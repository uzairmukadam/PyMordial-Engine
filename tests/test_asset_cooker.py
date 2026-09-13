"""Unit tests for Phase 4: Offline Asset Cooker & Virtual File System (VFS).

Validates:
- 32-byte cache-aligned .pm_mesh binary format and bounding volumes
- Pre-filtered .pm_tex format and mipmap generation
- .obj model parser and tangent generation
- .pak container creation, memory-mapped zero-copy reading, and Zstandard decompression
- ResourceCache deduplication and MegaBuffer 32-byte integration
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import moderngl

from engine.assets.mesh_format import (
    PMMesh,
    VERTEX_STRIDE,
    build_pm_mesh,
)
from engine.assets.texture_format import (
    TextureFormat,
    cook_image_to_pm_tex,
    PMTex,
)
from engine.assets.vfs import (
    PakWriter,
    PakReader,
    VFS,
)
from engine.assets.cooker import (
    cook_mesh,
    cook_material,
    cook_all_materials,
)
from engine.assets.resource_cache import ResourceCache
from engine.gfx.mega_buffer import MegaBuffer
from engine.gfx.texture_atlas import decode_material_folder


class TestAssetCooker:
    """Comprehensive test suite for the PyMordial Phase 4 asset system."""

    def test_pm_mesh_binary_packing_and_unpacking(self):
        """Validates 32-byte vertex stride, header bounds, and byte roundtrip."""
        positions = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        uvs = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tangents = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        mesh = build_pm_mesh(positions, normals, uvs, tangents, indices)

        assert mesh.vertex_count == 4
        assert mesh.index_count == 6
        assert mesh.vertices.dtype.itemsize == VERTEX_STRIDE

        # Verify Bounding Volumes
        bounds = mesh.bounds
        assert abs(bounds.sphere_center[0]) < 1e-5
        assert abs(bounds.sphere_center[1]) < 1e-5
        assert abs(bounds.sphere_radius - np.sqrt(2.0)) < 1e-4
        assert bounds.aabb_min == (-1.0, -1.0, 0.0)
        assert bounds.aabb_max == (1.0, 1.0, 0.0)

        # Serialize & Deserialize
        serialized = mesh.serialize()
        expected_len = 64 + 4 * 32 + 6 * 4
        assert len(serialized) == expected_len

        restored = PMMesh.from_bytes(serialized)
        assert restored.vertex_count == 4
        assert restored.index_count == 6
        assert np.allclose(restored.vertices["position"], positions)
        assert np.array_equal(restored.indices, indices)
        assert abs(restored.bounds.sphere_radius - bounds.sphere_radius) < 1e-4

    def test_obj_model_cooking_pipeline(self):
        """Validates Wavefront .obj ingestion, fan triangulation, and tangent baking."""
        obj_content = """# PyMordial Test Tetrahedron OBJ
v 0.0 1.0 0.0
v -1.0 -1.0 1.0
v 1.0 -1.0 1.0
v 0.0 -1.0 -1.0
vt 0.5 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 0.5
vn 0.0 1.0 0.0
vn 0.0 -1.0 0.0
vn 1.0 0.0 0.0
vn -1.0 0.0 0.0
f 1/1/1 2/2/4 3/3/3
f 1/1/1 3/3/3 4/4/2
f 1/1/1 4/4/2 2/2/4
f 2/2/2 4/4/2 3/3/2
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            obj_path = Path(tmp_dir) / "tetra.obj"
            pm_path = Path(tmp_dir) / "tetra.pm_mesh"
            obj_path.write_text(obj_content, encoding="utf-8")

            # Cook .obj -> .pm_mesh
            cook_mesh(obj_path, pm_path)
            assert pm_path.exists()

            cooked = PMMesh.load(pm_path)
            assert cooked.vertex_count > 0
            assert cooked.index_count == 12  # 4 triangle faces * 3 = 12 indices
            assert cooked.bounds.sphere_radius > 0.5

    def test_pm_tex_cooking_and_mipmaps(self):
        """Validates mipmap pyramid generation and .pm_tex binary structure."""
        # Create 64x64 synthetic gradient texture
        img = Image.new("RGBA", (64, 64), color=(255, 128, 64, 255))

        pm_tex = cook_image_to_pm_tex(img, target_format=TextureFormat.RGBA8_UNORM, generate_mips=True)

        # 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (7 levels)
        assert pm_tex.width == 64
        assert pm_tex.height == 64
        assert pm_tex.mip_count == 7
        assert pm_tex.format == TextureFormat.RGBA8_UNORM

        # Check first and last mip dimensions
        assert (pm_tex.mip_levels[0].width, pm_tex.mip_levels[0].height) == (64, 64)
        assert (pm_tex.mip_levels[-1].width, pm_tex.mip_levels[-1].height) == (1, 1)

        # Serialization roundtrip
        serialized = pm_tex.serialize()
        restored = PMTex.from_bytes(serialized)
        assert restored.width == 64
        assert restored.height == 64
        assert restored.mip_count == 7
        assert len(restored.get_mip_bytes(0)) == 64 * 64 * 4
        assert len(restored.get_mip_bytes(6)) == 1 * 1 * 4

    def test_vfs_pak_packaging_and_mmap_streaming(self):
        """Validates .pak creation, FNV-1a lookups, zero-copy mmap, and Zstd compression."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            pak_path = tmp / "game_data.pak"

            writer = PakWriter()
            # File 1: Uncompressed raw binary
            raw_mesh_bytes = b"PMMS_DUMMY_RAW_PAYLOAD_12345678"
            writer.add_file("meshes/hero.pm_mesh", raw_mesh_bytes, compress=False)

            # File 2: Compressed asset
            comp_bytes = b"A" * 2048  # Highly compressible string
            writer.add_file("textures/albedo.pm_tex", comp_bytes, compress=True)

            writer.write(pak_path)
            assert pak_path.exists()

            # Test PakReader
            with PakReader(pak_path) as reader:
                assert reader.has_file("meshes/hero.pm_mesh")
                assert reader.has_file("textures/albedo.pm_tex")
                assert not reader.has_file("non_existent.dat")

                # Verify uncompressed read returns memoryview (zero-copy)
                mesh_data = reader.read("meshes/hero.pm_mesh")
                assert isinstance(mesh_data, memoryview)
                assert bytes(mesh_data) == raw_mesh_bytes
                mesh_data.release()

                # Verify compressed read returns decompressed bytes
                tex_data = reader.read("textures/albedo.pm_tex")
                assert isinstance(tex_data, bytes)
                assert tex_data == comp_bytes

            # Test VFS Mount
            vfs = VFS()
            vfs.mount_pak(pak_path)
            assert vfs.exists("meshes/hero.pm_mesh")
            v_data = vfs.read("meshes/hero.pm_mesh")
            assert bytes(v_data) == raw_mesh_bytes
            if isinstance(v_data, memoryview):
                v_data.release()
            all_files = vfs.list_files()
            assert "meshes/hero.pm_mesh" in all_files
            assert "textures/albedo.pm_tex" in all_files
            vfs.close()

    def test_resource_cache_deduplication(self):
        """Validates that ResourceCache deduplicates loaded asset instances."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            # Create a simple valid mesh file
            positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            normals = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
            uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
            tangents = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float32)
            indices = np.array([0, 1, 2], dtype=np.uint32)
            mesh = build_pm_mesh(positions, normals, uvs, tangents, indices)
            mesh_path = tmp / "triangle.pm_mesh"
            mesh.save(mesh_path)

            vfs = VFS()
            vfs.mount_dir(tmp)
            cache = ResourceCache(vfs=vfs)

            m1 = cache.load_mesh("triangle.pm_mesh")
            m2 = cache.load_mesh("triangle.pm_mesh")
            # Should return the exact same Python object reference from cache
            assert m1 is m2
            assert m1.vertex_count == 3
            cache.close()

    def test_megabuffer_pm_mesh_integration(self):
        """Validates that MegaBuffer directly accepts 32-byte PMMesh instances."""
        ctx = moderngl.create_context(standalone=True)
        mega = MegaBuffer(ctx)

        initial_verts = mega.total_vertices
        initial_indices = mega.total_indices

        positions = np.array(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
        uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        tangents = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        custom_pm_mesh = build_pm_mesh(positions, normals, uvs, tangents, indices)

        # Register into MegaBuffer
        alloc = mega.add_pm_mesh("custom_tri", custom_pm_mesh, auto_bake=True)

        assert alloc.vertex_count == 3
        assert alloc.index_count == 3
        assert alloc.base_vertex == initial_verts
        assert alloc.first_index == initial_indices
        assert "custom_tri" in mega.allocations

        # Verify VAO compilation with 32-byte layout
        prog = ctx.program(
            vertex_shader="""#version 330 core
            in vec3 in_position;
            in vec4 in_normal;
            in vec2 in_uv;
            in vec4 in_tangent;
            void main() {
                gl_Position = vec4(in_position + in_normal.xyz + vec3(in_uv, in_tangent.x), 1.0);
            }
            """,
            fragment_shader="""#version 330 core
            out vec4 f_color;
            void main() {
                f_color = vec4(1.0);
            }
            """,
        )

        vao = mega.get_vao(prog)
        assert vao is not None

        # Verify duplicate registration returns existing allocation without rebaking
        initial_verts = mega.total_vertices
        alloc2 = mega.add_pm_mesh("custom_tri", custom_pm_mesh, auto_bake=True)
        assert alloc2 is alloc
        assert mega.total_vertices == initial_verts

        mega.destroy()
        ctx.release()

    def test_vfs_try_read_and_memoryview_slicing(self):
        """Validates that PakReader.try_read returns None on missing files and handles memoryviews."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pak_path = Path(tmp_dir) / "test_try_read.pak"
            writer = PakWriter()
            writer.add_file("data/sample.bin", b"SAMPLE_PAYLOAD_BYTES", compress=False)
            writer.write(pak_path)

            with PakReader(pak_path) as reader:
                assert reader.try_read("non_existent_file.dat") is None
                data = reader.try_read("data/sample.bin")
                assert data is not None
                assert bytes(data) == b"SAMPLE_PAYLOAD_BYTES"
                if isinstance(data, memoryview):
                    data.release()

    def test_pm_tex_r8_single_channel_format(self):
        """Validates R8_UNORM format cooking, single-channel byte size, and roundtrip."""
        img = Image.new("L", (64, 64), color=128)
        pm_tex = cook_image_to_pm_tex(img, target_format=TextureFormat.R8_UNORM, generate_mips=True)

        assert pm_tex.width == 64
        assert pm_tex.height == 64
        assert pm_tex.mip_count == 7
        assert pm_tex.format == TextureFormat.R8_UNORM
        # Mip 0 for R8 single-channel is 64*64*1 bytes
        assert len(pm_tex.get_mip_bytes(0)) == 64 * 64 * 1
        assert len(pm_tex.get_mip_bytes(6)) == 1 * 1 * 1

        serialized = pm_tex.serialize()
        restored = PMTex.from_bytes(serialized)
        assert restored.format == TextureFormat.R8_UNORM
        assert len(restored.get_mip_bytes(0)) == 4096
        assert restored.get_mip_bytes(0)[0] == 128

    def test_cook_texture_with_target_size(self):
        """Validates that cook_texture and cook_image_to_pm_tex resize non-square images."""
        img = Image.new("RGBA", (128, 64), color=(100, 150, 200, 255))
        pm_tex = cook_image_to_pm_tex(
            img,
            target_format=TextureFormat.RGBA8_UNORM,
            target_size=(32, 32),
            generate_mips=True,
        )
        assert pm_tex.width == 32
        assert pm_tex.height == 32
        assert len(pm_tex.get_mip_bytes(0)) == 32 * 32 * 4

    def test_cook_material_and_fast_path_decoding(self):
        """Validates full PBR material folder cooking and pure-binary fast path loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_mat = Path(tmp_dir) / "test_stone"
            src_mat.mkdir()
            out_mat = Path(tmp_dir) / "cooked_test_stone"

            # Create mock source images
            Image.new("RGBA", (32, 32), (200, 100, 50, 255)).save(src_mat / "diff.png")
            Image.new("RGBA", (32, 32), (128, 128, 255, 255)).save(src_mat / "nor.png")
            Image.new("L", (32, 32), 180).save(src_mat / "disp.png")
            # arm.png intentionally omitted to test fallback handling

            cook_material(src_mat, out_mat, target_size=(32, 32), generate_mips=True)

            assert (out_mat / "diff.pm_tex").exists()
            assert (out_mat / "nor.pm_tex").exists()
            assert (out_mat / "disp.pm_tex").exists()
            assert (out_mat / "arm.pm_tex").exists()

            # Test fast-path decoding
            layer = decode_material_folder(src_mat, 32, 32, cooked_folder=out_mat)
            assert layer.name == "test_stone"
            assert len(layer.diffuse_bytes) == 32 * 32 * 4
            assert len(layer.normal_bytes) == 32 * 32 * 4
            assert len(layer.disp_bytes) == 32 * 32 * 1
            assert len(layer.arm_bytes) == 32 * 32 * 4
            # Verify displacement byte content
            assert layer.disp_bytes[0] == 180

    def test_cook_all_materials_incremental(self):
        """Validates batch material cooking and incremental mtime skipping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_src = Path(tmp_dir) / "materials_src"
            root_out = Path(tmp_dir) / "materials_out"
            mat1 = root_src / "mat_a"
            mat2 = root_src / "mat_b"
            mat1.mkdir(parents=True)
            mat2.mkdir(parents=True)

            Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(mat1 / "diff.png")
            Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(mat2 / "diff.png")

            count = cook_all_materials(root_src, root_out, target_size=(16, 16), generate_mips=False)
            assert count == 2
            assert (root_out / "mat_a" / "diff.pm_tex").exists()
            assert (root_out / "mat_b" / "diff.pm_tex").exists()

            mtime_before = (root_out / "mat_a" / "diff.pm_tex").stat().st_mtime

            # Second call without force should skip re-cooking
            count2 = cook_all_materials(root_src, root_out, target_size=(16, 16), generate_mips=False, force=False)
            assert count2 == 2
            mtime_after = (root_out / "mat_a" / "diff.pm_tex").stat().st_mtime
            assert mtime_before == mtime_after

