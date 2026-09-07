"""Offline Asset Cooker CLI Tool for PyMordial Engine.

Compiles raw assets into fast, binary, memory-mappable structures:
- .obj / .glb / .gltf -> 32-byte cache-aligned .pm_mesh (with computed tangents and bounding volumes)
- .png / .jpg / .tga  -> Pre-filtered mipmapped .pm_tex
- Directory of cooked assets -> Contiguous .pak archive container
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import struct
import numpy as np
from PIL import Image

from engine.assets.mesh_format import PMMesh, build_pm_mesh
from engine.assets.texture_format import TextureFormat, cook_image_to_pm_tex
from engine.assets.vfs import PakWriter


def compute_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Computes smooth Gram-Schmidt tangents and bitangent signs using vectorized NumPy."""
    v_count = len(positions)
    if v_count == 0 or len(indices) == 0:
        return np.zeros((v_count, 4), dtype=np.float32)

    tri_count = len(indices) // 3
    tri_indices = indices[: tri_count * 3].reshape(tri_count, 3)
    i0 = tri_indices[:, 0]
    i1 = tri_indices[:, 1]
    i2 = tri_indices[:, 2]

    v0 = positions[i0, :3].astype(np.float32)
    v1 = positions[i1, :3].astype(np.float32)
    v2 = positions[i2, :3].astype(np.float32)

    w0 = uvs[i0, :2].astype(np.float32)
    w1 = uvs[i1, :2].astype(np.float32)
    w2 = uvs[i2, :2].astype(np.float32)

    delta_v1 = v1 - v0
    delta_v2 = v2 - v0

    delta_w1 = w1 - w0
    delta_w2 = w2 - w0

    s1 = delta_w1[:, 0:1]
    t1 = delta_w1[:, 1:2]
    s2 = delta_w2[:, 0:1]
    t2 = delta_w2[:, 1:2]

    det = s1 * t2 - s2 * t1
    denom = np.where(np.abs(det) < 1e-8, 1e-8, det)
    r = 1.0 / denom

    sdir = (delta_v1 * t2 - delta_v2 * t1) * r
    tdir = (delta_v2 * s1 - delta_v1 * s2) * r

    tan1 = np.zeros((v_count, 3), dtype=np.float32)
    tan2 = np.zeros((v_count, 3), dtype=np.float32)

    np.add.at(tan1, i0, sdir)
    np.add.at(tan1, i1, sdir)
    np.add.at(tan1, i2, sdir)

    np.add.at(tan2, i0, tdir)
    np.add.at(tan2, i1, tdir)
    np.add.at(tan2, i2, tdir)

    # Gram-Schmidt orthogonalization: T' = normalize(t - n * dot(n, t))
    n = normals[:, :3].astype(np.float32)
    dot_nt = np.sum(n * tan1, axis=1, keepdims=True)
    ortho_t = tan1 - n * dot_nt
    lengths = np.linalg.norm(ortho_t, axis=1, keepdims=True)

    fallback = np.zeros_like(ortho_t)
    fallback[:, 0] = 1.0
    valid_mask = lengths > 1e-6
    safe_lengths = np.where(valid_mask, lengths, 1.0)
    normalized_t = np.where(valid_mask, ortho_t / safe_lengths, fallback)

    # Calculate handedness / bitangent sign
    cross_nt = np.cross(n, tan1)
    handedness_dot = np.sum(cross_nt * tan2, axis=1)
    w = np.where(handedness_dot < 0.0, -1.0, 1.0)

    tangents = np.empty((v_count, 4), dtype=np.float32)
    tangents[:, :3] = normalized_t
    tangents[:, 3] = w
    return tangents


def parse_obj(filepath: str | Path) -> PMMesh:
    """Parses a Wavefront .obj file and compiles it into a PMMesh."""
    positions_pool: list[tuple[float, float, float]] = []
    normals_pool: list[tuple[float, float, float]] = []
    uvs_pool: list[tuple[float, float]] = []

    unique_vertex_map: dict[tuple[int, int, int], int] = {}
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    indices: list[int] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "v":
                positions_pool.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif cmd == "vn":
                normals_pool.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif cmd == "vt":
                uvs_pool.append((float(parts[1]), float(parts[2])))
            elif cmd == "f":
                face_vertices = parts[1:]
                face_indices = []

                for f_spec in face_vertices:
                    tokens = f_spec.split("/")
                    v_idx = int(tokens[0]) - 1 if tokens[0] else 0
                    vt_idx = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1] else -1
                    vn_idx = int(tokens[2]) - 1 if len(tokens) > 2 and tokens[2] else -1

                    key = (v_idx, vt_idx, vn_idx)
                    if key not in unique_vertex_map:
                        new_idx = len(positions)
                        unique_vertex_map[key] = new_idx

                        p = positions_pool[v_idx] if 0 <= v_idx < len(positions_pool) else (0.0, 0.0, 0.0)
                        uv = uvs_pool[vt_idx] if 0 <= vt_idx < len(uvs_pool) else (0.0, 0.0)
                        n = normals_pool[vn_idx] if 0 <= vn_idx < len(normals_pool) else (0.0, 1.0, 0.0)

                        positions.append(p)
                        uvs.append(uv)
                        normals.append(n)
                        face_indices.append(new_idx)
                    else:
                        face_indices.append(unique_vertex_map[key])

                # Fan triangulation for polygons with > 3 vertices
                for i in range(1, len(face_indices) - 1):
                    indices.extend([face_indices[0], face_indices[i], face_indices[i + 1]])

    pos_arr = np.array(positions, dtype=np.float32)
    norm_arr = np.array(normals, dtype=np.float32)
    uv_arr = np.array(uvs, dtype=np.float32)
    idx_arr = np.array(indices, dtype=np.uint32)

    # Compute smooth normal tangents
    tangents = compute_tangents(pos_arr, norm_arr, uv_arr, idx_arr)

    return build_pm_mesh(
        positions=pos_arr,
        normals=norm_arr,
        uvs=uv_arr,
        tangents=tangents,
        indices=idx_arr,
    )


def parse_glb(filepath: str | Path) -> PMMesh:
    """Parses a binary glTF (.glb) file and compiles the first primitive mesh."""
    raw = Path(filepath).read_bytes()
    if len(raw) < 12:
        raise ValueError("File too small to be valid .glb")

    magic, version, _ = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF":
        raise ValueError(f"Invalid .glb magic: {magic!r}")

    # Read chunks
    pos = 12
    json_chunk = None
    bin_chunk = None

    while pos < len(raw):
        chunk_len, chunk_type = struct.unpack("<I4s", raw[pos : pos + 8])
        pos += 8
        chunk_data = raw[pos : pos + chunk_len]
        pos += chunk_len

        if chunk_type == b"JSON":
            json_chunk = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk_data

    if json_chunk is None or bin_chunk is None:
        raise ValueError(".glb must contain both JSON and BIN chunks")

    meshes = json_chunk.get("meshes", [])
    if not meshes:
        raise ValueError(".glb contains no meshes")

    prim = meshes[0]["primitives"][0]
    accessors = json_chunk.get("accessors", [])
    buffer_views = json_chunk.get("bufferViews", [])

    def get_accessor_data(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = buffer_views[acc["bufferView"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        comp_type = acc["componentType"]
        type_str = acc["type"]

        # Component type mapping: 5126=FLOAT, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT
        dtypes = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}
        dt = dtypes.get(comp_type, np.float32)

        type_counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        c_count = type_counts.get(type_str, 1)

        total_elements = count * c_count
        byte_len = total_elements * np.dtype(dt).itemsize
        data_slice = bin_chunk[offset : offset + byte_len]

        arr = np.frombuffer(data_slice, dtype=dt, count=total_elements)
        if c_count > 1:
            arr = arr.reshape(count, c_count)
        return arr

    attributes = prim["attributes"]
    pos_arr = get_accessor_data(attributes["POSITION"]).astype(np.float32)

    if "NORMAL" in attributes:
        norm_arr = get_accessor_data(attributes["NORMAL"]).astype(np.float32)
    else:
        norm_arr = np.zeros_like(pos_arr)
        norm_arr[:, 1] = 1.0

    if "TEXCOORD_0" in attributes:
        uv_arr = get_accessor_data(attributes["TEXCOORD_0"]).astype(np.float32)
    else:
        uv_arr = np.zeros((len(pos_arr), 2), dtype=np.float32)

    if "indices" in prim:
        idx_arr = get_accessor_data(prim["indices"]).astype(np.uint32).flatten()
    else:
        idx_arr = np.arange(len(pos_arr), dtype=np.uint32)

    if "TANGENT" in attributes:
        tan_arr = get_accessor_data(attributes["TANGENT"]).astype(np.float32)
    else:
        tan_arr = compute_tangents(pos_arr, norm_arr, uv_arr, idx_arr)

    return build_pm_mesh(
        positions=pos_arr,
        normals=norm_arr,
        uvs=uv_arr,
        tangents=tan_arr,
        indices=idx_arr,
    )


def cook_mesh(input_path: str | Path, output_path: str | Path) -> None:
    """Ingests an .obj, .glb, or .gltf model and bakes it into a 32-byte .pm_mesh."""
    in_p = Path(input_path)
    out_p = Path(output_path)

    ext = in_p.suffix.lower()
    if ext == ".obj":
        mesh = parse_obj(in_p)
    elif ext in (".glb", ".gltf"):
        mesh = parse_glb(in_p)
    else:
        raise ValueError(f"Unsupported model extension '{ext}'. Supported: .obj, .glb, .gltf")

    mesh.save(out_p)
    print(
        f"[Cooker] Baked '{in_p.name}' -> '{out_p.name}': "
        f"{mesh.vertex_count} vertices, {mesh.index_count} indices, "
        f"Radius={mesh.bounds.sphere_radius:0.2f}m"
    )


def cook_texture(
    input_path: str | Path,
    output_path: str | Path,
    target_format: TextureFormat = TextureFormat.RGBA8_UNORM,
    generate_mips: bool = True,
    target_size: tuple[int, int] | None = None,
) -> None:
    """Ingests a source image (.png, .jpg, etc.) and bakes it into a mipmapped .pm_tex."""
    in_p = Path(input_path)
    out_p = Path(output_path)

    with Image.open(in_p) as img:
        tex = cook_image_to_pm_tex(
            img,
            target_format=target_format,
            generate_mips=generate_mips,
            target_size=target_size,
        )
        tex.save(out_p)

    print(
        f"[Cooker] Baked '{in_p.name}' -> '{out_p.name}': "
        f"{tex.width}x{tex.height}, {tex.mip_count} mips, Format={tex.format.name}"
    )


def cook_material(
    material_dir: str | Path,
    output_dir: str | Path,
    target_size: tuple[int, int] = (4096, 4096),
    generate_mips: bool = True,
    force: bool = False,
) -> None:
    """Bakes all 4 PBR channels of a material folder into pre-filtered .pm_tex files.

    Channels:
    - diff.png -> diff.pm_tex (RGBA8_UNORM)
    - nor.png  -> nor.pm_tex  (RGBA8_UNORM)
    - disp.png -> disp.pm_tex (R8_UNORM)
    - arm.png  -> arm.pm_tex  (RGBA8_UNORM)
    """
    in_d = Path(material_dir)
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    channel_configs = [
        ("diff.png", "diff.pm_tex", TextureFormat.RGBA8_UNORM, "RGBA", (255, 255, 255, 255)),
        ("nor.png", "nor.pm_tex", TextureFormat.RGBA8_UNORM, "RGBA", (128, 128, 255, 255)),
        ("disp.png", "disp.pm_tex", TextureFormat.R8_UNORM, "L", 128),
        ("arm.png", "arm.pm_tex", TextureFormat.RGBA8_UNORM, "RGBA", (255, 128, 0, 255)),
    ]

    for src_name, out_name, fmt, fallback_mode, fallback_color in channel_configs:
        src_path = in_d / src_name
        out_path = out_d / out_name

        # Incremental check: skip if output exists and is newer than source
        if not force and out_path.is_file():
            if src_path.is_file():
                if out_path.stat().st_mtime >= src_path.stat().st_mtime:
                    continue
            else:
                continue

        if src_path.is_file():
            with Image.open(src_path) as img:
                tex = cook_image_to_pm_tex(
                    img,
                    target_format=fmt,
                    generate_mips=generate_mips,
                    target_size=target_size,
                )
        else:
            fallback_img = Image.new(fallback_mode, target_size, fallback_color)
            tex = cook_image_to_pm_tex(
                fallback_img,
                target_format=fmt,
                generate_mips=generate_mips,
                target_size=target_size,
            )

        tex.save(out_path)
        print(f"[Cooker] Material '{in_d.name}' baked '{out_name}' ({tex.width}x{tex.height}, {fmt.name})")


def cook_all_materials(
    materials_root: str | Path,
    output_root: str | Path,
    target_size: tuple[int, int] = (4096, 4096),
    generate_mips: bool = True,
    force: bool = False,
) -> int:
    """Discovers and cooks all PBR material folders under materials_root.

    Returns the count of material folders cooked/verified.
    """
    in_root = Path(materials_root)
    out_root = Path(output_root)

    if not in_root.is_dir():
        print(f"[Cooker] Materials root '{in_root}' not found.")
        return 0

    subdirs = [d for d in in_root.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda d: d.name)

    cooked_count = 0
    for d in subdirs:
        out_dir = out_root / d.name
        cook_material(
            material_dir=d,
            output_dir=out_dir,
            target_size=target_size,
            generate_mips=generate_mips,
            force=force,
        )
        cooked_count += 1

    print(f"[Cooker] Processed {cooked_count} materials in '{output_root}'")
    return cooked_count


def pack_directory(
    input_dir: str | Path,
    output_pak: str | Path,
    compress: bool = False,
) -> None:
    """Recursively packages all cooked files in a directory into a .pak archive."""
    in_d = Path(input_dir).resolve()
    writer = PakWriter()

    file_count = 0
    total_bytes = 0
    for root, _, files in os.walk(in_d):
        for f in files:
            p = Path(root, f)
            rel_path = p.relative_to(in_d).as_posix()
            data = p.read_bytes()
            writer.add_file(rel_path, data, compress=compress)
            file_count += 1
            total_bytes += len(data)

    writer.write(output_pak)
    print(
        f"[Cooker] Packed {file_count} assets ({total_bytes / (1024 * 1024):0.2f} MB) -> '{Path(output_pak).name}'"
    )


def main() -> None:
    """CLI entry point for the offline cooker."""
    parser = argparse.ArgumentParser(description="PyMordial Engine Offline Asset Cooker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cook-mesh
    p_mesh = subparsers.add_parser("cook-mesh", help="Cooks an .obj/.glb model into a .pm_mesh")
    p_mesh.add_argument("input", help="Source model path")
    p_mesh.add_argument("output", help="Destination .pm_mesh path")

    # cook-tex
    p_tex = subparsers.add_parser("cook-tex", help="Cooks an image into a .pm_tex")
    p_tex.add_argument("input", help="Source image path")
    p_tex.add_argument("output", help="Destination .pm_tex path")
    p_tex.add_argument("--format", default="rgba8", choices=["rgba8", "bc7", "bc5", "r8"], help="Target GPU format")
    p_tex.add_argument("--no-mips", action="store_true", help="Disable pre-baked mipmap chain")
    p_tex.add_argument("--width", type=int, default=None, help="Target width resize")
    p_tex.add_argument("--height", type=int, default=None, help="Target height resize")

    # cook-material
    p_mat = subparsers.add_parser("cook-material", help="Cooks a PBR material folder into .pm_tex files")
    p_mat.add_argument("input", help="Source material folder")
    p_mat.add_argument("output", help="Destination material folder")
    p_mat.add_argument("--size", type=int, default=4096, help="Target resolution (default: 4096)")
    p_mat.add_argument("--no-mips", action="store_true", help="Disable pre-baked mipmap chain")
    p_mat.add_argument("--force", action="store_true", help="Force re-bake regardless of timestamps")

    # cook-materials
    p_mats = subparsers.add_parser("cook-materials", help="Cooks all PBR material folders in a directory")
    p_mats.add_argument("input_root", help="Source materials root directory (e.g. assets/textures)")
    p_mats.add_argument("output_root", help="Destination materials root directory")
    p_mats.add_argument("--size", type=int, default=4096, help="Target resolution (default: 4096)")
    p_mats.add_argument("--no-mips", action="store_true", help="Disable pre-baked mipmap chain")
    p_mats.add_argument("--force", action="store_true", help="Force re-bake regardless of timestamps")

    # pack
    p_pack = subparsers.add_parser("pack", help="Packs a directory of cooked assets into a .pak")
    p_pack.add_argument("input_dir", help="Directory containing cooked assets")
    p_pack.add_argument("output_pak", help="Destination .pak archive file")
    p_pack.add_argument("--compress", action="store_true", help="Enable Zstandard compression")

    args = parser.parse_args()

    if args.command == "cook-mesh":
        cook_mesh(args.input, args.output)
    elif args.command == "cook-tex":
        fmt_map = {
            "rgba8": TextureFormat.RGBA8_UNORM,
            "bc7": TextureFormat.BC7_UNORM,
            "bc5": TextureFormat.BC5_UNORM,
            "r8": TextureFormat.R8_UNORM,
        }
        target_sz = (args.width, args.height) if (args.width and args.height) else None
        cook_texture(
            args.input,
            args.output,
            target_format=fmt_map[args.format],
            generate_mips=not args.no_mips,
            target_size=target_sz,
        )
    elif args.command == "cook-material":
        cook_material(
            args.input,
            args.output,
            target_size=(args.size, args.size),
            generate_mips=not args.no_mips,
            force=args.force,
        )
    elif args.command == "cook-materials":
        cook_all_materials(
            args.input_root,
            args.output_root,
            target_size=(args.size, args.size),
            generate_mips=not args.no_mips,
            force=args.force,
        )
    elif args.command == "pack":
        pack_directory(args.input_dir, args.output_pak, compress=args.compress)


if __name__ == "__main__":
    main()
