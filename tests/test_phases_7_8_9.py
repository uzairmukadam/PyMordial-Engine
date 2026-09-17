"""Unit tests for Phase 7 (Normal Mapping), Phase 8 (HDR Emissive), Phase 9 (Deferred Decals), and Spot Light Interpolation Sync."""

import math
import numpy as np
import pytest

from engine.core.materials import MaterialDef, DisplacementMode
from engine.core.ecs import EntityManager
from engine.gfx.passes.decal_pass import DecalPass
from engine.core.math_utils import quat_nlerp, trs_to_mat4


class DummyBuffer:
    def __init__(self, size: int):
        self.size = size
        self.data = bytearray(size)

    def write(self, data: bytes, offset: int = 0) -> None:
        end = offset + len(data)
        self.data[offset:end] = data

    def bind_to_storage_buffer(self, binding: int) -> None:
        pass

    def release(self) -> None:
        pass


class DummyProgram:
    def __init__(self):
        self.uniforms = {}

    def get(self, name: str, default=None):
        return default

    def __contains__(self, name: str) -> bool:
        return True

    def __getitem__(self, name: str):
        class UniformProxy:
            value = 0
        return UniformProxy()

    def release(self) -> None:
        pass


class DummyTexture:
    def __init__(self, size, components, dtype):
        self.size = size
        self.components = components
        self.dtype = dtype
        self.filter = None

    def release(self) -> None:
        pass


class DummyContext:
    def __init__(self):
        self.viewport = (0, 0, 800, 600)

    def buffer(self, reserve: int) -> DummyBuffer:
        return DummyBuffer(reserve)

    def program(self, vertex_shader: str, fragment_shader: str):
        return DummyProgram()

    def vertex_array(self, prog, content):
        class DummyVAO:
            def release(self):
                pass
        return DummyVAO()

    def texture(self, size, components, dtype):
        return DummyTexture(size, components, dtype)

    def framebuffer(self, color_attachments=None, depth_attachment=None):
        class DummyFBO:
            def release(self):
                pass
        return DummyFBO()


class TestPhases789:
    """Tests for Phases 7, 8, 9 graphical systems."""

    def test_material_normal_strength_and_emissive(self):
        """Phase 7 & 8: Verifies normal_strength and emissive_intensity in MaterialDef."""
        mat = MaterialDef(
            name="neon_sign",
            material_id=42,
            color=(1.0, 0.2, 0.1),
            roughness=0.25,
            metallic=0.0,
            normal_strength=1.5,
            emissive_intensity=4.0,
        )
        assert mat.normal_strength == 1.5
        assert mat.emissive_intensity == 4.0

        # Floats must scale RGB color into HDR range by (1.0 + emissive_intensity) = 5.0x
        floats = mat.to_material_floats()
        assert np.isclose(floats[0], 5.0)
        assert np.isclose(floats[1], 1.0)
        assert np.isclose(floats[2], 0.5)
        assert np.isclose(floats[3], 0.25)

    def test_ecs_emissive_entity(self):
        """Phase 8: Verifies emissive_intensity support in EntityManager."""
        ecs = EntityManager(max_entities=32)
        eid = ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            color=(1.0, 0.9, 0.8),
            emissive_intensity=3.0,
        )
        dense = ecs.pool.get_dense_index(eid)
        assert dense >= 0

        # Stored color must be (1.0 + 3.0) * color = 4.0 * (1.0, 0.9, 0.8)
        stored_color = ecs.material_data[dense, 0:3]
        assert np.isclose(stored_color[0], 4.0)
        assert np.isclose(stored_color[1], 3.6)
        assert np.isclose(stored_color[2], 3.2)

        # Dynamic update via set_entity_material
        ecs.set_entity_material(eid, color=(1.0, 0.0, 0.0), emissive_intensity=5.0)
        updated = ecs.material_data[dense, 0:3]
        assert np.isclose(updated[0], 6.0)
        assert np.isclose(updated[1], 0.0)
        assert np.isclose(updated[2], 0.0)

    def test_decal_pass_data_and_matrix_inversion(self):
        """Phase 9: Verifies DecalPass adds decals and computes valid inverse model matrices."""
        ctx = DummyContext()
        decal_pass = DecalPass(ctx, width=800, height=600)
        assert decal_pass.active_decal_count == 0

        # Add a tire skidmark decal at (10, 0.5, 20) with yaw rotation
        pos = (10.0, 0.5, 20.0)
        rot = (0.0, 0.7071, 0.0, 0.7071)
        size = (0.32, 0.25, 0.75)
        idx = decal_pass.add_decal(
            position=pos,
            rotation_quat=rot,
            size=size,
            color=(0.04, 0.04, 0.05),
            opacity=0.90,
            roughness=0.96,
            decal_type=0,
        )
        assert idx == 0
        assert decal_pass.active_decal_count == 1

        # Check that inverse matrix transforms position back to origin (local center)
        offset = idx * decal_pass.DECAL_SIZE_FLOATS
        inv_mat = decal_pass._cpu_buffer[offset : offset + 16].reshape((4, 4), order="F")
        # Point at pos in homogeneous coords
        p_world = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
        p_local = inv_mat @ p_world
        assert np.allclose(p_local[:3], [0.0, 0.0, 0.0], atol=1e-4)

        # Verify color, opacity, roughness, and decal_type
        assert np.isclose(decal_pass._cpu_buffer[offset + 16], 0.04)
        assert np.isclose(decal_pass._cpu_buffer[offset + 19], 0.90)  # opacity
        assert np.isclose(decal_pass._cpu_buffer[offset + 20], 0.96)  # roughness
        assert np.isclose(decal_pass._cpu_buffer[offset + 21], 0.0)   # decal_type 0 = skidmark

        # Test clear
        decal_pass.clear()
        assert decal_pass.active_decal_count == 0

    def test_spotlight_subframe_interpolation(self):
        """Verifies sub-frame NLERP correctly aligns spot light coordinate math."""
        q_prev = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # 90-degree yaw rotation around Y
        q_curr = np.array([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float32)

        alpha = 0.5
        q_interp = quat_nlerp(q_prev, q_curr, alpha)
        # At alpha=0.5, angle is 45 degrees
        sin_45_half = math.sin(math.radians(22.5))
        cos_45_half = math.cos(math.radians(22.5))
        assert np.isclose(q_interp[1], sin_45_half, atol=1e-3)
        assert np.isclose(q_interp[3], cos_45_half, atol=1e-3)

        # Forward vector computed from interpolated quaternion
        qx, qy, qz, qw = q_interp
        fwd_x = -2.0 * (qx * qz + qw * qy)
        fwd_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
        # At 45 deg, fwd_x and fwd_z must have magnitude ~sqrt(0.5)
        assert np.isclose(abs(fwd_x), math.sqrt(0.5), atol=1e-2)
        assert np.isclose(abs(fwd_z), math.sqrt(0.5), atol=1e-2)
