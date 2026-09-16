"""Unit tests for Phase 1: Dynamic Spot Lights with Shadow Casting."""

import math
import numpy as np
import pytest

from engine.gfx.passes.clustered_lights import ClusteredLightingPass, PointLight, SpotLight
from engine.gfx.spot_shadows import SpotLightShadowMap
from engine.gfx.quality_presets import RenderConfig
from engine.debug.engine_tweaks import EngineTweaks
from engine.gfx.render_graph import RenderGraphContext


class DummyBuffer:
    """Mock ModernGL buffer for headless unit testing."""
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


class DummyContext:
    """Mock ModernGL context for headless unit testing."""
    def __init__(self):
        self.viewport = (0, 0, 800, 600)

    def buffer(self, reserve: int) -> DummyBuffer:
        return DummyBuffer(reserve)


class TestSpotLightData:
    """Tests SpotLight dataclass and ClusteredLightingPass buffer packing."""

    def test_spot_light_creation(self):
        spot = SpotLight(
            x=1.0, y=2.0, z=3.0,
            radius=25.0,
            dir_x=0.0, dir_y=-0.2, dir_z=-1.0,
            inner_cone_angle=math.radians(15.0),
            outer_cone_angle=math.radians(30.0),
            r=1.0, g=0.9, b=0.8,
            intensity=5.0,
            cast_shadow=True,
            shadow_bias=0.002,
        )
        assert spot.x == 1.0
        assert spot.radius == 25.0
        assert spot.cast_shadow is True
        assert spot.shadow_index == -1

    def test_clustered_lighting_pass_spot_packing(self):
        ctx = DummyContext()
        pass_instance = ClusteredLightingPass(ctx, max_lights=16, max_spot_lights=8)

        # Add point light
        pl = pass_instance.add_light((0.0, 1.0, 2.0), radius=10.0, color=(1.0, 0.5, 0.2), intensity=3.0)
        assert len(pass_instance.lights) == 1

        # Add spot lights: 5 total, 5 want shadows, but only 4 shadow slots exist
        for i in range(5):
            pass_instance.add_spot_light(
                position=(float(i), 2.0, 0.0),
                direction=(0.0, -0.1, -1.0),
                radius=30.0 + i,
                inner_cone_angle=math.radians(15.0),
                outer_cone_angle=math.radians(30.0),
                color=(1.0, 1.0, 1.0),
                intensity=4.0,
                cast_shadow=True,
                shadow_bias=0.0015,
            )

        assert len(pass_instance.spot_lights) == 5

        # Execute pass
        rg_ctx = RenderGraphContext(ctx=ctx, width=800, height=600, config=RenderConfig(), frame_context=None)
        pass_instance.execute(rg_ctx)

        assert rg_ctx.resources["point_light_count"] == 1
        assert rg_ctx.resources["spot_light_count"] == 5

        # Only first 4 should have shadow_index >= 0
        shadow_spots = rg_ctx.resources["shadow_spot_lights"]
        assert len(shadow_spots) == 4
        assert pass_instance.spot_lights[0].shadow_index == 0
        assert pass_instance.spot_lights[1].shadow_index == 1
        assert pass_instance.spot_lights[2].shadow_index == 2
        assert pass_instance.spot_lights[3].shadow_index == 3
        assert pass_instance.spot_lights[4].shadow_index == -1

        # Verify SSBO byte offsets
        # Point lights are in 0..8192
        point_data = np.frombuffer(pass_instance.ssbo_lights.data[:32], dtype=np.float32)
        assert pytest.approx(point_data[0]) == 0.0  # x
        assert pytest.approx(point_data[1]) == 1.0  # y
        assert pytest.approx(point_data[2]) == 2.0  # z
        assert pytest.approx(point_data[3]) == 10.0 # radius

        # Spot lights start at offset 8192
        spot_offset = 8192
        spot_data = np.frombuffer(pass_instance.ssbo_lights.data[spot_offset : spot_offset + 64], dtype=np.float32)
        assert pytest.approx(spot_data[0]) == 0.0  # x
        assert pytest.approx(spot_data[1]) == 2.0  # y
        assert pytest.approx(spot_data[2]) == 0.0  # z
        assert pytest.approx(spot_data[3]) == 30.0 # radius
        assert pytest.approx(spot_data[7]) == math.cos(math.radians(15.0)) # inner_cos
        assert pytest.approx(spot_data[12]) == math.cos(math.radians(30.0)) # outer_cos
        assert pytest.approx(spot_data[13]) == 0.0 # shadow_index = 0

    def test_clear_lights(self):
        ctx = DummyContext()
        pass_instance = ClusteredLightingPass(ctx, max_lights=16, max_spot_lights=8)
        pass_instance.add_light((0.0, 0.0, 0.0))
        pass_instance.add_spot_light((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        assert len(pass_instance.lights) == 1
        assert len(pass_instance.spot_lights) == 1

        pass_instance.clear_spot_lights()
        assert len(pass_instance.lights) == 1
        assert len(pass_instance.spot_lights) == 0

        pass_instance.clear()
        assert len(pass_instance.lights) == 0
        assert len(pass_instance.spot_lights) == 0


class TestSpotLightShadowMapMath:
    """Tests SpotLightShadowMap perspective matrix projection math."""

    def test_spot_matrix_computation(self):
        ctx = DummyContext()
        # Mock depth_texture and fbo creation to test matrix math
        spot_sm = SpotLightShadowMap.__new__(SpotLightShadowMap)
        spot_sm.ctx = ctx
        spot_sm.atlas_size = 2048
        spot_sm._proj_scratch = np.zeros(16, dtype=np.float32)
        spot_sm._view_scratch = np.zeros(16, dtype=np.float32)
        spot_sm.spot_matrices = [np.identity(4, dtype=np.float32).flatten() for _ in range(4)]

        vp = spot_sm.compute_spot_matrix(
            position=(0.0, 2.0, 5.0),
            direction=(0.0, 0.0, -1.0),
            outer_cone_angle_rad=math.radians(30.0),
            radius=35.0,
            spot_idx=0,
        )

        assert vp is not None
        assert vp.shape == (16,)
        # Reshape to (4, 4) in column-major order
        mat = vp.reshape((4, 4), order="F")

        # Test point directly in front of the spot light at distance 10m
        target_world = np.array([0.0, 2.0, -5.0, 1.0], dtype=np.float32)
        clip_pos = mat @ target_world
        ndc_pos = clip_pos[:3] / clip_pos[3]

        # Should be centered in X and Y (NDC coords ~ 0.0)
        assert pytest.approx(ndc_pos[0], abs=1e-3) == 0.0
        assert pytest.approx(ndc_pos[1], abs=1e-3) == 0.0
        # Depth in GL_ZERO_TO_ONE range: should be between 0.0 and 1.0
        assert 0.0 <= ndc_pos[2] <= 1.0

    def test_spot_quadrant_viewport(self):
        ctx = DummyContext()
        spot_sm = SpotLightShadowMap.__new__(SpotLightShadowMap)
        spot_sm.ctx = ctx
        spot_sm.atlas_size = 2048

        # Quadrant 0
        spot_sm.begin_spot(0)
        assert ctx.viewport == (0, 0, 1024, 1024)

        # Quadrant 1
        spot_sm.begin_spot(1)
        assert ctx.viewport == (1024, 0, 1024, 1024)

        # Quadrant 2
        spot_sm.begin_spot(2)
        assert ctx.viewport == (0, 1024, 1024, 1024)

        # Quadrant 3
        spot_sm.begin_spot(3)
        assert ctx.viewport == (1024, 1024, 1024, 1024)


class TestEngineTweaksSpotFields:
    """Tests spot light fields serialization and deserialization in EngineTweaks and RenderConfig."""

    def test_render_config_defaults(self):
        cfg = RenderConfig()
        assert cfg.spot_lights_enabled is True
        assert cfg.spot_shadows_enabled is True
        assert cfg.spot_shadow_resolution == 2048
        assert cfg.spot_light_radius == 35.0
        assert cfg.spot_cone_angle == 32.0
        assert cfg.spot_shadow_bias == 0.0015

    def test_engine_tweaks_serialization(self):
        tweaks = EngineTweaks()
        tweaks.spot_lights_enabled = False
        tweaks.spot_shadows_enabled = False
        tweaks.spot_light_radius = 42.0
        tweaks.spot_cone_angle = 28.5
        tweaks.spot_shadow_bias = 0.0025

        d = tweaks.to_dict()
        assert d["spot_lights_enabled"] is False
        assert d["spot_shadows_enabled"] is False
        assert d["spot_light_radius"] == 42.0
        assert d["spot_cone_angle"] == 28.5
        assert d["spot_shadow_bias"] == 0.0025

        # Restore into fresh instance
        restored = EngineTweaks()
        restored.from_dict(d)
        assert restored.spot_lights_enabled is False
        assert restored.spot_shadows_enabled is False
        assert restored.spot_light_radius == 42.0
        assert restored.spot_cone_angle == 28.5
        assert restored.spot_shadow_bias == 0.0025
