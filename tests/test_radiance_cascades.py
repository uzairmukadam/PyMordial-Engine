"""Unit and Integration Tests for Next-Gen Hybrid Radiance Cascades (SSRC + FFPC).

Verifies:
1. RadianceCascadesPass lifecycle, compute shader compilation, texture allocation, and resize.
2. Execution of SSRC interval raymarching with FFPC far-field probe integration.
3. Strict mutual exclusivity between Classic GI (LPV+SSGI) and Radiance Cascades (SSRC+FFPC).
4. Full render pipeline frame pass with G-Buffer debug mode 13 (Radiance Cascades visualization).
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import GIMode, RenderConfig
from engine.gfx.passes.radiance_cascades_pass import RadianceCascadesPass
from engine.gfx.pipeline import RenderPipeline


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestRadianceCascadesPass:
    def test_pass_initialization_and_textures(self, render_ctx: RenderContext):
        ctx = render_ctx.ctx
        rc_pass = RadianceCascadesPass(ctx, width=320, height=240)

        # Half resolution (160x120)
        assert rc_pass.width == 160
        assert rc_pass.height == 120
        assert rc_pass.raw_rc_tex.size == (160, 120)
        assert rc_pass.tex_a.size == (160, 120)
        assert rc_pass.tex_b.size == (160, 120)
        assert rc_pass.raw_rc_tex.dtype == "f2"

        # Resize test
        rc_pass.resize(640, 480)
        assert rc_pass.width == 320
        assert rc_pass.height == 240
        assert rc_pass.raw_rc_tex.size == (320, 240)

        rc_pass.destroy()

    def test_pass_execution_in_pipeline(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.gi_mode = GIMode.RADIANCE_CASCADES.value

        ecs = EntityManager(max_entities=10)
        # Create floor box
        ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(10.0, 0.2, 10.0),
            color=(0.7, 0.7, 0.7),
        )
        # Create red cube for color bleed
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.95, 0.10, 0.10),
            roughness=0.2,
            metallic=0.0,
        )

        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )

        # Verify filtered RC texture has data
        rc_tex = pipeline.rc_pass._current_filtered
        data = rc_tex.read()
        assert len(data) == 160 * 120 * 4 * 2  # RGBA16F (2 bytes per component)
        pixel_array = np.frombuffer(data, dtype=np.float16)
        assert np.any(pixel_array > 0.0)

        # Test debug mode 13 (Radiance Cascades visualization)
        pipeline.config.debug_gbuffer = 13
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.2,
        )
        final_pixels = pipeline.post_process.final_texture.read()
        final_arr = np.frombuffer(final_pixels, dtype=np.uint8)
        assert np.any(final_arr > 0)

        pipeline.destroy()

    def test_mutual_exclusivity(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=5)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.8, 0.4, 0.2))

        # 1. In RADIANCE_CASCADES mode:
        pipeline.config.gi_mode = GIMode.RADIANCE_CASCADES.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )
        assert pipeline._u_gi_enabled.value == 4
        # Verify Radiance Cascades ran
        assert pipeline.rc_pass._first_frame is False

        # 2. In HYBRID mode (Classic LPV + SSGI):
        pipeline.config.gi_mode = GIMode.HYBRID.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.2,
        )
        assert pipeline._u_gi_enabled.value == 3

        # 3. In OFF mode:
        pipeline.config.gi_mode = GIMode.OFF.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.3,
        )
        assert pipeline._u_gi_enabled.value == 0

        pipeline.destroy()
