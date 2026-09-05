"""Unit tests for Phase 2 ModernGL 4.5 rendering pipeline, Reversed-Z, and Quality Presets."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx import (
    GraphicsQuality,
    RenderConfig,
    get_quality_preset,
    RenderContext,
    RenderPipeline,
    HudOverlay,
)


class TestQualityPresets:
    def test_presets_exist_and_reversed_z_default(self):
        for q in [GraphicsQuality.LOW, GraphicsQuality.MEDIUM, GraphicsQuality.HIGH, GraphicsQuality.ULTRA, GraphicsQuality.CINEMATIC]:
            cfg = get_quality_preset(q)
            assert isinstance(cfg, RenderConfig)
            assert cfg.reverse_z is True
            assert cfg.csm_cascades >= 1
            assert cfg.shadow_resolution >= 1024

    def test_preset_scaling(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)

        assert low.shadow_resolution < cinematic.shadow_resolution
        assert low.pcf_samples <= cinematic.pcf_samples
        assert low.sscs_steps < cinematic.sscs_steps


class TestRenderPipelineHeadless:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_context_creation_and_reversed_z(self, render_ctx: RenderContext):
        assert render_ctx.ctx is not None
        # Verify depth func configured for Reversed-Z (GL_GREATER)
        assert render_ctx.depth_func_name == ">"

    def test_pipeline_render_frame_and_fbo_output(self, render_ctx: RenderContext):
        ecs = EntityManager(max_entities=100)

        # Entity 0: Ground Box
        ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(20.0, 0.5, 20.0),
            color=(0.3, 0.7, 0.3),
        )

        # Entity 1..5: PBR Spheres
        for i in range(5):
            ecs.create_entity(
                position=(float(i * 2.0 - 4.0), 1.0, 0.0),
                scale=(0.5, 0.5, 0.5),
                color=(0.8, 0.2, 0.2),
                roughness=0.1 + i * 0.2,
                metallic=0.0 + i * 0.2,
            )

        pipeline = RenderPipeline(render_ctx)

        # Render single frame
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 4.0, 8.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.5,
        )

        assert pipeline.output_texture_id > 0

        # Read back rendered pixels from final output texture
        raw_pixels = pipeline.post_process.final_texture.read()
        assert len(raw_pixels) == 320 * 240 * 4
        # Assert scene is not pitch black (rendered sky gradient & lit geometry)
        pixel_array = np.frombuffer(raw_pixels, dtype=np.uint8)
        assert np.any(pixel_array > 0)

        # Test dynamic quality preset switching
        ultra_config = get_quality_preset(GraphicsQuality.ULTRA)
        pipeline.apply_config(ultra_config)
        assert pipeline.config.shadow_resolution == 4096

        # Re-render with new preset
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 4.0, 8.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=1.0,
        )

        pipeline.destroy()

    def test_hud_overlay_render(self):
        render_ctx = RenderContext.create_headless(width=320, height=240)
        hud = HudOverlay(render_ctx.ctx, screen_width=320, screen_height=240, panel_width=200, panel_height=200)

        # Render HUD frame
        hud.render(
            fps=60.0,
            frame_time_ms=16.6,
            preset_name="HIGH",
            entity_count=42,
            cam_dist=15.0,
            cam_yaw=45.0,
            cam_pitch=30.0,
            sun_angle=1.2,
            tonemap_mode="ACES",
            status_message="Test Status",
            status_time=100.0,
        )

        assert hud.texture is not None
        hud.destroy()
        render_ctx.destroy()
