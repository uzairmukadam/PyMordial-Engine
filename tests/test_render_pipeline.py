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
        assert low.shadow_distance < cinematic.shadow_distance


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

    def test_matrix_scratchpad_in_place(self):
        from engine.core.math_utils import matrix_look_at, matrix_perspective, mat4_mul, mat4_inv
        import math

        eye = (0.0, 5.0, 10.0)
        target = (0.0, 0.0, 0.0)

        # Look-At matrix
        mat_alloc = matrix_look_at(eye, target)
        out_buf = np.zeros(16, dtype=np.float32)
        res = matrix_look_at(eye, target, out=out_buf)
        assert res is out_buf
        np.testing.assert_allclose(out_buf, mat_alloc, atol=1e-6)

        # Perspective matrix
        proj_alloc = matrix_perspective(math.radians(60.0), 16.0 / 9.0, near=0.1, far=100.0, reverse_z=True)
        proj_out = np.zeros(16, dtype=np.float32)
        res_p = matrix_perspective(math.radians(60.0), 16.0 / 9.0, near=0.1, far=100.0, reverse_z=True, out=proj_out)
        assert res_p is proj_out
        np.testing.assert_allclose(proj_out, proj_alloc, atol=1e-6)

        # Matrix Multiply
        vp_alloc = mat4_mul(proj_alloc, mat_alloc)
        vp_out = np.zeros(16, dtype=np.float32)
        res_vp = mat4_mul(proj_alloc, mat_alloc, out=vp_out)
        assert res_vp is vp_out
        np.testing.assert_allclose(vp_out, vp_alloc, atol=1e-6)

        # Matrix Inverse
        inv_alloc = mat4_inv(mat_alloc)
        inv_out = np.zeros(16, dtype=np.float32)
        res_inv = mat4_inv(mat_alloc, out=inv_out)
        assert res_inv is inv_out
        np.testing.assert_allclose(inv_out, inv_alloc, atol=1e-5)

    def test_csm_shadow_normal_offset_bias(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(20.0, 0.5, 20.0),
            color=(0.4, 0.4, 0.4),
        )
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.95, 0.75, 0.25),
            roughness=0.3,
            metallic=0.5,
        )
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.5,
            sun_dir=(0.5, -0.5, 0.5),
            sun_lux=4.0,
        )
        data = pipeline.post_process.final_texture.read()
        assert len(data) == 320 * 240 * 4
        pipeline.destroy()

    def test_pipeline_dynamic_resize_and_preset_reconfig(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0))

        # Dynamic resize
        pipeline.resize(400, 300)
        assert pipeline.g_buffer.width == 400
        assert pipeline.g_buffer.height == 300
        assert pipeline.post_process.width == 400
        assert pipeline.post_process.height == 300

        # Dynamic quality preset change
        low_config = get_quality_preset(GraphicsQuality.LOW)
        pipeline.apply_config(low_config)
        assert pipeline.config.shadow_resolution == low_config.shadow_resolution
        assert pipeline.csm.atlas_size == low_config.shadow_resolution

        # Render frame with new dimensions and configuration
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
        )
        data = pipeline.post_process.final_texture.read()
        assert len(data) == 400 * 300 * 4
        pipeline.destroy()

    def test_volumetric_fog_resolution_scaling(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0))

        # Check default resolution (HIGH -> 160x90x64)
        fog_pass = pipeline.volumetric_fog_pass
        assert (fog_pass.grid_w, fog_pass.grid_h, fog_pass.grid_d) == (160, 90, 64)

        # Dynamically scale to LOW (80x45x32)
        fog_pass.set_grid_resolution(80, 45, 32)
        assert (fog_pass.grid_w, fog_pass.grid_h, fog_pass.grid_d) == (80, 45, 32)

        # Render frame with point lights and fog
        pipeline.add_point_light(position=(0.0, 2.0, 0.0), radius=10.0, color=(1.0, 0.8, 0.4), intensity=3.0)
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
        )

        # Scale to ULTRA (200x112x80) via config
        ultra_config = get_quality_preset(GraphicsQuality.ULTRA)
        pipeline.apply_config(ultra_config)
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
        )
        assert (fog_pass.grid_w, fog_pass.grid_h, fog_pass.grid_d) == (200, 112, 80)

        data = pipeline.post_process.final_texture.read()
        assert len(data) == render_ctx.width * render_ctx.height * 4
        pipeline.destroy()
