"""Unit tests for Phase 2 ModernGL 4.5 rendering pipeline, Reversed-Z, and Quality Presets."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.debug.monitor import SystemMonitor
from engine.gfx import (
    GraphicsQuality,
    HudOverlay,
    RenderConfig,
    RenderContext,
    RenderPipeline,
    get_quality_preset,
)


class TestQualityPresets:
    def test_presets_exist_and_reversed_z_default(self):
        for q in [
            GraphicsQuality.LOW,
            GraphicsQuality.MEDIUM,
            GraphicsQuality.HIGH,
            GraphicsQuality.ULTRA,
            GraphicsQuality.CINEMATIC,
        ]:
            cfg = get_quality_preset(q)
            assert isinstance(cfg, RenderConfig)
            assert cfg.reverse_z is True
            assert cfg.csm_cascades >= 1
            assert cfg.shadow_resolution >= 1024

    def test_preset_scaling(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)

        assert low.shadow_resolution < cinematic.shadow_resolution
        assert low.shadow_distance < cinematic.shadow_distance
        assert low.csm_cascades <= cinematic.csm_cascades


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

    def test_gpu_timer_queries_return_without_waiting(self, render_ctx: RenderContext):
        monitor = SystemMonitor()
        try:
            for _ in range(8):
                if monitor.begin_gpu_frame():
                    render_ctx.ctx.clear()
                    monitor.end_gpu_frame()
            if monitor.gpu_timing_supported:
                assert monitor.gpu_frame_time_ms is not None
                assert monitor.gpu_frame_time_ms >= 0.0
        finally:
            monitor.destroy()

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
        assert pipeline.config.shadow_resolution == 8192

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
        import math

        from engine.core.math_utils import mat4_inv, mat4_mul, matrix_look_at, matrix_perspective

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

    def test_render_context_resize_propagates_to_pipeline(self, render_ctx: RenderContext):
        """RenderContext.resize must resize every pipeline target, not just the viewport.

        Regression: growing only the GL viewport left the G-Buffer, AO and HDR
        targets at the old size, so the final blit covered just the bottom-left
        fraction of the window (AO appeared scaled down to ~half height).
        """
        pipeline = RenderPipeline(render_ctx)
        render_ctx.pipeline = pipeline

        old_w, old_h = pipeline.g_buffer.width, pipeline.g_buffer.height
        new_w, new_h = old_w * 2, old_h * 2

        render_ctx.resize(new_w, new_h)

        assert pipeline.g_buffer.width == new_w
        assert pipeline.g_buffer.height == new_h
        assert pipeline.ao_pass.width == new_w
        assert pipeline.ao_pass.height == new_h
        assert pipeline.ao_pass.blur_pass.width == new_w
        assert pipeline.ao_pass.blur_pass.height == new_h
        assert pipeline.post_process.width == new_w
        assert pipeline.post_process.height == new_h
        assert pipeline.post_process.hdr_texture.size == (new_w, new_h)
        assert pipeline.post_process.final_fbo.size == (new_w, new_h)
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

    def test_destroy_releases_water_spot_and_particle(self, render_ctx: RenderContext):
        """Stage 1: pipeline.destroy() must release water, spot shadows, particles.

        Regression: these pipeline-owned objects were previously omitted from
        destroy(), leaking their GPU resources. Their destroy() methods are
        idempotent, so a second destroy() must not raise.
        """
        pipeline = RenderPipeline(render_ctx)
        assert pipeline.water_pass is not None
        assert pipeline.spot_shadow_map is not None
        assert pipeline.particle_pass is not None

        pipeline.destroy()

        # Idempotent: the objects null their handles, so a second call is safe.
        pipeline.water_pass.destroy()
        pipeline.spot_shadow_map.destroy()
        pipeline.particle_pass.destroy()


class TestSSRSpecularCompose:
    """Stage 4b: the second full deferred resolve is replaced by a lightweight
    SSR specular compose pass. These tests verify pass order, resource flow,
    configuration behavior, and lifecycle without brittle source snapshots.
    """

    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def _scene(self):
        ecs = EntityManager(max_entities=20)
        ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(20.0, 0.5, 20.0), color=(0.3, 0.7, 0.3))
        ecs.create_entity(position=(0.0, 0.5, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.9, 0.9),
                          roughness=0.05, metallic=1.0)
        return ecs

    def test_compose_pass_exists_and_owns_program(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.ssr_compose_pass is not None
            assert pipeline.ssr_compose_pass.prog is not None
            assert pipeline.ssr_compose_pass.vao is not None
        finally:
            pipeline.destroy()

    def test_ssr_enabled_runs_compose_and_single_base_resolve(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = True
            pipeline.apply_config(pipeline.config)
            calls = {"base": 0, "compose": 0}
            orig_run = RenderPipeline._run_deferred_resolve
            orig_compose = pipeline.ssr_compose_pass.execute

            def spy_run(self, ctx, w, h, env, sun, target_fbo, ssr_enabled, profiler_label="Deferred resolve"):
                if profiler_label == "Deferred base resolve":
                    calls["base"] += 1
                return orig_run(self, ctx, w, h, env, sun, target_fbo, ssr_enabled, profiler_label)

            def spy_compose(context):
                calls["compose"] += 1
                return orig_compose(context)

            RenderPipeline._run_deferred_resolve = spy_run
            pipeline.ssr_compose_pass.execute = spy_compose
            try:
                pipeline.render_frame(ecs=self._scene(), camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            finally:
                RenderPipeline._run_deferred_resolve = orig_run
            # Exactly one full deferred resolve (the base), and one compose.
            assert calls["base"] == 1
            assert calls["compose"] == 1
        finally:
            pipeline.destroy()

    def test_ssr_disabled_skips_compose(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = False
            pipeline.apply_config(pipeline.config)
            calls = {"compose": 0}
            orig_compose = pipeline.ssr_compose_pass.execute

            def spy_compose(context):
                calls["compose"] += 1
                return orig_compose(context)

            pipeline.ssr_compose_pass.execute = spy_compose
            pipeline.render_frame(ecs=self._scene(), camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            assert calls["compose"] == 0
        finally:
            pipeline.destroy()

    def test_compose_reads_base_hdr_and_writes_distinct_target(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = True
            pipeline.apply_config(pipeline.config)
            seen = {}
            orig_compose = pipeline.ssr_compose_pass.execute

            def spy_compose(context):
                seen["base_hdr_texture"] = context.resources.get("base_hdr_texture")
                seen["compose_target_fbo"] = context.resources.get("compose_target_fbo")
                seen["ssr_texture"] = context.resources.get("ssr_texture")
                return orig_compose(context)

            pipeline.ssr_compose_pass.execute = spy_compose
            pipeline.render_frame(ecs=self._scene(), camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            # Compose reads the base HDR and SSR output, writes to a different FBO.
            assert seen["base_hdr_texture"] is pipeline.post_process.base_hdr_texture
            assert seen["compose_target_fbo"] is pipeline.post_process.hdr_fbo
            assert seen["ssr_texture"] is not None
            # Source and destination are distinct textures (no feedback).
            assert pipeline.post_process.base_hdr_texture is not pipeline.post_process.hdr_texture
        finally:
            pipeline.destroy()

    def test_ssr_samples_base_hdr_not_composed(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = True
            pipeline.apply_config(pipeline.config)
            seen = {}
            orig_ssr = pipeline.ssr_pass.execute

            def spy_ssr(context):
                seen["base_hdr"] = context.resources.get("base_hdr")
                return orig_ssr(context)

            pipeline.ssr_pass.execute = spy_ssr
            pipeline.render_frame(ecs=self._scene(), camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            # SSR must trace against the pre-SSR base HDR, never the composed HDR.
            assert seen["base_hdr"] is pipeline.post_process.base_hdr_texture
        finally:
            pipeline.destroy()

    def test_compose_resize_and_destroy(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.ssr_compose_pass.resize(400, 300)
            assert pipeline.ssr_compose_pass.width == 400
            assert pipeline.ssr_compose_pass.height == 300
        finally:
            pipeline.destroy()
