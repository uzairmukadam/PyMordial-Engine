"""Unit and Integration Tests for Ray-Marched Global Illumination (RMGI).

Verifies:
1. RMGIPass lifecycle, compute shader compilation, texture allocation, and resize.
2. Execution of SSRC interval raymarching with FFPC far-field probe integration.
3. Strict mutual exclusivity between SSGI and Ray-Marched Global Illumination (RMGI).
4. Full render pipeline frame pass with G-Buffer debug mode 13 (RMGI visualization).
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.rmgi_pass import RMGIPass
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import GIMode, GraphicsQuality, get_quality_preset


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestRMGIPass:
    def test_pass_initialization_and_textures(self, render_ctx: RenderContext):
        ctx = render_ctx.ctx
        rmgi_pass = RMGIPass(ctx, width=320, height=240)

        # Full resolution by default
        assert rmgi_pass.width == 320
        assert rmgi_pass.height == 240
        assert rmgi_pass.raw_rmgi_tex.size == (320, 240)
        assert rmgi_pass.tex_a.size == (320, 240)
        assert rmgi_pass.tex_b.size == (320, 240)
        assert rmgi_pass.raw_rmgi_tex.dtype == "f2"

        # Resize test
        rmgi_pass.resize(640, 480)
        assert rmgi_pass.width == 640
        assert rmgi_pass.height == 480
        assert rmgi_pass.raw_rmgi_tex.size == (640, 480)

        # Explicit downscale still honoured
        rmgi_pass.resize(640, 480, 0.5)
        assert rmgi_pass.width == 320
        assert rmgi_pass.height == 240
        assert rmgi_pass.raw_rmgi_tex.size == (320, 240)

        rmgi_pass.destroy()

    def test_pass_execution_in_pipeline(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.gi_mode = GIMode.RMGI.value
        pipeline.config.rmgi_ray_count = 12

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
            # Pin the sun so the red cube's camera-facing side is lit. Without this
            # the assertion below would depend on the atmosphere's solar arc, which
            # is a separate concern from RMGI colour bleed.
            sun_dir=(0.3, -0.85, -0.4),
        )

        # Verify filtered RC texture has data
        assert pipeline.rmgi_pass._u_rm_ray_count.value == 12
        rmgi_tex = pipeline.rmgi_pass._current_history
        data = rmgi_tex.read()
        assert len(data) == 320 * 240 * 4 * 2  # RGBA16F (2 bytes per component)
        pixel_array = np.frombuffer(data, dtype=np.float16).reshape(240, 320, 4)
        assert np.any(pixel_array > 0.0)
        # The red object must tint nearby diffuse bounce instead of only contributing emissive light.
        assert pixel_array[..., 0].mean() > pixel_array[..., 1].mean() * 1.5
        surface_ids = np.frombuffer(pipeline.g_buffer.surface_id_texture.read(), dtype=np.float32)
        assert np.any(surface_ids == 1.0)
        assert np.any(surface_ids == 2.0)

        raw_frames = [
            np.frombuffer(pipeline.rmgi_pass.raw_rmgi_tex.read(), dtype=np.float16)
            .reshape(240, 320, 4)[..., :3]
            .astype(np.float32)
        ]
        filtered_frames = [pixel_array[..., :3].astype(np.float32)]
        raw_spatial_delta = np.mean(np.abs(raw_frames[0][1:] - raw_frames[0][:-1])) + np.mean(
            np.abs(raw_frames[0][:, 1:] - raw_frames[0][:, :-1])
        )
        filtered_spatial_delta = np.mean(np.abs(filtered_frames[0][1:] - filtered_frames[0][:-1])) + np.mean(
            np.abs(filtered_frames[0][:, 1:] - filtered_frames[0][:, :-1])
        )
        assert filtered_spatial_delta < raw_spatial_delta * 0.50
        raw_luminance = raw_frames[0] @ np.array((0.2126, 0.7152, 0.0722))
        assert raw_luminance.max() <= 2.01

        for frame in range(1, 4):
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 4.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.1 + frame / 30.0,
            )
            raw_frames.append(
                np.frombuffer(pipeline.rmgi_pass.raw_rmgi_tex.read(), dtype=np.float16)
                .reshape(240, 320, 4)[..., :3]
                .astype(np.float32)
            )
            filtered_frames.append(
                np.frombuffer(pipeline.rmgi_pass._current_history.read(), dtype=np.float16)
                .reshape(240, 320, 4)[..., :3]
                .astype(np.float32)
            )

        raw_frame_delta = np.mean(np.abs(raw_frames[-1] - raw_frames[-2]))
        filtered_frame_delta = np.mean(np.abs(filtered_frames[-1] - filtered_frames[-2]))
        assert filtered_frame_delta < raw_frame_delta * 0.5

        # Test debug mode 13 (RMGI visualization)
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

    def test_quality_presets_scale_rmgi_ray_count(self):
        assert get_quality_preset(GraphicsQuality.LOW).rmgi_ray_count == 8
        assert get_quality_preset(GraphicsQuality.MEDIUM).rmgi_ray_count == 12
        assert get_quality_preset(GraphicsQuality.HIGH).rmgi_ray_count == 16
        assert get_quality_preset(GraphicsQuality.ULTRA).rmgi_ray_count == 24
        assert get_quality_preset(GraphicsQuality.CINEMATIC).rmgi_ray_count == 32

    def test_mutual_exclusivity(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=5)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.8, 0.4, 0.2))

        # 1. In RMGI mode:
        pipeline.config.gi_mode = GIMode.RMGI.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )
        assert pipeline._u_gi_enabled.value == 4
        # Verify RMGI ran
        assert pipeline.rmgi_pass._first_frame is False

        # 2. In SSGI mode:
        pipeline.config.gi_mode = GIMode.SSGI.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.2,
        )
        assert pipeline._u_gi_enabled.value == 1

        # 3. In OFF mode:
        pipeline.config.gi_mode = GIMode.OFF.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.3,
        )
        assert pipeline._u_gi_enabled.value == 0
        assert pipeline.rmgi_pass._first_frame is True

        pipeline.destroy()

class TestRMGITraversalBackends:
    """Both traversal backends must run and share the same downstream output."""

    @pytest.fixture
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def _scene(self, ecs: EntityManager) -> None:
        ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))

    def _render(self, pipeline, ecs, frames=4):
        for i in range(frames):
            pipeline.render_frame(
                ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1)
            )

    def test_default_traversal_is_linear(self):
        assert get_quality_preset(GraphicsQuality.HIGH).rmgi_traversal_mode == "LINEAR"

    def test_linear_mode_runs(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "RMGI"
            pipeline.config.rmgi_traversal_mode = "LINEAR"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            self._render(pipeline, ecs)
            assert pipeline.rmgi_pass._first_frame is False
            data = np.frombuffer(pipeline.rmgi_pass.raw_rmgi_tex.read(), dtype=np.float16)
            assert np.all(np.isfinite(data))
        finally:
            pipeline.destroy()

    def test_hiz_mode_runs(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "RMGI"
            pipeline.config.rmgi_traversal_mode = "HIZ"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            self._render(pipeline, ecs)
            assert pipeline.rmgi_pass._first_frame is False
            data = np.frombuffer(pipeline.rmgi_pass.raw_rmgi_tex.read(), dtype=np.float16)
            assert np.all(np.isfinite(data))
            assert data.max() >= 0.0
        finally:
            pipeline.destroy()

    def test_hiz_mode_uses_shared_hiz_bounds(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "RMGI"
            pipeline.config.rmgi_traversal_mode = "HIZ"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            self._render(pipeline, ecs, frames=2)
            # The shared bounds hierarchy must exist for RMGI to reuse.
            assert pipeline.hiz_pass.hiz_bounds_texture is not None
        finally:
            pipeline.destroy()

    def test_both_modes_share_downstream_output(self, render_ctx: RenderContext):
        """Switching traversal must not recreate the merge/history resources."""
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "RMGI"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)

            pipeline.config.rmgi_traversal_mode = "LINEAR"
            self._render(pipeline, ecs, frames=2)
            tex_a_id = id(pipeline.rmgi_pass.tex_a)
            tex_b_id = id(pipeline.rmgi_pass.tex_b)
            merge_id = id(pipeline.rmgi_pass.merge_prog)

            pipeline.config.rmgi_traversal_mode = "HIZ"
            self._render(pipeline, ecs, frames=2)
            assert id(pipeline.rmgi_pass.tex_a) == tex_a_id
            assert id(pipeline.rmgi_pass.tex_b) == tex_b_id
            assert id(pipeline.rmgi_pass.merge_prog) == merge_id
        finally:
            pipeline.destroy()

    def test_hiz_mode_handles_sky_pixels(self):
        """Hi-Z must not introduce non-finite output on background pixels.

        Compares against the linear backend in the same fresh context so the
        assertion is independent of any shared-context test-ordering effects.
        """
        def _nonfinite(mode):
            ctx = RenderContext.create_headless(width=320, height=240)
            pipeline = RenderPipeline(ctx)
            try:
                pipeline.config.gi_mode = "RMGI"
                pipeline.config.rmgi_traversal_mode = mode
                ecs = EntityManager(max_entities=10)
                # No geometry: every pixel is background.
                self._render(pipeline, ecs, frames=2)
                data = np.frombuffer(pipeline.rmgi_pass.raw_rmgi_tex.read(), dtype=np.float16)
                return int(np.count_nonzero(~np.isfinite(data)))
            finally:
                pipeline.destroy()
                ctx.destroy()

        linear_bad = _nonfinite("LINEAR")
        hiz_bad = _nonfinite("HIZ")
        # Hi-Z must be no worse than the reference linear path on sky pixels.
        assert hiz_bad <= linear_bad

    def test_disabling_rmgi_unchanged(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "OFF"
            pipeline.config.rmgi_traversal_mode = "HIZ"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            self._render(pipeline, ecs, frames=1)
            assert pipeline.rmgi_pass._first_frame is True
        finally:
            pipeline.destroy()
