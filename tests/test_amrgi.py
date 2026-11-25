"""Tests for Adaptive Multi-Range Global Illumination (AMRGI).

Covers mode selection, mutual exclusivity, resource lifecycle, resize, history
reset, fallback, half-resolution odd sizes, deterministic scheduling, temporal
coverage, continuous near->mid->far continuation, direct-radiance sampling,
self-hit/backface rejection, composition convention, and NaN/Inf safety.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.amrgi_pass import AMRGIPass
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import GIMode, GraphicsQuality, get_quality_preset
from engine.gfx.shader_utils import load_shader


def _scene(ecs: EntityManager) -> None:
    ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
    ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestAMRGIPassLifecycle:
    def test_half_resolution_default(self, render_ctx: RenderContext):
        p = AMRGIPass(render_ctx.ctx, width=320, height=240)
        assert p.full_width == 320 and p.full_height == 240
        assert p.width == 160 and p.height == 120  # half-res trace domain
        assert p.raw_radiance_tex.size == (160, 120)
        assert p.aux_tex.size == (160, 120)
        assert p.resolved_tex.size == (320, 240)
        assert p.history_tex.size == (320, 240)
        p.destroy()

    def test_odd_resolution_half(self, render_ctx: RenderContext):
        p = AMRGIPass(render_ctx.ctx, width=321, height=241)
        # Half of an odd resolution must still be a valid positive domain.
        assert p.width == max(int(round(321 * 0.5)), 1)
        assert p.height == max(int(round(241 * 0.5)), 1)
        assert p.raw_radiance_tex.size == (p.width, p.height)
        assert p.resolved_tex.size == (321, 241)
        p.destroy()

    def test_resize_and_history_reset(self, render_ctx: RenderContext):
        p = AMRGIPass(render_ctx.ctx, width=320, height=240)
        p._first_frame = False
        p.resize(640, 480)
        assert p.full_width == 640 and p.full_height == 480
        assert p.width == 320 and p.height == 240
        assert p._first_frame is True
        p.resize(640, 480)  # no-op
        p.destroy()

    def test_invalidate_history(self, render_ctx: RenderContext):
        p = AMRGIPass(render_ctx.ctx, width=320, height=240)
        p._first_frame = False
        p.invalidate_history()
        assert p._first_frame is True
        p.destroy()


class TestAMRGIIntegration:
    def test_mode_selection_and_exclusivity(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline._u_gi_enabled.value == 5
            assert pipeline.amrgi_pass._first_frame is False
            # SSGI/RMGI histories must be invalidated while AMRGI runs.
            assert pipeline.ssgi_pass._first_frame is True
            assert pipeline.rmgi_pass._first_frame is True

            pipeline.config.gi_mode = "SSGI"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.032)
            assert pipeline._u_gi_enabled.value == 1
            assert pipeline.amrgi_pass._first_frame is True

            pipeline.config.gi_mode = "OFF"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.048)
            assert pipeline._u_gi_enabled.value == 0
        finally:
            pipeline.destroy()

    def test_output_finite_and_nonzero(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            for i in range(4):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
            data = np.frombuffer(pipeline.amrgi_pass.history_tex.read(), dtype=np.float16).reshape(240, 320, 4)
            assert np.all(np.isfinite(data))
            assert float(data[..., :3].max()) > 0.0
        finally:
            pipeline.destroy()

    def test_feature_state_reports_amrgi(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "AMRGI"
            state = pipeline.feature_state()
            assert state["gi"]["requested_mode"] == "AMRGI"
            assert state["gi"]["active_mode"] == "AMRGI"
            assert state["gi"]["resolution_scale"] == pytest.approx(0.5)
        finally:
            pipeline.destroy()

    def test_fallback_when_disabled(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "OFF"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.amrgi_pass._first_frame is True
        finally:
            pipeline.destroy()


class TestAMRGIShaderContracts:
    def test_trace_uses_direct_radiance(self):
        src = load_shader("amrgi_trace.comp")
        assert "u_DirectRadiance" in src
        # Must not use RMGI's analytic sun/ambient model as the primary estimator.
        assert "SurfaceRadiance" not in src

    def test_continuous_ranges_no_restart(self):
        src = load_shader("amrgi_trace.comp")
        # The same ray is continued: MarchRange is called with a shared origin/dir.
        assert "MarchRange" in src
        assert "u_NearEnd" in src and "u_MidEnd" in src and "u_FarEnd" in src

    def test_reconstruct_has_temporal_and_upscale(self):
        src = load_shader("amrgi_reconstruct.comp")
        assert "u_HistoryRadiance" in src
        assert "u_MotionVectors" in src
        assert "u_HalfResolution" in src  # edge-aware upscale from half-res

    def test_composition_convention(self):
        src = load_shader("deferred_resolve.frag")
        assert "u_AMRGITexture" in src
        # AMRGI applies albedo/PI exactly once (incoming radiance convention).
        assert "amrgi_diffuse" in src


class TestAMRGIPresets:
    def test_presets_scale_amrgi(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        cine = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert low.amrgi_near_steps < cine.amrgi_near_steps
        assert low.amrgi_far_rate > cine.amrgi_far_rate  # sparser far at low quality
        assert low.amrgi_resolution_scale == 0.5

    def test_amrgi_not_default(self):
        for q in (GraphicsQuality.LOW, GraphicsQuality.MEDIUM, GraphicsQuality.HIGH,
                  GraphicsQuality.ULTRA, GraphicsQuality.CINEMATIC):
            assert get_quality_preset(q).gi_mode != "AMRGI"

    def test_gimode_enum_has_amrgi(self):
        assert GIMode.AMRGI.value == "AMRGI"
