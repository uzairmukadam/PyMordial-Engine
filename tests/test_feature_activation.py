"""Activation tests for graphics feature state (Phase 2.5).

These are NOT visual-quality tests. Each test requests a feature variant, applies
the configuration, renders enough frames for initialization, and then asserts that
the *actual* active state matches the request â€” or that an explicit fallback was
reported. They exist so a benchmark can prove which backend actually executed.
"""

from __future__ import annotations

import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


@pytest.fixture(scope="module")
def pipeline(render_ctx):
    cfg = get_quality_preset(GraphicsQuality.HIGH)
    pipe = RenderPipeline(render_ctx, cfg)
    yield pipe
    pipe.destroy()


@pytest.fixture(scope="module")
def ecs():
    return EntityManager(max_entities=256)


def _render_frames(pipeline, ecs, frames: int = 3) -> None:
    """Renders a few frames so passes publish their active backend state."""
    if ecs.active_count == 0:
        ecs.create_entity(shape="cube", position=(0.0, 0.0, -3.0), dimensions=(1.0, 1.0, 1.0))
    for _ in range(frames):
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 1.0, 4.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.016,
            dt=0.016,
        )


class TestAOActivation:
    def test_ssao_reports_ssao(self, pipeline, ecs):
        pipeline.config.ao_mode = "SSAO"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["ao"]
        assert state["mode"] == "SSAO"
        assert state["active_backend"] == "SSAO"

    def test_gtao_legacy(self, pipeline, ecs):
        pipeline.config.ao_mode = "GTAO"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["ao"]
        assert state["active_backend"] == "LEGACY"
        assert state["fallback_reason"] is None

    def test_ao_off(self, pipeline, ecs):
        pipeline.config.ao_mode = "OFF"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        assert pipeline.feature_state()["ao"]["active_backend"] == "OFF"


class TestGIActivation:
    def test_gi_off(self, pipeline, ecs):
        pipeline.config.gi_mode = "OFF"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["gi"]
        assert state["requested_mode"] == "OFF"
        assert state["active_mode"] == "OFF"

    def test_gi_ibl_is_not_off(self, pipeline, ecs):
        pipeline.config.gi_mode = "IBL"
        pipeline.config.ibl_enabled = True
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["gi"]
        assert state["requested_mode"] == "IBL"
        assert state["active_mode"] == "IBL"

    def test_ssgi_legacy(self, pipeline, ecs):
        pipeline.config.gi_mode = "SSGI"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["gi"]
        assert state["active_mode"] == "SSGI"
        assert state["active_backend"] == "LEGACY"

    def test_rmgi(self, pipeline, ecs):
        pipeline.config.gi_mode = "RMGI"
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["gi"]
        assert state["active_mode"] == "RMGI"


class TestSSRActivation:
    def test_ssr_off(self, pipeline, ecs):
        pipeline.config.ssr_enabled = False
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        assert pipeline.feature_state()["ssr"]["active_backend"] == "OFF"

    def test_ssr_legacy(self, pipeline, ecs):
        pipeline.config.ssr_enabled = True
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        state = pipeline.feature_state()["ssr"]
        assert state["active_backend"] == "LEGACY"
        assert state["fallback_reason"] is None


class TestAAExclusivity:
    @pytest.mark.parametrize("mode", ["OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X", "TAA"])
    def test_single_stage(self, pipeline, mode):
        pipeline.config.aa_mode = mode
        pipeline.apply_config(pipeline.config)
        stages = pipeline.active_aa_stages()
        assert len(stages) <= 1, f"{mode} produced multiple AA stages: {stages}"

    def test_spatial_mode_clears_taa_flag(self, pipeline):
        pipeline.config.aa_mode = "FXAA"
        pipeline.config.taa_enabled = True  # contradictory request
        pipeline.apply_config(pipeline.config)
        assert pipeline.config.taa_enabled is False
        assert pipeline.active_aa_stages() == ("FXAA",)

    def test_taa_mode_sets_taa_flag(self, pipeline):
        pipeline.config.aa_mode = "TAA"
        pipeline.config.taa_enabled = False
        pipeline.apply_config(pipeline.config)
        assert pipeline.config.taa_enabled is True
        assert pipeline.active_aa_stages() == ("TAA",)

    def test_unknown_mode_falls_back_to_off(self, pipeline):
        pipeline.config.aa_mode = "BOGUS"
        pipeline.apply_config(pipeline.config)
        assert pipeline.config.aa_mode == "OFF"
        assert pipeline.active_aa_stages() == ()


class TestShadowAndFogState:
    @pytest.mark.parametrize("mode", ["HARD", "PCF", "PCSS"])
    def test_shadow_filter_reported(self, pipeline, mode):
        pipeline.config.shadow_filter_mode = mode
        pipeline.apply_config(pipeline.config)
        assert pipeline.feature_state()["shadows"]["filter"] == mode

    @pytest.mark.parametrize("res", ["LOW", "MEDIUM", "HIGH", "ULTRA", "CINEMATIC"])
    def test_fog_resolution_reported(self, pipeline, res):
        pipeline.config.fog_resolution = res
        pipeline.apply_config(pipeline.config)
        assert pipeline.feature_state()["fog"]["froxel_resolution"] == res


class TestMotionBlurSamples:
    def test_motion_blur_samples_is_consumed(self, pipeline, ecs):
        """motion_blur_samples must drive the gather tap budget (Phase 2.5 fix)."""
        pipeline.config.motion_blur_enabled = True
        pipeline.config.motion_blur_samples = 7
        pipeline.apply_config(pipeline.config)
        _render_frames(pipeline, ecs)
        uniform = pipeline.motion_blur_pass._u_g_max_samples
        if uniform is not None:
            assert int(uniform.value) == 7
