"""Tests for the P0 auto-exposure pass (Pass 2).

The controller is a GPU histogram plus a temporal adaptation step. These tests
verify the observable contract:

* Disabled by default, so the static ``RenderConfig.exposure`` is unchanged.
* Converges to a stable EV on a static scene.
* Adapts in the correct direction when the scene brightens or darkens.
* Clamps to the configured EV range.
* Publishes ``exposure_ev`` for the post-process pass to consume.
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.exposure_pass import HISTOGRAM_BINS, ExposurePass
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import RenderConfig


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _scene() -> EntityManager:
    ecs = EntityManager(max_entities=10)
    ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(20.0, 0.5, 20.0), color=(0.4, 0.4, 0.4))
    ecs.create_entity(
        position=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        color=(0.95, 0.75, 0.25),
        roughness=0.3,
        metallic=0.5,
    )
    return ecs


def _run(pipeline: RenderPipeline, ecs: EntityManager, frames: int, sun_lux: float = 4.0) -> None:
    for i in range(frames):
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1 + i / 60.0,
            sun_dir=(0.3, -0.85, -0.4),
            sun_lux=sun_lux,
            dt=1.0 / 60.0,
        )


class TestExposureConfig:
    def test_disabled_by_default(self):
        assert RenderConfig().auto_exposure_enabled is False

    def test_defaults_are_sane(self):
        cfg = RenderConfig()
        assert 0.0 < cfg.exposure_key < 1.0
        assert cfg.exposure_min_ev < cfg.exposure_max_ev
        assert cfg.exposure_adapt_speed_up > 0.0
        assert cfg.exposure_adapt_speed_down > 0.0


class TestExposurePass:
    def test_histogram_buffer_is_256_bins(self, render_ctx):
        assert HISTOGRAM_BINS == 256
        pass_ = ExposurePass(render_ctx.ctx)
        try:
            assert pass_.histogram_buffer.size == HISTOGRAM_BINS * 4
        finally:
            pass_.destroy()

    def test_disabled_publishes_none(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=False))
        try:
            _run(pipeline, _scene(), 2)
            assert pipeline.exposure_pass.adapted_ev == 0.0
        finally:
            pipeline.destroy()

    def test_converges_on_a_static_scene(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            _run(pipeline, _scene(), 30)
            ep = pipeline.exposure_pass
            assert abs(ep.adapted_ev - ep.target_ev) < 0.05, (
                f"exposure did not converge: adapted={ep.adapted_ev:.3f} target={ep.target_ev:.3f}"
            )
            assert ep.key_luminance > 0.0
        finally:
            pipeline.destroy()

    def test_brightening_lowers_exposure(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            ecs = _scene()
            _run(pipeline, ecs, 120, sun_lux=4.0)
            dim_ev = pipeline.exposure_pass.adapted_ev

            _run(pipeline, ecs, 120, sun_lux=40.0)
            bright_ev = pipeline.exposure_pass.adapted_ev

            # A 10x brighter scene must reduce exposure by roughly log2(10).
            assert bright_ev < dim_ev - 2.0, (dim_ev, bright_ev)
        finally:
            pipeline.destroy()

    def test_adapts_back_when_scene_darkens(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            ecs = _scene()
            _run(pipeline, ecs, 120, sun_lux=40.0)
            bright_ev = pipeline.exposure_pass.adapted_ev

            _run(pipeline, ecs, 120, sun_lux=4.0)
            recovered_ev = pipeline.exposure_pass.adapted_ev

            assert recovered_ev > bright_ev + 1.0, (bright_ev, recovered_ev)
        finally:
            pipeline.destroy()

    def test_respects_ev_clamp(self, render_ctx):
        cfg = RenderConfig(
            auto_exposure_enabled=True,
            exposure_min_ev=-1.0,
            exposure_max_ev=1.0,
        )
        pipeline = RenderPipeline(render_ctx, config=cfg)
        try:
            _run(pipeline, _scene(), 30, sun_lux=400.0)
            ev = pipeline.exposure_pass.adapted_ev
            assert -1.0 - 1e-3 <= ev <= 1.0 + 1e-3, ev
        finally:
            pipeline.destroy()

    def test_exposure_multiplier_matches_ev(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            _run(pipeline, _scene(), 10)
            ep = pipeline.exposure_pass
            assert abs(ep.exposure_multiplier - 2.0**ep.adapted_ev) < 1e-5
        finally:
            pipeline.destroy()

    def test_auto_exposure_changes_the_final_image(self, render_ctx):
        """The adapted EV must actually reach the post-process pass."""
        static = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=False))
        auto = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            _run(static, _scene(), 20)
            _run(auto, _scene(), 20)
            a = np.frombuffer(static.post_process.final_texture.read(), dtype=np.uint8)
            b = np.frombuffer(auto.post_process.final_texture.read(), dtype=np.uint8)
            assert np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() > 1.0
        finally:
            static.destroy()
            auto.destroy()

    def test_invalidate_history_snaps_to_target(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            ecs = _scene()
            _run(pipeline, ecs, 30, sun_lux=4.0)
            pipeline.exposure_pass.invalidate_history()
            _run(pipeline, ecs, 1, sun_lux=40.0)
            ep = pipeline.exposure_pass
            # After invalidation the first frame snaps straight to the target.
            assert abs(ep.adapted_ev - ep.target_ev) < 0.05
        finally:
            pipeline.destroy()
