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
        assert 0.0 <= cfg.exposure_low_percentile < cfg.exposure_high_percentile <= 1.0
        assert cfg.exposure_middle_gray > 0.0
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


class TestAutoExposureReachability:
    """Phase 3.3: the controller must be reachable from the debug UI and config.

    The original defect was not in the exposure math (which was correct) but in
    reachability: ``auto_exposure_enabled`` defaulted to False and no preset,
    debug control, or settings path could turn it on, so the feature was inert.
    """

    def test_engine_tweaks_exposes_auto_exposure_fields(self):
        from engine.debug.engine_tweaks import EngineTweaks

        et = EngineTweaks()
        assert et.auto_exposure_enabled is False
        assert et.exposure_low_percentile == pytest.approx(0.10)
        assert et.exposure_high_percentile == pytest.approx(0.90)
        assert et.exposure_middle_gray == pytest.approx(0.18)
        assert et.exposure_min_ev == pytest.approx(-8.0)
        assert et.exposure_max_ev == pytest.approx(12.0)
        assert et.exposure_adapt_speed_up == pytest.approx(1.5)
        assert et.exposure_adapt_speed_down == pytest.approx(4.0)

    def test_engine_tweaks_round_trips_auto_exposure(self):
        from engine.debug.engine_tweaks import EngineTweaks

        et = EngineTweaks()
        et.auto_exposure_enabled = True
        et.exposure_low_percentile = 0.05
        et.exposure_high_percentile = 0.95
        et.exposure_middle_gray = 0.22
        et.exposure_min_ev = -6.0
        et.exposure_max_ev = 10.0
        et.exposure_adapt_speed_up = 2.5
        et.exposure_adapt_speed_down = 5.5

        restored = EngineTweaks()
        restored.from_dict(et.to_dict())
        assert restored.auto_exposure_enabled is True
        assert restored.exposure_low_percentile == pytest.approx(0.05)
        assert restored.exposure_high_percentile == pytest.approx(0.95)
        assert restored.exposure_middle_gray == pytest.approx(0.22)
        assert restored.exposure_min_ev == pytest.approx(-6.0)
        assert restored.exposure_max_ev == pytest.approx(10.0)
        assert restored.exposure_adapt_speed_up == pytest.approx(2.5)
        assert restored.exposure_adapt_speed_down == pytest.approx(5.5)

    def test_engine_tweaks_applies_auto_exposure_to_pipeline_config(self):
        from engine.debug.engine_tweaks import EngineTweaks

        class _FakePipeline:
            def __init__(self):
                self.config = RenderConfig()

        et = EngineTweaks()
        et.auto_exposure_enabled = True
        et.exposure_low_percentile = 0.08
        et.exposure_high_percentile = 0.92
        et.exposure_middle_gray = 0.20
        et.exposure_min_ev = -5.0
        et.exposure_max_ev = 9.0
        et.exposure_adapt_speed_up = 3.0
        et.exposure_adapt_speed_down = 6.0

        pipeline = _FakePipeline()
        et.apply_to_pipeline(pipeline, force=True)
        cfg = pipeline.config
        assert cfg.auto_exposure_enabled is True
        assert cfg.exposure_low_percentile == pytest.approx(0.08)
        assert cfg.exposure_high_percentile == pytest.approx(0.92)
        assert cfg.exposure_middle_gray == pytest.approx(0.20)
        assert cfg.exposure_min_ev == pytest.approx(-5.0)
        assert cfg.exposure_max_ev == pytest.approx(9.0)
        assert cfg.exposure_adapt_speed_up == pytest.approx(3.0)
        assert cfg.exposure_adapt_speed_down == pytest.approx(6.0)

    def test_engine_tweaks_syncs_auto_exposure_from_config(self):
        from engine.debug.engine_tweaks import EngineTweaks

        cfg = RenderConfig()
        cfg.auto_exposure_enabled = True
        cfg.exposure_low_percentile = 0.15
        cfg.exposure_high_percentile = 0.85
        cfg.exposure_middle_gray = 0.16
        et = EngineTweaks()
        et.sync_from_config(cfg)
        assert et.auto_exposure_enabled is True
        assert et.exposure_low_percentile == pytest.approx(0.15)
        assert et.exposure_high_percentile == pytest.approx(0.85)
        assert et.exposure_middle_gray == pytest.approx(0.16)

    def test_reset_restores_auto_exposure_defaults(self):
        from engine.debug.engine_tweaks import EngineTweaks

        et = EngineTweaks()
        et.auto_exposure_enabled = True
        et.exposure_low_percentile = 0.3
        et.exposure_middle_gray = 0.4
        et.reset_to_defaults()
        assert et.auto_exposure_enabled is False
        assert et.exposure_low_percentile == pytest.approx(0.10)
        assert et.exposure_high_percentile == pytest.approx(0.90)
        assert et.exposure_middle_gray == pytest.approx(0.18)

    def test_enabling_auto_exposure_changes_final_image(self, render_ctx):
        """End-to-end: the toggle must actually reach the post-process."""
        static = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=False))
        auto = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=True))
        try:
            _run(static, _scene(), 20)
            _run(auto, _scene(), 20)
            assert auto.exposure_pass.adapted_ev != pytest.approx(0.0, abs=1e-3)
            a = np.frombuffer(static.post_process.final_texture.read(), dtype=np.uint8)
            b = np.frombuffer(auto.post_process.final_texture.read(), dtype=np.uint8)
            assert np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() > 1.0
        finally:
            static.destroy()
            auto.destroy()


class TestPercentileTrimmedMetering:
    """Phase 3.3B: percentile-trimmed weighted mean of log2 luminance.

    The previous single-bin percentile estimator produced step discontinuities:
    a two-mode scene flipping from 51% dark to 49% dark jumped the target by
    ~11.9 stops. These tests pin the continuous replacement.
    """

    W = 64
    H = 64
    DARK = 0.002
    MEDIUM = 0.18
    BRIGHT = 8.0
    EXTREME = 5000.0

    class _Ctx:
        def __init__(self, gl, config, tex, dt=1.0 / 60.0):
            self.ctx = gl
            self.config = config
            self.width = TestPercentileTrimmedMetering.W
            self.height = TestPercentileTrimmedMetering.H
            self.delta_time = dt
            self.resources = {"hdr_color": tex}

    def _tex(self, gl, values):
        """Build an HDR texture from a flat list of per-pixel luminance values."""
        n = self.W * self.H
        assert len(values) == n, f"expected {n} values, got {len(values)}"
        arr = np.asarray(values, dtype=np.float32).reshape(self.H, self.W)
        data = np.zeros((self.H, self.W, 4), dtype=np.float16)
        data[..., 0] = arr
        data[..., 1] = arr
        data[..., 2] = arr
        data[..., 3] = 1.0
        tex = gl.texture((self.W, self.H), 4, dtype="f2")
        tex.write(data.tobytes())
        return tex

    def _measure(self, gl, values, **cfg_overrides):
        cfg = RenderConfig(auto_exposure_enabled=True)
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        tex = self._tex(gl, values)
        ep = ExposurePass(gl)
        try:
            ep.execute(self._Ctx(gl, cfg, tex))
            return ep.key_luminance, ep.target_ev, ep.exposure_multiplier
        finally:
            ep.destroy()
            tex.release()

    def _mixed(self, dark_frac):
        n = self.W * self.H
        n_dark = int(round(n * dark_frac))
        return [self.DARK] * n_dark + [self.BRIGHT] * (n - n_dark)

    # ---- 1-3: constant inputs -------------------------------------------

    def test_constant_dark_input(self, render_ctx):
        key, target, mult = self._measure(render_ctx.ctx, [self.DARK] * (self.W * self.H))
        # One histogram bin spans 28/256 = 0.109 stops, i.e. +/-7.8% in luminance.
        assert key == pytest.approx(self.DARK, rel=0.10)
        assert target > 0.0
        assert mult > 1.0

    def test_constant_middle_gray_input(self, render_ctx):
        key, target, mult = self._measure(render_ctx.ctx, [self.MEDIUM] * (self.W * self.H))
        assert key == pytest.approx(self.MEDIUM, rel=0.10)
        # A middle-grey scene should land near unit exposure.
        assert abs(target) < 0.15
        assert mult == pytest.approx(1.0, abs=0.15)

    def test_constant_bright_input(self, render_ctx):
        key, target, mult = self._measure(render_ctx.ctx, [self.BRIGHT] * (self.W * self.H))
        assert key == pytest.approx(self.BRIGHT, rel=0.10)
        assert target < 0.0
        assert mult < 1.0

    # ---- 4-5: the key acceptance test -----------------------------------

    def test_51_dark_49_bright_is_continuous(self, render_ctx):
        key_a, target_a, mult_a = self._measure(render_ctx.ctx, self._mixed(0.51))
        key_b, target_b, mult_b = self._measure(render_ctx.ctx, self._mixed(0.49))
        # The old estimator jumped ~11.9 stops here. The trimmed mean must not.
        assert abs(target_b - target_a) < 1.0, (
            f"target jumped {abs(target_b - target_a):.3f} stops between 51/49 and 49/51"
        )
        # Brighter scene (49% dark) must not raise exposure.
        assert mult_b <= mult_a + 1e-3

    def test_49_dark_51_bright_direction(self, render_ctx):
        _, target_49, mult_49 = self._measure(render_ctx.ctx, self._mixed(0.49))
        _, target_51, mult_51 = self._measure(render_ctx.ctx, self._mixed(0.51))
        # More bright content -> lower target -> lower multiplier.
        assert target_49 < target_51
        assert mult_49 < mult_51

    # ---- 6-7: highlight robustness --------------------------------------

    def test_90_medium_10_extreme_highlight(self, render_ctx):
        n = self.W * self.H
        n_med = int(round(n * 0.9))
        values = [self.MEDIUM] * n_med + [self.EXTREME] * (n - n_med)
        key, target, mult = self._measure(render_ctx.ctx, values)
        # The 10% extreme tail is trimmed away, so the key stays near medium.
        assert key == pytest.approx(self.MEDIUM, rel=0.10)
        assert abs(target) < 0.2

    def test_10_very_dark_90_medium(self, render_ctx):
        n = self.W * self.H
        n_dark = int(round(n * 0.1))
        values = [self.DARK] * n_dark + [self.MEDIUM] * (n - n_dark)
        key, target, mult = self._measure(render_ctx.ctx, values)
        # The 10% dark tail is trimmed away, so the key stays near medium.
        assert key == pytest.approx(self.MEDIUM, rel=0.10)
        assert abs(target) < 0.2

    # ---- 8: empty / invalid histogram -----------------------------------

    def test_empty_histogram_preserves_previous_state(self, render_ctx):
        gl = render_ctx.ctx
        cfg = RenderConfig(auto_exposure_enabled=True)
        good = self._tex(gl, [self.BRIGHT] * (self.W * self.H))
        bad = gl.texture((self.W, self.H), 4, dtype="f2")
        data = np.zeros((self.H, self.W, 4), dtype=np.float16)
        data[..., :3] = np.nan
        data[..., 3] = 1.0
        bad.write(data.tobytes())

        ep = ExposurePass(gl)
        try:
            ctx = self._Ctx(gl, cfg, good)
            for _ in range(4):
                ep.execute(ctx)
            held = ep.adapted_ev
            assert np.isfinite(held)

            ctx.resources["hdr_color"] = bad
            for _ in range(4):
                ep.execute(ctx)
            assert ep.adapted_ev == pytest.approx(held, abs=1e-3)
            assert np.isfinite(ep.adapted_ev)
        finally:
            ep.destroy()
            good.release()
            bad.release()

    # ---- 9-10: adaptation monotonicity ----------------------------------

    def test_dark_to_bright_adaptation_is_monotonic(self, render_ctx):
        gl = render_ctx.ctx
        cfg = RenderConfig(auto_exposure_enabled=True)
        dark = self._tex(gl, [self.DARK] * (self.W * self.H))
        bright = self._tex(gl, [self.BRIGHT] * (self.W * self.H))
        ep = ExposurePass(gl)
        try:
            ctx = self._Ctx(gl, cfg, dark)
            ep.execute(ctx)
            start = ep.adapted_ev
            ctx.resources["hdr_color"] = bright
            history = []
            for _ in range(60):
                ep.execute(ctx)
                history.append(ep.adapted_ev)
            # Monotonic decrease toward the (lower) bright-scene target.
            assert all(b <= a + 1e-4 for a, b in zip(history, history[1:]))
            assert history[-1] < start
        finally:
            ep.destroy()
            dark.release()
            bright.release()

    def test_bright_to_dark_adaptation_is_monotonic(self, render_ctx):
        gl = render_ctx.ctx
        cfg = RenderConfig(auto_exposure_enabled=True)
        dark = self._tex(gl, [self.DARK] * (self.W * self.H))
        bright = self._tex(gl, [self.BRIGHT] * (self.W * self.H))
        ep = ExposurePass(gl)
        try:
            ctx = self._Ctx(gl, cfg, bright)
            ep.execute(ctx)
            start = ep.adapted_ev
            ctx.resources["hdr_color"] = dark
            history = []
            for _ in range(60):
                ep.execute(ctx)
                history.append(ep.adapted_ev)
            assert all(b >= a - 1e-4 for a, b in zip(history, history[1:]))
            assert history[-1] > start
        finally:
            ep.destroy()
            dark.release()
            bright.release()

    # ---- 11: manual exposure unchanged ----------------------------------

    def test_manual_exposure_unaffected_by_metering_change(self, render_ctx):
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(auto_exposure_enabled=False))
        try:
            _run(pipeline, _scene(), 5)
            assert pipeline.exposure_pass.adapted_ev == pytest.approx(0.0)
            assert pipeline.post_process._u_exposure.value == pytest.approx(
                pipeline.config.exposure
            )
        finally:
            pipeline.destroy()
