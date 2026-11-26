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
            pipeline.config.amrgi_resolution_scale = 0.5
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
        assert "amrgi_diffuse" in src

    def test_composition_applies_receiver_brdf_once(self):
        src = load_shader("deferred_resolve.frag")
        # The trace already returns a cosine-weighted mean of incoming radiance
        # (irradiance/PI). Cosine-weighted sampling cancels cos/pdf, so the
        # receiver BRDF is albedo only -- a second /PI would darken by PI.
        assert "amrgi_sample.rgb, vec3(0.0), vec3(30.0)) * albedo * (1.0 - metallic)" in src
        assert "amrgi_sample.rgb * (albedo / PI)" not in src
        assert "amrgi_sample.rgb * albedo / PI" not in src

    def test_firefly_knee_is_identity_below_threshold(self):
        src = load_shader("amrgi_trace.comp")
        # Firefly control must be a soft knee that leaves ordinary radiance
        # untouched; an unconditional 1/(1+lum*k) compresses every sample.
        assert "if (lum > u_RadianceClamp) L *= u_RadianceClamp / lum;" in src
        assert "1.0 / (1.0 + lum * 0.75)" not in src

    def test_diagnostic_views_present(self):
        src = load_shader("deferred_resolve.frag")
        # AMRGI diagnostic debug views (25-29) must exist.
        for mode in ("== 25", "== 26", "== 27", "== 28", "== 29"):
            assert mode in src
        assert "u_AMRGIRawTexture" in src
        assert "u_AMRGIAuxTexture" in src

    def test_ray_generator_uses_two_independent_values(self):
        src = load_shader("amrgi_trace.comp")
        # The cosine hemisphere must take two independent low-discrepancy inputs,
        # not derive both phi and cosTheta from one scalar (1D spiral bug).
        assert "CosineHemisphereRay(vec3 n, float u1, float u2)" in src

    def test_seed_uses_decorrelated_integer_hash(self):
        src = load_shader("amrgi_trace.comp")
        # AMRGI-5.1: the smooth gradient hash produced a visible checkerboard
        # (corr(rot,rot2) = -0.41, spatial autocorr lag1 = -0.5). Seeds must come
        # from an independent integer hash per (pixel, frame, dimension).
        assert "uint HashUint(uint x)" in src
        assert "float Hash01(uvec3 v)" in src
        assert "Hash01(uvec3(uint(pixel.x), uint(pixel.y), frameU * 2u + 0u))" in src
        assert "Hash01(uvec3(uint(pixel.x), uint(pixel.y), frameU * 2u + 1u))" in src
        # The old correlated derivation must be gone.
        assert "fract(noise + frameIndex * 0.61803398875)" not in src
        assert "fract(noise * 1.32471795724" not in src

    def test_seed_depends_on_pixel_frame_and_dimension(self):
        src = load_shader("amrgi_trace.comp")
        # rot and rot2 must differ only in the dimension term, so they are
        # independent but both vary with pixel and frame.
        assert "frameU * 2u + 0u" in src and "frameU * 2u + 1u" in src
        assert "uint(pixel.x)" in src and "uint(pixel.y)" in src

    def test_history_length_texture_published(self):
        src = open("engine/gfx/passes/amrgi_pass.py", encoding="utf-8").read()
        assert 'context.resources["amrgi_length_texture"] = self.length_tex' in src

    def test_thickness_scales_with_step(self):
        src = load_shader("amrgi_trace.comp")
        # Thickness must cover the inter-sample gap so coarse steps cannot skip surfaces.
        assert "step_len * 1.35" in src

    def test_spatial_excludes_misses(self):
        src = load_shader("amrgi_reconstruct.comp")
        # Miss samples (confidence <= 0) must not dilute the spatial denominator.
        assert "s.a <= 1e-4" in src


class TestAMRGITemporalStability:
    """AMRGI-5: explicit sample state + history-length running mean."""

    def test_trace_encodes_sample_state(self):
        src = load_shader("amrgi_trace.comp")
        # A miss must be distinguishable from "no information this frame".
        assert "stateAlpha" in src
        assert "0.5 + 0.5 * newStreak" in src

    def test_reconstruct_reads_aux_state(self):
        src = load_shader("amrgi_reconstruct.comp")
        assert "u_AuxState" in src
        assert "currentHit" in src
        assert "missStreak" in src

    def test_miss_is_not_treated_as_black(self):
        src = load_shader("amrgi_reconstruct.comp")
        # On an evaluated miss with valid history the estimate is preserved, not
        # blended toward zero.
        assert "current = historyRGB;" in src
        assert "newLen = prevLen;" in src

    def test_history_length_running_mean(self):
        src = load_shader("amrgi_reconstruct.comp")
        # alpha = 1/length, not a fixed EMA.
        assert "1.0 / max(newLen, 1.0)" in src
        assert "u_MaxHistory" in src
        assert "u_OutputLength" in src

    def test_history_length_reset_on_reject(self):
        src = load_shader("amrgi_reconstruct.comp")
        # No usable history -> restart accumulation at length 1.
        assert "newLen = 1.0;" in src

    def test_geometry_rejection_present(self):
        src = load_shader("amrgi_reconstruct.comp")
        # Depth + motion + surface-ID must all gate history acceptance.
        assert "depthConf" in src
        assert "motionConf" in src
        assert "geomConf > 0.5" in src

    def test_depth_compared_against_nearest_gbuffer(self):
        src = load_shader("amrgi_reconstruct.comp")
        # AMRGI-5.2: history.a is the bilinearly-filtered resolved depth, which is
        # meaningless across a depth edge and caused spurious intermittent
        # rejection. The comparison must use the NEAREST G-buffer depth at the
        # reprojected location instead.
        assert "prevRawDepth = texture(u_GBufferDepth, prevUV).r" in src
        assert "prevLinearDepth" in src
        assert "abs(prevLinearDepth - expectedPrevDepth)" in src
        assert "abs(history.a - expectedPrevDepth)" not in src

    def test_length_texture_lifecycle(self, render_ctx: RenderContext):
        p = AMRGIPass(render_ctx.ctx, width=320, height=240)
        try:
            assert p.length_tex.size == (320, 240)
            assert p.prev_length_tex.size == (320, 240)
            p.resize(640, 480)
            assert p.length_tex.size == (640, 480)
        finally:
            p.destroy()

    def test_static_convergence_reduces_variance(self, render_ctx: RenderContext):
        """A static scene must converge: late-frame variance < early-frame."""
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.amrgi_max_history = 32.0
            pipeline.apply_config(pipeline.config)
            means = []
            for i in range(60):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
                d = np.frombuffer(pipeline.amrgi_pass.history_tex.read(), dtype=np.float16).reshape(240, 320, 4)
                means.append(float(d[..., :3].mean()))
            early = np.std(means[5:20])
            late = np.std(means[40:60])
            assert late <= early + 1e-6
        finally:
            pipeline.destroy()

    def test_no_nan_inf_after_temporal(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.apply_config(pipeline.config)
            for i in range(20):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
            d = np.frombuffer(pipeline.amrgi_pass.history_tex.read(), dtype=np.float16).reshape(240, 320, 4)
            assert np.all(np.isfinite(d))
        finally:
            pipeline.destroy()

    def test_history_length_capped(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.amrgi_max_history = 8.0
            pipeline.apply_config(pipeline.config)
            for i in range(40):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
            ln = np.frombuffer(pipeline.amrgi_pass.prev_length_tex.read(), dtype=np.float16).reshape(240, 320, 4)
            assert float(ln[..., 0].max()) <= 8.0 + 1e-3
        finally:
            pipeline.destroy()


class TestAMRGIPresets:
    def test_presets_scale_amrgi(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        cine = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert low.amrgi_near_steps < cine.amrgi_near_steps
        assert low.amrgi_far_rate > cine.amrgi_far_rate  # sparser far at low quality
        assert low.amrgi_resolution_scale == 0.25
        assert cine.amrgi_resolution_scale == 1.0
        assert low.amrgi_ray_count < cine.amrgi_ray_count

    def test_amrgi_not_default(self):
        for q in (GraphicsQuality.LOW, GraphicsQuality.MEDIUM, GraphicsQuality.HIGH,
                  GraphicsQuality.ULTRA, GraphicsQuality.CINEMATIC):
            assert get_quality_preset(q).gi_mode != "AMRGI"

    def test_gimode_enum_has_amrgi(self):
        assert GIMode.AMRGI.value == "AMRGI"


class TestAMRGIParameterIndependence:
    """Vary one parameter at a time; verify intended effect and no coupling."""

    def _render(self, render_ctx, **overrides):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            for k, v in overrides.items():
                setattr(pipeline.config, k, v)
            for i in range(4):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
            data = np.frombuffer(pipeline.amrgi_pass.history_tex.read(), dtype=np.float16).reshape(240, 320, 4)
            return data
        finally:
            pipeline.destroy()

    def test_resolution_does_not_change_ray_budget(self, render_ctx: RenderContext):
        # Resolution changes the trace domain size, not the ray/step uniforms.
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.amrgi_ray_count = 1
            pipeline.config.amrgi_resolution_scale = 0.25
            pipeline.apply_config(pipeline.config)
            assert pipeline.amrgi_pass.width == 80
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.amrgi_pass._u_trace["u_RayCount"].value == 1
            pipeline.config.amrgi_resolution_scale = 1.0
            pipeline.apply_config(pipeline.config)
            assert pipeline.amrgi_pass.width == 320
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.032)
            assert pipeline.amrgi_pass._u_trace["u_RayCount"].value == 1
        finally:
            pipeline.destroy()

    def test_ray_count_does_not_brighten(self, render_ctx: RenderContext):
        d1 = self._render(render_ctx, amrgi_ray_count=1)
        d4 = self._render(render_ctx, amrgi_ray_count=4)
        m1 = float(d1[..., :3].mean())
        m4 = float(d4[..., :3].mean())
        # More rays must converge to ~same brightness, not become brighter.
        assert m4 < m1 * 1.5 + 1e-4

    def test_range_ordering_clamped(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "AMRGI"
            # Deliberately invalid ordering: near > mid > far.
            pipeline.config.amrgi_near_end = 20.0
            pipeline.config.amrgi_mid_end = 5.0
            pipeline.config.amrgi_far_end = 1.0
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            u = pipeline.amrgi_pass._u_trace
            assert u["u_NearEnd"].value < u["u_MidEnd"].value < u["u_FarEnd"].value
        finally:
            pipeline.destroy()

    def test_intensity_scales_brightness_only(self, render_ctx: RenderContext):
        d1 = self._render(render_ctx, amrgi_intensity=1.0)
        d2 = self._render(render_ctx, amrgi_intensity=2.0)
        m1 = float(d1[..., :3].mean())
        m2 = float(d2[..., :3].mean())
        assert m2 > m1  # intensity raises brightness

    def test_all_resolutions_finite(self, render_ctx: RenderContext):
        for scale in (0.25, 0.5, 0.75, 1.0):
            d = self._render(render_ctx, amrgi_resolution_scale=scale)
            assert np.all(np.isfinite(d))

class TestAMRGIRayCountRange:
    """AMRGI-5.3: debug ray-count control extended to 1..32."""

    SUPPORTED = (1, 2, 4, 8, 12, 16, 24, 32)

    def _render_with(self, render_ctx, rays):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.amrgi_ray_count = rays
            pipeline.apply_config(pipeline.config)
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            return pipeline.amrgi_pass._u_trace["u_RayCount"].value
        finally:
            pipeline.destroy()

    @pytest.mark.parametrize("rays", (1, 2, 4, 8, 12, 16, 24, 32))
    def test_supported_ray_counts_reach_uniform(self, render_ctx: RenderContext, rays):
        assert self._render_with(render_ctx, rays) == rays

    @pytest.mark.parametrize("rays", (1, 2, 4, 8, 12, 16, 24, 32))
    def test_supported_ray_counts_preserved_in_config(self, render_ctx: RenderContext, rays):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.amrgi_ray_count = rays
            pipeline.apply_config(pipeline.config)
            assert pipeline.config.amrgi_ray_count == rays
        finally:
            pipeline.destroy()

    @pytest.mark.parametrize("bad,expected", [(0, 1), (-5, 1), (33, 32), (100, 32)])
    def test_out_of_range_clamped_safely(self, render_ctx: RenderContext, bad, expected):
        assert self._render_with(render_ctx, bad) == expected

    def test_shader_clamp_is_32(self):
        src = load_shader("amrgi_trace.comp")
        assert "int rayCount = clamp(u_RayCount, 1, 32);" in src
        assert "clamp(u_RayCount, 1, 4)" not in src

    def test_pass_clamp_is_32(self):
        src = open("engine/gfx/passes/amrgi_pass.py", encoding="utf-8").read()
        assert 'max(1, min(32, int(getattr(cfg, "amrgi_ray_count", 1))))' in src

    def test_debug_slider_range_is_32(self):
        src = open("engine/debug/debug_menu.py", encoding="utf-8").read()
        assert 'imgui.slider_int("Rays / Sample"' in src
        assert "1, 32)" in src

    def test_debug_slider_invalidates_history(self):
        src = open("engine/debug/debug_menu.py", encoding="utf-8").read()
        # Changing the sample budget must drop incompatible temporal history.
        assert "invalidate_history()" in src

    def test_debug_shows_active_ray_count(self):
        src = open("engine/debug/debug_menu.py", encoding="utf-8").read()
        assert "AMRGI Rays:" in src

    def test_presets_unchanged_by_debug_range(self):
        # The debug range extension must not alter shipped preset values.
        low = get_quality_preset(GraphicsQuality.LOW)
        med = get_quality_preset(GraphicsQuality.MEDIUM)
        high = get_quality_preset(GraphicsQuality.HIGH)
        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        cine = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert (low.amrgi_ray_count, med.amrgi_ray_count, high.amrgi_ray_count,
                ultra.amrgi_ray_count, cine.amrgi_ray_count) == (1, 1, 2, 2, 4)

    def test_normalization_unchanged(self):
        src = load_shader("amrgi_trace.comp")
        # Estimator must still divide by the launched ray count.
        assert "radiance = radiance / float(rayCount) * u_Intensity;" in src
        assert "confidence = confidence / float(rayCount);" in src
