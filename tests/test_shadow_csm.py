"""Hard-shadow cascaded shadow map tests.

The shadow system is intentionally narrow: PSSM splits, a fixed 2x2 atlas,
texel-snapped fitting, and a single hardware depth comparison. These tests pin
the properties that make a hard shadow look correct, and pin the removal of the
filter-mode machinery so it cannot creep back in unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from engine.debug.engine_tweaks import EngineTweaks
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset
from engine.gfx.shadow_csm import MAX_CASTER_REACH, MAX_SPAN_RATIO, CascadedShadowMap

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
RESOLVE_SHADER = SHADER_DIR / "deferred_resolve.frag"
SHADOWS_INCLUDE = SHADER_DIR / "common" / "shadows.glsl"
VOLUMETRICS_INCLUDE = SHADER_DIR / "common" / "volumetrics.glsl"

ALL_PRESETS = (
    GraphicsQuality.LOW,
    GraphicsQuality.MEDIUM,
    GraphicsQuality.HIGH,
    GraphicsQuality.ULTRA,
    GraphicsQuality.CINEMATIC,
)


@pytest.fixture(scope="module")
def headless_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _resolve_source() -> str:
    """Returns the resolve shader with its #include directives expanded.

    The shadow sampling code lives in ``shaders/common/csm_sampling.glsl`` so the
    direct-radiance pass can share it. Inspecting the expanded source keeps these
    tests pinned to what the driver actually compiles.
    """
    from engine.gfx.shader_utils import load_shader

    return load_shader("deferred_resolve.frag")


def _legacy_rects(atlas_size: int) -> list[tuple[int, int, int, int]]:
    half = atlas_size // 2
    return [
        (0, 0, half, half),
        (half, 0, half, half),
        (0, half, half, half),
        (half, half, half, half),
    ]


class TestRemovedShadowMachinery:
    """PCSS and its supporting knobs stay gone; PCF is a separate, simpler tier."""

    def test_config_has_no_pcss_fields(self) -> None:
        cfg = RenderConfig()
        for removed in (
            "shadow_mode",
            "shadow_softness",
            "sun_angular_radius",
            "shadow_penumbra_scale",
            "shadow_offset_x",
            "shadow_offset_y",
            "csm_split_mode",
            "csm_stabilization",
            "csm_nonuniform_atlas",
        ):
            assert not hasattr(cfg, removed), f"RenderConfig.{removed} should have been removed."

    @pytest.mark.parametrize("quality", ALL_PRESETS)
    def test_presets_have_no_pcss_fields(self, quality: GraphicsQuality) -> None:
        preset = get_quality_preset(quality)
        assert not hasattr(preset, "shadow_mode")
        assert not hasattr(preset, "shadow_softness")
        assert not hasattr(preset, "shadow_penumbra_scale")

    def test_shader_has_no_pcss_uniforms(self) -> None:
        source = _resolve_source()
        # u_SunAngularRadius is intentionally retained: csm_sampling.glsl uses it
        # to derive a physically-based penumbra from the sun's solid angle.
        for removed in (
            "u_ShadowMode",
            "u_ShadowSoftness",
            "u_ShadowPenumbraScale",
            "u_ShadowOffset",
            "POISSON_DISK",
        ):
            assert removed not in source, f"{removed} should have been removed from the resolve shader."

    def test_engine_tweaks_has_no_pcss_fields(self) -> None:
        tweaks = EngineTweaks()
        for removed in ("shadow_mode", "shadow_softness", "csm_split_mode"):
            assert not hasattr(tweaks, removed), f"EngineTweaks.{removed} should have been removed."


class TestHardShadowPath:
    """One receiver setup, one comparison tap, one hard filter."""

    def test_shader_has_a_single_hard_filter(self) -> None:
        source = _resolve_source()
        assert "float SampleHardShadow(" in source
        assert "SampleCascadeShadow(cascade, c0)" in source

    def test_hard_filter_uses_the_comparison_sampler(self) -> None:
        source = _resolve_source()
        start = source.index("float ShadowCmpTap(")
        end = source.index("float SampleHardShadow(", start)
        body = source[start:end]
        assert "u_ShadowAtlasCmp" in body, (
            "The hard shadow tap must go through the hardware comparison sampler so one "
            "fetch resolves a bilinear 2x2 footprint."
        )

    def test_hard_filter_resolves_across_a_screen_pixel(self) -> None:
        source = _resolve_source()
        start = source.index("float SampleHardShadow(")
        end = source.index("float CalculateCSMShadow(", start)
        body = source[start:end]
        assert "screen_texels" in body, (
            "HARD must resolve across a screen pixel; a lone texel tap stair-steps at "
            "1080p where a texel spans several pixels."
        )

    def test_taps_use_the_receiver_plane_depth(self) -> None:
        source = _resolve_source()
        match = re.search(r"float ShadowCmpTap\(ShadowCascade c, vec2 uv\) \{(.*?)\n\}", source, re.DOTALL)
        assert match is not None, "Could not locate ShadowCmpTap."
        assert "ShadowPlaneDepth" in match.group(1), (
            "Each tap must evaluate the receiver plane at its own UV. A single constant "
            "bias has to cover the worst tap in the footprint, which detaches shadows."
        )

    def test_csm_exposes_a_compare_sampler(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            assert csm.compare_sampler is not None
            assert csm.compare_sampler.compare_func == "<="
        finally:
            csm.destroy()

    def test_tap_phase_is_world_anchored(self) -> None:
        source = _resolve_source()
        assert "ShadowTapPhase" in source
        assert "u_ShadowFrameIndex" not in source


class TestPCFFiltering:
    """PCF is a near-cascade quality tier layered on the hard path."""

    def test_config_exposes_pcf_parameters(self) -> None:
        cfg = RenderConfig()
        assert cfg.shadow_filter_mode in ("HARD", "PCF")
        assert 1 <= cfg.pcf_cascade_limit <= 4
        assert 4 <= cfg.pcf_samples <= 64
        assert cfg.pcf_radius > 0.0
        assert cfg.pcf_falloff > 0.0

    def test_shader_declares_pcf_uniforms(self) -> None:
        source = _resolve_source()
        for uniform in (
            "u_ShadowFilterMode",
            "u_PCFCascadeLimit",
            "u_PCFSamples",
            "u_PCFRadius",
            "u_PCFFalloff",
        ):
            assert uniform in source, f"{uniform} must be declared in the resolve shader."

    def test_pcf_is_limited_to_near_cascades(self) -> None:
        """The dispatcher must gate PCF on the cascade index."""
        source = _resolve_source()
        start = source.index("float SampleCascadeShadow(")
        end = source.index("float CalculateCSMShadow(", start)
        body = source[start:end]
        assert "u_ShadowFilterMode >= 1" in body, "PCF must be gated on the filter mode."
        assert "cascade < u_PCFCascadeLimit" in body, (
            "PCF must be limited to cascades below u_PCFCascadeLimit so the far "
            "cascade keeps the hard path."
        )

    def test_pcf_radius_is_floored_at_the_screen_footprint(self) -> None:
        """A filter narrower than a screen pixel still stair-steps."""
        source = _resolve_source()
        start = source.index("float SamplePCFShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "screen_texels" in body, (
            "PCF must floor its radius at the screen footprint, otherwise the "
            "penumbra is thinner than a pixel and the edge stair-steps."
        )

    def test_pcf_tap_count_is_clamped(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCFShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "clamp(u_PCFSamples" in body, "The tap count must be clamped to a safe range."

    def test_pcf_falloff_shapes_the_kernel(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCFShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "u_PCFFalloff" in body, "The falloff exponent must shape the radial weight."

    def test_all_cascade_samples_use_the_dispatcher(self) -> None:
        """Every cascade sample must route through the dispatcher, not hard-only."""
        source = _resolve_source()
        start = source.index("float CalculateCSMShadow(")
        body = source[start:]
        assert "SampleCascadeShadow(cascade, c0)" in body
        assert "SampleCascadeShadow(next, cn)" in body
        assert "SampleCascadeShadow(cascade + 1, c2)" in body
        # No direct hard-only calls may remain in the cascade evaluation.
        assert "SampleHardShadow(c0)" not in body
        assert "SampleHardShadow(cn)" not in body
        assert "SampleHardShadow(c2)" not in body

    def test_presets_scale_pcf_quality(self) -> None:
        low = get_quality_preset(GraphicsQuality.LOW)
        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert low.shadow_filter_mode == "HARD"
        assert cinematic.shadow_filter_mode == "PCSS"
        assert low.pcf_samples < cinematic.pcf_samples
        assert low.pcf_radius < cinematic.pcf_radius

    def test_engine_tweaks_round_trips_pcf_parameters(self) -> None:
        tweaks = EngineTweaks()
        tweaks.shadow_filter_mode = "PCF"
        tweaks.pcf_cascade_limit = 2
        tweaks.pcf_samples = 32
        tweaks.pcf_radius = 2.5
        tweaks.pcf_falloff = 1.5
        data = tweaks.to_dict()
        assert data["shadow_filter_mode"] == "PCF"
        assert data["pcf_cascade_limit"] == 2
        assert data["pcf_samples"] == 32
        assert data["pcf_radius"] == 2.5
        assert data["pcf_falloff"] == 1.5

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.shadow_filter_mode == "PCF"
        assert restored.pcf_cascade_limit == 2
        assert restored.pcf_samples == 32


class TestPCSSFiltering:
    """PCSS is the narrowest tier: variable penumbra on the nearest cascade."""

    def test_config_exposes_pcss_parameters(self) -> None:
        cfg = RenderConfig()
        assert cfg.pcss_cascade_limit >= 1
        assert 4 <= cfg.pcss_search_samples <= 64
        assert cfg.pcss_search_radius > 0.0
        assert cfg.pcss_light_size > 0.0
        assert 4 <= cfg.pcss_samples <= 64
        assert cfg.pcss_falloff > 0.0
        assert cfg.pcss_max_radius > 0.0

    def test_shader_declares_pcss_uniforms(self) -> None:
        source = _resolve_source()
        for uniform in (
            "u_PCSS_CascadeLimit",
            "u_PCSSSearchSamples",
            "u_PCSSSearchRadius",
            "u_PCSSLightSize",
            "u_PCSSSamples",
            "u_PCSSFalloff",
            "u_PCSSMaxRadius",
        ):
            assert uniform in source, f"{uniform} must be declared in the resolve shader."

    def test_pcss_is_limited_to_near_cascades(self) -> None:
        source = _resolve_source()
        start = source.index("float SampleCascadeShadow(")
        end = source.index("float CalculateCSMShadow(", start)
        body = source[start:end]
        assert "u_ShadowFilterMode == 2" in body, "PCSS must be gated on filter mode 2."
        assert "cascade < u_PCSS_CascadeLimit" in body, (
            "PCSS must be limited to cascades below u_PCSS_CascadeLimit."
        )

    def test_pcss_has_three_stages(self) -> None:
        """Blocker search -> penumbra estimate -> disk filter."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "blocker_sum" in body, "PCSS needs a blocker search pass."
        assert "penumbra_world" in body, "PCSS needs a penumbra estimate."
        assert "radius_texels" in body, "PCSS needs a filter radius derived from the gap."

    def test_pcss_penumbra_uses_the_blocker_gap(self) -> None:
        """The penumbra must scale with the receiver-to-blocker gap, not a constant."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "gap_world" in body
        assert "u_PCSSLightSize" in body, "The light size must scale the penumbra."

    def test_pcss_penumbra_uses_world_texel_conversion(self) -> None:
        """Converting metres to texels via inv_rad * cascade_texels is 2x too large."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "c.world_texel" in body, (
            "The penumbra must convert metres to texels with world_texel; deriving it "
            "from inv_rad * cascade_texels is 2x too large because a cascade spans "
            "2 * radius in world space."
        )

    def test_pcss_filter_is_bounded_by_the_search(self) -> None:
        """An unanchored penumbra reads as a blurry blob."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "u_PCSSMaxRadius" in body
        assert "clamp(penumbra_texels" in body

    def test_pcss_returns_lit_when_no_blockers(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "blocker_count == 0" in body

    def test_pcss_tracks_the_closest_blocker(self) -> None:
        """Overlapping shadows must not inherit the far caster's softness."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "close_blocker" in body, (
            "PCSS must track a closeness-weighted blocker so a near caster keeps its "
            "sharp edge when its shadow overlaps a far caster's soft shadow."
        )
        assert "close_gap" in body

    def test_closest_blocker_is_closeness_weighted_not_a_max(self) -> None:
        """A raw max over noisy taps is what makes a PCSS penumbra blotchy."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "close_weight_sum" in body, (
            "The closest-blocker estimate must be a closeness-weighted average, not a "
            "max: a max lets one outlier tap dominate the whole pixel."
        )
        assert "closest_blocker = max(" not in body

    def test_pcss_blends_average_and_closest_gap(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "u_PCSSClosestBlockerWeight" in body, (
            "The average/closest blend must be exposed so the classic PCSS look "
            "remains reachable."
        )
        assert "mix(avg_gap, close_gap" in body

    def test_pcss_filter_uses_a_gaussian_falloff(self) -> None:
        """A polynomial cutoff leaves a visible rim where the filter stops."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "exp(-" in body, (
            "The penumbra filter must use a Gaussian falloff; pow(1 - r2, n) has a "
            "hard rim at the disk edge that reads as a visible ring."
        )
        assert "pow(1.0 - r2" not in body

    def test_pcss_tap_count_scales_with_radius(self) -> None:
        """A fixed budget over a wide penumbra under-samples into banding."""
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "radius_texels_clamped * radius_texels_clamped" in body, (
            "The tap count must scale with the squared radius so taps-per-texel "
            "stays roughly constant."
        )

    def test_closest_blocker_weight_defaults_to_closest(self) -> None:
        cfg = RenderConfig()
        assert cfg.pcss_closest_blocker_weight == 1.0, (
            "The default must favour the closest blocker, otherwise overlapping "
            "shadows soften the near caster."
        )

    def test_pcss_presets_are_configured(self) -> None:
        high = get_quality_preset(GraphicsQuality.HIGH)
        assert high.shadow_filter_mode == "PCSS"
        assert high.pcss_cascade_limit == 1, "PCSS is limited to cascade 1 by default."
        assert high.pcss_search_radius > 0.0
        assert high.pcss_light_size > 0.0

    def test_medium_disables_pcss(self) -> None:
        """MEDIUM stays on PCF; PCSS is a HIGH-and-above tier."""
        medium = get_quality_preset(GraphicsQuality.MEDIUM)
        assert medium.shadow_filter_mode == "PCF"
        assert medium.pcss_cascade_limit == 0

    def test_engine_tweaks_round_trips_pcss_parameters(self) -> None:
        tweaks = EngineTweaks()
        tweaks.shadow_filter_mode = "PCSS"
        tweaks.pcss_cascade_limit = 2
        tweaks.pcss_search_samples = 20
        tweaks.pcss_search_radius = 40.0
        tweaks.pcss_light_size = 0.15
        tweaks.pcss_samples = 28
        tweaks.pcss_falloff = 1.5
        tweaks.pcss_max_radius = 80.0
        tweaks.pcss_closest_blocker_weight = 0.75
        data = tweaks.to_dict()
        assert data["shadow_filter_mode"] == "PCSS"
        assert data["pcss_cascade_limit"] == 2
        assert data["pcss_search_samples"] == 20
        assert data["pcss_search_radius"] == 40.0
        assert data["pcss_light_size"] == 0.15
        assert data["pcss_samples"] == 28
        assert data["pcss_falloff"] == 1.5
        assert data["pcss_max_radius"] == 80.0
        assert data["pcss_closest_blocker_weight"] == 0.75

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.shadow_filter_mode == "PCSS"
        assert restored.pcss_cascade_limit == 2
        assert restored.pcss_light_size == 0.15
        assert restored.pcss_closest_blocker_weight == 0.75


class TestExplicitCascadeDistances:
    """Each cascade's far plane can be authored directly."""

    def test_explicit_distances_are_used_verbatim(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            csm.update_splits(500.0, cascade_count=4, cascade_distances=(20.0, 60.0, 160.0, 500.0))
            assert csm.split_distances == pytest.approx((20.0, 60.0, 160.0, 500.0))
            assert csm.cascade_distances == pytest.approx((20.0, 60.0, 160.0, 500.0))
            assert csm.max_distance == pytest.approx(500.0)
        finally:
            csm.destroy()

    def test_explicit_distances_are_forced_increasing(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            # Out-of-order input must be repaired, not accepted.
            csm.update_splits(500.0, cascade_count=4, cascade_distances=(100.0, 50.0, 50.0, 20.0))
            splits = csm.split_distances
            assert all(splits[i] < splits[i + 1] for i in range(3)), splits
        finally:
            csm.destroy()

    def test_explicit_distances_override_pssm(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            csm.update_splits(500.0, cascade_count=4, lambda_val=0.85)
            pssm = csm.split_distances
            csm.update_splits(500.0, cascade_count=4, cascade_distances=(5.0, 15.0, 45.0, 500.0))
            explicit = csm.split_distances
            assert explicit != pssm
            assert explicit[0] == pytest.approx(5.0)
        finally:
            csm.destroy()

    def test_clearing_explicit_distances_restores_pssm(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            csm.update_splits(500.0, cascade_count=4, cascade_distances=(5.0, 15.0, 45.0, 500.0))
            assert csm.cascade_distances is not None
            csm.update_splits(500.0, cascade_count=4, lambda_val=0.85)
            assert csm.cascade_distances is None
            assert csm.split_distances[-1] == pytest.approx(500.0)
        finally:
            csm.destroy()

    def test_cascade_count_selects_the_matching_far_plane(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=2)
        try:
            csm.update_splits(500.0, cascade_count=2, cascade_distances=(20.0, 60.0, 160.0, 500.0))
            # With two cascades the far plane is cascade 1's end, not cascade 3's.
            assert csm.max_distance == pytest.approx(60.0)
        finally:
            csm.destroy()


class TestCascadeDistance:
    """``shadow_distance`` is the single knob that moves the distribution."""

    @pytest.mark.parametrize("distance", (60.0, 120.0, 250.0, 500.0, 800.0, 1200.0))
    def test_last_split_equals_shadow_distance(self, headless_ctx: RenderContext, distance: float) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4, max_distance=distance)
        try:
            assert csm.split_distances[-1] == pytest.approx(distance)
        finally:
            csm.destroy()

    @pytest.mark.parametrize("distance", (120.0, 250.0, 500.0, 800.0, 1200.0))
    def test_near_cascade_stays_near_field(self, headless_ctx: RenderContext, distance: float) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=4096, cascade_count=4, max_distance=distance)
        try:
            # Cascade 0 must cover a small fraction of the total distance. A
            # flattened distribution pushes it towards distance / cascade_count.
            assert csm.split_distances[0] < distance * 0.10, (
                f"At a {distance:.0f}m shadow distance cascade 0 reaches "
                f"{csm.split_distances[0]:.1f}m, which wastes near-field texel density."
            )
        finally:
            csm.destroy()

    def test_splits_are_strictly_increasing(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4, max_distance=500.0)
        try:
            splits = csm.split_distances
            assert all(splits[i] < splits[i + 1] for i in range(len(splits) - 1))
        finally:
            csm.destroy()

    def test_lambda_changes_the_distribution(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=4096, cascade_count=4, max_distance=500.0)
        try:
            csm.update_splits(500.0, cascade_count=4, lambda_val=0.5)
            loose = csm.split_distances[0]
            csm.update_splits(500.0, cascade_count=4, lambda_val=0.95)
            tight = csm.split_distances[0]
            assert tight < loose, (
                f"lambda 0.95 gave cascade 0 = {tight:.1f}m but lambda 0.5 gave {loose:.1f}m; "
                "the split guard is overriding the requested distribution."
            )
        finally:
            csm.destroy()

    def test_span_ratio_guard_sits_above_natural_pssm(self) -> None:
        assert MAX_SPAN_RATIO >= 7.5, (
            f"MAX_SPAN_RATIO={MAX_SPAN_RATIO} is below the natural PSSM ratio at long "
            "shadow distances, so the bisection will override the requested lambda."
        )

    def test_cascade_count_is_clamped(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=9)
        try:
            assert csm.cascade_count == 4
            csm.update_splits(500.0, cascade_count=0)
            assert csm.cascade_count == 1
        finally:
            csm.destroy()


class TestCascadeAtlas:
    """The atlas is a fixed 2x2 quadrant layout."""

    @pytest.mark.parametrize("atlas_size", (1024, 2048, 4096))
    def test_rects_are_the_2x2_quadrants(self, headless_ctx: RenderContext, atlas_size: int) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=atlas_size, cascade_count=4)
        try:
            assert csm.cascade_rects == _legacy_rects(atlas_size)
            inv = 1.0 / float(atlas_size)
            expected_uv = [(x * inv, y * inv, w * inv, h * inv) for (x, y, w, h) in _legacy_rects(atlas_size)]
            for actual, expected in zip(csm.cascade_rect_uv, expected_uv):
                assert actual == pytest.approx(expected)
        finally:
            csm.destroy()

    def test_begin_cascade_uses_per_cascade_rect(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=4096, cascade_count=4)
        try:
            for index, rect in enumerate(csm.cascade_rects):
                csm.begin_cascade(index)
                assert tuple(headless_ctx.ctx.viewport) == rect
        finally:
            csm.destroy()

    def test_resize_atlas_rebuilds_rects(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            csm.resize_atlas(2048)
            assert csm.atlas_size == 2048
            assert csm.cascade_rects == _legacy_rects(2048)
        finally:
            csm.destroy()

    def test_shader_uses_rect_uniform(self) -> None:
        source = _resolve_source()
        assert re.search(r"uniform\s+vec4\s+u_CascadeAtlasRect\[4\]", source) is not None
        # The hardcoded 2x2 quadrant derivation must not appear in the cascade path.
        assert "float(cascade % 2) * 0.5" not in source

    def test_volumetrics_uses_rect_uniform(self) -> None:
        source = VOLUMETRICS_INCLUDE.read_text(encoding="utf-8")
        assert "u_CascadeAtlasRect" in source
        assert "float(cascade % 2) * 0.5" not in source


class TestCascadeFitting:
    """Fitting must be stable under camera motion and bounded in depth."""

    def test_texel_snapping_is_always_applied(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "engine" / "gfx" / "shadow_csm.py").read_text(
            encoding="utf-8"
        )
        assert "math.floor(center_x / texel_world + 0.5)" in source
        assert "math.floor(center_y / texel_world + 0.5)" in source

    def test_slab_origin_is_quantised(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "engine" / "gfx" / "shadow_csm.py").read_text(
            encoding="utf-8"
        )
        assert "depth_quantum" in source
        assert "math.floor(sphere_center_z / depth_quantum)" in source

    def test_caster_reach_is_bounded(self) -> None:
        assert MAX_CASTER_REACH > 0.0
        assert MAX_CASTER_REACH <= 8.0, (
            "A reach beyond a few slice radii re-admits the kilometre-scale slab "
            "that destroys depth precision."
        )

    def test_ground_plane_does_not_inflate_the_slab(self, headless_ctx: RenderContext) -> None:
        # A 10km ground quad overlaps every cascade in XY. Without the clamp it
        # drags the slab to ~17km for a 60m shadow distance.
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, max_distance=60.0)
        try:
            csm.update_splits(max_distance=60.0, cascade_count=4, lambda_val=0.85)
            aabbs = np.array(
                [
                    [0.0, 0.0, 0.0, 10000.0, 500.0, 10000.0],
                    [0.0, 1.0, 0.0, 0.5, 1.0, 0.5],
                ],
                dtype=np.float32,
            )
            matrices = csm.compute_cascade_matrices(
                camera_pos=np.array([0.0, 3.0, 7.0]),
                camera_forward=np.array([0.0, -0.3, -1.0]),
                sun_dir=np.array([0.4, -0.8, 0.3]),
                aabbs=aabbs,
                active_count=2,
            )
            for index in range(4):
                matrix = matrices[index].reshape((4, 4), order="F")
                depth_range = 1.0 / float(np.linalg.norm(matrix[2, :3]))
                radius = 1.0 / float(np.linalg.norm(matrix[0, :3]))
                assert depth_range < radius * 12.0, (
                    f"cascade {index} slab {depth_range:.1f}m is not proportional to "
                    f"its {radius:.1f}m radius; a huge caster inflated it."
                )
        finally:
            csm.destroy()

    def test_depth_range_is_yaw_invariant(self, headless_ctx: RenderContext) -> None:
        """Turning the camera must not re-scale the depth encoding."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, max_distance=200.0)
        try:
            sun = np.array([0.4, -0.8, 0.3])
            ranges = []
            for yaw_deg in (0.0, 30.0, 60.0, 90.0):
                yaw = np.radians(yaw_deg)
                forward = np.array([np.sin(yaw), -0.2, -np.cos(yaw)])
                matrices = csm.compute_cascade_matrices(
                    camera_pos=np.array([0.0, 3.0, 7.0]),
                    camera_forward=forward,
                    sun_dir=sun,
                )
                matrix = matrices[0].reshape((4, 4), order="F")
                ranges.append(1.0 / float(np.linalg.norm(matrix[2, :3])))
            # The bounding-sphere fit makes the slab a constant for a given slice.
            assert max(ranges) - min(ranges) < 1e-3, f"depth range varied with yaw: {ranges}"
        finally:
            csm.destroy()

    def test_unused_cascades_mirror_the_last(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=2, max_distance=100.0)
        try:
            matrices = csm.compute_cascade_matrices(
                camera_pos=np.array([0.0, 3.0, 7.0]),
                camera_forward=np.array([0.0, -0.3, -1.0]),
                sun_dir=np.array([0.4, -0.8, 0.3]),
            )
            assert np.allclose(matrices[2], matrices[1])
            assert np.allclose(matrices[3], matrices[1])
        finally:
            csm.destroy()


class TestCascadeCulling:
    def test_cull_cascades_returns_one_list_per_cascade(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4, max_distance=100.0)
        try:
            aabbs = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, -50.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            batches = [("cube", 2, 0, False, True)]
            result = csm.cull_cascades(aabbs=aabbs, active_count=2, draw_batches=batches, cascade_count=4)
            assert len(result) == 4
        finally:
            csm.destroy()


class TestCascadeDebugModes:
    def test_shader_handles_cascade_debug_modes(self) -> None:
        source = _resolve_source()
        assert "u_GBufferDebug == 14" in source
        assert "u_GBufferDebug == 15" in source
        assert "u_GBufferDebug == 16" in source

    def test_cascade_bounds_mode_is_registered(self) -> None:
        from engine.debug.engine_tweaks import GBufferDebugMode

        assert GBufferDebugMode.CASCADE_BOUNDS == 16

    def test_cascade_bounds_draws_split_boundaries(self) -> None:
        """The bounds view must mark every split distance, not just tint bands."""
        source = _resolve_source()
        start = source.index("u_GBufferDebug == 16")
        end = source.index("// Subtle direct contact crevice", start)
        body = source[start:end]
        assert "u_CascadeSplits[i]" in body, (
            "The cascade bounds view must iterate the split distances so each "
            "cascade's start and end is visible."
        )
        assert "view_depth" in body

    def test_engine_tweaks_round_trips_cascade_distance(self) -> None:
        tweaks = EngineTweaks()
        tweaks.shadow_distance = 750.0
        tweaks.csm_cascades = 3
        data = tweaks.to_dict()
        assert data["shadow_distance"] == 750.0
        assert data["csm_cascades"] == 3

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.shadow_distance == 750.0
        assert restored.csm_cascades == 3

    def test_engine_tweaks_round_trips_cascade_distances(self) -> None:
        tweaks = EngineTweaks()
        tweaks.csm_cascade_distances = (10.0, 40.0, 120.0, 400.0)
        data = tweaks.to_dict()
        assert data["csm_cascade_distances"] == [10.0, 40.0, 120.0, 400.0]

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.csm_cascade_distances == (10.0, 40.0, 120.0, 400.0)
