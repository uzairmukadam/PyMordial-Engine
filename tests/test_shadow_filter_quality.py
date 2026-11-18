"""Shadow filter quality invariants.

These guard the properties that make the three modes actually look different and
actually attach to their casters. Each one corresponds to a defect that was
visible in a rendered frame:

* a split guard that overrode the requested lambda and flattened the cascades,
  wasting near-field texel density;
* a constant depth bias large enough to detach shadows from their casters;
* a physical-only penumbra that was sub-texel and therefore indistinguishable
  from a hard shadow;
* a depth slab stretched to kilometres by a large ground plane, collapsing
  depth precision;
* a filter narrower than a screen pixel, which stair-steps no matter how many
  taps it takes;
* a lit notch at the cascade boundary where a receiver fell outside its rect.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.shadow_csm import MAX_SPAN_RATIO, CascadedShadowMap

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
RESOLVE_SHADER = SHADER_DIR / "deferred_resolve.frag"
SHADOWS_INCLUDE = SHADER_DIR / "common" / "shadows.glsl"


@pytest.fixture(scope="module")
def headless_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _resolve_source() -> str:
    return RESOLVE_SHADER.read_text(encoding="utf-8")


class TestSplitGuardDoesNotFlattenCascades:
    """The span guard must never override the requested split distribution."""

    def test_guard_sits_above_natural_pssm_ratio(self) -> None:
        # The natural ratio grows with shadow distance; the guard must clear the
        # widest supported configuration or it silently becomes the split policy.
        assert MAX_SPAN_RATIO >= 7.5, (
            f"MAX_SPAN_RATIO={MAX_SPAN_RATIO} is below the natural PSSM ratio at long "
            "shadow distances, so the bisection will override the requested lambda."
        )

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

    def test_lambda_still_changes_the_distribution(self, headless_ctx: RenderContext) -> None:
        """A higher lambda must produce a tighter near cascade, not the same one."""
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


class TestHardwareComparisonPath:
    """The filters must use a comparison sampler, not manual per-texel compares."""

    def test_resolve_binds_a_comparison_sampler(self) -> None:
        source = _resolve_source()
        assert "sampler2DShadow u_ShadowAtlasCmp" in source, (
            "The resolve shader must bind the atlas through a sampler2DShadow so one tap "
            "resolves a bilinear 2x2 PCF footprint."
        )

    def test_raw_sampler_is_kept_for_blocker_search(self) -> None:
        source = _resolve_source()
        assert "sampler2D u_ShadowAtlas" in source, (
            "The blocker search needs raw depth, which a comparison sampler cannot return."
        )

    def test_csm_exposes_a_compare_sampler(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=1024, cascade_count=4)
        try:
            assert csm.compare_sampler is not None
            assert csm.compare_sampler.compare_func == "<="
        finally:
            csm.destroy()

    def test_spot_shadow_map_exposes_a_compare_sampler(self, headless_ctx: RenderContext) -> None:
        from engine.gfx.spot_shadows import SpotLightShadowMap

        spot = SpotLightShadowMap(headless_ctx.ctx, atlas_size=512)
        try:
            assert spot.compare_sampler is not None
            assert spot.compare_sampler.compare_func == "<="
        finally:
            spot.destroy()


class TestReceiverPlaneBias:
    """Bias must be evaluated per tap, not applied as one constant."""

    def test_plane_depth_helper_exists(self) -> None:
        source = SHADOWS_INCLUDE.read_text(encoding="utf-8")
        assert "float ShadowPlaneDepth(" in source

    def test_taps_use_the_plane_depth(self) -> None:
        source = _resolve_source()
        match = re.search(r"float ShadowCmpTap\(ShadowCascade c, vec2 uv\) \{(.*?)\n\}", source, re.DOTALL)
        assert match is not None, "Could not locate ShadowCmpTap."
        body = match.group(1)
        assert "ShadowPlaneDepth" in body, (
            "Each tap must evaluate the receiver plane at its own UV. A single constant "
            "bias has to cover the worst tap in the footprint, which detaches shadows."
        )


class TestPenumbraScale:
    """PCSS must be able to produce a visible penumbra, not a hard edge."""

    def test_config_exposes_penumbra_scale(self) -> None:
        cfg = RenderConfig()
        assert cfg.shadow_penumbra_scale >= 1.0

    def test_shader_declares_penumbra_scale(self) -> None:
        assert "u_ShadowPenumbraScale" in _resolve_source()

    def test_pcss_uses_the_penumbra_scale(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "u_ShadowPenumbraScale" in body, (
            "PCSS must widen the physical light source, otherwise the penumbra is "
            "sub-texel and the mode is indistinguishable from HARD."
        )


class TestModesAreDistinct:
    """Each mode must dispatch to its own filter, not share one implementation."""

    def test_dispatch_covers_all_modes(self) -> None:
        source = _resolve_source()
        start = source.index("float SampleCascadeShadow(")
        end = source.index("float CalculateCSMShadow(", start)
        body = source[start:end]
        assert "SampleHardShadow" in body
        assert "SamplePCFShadow" in body
        assert "SamplePCSSShadow" in body

    def test_ultra_differs_from_pcss(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "ultra ?" in body, "PCSS_ULTRA must widen the search or raise the tap budget."

    def test_tap_phase_is_world_anchored(self) -> None:
        # The tap phase must be derived from shadow-space position, never from a
        # frame counter or screen position. A time-varying phase reads as a
        # "refresh scan" sweeping the penumbra; a screen-space phase crawls
        # whenever the camera moves.
        source = _resolve_source()
        assert "ShadowTapPhase" in source
        assert "u_ShadowFrameIndex" not in source
        assert "ShadowTemporalRotation" not in source


class TestDepthSlabStaysProportional:
    """A large ground plane must not stretch the depth slab to kilometres."""

    def test_caster_reach_is_bounded(self) -> None:
        from engine.gfx.shadow_csm import MAX_CASTER_REACH

        assert MAX_CASTER_REACH > 0.0
        assert MAX_CASTER_REACH <= 8.0, (
            "A reach beyond a few slice radii re-admits the kilometre-scale slab "
            "that destroys depth precision."
        )

    def test_ground_plane_does_not_inflate_the_slab(self, headless_ctx: RenderContext) -> None:
        # A 10km ground quad overlaps every cascade in XY. Before the clamp it
        # dragged the slab to ~17km for a 60m shadow distance.
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

    def test_slab_origin_is_quantised(self) -> None:
        # The slab origin must be snapped, otherwise it re-scales the depth
        # encoding every frame and the shadow edge creeps as the camera moves.
        source = (Path(__file__).resolve().parent.parent / "engine" / "gfx" / "shadow_csm.py").read_text(
            encoding="utf-8"
        )
        assert "depth_quantum" in source
        assert "math.floor(sphere_center_z / depth_quantum)" in source


class TestFilterIsScreenSpaceAware:
    """The filter must never be narrower than a screen pixel."""

    def test_cascade_carries_a_screen_footprint(self) -> None:
        source = _resolve_source()
        assert "screen_texels" in source
        assert "c.screen_texels = max(length(uv_dx)" in source

    def test_pcf_radius_respects_the_screen_footprint(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCFShadow(")
        end = source.index("float SamplePCSSShadow(", start)
        body = source[start:end]
        assert "c.screen_texels" in body, (
            "PCF must floor its radius at the screen footprint, otherwise the "
            "penumbra is thinner than a pixel and the edge stair-steps."
        )

    def test_pcss_radius_respects_the_screen_footprint(self) -> None:
        source = _resolve_source()
        start = source.index("float SamplePCSSShadow(")
        end = source.index("float SampleCascadeShadow(", start)
        body = source[start:end]
        assert "c.screen_texels" in body

    def test_hard_mode_resolves_across_a_pixel(self) -> None:
        source = _resolve_source()
        start = source.index("float SampleHardShadow(")
        end = source.index("float SamplePCFShadow(", start)
        body = source[start:end]
        assert "screen_texels" in body, (
            "HARD must resolve across a screen pixel; a lone texel tap "
            "stair-steps at 1080p where a texel spans several pixels."
        )


class TestCascadeBoundaryHasNoHole:
    """A receiver outside its cascade rect must fall through, not go lit."""

    def test_out_of_bounds_walks_forward(self) -> None:
        source = _resolve_source()
        start = source.index("float CalculateCSMShadow(")
        body = source[start:]
        assert "for (int next = cascade + 1; next < u_CascadeCount; ++next)" in body, (
            "A receiver outside its selected cascade's rect must walk forward to "
            "the first cascade that covers it, otherwise a lit notch appears at "
            "the cascade boundary."
        )
