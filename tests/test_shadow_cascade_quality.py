"""Shadow cascade quality tests: PCSS_ULTRA mode and per-cascade atlas allocation.

The per-cascade atlas lets near cascades claim more texels than far ones. That
only helps if the legacy 2x2 layout is reproduced exactly when the feature is
off, and if the texel-density step between neighbours stays bounded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset
from engine.gfx.shadow_csm import MAX_SPAN_RATIO, CascadedShadowMap

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
RESOLVE_SHADER = SHADER_DIR / "deferred_resolve.frag"

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


def _legacy_rects(atlas_size: int) -> list[tuple[int, int, int, int]]:
    half = atlas_size // 2
    return [
        (0, 0, half, half),
        (half, 0, half, half),
        (0, half, half, half),
        (half, half, half, half),
    ]


class TestPCSSUltraMode:
    """PCSS_ULTRA must be a first-class shadow mode everywhere."""

    def test_render_config_accepts_pcss_ultra(self) -> None:
        cfg = RenderConfig()
        cfg.shadow_mode = "PCSS_ULTRA"
        assert cfg.shadow_mode == "PCSS_ULTRA"

    @pytest.mark.parametrize("quality", ALL_PRESETS)
    def test_all_presets_accept_pcss_ultra(self, quality: GraphicsQuality) -> None:
        preset = get_quality_preset(quality)
        preset.shadow_mode = "PCSS_ULTRA"
        assert preset.shadow_mode == "PCSS_ULTRA"

    def test_engine_tweaks_round_trips_pcss_ultra(self) -> None:
        tweaks = EngineTweaks()
        tweaks.shadow_mode = "PCSS_ULTRA"
        data = tweaks.to_dict()
        assert data["shadow_mode"] == "PCSS_ULTRA"

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.shadow_mode == "PCSS_ULTRA"

    def test_shader_implements_mode_three(self) -> None:
        source = RESOLVE_SHADER.read_text(encoding="utf-8")
        assert "u_ShadowMode == 3" in source, "PCSS_ULTRA must be handled as u_ShadowMode == 3."
        # The ultra path must use a Poisson disk, not the Vogel spiral.
        assert "PoissonDiskSample" in source
        assert "POISSON_DISK" in source

    def test_ultra_keeps_physical_penumbra(self) -> None:
        source = RESOLVE_SHADER.read_text(encoding="utf-8")
        ultra_start = source.index("u_ShadowMode == 3")
        ultra_end = source.index("MODE 2:", ultra_start)
        ultra_body = source[ultra_start:ultra_end]
        assert "u_SunAngularRadius" in ultra_body, (
            "PCSS_ULTRA must derive its penumbra from the physical solar angular radius."
        )

    def test_ultra_has_separable_blocker_and_filter_passes(self) -> None:
        source = RESOLVE_SHADER.read_text(encoding="utf-8")
        ultra_start = source.index("u_ShadowMode == 3")
        ultra_end = source.index("MODE 2:", ultra_start)
        ultra_body = source[ultra_start:ultra_end]
        assert "search_radius_u" in ultra_body, "PCSS_ULTRA needs a coarse blocker search pass."
        assert "filter_radius_u" in ultra_body, "PCSS_ULTRA needs a tight filter pass."
        assert "cascade_budget" in ultra_body, "PCSS_ULTRA needs a per-cascade tap budget."


class TestCascadeAtlasRects:
    """Per-cascade rects must be exact when off and near-biased when on."""

    def test_default_flag_is_off(self) -> None:
        assert RenderConfig().csm_nonuniform_atlas is False

    @pytest.mark.parametrize("atlas_size", (1024, 2048, 4096))
    def test_uniform_rects_match_legacy_2x2(self, headless_ctx: RenderContext, atlas_size: int) -> None:
        csm = CascadedShadowMap(
            headless_ctx.ctx, atlas_size=atlas_size, cascade_count=4, nonuniform_atlas=False
        )
        try:
            assert csm.cascade_rects == _legacy_rects(atlas_size)
            inv = 1.0 / float(atlas_size)
            expected_uv = [
                (x * inv, y * inv, w * inv, h * inv) for (x, y, w, h) in _legacy_rects(atlas_size)
            ]
            for actual, expected in zip(csm.cascade_rect_uv, expected_uv):
                assert actual == pytest.approx(expected)
        finally:
            csm.destroy()

    def test_nonuniform_rects_favour_near_cascades(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(
            headless_ctx.ctx, atlas_size=4096, cascade_count=4, nonuniform_atlas=True
        )
        try:
            widths = [rect[2] for rect in csm.cascade_rects]
            assert widths[0] > widths[2], "Near cascade must get more texels than a far one."
            assert widths[1] > widths[3], "Near cascade must get more texels than a far one."
            assert widths[0] == widths[1]
            assert widths[2] == widths[3]
            assert widths[0] == 2048 and widths[2] == 1024
        finally:
            csm.destroy()

    def test_nonuniform_rects_stay_inside_atlas(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(
            headless_ctx.ctx, atlas_size=4096, cascade_count=4, nonuniform_atlas=True
        )
        try:
            for x, y, w, h in csm.cascade_rects:
                assert x >= 0 and y >= 0
                assert x + w <= csm.atlas_size
                assert y + h <= csm.atlas_size
        finally:
            csm.destroy()

    def test_begin_cascade_uses_per_cascade_rect(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(
            headless_ctx.ctx, atlas_size=4096, cascade_count=4, nonuniform_atlas=True
        )
        try:
            for index, rect in enumerate(csm.cascade_rects):
                csm.begin_cascade(index)
                assert tuple(headless_ctx.ctx.viewport) == rect
        finally:
            csm.destroy()

    def test_toggling_flag_rebuilds_rects(self, headless_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=4096, cascade_count=4)
        try:
            assert csm.cascade_rects == _legacy_rects(4096)
            csm.set_nonuniform_atlas(True)
            assert csm.cascade_rects[0][2] == 2048
            assert csm.cascade_rects[2][2] == 1024
            csm.set_nonuniform_atlas(False)
            assert csm.cascade_rects == _legacy_rects(4096)
        finally:
            csm.destroy()

    def test_shader_uses_rect_uniform(self) -> None:
        source = RESOLVE_SHADER.read_text(encoding="utf-8")
        assert "u_CascadeAtlasRect" in source
        assert re.search(r"uniform\s+vec4\s+u_CascadeAtlasRect\[4\]", source) is not None
        # The hardcoded 2x2 quadrant derivation must be gone from the cascade path.
        assert "float(cascade % 2) * 0.5" not in source


class TestCascadeTexelDensity:
    """Adjacent cascades must not differ wildly in texel density."""

    @pytest.mark.parametrize("nonuniform", (False, True))
    def test_adjacent_texel_density_ratio_bounded(
        self, headless_ctx: RenderContext, nonuniform: bool
    ) -> None:
        csm = CascadedShadowMap(
            headless_ctx.ctx,
            atlas_size=4096,
            cascade_count=4,
            max_distance=500.0,
            nonuniform_atlas=nonuniform,
        )
        try:
            splits = csm.split_distances
            spans: list[float] = []
            previous = 0.1
            for split in splits:
                spans.append(split - previous)
                previous = split

            # Texels per world unit for each cascade: rect size / slice span.
            densities = [
                csm.cascade_rects[index][2] / max(spans[index], 1e-6)
                for index in range(len(spans))
            ]
            ratios = [
                densities[index] / max(densities[index + 1], 1e-9)
                for index in range(len(densities) - 1)
            ]
            worst = max(ratios)
            assert worst <= MAX_SPAN_RATIO, (
                f"Adjacent cascade texel-density ratio reaches {worst:.2f}x "
                f"(limit {MAX_SPAN_RATIO}x). Densities: {[round(d, 2) for d in densities]}"
            )
        finally:
            csm.destroy()

    def test_nonuniform_improves_near_cascade_density(self, headless_ctx: RenderContext) -> None:
        uniform = CascadedShadowMap(headless_ctx.ctx, atlas_size=4096, cascade_count=4)
        nonuniform = CascadedShadowMap(
            headless_ctx.ctx, atlas_size=4096, cascade_count=4, nonuniform_atlas=True
        )
        try:
            # Near cascades keep their texels while far cascades shrink, so the
            # near/far texel ratio improves without growing the atlas.
            assert nonuniform.cascade_rects[0][2] >= uniform.cascade_rects[0][2]
            assert nonuniform.cascade_rects[3][2] < uniform.cascade_rects[3][2]
            uniform_ratio = uniform.cascade_rects[0][2] / uniform.cascade_rects[3][2]
            nonuniform_ratio = nonuniform.cascade_rects[0][2] / nonuniform.cascade_rects[3][2]
            assert nonuniform_ratio > uniform_ratio
        finally:
            uniform.destroy()
            nonuniform.destroy()


class TestCascadeDebugModes:
    """The cascade debug visualizations must be reachable from the F2 surface."""

    def test_debug_modes_exist(self) -> None:
        assert GBufferDebugMode.CASCADE_INDEX == 14
        assert GBufferDebugMode.CASCADE_COVERAGE == 15

    def test_shader_handles_cascade_debug_modes(self) -> None:
        source = RESOLVE_SHADER.read_text(encoding="utf-8")
        assert "u_GBufferDebug == 14" in source
        assert "u_GBufferDebug == 15" in source

    def test_engine_tweaks_round_trips_nonuniform_flag(self) -> None:
        tweaks = EngineTweaks()
        assert tweaks.csm_nonuniform_atlas is False
        tweaks.csm_nonuniform_atlas = True
        data = tweaks.to_dict()
        assert data["csm_nonuniform_atlas"] is True

        restored = EngineTweaks()
        restored.from_dict(data)
        assert restored.csm_nonuniform_atlas is True
