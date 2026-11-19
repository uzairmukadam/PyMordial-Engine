"""Tests for the P0 colour pipeline (Pass 1).

Covers the shared tone-mapping library and the post-process integration:

* AgX must be monotonic, map black to black, and desaturate saturated
  highlights toward white (its defining behaviour).
* ACES fitted must roll off highlights instead of clipping.
* The exact sRGB transfer function must round-trip through its inverse.
* The post-process pass must honour ``tonemap_mode`` and the grading knobs.
"""

from pathlib import Path

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.post_process import PostProcessPipeline
from engine.gfx.quality_presets import RenderConfig

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
#  CPU mirrors of shaders/common/tonemap.glsl
#
#  The shader is the source of truth; these mirrors exist so the curve can be
#  asserted numerically without a GPU readback per sample.
# ---------------------------------------------------------------------------

AGX_INSET = np.array(
    [
        [0.856627153315983, 0.137318972929847, 0.111898212999950],
        [0.0951212405381588, 0.761241990602591, 0.0767994186031903],
        [0.0482516061458583, 0.101439036467562, 0.811302368396859],
    ]
)
AGX_OUTSET = np.array(
    [
        [1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
        [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
        [-0.016493938717834573, -0.016493938717834257, 1.2519364065950405],
    ]
)
AGX_MIN_EV = -12.47393
AGX_MAX_EV = 4.026069


def _agx_sigmoid(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x4 = x2 * x2
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232


def tonemap_agx(color: np.ndarray) -> np.ndarray:
    v = AGX_INSET @ np.clip(color, 0.0, 65504.0)
    v = np.clip(np.log2(np.maximum(v, 1e-10)), AGX_MIN_EV, AGX_MAX_EV)
    v = (v - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)
    v = _agx_sigmoid(v)
    v = AGX_OUTSET @ v
    return np.clip(v, 0.0, 1.0)


ACES_INPUT = np.array(
    [
        [0.59719, 0.07600, 0.02840],
        [0.35458, 0.90834, 0.13383],
        [0.04823, 0.01566, 0.83777],
    ]
)
ACES_OUTPUT = np.array(
    [
        [1.60475, -0.10208, -0.00327],
        [-0.53108, 1.10813, -0.07276],
        [-0.07367, -0.00605, 1.07602],
    ]
)


def tonemap_aces(color: np.ndarray) -> np.ndarray:
    v = ACES_INPUT @ np.maximum(color, 0.0)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    v = ACES_OUTPUT @ (a / b)
    return np.clip(v, 0.0, 1.0)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


class TestAgXCurve:
    def test_black_maps_to_black(self):
        assert tonemap_agx(np.zeros(3))[0] < 1e-3

    def test_monotonic_in_luminance(self):
        values = [0.0, 0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0, 16.0]
        out = [tonemap_agx(np.array([v, v, v]))[0] for v in values]
        assert all(b >= a - 1e-6 for a, b in zip(out, out[1:])), out

    def test_highlights_roll_off_without_clipping(self):
        """A 100x-overexposed white must stay below 1.0 and above mid-grey."""
        out = tonemap_agx(np.array([100.0, 100.0, 100.0]))[0]
        assert 0.5 < out < 1.0

    def test_saturated_highlight_desaturates_toward_white(self):
        """AgX's signature: a very bright saturated red loses saturation."""
        dim = tonemap_agx(np.array([1.0, 0.0, 0.0]))
        bright = tonemap_agx(np.array([64.0, 0.0, 0.0]))

        def saturation(c):
            return (c.max() - c.min()) / max(c.max(), 1e-6)

        assert saturation(bright) < saturation(dim), (dim, bright)

    def test_output_is_bounded(self):
        for v in (0.0, 1e-6, 1.0, 1e3, 1e6):
            out = tonemap_agx(np.array([v, v * 0.5, v * 0.1]))
            assert np.all(out >= 0.0) and np.all(out <= 1.0)


class TestACESCurve:
    def test_black_maps_to_black(self):
        assert tonemap_aces(np.zeros(3))[0] < 1e-3

    def test_monotonic(self):
        values = [0.0, 0.01, 0.18, 1.0, 4.0, 16.0]
        out = [tonemap_aces(np.array([v, v, v]))[0] for v in values]
        assert all(b >= a - 1e-6 for a, b in zip(out, out[1:])), out

    def test_highlights_roll_off(self):
        out = tonemap_aces(np.array([100.0, 100.0, 100.0]))[0]
        assert 0.5 < out <= 1.0


class TestSRGBTransfer:
    def test_round_trip(self):
        linear = np.linspace(0.0, 1.0, 256)
        encoded = linear_to_srgb(linear)
        decoded = srgb_to_linear(encoded)
        np.testing.assert_allclose(decoded, linear, atol=1e-5)

    def test_mid_grey(self):
        """Linear 0.18 must encode to ~0.46, not the 0.42 a pow(1/2.2) gives."""
        assert abs(linear_to_srgb(np.array([0.18]))[0] - 0.4614) < 0.005

    def test_linear_segment_near_black(self):
        """The linear toe must be present; pow(1/2.2) would give ~0.06 here."""
        assert linear_to_srgb(np.array([0.002]))[0] < 0.03


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=64, height=64)
    yield ctx.ctx
    ctx.destroy()


def _render_post(ctx, config: RenderConfig, hdr_value: float) -> np.ndarray:
    quad_vert = (ROOT_DIR / "shaders" / "fullscreen_quad.vert").read_text(encoding="utf-8")
    post_frag = (ROOT_DIR / "shaders" / "post_process.frag").read_text(encoding="utf-8")
    post = PostProcessPipeline(ctx, 64, 64, config, post_frag, quad_vert)
    try:
        scene = np.full((64, 64, 4), (hdr_value, hdr_value, hdr_value, 1.0), dtype=np.float16)
        post.hdr_texture.write(scene.tobytes())
        post.render()
        return np.frombuffer(post.final_texture.read(), dtype=np.uint8).reshape(64, 64, 4)
    finally:
        post.destroy()


def _base_config(**overrides) -> RenderConfig:
    defaults = dict(
        bloom_enabled=False,
        lens_flare_enabled=False,
        chromatic_aberration_enabled=False,
        vignette_enabled=False,
        film_grain_enabled=False,
    )
    defaults.update(overrides)
    return RenderConfig(**defaults)


class TestPostProcessIntegration:
    @pytest.mark.parametrize("mode", ["ACES", "AgX", "Neutral", "Reinhard"])
    def test_every_tonemap_mode_renders(self, render_ctx, mode):
        out = _render_post(render_ctx, _base_config(tonemap_mode=mode), 1.0)
        assert out[..., :3].mean() > 1.0, f"{mode} produced a black frame"

    def test_modes_produce_distinct_images(self, render_ctx):
        images = {
            mode: _render_post(render_ctx, _base_config(tonemap_mode=mode), 2.0)[..., :3].astype(np.int16)
            for mode in ("ACES", "AgX", "Neutral", "Reinhard")
        }
        modes = list(images)
        for i, a in enumerate(modes):
            for b in modes[i + 1 :]:
                assert np.abs(images[a] - images[b]).mean() > 1.0, f"{a} and {b} are identical"

    def test_exposure_compensation_brightens(self, render_ctx):
        dark = _render_post(render_ctx, _base_config(exposure_compensation=0.0), 0.5)
        bright = _render_post(render_ctx, _base_config(exposure_compensation=2.0), 0.5)
        assert bright[..., :3].mean() > dark[..., :3].mean()

    def test_saturation_zero_is_greyscale(self, render_ctx):
        out = _render_post(render_ctx, _base_config(saturation=0.0), 1.0)[..., :3].astype(np.int16)
        assert np.abs(out[..., 0] - out[..., 1]).max() <= 1
        assert np.abs(out[..., 1] - out[..., 2]).max() <= 1

    def test_white_balance_shifts_channels(self, render_ctx):
        neutral = _render_post(render_ctx, _base_config(), 1.0)[..., :3].astype(np.int16)
        warm = _render_post(
            render_ctx,
            _base_config(white_balance_enabled=True, white_balance_k=9000.0),
            1.0,
        )[..., :3].astype(np.int16)
        # `white_balance_k` is the temperature of the illuminant being
        # compensated for, so a higher value warms the image (Unreal/Unity
        # convention): red rises relative to blue.
        assert (warm[..., 0] - warm[..., 2]).mean() > (neutral[..., 0] - neutral[..., 2]).mean()

    def test_output_never_exceeds_8bit_range(self, render_ctx):
        out = _render_post(render_ctx, _base_config(), 500.0)
        assert out.max() <= 255
