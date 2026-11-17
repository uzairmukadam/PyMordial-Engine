"""Tests for the shared blur library and the reusable ``BlurPass`` helper.

Verifies that ``shaders/common/blur.glsl`` is the single canonical source of the
Gaussian kernels, depth linearisation and octahedral normal decode, that every
blur consumer includes it instead of keeping private copies, and that
``BlurPass`` owns a working ping-pong pair.
"""

from __future__ import annotations

from pathlib import Path

import moderngl
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.passes.blur_pass import BlurPass
from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()
INCLUDE_REL = "shaders/common/blur.glsl"

CONSUMERS = (
    "bilateral_blur.frag",
    "ssr_blur.frag",
    "ssgi_blur.frag",
)

REQUIRED_SYMBOLS = (
    "float GaussianWeight1D(float offset, float sigma)",
    "const float GAUSSIAN_5[5]",
    "vec4 SeparableBlur(sampler2D tex, vec2 uv, vec2 direction, float radius)",
    "vec4 KawaseBlur(sampler2D tex, vec2 uv, vec2 texel, float offset)",
    "vec4 BilateralBlurDepth(sampler2D tex, sampler2D depth, vec2 uv, vec2 direction,",
    "vec4 BilateralBlurDepthNormal(sampler2D tex, sampler2D depth, sampler2D normal,",
    "float LinearizeViewDepth(float raw_depth, vec2 uv, mat4 inv_proj)",
    "vec3 OctahedralDecode(vec2 f)",
    "struct BlurParams",
)


def _include_path() -> Path:
    return SHADER_DIR / "common" / "blur.glsl"


def _read_consumer(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


def test_include_file_exists_and_has_guard():
    path = _include_path()
    assert path.is_file(), f"Missing shared include at {path}"
    text = path.read_text(encoding="utf-8")
    assert "#ifndef BLUR_GLSL" in text
    assert "#define BLUR_GLSL" in text
    assert "#endif" in text


@pytest.mark.parametrize("declaration", REQUIRED_SYMBOLS)
def test_include_declares_required_symbols(declaration: str):
    text = _include_path().read_text(encoding="utf-8")
    assert declaration in text, f"blur.glsl is missing '{declaration}'"


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumer_includes_shared_library(consumer: str):
    text = _read_consumer(consumer)
    assert f'#include "{INCLUDE_REL}"' in text, f"{consumer} does not include {INCLUDE_REL}"


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("symbol", ("OctahedralDecode", "GetLinearViewDepth"))
def test_consumer_does_not_redefine_shared_helpers(consumer: str, symbol: str):
    text = _read_consumer(consumer)
    assert f"vec3 {symbol}(" not in text, f"{consumer} still defines its own {symbol}"
    assert f"float {symbol}(" not in text, f"{consumer} still defines its own {symbol}"


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumer_expands_to_single_definition(consumer: str):
    """The recursive include must yield exactly one definition of each helper."""
    expanded = load_shader(consumer)
    assert expanded.count("vec3 OctahedralDecode(") == 1, (
        f"{consumer} expands to {expanded.count('vec3 OctahedralDecode(')} definitions of OctahedralDecode"
    )
    assert expanded.count("float LinearizeViewDepth(") == 1, (
        f"{consumer} expands to {expanded.count('float LinearizeViewDepth(')} definitions of LinearizeViewDepth"
    )


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=64, height=64)
    yield ctx
    ctx.destroy()


def test_blur_pass_lifecycle_and_output(render_ctx: RenderContext):
    ctx = render_ctx.ctx
    blur = BlurPass(ctx, 32, 24, components=1, dtype="f1")
    try:
        assert blur.width == 32
        assert blur.height == 24
        assert len(blur.textures) == 2
        assert len(blur.fbos) == 2
        assert blur.textures[0].size == (32, 24)

        blur.resize(48, 36)
        assert blur.width == 48
        assert blur.height == 36
        assert blur.textures[0].size == (48, 36)
        assert blur.textures[1].size == (48, 36)

        # Resizing to the same size must be a no-op that keeps the same textures.
        existing = blur.textures[0]
        blur.resize(48, 36)
        assert blur.textures[0] is existing

        source = ctx.texture((48, 36), 1, dtype="f1")
        source.filter = (moderngl.LINEAR, moderngl.LINEAR)
        source.write(bytes([255]) * (48 * 36))
        depth = ctx.texture((48, 36), 1, dtype="f4")
        depth.write(b"\x00\x00\x80\x3f" * (48 * 36))

        result = blur.blur_hv(source, depth)
        assert result is blur.textures[1]
        data = result.read()
        assert len(data) == 48 * 36
        assert any(byte > 0 for byte in data)

        source.release()
        depth.release()
    finally:
        blur.destroy()
