"""Tests for the shared volumetric scattering library.

Verifies that ``shaders/common/volumetrics.glsl`` is the single canonical source
of the phase functions, noise and CSM sampling helpers, that every volumetric
consumer includes it instead of keeping private copies, and that the CPU-side
``VolumetricSettings`` surface mirrors the render config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.gfx.quality_presets import RenderConfig
from engine.gfx.shader_utils import get_shader_dir, load_shader
from engine.gfx.volumetrics import VolumetricSettings

SHADER_DIR = get_shader_dir()
INCLUDE_REL = "shaders/common/volumetrics.glsl"

CONSUMERS = (
    "volumetric_fog_inject.comp",
    "particle_render.frag",
    "sky_cloud.frag",
)

REQUIRED_FUNCTIONS = (
    "float HenyeyGreenstein(float cosTheta, float g)",
    "float DualLobeHenyeyGreenstein(float cosTheta, float g)",
    "float CornetteShanksPhase(float cosTheta, float g)",
    "float Hash31(vec3 p)",
    "float ValueNoise3D(vec3 p)",
    "float InterleavedGradientNoise(vec2 screenPos)",
    "float SampleCascadeShadow(vec3 worldPos, float viewDepth)",
    "float FroxelSliceToViewDepth(float sliceFraction, float near, float far)",
    "vec3 BeerLambertIntegrate(vec3 scattering, float extinction, float deltaZ)",
    "struct VolumetricMedium",
)


def _include_path() -> Path:
    return SHADER_DIR / "common" / "volumetrics.glsl"


def _read_consumer(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


def test_include_file_exists_and_has_guard():
    path = _include_path()
    assert path.is_file(), f"Missing shared include at {path}"
    text = path.read_text(encoding="utf-8")
    assert "#ifndef VOLUMETRICS_GLSL" in text
    assert "#define VOLUMETRICS_GLSL" in text
    assert "#endif" in text


@pytest.mark.parametrize("declaration", REQUIRED_FUNCTIONS)
def test_include_declares_required_symbols(declaration: str):
    text = _include_path().read_text(encoding="utf-8")
    assert declaration in text, f"volumetrics.glsl is missing '{declaration}'"


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumer_includes_shared_library(consumer: str):
    text = _read_consumer(consumer)
    assert f'#include "{INCLUDE_REL}"' in text, f"{consumer} does not include {INCLUDE_REL}"


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("symbol", ("HenyeyGreenstein", "ValueNoise3D", "Hash31"))
def test_consumer_does_not_redefine_shared_helpers(consumer: str, symbol: str):
    text = _read_consumer(consumer)
    assert f"float {symbol}(" not in text, f"{consumer} still defines its own {symbol}"


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumer_expands_to_single_definition(consumer: str):
    """The recursive include must yield exactly one definition of each helper."""
    expanded = load_shader(consumer)
    for symbol in ("HenyeyGreenstein", "ValueNoise3D", "Hash31"):
        assert expanded.count(f"float {symbol}(") == 1, (
            f"{consumer} expands to {expanded.count(f'float {symbol}(')} definitions of {symbol}"
        )


def test_volumetric_settings_defaults():
    settings = VolumetricSettings()
    assert settings.density == pytest.approx(0.005)
    assert settings.height_falloff == pytest.approx(0.10)
    assert settings.anisotropy == pytest.approx(0.65)
    assert settings.ambient == pytest.approx(0.35)
    assert settings.distance == pytest.approx(400.0)


def test_volumetric_settings_from_render_config_round_trips():
    config = RenderConfig(
        fog_density=0.012,
        fog_height_falloff=0.25,
        fog_anisotropy=0.42,
        fog_ambient=0.60,
        fog_distance=250.0,
    )
    settings = VolumetricSettings.from_render_config(config)
    assert settings.density == pytest.approx(0.012)
    assert settings.height_falloff == pytest.approx(0.25)
    assert settings.anisotropy == pytest.approx(0.42)
    assert settings.ambient == pytest.approx(0.60)
    assert settings.distance == pytest.approx(250.0)


def test_volumetric_settings_from_partial_config_uses_defaults():
    class Partial:
        fog_density = 0.02

    settings = VolumetricSettings.from_render_config(Partial())
    assert settings.density == pytest.approx(0.02)
    assert settings.height_falloff == pytest.approx(0.10)
    assert settings.distance == pytest.approx(400.0)
