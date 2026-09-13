"""Unit tests for Physical Atmosphere & Dynamic Day-Night Cycle."""

import pytest
from engine.gfx.atmosphere import (
    AtmosphereConfig,
    AtmosphereSystem,
    ATMOSPHERE_PRESETS,
)
from engine.debug.engine_tweaks import EngineTweaks


def test_atmosphere_config_defaults() -> None:
    config = AtmosphereConfig()
    assert config.time_of_day == 12.0
    assert config.day_speed == 0.0
    assert len(config.rayleigh_beta) == 3
    assert config.rayleigh_beta[0] < config.rayleigh_beta[2]  # Red scatters less than blue
    assert config.turbidity == 1.0
    assert config.sun_intensity == 4.0
    assert config.moon_intensity == 0.40


def test_atmosphere_presets_structure() -> None:
    expected_presets = [
        "EARTH_DAY",
        "EARTH_SUNSET",
        "EARTH_NIGHT",
        "ALIEN_CYAN_PURPLE",
        "ALIEN_CRIMSON_MARS",
    ]
    for p in expected_presets:
        assert p in ATMOSPHERE_PRESETS
        preset_data = ATMOSPHERE_PRESETS[p]
        assert "rayleigh_beta" in preset_data
        assert "mie_beta" in preset_data
        assert "sun_intensity" in preset_data
        assert len(preset_data["rayleigh_beta"]) == 3


def test_atmosphere_system_apply_preset() -> None:
    atmo = AtmosphereSystem()
    atmo.apply_preset("ALIEN_CYAN_PURPLE")
    assert atmo.config.preset_name == "ALIEN_CYAN_PURPLE"
    assert atmo.config.time_of_day == 13.5
    # Alien cyan/purple has high red scattering
    assert atmo.config.rayleigh_beta[0] == 32.0e-6

    atmo.apply_preset("EARTH_NIGHT")
    assert atmo.config.preset_name == "EARTH_NIGHT"
    assert atmo.config.time_of_day == 0.0


def test_solar_kinematics_noon() -> None:
    atmo = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, azimuth_deg=45.0, latitude_deg=35.0))
    sun_dir = atmo.compute_sun_vector()
    # At noon, sun vector points downwards (negative Y)
    assert sun_dir[1] < -0.5

    light_dir, lux, col, ambient = atmo.compute_lighting()
    # At noon, lux is high
    assert lux > 2.0
    assert light_dir[1] < -0.5
    assert ambient > 0.02


def test_solar_kinematics_midnight() -> None:
    atmo = AtmosphereSystem(AtmosphereConfig(time_of_day=0.0, azimuth_deg=45.0, latitude_deg=35.0))
    sun_dir = atmo.compute_sun_vector()
    # At midnight, sun vector points upwards (sun is below horizon on other side of earth)
    assert sun_dir[1] > 0.5

    light_dir, lux, col, ambient = atmo.compute_lighting()
    # At midnight, moon is the directional caster, pointing downwards opposite to sun
    assert light_dir[1] < -0.5
    assert lux == pytest.approx(atmo.config.moon_intensity, rel=1e-3)
    assert col == atmo.config.moon_color


def test_time_advancement() -> None:
    atmo = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, day_speed=0.0))
    atmo.update(1.0)
    # Paused
    assert atmo.config.time_of_day == 12.0

    atmo.config.day_speed = 1.0  # 1 real minute per 24 hours
    atmo.update(30.0)  # 30 seconds = 12 hours
    assert atmo.config.time_of_day == pytest.approx(0.0, abs=0.1)


def test_engine_tweaks_preset_integration() -> None:
    tweaks = EngineTweaks()
    tweaks.apply_atmo_preset("ALIEN_CRIMSON_MARS")
    assert tweaks.atmo_preset == "ALIEN_CRIMSON_MARS"
    assert tweaks.time_of_day == 14.0
    assert tweaks.turbidity == 4.5
    assert tweaks.rayleigh_r == pytest.approx(4.0, rel=1e-3)
    assert tweaks.rayleigh_b == pytest.approx(26.0, rel=1e-3)
