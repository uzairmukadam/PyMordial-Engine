"""Unit tests for Physical Atmosphere & Dynamic Day-Night Cycle."""

import math

import pytest
from engine.gfx.atmosphere import (
    AtmosphereConfig,
    AtmosphereSystem,
    ATMOSPHERE_PRESETS,
)
from engine.debug.engine_tweaks import EngineTweaks
from engine.modules.atmosphere.global_light import GlobalLightSystem


def test_atmosphere_config_defaults() -> None:
    config = AtmosphereConfig()
    assert config.time_of_day == 12.0
    assert config.day_speed == 0.0
    assert len(config.rayleigh_beta) == 3
    assert config.rayleigh_beta[0] < config.rayleigh_beta[2]  # Red scatters less than blue
    assert config.turbidity == 1.0
    assert config.sun_intensity == 4.0
    assert config.moon_intensity == 0.40
    assert config.sun_disc_size == pytest.approx(0.00465)
    assert config.moon_disc_size == pytest.approx(0.00450)
    assert config.cloud_enabled is True
    assert config.sky_mode == "ATMOSPHERE"


def test_atmosphere_presets_structure() -> None:
    expected_presets = [
        "EARTH_DAY",
        "EARTH_SUNSET",
        "EARTH_NIGHT",
        "ALIEN_CYAN_PURPLE",
        "ALIEN_CRIMSON_MARS",
        "SPACE",
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


def test_moon_has_independent_orbital_plane() -> None:
    """The moon must not simply retrace the sun's arc."""
    from engine.modules.atmosphere.global_light import GlobalLightSystem

    base = AtmosphereConfig(
        time_of_day=12.0, latitude_deg=35.0, declination_deg=0.0, azimuth_deg=45.0
    )
    default = GlobalLightSystem(base)
    default_dot = sum(a * b for a, b in zip(default.sun_dir, default.moon_dir))

    tilted = AtmosphereConfig(
        time_of_day=12.0,
        latitude_deg=35.0,
        declination_deg=0.0,
        azimuth_deg=45.0,
        moon_declination_deg=20.0,
        moon_azimuth_offset_deg=60.0,
    )
    offset = GlobalLightSystem(tilted)
    offset_dot = sum(a * b for a, b in zip(offset.sun_dir, offset.moon_dir))

    # The default antipodal moon is nearly opposite the sun; tilting its plane
    # must visibly separate the two paths.
    assert default_dot < -0.99
    assert offset_dot > default_dot + 0.2, (
        f"Moon declination/azimuth offsets did not change its path "
        f"(dot {default_dot:.3f} -> {offset_dot:.3f})."
    )


def test_moon_declination_changes_elevation() -> None:
    """A lunar declination offset must move the moon off the solar arc."""
    from engine.modules.atmosphere.global_light import GlobalLightSystem

    flat = GlobalLightSystem(
        AtmosphereConfig(time_of_day=0.0, latitude_deg=35.0, moon_declination_deg=0.0)
    )
    tilted = GlobalLightSystem(
        AtmosphereConfig(time_of_day=0.0, latitude_deg=35.0, moon_declination_deg=25.0)
    )
    assert abs(tilted.moon_dir[1] - flat.moon_dir[1]) > 0.05, (
        "Moon elevation did not respond to moon_declination_deg."
    )


def test_engine_tweaks_preset_integration() -> None:
    tweaks = EngineTweaks()
    tweaks.apply_atmo_preset("ALIEN_CRIMSON_MARS")
    assert tweaks.atmo_preset == "ALIEN_CRIMSON_MARS"
    assert tweaks.time_of_day == 14.0
    assert tweaks.turbidity == 4.5
    assert tweaks.rayleigh_r == pytest.approx(4.0, rel=1e-3)
    assert tweaks.rayleigh_b == pytest.approx(26.0, rel=1e-3)

    tweaks.apply_atmo_preset("SPACE")
    assert tweaks.sky_mode == "SPACE"
    assert tweaks.cloud_enabled is False
    assert tweaks.star_density == pytest.approx(1.6)

    restored = EngineTweaks()
    restored.from_dict(tweaks.to_dict())
    assert restored.sky_mode == "SPACE"
    assert restored.space_zenith == pytest.approx(tweaks.space_zenith)
    assert restored.nebula_color == pytest.approx(tweaks.nebula_color)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def test_moon_is_not_antipodal_when_orbit_offset_differs() -> None:
    """The moon must follow its own hour angle, not mirror the sun."""
    system = AtmosphereSystem(
        AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0, moon_orbit_offset_hours=6.0)
    )
    sun_dir = system.compute_sun_vector()
    moon_dir = system.compute_moon_vector()
    # Antipodal would give a dot product of exactly -1.
    assert _dot(sun_dir, moon_dir) > -0.99

    # The default 12h offset keeps the historical antipodal full moon.
    default_system = AtmosphereSystem(
        AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0, moon_orbit_offset_hours=12.0)
    )
    assert _dot(default_system.compute_sun_vector(), default_system.compute_moon_vector()) == pytest.approx(
        -1.0, abs=1e-6
    )


def test_moon_phase_is_new_at_conjunction_and_full_at_opposition() -> None:
    conjunction = AtmosphereSystem(
        AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0, moon_orbit_offset_hours=0.0)
    )
    assert conjunction.moon_phase == pytest.approx(0.0, abs=1e-6)

    opposition = AtmosphereSystem(
        AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0, moon_orbit_offset_hours=12.0)
    )
    assert opposition.moon_phase == pytest.approx(1.0, abs=1e-6)


def test_moon_phase_modulates_moon_intensity() -> None:
    """A new moon must contribute no direct light; a full moon its full intensity."""
    new_moon = AtmosphereSystem(
        AtmosphereConfig(time_of_day=0.0, latitude_deg=35.0, moon_orbit_offset_hours=0.0)
    )
    assert new_moon.lights.moon_lux == pytest.approx(0.0, abs=1e-6)

    full_moon = AtmosphereSystem(
        AtmosphereConfig(time_of_day=0.0, latitude_deg=35.0, moon_orbit_offset_hours=12.0)
    )
    assert full_moon.lights.moon_lux == pytest.approx(full_moon.config.moon_intensity, rel=1e-6)


def test_sky_grading_round_trips_through_engine_tweaks() -> None:
    tweaks = EngineTweaks()
    tweaks.sky_exposure = 1.75
    tweaks.sky_saturation = 0.65
    tweaks.declination_deg = 12.5
    tweaks.moon_orbit_offset_hours = 7.25

    restored = EngineTweaks()
    restored.from_dict(tweaks.to_dict())
    assert restored.sky_exposure == pytest.approx(1.75)
    assert restored.sky_saturation == pytest.approx(0.65)
    assert restored.declination_deg == pytest.approx(12.5)
    assert restored.moon_orbit_offset_hours == pytest.approx(7.25)


def test_global_light_primary_picks_sun_above_horizon() -> None:
    lights = GlobalLightSystem(AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0))
    lights.refresh()
    assert lights.sun_dir[1] < 0.0  # sun above the horizon
    assert lights.primary_dir == lights.sun_dir
    assert lights.primary_lux == pytest.approx(lights.sun_lux)


def test_global_light_primary_picks_moon_when_sun_is_below_horizon() -> None:
    lights = GlobalLightSystem(
        AtmosphereConfig(time_of_day=0.0, latitude_deg=35.0, moon_orbit_offset_hours=12.0)
    )
    lights.refresh()
    assert lights.sun_dir[1] > 0.0  # sun below the horizon
    assert lights.moon_dir[1] < 0.0  # full moon above the horizon
    assert lights.primary_dir == lights.moon_dir
    assert lights.primary_lux == pytest.approx(lights.moon_lux)


def test_global_light_primary_dir_is_unit_length() -> None:
    lights = GlobalLightSystem(AtmosphereConfig(time_of_day=5.0, latitude_deg=35.0))
    for hour in (0.0, 6.0, 12.0, 18.0, 23.0):
        lights.config.time_of_day = hour
        lights.refresh()
        length = math.sqrt(_dot(lights.primary_dir, lights.primary_dir))
        assert length == pytest.approx(1.0, abs=1e-6)
