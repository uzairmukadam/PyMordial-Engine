"""Atmosphere & Sky Module Configuration.

Declares the tunable sky, planetary atmosphere, cloud, and day-night parameters
along with the built-in planet presets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class AtmosphereConfig:
    """Configurable sky, planetary atmosphere, clouds, and day-night parameters."""

    sky_mode: str = "ATMOSPHERE"  # ATMOSPHERE or SPACE
    # Time of Day (0.0 to 24.0 hours) and orbital speed multiplier (0.0 = paused)
    time_of_day: float = 12.0
    day_speed: float = 0.0          # Multiplier: e.g. 1.0 = 1 min full cycle, 0.0 = manual/paused
    azimuth_deg: float = 45.0       # Sun orbit azimuth heading in degrees
    latitude_deg: float = 35.0      # Earth/planetary latitude affecting sun tilt
    declination_deg: float = 0.0    # Solar declination (0.0 = equinox, +/-23.44 = solstice)
    # Lunar hour-angle offset from the sun. 12.0 h keeps the historical antipodal
    # moon (full moon rising as the sun sets); any other value gives the moon its
    # own independent rise/set cycle.
    moon_orbit_offset_hours: float = 12.0
    # Lunar orbital plane, independent of the solar one. The moon's declination
    # varies +/-28.5 deg over the nodal cycle, so it does not trace the sun's arc.
    # Default 0.0 keeps the historical antipodal moon; set a non-zero value to
    # give the moon its own arc.
    moon_declination_deg: float = 0.0
    # Extra azimuth heading applied only to the moon, so its rise/set points can
    # be placed independently of the sun's.
    moon_azimuth_offset_deg: float = 0.0

    # Wavelength-dependent scattering coefficients (m^-1)
    # Earth physical constants (Sébastien Hillaire Eurographics 2020)
    rayleigh_beta: tuple[float, float, float] = (5.802e-6, 13.558e-6, 33.100e-6)
    mie_beta: float = 3.996e-6
    mie_asymmetry: float = 0.80     # Mie forward scattering factor (g)
    ozone_beta: tuple[float, float, float] = (0.650e-6, 1.881e-6, 0.085e-6)
    turbidity: float = 1.0          # Atmospheric aerosol turbidity / haziness
    planet_radius: float = 6360000.0
    atmosphere_radius: float = 6460000.0
    h_r: float = 8000.0
    h_m: float = 1200.0
    ozone_center: float = 25000.0
    ozone_width: float = 15000.0

    # Celestial Bodies
    sun_intensity: float = 4.0
    sun_color: tuple[float, float, float] = (1.0, 0.98, 0.92)
    sun_disc_size: float = 0.00465
    moon_intensity: float = 0.40
    moon_color: tuple[float, float, float] = (0.70, 0.82, 1.0)
    moon_disc_size: float = 0.00450

    # Night Sky & Ground Ambient
    ground_color: tuple[float, float, float] = (0.15, 0.15, 0.15)
    night_zenith: tuple[float, float, float] = (0.012, 0.025, 0.050)
    night_horizon: tuple[float, float, float] = (0.035, 0.055, 0.090)
    star_intensity: float = 1.0
    star_density: float = 1.0
    space_zenith: tuple[float, float, float] = (0.0015, 0.0020, 0.0060)
    nebula_color: tuple[float, float, float] = (0.16, 0.055, 0.28)
    nebula_intensity: float = 0.35

    # Sky grading applied in the resolve shader (before the night blend)
    sky_exposure: float = 1.0       # Linear radiance multiplier on the sky-view LUT
    sky_saturation: float = 1.0     # Luminance-preserving saturation (1.0 = untouched)

    # Procedural cloud deck, positioned in the world's +Y half-space.
    cloud_enabled: bool = True
    cloud_coverage: float = 0.46
    cloud_opacity: float = 0.88
    cloud_altitude: float = 1800.0
    cloud_scale: float = 18000.0
    cloud_layer_separation: float = 6500.0
    cloud_secondary_opacity: float = 0.22
    cloud_wind: tuple[float, float] = (8.0, 2.0)

    preset_name: str = "EARTH_DAY"


_ATMOSPHERE_PRESET_BASE: dict[str, Any] = {
    "sky_mode": "ATMOSPHERE",
    "day_speed": 0.0,
    "azimuth_deg": 45.0,
    "latitude_deg": 35.0,
    "declination_deg": 0.0,
    "moon_orbit_offset_hours": 12.0,
    "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
    "mie_beta": 3.996e-6,
    "mie_asymmetry": 0.80,
    "ozone_beta": (0.650e-6, 1.881e-6, 0.085e-6),
    "turbidity": 1.0,
    "planet_radius": 6360000.0,
    "atmosphere_radius": 6460000.0,
    "h_r": 8000.0,
    "h_m": 1200.0,
    "ozone_center": 25000.0,
    "ozone_width": 15000.0,
    "sun_intensity": 4.0,
    "sun_color": (1.0, 0.98, 0.92),
    "sun_disc_size": 0.00465,
    "moon_intensity": 0.40,
    "moon_color": (0.70, 0.82, 1.0),
    "moon_disc_size": 0.00450,
    "ground_color": (0.15, 0.15, 0.15),
    "night_zenith": (0.012, 0.025, 0.050),
    "night_horizon": (0.035, 0.055, 0.090),
    "star_intensity": 1.0,
    "star_density": 1.0,
    "space_zenith": (0.0015, 0.0020, 0.0060),
    "nebula_color": (0.16, 0.055, 0.28),
    "nebula_intensity": 0.35,
    "sky_exposure": 1.0,
    "sky_saturation": 1.0,
    "cloud_enabled": True,
    "cloud_coverage": 0.46,
    "cloud_opacity": 0.88,
    "cloud_altitude": 1800.0,
    "cloud_scale": 18000.0,
    "cloud_layer_separation": 6500.0,
    "cloud_secondary_opacity": 0.22,
    "cloud_wind": (8.0, 2.0),
}


ATMOSPHERE_PRESETS: dict[str, dict[str, Any]] = {
    "EARTH_DAY": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "ATMOSPHERE",
        "time_of_day": 12.0,
        "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
        # Retuned for realism: the physical Hillaire aerosol coefficient (3.996e-6)
        # plus turbidity 1.0 leaves a milky, desaturated zenith. Dropping Mie to
        # 2.2e-6 keeps the aerosol haze thin so Rayleigh scattering dominates and
        # the zenith reads as a deep natural blue while the horizon stays pale.
        "mie_beta": 2.2e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 1.0,
        "sun_intensity": 4.0,
        "sun_color": (1.0, 0.98, 0.92),
        "ground_color": (0.15, 0.15, 0.15),
        "night_zenith": (0.012, 0.025, 0.050),
        "night_horizon": (0.035, 0.055, 0.090),
        "star_intensity": 1.0,
        "cloud_enabled": True,
        "cloud_coverage": 0.46,
    },
    "EARTH_SUNSET": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "ATMOSPHERE",
        "time_of_day": 18.0,
        "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
        # Retuned for realism: a sunset needs a dense, forward-scattering aerosol
        # layer to carry the long-path orange/red glow along the horizon. Mie is
        # raised to 9.0e-6 (2.25x the physical value) so the horizon warms strongly
        # while the zenith above it stays deep. Turbidity is left at 1.0 because it
        # multiplies extinction and would otherwise darken the whole sky instead of
        # only thickening the haze.
        "mie_beta": 9.0e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 1.0,
        "sun_intensity": 3.5,
        "sun_color": (1.0, 0.70, 0.40),
        "ground_color": (0.15, 0.12, 0.10),
        "night_zenith": (0.015, 0.02, 0.04),
        "night_horizon": (0.06, 0.04, 0.03),
        "star_intensity": 1.0,
        "cloud_enabled": True,
        "cloud_coverage": 0.55,
    },
    "EARTH_NIGHT": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "ATMOSPHERE",
        "time_of_day": 0.0,
        "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
        "mie_beta": 3.996e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 1.0,
        "sun_intensity": 0.0,
        "sun_color": (0.70, 0.82, 1.0),
        "ground_color": (0.05, 0.06, 0.08),
        "night_zenith": (0.008, 0.018, 0.040),
        "night_horizon": (0.025, 0.040, 0.070),
        "star_intensity": 1.5,
        "cloud_enabled": True,
        "cloud_coverage": 0.30,
        "cloud_opacity": 0.55,
    },
    "ALIEN_CYAN_PURPLE": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "ATMOSPHERE",
        # High red/blue scattering creates rich magenta/purple zenith with electric cyan horizon
        "time_of_day": 13.5,
        "rayleigh_beta": (32.0e-6, 8.0e-6, 22.0e-6),
        "mie_beta": 28.0e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 2.5,
        "sun_intensity": 4.5,
        "sun_color": (0.90, 1.0, 0.95),
        "ground_color": (0.12, 0.18, 0.22),
        "night_zenith": (0.04, 0.015, 0.06),
        "night_horizon": (0.02, 0.06, 0.08),
        "star_intensity": 1.2,
        "cloud_enabled": True,
        "cloud_coverage": 0.50,
        "space_zenith": (0.002, 0.001, 0.006),
        "nebula_color": (0.10, 0.03, 0.24),
    },
    "ALIEN_CRIMSON_MARS": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "ATMOSPHERE",
        # Heavy dusty atmosphere: weak blue scattering, high red transmission -> Martian amber sky
        "time_of_day": 14.0,
        "rayleigh_beta": (4.0e-6, 14.0e-6, 26.0e-6),
        "mie_beta": 60.0e-6,
        "mie_asymmetry": 0.85,
        "turbidity": 4.5,
        "sun_intensity": 3.2,
        "sun_color": (1.0, 0.80, 0.60),
        "ground_color": (0.28, 0.14, 0.08),
        "night_zenith": (0.03, 0.01, 0.015),
        "night_horizon": (0.06, 0.02, 0.02),
        "star_intensity": 0.8,
        "cloud_enabled": False,
    },
    "SPACE": {
        **_ATMOSPHERE_PRESET_BASE,
        "sky_mode": "SPACE",
        "time_of_day": 12.0,
        "sun_intensity": 4.0,
        "star_intensity": 1.4,
        "star_density": 1.6,
        "cloud_enabled": False,
        "space_zenith": (0.0015, 0.0020, 0.0060),
        "nebula_color": (0.16, 0.055, 0.28),
        "nebula_intensity": 0.55,
    },
}
