"""Physical Atmosphere & Dynamic Day-Night Cycle for PyMordial Engine.

Provides analytical Rayleigh & Mie atmospheric scattering models, 24-hour solar
orbital mechanics, procedural starry night skies, and planet presets (Earth, Alien).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class AtmosphereConfig:
    """Configurable parameters for physical atmospheric scattering and day-night cycle."""

    # Time of Day (0.0 to 24.0 hours) and orbital speed multiplier (0.0 = paused)
    time_of_day: float = 12.0
    day_speed: float = 0.0          # Multiplier: e.g. 1.0 = 1 min full cycle, 0.0 = manual/paused
    azimuth_deg: float = 45.0       # Sun orbit azimuth heading in degrees
    latitude_deg: float = 35.0      # Earth/planetary latitude affecting sun tilt

    # Wavelength-dependent scattering coefficients (m^-1)
    # Earth physical constants (Sébastien Hillaire Eurographics 2020)
    rayleigh_beta: tuple[float, float, float] = (5.802e-6, 13.558e-6, 33.100e-6)
    mie_beta: float = 3.996e-6
    mie_asymmetry: float = 0.80     # Mie forward scattering factor (g)
    ozone_beta: tuple[float, float, float] = (0.650e-6, 1.881e-6, 0.085e-6)
    turbidity: float = 1.0          # Atmospheric aerosol turbidity / haziness

    # Celestial Bodies
    sun_intensity: float = 4.0
    sun_color: tuple[float, float, float] = (1.0, 0.98, 0.92)
    sun_disc_size: float = 0.045
    moon_intensity: float = 0.40
    moon_color: tuple[float, float, float] = (0.70, 0.82, 1.0)
    moon_disc_size: float = 0.040

    # Night Sky & Ground Ambient
    ground_color: tuple[float, float, float] = (0.15, 0.15, 0.15)
    night_zenith: tuple[float, float, float] = (0.012, 0.025, 0.050)
    night_horizon: tuple[float, float, float] = (0.035, 0.055, 0.090)
    star_intensity: float = 1.0
    star_density: float = 1.0

    preset_name: str = "EARTH_DAY"


ATMOSPHERE_PRESETS: dict[str, dict[str, Any]] = {
    "EARTH_DAY": {
        "time_of_day": 12.0,
        "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
        "mie_beta": 3.996e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 1.0,
        "sun_intensity": 4.0,
        "sun_color": (1.0, 0.98, 0.92),
        "ground_color": (0.15, 0.15, 0.15),
        "night_zenith": (0.012, 0.025, 0.050),
        "night_horizon": (0.035, 0.055, 0.090),
        "star_intensity": 1.0,
    },
    "EARTH_SUNSET": {
        "time_of_day": 18.0,
        "rayleigh_beta": (5.802e-6, 13.558e-6, 33.100e-6),
        "mie_beta": 6.0e-6,
        "mie_asymmetry": 0.80,
        "turbidity": 1.2,
        "sun_intensity": 3.5,
        "sun_color": (1.0, 0.70, 0.40),
        "ground_color": (0.15, 0.12, 0.10),
        "night_zenith": (0.015, 0.02, 0.04),
        "night_horizon": (0.06, 0.04, 0.03),
        "star_intensity": 1.0,
    },
    "EARTH_NIGHT": {
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
    },
    "ALIEN_CYAN_PURPLE": {
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
    },
    "ALIEN_CRIMSON_MARS": {
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
    },
}


class AtmosphereSystem:
    """Manages physical sky simulation, solar orbit, and lighting synchronization."""

    __slots__ = ("config",)

    def __init__(self, config: AtmosphereConfig | None = None) -> None:
        self.config = config if config is not None else AtmosphereConfig()

    def update(self, dt: float) -> None:
        """Advances time of day if day_speed > 0."""
        if self.config.day_speed > 0.0:
            # 1.0 day_speed = 1 real-world minute per full 24h cycle
            delta_hours = (dt * self.config.day_speed) * (24.0 / 60.0)
            self.config.time_of_day = (self.config.time_of_day + delta_hours) % 24.0

    def apply_preset(self, preset_name: str) -> None:
        """Applies a named planet atmosphere preset."""
        preset = ATMOSPHERE_PRESETS.get(preset_name.upper(), None)
        if preset is not None:
            self.config.preset_name = preset_name.upper()
            for k, v in preset.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

    def compute_sun_vector(self) -> tuple[float, float, float]:
        """Calculates normalized direction from light to scene (points downwards towards ground)."""
        # Time to solar hour angle: 12:00 = 0 rad (zenith), 06:00 = -pi/2, 18:00 = +pi/2, 00:00 = pi
        hour_angle = ((self.config.time_of_day - 12.0) / 12.0) * math.pi
        lat_rad = math.radians(self.config.latitude_deg)
        az_rad = math.radians(self.config.azimuth_deg)

        # Solar elevation angle
        # sin(elevation) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(hour_angle)
        # Using equinox (declination = 0): sin(el) = cos(lat) * cos(hour_angle)
        sin_elevation = math.cos(lat_rad) * math.cos(hour_angle)
        elevation = math.asin(max(-1.0, min(1.0, sin_elevation)))

        # Solar azimuth
        cos_az = math.sin(hour_angle) / max(math.cos(elevation), 1e-5)
        sin_az = -math.tan(lat_rad) * math.tan(elevation) + math.sin(lat_rad) * math.cos(hour_angle) / max(math.cos(elevation), 1e-5)
        total_az = math.atan2(cos_az, sin_az) + az_rad

        # Sun position in the sky (pointing from origin to sun)
        sun_sky_x = math.cos(elevation) * math.sin(total_az)
        sun_sky_y = math.sin(elevation)
        sun_sky_z = math.cos(elevation) * math.cos(total_az)

        # Light direction points from sun towards scene: (-x, -y, -z)
        return (-sun_sky_x, -sun_sky_y, -sun_sky_z)

    def compute_lighting(self) -> tuple[tuple[float, float, float], float, tuple[float, float, float], float]:
        """Computes current primary directional light vector, lux, color, and ambient factor.

        Returns:
            (light_dir, lux, light_color, ambient_factor)
        """
        sun_dir = self.compute_sun_vector()
        sun_elevation = -sun_dir[1]  # positive when sun is above horizon

        if sun_elevation >= -0.05:
            # Daytime / Twilight: Sun is the primary directional light
            sun_t = max(0.0, min(1.0, (sun_elevation + 0.05) / 0.35))
            # Smooth transition from sunset warm amber to bright noon white
            noon_col = self.config.sun_color
            sunset_col = (1.0, 0.60, 0.30)
            col = (
                noon_col[0] * sun_t + sunset_col[0] * (1.0 - sun_t),
                noon_col[1] * sun_t + sunset_col[1] * (1.0 - sun_t),
                noon_col[2] * sun_t + sunset_col[2] * (1.0 - sun_t),
            )
            lux = self.config.sun_intensity * max(0.05, min(1.0, sun_elevation * 1.5))
            ambient = 0.02 + 0.03 * max(0.0, sun_elevation)
            return (sun_dir, lux, col, ambient)
        else:
            # Night: Moon is the primary directional caster (pointing opposite to sun)
            moon_dir = (-sun_dir[0], -sun_dir[1], -sun_dir[2])
            col = self.config.moon_color
            lux = self.config.moon_intensity
            ambient = 0.015
            return (moon_dir, lux, col, ambient)
