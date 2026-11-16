"""Atmosphere & Sky Module Simulation.

Owns the solar orbit, day-night lighting derivation, and preset application. This
is pure CPU math with no GPU dependency, so it can be unit tested headlessly.
"""

from __future__ import annotations

import math

from engine.modules.atmosphere.config import ATMOSPHERE_PRESETS, AtmosphereConfig


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


__all__ = ["AtmosphereSystem"]
