"""Unified global light surface for the atmosphere module.

The sky, shadows, fog, particles, and clouds must all agree on a single
directional light. This module derives the sun and moon independently from the
same East-North-Up solar geometry and exposes one ``primary`` body (whichever is
above the horizon) so every consumer reads a consistent source.

Pure CPU math with no GPU dependency, so it is unit testable headlessly.
"""

from __future__ import annotations

import math

from engine.modules.atmosphere.config import AtmosphereConfig

# Elevation (radians) below which a body stops contributing direct light.
_HORIZON_EPS = 0.0

# Sun colour ramp: below this elevation the light is fully sunset amber, above
# the upper bound it is the configured noon colour.
_SUNSET_RAMP_LOW = -0.05
_SUNSET_RAMP_HIGH = 0.30

_SUNSET_COLOR = (1.0, 0.60, 0.30)


def _sky_vector(
    hour_angle: float,
    lat_rad: float,
    decl_rad: float,
    az_rad: float,
) -> tuple[float, float, float]:
    """Unit vector pointing from the observer toward a body in ENU geometry.

    Deriving the vector directly (rather than from azimuth alone) keeps the arc
    continuous through culmination, where an azimuth-only formulation degenerates.
    """
    cos_decl = math.cos(decl_rad)
    sin_decl = math.sin(decl_rad)
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_hour = math.cos(hour_angle)

    x_east = -cos_decl * math.sin(hour_angle)
    y_north = sin_decl * cos_lat - cos_decl * sin_lat * cos_hour
    z_up = sin_decl * sin_lat + cos_decl * cos_lat * cos_hour

    elevation = math.asin(max(-1.0, min(1.0, z_up)))
    # Compass azimuth: 0 = north (+Z), 90 = east (+X)
    total_az = math.atan2(x_east, y_north) + az_rad

    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.sin(total_az),
        math.sin(elevation),
        cos_elevation * math.cos(total_az),
    )


def hour_angle_for(time_of_day: float) -> float:
    """Solar hour angle in radians: 12:00 = 0, 06:00 = -pi/2, 18:00 = +pi/2."""
    return ((time_of_day - 12.0) / 12.0) * math.pi


class GlobalLightSystem:
    """Derives sun/moon directions, colours, and the active primary light.

    Results are cached against the orbital configuration so the per-frame cost is
    a single tuple comparison when nothing changed.
    """

    __slots__ = (
        "config",
        "sun_dir",
        "sun_color",
        "sun_lux",
        "moon_dir",
        "moon_color",
        "moon_lux",
        "moon_phase",
        "ambient",
        "primary_dir",
        "primary_color",
        "primary_lux",
        "_key",
    )

    def __init__(self, config: AtmosphereConfig | None = None) -> None:
        self.config = config if config is not None else AtmosphereConfig()
        self.sun_dir: tuple[float, float, float] = (0.0, -1.0, 0.0)
        self.sun_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.sun_lux: float = 0.0
        self.moon_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)
        self.moon_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.moon_lux: float = 0.0
        self.moon_phase: float = 1.0
        self.ambient: float = 0.0
        self.primary_dir: tuple[float, float, float] = (0.0, -1.0, 0.0)
        self.primary_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.primary_lux: float = 0.0
        self._key: tuple | None = None
        self.refresh()

    # ---------------- Derivation ----------------

    def refresh(self) -> None:
        """Recomputes every derived light value from the current configuration."""
        cfg = self.config
        key = (
            cfg.time_of_day,
            cfg.latitude_deg,
            cfg.declination_deg,
            cfg.azimuth_deg,
            cfg.moon_orbit_offset_hours,
            cfg.moon_declination_deg,
            cfg.moon_azimuth_offset_deg,
            cfg.sun_intensity,
            cfg.sun_color,
            cfg.moon_intensity,
            cfg.moon_color,
        )
        if key == self._key:
            return
        self._key = key

        lat_rad = math.radians(cfg.latitude_deg)
        decl_rad = math.radians(cfg.declination_deg)
        az_rad = math.radians(cfg.azimuth_deg)

        sun_sky = _sky_vector(hour_angle_for(cfg.time_of_day), lat_rad, decl_rad, az_rad)
        # The moon orbits on its own plane: independent declination and azimuth
        # heading, so it does not simply retrace the sun's arc.
        moon_sky = _sky_vector(
            hour_angle_for(cfg.time_of_day + cfg.moon_orbit_offset_hours),
            lat_rad,
            math.radians(cfg.moon_declination_deg),
            az_rad + math.radians(cfg.moon_azimuth_offset_deg),
        )

        # Light directions point from the body into the scene.
        self.sun_dir = (-sun_sky[0], -sun_sky[1], -sun_sky[2])
        self.moon_dir = (-moon_sky[0], -moon_sky[1], -moon_sky[2])

        sun_elevation = sun_sky[1]
        moon_elevation = moon_sky[1]

        # Phase from the sun-moon angular separation: 0 at conjunction (new moon),
        # 1 at opposition (full moon).
        separation = math.acos(
            max(-1.0, min(1.0, sun_sky[0] * moon_sky[0] + sun_sky[1] * moon_sky[1] + sun_sky[2] * moon_sky[2]))
        )
        self.moon_phase = 0.5 * (1.0 - math.cos(separation))

        # Sun: warm amber near the horizon ramping to the configured noon colour.
        ramp = max(0.0, min(1.0, (sun_elevation - _SUNSET_RAMP_LOW) / (_SUNSET_RAMP_HIGH - _SUNSET_RAMP_LOW)))
        noon = cfg.sun_color
        self.sun_color = (
            noon[0] * ramp + _SUNSET_COLOR[0] * (1.0 - ramp),
            noon[1] * ramp + _SUNSET_COLOR[1] * (1.0 - ramp),
            noon[2] * ramp + _SUNSET_COLOR[2] * (1.0 - ramp),
        )
        self.sun_lux = cfg.sun_intensity * max(0.05, min(1.0, sun_elevation * 1.5))

        # Moon: brightness scales with illuminated fraction, so a new moon is dark.
        self.moon_color = cfg.moon_color
        self.moon_lux = cfg.moon_intensity * self.moon_phase

        # Primary body: whichever is above the horizon; otherwise the higher one so
        # there is always a well-defined directional light.
        if sun_elevation > _HORIZON_EPS:
            self.primary_dir = self.sun_dir
            self.primary_color = self.sun_color
            self.primary_lux = self.sun_lux
            self.ambient = 0.02 + 0.03 * max(0.0, sun_elevation)
        elif moon_elevation > _HORIZON_EPS:
            self.primary_dir = self.moon_dir
            self.primary_color = self.moon_color
            self.primary_lux = self.moon_lux
            self.ambient = 0.015
        elif sun_elevation >= moon_elevation:
            self.primary_dir = self.sun_dir
            self.primary_color = self.sun_color
            self.primary_lux = 0.0
            self.ambient = 0.012
        else:
            self.primary_dir = self.moon_dir
            self.primary_color = self.moon_color
            self.primary_lux = 0.0
            self.ambient = 0.012

    # ---------------- Convenience ----------------

    def lighting(self) -> tuple[tuple[float, float, float], float, tuple[float, float, float], float]:
        """Returns ``(light_dir, lux, light_color, ambient_factor)`` for the primary body."""
        self.refresh()
        return (self.primary_dir, self.primary_lux, self.primary_color, self.ambient)


__all__ = ["GlobalLightSystem", "hour_angle_for"]
