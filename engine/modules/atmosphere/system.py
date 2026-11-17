"""Atmosphere & Sky Module Simulation.

Owns the solar orbit, day-night lighting derivation, and preset application. This
is pure CPU math with no GPU dependency, so it can be unit tested headlessly.
"""

from __future__ import annotations

from engine.modules.atmosphere.config import ATMOSPHERE_PRESETS, AtmosphereConfig
from engine.modules.atmosphere.global_light import GlobalLightSystem, hour_angle_for


class AtmosphereSystem:
    """Manages physical sky simulation, solar orbit, and lighting synchronization."""

    __slots__ = ("config", "lights")

    def __init__(self, config: AtmosphereConfig | None = None) -> None:
        self.config = config if config is not None else AtmosphereConfig()
        self.lights = GlobalLightSystem(self.config)

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
        self.lights.refresh()
        return self.lights.sun_dir

    def compute_moon_vector(self) -> tuple[float, float, float]:
        """Calculates normalized moon light direction (points downwards towards ground)."""
        self.lights.refresh()
        return self.lights.moon_dir

    @property
    def moon_phase(self) -> float:
        """Illuminated fraction of the moon: 0.0 at new moon, 1.0 at full moon."""
        self.lights.refresh()
        return self.lights.moon_phase

    def compute_lighting(self) -> tuple[tuple[float, float, float], float, tuple[float, float, float], float]:
        """Computes current primary directional light vector, lux, color, and ambient factor.

        Returns:
            (light_dir, lux, light_color, ambient_factor)
        """
        return self.lights.lighting()


__all__ = ["AtmosphereSystem", "hour_angle_for"]
