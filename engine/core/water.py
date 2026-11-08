"""Per-entity rendering properties for ECS water surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WaterProperties:
    """Optical and wave parameters stored on one water entity."""

    wave_amplitude: float = 0.15
    wave_speed: float = 1.0
    wave_steepness: float = 0.8
    refraction_enabled: bool = True
    refraction_strength: float = 0.03
    foam_enabled: bool = True
    foam_threshold: float = 0.40
    foam_scale: float = 6.0
    foam_intensity: float = 1.0
    clarity: float = 4.0
    roughness: float = 0.05
    color_shallow: tuple[float, float, float] = (0.05, 0.45, 0.55)
    color_deep: tuple[float, float, float] = (0.005, 0.04, 0.15)
    extinction: tuple[float, float, float] = (0.35, 0.12, 0.06)
    foam_color: tuple[float, float, float] = (0.92, 0.96, 1.0)
