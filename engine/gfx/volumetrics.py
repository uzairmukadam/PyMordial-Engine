"""Shared volumetric medium tuning for fog, particles and clouds.

The GPU side of volumetric scattering lives in ``shaders/common/volumetrics.glsl``.
This module is the CPU-side counterpart: a single small settings surface so that
fog, particle and cloud shading read the same tuning defaults instead of each
pass inventing its own.

Nothing here changes rendering behaviour on its own; it is a convenience surface
that mirrors the ``RenderConfig`` fog fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["VolumetricSettings"]

# Canonical defaults, mirrored by the ``RenderConfig`` fog fields.
_DEFAULT_DENSITY = 0.005
_DEFAULT_HEIGHT_FALLOFF = 0.10
_DEFAULT_ANISOTROPY = 0.65
_DEFAULT_AMBIENT = 0.35
_DEFAULT_DISTANCE = 400.0


@dataclass(slots=True)
class VolumetricSettings:
    """Tuning for a participating volumetric medium.

    Attributes:
        density: Extinction coefficient at the medium's base height (1/m).
        height_falloff: Exponential height falloff rate (1/m).
        anisotropy: Henyey-Greenstein ``g``; positive is forward scattering.
        ambient: Ambient hemisphere in-scattering multiplier.
        distance: Maximum integration distance in metres.
    """

    density: float = _DEFAULT_DENSITY
    height_falloff: float = _DEFAULT_HEIGHT_FALLOFF
    anisotropy: float = _DEFAULT_ANISOTROPY
    ambient: float = _DEFAULT_AMBIENT
    distance: float = _DEFAULT_DISTANCE

    @classmethod
    def from_render_config(cls, config: Any) -> "VolumetricSettings":
        """Builds settings from a ``RenderConfig``-like object.

        Missing attributes fall back to the canonical defaults, so partial
        configs (or plain objects) are accepted.
        """
        return cls(
            density=float(getattr(config, "fog_density", _DEFAULT_DENSITY)),
            height_falloff=float(getattr(config, "fog_height_falloff", _DEFAULT_HEIGHT_FALLOFF)),
            anisotropy=float(getattr(config, "fog_anisotropy", _DEFAULT_ANISOTROPY)),
            ambient=float(getattr(config, "fog_ambient", _DEFAULT_AMBIENT)),
            distance=float(getattr(config, "fog_distance", _DEFAULT_DISTANCE)),
        )
