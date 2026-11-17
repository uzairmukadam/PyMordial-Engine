"""PyMordial Atmosphere & Sky Engine Module.

Provides physically based atmospheric scattering (Sébastien Hillaire,
Eurographics 2020), a 24-hour solar cycle, procedural star fields, planet
presets, and a procedural cloud deck.

The module is enabled by default and can be disabled or configured from project
configuration without editing ``RenderPipeline``:

    ProjectConfig(
        atmosphere_enabled=False,                       # disable entirely
        atmosphere_settings={"preset": "EARTH_SUNSET"},  # configure
    )
"""

from __future__ import annotations

from typing import Any

import moderngl

from engine.modules.atmosphere.config import ATMOSPHERE_PRESETS, AtmosphereConfig
from engine.modules.atmosphere.environment import AtmosphereEnvironment
from engine.modules.atmosphere.global_light import GlobalLightSystem
from engine.modules.atmosphere.module import AtmosphereModule
from engine.modules.atmosphere.system import AtmosphereSystem

MODULE_MANIFEST = AtmosphereModule.MODULE_MANIFEST


def create_atmosphere_environment(
    ctx: moderngl.Context,
    settings: dict[str, Any] | None = None,
) -> AtmosphereEnvironment:
    """Builds the atmosphere environment provider from project settings.

    Recognized settings keys:
        ``preset``: name of an entry in ``ATMOSPHERE_PRESETS``.
        ``time_of_day``, ``day_speed``, ``azimuth_deg``, ``latitude_deg``,
        ``declination_deg``, ``cloud_enabled``: direct ``AtmosphereConfig``
        overrides.
    """
    config = AtmosphereConfig()
    settings = dict(settings or {})

    preset = settings.pop("preset", None)
    if preset:
        system = AtmosphereSystem(config)
        system.apply_preset(str(preset))
        config = system.config

    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return AtmosphereEnvironment(ctx, config)


__all__ = [
    "ATMOSPHERE_PRESETS",
    "AtmosphereConfig",
    "AtmosphereEnvironment",
    "AtmosphereModule",
    "AtmosphereSystem",
    "GlobalLightSystem",
    "MODULE_MANIFEST",
    "create_atmosphere_environment",
]
