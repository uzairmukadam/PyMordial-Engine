"""Deprecated import path for the Atmosphere & Sky engine module.

The implementation now lives in :mod:`engine.modules.atmosphere`. This module is
kept so existing imports keep working; new code should import from the module
package directly.
"""

from __future__ import annotations

from engine.modules.atmosphere.config import ATMOSPHERE_PRESETS, AtmosphereConfig
from engine.modules.atmosphere.system import AtmosphereSystem

__all__ = ["ATMOSPHERE_PRESETS", "AtmosphereConfig", "AtmosphereSystem"]
