"""Built-in Entity Scripts for PyMordial Mapping System."""

from __future__ import annotations

from engine.world.mapping.scripts.interactive_lever import InteractiveLeverScript
from engine.world.mapping.scripts.moving_platform import MovingPlatformScript
from engine.world.mapping.scripts.sliding_gate import SlidingGateScript

__all__ = [
    "MovingPlatformScript",
    "InteractiveLeverScript",
    "SlidingGateScript",
]
