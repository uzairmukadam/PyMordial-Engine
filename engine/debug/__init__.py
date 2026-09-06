"""Dedicated 3-Tiered Debug Subsystem for PyMordial Engine.

Exposes:
- Tier 1: SystemMonitor, ProfileScope (CPU profiling, 1% low, memory gauges)
- Tier 2: EngineTweaks, GBufferDebugMode (Presets, tonemapping, wireframes, sun)
- Tier 3: GameTweaks, TweakType, TweakItem (Declarative developer inspector)
- DebugDraw: Immediate-mode 3D wireframe gizmo renderer
- DebugMenu: Glassmorphic 3-tab debug overlay UI
- DebugToast: On-screen notification toasts
"""

from engine.debug.monitor import SystemMonitor, ProfileScope
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.debug.game_tweaks import GameTweaks, TweakType, TweakItem
from engine.debug.debug_draw import DebugDraw
from engine.debug.debug_menu import DebugMenu
from engine.debug.toast import DebugToast

__all__ = [
    "SystemMonitor",
    "ProfileScope",
    "EngineTweaks",
    "GBufferDebugMode",
    "GameTweaks",
    "TweakType",
    "TweakItem",
    "DebugDraw",
    "DebugMenu",
    "DebugToast",
]
