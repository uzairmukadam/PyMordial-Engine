"""Tier 2 Debug Subsystem: Engine Graphics, G-Buffer & Pipeline Tweaks.

Provides standardized controls for runtime quality presets, tonemapping operators,
G-Buffer inspection modes, visual 3D wireframes, and shadow parameters.
"""

from __future__ import annotations
from enum import IntEnum
from typing import Callable
from engine.gfx.quality_presets import GraphicsQuality, GIMode


class GBufferDebugMode(IntEnum):
    """G-Buffer target debug visualization modes."""

    DISABLED = 0       # Standard deferred Cook-Torrance PBR resolve
    ALBEDO = 1         # RT0: Base Color RGB
    NORMALS = 2        # RT1: World-space Normals (mapped to [0, 1])
    MATERIAL = 3       # RT2: Roughness, Metallic & Material parameters
    DEPTH = 4          # RT3: Reversed-Z Depth linearized
    SHADOW_ATLAS = 5   # Cascaded Shadow Maps depth atlas


class EngineTweaks:
    """Central registry of engine graphical, lighting, and pipeline tweaks."""

    __slots__ = (
        "quality_preset",
        "tonemap_mode",
        "exposure",
        "gbuffer_debug",
        "show_wireframe",
        "show_physics_colliders",
        "show_bounds",
        "show_sun_ray",
        "sun_angle_deg",
        "sun_elevation_deg",
        "sun_lux",
        "sscs_enabled",
        "sscs_steps",
        "sscs_thickness",
        "ao_mode",
        "ao_intensity",
        "gi_mode",
        "ssgi_steps",
        "ssgi_intensity",
        "lpv_intensity",
        "ibl_enabled",
        "ssr_enabled",
        "ssr_steps",
        "taa_enabled",
        "point_lights_enabled",
        "vsync_enabled",
        "uncapped_fps",
        "_on_quality_changed",
        "_on_vsync_changed",
    )

    def __init__(self) -> None:
        # Default to CUSTOM so individual element tweaks are not overridden
        self.quality_preset = GraphicsQuality.CUSTOM
        self.tonemap_mode = "ACES"
        self.exposure = 1.0
        self.gbuffer_debug = GBufferDebugMode.DISABLED

        self.show_wireframe = False
        # Disabled by default per user request
        self.show_physics_colliders = False
        self.show_bounds = False
        self.show_sun_ray = False

        self.sun_angle_deg = 45.0
        self.sun_elevation_deg = 65.0
        self.sun_lux = 4.0

        self.sscs_enabled = True
        self.sscs_steps = 16
        self.sscs_thickness = 0.05

        # Phase 5 High-End Graphics Settings (Individually customizable)
        self.ao_mode = "GTAO"
        self.ao_intensity = 1.0
        self.gi_mode = GIMode.HYBRID.value
        self.ssgi_steps = 12
        self.ssgi_intensity = 1.2
        self.lpv_intensity = 1.0
        self.ibl_enabled = True
        self.ssr_enabled = True
        self.ssr_steps = 24
        self.taa_enabled = True
        self.point_lights_enabled = True

        # VSync disabled by default per user request
        self.vsync_enabled = False
        self.uncapped_fps = True
        self._on_quality_changed: list[Callable[[GraphicsQuality], None]] = []
        self._on_vsync_changed: list[Callable[[bool], None]] = []

    def set_quality_preset(self, preset: GraphicsQuality) -> None:
        """Updates the graphics preset and triggers registered reconfigure listeners."""
        self.quality_preset = preset
        for cb in self._on_quality_changed:
            cb(preset)

    set_preset = set_quality_preset

    def mark_custom(self) -> None:
        """Switches quality preset mode to CUSTOM when individual elements are modified."""
        self.quality_preset = GraphicsQuality.CUSTOM

    def set_vsync(self, enabled: bool) -> None:
        """Toggles VSync and notifies registered listeners."""
        self.vsync_enabled = enabled
        for cb in self._on_vsync_changed:
            cb(enabled)

    def on_vsync_changed(self, listener: Callable[[bool], None]) -> None:
        """Registers a listener called whenever VSync is toggled."""
        self._on_vsync_changed.append(listener)

    def cycle_quality_preset(self) -> GraphicsQuality:
        """Cycles to the next preset: CUSTOM -> LOW -> MED -> HIGH -> ULTRA -> CINEMATIC."""
        presets = [
            GraphicsQuality.CUSTOM,
            GraphicsQuality.LOW,
            GraphicsQuality.MEDIUM,
            GraphicsQuality.HIGH,
            GraphicsQuality.ULTRA,
            GraphicsQuality.CINEMATIC,
        ]
        curr_idx = presets.index(self.quality_preset) if self.quality_preset in presets else 0
        next_preset = presets[(curr_idx + 1) % len(presets)]
        self.set_quality_preset(next_preset)
        return next_preset

    def cycle_tonemapper(self) -> str:
        """Cycles between ACES, AgX, and Reinhard tonemapping operators."""
        modes = ["ACES", "AgX", "Reinhard"]
        curr_idx = modes.index(self.tonemap_mode) if self.tonemap_mode in modes else 0
        self.tonemap_mode = modes[(curr_idx + 1) % len(modes)]
        return self.tonemap_mode

    def cycle_gbuffer_debug(self) -> GBufferDebugMode:
        """Cycles through G-Buffer visualization inspection modes."""
        modes = list(GBufferDebugMode)
        curr_idx = modes.index(self.gbuffer_debug)
        self.gbuffer_debug = modes[(curr_idx + 1) % len(modes)]
        return self.gbuffer_debug

    def on_quality_changed(self, listener: Callable[[GraphicsQuality], None]) -> None:
        """Registers a listener called whenever the graphics preset changes."""
        self._on_quality_changed.append(listener)

    def toggle_wireframe(self) -> bool:
        """Toggles polygon wireframe rasterization mode."""
        self.show_wireframe = not self.show_wireframe
        return self.show_wireframe

    def toggle_physics_colliders(self) -> bool:
        """Toggles immediate-mode physics colliders & gizmos."""
        self.show_physics_colliders = not self.show_physics_colliders
        return self.show_physics_colliders
