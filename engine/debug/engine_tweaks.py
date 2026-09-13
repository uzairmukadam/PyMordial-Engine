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
    SHADOW_MASK = 6    # Direct Cascaded Shadow Map occlusion mask
    AO = 7             # Ambient Occlusion (SSAO / HBAO / GTAO) mask
    SSGI = 8           # Screen-Space Global Illumination indirect diffuse
    LPV = 9            # Light Propagation Volumes 3D irradiance slice
    GI_TOTAL = 10      # Consolidated Total Indirect Diffuse
    HIZ = 11           # Hierarchical-Z (Hi-Z) Depth Pyramid Mip Level


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
        "shadow_resolution",
        "shadow_mode",
        "shadow_softness",
        "shadow_bias",
        "shadow_normal_bias",
        "shadow_offset_x",
        "shadow_offset_y",
        "shadow_distance",
        "csm_cascades",
        "sscs_enabled",
        "sscs_steps",
        "sscs_thickness",
        "sscs_ray_distance",
        "sscs_max_distance",
        "ao_mode",
        "ao_intensity",
        "ao_radius",
        "gi_mode",
        "ssgi_steps",
        "ssgi_rays",
        "ssgi_thickness",
        "ssgi_ray_distance",
        "ssgi_intensity",
        "lpv_intensity",
        "ibl_enabled",
        "ssr_enabled",
        "ssr_steps",
        "ssr_thickness",
        "ssr_max_roughness",
        "aa_mode",
        "taa_enabled",
        "taa_feedback",
        "taa_sharpness",
        "point_lights_enabled",
        "pom_enabled",
        "pom_height_scale",
        "pom_self_shadow",
        "disp_near_radius",
        "disp_mid_radius",
        "tess_enabled",
        "tess_max_level",
        "tess_med_level",
        "tess_displacement_scale",
        "frustum_cull_enabled",
        "ssdm_enabled",
        "ssdm_scale",
        "vsync_enabled",
        "uncapped_fps",
        "volumetric_fog_enabled",
        "fog_resolution",
        "fog_point_lights",
        "fog_density",
        "fog_height_falloff",
        "fog_anisotropy",
        "fog_distance",
        "fog_ambient",
        "fog_debug_mode",
        "time_of_day",
        "day_speed",
        "atmo_preset",
        "rayleigh_r",
        "rayleigh_g",
        "rayleigh_b",
        "mie_coeff",
        "turbidity",
        "star_intensity",
        "particles_enabled",
        "particle_count",
        "particle_mode",
        "particle_size",
        "particle_turbulence",
        "particle_brightness",
        "dof_enabled",
        "dof_focus_distance",
        "dof_focal_length",
        "dof_aperture",
        "dof_bokeh_shape",
        "dof_anamorphic_ratio",
        "dof_max_coc",
        "motion_blur_enabled",
        "motion_blur_samples",
        "motion_blur_intensity",
        "motion_blur_max_radius",
        "lens_flare_enabled",
        "lens_flare_threshold",
        "lens_flare_streak_intensity",
        "lens_flare_streak_width",
        "lens_flare_ghost_intensity",
        "lens_flare_halo_intensity",
        "chromatic_aberration_enabled",
        "chromatic_aberration_intensity",
        "vignette_enabled",
        "vignette_intensity",
        "vignette_roundness",
        "vignette_smoothness",
        "film_grain_enabled",
        "film_grain_intensity",
        "water_enabled",
        "water_height",
        "water_wave_amplitude",
        "water_wave_speed",
        "water_wave_steepness",
        "water_refraction_enabled",
        "water_refraction_strength",
        "water_foam_enabled",
        "water_foam_threshold",
        "water_foam_scale",
        "water_foam_intensity",
        "water_clarity",
        "water_roughness",
        "water_color_shallow",
        "water_color_deep",
        "hiz_debug_mip",
        "_on_quality_changed",
        "_on_vsync_changed",
    )

    def __init__(self) -> None:
        # Default to CUSTOM so individual element tweaks are not overridden
        self.quality_preset = GraphicsQuality.CUSTOM
        self.tonemap_mode = "ACES"
        self.exposure = 1.0
        self.gbuffer_debug = GBufferDebugMode.DISABLED
        self.hiz_debug_mip = 0

        self.show_wireframe = False
        # Disabled by default per user request
        self.show_physics_colliders = False
        self.show_bounds = False
        self.show_sun_ray = False

        self.sun_angle_deg = 45.0
        self.sun_elevation_deg = 65.0
        self.sun_lux = 4.0

        # Phase 6: Micro-Geometry Displacement Controls
        self.pom_enabled = True
        self.pom_height_scale = 1.0
        self.pom_self_shadow = True

        self.disp_near_radius = 8.0
        self.disp_mid_radius = 25.0
        self.tess_enabled = True
        self.tess_max_level = 24.0
        self.tess_med_level = 8.0
        self.tess_displacement_scale = 1.0
        self.frustum_cull_enabled = True

        self.ssdm_enabled = True
        self.ssdm_scale = 1.0

        # Shadow Settings (AAA Default: 4096 Atlas, PCSS, zero delta offset)
        self.shadow_resolution = 4096
        self.shadow_mode = "PCSS"
        self.shadow_softness = 1.2
        self.shadow_bias = 0.0015
        self.shadow_normal_bias = 0.0010
        self.shadow_offset_x = 0.0
        self.shadow_offset_y = 0.0
        self.shadow_distance = 500.0
        self.csm_cascades = 4
        self.sscs_enabled = False
        self.sscs_steps = 16
        self.sscs_thickness = 0.10
        self.sscs_ray_distance = 0.20
        self.sscs_max_distance = 50.0

        # Phase 5 High-End Graphics Settings (Individually customizable)
        self.ao_mode = "GTAO"
        self.ao_intensity = 1.2
        self.ao_radius = 0.75
        self.gi_mode = GIMode.HYBRID.value
        self.ssgi_steps = 16
        self.ssgi_rays = 8
        self.ssgi_thickness = 0.35
        self.ssgi_ray_distance = 3.0
        self.ssgi_intensity = 1.5
        self.lpv_intensity = 1.0
        self.ibl_enabled = True
        self.ssr_enabled = True
        self.ssr_steps = 32
        self.ssr_thickness = 0.40
        self.ssr_max_roughness = 0.65
        self.aa_mode = "OFF"
        self.taa_enabled = False
        self.taa_feedback = 0.92
        self.point_lights_enabled = True

        # Froxel Volumetric Fog & Atmospheric Light Scattering
        self.volumetric_fog_enabled = True
        self.fog_resolution = "HIGH"
        self.fog_point_lights = True
        self.fog_density = 0.005
        self.fog_height_falloff = 0.10
        self.fog_anisotropy = 0.65
        self.fog_distance = 400.0
        self.fog_ambient = 0.35
        self.fog_debug_mode = 0

        # Physical Atmosphere & Day-Night Cycle
        self.time_of_day = 12.0
        self.day_speed = 0.0          # 0.0 = manual / paused
        self.atmo_preset = "EARTH_DAY"
        self.rayleigh_r = 5.802       # m^-1 * 1e6
        self.rayleigh_g = 13.558
        self.rayleigh_b = 33.100
        self.mie_coeff = 3.996        # m^-1 * 1e6
        self.turbidity = 1.0
        self.star_intensity = 1.0

        # Phase 2: Volumetric Particles & Dust Motes
        self.particles_enabled = True
        self.particle_count = 16384
        self.particle_mode = "DUST_MOTES"
        self.particle_size = 1.0
        self.particle_turbulence = 0.85
        self.particle_brightness = 2.5

        # Phase 3: Cinematic Camera Optics & Lens Effects
        self.dof_enabled = True
        self.dof_focus_distance = 5.0
        self.dof_focal_length = 50.0
        self.dof_aperture = 2.8
        self.dof_bokeh_shape = "CIRCULAR"
        self.dof_anamorphic_ratio = 1.0
        self.dof_max_coc = 24.0

        self.motion_blur_enabled = True
        self.motion_blur_samples = 12
        self.motion_blur_intensity = 1.0
        self.motion_blur_max_radius = 32.0

        self.lens_flare_enabled = True
        self.lens_flare_threshold = 1.8
        self.lens_flare_streak_intensity = 0.6
        self.lens_flare_streak_width = 32.0
        self.lens_flare_ghost_intensity = 0.35
        self.lens_flare_halo_intensity = 0.25

        self.chromatic_aberration_enabled = True
        self.chromatic_aberration_intensity = 0.005
        self.vignette_enabled = True
        self.vignette_intensity = 0.35
        self.vignette_roundness = 0.85
        self.vignette_smoothness = 0.50
        self.film_grain_enabled = True
        self.film_grain_intensity = 0.04

        # Phase 4: Dynamic Water & Screen-Space Refraction
        self.water_enabled = True
        self.water_height = 0.0
        self.water_wave_amplitude = 0.15
        self.water_wave_speed = 1.0
        self.water_wave_steepness = 0.8
        self.water_refraction_enabled = True
        self.water_refraction_strength = 0.03
        self.water_foam_enabled = True
        self.water_foam_threshold = 0.40
        self.water_foam_scale = 6.0
        self.water_foam_intensity = 1.0
        self.water_clarity = 4.0
        self.water_roughness = 0.05
        self.water_color_shallow = (0.05, 0.45, 0.55)
        self.water_color_deep = (0.005, 0.04, 0.15)

        # VSync disabled by default per user request
        self.vsync_enabled = False
        self.uncapped_fps = True
        self._on_quality_changed: list[Callable[[GraphicsQuality], None]] = []
        self._on_vsync_changed: list[Callable[[bool], None]] = []

    def apply_atmo_preset(self, preset_name: str) -> None:
        """Applies a named planet atmosphere preset to tweak fields."""
        from engine.gfx.atmosphere import ATMOSPHERE_PRESETS
        preset = ATMOSPHERE_PRESETS.get(preset_name.upper())
        if preset is not None:
            self.atmo_preset = preset_name.upper()
            if "time_of_day" in preset:
                self.time_of_day = float(preset["time_of_day"])
            if "rayleigh_beta" in preset:
                rb = preset["rayleigh_beta"]
                self.rayleigh_r = float(rb[0] * 1e6)
                self.rayleigh_g = float(rb[1] * 1e6)
                self.rayleigh_b = float(rb[2] * 1e6)
            if "mie_beta" in preset:
                self.mie_coeff = float(preset["mie_beta"] * 1e6)
            if "turbidity" in preset:
                self.turbidity = float(preset["turbidity"])
            if "star_intensity" in preset:
                self.star_intensity = float(preset["star_intensity"])

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
