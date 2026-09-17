"""Tier 2 Debug Subsystem: Engine Graphics, G-Buffer & Pipeline Tweaks.

Provides standardized controls for runtime quality presets, tonemapping operators,
G-Buffer inspection modes, visual 3D wireframes, and shadow parameters.
"""

from __future__ import annotations
import json
from pathlib import Path
from enum import IntEnum
from typing import Any, Callable
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
        "spot_lights_enabled",
        "spot_shadows_enabled",
        "spot_light_radius",
        "spot_cone_angle",
        "spot_shadow_bias",
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
        "bloom_enabled",
        "bloom_intensity",
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
        "_version",
        "_applied_version",
    )

    def __init__(self) -> None:
        super().__setattr__("_version", 0)
        super().__setattr__("_applied_version", -1)
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

        self.disp_near_radius = 120.0
        self.disp_mid_radius = 300.0
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
        self.taa_sharpness = 0.50
        self.point_lights_enabled = True
        self.spot_lights_enabled = True
        self.spot_shadows_enabled = True
        self.spot_light_radius = 35.0
        self.spot_cone_angle = 32.0
        self.spot_shadow_bias = 0.0015

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

        # Phase 2: Volumetric Particles & Dust Motes (Scene-specific)
        self.particles_enabled = False
        self.particle_count = 16384
        self.particle_mode = "OFF"
        self.particle_size = 1.0
        self.particle_turbulence = 0.85
        self.particle_brightness = 2.5

        # Phase 3: Cinematic Camera Optics & Lens Effects
        self.bloom_enabled = True
        self.bloom_intensity = 0.045

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

        # Phase 4: Dynamic Water & Screen-Space Refraction (Scene-specific)
        self.water_enabled = False
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

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name not in ("_version", "_applied_version"):
            try:
                super().__setattr__("_version", getattr(self, "_version", 0) + 1)
            except AttributeError:
                pass

    def apply_to_pipeline(self, pipeline: Any, force: bool = False) -> None:
        """Synchronizes all graphical and post-processing properties to the active pipeline."""
        if pipeline is None:
            return

        if not force and self._version == self._applied_version:
            return

        p_cfg = getattr(pipeline, "config", None)
        if p_cfg is not None:
            # 1. Post-Processing, Optics & Tone-Mapping
            p_cfg.tonemap_mode = self.tonemap_mode
            p_cfg.exposure = float(self.exposure)
            p_cfg.bloom_enabled = bool(self.bloom_enabled)
            p_cfg.bloom_intensity = float(self.bloom_intensity)
            p_cfg.chromatic_aberration_enabled = bool(self.chromatic_aberration_enabled)
            p_cfg.chromatic_aberration_intensity = float(self.chromatic_aberration_intensity)
            p_cfg.vignette_enabled = bool(self.vignette_enabled)
            p_cfg.vignette_intensity = float(self.vignette_intensity)
            p_cfg.vignette_roundness = float(self.vignette_roundness)
            p_cfg.vignette_smoothness = float(self.vignette_smoothness)
            p_cfg.film_grain_enabled = bool(self.film_grain_enabled)
            p_cfg.film_grain_intensity = float(self.film_grain_intensity)

            # 2. Cinematic Camera Optics (DoF, Motion Blur, Lens Flare)
            p_cfg.dof_enabled = bool(self.dof_enabled)
            p_cfg.dof_focus_distance = float(self.dof_focus_distance)
            p_cfg.dof_focal_length = float(self.dof_focal_length)
            p_cfg.dof_aperture = float(self.dof_aperture)
            p_cfg.dof_bokeh_shape = self.dof_bokeh_shape
            p_cfg.dof_anamorphic_ratio = float(self.dof_anamorphic_ratio)
            p_cfg.dof_max_coc = float(self.dof_max_coc)

            p_cfg.motion_blur_enabled = bool(self.motion_blur_enabled)
            p_cfg.motion_blur_samples = int(self.motion_blur_samples)
            p_cfg.motion_blur_intensity = float(self.motion_blur_intensity)
            p_cfg.motion_blur_max_radius = float(self.motion_blur_max_radius)

            p_cfg.lens_flare_enabled = bool(self.lens_flare_enabled)
            p_cfg.lens_flare_threshold = float(self.lens_flare_threshold)
            p_cfg.lens_flare_streak_intensity = float(self.lens_flare_streak_intensity)
            p_cfg.lens_flare_streak_width = float(self.lens_flare_streak_width)
            p_cfg.lens_flare_ghost_intensity = float(self.lens_flare_ghost_intensity)
            p_cfg.lens_flare_halo_intensity = float(self.lens_flare_halo_intensity)

            # 3. Global Illumination & Ambient Occlusion
            p_cfg.gi_mode = self.gi_mode
            p_cfg.ssgi_steps = int(self.ssgi_steps)
            p_cfg.ssgi_rays = int(self.ssgi_rays)
            p_cfg.ssgi_thickness = float(self.ssgi_thickness)
            p_cfg.ssgi_ray_distance = float(self.ssgi_ray_distance)
            p_cfg.ssgi_intensity = float(self.ssgi_intensity)
            p_cfg.lpv_intensity = float(self.lpv_intensity)
            p_cfg.ao_mode = self.ao_mode
            p_cfg.ao_intensity = float(self.ao_intensity)
            p_cfg.ao_radius = float(self.ao_radius)

            # 4. Reflections & Anti-Aliasing
            p_cfg.ibl_enabled = bool(self.ibl_enabled)
            p_cfg.ssr_enabled = bool(self.ssr_enabled)
            p_cfg.ssr_steps = int(self.ssr_steps)
            p_cfg.ssr_thickness = float(self.ssr_thickness)
            p_cfg.ssr_max_roughness = float(self.ssr_max_roughness)
            p_cfg.aa_mode = getattr(self, "aa_mode", "OFF")
            p_cfg.clustered_lights_enabled = bool(self.point_lights_enabled)
            p_cfg.spot_lights_enabled = bool(self.spot_lights_enabled)
            p_cfg.spot_shadows_enabled = bool(self.spot_shadows_enabled)
            p_cfg.spot_light_radius = float(self.spot_light_radius)
            p_cfg.spot_cone_angle = float(self.spot_cone_angle)
            p_cfg.spot_shadow_bias = float(self.spot_shadow_bias)

            # 5. Shadows & CSM
            p_cfg.shadow_mode = self.shadow_mode
            p_cfg.shadow_softness = float(self.shadow_softness)
            p_cfg.shadow_bias = float(self.shadow_bias)
            p_cfg.shadow_normal_bias = float(self.shadow_normal_bias)
            p_cfg.shadow_distance = float(self.shadow_distance)
            p_cfg.csm_cascades = int(self.csm_cascades)
            p_cfg.shadow_resolution = int(self.shadow_resolution)

            if hasattr(pipeline, "csm") and pipeline.csm is not None:
                if self.shadow_resolution != pipeline.csm.atlas_size:
                    pipeline.csm.resize_atlas(self.shadow_resolution)
                if (
                    self.shadow_distance != pipeline.csm.max_distance
                    or self.csm_cascades != pipeline.csm.cascade_count
                ):
                    pipeline.csm.update_splits(self.shadow_distance, self.csm_cascades)

            if hasattr(pipeline, "spot_shadow_map") and pipeline.spot_shadow_map is not None:
                spot_res = getattr(p_cfg, "spot_shadow_resolution", 2048)
                if spot_res != pipeline.spot_shadow_map.atlas_size:
                    pipeline.spot_shadow_map.resize_atlas(spot_res)

            # 6. Micro-Geometry (POM, SSDM, Tessellation)
            p_cfg.pom_enabled = bool(self.pom_enabled)
            p_cfg.pom_height_scale = float(self.pom_height_scale)
            p_cfg.pom_self_shadow = bool(self.pom_self_shadow)
            p_cfg.disp_near_radius = float(self.disp_near_radius)
            p_cfg.disp_mid_radius = float(self.disp_mid_radius)
            p_cfg.tess_enabled = bool(self.tess_enabled)
            p_cfg.tess_max_level = float(self.tess_max_level)
            p_cfg.tess_med_level = float(self.tess_med_level)
            p_cfg.tess_displacement_scale = float(self.tess_displacement_scale)
            p_cfg.frustum_cull_enabled = bool(self.frustum_cull_enabled)
            p_cfg.ssdm_enabled = bool(self.ssdm_enabled)
            p_cfg.ssdm_scale = float(self.ssdm_scale)

            # 7. Froxel Volumetric Fog
            p_cfg.volumetric_fog_enabled = bool(self.volumetric_fog_enabled)
            p_cfg.fog_resolution = str(self.fog_resolution)
            p_cfg.fog_point_lights = bool(self.fog_point_lights)
            p_cfg.fog_density = float(self.fog_density)
            p_cfg.fog_height_falloff = float(self.fog_height_falloff)
            p_cfg.fog_anisotropy = float(self.fog_anisotropy)
            p_cfg.fog_distance = float(self.fog_distance)
            p_cfg.fog_ambient = float(self.fog_ambient)
            p_cfg.fog_debug_mode = int(self.fog_debug_mode)

            # 8. Dynamic Water Simulation
            p_cfg.water_enabled = bool(self.water_enabled)
            p_cfg.water_height = float(self.water_height)
            p_cfg.water_wave_amplitude = float(self.water_wave_amplitude)
            p_cfg.water_wave_speed = float(self.water_wave_speed)
            p_cfg.water_wave_steepness = float(self.water_wave_steepness)
            p_cfg.water_refraction_enabled = bool(self.water_refraction_enabled)
            p_cfg.water_refraction_strength = float(self.water_refraction_strength)
            p_cfg.water_foam_enabled = bool(self.water_foam_enabled)
            p_cfg.water_foam_threshold = float(self.water_foam_threshold)
            p_cfg.water_clarity = float(self.water_clarity)
            p_cfg.water_roughness = float(self.water_roughness)
            if hasattr(pipeline, "water_pass") and pipeline.water_pass is not None:
                pipeline.water_pass.enabled = self.water_enabled

            # 9. GPU Particles
            p_cfg.particles_enabled = bool(self.particles_enabled)
            p_cfg.particle_count = int(self.particle_count)
            p_cfg.particle_mode = str(self.particle_mode)
            p_cfg.particle_size = float(self.particle_size)
            p_cfg.particle_turbulence = float(self.particle_turbulence)
            if hasattr(pipeline, "particle_pass") and pipeline.particle_pass is not None:
                pipeline.particle_pass.set_mode(self.particle_mode if self.particles_enabled else "OFF")
                pipeline.particle_pass.set_active_count(self.particle_count if self.particles_enabled else 0)
                pipeline.particle_pass.base_size_multiplier = self.particle_size
                pipeline.particle_pass.turbulence_strength = self.particle_turbulence
                pipeline.particle_pass.sun_scatter_intensity = getattr(self, "particle_brightness", 2.5)

            # 10. Debug Displays & Sun
            p_cfg.wireframe = bool(self.show_wireframe)
            p_cfg.debug_gbuffer = int(self.gbuffer_debug)
            p_cfg.sun_intensity = float(self.sun_lux)

            # Ensure post_process config is kept in lockstep
            if hasattr(pipeline, "post_process") and pipeline.post_process is not None:
                pipeline.post_process.config = p_cfg

        # 11. Atmosphere System
        if hasattr(pipeline, "atmosphere") and pipeline.atmosphere is not None:
            atmo = pipeline.atmosphere
            if hasattr(atmo, "config") and atmo.config is not None:
                atmo.config.day_speed = float(self.day_speed)
                if self.day_speed > 0.0:
                    self.time_of_day = float(atmo.config.time_of_day)
                else:
                    atmo.config.time_of_day = float(self.time_of_day)
                atmo.config.rayleigh_beta = (
                    float(self.rayleigh_r * 1e-6),
                    float(self.rayleigh_g * 1e-6),
                    float(self.rayleigh_b * 1e-6),
                )
                atmo.config.mie_beta = float(self.mie_coeff * 1e-6)
                atmo.config.turbidity = float(self.turbidity)
                atmo.config.star_intensity = float(self.star_intensity)

        self._applied_version = self._version

    def to_dict(self) -> dict[str, Any]:
        """Serializes current graphics configuration into a JSON-compatible dictionary."""
        return {
            "quality_preset": self.quality_preset.value if hasattr(self.quality_preset, "value") else str(self.quality_preset),
            "show_wireframe": bool(self.show_wireframe),
            "vsync_enabled": bool(self.vsync_enabled),
            "tonemap_mode": str(self.tonemap_mode),
            "exposure": float(self.exposure),
            "bloom_enabled": bool(self.bloom_enabled),
            "bloom_intensity": float(self.bloom_intensity),
            "chromatic_aberration_enabled": bool(self.chromatic_aberration_enabled),
            "chromatic_aberration_intensity": float(self.chromatic_aberration_intensity),
            "vignette_enabled": bool(self.vignette_enabled),
            "vignette_intensity": float(self.vignette_intensity),
            "vignette_roundness": float(self.vignette_roundness),
            "vignette_smoothness": float(self.vignette_smoothness),
            "film_grain_enabled": bool(self.film_grain_enabled),
            "film_grain_intensity": float(self.film_grain_intensity),
            "dof_enabled": bool(self.dof_enabled),
            "dof_focus_distance": float(self.dof_focus_distance),
            "dof_focal_length": float(self.dof_focal_length),
            "dof_aperture": float(self.dof_aperture),
            "dof_bokeh_shape": str(self.dof_bokeh_shape),
            "dof_anamorphic_ratio": float(self.dof_anamorphic_ratio),
            "dof_max_coc": float(self.dof_max_coc),
            "motion_blur_enabled": bool(self.motion_blur_enabled),
            "motion_blur_samples": int(self.motion_blur_samples),
            "motion_blur_intensity": float(self.motion_blur_intensity),
            "motion_blur_max_radius": float(self.motion_blur_max_radius),
            "lens_flare_enabled": bool(self.lens_flare_enabled),
            "lens_flare_threshold": float(self.lens_flare_threshold),
            "lens_flare_streak_intensity": float(self.lens_flare_streak_intensity),
            "lens_flare_streak_width": float(self.lens_flare_streak_width),
            "lens_flare_ghost_intensity": float(self.lens_flare_ghost_intensity),
            "lens_flare_halo_intensity": float(self.lens_flare_halo_intensity),
            "gi_mode": str(self.gi_mode),
            "ssgi_steps": int(self.ssgi_steps),
            "ssgi_rays": int(self.ssgi_rays),
            "ssgi_thickness": float(self.ssgi_thickness),
            "ssgi_ray_distance": float(self.ssgi_ray_distance),
            "ssgi_intensity": float(self.ssgi_intensity),
            "lpv_intensity": float(self.lpv_intensity),
            "ao_mode": str(self.ao_mode),
            "ao_intensity": float(self.ao_intensity),
            "ao_radius": float(self.ao_radius),
            "ibl_enabled": bool(self.ibl_enabled),
            "ssr_enabled": bool(self.ssr_enabled),
            "ssr_steps": int(self.ssr_steps),
            "ssr_thickness": float(self.ssr_thickness),
            "ssr_max_roughness": float(self.ssr_max_roughness),
            "aa_mode": str(getattr(self, "aa_mode", "OFF")),
            "point_lights_enabled": bool(self.point_lights_enabled),
            "spot_lights_enabled": bool(self.spot_lights_enabled),
            "spot_shadows_enabled": bool(self.spot_shadows_enabled),
            "spot_light_radius": float(self.spot_light_radius),
            "spot_cone_angle": float(self.spot_cone_angle),
            "spot_shadow_bias": float(self.spot_shadow_bias),
            "shadow_mode": str(self.shadow_mode),
            "shadow_softness": float(self.shadow_softness),
            "shadow_bias": float(self.shadow_bias),
            "shadow_normal_bias": float(self.shadow_normal_bias),
            "shadow_distance": float(self.shadow_distance),
            "csm_cascades": int(self.csm_cascades),
            "shadow_resolution": int(self.shadow_resolution),
            "volumetric_fog_enabled": bool(self.volumetric_fog_enabled),
            "fog_resolution": str(self.fog_resolution),
            "fog_point_lights": bool(self.fog_point_lights),
            "fog_density": float(self.fog_density),
            "fog_height_falloff": float(self.fog_height_falloff),
            "fog_anisotropy": float(self.fog_anisotropy),
            "fog_distance": float(self.fog_distance),
            "fog_ambient": float(self.fog_ambient),
            "fog_debug_mode": int(self.fog_debug_mode),
            "sun_lux": float(self.sun_lux),
            "time_of_day": float(self.time_of_day),
            "day_speed": float(self.day_speed),
            "atmo_preset": str(self.atmo_preset),
            "rayleigh_r": float(self.rayleigh_r),
            "rayleigh_g": float(self.rayleigh_g),
            "rayleigh_b": float(self.rayleigh_b),
            "mie_coeff": float(self.mie_coeff),
            "turbidity": float(self.turbidity),
            "star_intensity": float(self.star_intensity),
            "uncapped_fps": bool(self.uncapped_fps),
            "pom_enabled": bool(self.pom_enabled),
            "pom_height_scale": float(self.pom_height_scale),
            "pom_self_shadow": bool(self.pom_self_shadow),
            "disp_near_radius": float(self.disp_near_radius),
            "disp_mid_radius": float(self.disp_mid_radius),
            "tess_enabled": bool(self.tess_enabled),
            "tess_max_level": float(self.tess_max_level),
            "tess_med_level": float(self.tess_med_level),
            "tess_displacement_scale": float(self.tess_displacement_scale),
            "frustum_cull_enabled": bool(self.frustum_cull_enabled),
            "ssdm_enabled": bool(self.ssdm_enabled),
            "ssdm_scale": float(self.ssdm_scale),
            "water_enabled": bool(self.water_enabled),
            "particles_enabled": bool(self.particles_enabled),
            "show_bounds": bool(self.show_bounds),
            "show_physics_colliders": bool(self.show_physics_colliders),
            "shadow_offset_x": float(self.shadow_offset_x),
            "shadow_offset_y": float(self.shadow_offset_y),
            "sscs_enabled": bool(self.sscs_enabled),
            "sscs_steps": int(self.sscs_steps),
            "sscs_thickness": float(self.sscs_thickness),
            "sscs_ray_distance": float(self.sscs_ray_distance),
            "sscs_max_distance": float(self.sscs_max_distance),
            "taa_enabled": bool(self.taa_enabled),
            "taa_feedback": float(self.taa_feedback),
            "taa_sharpness": float(self.taa_sharpness),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restores graphics configuration from a dictionary."""
        if not isinstance(data, dict):
            return

        if "quality_preset" in data:
            try:
                self.quality_preset = GraphicsQuality(data["quality_preset"])
            except Exception:
                self.quality_preset = GraphicsQuality.CUSTOM

        for key, val in data.items():
            if hasattr(self, key) and key != "quality_preset":
                current = getattr(self, key)
                if isinstance(current, bool):
                    setattr(self, key, bool(val))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(self, key, int(val))
                elif isinstance(current, float):
                    setattr(self, key, float(val))
                elif isinstance(current, str):
                    setattr(self, key, str(val))
                else:
                    setattr(self, key, val)

    def save_to_file(self, filepath: str | Path) -> bool:
        """Writes configuration to JSON file in the target directory."""
        try:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def load_from_file(self, filepath: str | Path) -> bool:
        """Loads configuration from JSON file in the target directory."""
        try:
            p = Path(filepath)
            if not p.is_file():
                return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
            return True
        except Exception:
            return False
