"""Tier 2 Debug Subsystem: Engine Graphics, G-Buffer & Pipeline Tweaks.

Provides standardized controls for runtime quality presets, tonemapping operators,
G-Buffer inspection modes, visual 3D wireframes, and shadow parameters.
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

from engine.gfx.quality_presets import GIMode, GraphicsQuality
from engine.logging import log_error


class GBufferDebugMode(IntEnum):
    """G-Buffer target debug visualization modes."""

    DISABLED = 0  # Standard deferred Cook-Torrance PBR resolve
    ALBEDO = 1  # RT0: Base Color RGB
    NORMALS = 2  # RT1: World-space Normals (mapped to [0, 1])
    MATERIAL = 3  # RT2: Roughness, Metallic & Material parameters
    DEPTH = 4  # RT3: Reversed-Z depth
    SHADOW_ATLAS = 5  # Cascaded Shadow Maps depth atlas
    SHADOW_MASK = 6  # Direct Cascaded Shadow Map occlusion mask
    AO = 7  # Ambient Occlusion (SSAO / GTAO) mask
    SSGI = 8  # Screen-Space Global Illumination indirect diffuse
    GI_TOTAL = 10  # Consolidated Total Indirect Diffuse
    HIZ = 11  # Hierarchical-Z (Hi-Z) Depth Pyramid Mip Level
    SPOT_LIGHTS = 12  # Accumulated dynamic spot-light contribution
    RADIANCE_CASCADES = 13  # Radiance Cascades indirect diffuse
    CASCADE_INDEX = 14  # Selected CSM cascade index (false colour)
    CASCADE_COVERAGE = 15  # Per-cascade coverage / border fade
    CASCADE_BOUNDS = 16  # Cascade start/end bands with split-distance overlay


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
        "shadow_bias",
        "shadow_normal_bias",
        "shadow_distance",
        "csm_split_lambda",
        "csm_cascade_distances",
        "csm_cascades",
        "ao_mode",
        "ao_intensity",
        "ao_radius",
        "ao_resolution_scale",
        "gi_mode",
        "ssgi_steps",
        "ssgi_rays",
        "ssgi_thickness",
        "ssgi_ray_distance",
        "ssgi_intensity",
        "ssgi_resolution_scale",
        "rc_ray_count",
        "rc_intensity",
        "rc_interval_c0",
        "rc_interval_c1",
        "rc_interval_c2",
        "rc_steps_c0",
        "rc_steps_c1",
        "rc_thickness",
        "rc_temporal_blend",
        "rc_resolution_scale",
        "ibl_enabled",
        "ssr_enabled",
        "ssr_steps",
        "ssr_thickness",
        "ssr_max_roughness",
        "ssr_max_distance",
        "ssr_resolution_scale",
        "fxaa_subpixel",
        "fxaa_edge_threshold",
        "smaa_threshold",
        "aa_mode",
        "taa_enabled",
        "taa_feedback",
        "taa_sharpness",
        "taa_gamma",
        "taa_max_feedback",
        "taa_min_feedback",
        "taa_variance_gamma",
        "point_lights_enabled",
        "spot_lights_enabled",
        "spot_shadows_enabled",
        "spot_shadow_resolution",
        "spot_light_radius",
        "spot_cone_angle",
        "spot_shadow_bias",
        "pom_enabled",
        "pom_min_samples",
        "pom_max_samples",
        "pom_height_scale",
        "pom_self_shadow",
        "disp_near_radius",
        "disp_mid_radius",
        "tess_enabled",
        "tess_max_level",
        "tess_med_level",
        "tess_displacement_scale",
        "frustum_cull_enabled",
        "hiz_cull_enabled",
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
        "azimuth_deg",
        "latitude_deg",
        "declination_deg",
        "moon_orbit_offset_hours",
        "moon_declination_deg",
        "moon_azimuth_offset_deg",
        "sky_exposure",
        "sky_saturation",
        "atmo_preset",
        "sky_mode",
        "rayleigh_r",
        "rayleigh_g",
        "rayleigh_b",
        "mie_coeff",
        "mie_asymmetry",
        "turbidity",
        "star_intensity",
        "star_density",
        "cloud_enabled",
        "cloud_coverage",
        "cloud_opacity",
        "cloud_altitude",
        "cloud_scale",
        "cloud_layer_separation",
        "cloud_secondary_opacity",
        "cloud_wind_x",
        "cloud_wind_z",
        "space_zenith",
        "nebula_color",
        "nebula_intensity",
        "sun_color",
        "moon_color",
        "moon_intensity",
        "ground_color",
        "night_zenith",
        "night_horizon",
        "sun_disc_size",
        "moon_disc_size",
        "planet_radius",
        "atmosphere_radius",
        "h_r",
        "h_m",
        "ozone_center",
        "ozone_width",
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
        "lens_flare_mode",
        "lens_flare_threshold",
        "lens_flare_streak_intensity",
        "lens_flare_streak_width",
        "lens_flare_ghost_intensity",
        "lens_flare_halo_intensity",
        "lens_flare_tint",
        "lens_flare_resolution_scale",
        "solar_flare_blade_count",
        "solar_flare_bloom_intensity",
        "solar_flare_starburst_intensity",
        "solar_flare_occlusion_fade",
        "chromatic_aberration_enabled",
        "chromatic_aberration_intensity",
        "vignette_enabled",
        "vignette_intensity",
        "vignette_roundness",
        "vignette_smoothness",
        "film_grain_enabled",
        "film_grain_intensity",
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
        self.pom_min_samples = 8
        self.pom_max_samples = 64
        self.pom_height_scale = 1.0
        self.pom_self_shadow = True

        self.disp_near_radius = 120.0
        self.disp_mid_radius = 300.0
        self.tess_enabled = True
        self.tess_max_level = 24.0
        self.tess_med_level = 8.0
        self.tess_displacement_scale = 1.0
        self.frustum_cull_enabled = True
        self.hiz_cull_enabled = False

        self.ssdm_enabled = True
        self.ssdm_scale = 1.0

        # Shadow Settings (Default: 8192 Atlas, hard shadows)
        self.shadow_resolution = 8192
        self.shadow_bias = 0.0015
        self.shadow_normal_bias = 0.0010
        self.shadow_distance = 500.0
        self.csm_split_lambda = 0.85
        # Explicit far distance of each cascade in meters. Cascade 0 is the
        # tightest near-field slice; cascade 3 reaches the far plane.
        self.csm_cascade_distances = (20.0, 60.0, 160.0, 500.0)
        self.csm_cascades = 4

        # Phase 5 High-End Graphics Settings (Individually customizable)
        self.ao_mode = "GTAO"
        self.ao_intensity = 1.2
        self.ao_radius = 0.35
        self.ao_resolution_scale = 1.0
        self.gi_mode = GIMode.SSGI.value
        self.ssgi_steps = 16
        self.ssgi_rays = 8
        self.ssgi_thickness = 0.35
        self.ssgi_ray_distance = 3.0
        self.ssgi_intensity = 1.5
        self.ssgi_resolution_scale = 1.0
        self.rc_ray_count = 16
        self.rc_intensity = 1.0
        self.rc_interval_c0 = 0.40
        self.rc_interval_c1 = 2.50
        self.rc_interval_c2 = 40.0
        self.rc_steps_c0 = 8
        self.rc_steps_c1 = 12
        self.rc_thickness = 0.15
        self.rc_temporal_blend = 0.96
        self.rc_resolution_scale = 1.0
        self.ibl_enabled = True
        self.ssr_enabled = True
        self.ssr_steps = 64
        self.ssr_thickness = 0.10
        self.ssr_max_roughness = 0.65
        self.ssr_max_distance = 30.0
        self.ssr_resolution_scale = 1.0
        self.fxaa_subpixel = 0.75
        self.fxaa_edge_threshold = 0.125
        self.smaa_threshold = 0.08
        self.aa_mode = "OFF"
        self.taa_enabled = False
        self.taa_feedback = 0.95
        self.taa_sharpness = 0.35
        self.taa_gamma = 1.25
        self.taa_max_feedback = 0.95
        self.taa_min_feedback = 0.78
        self.taa_variance_gamma = 1.25
        self.point_lights_enabled = True
        self.spot_lights_enabled = True
        self.spot_shadows_enabled = True
        self.spot_shadow_resolution = 2048
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
        self.day_speed = 0.0  # 0.0 = manual / paused
        self.azimuth_deg = 45.0
        self.latitude_deg = 35.0
        self.declination_deg = 0.0
        self.moon_orbit_offset_hours = 12.0
        self.moon_declination_deg = 0.0
        self.moon_azimuth_offset_deg = 0.0
        self.sky_exposure = 1.0
        self.sky_saturation = 1.0
        self.atmo_preset = "EARTH_DAY"
        self.rayleigh_r = 5.802  # m^-1 * 1e6
        self.rayleigh_g = 13.558
        self.rayleigh_b = 33.100
        self.mie_coeff = 3.996  # m^-1 * 1e6
        self.mie_asymmetry = 0.80
        self.turbidity = 1.0
        self.star_intensity = 1.0
        self.star_density = 1.0
        self.sky_mode = "ATMOSPHERE"
        self.cloud_enabled = True
        self.cloud_coverage = 0.46
        self.cloud_opacity = 0.88
        self.cloud_altitude = 1800.0
        self.cloud_scale = 18000.0
        self.cloud_layer_separation = 6500.0
        self.cloud_secondary_opacity = 0.22
        self.cloud_wind_x = 8.0
        self.cloud_wind_z = 2.0
        self.space_zenith = (0.0015, 0.0020, 0.0060)
        self.nebula_color = (0.16, 0.055, 0.28)
        self.nebula_intensity = 0.35
        self.sun_color = (1.0, 0.98, 0.92)
        self.moon_color = (0.70, 0.82, 1.0)
        self.moon_intensity = 0.40
        self.ground_color = (0.15, 0.15, 0.15)
        self.night_zenith = (0.012, 0.025, 0.050)
        self.night_horizon = (0.035, 0.055, 0.090)
        self.sun_disc_size = 0.00465
        self.moon_disc_size = 0.00450
        self.planet_radius = 6360000.0
        self.atmosphere_radius = 6460000.0
        self.h_r = 8000.0
        self.h_m = 1200.0
        self.ozone_center = 25000.0
        self.ozone_width = 15000.0

        # Phase 2: Volumetric Particles & Dust Motes (Scene-specific)
        self.particles_enabled = False
        self.particle_count = 16384
        self.particle_mode = "OFF"
        self.particle_size = 1.0
        self.particle_turbulence = 0.85
        self.particle_brightness = 2.5

        # Phase 3: Cinematic Camera Optics & Lens Effects
        self.bloom_enabled = True
        self.bloom_intensity = 0.12

        self.dof_enabled = False
        self.dof_focus_distance = 5.0
        self.dof_focal_length = 50.0
        self.dof_aperture = 2.8
        self.dof_bokeh_shape = "CIRCULAR"
        self.dof_anamorphic_ratio = 1.0
        self.dof_max_coc = 24.0

        self.motion_blur_enabled = True
        self.motion_blur_samples = 12
        self.motion_blur_intensity = 1.5
        self.motion_blur_max_radius = 48.0

        self.lens_flare_enabled = True
        self.lens_flare_mode = "ANAMORPHIC"
        self.lens_flare_threshold = 1.8
        self.lens_flare_streak_intensity = 0.6
        self.lens_flare_streak_width = 32.0
        self.lens_flare_ghost_intensity = 0.35
        self.lens_flare_halo_intensity = 0.25
        self.lens_flare_tint = (0.25, 0.65, 1.0)
        self.lens_flare_resolution_scale = 1.0
        self.solar_flare_blade_count = 6
        self.solar_flare_bloom_intensity = 1.0
        self.solar_flare_starburst_intensity = 0.6
        self.solar_flare_occlusion_fade = 1.0

        self.chromatic_aberration_enabled = True
        self.chromatic_aberration_intensity = 0.005
        self.vignette_enabled = True
        self.vignette_intensity = 0.35
        self.vignette_roundness = 0.85
        self.vignette_smoothness = 0.50
        self.film_grain_enabled = True
        self.film_grain_intensity = 0.03

        # VSync disabled by default per user request
        self.vsync_enabled = False
        self.uncapped_fps = True
        self._on_quality_changed: list[Callable[[GraphicsQuality], None]] = []
        self._on_vsync_changed: list[Callable[[bool], None]] = []

    def apply_atmo_preset(self, preset_name: str) -> None:
        """Applies a named planet atmosphere preset to tweak fields."""
        from engine.modules.atmosphere.config import ATMOSPHERE_PRESETS

        preset = ATMOSPHERE_PRESETS.get(preset_name.upper())
        if preset is not None:
            self.atmo_preset = preset_name.upper()
            for name in (
                "sky_mode",
                "time_of_day",
                "day_speed",
                "azimuth_deg",
                "latitude_deg",
                "declination_deg",
                "moon_orbit_offset_hours",
                "moon_declination_deg",
                "moon_azimuth_offset_deg",
                "sky_exposure",
                "sky_saturation",
                "sun_intensity",
                "mie_asymmetry",
                "star_intensity",
                "star_density",
                "cloud_enabled",
                "cloud_coverage",
                "cloud_opacity",
                "cloud_altitude",
                "cloud_scale",
                "cloud_layer_separation",
                "cloud_secondary_opacity",
                "space_zenith",
                "nebula_color",
                "nebula_intensity",
                "sun_color",
                "moon_color",
                "moon_intensity",
                "ground_color",
                "night_zenith",
                "night_horizon",
                "sun_disc_size",
                "moon_disc_size",
                "planet_radius",
                "atmosphere_radius",
                "h_r",
                "h_m",
                "ozone_center",
                "ozone_width",
            ):
                if name in preset:
                    value = preset[name]
                    setattr(self, "sun_lux" if name == "sun_intensity" else name, value)
            if "cloud_wind" in preset:
                self.cloud_wind_x, self.cloud_wind_z = preset["cloud_wind"]
            if "rayleigh_beta" in preset:
                rb = preset["rayleigh_beta"]
                self.rayleigh_r = float(rb[0] * 1e6)
                self.rayleigh_g = float(rb[1] * 1e6)
                self.rayleigh_b = float(rb[2] * 1e6)
            if "mie_beta" in preset:
                self.mie_coeff = float(preset["mie_beta"] * 1e6)
            if "turbidity" in preset:
                self.turbidity = float(preset["turbidity"])

    def set_quality_preset(self, preset: GraphicsQuality, preserve_creative_optics: bool = True) -> None:
        """Updates the graphics preset and triggers registered reconfigure listeners.

        Args:
            preset: Target hardware graphics quality preset.
            preserve_creative_optics: If True, preserves player artistic camera choices
                (DoF/lens blur, motion blur, lens flare, vignette, chromatic aberration, film grain)
                rather than resetting them to hardware tier defaults.
        """
        from engine.gfx.quality_presets import QUALITY_PRESETS

        self.quality_preset = preset
        if preset != GraphicsQuality.CUSTOM and preset in QUALITY_PRESETS:
            cfg = QUALITY_PRESETS[preset]
            self.sync_from_config(cfg, preserve_creative_optics=preserve_creative_optics)
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

    def reset_to_defaults(self) -> None:
        """Restores every tweak while retaining registered change listeners."""
        defaults = type(self)()
        excluded = {
            "_on_quality_changed",
            "_on_vsync_changed",
            "_version",
            "_applied_version",
        }
        for name in self.__slots__:
            if name not in excluded:
                setattr(self, name, getattr(defaults, name))
        for cb in self._on_quality_changed:
            cb(self.quality_preset)
        for cb in self._on_vsync_changed:
            cb(self.vsync_enabled)

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
            p_cfg.lens_flare_mode = str(self.lens_flare_mode)
            p_cfg.lens_flare_threshold = float(self.lens_flare_threshold)
            p_cfg.lens_flare_streak_intensity = float(self.lens_flare_streak_intensity)
            p_cfg.lens_flare_streak_width = float(self.lens_flare_streak_width)
            p_cfg.lens_flare_ghost_intensity = float(self.lens_flare_ghost_intensity)
            p_cfg.lens_flare_halo_intensity = float(self.lens_flare_halo_intensity)
            p_cfg.lens_flare_tint = tuple(float(c) for c in self.lens_flare_tint)
            p_cfg.lens_flare_resolution_scale = float(self.lens_flare_resolution_scale)
            p_cfg.solar_flare_blade_count = int(self.solar_flare_blade_count)
            p_cfg.solar_flare_bloom_intensity = float(self.solar_flare_bloom_intensity)
            p_cfg.solar_flare_starburst_intensity = float(self.solar_flare_starburst_intensity)
            p_cfg.solar_flare_occlusion_fade = float(self.solar_flare_occlusion_fade)

            # 3. Global Illumination & Ambient Occlusion
            p_cfg.gi_mode = self.gi_mode
            p_cfg.ssgi_steps = int(self.ssgi_steps)
            p_cfg.ssgi_rays = int(self.ssgi_rays)
            p_cfg.ssgi_thickness = float(self.ssgi_thickness)
            p_cfg.ssgi_ray_distance = float(self.ssgi_ray_distance)
            p_cfg.ssgi_intensity = float(self.ssgi_intensity)
            p_cfg.ssgi_resolution_scale = float(self.ssgi_resolution_scale)
            p_cfg.rc_ray_count = int(self.rc_ray_count)
            p_cfg.rc_intensity = float(self.rc_intensity)
            p_cfg.rc_interval_c0 = float(self.rc_interval_c0)
            p_cfg.rc_interval_c1 = float(self.rc_interval_c1)
            p_cfg.rc_interval_c2 = float(self.rc_interval_c2)
            p_cfg.rc_steps_c0 = int(self.rc_steps_c0)
            p_cfg.rc_steps_c1 = int(self.rc_steps_c1)
            p_cfg.rc_thickness = float(self.rc_thickness)
            p_cfg.rc_temporal_blend = float(self.rc_temporal_blend)
            p_cfg.rc_resolution_scale = float(self.rc_resolution_scale)
            p_cfg.ao_mode = self.ao_mode
            p_cfg.ao_intensity = float(self.ao_intensity)
            p_cfg.ao_radius = float(self.ao_radius)

            # 4. Reflections & Anti-Aliasing
            p_cfg.ibl_enabled = bool(self.ibl_enabled)
            p_cfg.ssr_enabled = bool(self.ssr_enabled)
            p_cfg.ssr_steps = int(self.ssr_steps)
            p_cfg.ssr_thickness = float(self.ssr_thickness)
            p_cfg.ssr_max_roughness = float(self.ssr_max_roughness)
            p_cfg.ssr_max_distance = float(self.ssr_max_distance)
            p_cfg.ssr_resolution_scale = float(self.ssr_resolution_scale)
            p_cfg.aa_mode = getattr(self, "aa_mode", "OFF")
            p_cfg.taa_enabled = p_cfg.aa_mode == "TAA" or bool(getattr(self, "taa_enabled", False))
            p_cfg.taa_feedback = float(getattr(self, "taa_feedback", 0.95))
            p_cfg.taa_sharpness = float(getattr(self, "taa_sharpness", 0.35))
            p_cfg.taa_gamma = float(getattr(self, "taa_gamma", 1.25))
            p_cfg.taa_max_feedback = float(getattr(self, "taa_max_feedback", 0.95))
            p_cfg.taa_min_feedback = float(getattr(self, "taa_min_feedback", 0.78))
            p_cfg.taa_variance_gamma = float(getattr(self, "taa_variance_gamma", 1.25))
            p_cfg.fxaa_subpixel = float(self.fxaa_subpixel)
            p_cfg.fxaa_edge_threshold = float(self.fxaa_edge_threshold)
            p_cfg.smaa_threshold = float(self.smaa_threshold)
            p_cfg.clustered_lights_enabled = bool(self.point_lights_enabled)
            p_cfg.spot_lights_enabled = bool(self.spot_lights_enabled)
            p_cfg.spot_shadows_enabled = bool(self.spot_shadows_enabled)
            p_cfg.spot_shadow_resolution = int(self.spot_shadow_resolution)
            p_cfg.spot_light_radius = float(self.spot_light_radius)
            p_cfg.spot_cone_angle = float(self.spot_cone_angle)
            p_cfg.spot_shadow_bias = float(self.spot_shadow_bias)

            # 5. Shadows & CSM
            p_cfg.shadow_bias = float(self.shadow_bias)
            p_cfg.shadow_normal_bias = float(self.shadow_normal_bias)
            p_cfg.shadow_distance = float(self.shadow_distance)
            p_cfg.csm_split_lambda = float(self.csm_split_lambda)
            p_cfg.csm_cascade_distances = tuple(float(v) for v in self.csm_cascade_distances)
            p_cfg.csm_cascades = int(self.csm_cascades)
            p_cfg.shadow_resolution = int(self.shadow_resolution)

            if hasattr(pipeline, "csm") and pipeline.csm is not None:
                if self.shadow_resolution != pipeline.csm.atlas_size:
                    pipeline.csm.resize_atlas(self.shadow_resolution)
                if (
                    self.csm_cascades != pipeline.csm.cascade_count
                    or self.csm_split_lambda != pipeline.csm.split_lambda
                    or tuple(self.csm_cascade_distances) != pipeline.csm.cascade_distances
                ):
                    pipeline.csm.update_splits(
                        self.shadow_distance,
                        self.csm_cascades,
                        self.csm_split_lambda,
                        tuple(float(v) for v in self.csm_cascade_distances),
                    )

            if hasattr(pipeline, "spot_shadow_map") and pipeline.spot_shadow_map is not None:
                spot_res = getattr(p_cfg, "spot_shadow_resolution", 2048)
                if spot_res != pipeline.spot_shadow_map.atlas_size:
                    pipeline.spot_shadow_map.resize_atlas(spot_res)

            # 6. Micro-Geometry (POM, SSDM, Tessellation)
            p_cfg.pom_enabled = bool(self.pom_enabled)
            p_cfg.pom_min_samples = int(self.pom_min_samples)
            p_cfg.pom_max_samples = int(self.pom_max_samples)
            p_cfg.pom_height_scale = float(self.pom_height_scale)
            p_cfg.pom_self_shadow = bool(self.pom_self_shadow)
            p_cfg.disp_near_radius = float(self.disp_near_radius)
            p_cfg.disp_mid_radius = float(self.disp_mid_radius)
            p_cfg.tess_enabled = bool(self.tess_enabled)
            p_cfg.tess_max_level = float(self.tess_max_level)
            p_cfg.tess_med_level = float(self.tess_med_level)
            p_cfg.tess_displacement_scale = float(self.tess_displacement_scale)
            p_cfg.frustum_cull_enabled = bool(self.frustum_cull_enabled)
            p_cfg.hiz_cull_enabled = bool(self.hiz_cull_enabled)
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

        # 11. Environment Provider (atmosphere module)
        # The pipeline exposes the provider as `environment`; `atmosphere` is a
        # compatibility alias for the same object. Tweaks write through the
        # provider's public config.
        env = getattr(pipeline, "atmosphere", None) or getattr(pipeline, "environment", None)
        if env is not None:
            atmo = env
            if hasattr(atmo, "config") and atmo.config is not None:
                atmo.config.day_speed = float(self.day_speed)
                atmo.config.azimuth_deg = float(self.azimuth_deg)
                atmo.config.latitude_deg = float(self.latitude_deg)
                atmo.config.declination_deg = float(self.declination_deg)
                atmo.config.moon_orbit_offset_hours = float(self.moon_orbit_offset_hours)
                atmo.config.moon_declination_deg = float(self.moon_declination_deg)
                atmo.config.moon_azimuth_offset_deg = float(self.moon_azimuth_offset_deg)
                atmo.config.sky_exposure = float(self.sky_exposure)
                atmo.config.sky_saturation = float(self.sky_saturation)
                if self.day_speed > 0.0:
                    self.time_of_day = float(atmo.config.time_of_day)
                else:
                    atmo.config.time_of_day = float(self.time_of_day)
                atmo.config.preset_name = str(self.atmo_preset)
                atmo.config.rayleigh_beta = (
                    float(self.rayleigh_r * 1e-6),
                    float(self.rayleigh_g * 1e-6),
                    float(self.rayleigh_b * 1e-6),
                )
                atmo.config.mie_beta = float(self.mie_coeff * 1e-6)
                atmo.config.mie_asymmetry = float(self.mie_asymmetry)
                atmo.config.turbidity = float(self.turbidity)
                atmo.config.star_intensity = float(self.star_intensity)
                atmo.config.star_density = float(self.star_density)
                atmo.config.sun_intensity = float(self.sun_lux)
                atmo.config.sky_mode = str(self.sky_mode)
                atmo.config.cloud_enabled = bool(self.cloud_enabled)
                atmo.config.cloud_coverage = float(self.cloud_coverage)
                atmo.config.cloud_opacity = float(self.cloud_opacity)
                atmo.config.cloud_altitude = float(self.cloud_altitude)
                atmo.config.cloud_scale = float(self.cloud_scale)
                atmo.config.cloud_layer_separation = float(self.cloud_layer_separation)
                atmo.config.cloud_secondary_opacity = float(self.cloud_secondary_opacity)
                atmo.config.cloud_wind = (float(self.cloud_wind_x), float(self.cloud_wind_z))
                atmo.config.space_zenith = tuple(self.space_zenith)
                atmo.config.nebula_color = tuple(self.nebula_color)
                atmo.config.nebula_intensity = float(self.nebula_intensity)
                atmo.config.sun_color = tuple(self.sun_color)
                atmo.config.moon_color = tuple(self.moon_color)
                atmo.config.moon_intensity = float(self.moon_intensity)
                atmo.config.ground_color = tuple(self.ground_color)
                atmo.config.night_zenith = tuple(self.night_zenith)
                atmo.config.night_horizon = tuple(self.night_horizon)
                atmo.config.sun_disc_size = float(self.sun_disc_size)
                atmo.config.moon_disc_size = float(self.moon_disc_size)
                atmo.config.planet_radius = float(self.planet_radius)
                atmo.config.atmosphere_radius = float(self.atmosphere_radius)
                atmo.config.h_r = float(self.h_r)
                atmo.config.h_m = float(self.h_m)
                atmo.config.ozone_center = float(self.ozone_center)
                atmo.config.ozone_width = float(self.ozone_width)

        self._applied_version = self._version

    sync_to_pipeline = apply_to_pipeline

    CREATIVE_OPTICS_PROPERTIES = frozenset(
        {
            "dof_enabled",
            "motion_blur_enabled",
            "lens_flare_enabled",
            "chromatic_aberration_enabled",
            "vignette_enabled",
            "film_grain_enabled",
        }
    )

    def sync_from_config(self, cfg: Any, preserve_creative_optics: bool = False) -> None:
        """Copies graphics properties from a RenderConfig instance."""
        if cfg is None:
            return
        for prop in (
            "tonemap_mode",
            "exposure",
            "bloom_enabled",
            "bloom_intensity",
            "chromatic_aberration_enabled",
            "chromatic_aberration_intensity",
            "vignette_enabled",
            "vignette_intensity",
            "vignette_roundness",
            "vignette_smoothness",
            "film_grain_enabled",
            "film_grain_intensity",
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
            "lens_flare_mode",
            "lens_flare_threshold",
            "lens_flare_streak_intensity",
            "lens_flare_streak_width",
            "lens_flare_ghost_intensity",
            "lens_flare_halo_intensity",
            "lens_flare_tint",
            "lens_flare_resolution_scale",
            "solar_flare_blade_count",
            "solar_flare_bloom_intensity",
            "solar_flare_starburst_intensity",
            "solar_flare_occlusion_fade",
            "gi_mode",
            "ssgi_steps",
            "ssgi_rays",
            "ssgi_thickness",
            "ssgi_ray_distance",
            "ssgi_intensity",
            "ssgi_resolution_scale",
            "rc_ray_count",
            "rc_intensity",
            "rc_interval_c0",
            "rc_interval_c1",
            "rc_interval_c2",
            "rc_steps_c0",
            "rc_steps_c1",
            "rc_thickness",
            "rc_temporal_blend",
            "rc_resolution_scale",
            "ao_mode",
            "ao_intensity",
            "ao_radius",
            "ao_resolution_scale",
            "ibl_enabled",
            "ssr_enabled",
            "ssr_steps",
            "ssr_thickness",
            "ssr_max_roughness",
            "ssr_max_distance",
            "ssr_resolution_scale",
            "fxaa_subpixel",
            "fxaa_edge_threshold",
            "smaa_threshold",
            "spot_shadow_resolution",
            "aa_mode",
            "taa_enabled",
            "taa_feedback",
            "taa_sharpness",
            "taa_gamma",
            "taa_max_feedback",
            "taa_min_feedback",
            "taa_variance_gamma",
            "shadow_bias",
            "shadow_normal_bias",
            "shadow_distance",
            "csm_split_lambda",
            "csm_cascade_distances",
            "csm_cascades",
            "shadow_resolution",
            "pom_enabled",
            "pom_min_samples",
            "pom_max_samples",
            "pom_height_scale",
            "pom_self_shadow",
            "disp_near_radius",
            "disp_mid_radius",
            "tess_enabled",
            "tess_max_level",
            "tess_med_level",
            "tess_displacement_scale",
            "frustum_cull_enabled",
            "hiz_cull_enabled",
            "ssdm_enabled",
            "ssdm_scale",
            "volumetric_fog_enabled",
            "fog_resolution",
            "fog_point_lights",
            "fog_density",
            "fog_height_falloff",
            "fog_anisotropy",
            "fog_distance",
            "fog_ambient",
            "fog_debug_mode",
            "particles_enabled",
            "particle_count",
            "particle_mode",
            "particle_size",
            "particle_turbulence",
        ):
            if preserve_creative_optics and prop in self.CREATIVE_OPTICS_PROPERTIES:
                continue
            if hasattr(cfg, prop):
                setattr(self, prop, getattr(cfg, prop))
        if hasattr(cfg, "clustered_lights_enabled"):
            self.point_lights_enabled = bool(cfg.clustered_lights_enabled)
        if hasattr(cfg, "spot_lights_enabled"):
            self.spot_lights_enabled = bool(cfg.spot_lights_enabled)
        if hasattr(cfg, "spot_shadows_enabled"):
            self.spot_shadows_enabled = bool(cfg.spot_shadows_enabled)
        if hasattr(cfg, "wireframe"):
            self.show_wireframe = bool(cfg.wireframe)
        if hasattr(cfg, "debug_gbuffer"):
            self.gbuffer_debug = GBufferDebugMode(cfg.debug_gbuffer)
        if hasattr(cfg, "sun_intensity"):
            self.sun_lux = float(cfg.sun_intensity)

    def sync_from_pipeline(self, pipeline: Any) -> None:
        """Initializes tweak fields from active pipeline and marks versions as synchronized."""
        if pipeline is None:
            return
        p_cfg = getattr(pipeline, "config", None)
        if p_cfg is not None:
            self.sync_from_config(p_cfg)
        if hasattr(pipeline, "atmosphere") and pipeline.atmosphere is not None:
            atmo = pipeline.atmosphere
            if hasattr(atmo, "config") and atmo.config is not None:
                config = atmo.config
                self.time_of_day = float(getattr(config, "time_of_day", 12.0))
                self.day_speed = float(getattr(config, "day_speed", 0.0))
                self.azimuth_deg = float(getattr(config, "azimuth_deg", 45.0))
                self.latitude_deg = float(getattr(config, "latitude_deg", 35.0))
                self.declination_deg = float(getattr(config, "declination_deg", 0.0))
                self.moon_orbit_offset_hours = float(getattr(config, "moon_orbit_offset_hours", 12.0))
                self.moon_declination_deg = float(getattr(config, "moon_declination_deg", 0.0))
                self.moon_azimuth_offset_deg = float(getattr(config, "moon_azimuth_offset_deg", 0.0))
                self.sky_exposure = float(getattr(config, "sky_exposure", 1.0))
                self.sky_saturation = float(getattr(config, "sky_saturation", 1.0))
                self.atmo_preset = str(getattr(config, "preset_name", "CUSTOM"))
                self.sky_mode = str(getattr(config, "sky_mode", "ATMOSPHERE"))
                beta = getattr(config, "rayleigh_beta", (5.802e-6, 13.558e-6, 33.100e-6))
                self.rayleigh_r, self.rayleigh_g, self.rayleigh_b = (float(v * 1e6) for v in beta)
                self.mie_coeff = float(getattr(config, "mie_beta", 3.996e-6) * 1e6)
                self.mie_asymmetry = float(getattr(config, "mie_asymmetry", 0.80))
                self.turbidity = float(getattr(config, "turbidity", 1.0))
                self.sun_lux = float(getattr(config, "sun_intensity", 4.0))
                for name in (
                    "star_intensity",
                    "star_density",
                    "cloud_enabled",
                    "cloud_coverage",
                    "cloud_opacity",
                    "cloud_altitude",
                    "cloud_scale",
                    "cloud_layer_separation",
                    "cloud_secondary_opacity",
                    "nebula_intensity",
                    "sun_color",
                    "moon_color",
                    "moon_intensity",
                    "ground_color",
                    "night_zenith",
                    "night_horizon",
                    "space_zenith",
                    "nebula_color",
                    "sun_disc_size",
                    "moon_disc_size",
                    "planet_radius",
                    "atmosphere_radius",
                    "h_r",
                    "h_m",
                    "ozone_center",
                    "ozone_width",
                ):
                    if hasattr(config, name):
                        setattr(self, name, getattr(config, name))
                cloud_wind = getattr(config, "cloud_wind", (8.0, 2.0))
                self.cloud_wind_x, self.cloud_wind_z = (float(v) for v in cloud_wind)
        super().__setattr__("_applied_version", self._version)

    def to_dict(self) -> dict[str, Any]:
        """Serializes current graphics configuration into a JSON-compatible dictionary."""
        return {
            "quality_preset": self.quality_preset.value
            if hasattr(self.quality_preset, "value")
            else str(self.quality_preset),
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
            "lens_flare_mode": str(self.lens_flare_mode),
            "lens_flare_threshold": float(self.lens_flare_threshold),
            "lens_flare_streak_intensity": float(self.lens_flare_streak_intensity),
            "lens_flare_streak_width": float(self.lens_flare_streak_width),
            "lens_flare_ghost_intensity": float(self.lens_flare_ghost_intensity),
            "lens_flare_halo_intensity": float(self.lens_flare_halo_intensity),
            "lens_flare_tint": [float(c) for c in self.lens_flare_tint],
            "lens_flare_resolution_scale": float(self.lens_flare_resolution_scale),
            "solar_flare_blade_count": int(self.solar_flare_blade_count),
            "solar_flare_bloom_intensity": float(self.solar_flare_bloom_intensity),
            "solar_flare_starburst_intensity": float(self.solar_flare_starburst_intensity),
            "solar_flare_occlusion_fade": float(self.solar_flare_occlusion_fade),
            "gi_mode": str(self.gi_mode),
            "ssgi_steps": int(self.ssgi_steps),
            "ssgi_rays": int(self.ssgi_rays),
            "ssgi_thickness": float(self.ssgi_thickness),
            "ssgi_ray_distance": float(self.ssgi_ray_distance),
            "ssgi_intensity": float(self.ssgi_intensity),
            "ssgi_resolution_scale": float(self.ssgi_resolution_scale),
            "rc_ray_count": int(self.rc_ray_count),
            "rc_intensity": float(self.rc_intensity),
            "rc_interval_c0": float(self.rc_interval_c0),
            "rc_interval_c1": float(self.rc_interval_c1),
            "rc_interval_c2": float(self.rc_interval_c2),
            "rc_steps_c0": int(self.rc_steps_c0),
            "rc_steps_c1": int(self.rc_steps_c1),
            "rc_thickness": float(self.rc_thickness),
            "rc_temporal_blend": float(self.rc_temporal_blend),
            "rc_resolution_scale": float(self.rc_resolution_scale),
            "ao_mode": str(self.ao_mode),
            "ao_intensity": float(self.ao_intensity),
            "ao_radius": float(self.ao_radius),
            "ao_resolution_scale": float(self.ao_resolution_scale),
            "ibl_enabled": bool(self.ibl_enabled),
            "ssr_enabled": bool(self.ssr_enabled),
            "ssr_steps": int(self.ssr_steps),
            "ssr_thickness": float(self.ssr_thickness),
            "ssr_max_roughness": float(self.ssr_max_roughness),
            "ssr_max_distance": float(self.ssr_max_distance),
            "ssr_resolution_scale": float(self.ssr_resolution_scale),
            "fxaa_subpixel": float(self.fxaa_subpixel),
            "fxaa_edge_threshold": float(self.fxaa_edge_threshold),
            "smaa_threshold": float(self.smaa_threshold),
            "aa_mode": str(getattr(self, "aa_mode", "OFF")),
            "point_lights_enabled": bool(self.point_lights_enabled),
            "spot_lights_enabled": bool(self.spot_lights_enabled),
            "spot_shadows_enabled": bool(self.spot_shadows_enabled),
            "spot_shadow_resolution": int(self.spot_shadow_resolution),
            "spot_light_radius": float(self.spot_light_radius),
            "spot_cone_angle": float(self.spot_cone_angle),
            "spot_shadow_bias": float(self.spot_shadow_bias),
            "shadow_bias": float(self.shadow_bias),
            "shadow_normal_bias": float(self.shadow_normal_bias),
            "shadow_distance": float(self.shadow_distance),
            "csm_split_lambda": float(self.csm_split_lambda),
            "csm_cascade_distances": [float(v) for v in self.csm_cascade_distances],
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
            "azimuth_deg": float(self.azimuth_deg),
            "latitude_deg": float(self.latitude_deg),
            "declination_deg": float(self.declination_deg),
            "moon_orbit_offset_hours": float(self.moon_orbit_offset_hours),
            "moon_declination_deg": float(self.moon_declination_deg),
            "moon_azimuth_offset_deg": float(self.moon_azimuth_offset_deg),
            "sky_exposure": float(self.sky_exposure),
            "sky_saturation": float(self.sky_saturation),
            "atmo_preset": str(self.atmo_preset),
            "rayleigh_r": float(self.rayleigh_r),
            "rayleigh_g": float(self.rayleigh_g),
            "rayleigh_b": float(self.rayleigh_b),
            "mie_coeff": float(self.mie_coeff),
            "mie_asymmetry": float(self.mie_asymmetry),
            "turbidity": float(self.turbidity),
            "star_intensity": float(self.star_intensity),
            "star_density": float(self.star_density),
            "sky_mode": str(self.sky_mode),
            "cloud_enabled": bool(self.cloud_enabled),
            "cloud_coverage": float(self.cloud_coverage),
            "cloud_opacity": float(self.cloud_opacity),
            "cloud_altitude": float(self.cloud_altitude),
            "cloud_scale": float(self.cloud_scale),
            "cloud_layer_separation": float(self.cloud_layer_separation),
            "cloud_secondary_opacity": float(self.cloud_secondary_opacity),
            "cloud_wind_x": float(self.cloud_wind_x),
            "cloud_wind_z": float(self.cloud_wind_z),
            "space_zenith": self.space_zenith,
            "nebula_color": self.nebula_color,
            "nebula_intensity": float(self.nebula_intensity),
            "sun_color": self.sun_color,
            "moon_color": self.moon_color,
            "moon_intensity": float(self.moon_intensity),
            "ground_color": self.ground_color,
            "night_zenith": self.night_zenith,
            "night_horizon": self.night_horizon,
            "sun_disc_size": float(self.sun_disc_size),
            "moon_disc_size": float(self.moon_disc_size),
            "planet_radius": float(self.planet_radius),
            "atmosphere_radius": float(self.atmosphere_radius),
            "h_r": float(self.h_r),
            "h_m": float(self.h_m),
            "ozone_center": float(self.ozone_center),
            "ozone_width": float(self.ozone_width),
            "uncapped_fps": bool(self.uncapped_fps),
            "pom_enabled": bool(self.pom_enabled),
            "pom_min_samples": int(self.pom_min_samples),
            "pom_max_samples": int(self.pom_max_samples),
            "pom_height_scale": float(self.pom_height_scale),
            "pom_self_shadow": bool(self.pom_self_shadow),
            "disp_near_radius": float(self.disp_near_radius),
            "disp_mid_radius": float(self.disp_mid_radius),
            "tess_enabled": bool(self.tess_enabled),
            "tess_max_level": float(self.tess_max_level),
            "tess_med_level": float(self.tess_med_level),
            "tess_displacement_scale": float(self.tess_displacement_scale),
            "frustum_cull_enabled": bool(self.frustum_cull_enabled),
            "hiz_cull_enabled": bool(self.hiz_cull_enabled),
            "ssdm_enabled": bool(self.ssdm_enabled),
            "ssdm_scale": float(self.ssdm_scale),
            "particles_enabled": bool(self.particles_enabled),
            "show_bounds": bool(self.show_bounds),
            "show_physics_colliders": bool(self.show_physics_colliders),
            "taa_enabled": bool(self.taa_enabled),
            "taa_feedback": float(self.taa_feedback),
            "taa_sharpness": float(self.taa_sharpness),
            "taa_max_feedback": float(getattr(self, "taa_max_feedback", 0.95)),
            "taa_min_feedback": float(getattr(self, "taa_min_feedback", 0.78)),
            "taa_variance_gamma": float(getattr(self, "taa_variance_gamma", 1.25)),
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
                elif isinstance(current, tuple):
                    setattr(self, key, tuple(val))
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
        except (OSError, TypeError, ValueError) as exc:
            log_error("DebugMenu", f"Failed to save graphics configuration to {filepath}: {exc}")
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
