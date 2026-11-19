"""AAA-Style Graphics Quality Presets and Tunable Render Settings for PyMordial Engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class GraphicsQuality(str, Enum):
    CUSTOM = "custom"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CINEMATIC = "cinematic"


class GIMode(str, Enum):
    OFF = "OFF"
    IBL = "IBL"
    SSGI = "SSGI"
    RADIANCE_CASCADES = "RADIANCE_CASCADES"


FROXEL_RESOLUTIONS: dict[str, tuple[int, int, int]] = {
    "LOW": (80, 45, 32),
    "MEDIUM": (120, 68, 48),
    "HIGH": (160, 90, 64),
    "ULTRA": (200, 112, 80),
    "CINEMATIC": (240, 135, 96),
}


@dataclass
class RenderConfig:
    """Configurable graphics and shader parameters for AAA quality and player scaling."""

    # Shadow Settings (hard-shadow cascaded shadow maps)
    shadow_resolution: int = 8192  # CSM Atlas size (1024, 2048, 4096, 8192)
    shadow_bias: float = 0.0015  # Base world-space depth bias (meters)
    shadow_normal_bias: float = 0.0010  # Base normal offset bias scale
    # Slope-scaled bias clamp: bias = clamp(base * (1 + slope * scale), base * min, base * max)
    shadow_bias_slope_scale: float = 2.0
    shadow_bias_min_scale: float = 0.5
    shadow_bias_max_scale: float = 6.0
    csm_cascades: int = 4  # Number of shadow cascades (1 to 4)
    shadow_distance: float = 500.0  # Far plane of the last cascade (meters)
    csm_split_lambda: float = 0.85  # Practical Split Scheme (PSSM) log-linear blend factor
    # Explicit far distance of each cascade in meters. When set, these override
    # the PSSM distribution entirely, so each cascade's reach is authored
    # directly. ``None`` falls back to PSSM splits derived from shadow_distance.
    csm_cascade_distances: tuple[float, float, float, float] | None = None

    # Shadow filtering tiers. Each tier is limited to a prefix of the cascades;
    # the far cascades keep the cheaper path because their texels already span
    # several screen pixels.
    shadow_filter_mode: str = "PCF"  # "HARD", "PCF" or "PCSS"
    pcf_cascade_limit: int = 3  # cascades with index < this use PCF
    pcf_samples: int = 16  # Vogel disk taps (4 to 64)
    pcf_radius: float = 1.5  # filter radius in cascade texels
    pcf_falloff: float = 2.0  # radial weight exponent (1 = cone, 2 = bell)

    # PCSS (percentage-closer soft shadows). The penumbra is derived from the
    # receiver-to-blocker gap, so contact shadows stay sharp and only elevated
    # casters soften.
    pcss_cascade_limit: int = 1  # cascades with index < this use PCSS
    pcss_search_samples: int = 16  # blocker search taps (4 to 64)
    pcss_search_radius: float = 48.0  # blocker search radius in cascade texels
    pcss_light_size: float = 0.10  # light-source size factor (penumbra scale)
    pcss_samples: int = 24  # penumbra filter taps (4 to 64)
    pcss_falloff: float = 2.0  # radial weight exponent
    pcss_max_radius: float = 96.0  # penumbra radius clamp in cascade texels
    # 0.0 = average blocker gap (classic PCSS), 1.0 = closest blocker gap.
    # The closest blocker keeps overlapping shadows sharp instead of letting a
    # far caster's softness bleed onto a near caster's edge.
    pcss_closest_blocker_weight: float = 1.0

    # Depth Precision & GPU Culling
    reverse_z: bool = True  # Reversed-Z 32F floating-point depth buffer
    hiz_cull_enabled: bool = False  # GPU-Driven Hi-Z Occlusion Culling
    frustum_cull_enabled: bool = True  # CPU vectorized frustum culling of draw batches
    hiz_debug_mip: int = 0  # Hi-Z debug visualization mip level (0 = off)
    max_entities: int = 100_000  # GPU entity capacity for SSBOs and culling scratch

    # Ambient Occlusion (GTAO / SSAO)
    ao_mode: str = "GTAO"  # "OFF", "SSAO", "GTAO"
    ao_intensity: float = 1.0
    ao_radius: float = 0.35
    ao_resolution_scale: float = 1.0  # 1.0 = full res; 0.5 = half res (cheaper, softer)
    # GTAO backend: "LEGACY" (horizon march) or "HIZ_BOUNDS" (bounds-rejected).
    ao_backend: str = "LEGACY"
    ao_debug_view: int = 0  # 0=final, 1=mip, 2=occupancy, 3=rejected, 4=refined

    # Global Illumination (SSGI vs Radiance Cascades)
    gi_mode: str = "RADIANCE_CASCADES"  # "OFF", "IBL", "SSGI", "RADIANCE_CASCADES"
    ssgi_steps: int = 16
    ssgi_rays: int = 8
    ssgi_ray_distance: float = 3.0
    ssgi_thickness: float = 0.35
    ssgi_intensity: float = 1.5
    ssgi_resolution_scale: float = 1.0  # 1.0 = full res; 0.5 = half res (cheaper, softer)
    # Temporal accumulation. A single-frame SSGI estimate is inherently noisy;
    # accumulating across frames is what converges it and keeps it stable while
    # the camera moves.
    ssgi_temporal_enabled: bool = True
    ssgi_temporal_blend: float = 0.90  # history weight (0 = off, 0.95 = heavy)
    # Tracing backend: "LEGACY" (fixed-step marcher) or "HIZ_BOUNDS" (shared
    # hierarchical tracer over the RG32F linear-depth bounds hierarchy).
    ssgi_backend: str = "LEGACY"
    ssgi_hiz_max_iterations: int = 64  # iteration budget for the shared tracer
    ssgi_debug_view: int = 0  # 0=off, 1=raw, 2=hit UV, 3=hit dist, 4=hit ratio, 5=iters, 6=miss, 7=self, 8=surface
    ssgi_surface_reject: bool = False  # reject hits on the receiver's own surface ID

    # Hybrid Radiance Cascades (SSRC + FFPC)
    rc_cascades: int = 3
    rc_ray_count: int = 16
    rc_interval_c0: float = 0.40
    rc_interval_c1: float = 2.50
    rc_interval_c2: float = 40.0
    rc_steps_c0: int = 8
    rc_steps_c1: int = 12
    rc_thickness: float = 0.15
    rc_intensity: float = 1.0
    rc_temporal_blend: float = 0.96
    rc_resolution_scale: float = 1.0  # 1.0 = full res; 0.5 = half res (cheaper, softer)
    # Backend: "LEGACY" (per-pixel raymarch) or "CASCADES" (explicit probe cascades).
    rc_backend: str = "LEGACY"
    rc_hiz_max_iterations: int = 64  # iteration budget for cascade interval tracing
    rc_temporal_enabled: bool = True
    rc_debug_view: int = 0  # reconstruction: 0=irradiance, 1=irradiance, 2=confidence, 3=rejected
    rc_temporal_debug_view: int = 0  # 0=resolved, 1=current, 2=depth conf, 3=motion conf, 4=history weight

    # Image-Based Lighting & Screen-Space Reflections (SSR)
    ibl_enabled: bool = True
    ssr_enabled: bool = True
    ssr_resolution_scale: float = 1.0  # 1.0 = full res; 0.5 = half res (cheaper, softer)
    ssr_mode: str = "HIZ"  # "HIZ" (Hi-Z accelerated) or "LINEAR" (legacy march)
    # Tracing backend: "LEGACY" (in-shader march) or "HIZ_BOUNDS" (shared
    # hierarchical tracer over the RG32F linear-depth bounds hierarchy).
    ssr_backend: str = "LEGACY"
    ssr_hiz_max_iterations: int = 96  # iteration budget for the shared tracer
    ssr_debug_view: int = 0  # 0=off, 1=hit UV, 2=distance, 3=iterations, 4=miss reason, 5=final mip
    ssr_steps: int = 48
    ssr_max_roughness: float = 0.65
    ssr_thickness: float = 0.12
    ssr_max_distance: float = 30.0

    # Anti-Aliasing (Mutually Exclusive: "OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X", "TAA")
    aa_mode: str = "OFF"
    taa_enabled: bool = False
    taa_feedback: float = 0.82
    taa_sharpness: float = 0.35
    taa_gamma: float = 1.25
    taa_max_feedback: float = 0.95
    taa_min_feedback: float = 0.78
    taa_variance_gamma: float = 1.25
    fxaa_subpixel: float = 0.75
    fxaa_edge_threshold: float = 0.125
    smaa_threshold: float = 0.08

    # Clustered Local Lighting & Spot Lights
    clustered_lights_enabled: bool = True
    max_point_lights: int = 256
    max_spot_lights: int = 64
    spot_lights_enabled: bool = True
    spot_shadows_enabled: bool = True
    spot_shadow_resolution: int = 2048  # Atlas size (1024 or 2048)
    spot_light_radius: float = 35.0
    spot_cone_angle: float = 32.0  # Outer cone angle in degrees
    spot_shadow_bias: float = 0.0015
    sun_intensity: float = 4.0  # Sun lux intensity
    ambient_factor: float = 0.04

    # Post-Processing & Atmosphere
    bloom_enabled: bool = True
    bloom_iterations: int = 5  # Dual-filter pyramid mip passes (3 to 6)
    bloom_threshold: float = 1.0
    bloom_intensity: float = 0.12
    tonemap_mode: str = "ACES"  # "ACES" (fitted RRT+ODT), "AgX", "Neutral", "Reinhard"
    exposure: float = 1.0

    # ---- Graphics Upgrade: colour pipeline (Pass 1) ----
    # EV stops applied on top of `exposure`. Positive brightens.
    exposure_compensation: float = 0.0
    # White balance in Kelvin; 6500 K is a no-op. Disabled by default so the
    # existing look is unchanged until a project opts in.
    white_balance_enabled: bool = False
    white_balance_k: float = 6500.0
    # ASC-CDL style grading applied in display-referred space after the curve.
    contrast: float = 1.0
    saturation: float = 1.0
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ---- Graphics Upgrade: auto-exposure (Pass 2) ----
    # When enabled, a GPU log-luminance histogram drives the exposure and
    # `exposure` becomes a bias on top of the adapted value.
    auto_exposure_enabled: bool = False
    exposure_key: float = 0.5  # Target percentile of the luminance histogram
    exposure_min_ev: float = -8.0  # Clamp on the adapted EV100
    exposure_max_ev: float = 12.0
    exposure_adapt_speed_up: float = 1.5  # EV/s when the scene brightens
    exposure_adapt_speed_down: float = 4.0  # EV/s when the scene darkens
    exposure_min_log_luminance: float = -12.0  # Histogram bin 0 lower bound
    exposure_max_log_luminance: float = 16.0  # Histogram bin 255 upper bound

    # ---- Graphics Upgrade: energy-conserving BRDF (Pass 3) ----
    # Restores the energy single-scatter GGX drops between microfacets, so rough
    # metals stop going dark. Requires the BRDF LUT's blue channel (directional
    # albedo), which is generated automatically.
    energy_compensation_enabled: bool = True
    # Folds unresolved normal-map variance into roughness to stop specular
    # aliasing on high-frequency detail.
    specular_aa_enabled: bool = True
    specular_aa_strength: float = 1.0

    # ---- Graphics Upgrade: per-object motion vectors (Pass 4) ----
    # Reprojects with the previous frame's model matrix so moving objects report
    # their own velocity. Without this TAA smears anything that moves.
    object_motion_vectors_enabled: bool = True

    volumetric_fog_enabled: bool = True
    fog_resolution: str = "HIGH"  # "LOW", "MEDIUM", "HIGH", "ULTRA", "CINEMATIC"
    fog_point_lights: bool = True  # Volumetric local point light scattering
    fog_density: float = 0.005
    fog_height_falloff: float = 0.10
    fog_anisotropy: float = 0.65
    fog_distance: float = 400.0
    fog_ambient: float = 0.35
    fog_debug_mode: int = 0  # 0=Normal, 1=Inscattering Only, 2=Transmittance Only

    # Debug Visualization
    debug_gbuffer: int = 0  # 0=Disabled, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas, 6=ShadowMask, 14=CascadeIndex, 15=CascadeCoverage, 17..22=HiZ bounds
    wireframe: bool = False  # ModernGL polygon line rasterization mode

    # Phase 6: Parallax Occlusion Mapping (POM)
    pom_enabled: bool = True
    pom_min_samples: int = 8  # Min raymarch steps (glancing)
    pom_max_samples: int = 64  # Max raymarch steps (head-on)
    pom_height_scale: float = 1.0  # Multiplier for per-material POM depth
    pom_self_shadow: bool = True  # Enable self-shadowing from sun direction

    # Phase 6: Screen-Space Displacement Mapping (SSDM)
    ssdm_enabled: bool = True
    ssdm_scale: float = 1.0  # Multiplier for per-material SSDM depth
    ssdm_max_distance: float = 30.0  # Camera distance fade-out

    # Phase 6: Camera Radius-Based Micro-Geometry & Hardware Tessellation
    disp_near_radius: float = 120.0  # Near radius for highest quality tessellation & POM self-shadowing
    disp_mid_radius: float = 300.0  # Medium radius boundary & POM/SSDM cutoff
    tess_enabled: bool = True
    tess_max_level: float = 24.0  # Max hardware tessellation subdivision level (near radius)
    tess_med_level: float = 8.0  # Medium quality tessellation level (mid radius)
    tess_displacement_scale: float = 1.0  # Multiplier for per-material tessellation displacement depth

    # Phase 2: Optional legacy camera-centered particle field; local effects use app.particles emitters.
    particles_enabled: bool = False
    particle_count: int = 0  # Legacy camera-centered field; local emitters allocate on demand
    particle_mode: str = "OFF"  # "DUST_MOTES", "EMBERS", "FIREFLIES", "OFF"
    particle_size: float = 1.0  # Multiplier for particle radius
    particle_turbulence: float = 0.85  # Curl noise turbulence strength

    # Phase 3: Cinematic Camera Optics & Lens Effects
    # Bokeh Depth of Field
    dof_enabled: bool = False
    dof_focus_distance: float = 5.0  # Focus plane distance in meters
    dof_focal_length: float = 50.0  # Lens focal length in mm
    dof_aperture: float = 2.8  # Aperture f-stop (e.g. 1.4, 2.8, 5.6)
    dof_bokeh_shape: str = "CIRCULAR"  # "CIRCULAR", "HEXAGONAL", "ANAMORPHIC"
    dof_anamorphic_ratio: float = 1.0  # Horizontal anamorphic squeeze ratio (1.0 to 2.0)
    dof_max_coc: float = 24.0  # Max blur radius in pixels

    # Velocity Motion Blur
    motion_blur_enabled: bool = False
    motion_blur_samples: int = 12  # Directional blur taps (4 to 32)
    motion_blur_intensity: float = 1.5  # Shutter angle / blur strength scale
    motion_blur_max_radius: float = 48.0  # Max blur vector in pixels
    motion_blur_tile_size: int = 16  # Tile footprint (px) for the tile-max reduction
    motion_blur_max_samples: int = 32  # Upper bound on gather taps per pixel

    # Anamorphic Lens Flare & Ghost Reflections
    lens_flare_enabled: bool = True
    lens_flare_mode: str = "ANAMORPHIC"  # "OFF", "ANAMORPHIC", "SOLAR"
    lens_flare_threshold: float = 1.8  # Luminance cutoff for specular streaks
    lens_flare_streak_intensity: float = 0.6
    lens_flare_streak_width: float = 32.0  # Horizontal spread in pixels
    lens_flare_ghost_intensity: float = 0.0
    lens_flare_halo_intensity: float = 0.25
    lens_flare_tint: tuple[float, float, float] = (0.25, 0.65, 1.0)  # Anamorphic blue/cyan
    lens_flare_resolution_scale: float = 1.0  # 1.0 = full res; 0.5 = half res (cheaper, softer)

    # Solar Flare (lens_flare_mode == "SOLAR")
    solar_flare_blade_count: int = 6  # Aperture blades -> diffraction spike count
    solar_flare_bloom_intensity: float = 1.0  # Sun disc bloom weight
    solar_flare_starburst_intensity: float = 0.6  # Diffraction spike weight
    solar_flare_occlusion_fade: float = 1.0  # 0 = ignore geometry, 1 = full depth occlusion

    # Optical Lens Imperfections
    chromatic_aberration_enabled: bool = True
    chromatic_aberration_intensity: float = 0.005  # Radial spectral separation
    vignette_enabled: bool = True
    vignette_intensity: float = 0.35
    vignette_roundness: float = 0.85
    vignette_smoothness: float = 0.50
    film_grain_enabled: bool = False
    film_grain_intensity: float = 0.03


QUALITY_PRESETS: dict[GraphicsQuality, RenderConfig] = {
    GraphicsQuality.CUSTOM: RenderConfig(),
    GraphicsQuality.LOW: RenderConfig(
        shadow_resolution=1024,
        shadow_bias=0.0020,
        csm_cascades=2,
        shadow_distance=120.0,
        csm_cascade_distances=(12.0, 45.0, 80.0, 120.0),
        shadow_filter_mode="HARD",
        ao_mode="SSAO",
        gi_mode="OFF",
        rc_ray_count=8,
        ibl_enabled=True,
        ssr_enabled=False,
        aa_mode="FXAA",
        taa_enabled=False,
        clustered_lights_enabled=False,
        bloom_enabled=False,
        bloom_iterations=3,
        volumetric_fog_enabled=False,
        fog_resolution="LOW",
        fog_point_lights=False,
        fog_density=0.002,
        pom_enabled=False,
        pom_min_samples=4,
        pom_max_samples=16,
        pom_self_shadow=False,
        ssdm_enabled=False,
        tess_enabled=False,
        particles_enabled=False,
        particle_count=0,
        particle_mode="OFF",
        dof_enabled=False,
        motion_blur_enabled=False,
        lens_flare_enabled=False,
        chromatic_aberration_enabled=False,
        vignette_enabled=True,
        vignette_intensity=0.20,
        film_grain_enabled=False,
        film_grain_intensity=0.03,
    ),
    GraphicsQuality.MEDIUM: RenderConfig(
        shadow_resolution=2048,
        shadow_bias=0.0018,
        csm_cascades=3,
        shadow_distance=250.0,
        csm_cascade_distances=(14.0, 45.0, 110.0, 250.0),
        shadow_filter_mode="PCF",
        pcf_samples=8,
        pcf_radius=1.0,
        pcf_falloff=2.0,
        pcss_cascade_limit=0,
        ao_mode="GTAO",
        gi_mode="SSGI",
        ssgi_steps=8,
        ssgi_resolution_scale=1.0,
        rc_ray_count=12,
        rc_resolution_scale=1.0,
        ibl_enabled=True,
        ssr_enabled=False,
        ssr_resolution_scale=1.0,
        aa_mode="TAA",
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=4,
        bloom_intensity=0.07,
        volumetric_fog_enabled=True,
        fog_resolution="LOW",
        fog_point_lights=False,
        fog_density=0.004,
        pom_enabled=True,
        pom_min_samples=8,
        pom_max_samples=32,
        pom_self_shadow=False,
        ssdm_enabled=False,
        tess_enabled=False,
        particles_enabled=False,
        particle_count=0,
        particle_mode="OFF",
        dof_enabled=False,
        motion_blur_enabled=False,
        motion_blur_samples=6,
        lens_flare_enabled=False,
        chromatic_aberration_enabled=True,
        chromatic_aberration_intensity=0.003,
        vignette_enabled=True,
        film_grain_enabled=False,
        film_grain_intensity=0.03,
    ),
    GraphicsQuality.HIGH: RenderConfig(
        shadow_resolution=8192,
        shadow_bias=0.0015,
        csm_cascades=4,
        shadow_distance=500.0,
        csm_cascade_distances=(20.0, 60.0, 160.0, 500.0),
        shadow_filter_mode="PCSS",
        pcf_samples=16,
        pcf_radius=1.5,
        pcf_falloff=2.0,
        pcss_cascade_limit=1,
        pcss_search_samples=16,
        pcss_search_radius=48.0,
        pcss_light_size=0.10,
        pcss_samples=24,
        pcss_falloff=2.0,
        pcss_max_radius=96.0,
        ao_mode="GTAO",
        gi_mode="RADIANCE_CASCADES",
        rc_ray_count=16,
        rc_resolution_scale=1.0,
        ssgi_steps=12,
        ssgi_resolution_scale=1.0,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_resolution_scale=1.0,
        ssr_steps=64,
        ssr_thickness=0.10,
        ssr_max_distance=30.0,
        aa_mode="TAA",
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=5,
        bloom_intensity=0.10,
        volumetric_fog_enabled=True,
        fog_resolution="HIGH",
        fog_point_lights=True,
        fog_density=0.005,
        pom_enabled=True,
        pom_min_samples=8,
        pom_max_samples=64,
        pom_self_shadow=True,
        ssdm_enabled=True,
        ssdm_scale=0.04,
        tess_enabled=True,
        tess_max_level=16.0,
        tess_displacement_scale=0.10,
        particles_enabled=False,
        particle_count=0,
        particle_mode="OFF",
        dof_enabled=False,
        dof_bokeh_shape="CIRCULAR",
        motion_blur_enabled=False,
        motion_blur_samples=12,
        lens_flare_enabled=True,
        chromatic_aberration_enabled=True,
        chromatic_aberration_intensity=0.005,
        vignette_enabled=True,
        film_grain_enabled=False,
        film_grain_intensity=0.03,
    ),
    GraphicsQuality.ULTRA: RenderConfig(
        shadow_resolution=8192,
        shadow_bias=0.0015,
        csm_cascades=4,
        shadow_distance=800.0,
        csm_cascade_distances=(25.0, 80.0, 250.0, 800.0),
        shadow_filter_mode="PCSS",
        pcf_samples=24,
        pcf_radius=1.75,
        pcf_falloff=2.0,
        pcss_cascade_limit=1,
        pcss_search_samples=24,
        pcss_search_radius=64.0,
        pcss_light_size=0.12,
        pcss_samples=32,
        pcss_falloff=2.0,
        pcss_max_radius=128.0,
        ao_mode="GTAO",
        gi_mode="RADIANCE_CASCADES",
        rc_ray_count=24,
        rc_resolution_scale=1.0,
        ssgi_steps=16,
        ssgi_resolution_scale=1.0,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_resolution_scale=1.0,
        ssr_steps=80,
        ssr_thickness=0.08,
        ssr_max_distance=40.0,
        aa_mode="TAA",
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
        bloom_intensity=0.12,
        volumetric_fog_enabled=True,
        fog_resolution="ULTRA",
        fog_point_lights=True,
        fog_density=0.006,
        pom_enabled=True,
        pom_min_samples=12,
        pom_max_samples=80,
        pom_self_shadow=True,
        ssdm_enabled=True,
        ssdm_scale=0.05,
        tess_enabled=True,
        tess_max_level=24.0,
        tess_displacement_scale=0.10,
        particles_enabled=False,
        particle_count=0,
        particle_mode="OFF",
        dof_enabled=False,
        dof_bokeh_shape="HEXAGONAL",
        motion_blur_enabled=False,
        motion_blur_samples=16,
        lens_flare_enabled=True,
        chromatic_aberration_enabled=True,
        chromatic_aberration_intensity=0.007,
        vignette_enabled=True,
        film_grain_enabled=False,
        film_grain_intensity=0.03,
    ),
    GraphicsQuality.CINEMATIC: RenderConfig(
        shadow_resolution=8192,
        shadow_bias=0.0012,
        csm_cascades=4,
        shadow_distance=1200.0,
        csm_cascade_distances=(30.0, 110.0, 380.0, 1200.0),
        shadow_filter_mode="PCSS",
        pcf_samples=32,
        pcf_radius=2.0,
        pcf_falloff=2.0,
        pcss_cascade_limit=1,
        pcss_search_samples=32,
        pcss_search_radius=80.0,
        pcss_light_size=0.15,
        pcss_samples=40,
        pcss_falloff=2.0,
        pcss_max_radius=160.0,
        ao_mode="GTAO",
        gi_mode="RADIANCE_CASCADES",
        rc_ray_count=32,
        rc_resolution_scale=1.0,
        ssgi_steps=24,
        ssgi_resolution_scale=1.0,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_resolution_scale=1.0,
        ssr_steps=96,
        ssr_thickness=0.06,
        ssr_max_distance=50.0,
        aa_mode="TAA",
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
        bloom_intensity=0.14,
        volumetric_fog_enabled=True,
        fog_resolution="CINEMATIC",
        fog_point_lights=True,
        fog_density=0.008,
        pom_enabled=True,
        pom_min_samples=16,
        pom_max_samples=128,
        pom_self_shadow=True,
        ssdm_enabled=True,
        ssdm_scale=0.06,
        tess_enabled=True,
        tess_max_level=32.0,
        tess_displacement_scale=0.12,
        particles_enabled=False,
        particle_count=0,
        particle_mode="OFF",
        dof_enabled=False,
        dof_bokeh_shape="ANAMORPHIC",
        dof_anamorphic_ratio=2.0,
        motion_blur_enabled=False,
        motion_blur_samples=24,
        lens_flare_enabled=True,
        chromatic_aberration_enabled=True,
        chromatic_aberration_intensity=0.008,
        vignette_enabled=True,
        film_grain_enabled=False,
        film_grain_intensity=0.03,
    ),
}


def get_quality_preset(quality: GraphicsQuality | str) -> RenderConfig:
    """Returns an independent copy of the requested render quality preset."""
    if isinstance(quality, str):
        quality = GraphicsQuality(quality.lower())
    cfg = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[GraphicsQuality.HIGH])
    return replace(cfg)
