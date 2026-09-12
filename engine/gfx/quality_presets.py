"""AAA-Style Graphics Quality Presets and Tunable Render Settings for PyMordial Engine."""

from __future__ import annotations
from dataclasses import dataclass
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
    SSGI = "SSGI"
    LPV = "LPV"
    HYBRID = "HYBRID"


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

    # Shadow Settings
    shadow_resolution: int = 4096  # CSM Atlas size (1024, 2048, 4096)
    shadow_mode: str = "PCSS"      # "HARD", "PCF", "PCSS"
    shadow_softness: float = 1.2   # Penumbra scale & PCF radius (0.2 to 3.0)
    shadow_bias: float = 0.0015    # Base depth bias
    shadow_normal_bias: float = 0.0010  # Base normal offset bias scale
    shadow_offset_x: float = 0.0   # Light-space alignment offset X (delta from calibrated base)
    shadow_offset_y: float = 0.0   # Light-space alignment offset Y (delta from calibrated base)
    csm_cascades: int = 4          # Number of shadow cascades (1 to 4)
    pcf_samples: int = 24          # Vogel PCF taps (8 to 32)
    shadow_distance: float = 500.0 # Maximum shadow distance in meters (improved from 150.0m)
    csm_split_lambda: float = 0.85 # Practical Split Scheme (PSSM) log-linear blend factor

    # Screen-Space Contact Shadows (Deprecated/Removed in favor of GTAO)
    sscs_enabled: bool = False
    sscs_steps: int = 16           # Depth raymarching steps (4 to 32)
    sscs_ray_distance: float = 0.20 # Metric trace distance (0.05m to 1.5m)
    sscs_max_distance: float = 50.0# Camera distance fade out (10.0m to 100.0m)
    sscs_thickness: float = 0.10   # Physical metric thickness in meters

    # Depth Precision
    reverse_z: bool = True         # Reversed-Z 32F floating-point depth buffer

    # Ambient Occlusion (GTAO / HBAO / SSAO)
    ao_mode: str = "GTAO"          # "OFF", "SSAO", "HBAO", "GTAO"
    ao_intensity: float = 1.0
    ao_radius: float = 0.75

    # Global Illumination (SSGI & LPV)
    gi_mode: str = "HYBRID"        # "OFF", "SSGI", "LPV", "HYBRID"
    ssgi_steps: int = 16
    ssgi_rays: int = 8
    ssgi_ray_distance: float = 3.0
    ssgi_thickness: float = 0.35
    ssgi_intensity: float = 1.5
    lpv_intensity: float = 1.0

    # Image-Based Lighting & Screen-Space Reflections (SSR)
    ibl_enabled: bool = True
    ssr_enabled: bool = True
    ssr_steps: int = 32
    ssr_max_roughness: float = 0.65
    ssr_thickness: float = 0.40
    ssr_max_distance: float = 20.0

    # Anti-Aliasing (Mutually Exclusive: "OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X")
    aa_mode: str = "OFF"
    taa_enabled: bool = False
    taa_feedback: float = 0.92
    taa_sharpness: float = 0.35
    fxaa_subpixel: float = 0.75
    fxaa_edge_threshold: float = 0.125
    smaa_threshold: float = 0.08

    # Clustered Local Lighting
    clustered_lights_enabled: bool = True
    max_point_lights: int = 256
    sun_intensity: float = 4.0     # Sun lux intensity
    ambient_factor: float = 0.04

    # Post-Processing & Atmosphere
    bloom_enabled: bool = True
    bloom_iterations: int = 5      # Dual-filter pyramid mip passes (3 to 6)
    bloom_threshold: float = 1.0
    bloom_intensity: float = 0.045
    tonemap_mode: str = "ACES"     # "ACES", "AgX", "Reinhard"
    exposure: float = 1.0

    volumetric_fog_enabled: bool = True
    fog_resolution: str = "HIGH"    # "LOW", "MEDIUM", "HIGH", "ULTRA", "CINEMATIC"
    fog_point_lights: bool = True   # Volumetric local point light scattering
    fog_density: float = 0.005
    fog_height_falloff: float = 0.10
    fog_anisotropy: float = 0.65
    fog_distance: float = 400.0
    fog_ambient: float = 0.35
    fog_debug_mode: int = 0         # 0=Normal, 1=Inscattering Only, 2=Transmittance Only

    # Debug Visualization
    debug_gbuffer: int = 0         # 0=Disabled, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas
    wireframe: bool = False        # ModernGL polygon line rasterization mode

    # Phase 6: Parallax Occlusion Mapping (POM)
    pom_enabled: bool = True
    pom_min_samples: int = 8       # Min raymarch steps (glancing)
    pom_max_samples: int = 64      # Max raymarch steps (head-on)
    pom_height_scale: float = 1.0  # Multiplier for per-material POM depth
    pom_self_shadow: bool = True   # Enable self-shadowing from sun direction

    # Phase 6: Screen-Space Displacement Mapping (SSDM)
    ssdm_enabled: bool = True
    ssdm_scale: float = 1.0        # Multiplier for per-material SSDM depth
    ssdm_max_distance: float = 30.0  # Camera distance fade-out

    # Phase 6: Camera Radius-Based Micro-Geometry & Hardware Tessellation
    disp_near_radius: float = 8.0         # Near radius for highest quality tessellation
    disp_mid_radius: float = 25.0         # Medium radius boundary & POM/SSDM cutoff
    tess_enabled: bool = True
    tess_max_level: float = 24.0          # Max hardware tessellation subdivision level (near radius)
    tess_med_level: float = 8.0           # Medium quality tessellation level (mid radius)
    tess_displacement_scale: float = 1.0  # Multiplier for per-material tessellation displacement depth
    tess_distance_min: float = 2.0        # Deprecated: alias for near distance
    tess_distance_max: float = 30.0       # Deprecated: alias for far distance


QUALITY_PRESETS: dict[GraphicsQuality, RenderConfig] = {
    GraphicsQuality.CUSTOM: RenderConfig(),
    GraphicsQuality.LOW: RenderConfig(
        shadow_resolution=1024,
        shadow_mode="HARD",
        shadow_softness=0.8,
        shadow_bias=0.0020,
        csm_cascades=2,
        pcf_samples=4,
        shadow_distance=120.0,
        sscs_enabled=False,
        sscs_steps=4,
        ao_mode="OFF",
        gi_mode="OFF",
        ibl_enabled=False,
        ssr_enabled=False,
        aa_mode="OFF",
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
    ),
    GraphicsQuality.MEDIUM: RenderConfig(
        shadow_resolution=2048,
        shadow_mode="PCF",
        shadow_softness=1.0,
        shadow_bias=0.0018,
        csm_cascades=3,
        pcf_samples=16,
        shadow_distance=250.0,
        sscs_enabled=False,
        sscs_steps=8,
        ao_mode="SSAO",
        gi_mode="SSGI",
        ssgi_steps=8,
        ibl_enabled=True,
        ssr_enabled=False,
        aa_mode="OFF",
        taa_enabled=False,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=4,
        volumetric_fog_enabled=True,
        fog_resolution="MEDIUM",
        fog_point_lights=True,
        fog_density=0.004,
        pom_enabled=True,
        pom_min_samples=8,
        pom_max_samples=32,
        pom_self_shadow=False,
        ssdm_enabled=False,
        tess_enabled=False,
    ),
    GraphicsQuality.HIGH: RenderConfig(
        shadow_resolution=4096,
        shadow_mode="PCSS",
        shadow_softness=1.2,
        shadow_bias=0.0015,
        csm_cascades=4,
        pcf_samples=24,
        shadow_distance=500.0,
        sscs_enabled=False,
        sscs_steps=12,
        ao_mode="GTAO",
        gi_mode="SSGI",
        ssgi_steps=12,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=20,
        aa_mode="OFF",
        taa_enabled=False,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=5,
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
    ),
    GraphicsQuality.ULTRA: RenderConfig(
        shadow_resolution=4096,
        shadow_mode="PCSS",
        shadow_softness=1.2,
        shadow_bias=0.0015,
        csm_cascades=4,
        pcf_samples=32,
        shadow_distance=800.0,
        sscs_enabled=False,
        sscs_steps=16,
        ao_mode="GTAO",
        gi_mode="HYBRID",
        ssgi_steps=16,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=28,
        aa_mode="OFF",
        taa_enabled=False,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
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
    ),
    GraphicsQuality.CINEMATIC: RenderConfig(
        shadow_resolution=4096,
        shadow_mode="PCSS",
        shadow_softness=1.4,
        shadow_bias=0.0012,
        csm_cascades=4,
        pcf_samples=32,
        shadow_distance=1200.0,
        sscs_enabled=False,
        sscs_steps=24,
        ao_mode="GTAO",
        gi_mode="HYBRID",
        ssgi_steps=24,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=40,
        aa_mode="OFF",
        taa_enabled=False,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
        bloom_intensity=0.06,
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
    ),
}


def get_quality_preset(quality: GraphicsQuality | str) -> RenderConfig:
    """Returns a copy of the requested render quality preset."""
    if isinstance(quality, str):
        quality = GraphicsQuality(quality.lower())
    cfg = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[GraphicsQuality.HIGH])
    return RenderConfig(**cfg.__dict__)
