"""AAA-Style Graphics Quality Presets and Tunable Render Settings for PyMordial Engine."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class GraphicsQuality(str, Enum):
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


@dataclass
class RenderConfig:
    """Configurable graphics and shader parameters for AAA quality and player scaling."""

    # Shadow Settings
    shadow_resolution: int = 4096  # CSM Atlas size (1024, 2048, 4096)
    csm_cascades: int = 4          # Number of shadow cascades (1 to 4)
    pcf_samples: int = 16          # Poisson PCF taps (4, 8, 16)
    shadow_distance: float = 150.0 # Maximum shadow distance in meters

    # Screen-Space Contact Shadows (SSCS)
    sscs_enabled: bool = True
    sscs_steps: int = 12           # Depth raymarching steps (<10m)
    sscs_max_distance: float = 10.0
    sscs_thickness: float = 0.05

    # Depth Precision
    reverse_z: bool = True         # Reversed-Z 32F floating-point depth buffer

    # Ambient Occlusion (GTAO / SSAO)
    ao_mode: str = "GTAO"          # "OFF", "SSAO", "GTAO"
    ao_intensity: float = 1.0
    ao_radius: float = 0.75

    # Global Illumination (SSGI & LPV)
    gi_mode: str = "HYBRID"        # "OFF", "SSGI", "LPV", "HYBRID"
    ssgi_steps: int = 12
    ssgi_ray_distance: float = 2.5
    ssgi_intensity: float = 1.2
    lpv_intensity: float = 1.0

    # Image-Based Lighting & Screen-Space Reflections (SSR)
    ibl_enabled: bool = True
    ssr_enabled: bool = True
    ssr_steps: int = 24
    ssr_max_roughness: float = 0.65

    # Temporal Anti-Aliasing (TAA)
    taa_enabled: bool = True

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
    fog_density: float = 0.015
    fog_height_falloff: float = 0.10

    # Debug Visualization
    debug_gbuffer: int = 0         # 0=Disabled, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas
    wireframe: bool = False        # ModernGL polygon line rasterization mode


QUALITY_PRESETS: dict[GraphicsQuality, RenderConfig] = {
    GraphicsQuality.LOW: RenderConfig(
        shadow_resolution=1024,
        csm_cascades=2,
        pcf_samples=4,
        shadow_distance=60.0,
        sscs_enabled=False,
        sscs_steps=4,
        ao_mode="OFF",
        gi_mode="OFF",
        ibl_enabled=False,
        ssr_enabled=False,
        taa_enabled=False,
        clustered_lights_enabled=False,
        bloom_enabled=False,
        bloom_iterations=3,
        volumetric_fog_enabled=False,
        fog_density=0.005,
    ),
    GraphicsQuality.MEDIUM: RenderConfig(
        shadow_resolution=2048,
        csm_cascades=3,
        pcf_samples=8,
        shadow_distance=100.0,
        sscs_enabled=True,
        sscs_steps=8,
        ao_mode="SSAO",
        gi_mode="SSGI",
        ssgi_steps=8,
        ibl_enabled=True,
        ssr_enabled=False,
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=4,
        volumetric_fog_enabled=True,
        fog_density=0.010,
    ),
    GraphicsQuality.HIGH: RenderConfig(
        shadow_resolution=2048,
        csm_cascades=4,
        pcf_samples=16,
        shadow_distance=150.0,
        sscs_enabled=True,
        sscs_steps=12,
        ao_mode="GTAO",
        gi_mode="SSGI",
        ssgi_steps=12,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=20,
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=5,
        volumetric_fog_enabled=True,
        fog_density=0.015,
    ),
    GraphicsQuality.ULTRA: RenderConfig(
        shadow_resolution=4096,
        csm_cascades=4,
        pcf_samples=16,
        shadow_distance=200.0,
        sscs_enabled=True,
        sscs_steps=16,
        ao_mode="GTAO",
        gi_mode="HYBRID",
        ssgi_steps=16,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=28,
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
        volumetric_fog_enabled=True,
        fog_density=0.018,
    ),
    GraphicsQuality.CINEMATIC: RenderConfig(
        shadow_resolution=4096,
        csm_cascades=4,
        pcf_samples=16,
        shadow_distance=300.0,
        sscs_enabled=True,
        sscs_steps=24,
        ao_mode="GTAO",
        gi_mode="HYBRID",
        ssgi_steps=24,
        ibl_enabled=True,
        ssr_enabled=True,
        ssr_steps=40,
        taa_enabled=True,
        clustered_lights_enabled=True,
        bloom_enabled=True,
        bloom_iterations=6,
        bloom_intensity=0.06,
        volumetric_fog_enabled=True,
        fog_density=0.022,
    ),
}


def get_quality_preset(quality: GraphicsQuality | str) -> RenderConfig:
    """Returns a copy of the requested render quality preset."""
    if isinstance(quality, str):
        quality = GraphicsQuality(quality.lower())
    cfg = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[GraphicsQuality.HIGH])
    return RenderConfig(**cfg.__dict__)
