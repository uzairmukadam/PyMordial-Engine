"""Unified Master Render Graph Coordinator for PyMordial Engine.

Coordinates the full ModernGL 4.5 MDI geometry pass, Reversed-Z depth, Cascaded Shadow Maps,
Image-Based Lighting (IBL), GTAO, SSGI, LPV, Clustered Local Lighting, SSR, TAA, and Tone-Mapping.
"""

from __future__ import annotations
from pathlib import Path
import math
from typing import Callable
import numpy as np
import moderngl

from engine.core.ecs import EntityManager
from engine.core.math_utils import matrix_perspective, matrix_look_at, mat4_mul, mat4_inv
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.frame_context import FrameContext
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.g_buffer import GBuffer
from engine.gfx.shadow_csm import CascadedShadowMap
from engine.gfx.post_process import PostProcessPipeline
from engine.gfx.render_graph import RenderGraphContext
from engine.gfx.passes.ibl import IBLPass
from engine.gfx.passes.ao_pass import AmbientOcclusionPass
from engine.gfx.passes.ssgi_pass import SSGIPass
from engine.gfx.passes.lpv_pass import LPVPass
from engine.gfx.passes.clustered_lights import ClusteredLightingPass, PointLight
from engine.gfx.passes.ssr_pass import SSRPass
from engine.gfx.passes.taa_pass import TAAPass
from engine.gfx.passes.fxaa_pass import FXAAPass
from engine.gfx.passes.smaa_pass import SMAAPass
from engine.gfx.passes.volumetric_fog_pass import VolumetricFogPass
from engine.gfx.texture_atlas import TextureArrayAtlas
from engine.gfx.passes.ssdm_pass import SSDMPass
from engine.gfx.passes.sky_atmosphere_pass import SkyAtmospherePass
from engine.gfx.passes.particle_pass import ParticleSystemPass
from engine.gfx.passes.camera_optics_pass import CameraOpticsPass
from engine.gfx.passes.water_pass import WaterPass
from engine.gfx.passes.hiz_pass import HiZPass
from engine.gfx.atmosphere import AtmosphereSystem
from engine.assets.resource_cache import ResourceCache
from engine.debug.debug_draw import DebugDraw
from engine.events import subscribe_event, unsubscribe_event, WindowResizeEvent

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


SHADER_DIR = Path(__file__).resolve().parent.parent.parent / "shaders"
ROOT_DIR = SHADER_DIR.parent

_SHADER_FILE_CACHE: dict[Path, str] = {}


def _read_shader_file(path: Path) -> str:
    """Reads a GLSL shader file with memory caching to avoid redundant disk I/O."""
    if path not in _SHADER_FILE_CACHE:
        _SHADER_FILE_CACHE[path] = path.read_text(encoding="utf-8")
    return _SHADER_FILE_CACHE[path]


def _load_shader(rel_path: str | Path, visited: set[Path] | None = None) -> str:
    """Loads a GLSL shader file and recursively expands #include directives."""
    if visited is None:
        visited = set()
    if isinstance(rel_path, str):
        path = ROOT_DIR / rel_path if (ROOT_DIR / rel_path).is_file() else SHADER_DIR / rel_path
    else:
        path = rel_path
    path = path.resolve()
    if path in visited:
        return ""
    visited.add(path)
    raw = _read_shader_file(path)
    lines: list[str] = []
    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("#include"):
            start_q = line.find('"')
            end_q = line.rfind('"')
            if start_q != -1 and end_q > start_q:
                inc_rel = line[start_q + 1 : end_q]
                inc_path = ROOT_DIR / inc_rel if (ROOT_DIR / inc_rel).is_file() else SHADER_DIR / inc_rel
                inc_content = _load_shader(inc_path, visited)
                lines.append(inc_content)
            else:
                lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


class RenderPipeline:
    """Coordinates the full ModernGL 4.5 MDI, Reversed-Z, CSM, and deferred resolve pipeline."""

    __slots__ = (
        "ctx_wrapper",
        "ctx",
        "config",
        "frame_context",
        "mega_buffer",
        "mdi",
        "g_buffer",
        "csm",
        "ibl_pass",
        "ao_pass",
        "ssgi_pass",
        "lpv_pass",
        "lights_pass",
        "ssr_pass",
        "taa_pass",
        "fxaa_pass",
        "smaa_pass",
        "volumetric_fog_pass",
        "_prev_vp_mat",
        "post_process",
        "resources",
        "debug",
        "ssbo_transforms",
        "ssbo_materials",
        "gbuffer_prog",
        "csm_prog",
        "resolve_prog",
        "resolve_vao",
        "gbuffer_vao",
        "csm_vao",
        "texture_atlas",
        "ssdm_pass",
        "hiz_pass",
        "tess_mdi",
        "gbuffer_tess_prog",
        "gbuffer_tess_vao",
        "csm_tess_prog",
        "csm_tess_vao",
        "_view_mat",
        "_proj_mat",
        "_vp_mat",
        "_inv_proj",
        "_inv_view",
        "_cam_pos",
        "_cam_fwd",
        "_sun_v",
        "_screen_size_tuple",
        "_zero_jitter",
        "_mie_beta_tuple",
        "_prev_mie_beta",
        "_atmo_sun_tuple",
        "_cube_alloc",
        "_sphere_alloc",
        "_capsule_alloc",
        "_csm_cascade_idx_uniform",
        "_u_pcf_samples",
        "_u_shadow_mode",
        "_u_shadow_softness",
        "_u_shadow_bias",
        "_u_shadow_normal_bias",
        "_u_shadow_offset",
        "_u_cascade_count",
        "_u_gbuffer_debug",
        "_u_hiz_debug_mip",
        "_u_gbuffer_prev_vp",
        "_u_ao_enabled",
        "_u_gi_enabled",
        "_u_ibl_enabled",
        "_u_ssr_enabled",
        "_u_point_light_count",
        "_u_lpv_min",
        "_u_lpv_size",
        "_u_gbuffer_pom_enabled",
        "_u_gbuffer_pom_min_samples",
        "_u_gbuffer_pom_max_samples",
        "_u_gbuffer_pom_height_scale",
        "_u_gbuffer_pom_scale_mult",
        "_u_gbuffer_pom_self_shadow",
        "_u_gbuffer_disp_near_radius",
        "_u_gbuffer_disp_mid_radius",
        "_u_tess_enabled",
        "_u_tess_frustum_cull",
        "_u_tess_near_radius",
        "_u_tess_mid_radius",
        "_u_tess_max_level",
        "_u_tess_med_level",
        "_u_tess_disp_scale",
        "_u_tess_pom_scale_mult",
        "_u_tess_prev_vp",
        "_u_csm_tess_cascade_idx",
        "_u_csm_tess_near_radius",
        "_u_csm_tess_mid_radius",
        "_u_csm_tess_max_level",
        "_u_csm_tess_med_level",
        "_u_csm_tess_disp_scale",
        "atmosphere",
        "sky_atmosphere_pass",
        "_u_rayleigh_beta",
        "_u_mie_beta",
        "_u_mie_g",
        "_u_ozone_beta",
        "_u_atmo_ground_col",
        "_u_night_zenith",
        "_u_night_horizon",
        "_u_sun_disc_size",
        "_u_moon_disc_size",
        "_u_moon_color",
        "_u_star_intensity",
        "_u_star_density",
        "_u_atmo_turbidity",
        "_u_atmo_sun_dir",
        "particle_pass",
        "camera_optics_pass",
        "water_pass",
        "_prev_time",
        "_graph_context",
    )

    def __init__(
        self,
        ctx_wrapper: RenderContext,
        config: RenderConfig | None = None,
        resources: ResourceCache | None = None,
    ) -> None:
        self.ctx_wrapper = ctx_wrapper
        self.ctx = ctx_wrapper.ctx
        self.config = config if config is not None else ctx_wrapper.config
        self.resources = resources if resources is not None else ResourceCache()
        self.debug = DebugDraw(self.ctx)
        self._prev_time = 0.0

        w, h = self.ctx_wrapper.width, self.ctx_wrapper.height

        # 1. Initialize UBO 0 (std140 binding 0)
        self.frame_context = FrameContext(self.ctx)

        # 2. Initialize Shared Mega-Buffer & MDI Batcher
        self.mega_buffer = MegaBuffer(self.ctx)
        self.mdi = MultiDrawIndirect(self.ctx)

        # 3. Initialize MRT G-Buffer (Reversed-Z 32F)
        self.g_buffer = GBuffer(
            self.ctx,
            w,
            h,
            reverse_z=self.config.reverse_z,
        )

        # 4. Initialize Cascaded Shadow Maps (CSM)
        self.csm = CascadedShadowMap(
            self.ctx,
            atlas_size=self.config.shadow_resolution,
            cascade_count=self.config.csm_cascades,
            max_distance=self.config.shadow_distance,
        )

        # 5. Initialize SSBO 1 (Transforms) & SSBO 2 (Materials)
        max_entities = 100_000
        self.ssbo_transforms = self.ctx.buffer(reserve=max_entities * 16 * 4)
        self.ssbo_transforms.bind_to_storage_buffer(binding=1)

        self.ssbo_materials = self.ctx.buffer(reserve=max_entities * 8 * 4)
        self.ssbo_materials.bind_to_storage_buffer(binding=2)

        # 6. Initialize Phase 5 High-End Graphics Passes
        self.ibl_pass = IBLPass(self.ctx)
        self.ao_pass = AmbientOcclusionPass(self.ctx, w, h)
        self.ssgi_pass = SSGIPass(self.ctx, w, h)
        self.lpv_pass = LPVPass(self.ctx)
        self.lights_pass = ClusteredLightingPass(self.ctx, max_lights=self.config.max_point_lights)
        self.ssr_pass = SSRPass(self.ctx, w, h)
        self.taa_pass = TAAPass(self.ctx, w, h)
        self.fxaa_pass = FXAAPass(self.ctx, w, h)
        self.smaa_pass = SMAAPass(self.ctx, w, h)
        self.volumetric_fog_pass = VolumetricFogPass(self.ctx, w, h)
        self._prev_vp_mat = np.identity(4, dtype=np.float32).flatten()

        # Pre-allocated scratch buffers for zero-allocation frame rendering
        self._view_mat = np.zeros(16, dtype=np.float32)
        self._proj_mat = np.zeros(16, dtype=np.float32)
        self._vp_mat = np.zeros(16, dtype=np.float32)
        self._inv_proj = np.zeros(16, dtype=np.float32)
        self._inv_view = np.zeros(16, dtype=np.float32)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_fwd = np.zeros(3, dtype=np.float32)
        self._sun_v = np.zeros(3, dtype=np.float32)
        self._screen_size_tuple = (float(w), float(h))
        self._zero_jitter = (0.0, 0.0)
        self._mie_beta_tuple = (0.0, 0.0, 0.0)
        self._prev_mie_beta = -1.0
        self._atmo_sun_tuple = (0.0, 0.0, 0.0)

        # 7. Compile Core Programs
        quad_vert = _load_shader("fullscreen_quad.vert")
        gbuffer_vert = _load_shader("gbuffer.vert")
        gbuffer_frag = _load_shader("gbuffer.frag")
        csm_vert = _load_shader("csm_depth.vert")
        csm_frag = _load_shader("csm_depth.frag")
        resolve_frag = _load_shader("deferred_resolve.frag")
        post_frag = _load_shader("post_process.frag")

        self.gbuffer_prog = self.ctx.program(
            vertex_shader=gbuffer_vert,
            fragment_shader=gbuffer_frag,
        )
        self.gbuffer_vao = self.mega_buffer.get_vao(self.gbuffer_prog)

        self.csm_prog = self.ctx.program(
            vertex_shader=csm_vert,
            fragment_shader=csm_frag,
        )
        self.csm_vao = self.mega_buffer.get_vao(self.csm_prog)

        # Hardware Tessellation Programs & VAOs (mode=PATCHES)
        gbuffer_tess_vert = _load_shader("gbuffer_tess.vert")
        gbuffer_tess_tesc = _load_shader("gbuffer_tess.tesc")
        gbuffer_tess_tese = _load_shader("gbuffer_tess.tese")
        self.gbuffer_tess_prog = self.ctx.program(
            vertex_shader=gbuffer_tess_vert,
            tess_control_shader=gbuffer_tess_tesc,
            tess_evaluation_shader=gbuffer_tess_tese,
            fragment_shader=gbuffer_frag,
        )
        self.gbuffer_tess_vao = self.mega_buffer.get_vao(self.gbuffer_tess_prog, mode=self.ctx.PATCHES)

        csm_tess_vert = _load_shader("csm_tess.vert")
        csm_tess_tesc = _load_shader("csm_tess.tesc")
        csm_tess_tese = _load_shader("csm_tess.tese")
        self.csm_tess_prog = self.ctx.program(
            vertex_shader=csm_tess_vert,
            tess_control_shader=csm_tess_tesc,
            tess_evaluation_shader=csm_tess_tese,
            fragment_shader=csm_frag,
        )
        self.csm_tess_vao = self.mega_buffer.get_vao(self.csm_tess_prog, mode=self.ctx.PATCHES)
        self.tess_mdi = MultiDrawIndirect(self.ctx)

        # Texture Array Atlas for PBR Materials
        self.texture_atlas = TextureArrayAtlas(self.ctx, width=2048, height=2048, max_layers=32)

        # SSDM Post-G-Buffer Pass
        self.ssdm_pass = SSDMPass(self.ctx, w, h)

        # Hierarchical-Z (Hi-Z) Depth Pyramid Pass
        self.hiz_pass = HiZPass(self.ctx, w, h, reverse_z=self.config.reverse_z)

        self.resolve_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=resolve_frag,
        )
        self.resolve_vao = self.ctx.vertex_array(self.resolve_prog, [])

        # Assign texture unit bindings for resolve pass
        tex_uniforms = {
            "u_GBufferAlbedoRoughness": 0,
            "u_GBufferNormalMetallic": 1,
            "u_GBufferDepth": 2,
            "u_ShadowAtlas": 3,
            "u_AOTexture": 4,
            "u_SSGITexture": 5,
            "u_BRDFLUT": 6,
            "u_EnvironmentMap": 7,
            "u_SSRTexture": 8,
            "u_LPVVolume": 9,
            "u_SkyViewLUT": 10,
            "u_TransmittanceLUT": 11,
            "u_HiZTexture": 14,
        }
        for name, unit in tex_uniforms.items():
            if name in self.resolve_prog:
                self.resolve_prog[name].value = unit

        # Pre-cache mesh allocations and uniforms
        self._cube_alloc = self.mega_buffer.allocations["cube"]
        self._sphere_alloc = self.mega_buffer.allocations["sphere"]
        self._capsule_alloc = self.mega_buffer.allocations["capsule"]
        self._csm_cascade_idx_uniform = self.csm_prog.get("u_CascadeIndex", None)

        self._u_pcf_samples = self.resolve_prog.get("u_PCF_Samples", None)
        self._u_shadow_mode = self.resolve_prog.get("u_ShadowMode", None)
        self._u_shadow_softness = self.resolve_prog.get("u_ShadowSoftness", None)
        self._u_shadow_bias = self.resolve_prog.get("u_ShadowBias", None)
        self._u_shadow_normal_bias = self.resolve_prog.get("u_ShadowNormalBias", None)
        self._u_shadow_offset = self.resolve_prog.get("u_ShadowOffset", None)
        self._u_cascade_count = self.resolve_prog.get("u_CascadeCount", None)
        self._u_gbuffer_debug = self.resolve_prog.get("u_GBufferDebug", None)
        self._u_hiz_debug_mip = self.resolve_prog.get("u_HiZDebugMip", None)
        self._u_gbuffer_prev_vp = self.gbuffer_prog.get("u_PrevViewProjection", None)

        self._u_ao_enabled = self.resolve_prog.get("u_AOEnabled", None)
        self._u_gi_enabled = self.resolve_prog.get("u_GIEnabled", None)
        self._u_ibl_enabled = self.resolve_prog.get("u_IBLEnabled", None)
        self._u_ssr_enabled = self.resolve_prog.get("u_SSREnabled", None)
        self._u_point_light_count = self.resolve_prog.get("u_PointLightCount", None)
        self._u_lpv_min = self.resolve_prog.get("u_LPV_Min", None)
        self._u_lpv_size = self.resolve_prog.get("u_LPV_Size", None)

        # Pre-cache POM and micro-geometry uniform handles
        self._u_gbuffer_pom_enabled = self.gbuffer_prog.get("u_POMEnabled", None)
        self._u_gbuffer_pom_min_samples = self.gbuffer_prog.get("u_POMMinSamples", None)
        self._u_gbuffer_pom_max_samples = self.gbuffer_prog.get("u_POMMaxSamples", None)
        self._u_gbuffer_pom_height_scale = self.gbuffer_prog.get("u_POMHeightScale", None)
        self._u_gbuffer_pom_scale_mult = self.gbuffer_prog.get("u_POMScaleMultiplier", None)
        self._u_gbuffer_pom_self_shadow = self.gbuffer_prog.get("u_POMSelfShadow", None)
        self._u_gbuffer_disp_near_radius = self.gbuffer_prog.get("u_DispNearRadius", None)
        self._u_gbuffer_disp_mid_radius = self.gbuffer_prog.get("u_DispMidRadius", None)

        self._u_tess_enabled = self.gbuffer_tess_prog.get("u_TessEnabled", None)
        self._u_tess_frustum_cull = self.gbuffer_tess_prog.get("u_FrustumCullEnabled", None)
        self._u_tess_near_radius = self.gbuffer_tess_prog.get("u_DispNearRadius", None)
        self._u_tess_mid_radius = self.gbuffer_tess_prog.get("u_DispMidRadius", None)
        self._u_tess_max_level = self.gbuffer_tess_prog.get("u_TessMaxLevel", None)
        self._u_tess_med_level = self.gbuffer_tess_prog.get("u_TessMedLevel", None)
        self._u_tess_disp_scale = self.gbuffer_tess_prog.get("u_TessDisplacementScale", None)
        self._u_tess_pom_scale_mult = self.gbuffer_tess_prog.get("u_POMScaleMultiplier", None)
        self._u_tess_prev_vp = self.gbuffer_tess_prog.get("u_PrevViewProjection", None)

        self._u_csm_tess_cascade_idx = self.csm_tess_prog.get("u_CascadeIndex", None)
        self._u_csm_tess_near_radius = self.csm_tess_prog.get("u_DispNearRadius", None)
        self._u_csm_tess_mid_radius = self.csm_tess_prog.get("u_DispMidRadius", None)
        self._u_csm_tess_max_level = self.csm_tess_prog.get("u_TessMaxLevel", None)
        self._u_csm_tess_med_level = self.csm_tess_prog.get("u_TessMedLevel", None)
        self._u_csm_tess_disp_scale = self.csm_tess_prog.get("u_TessDisplacementScale", None)

        # 8. Physical Atmosphere & Day-Night Simulation (Sébastien Hillaire Eurographics 2020)
        self.atmosphere = AtmosphereSystem()
        self.sky_atmosphere_pass = SkyAtmospherePass(self.ctx)
        self._u_rayleigh_beta = self.resolve_prog.get("u_RayleighBeta", None)
        self._u_mie_beta = self.resolve_prog.get("u_MieBeta", None)
        self._u_mie_g = self.resolve_prog.get("u_MieG", None)
        self._u_ozone_beta = self.resolve_prog.get("u_OzoneBeta", None)
        self._u_atmo_ground_col = self.resolve_prog.get("u_AtmosphereGroundColor", None)
        self._u_night_zenith = self.resolve_prog.get("u_NightZenithColor", None)
        self._u_night_horizon = self.resolve_prog.get("u_NightHorizonColor", None)
        self._u_sun_disc_size = self.resolve_prog.get("u_SunDiscSize", None)
        self._u_moon_disc_size = self.resolve_prog.get("u_MoonDiscSize", None)
        self._u_moon_color = self.resolve_prog.get("u_MoonColor", None)
        self._u_star_intensity = self.resolve_prog.get("u_StarIntensity", None)
        self._u_star_density = self.resolve_prog.get("u_StarDensity", None)
        self._u_atmo_turbidity = self.resolve_prog.get("u_AtmosphereTurbidity", None)
        self._u_atmo_sun_dir = self.resolve_prog.get("u_AtmosphereSunDir", None)

        # 8.5. GPU Compute Particle System & Volumetric VFX (Phase 2)
        part_enabled = getattr(self.config, "particles_enabled", True)
        part_count = getattr(self.config, "particle_count", 16384)
        part_mode = getattr(self.config, "particle_mode", "DUST_MOTES")
        self.particle_pass = ParticleSystemPass(
            ctx=self.ctx,
            max_particles=65536,
            active_count=part_count if part_enabled else 0,
            mode=part_mode if part_enabled else "OFF",
        )

        # 8.6. Cinematic Camera Optics & Lens Effects (Phase 3)
        self.camera_optics_pass = CameraOpticsPass(self.ctx, w, h)

        # 8.7. Dynamic Water Simulation & Screen-Space Refraction (Phase 4)
        self.water_pass = WaterPass(self.ctx, w, h)

        # 9. Post-Processing & Tonemapping Pipeline
        self.post_process = PostProcessPipeline(
            self.ctx,
            w,
            h,
            self.config,
            post_frag,
            quad_vert,
        )

        # Reusable RenderGraphContext container
        self._graph_context = RenderGraphContext(
            ctx=self.ctx,
            width=w,
            height=h,
            config=self.config,
            frame_context=self.frame_context,
        )

        # Automatically synchronize framebuffers with window resolution changes
        subscribe_event(WindowResizeEvent, self._on_window_resize, priority=90)

    def add_point_light(
        self,
        position: tuple[float, float, float],
        radius: float = 10.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 2.0,
    ) -> PointLight:
        """Adds a dynamic local point light to the active frame."""
        return self.lights_pass.add_light(position, radius, color, intensity)

    def clear_point_lights(self) -> None:
        """Clears all active dynamic point lights."""
        self.lights_pass.clear()

    def apply_config(self, new_config: RenderConfig) -> None:
        """Applies dynamic graphics quality configuration changes."""
        self.config = new_config
        self.csm.resize_atlas(new_config.shadow_resolution)
        self.csm.update_splits(
            new_config.shadow_distance,
            new_config.csm_cascades,
            getattr(new_config, "csm_split_lambda", 0.85),
        )
        self.post_process.config = new_config
        self.ctx_wrapper.config = new_config
        self.ctx_wrapper.ctx.depth_func = ">" if new_config.reverse_z else "<"
        self._graph_context.config = new_config

        part_enabled = getattr(new_config, "particles_enabled", True)
        part_count = getattr(new_config, "particle_count", 16384)
        part_mode = getattr(new_config, "particle_mode", "DUST_MOTES")
        if self.particle_pass is not None:
            self.particle_pass.set_mode(part_mode if part_enabled else "OFF")
            self.particle_pass.set_active_count(part_count if part_enabled else 0)
            self.particle_pass.base_size_multiplier = getattr(new_config, "particle_size", 1.0)
            self.particle_pass.turbulence_strength = getattr(new_config, "particle_turbulence", 0.85)

        if self.water_pass is not None:
            self.water_pass.enabled = getattr(new_config, "water_enabled", True)

    def resize(self, width: int, height: int) -> None:
        """Resizes MRT G-Buffer and all post-processing/intermediate pass buffers."""
        self.ctx_wrapper.width = width
        self.ctx_wrapper.height = height
        self._graph_context.width = width
        self._graph_context.height = height
        self._screen_size_tuple = (float(width), float(height))

        self.g_buffer.resize(width, height)
        self.ssdm_pass.resize(width, height)
        self.hiz_pass.resize(width, height)
        self.ao_pass.resize(width, height)
        self.ssgi_pass.resize(width, height)
        self.ssr_pass.resize(width, height)
        self.taa_pass.resize(width, height)
        self.fxaa_pass.resize(width, height)
        self.smaa_pass.resize(width, height)
        self.volumetric_fog_pass.resize(width, height)
        self.camera_optics_pass.resize(width, height)
        self.water_pass.resize(width, height)
        self.post_process.resize(width, height)

        if gl is not None:
            try:
                while gl.glGetError() != 0:
                    pass
            except Exception:
                pass

    def _on_window_resize(self, event: WindowResizeEvent) -> None:
        self.resize(event.width, event.height)

    def load_cooked_mesh(self, name: str, vpath: str) -> MeshAllocation:
        """Loads a cooked .pm_mesh from VFS and registers it into the MegaBuffer."""
        cached = self.resources.get_gpu_mesh(name)
        if cached is not None:
            return cached
        mesh = self.resources.load_mesh(vpath)
        alloc = self.mega_buffer.add_pm_mesh(name, mesh)
        self.resources.register_gpu_mesh(name, alloc)
        self.csm_vao = self.mega_buffer.get_vao(self.csm_prog)
        self.gbuffer_vao = self.mega_buffer.get_vao(self.gbuffer_prog)
        self.gbuffer_tess_vao = self.mega_buffer.get_vao(self.gbuffer_tess_prog, mode=self.ctx.PATCHES)
        self.csm_tess_vao = self.mega_buffer.get_vao(self.csm_tess_prog, mode=self.ctx.PATCHES)
        return alloc

    def load_cooked_texture(self, vpath: str) -> moderngl.Texture:
        return self.resources.load_gpu_texture(vpath, self.ctx)

    def load_materials(
        self,
        textures_dir: str | Path = "assets/textures",
        resolution: int = 4096,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Loads PBR material textures from folder into the texture array atlas."""
        if self.texture_atlas.width != resolution or self.texture_atlas.height != resolution:
            self.texture_atlas.destroy()
            self.texture_atlas = TextureArrayAtlas(self.ctx, width=resolution, height=resolution, max_layers=32)
        return self.texture_atlas.load_materials_from_folder(textures_dir, progress_callback=progress_callback)

    def render_frame(
        self,
        ecs: EntityManager,
        camera_pos: tuple[float, float, float] | np.ndarray = (0.0, 5.0, 10.0),
        camera_target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        time_elapsed: float = 0.0,
        sun_dir: tuple[float, float, float] = (0.35, -0.85, 0.40),
        sun_lux: float = 4.0,
        fovy_deg: float = 60.0,
        draw_batches: list[tuple] | None = None,
        debug_draw: DebugDraw | None = None,
        dt: float | None = None,
    ) -> None:
        """Executes full multi-pass high-fidelity deferred rendering pipeline."""
        active_count = ecs.active_count
        if active_count <= 0:
            return

        w, h = self.ctx_wrapper.width, self.ctx_wrapper.height
        aspect = w / (h if h > 0 else 1)
        if w != self.hiz_pass.width or h != self.hiz_pass.height:
            self.hiz_pass.resize(w, h)

        # 1. Update SSBO 1 & SSBO 2 from ECS contiguous memory tables
        self.ssbo_transforms.write(ecs.get_active_transforms_view())
        self.ssbo_materials.write(ecs.get_active_materials_view())

        # 2. Camera Matrices & Subpixel TAA Jitter
        self._cam_pos[0] = float(camera_pos[0])
        self._cam_pos[1] = float(camera_pos[1])
        self._cam_pos[2] = float(camera_pos[2])

        matrix_look_at(camera_pos, camera_target, up=(0.0, 1.0, 0.0), out=self._view_mat)

        # Synchronize CSM splits with active shadow distance and cascade count
        if (
            self.csm.max_distance != self.config.shadow_distance
            or self.csm.cascade_count != self.config.csm_cascades
        ):
            self.csm.update_splits(
                self.config.shadow_distance,
                self.config.csm_cascades,
                getattr(self.config, "csm_split_lambda", 0.85),
            )

        matrix_perspective(
            math.radians(fovy_deg),
            aspect,
            near=0.1,
            far=max(2000.0, self.config.shadow_distance * 3.0),
            reverse_z=self.config.reverse_z,
            out=self._proj_mat,
        )

        jitter_x, jitter_y = 0.0, 0.0
        aa_mode = getattr(self.config, "aa_mode", "OFF")
        if aa_mode in ("SMAA_2X", "SMAA_4X"):
            jitter_x, jitter_y = self.smaa_pass.get_jitter(w, h, aa_mode)

        if jitter_x != 0.0 or jitter_y != 0.0:
            self._proj_mat[8] += jitter_x * 2.0
            self._proj_mat[9] += jitter_y * 2.0

        mat4_mul(self._proj_mat, self._view_mat, out=self._vp_mat)
        mat4_inv(self._proj_mat, out=self._inv_proj)
        mat4_inv(self._view_mat, out=self._inv_view)

        fx = float(camera_target[0] - camera_pos[0])
        fy = float(camera_target[1] - camera_pos[1])
        fz = float(camera_target[2] - camera_pos[2])
        inv_fwd_len = 1.0 / (math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-6)
        self._cam_fwd[0] = fx * inv_fwd_len
        self._cam_fwd[1] = fy * inv_fwd_len
        self._cam_fwd[2] = fz * inv_fwd_len

        # 3. Advance atmosphere day-night simulation and synchronise lighting
        if dt is None:
            calc_dt = max(0.0, min(0.1, time_elapsed - self._prev_time)) if self._prev_time > 0.0 else 0.016
        else:
            calc_dt = dt
        self._prev_time = time_elapsed

        if calc_dt > 0.0:
            self.atmosphere.update(calc_dt)

        atmo_sun_dir = self.atmosphere.compute_sun_vector()
        atmo_dir, atmo_lux, atmo_col, atmo_ambient = self.atmosphere.compute_lighting()
        is_custom_sun = (sun_dir != (0.35, -0.85, 0.40) and self.atmosphere.config.day_speed == 0.0)
        true_sun_dir = sun_dir if is_custom_sun else atmo_sun_dir
        active_sun_dir = sun_dir if is_custom_sun else atmo_dir
        active_sun_lux = sun_lux if (sun_lux != 4.0 and is_custom_sun) else atmo_lux
        active_sun_col = atmo_col
        active_ambient = atmo_ambient

        self._sun_v[0] = float(active_sun_dir[0])
        self._sun_v[1] = float(active_sun_dir[1])
        self._sun_v[2] = float(active_sun_dir[2])
        csm_matrices = self.csm.compute_cascade_matrices(self._cam_pos, self._cam_fwd, self._sun_v)

        # 4. Upload to UBO 0
        self.frame_context.update(
            view_mat=self._view_mat,
            proj_mat=self._proj_mat,
            view_proj_mat=self._vp_mat,
            inv_proj_mat=self._inv_proj,
            inv_view_mat=self._inv_view,
            camera_pos=self._cam_pos,
            time_elapsed=time_elapsed,
            screen_size=self._screen_size_tuple,
            jitter=(jitter_x, jitter_y) if (jitter_x != 0.0 or jitter_y != 0.0) else self._zero_jitter,
            sun_dir=active_sun_dir,
            sun_lux=active_sun_lux,
            sun_color=active_sun_col,
            ambient_factor=active_ambient,
            csm_matrices=csm_matrices,
            cascade_splits=self.csm.split_distances,
            fog_density=self.config.fog_density,
            fog_height_falloff=self.config.fog_height_falloff,
        )

        # Prepare MDI batch commands (standard and tessellated)
        self.mdi.begin_frame()
        self.tess_mdi.begin_frame()
        if draw_batches is not None:
            for item in draw_batches:
                if len(item) == 4:
                    mesh_item, count, base_inst, is_tess = item
                else:
                    mesh_item, count, base_inst = item
                    is_tess = False

                if count > 0:
                    alloc = (
                        self.mega_buffer.allocations[mesh_item]
                        if isinstance(mesh_item, str)
                        else mesh_item
                    )
                    if is_tess:
                        self.tess_mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
                    else:
                        self.mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
        else:
            self.mdi.add_command(self._cube_alloc, instance_count=1, base_instance=0)
            if active_count > 1:
                self.mdi.add_command(self._sphere_alloc, instance_count=active_count - 1, base_instance=1)

        # Setup Graph Context Blackboard
        ctx = self._graph_context
        ctx.camera_pos = (self._cam_pos[0], self._cam_pos[1], self._cam_pos[2])
        ctx.sun_dir = sun_dir
        ctx.sun_lux = sun_lux
        ctx.time_elapsed = time_elapsed
        ctx.resources["g_buffer"] = self.g_buffer
        ctx.resources["csm"] = self.csm

        # ---- PASS 1: Cascaded Shadow Maps Pass ----
        self.csm.fbo.use()
        self.csm.clear()
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Bind PBR texture array atlases to units 10..13 (essential for csm_tess_prog displacement sampling)
        self.texture_atlas.bind(10, 11, 12, 13)

        for c in range(self.config.csm_cascades):
            self.csm.begin_cascade(c)
            if self._csm_cascade_idx_uniform is not None:
                self._csm_cascade_idx_uniform.value = c
            self.mdi.submit(self.csm_vao, self.csm_prog)
            if getattr(self.config, "tess_enabled", True) and self.tess_mdi.command_count > 0:
                self.ctx.patch_vertices = 3
                if self._u_csm_tess_cascade_idx is not None:
                    self._u_csm_tess_cascade_idx.value = c
                if self._u_csm_tess_near_radius is not None:
                    self._u_csm_tess_near_radius.value = getattr(self.config, "disp_near_radius", 8.0)
                if self._u_csm_tess_mid_radius is not None:
                    self._u_csm_tess_mid_radius.value = getattr(self.config, "disp_mid_radius", 25.0)
                if self._u_csm_tess_max_level is not None:
                    self._u_csm_tess_max_level.value = getattr(self.config, "tess_max_level", 24.0)
                if self._u_csm_tess_med_level is not None:
                    self._u_csm_tess_med_level.value = getattr(self.config, "tess_med_level", 8.0)
                if self._u_csm_tess_disp_scale is not None:
                    self._u_csm_tess_disp_scale.value = getattr(self.config, "tess_displacement_scale", 1.0)
                self.texture_atlas.upload_depths(self.csm_tess_prog)
                self.tess_mdi.submit(self.csm_tess_vao, self.csm_tess_prog)

        # ---- PASS 2: G-Buffer Pass (Reversed-Z) ----
        self.g_buffer.clear()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.depth_func = ">" if self.config.reverse_z else "<"
        self.ctx.enable(moderngl.CULL_FACE)

        # Bind PBR texture array atlases to units 10..13
        self.texture_atlas.bind(10, 11, 12, 13)

        # Set POM and Radius uniforms on gbuffer_prog
        if self._u_gbuffer_pom_enabled is not None:
            self._u_gbuffer_pom_enabled.value = 1 if getattr(self.config, "pom_enabled", True) else 0
        if self._u_gbuffer_pom_min_samples is not None:
            self._u_gbuffer_pom_min_samples.value = getattr(self.config, "pom_min_samples", 8)
        if self._u_gbuffer_pom_max_samples is not None:
            self._u_gbuffer_pom_max_samples.value = getattr(self.config, "pom_max_samples", 64)
        if self._u_gbuffer_pom_height_scale is not None:
            self._u_gbuffer_pom_height_scale.value = getattr(self.config, "pom_height_scale", 1.0)
        if self._u_gbuffer_pom_scale_mult is not None:
            self._u_gbuffer_pom_scale_mult.value = getattr(self.config, "pom_height_scale", 1.0)
        if self._u_gbuffer_pom_self_shadow is not None:
            self._u_gbuffer_pom_self_shadow.value = 1 if getattr(self.config, "pom_self_shadow", True) else 0
        if self._u_gbuffer_disp_near_radius is not None:
            self._u_gbuffer_disp_near_radius.value = getattr(self.config, "disp_near_radius", 8.0)
        if self._u_gbuffer_disp_mid_radius is not None:
            self._u_gbuffer_disp_mid_radius.value = getattr(self.config, "disp_mid_radius", 25.0)
        self.texture_atlas.upload_depths(self.gbuffer_prog)

        is_wireframe = getattr(self.config, "wireframe", False)
        if is_wireframe:
            self.ctx.wireframe = True
        try:
            if self._u_gbuffer_prev_vp is not None:
                self._u_gbuffer_prev_vp.write(self._prev_vp_mat.tobytes())
            self.mdi.submit(self.gbuffer_vao, self.gbuffer_prog)

            # Submit Hardware Tessellation batches (mode=PATCHES)
            if getattr(self.config, "tess_enabled", True) and self.tess_mdi.command_count > 0:
                self.ctx.patch_vertices = 3
                if self._u_tess_enabled is not None:
                    self._u_tess_enabled.value = 1
                if self._u_tess_frustum_cull is not None:
                    self._u_tess_frustum_cull.value = 1 if getattr(self.config, "frustum_cull_enabled", True) else 0
                if self._u_tess_near_radius is not None:
                    self._u_tess_near_radius.value = getattr(self.config, "disp_near_radius", 8.0)
                if self._u_tess_mid_radius is not None:
                    self._u_tess_mid_radius.value = getattr(self.config, "disp_mid_radius", 25.0)
                if self._u_tess_max_level is not None:
                    self._u_tess_max_level.value = getattr(self.config, "tess_max_level", 24.0)
                if self._u_tess_med_level is not None:
                    self._u_tess_med_level.value = getattr(self.config, "tess_med_level", 8.0)
                if self._u_tess_disp_scale is not None:
                    self._u_tess_disp_scale.value = getattr(self.config, "tess_displacement_scale", 1.0)
                if self._u_tess_pom_scale_mult is not None:
                    self._u_tess_pom_scale_mult.value = getattr(self.config, "pom_height_scale", 1.0)
                self.texture_atlas.upload_depths(self.gbuffer_tess_prog)
                if self._u_tess_prev_vp is not None:
                    self._u_tess_prev_vp.write(self._prev_vp_mat.tobytes())
                self.tess_mdi.submit(self.gbuffer_tess_vao, self.gbuffer_tess_prog)
        finally:
            if is_wireframe:
                self.ctx.wireframe = False

        # ---- PASS 2.5: Screen-Space Displacement Mapping (SSDM) ----
        if getattr(self.config, "ssdm_enabled", True):
            self.ssdm_pass.execute(
                ctx=ctx,
                g_buffer=self.g_buffer,
                displacement_array=self.texture_atlas.displacement_array,
                enabled=self.config.ssdm_enabled,
                scale=getattr(self.config, "ssdm_scale", 1.0),
                max_distance=getattr(self.config, "ssdm_max_distance", 30.0),
                disp_mid_radius=getattr(self.config, "disp_mid_radius", 25.0),
                material_depths=self.texture_atlas.material_depths,
            )

        # ---- PASS 2.6: Hierarchical-Z (Hi-Z) Depth Pyramid Pass ----
        self.hiz_pass.execute(ctx)

        # ---- PASS 3: Ambient Occlusion Pass (GTAO / SSAO) ----
        self.ao_pass.execute(ctx)

        # ---- PASS 4: Clustered Dynamic Local Lights (SSBO 3) ----
        # ---- PASS 5: Global Illumination (SSGI + LPV) ----
        ctx.resources["scene_color"] = getattr(self.post_process, "hdr_texture", None) or self.g_buffer.albedo_roughness_texture
        self.ssgi_pass.execute(ctx)
        self.lpv_pass.execute(ctx)

        # ---- PASS 6: Image-Based Lighting & Screen-Space Reflections (SSR) ----
        self.ibl_pass.execute(ctx)
        self.ssr_pass.execute(ctx)

        # ---- PASS 6.5: Physical Sky & Atmosphere (Sébastien Hillaire Eurographics 2020) ----
        self.sky_atmosphere_pass.update(
            atmo_config=self.atmosphere.config,
            camera_pos=(float(self._cam_pos[0]), float(self._cam_pos[1]), float(self._cam_pos[2])),
            sun_dir=true_sun_dir,
            sun_color=self.atmosphere.config.sun_color,
            sun_intensity=sun_lux,
        )

        # ---- PASS 7: Consolidated Deferred Resolve ----
        self.post_process.hdr_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Bind all required texture units
        self.g_buffer.bind_textures(base_unit=0)     # 0: AlbedoRough, 1: NormalMetal, 2: Depth
        self.csm.depth_texture.use(location=3)      # 3: ShadowAtlas

        ao_tex = ctx.resources.get("ao_texture", self.ao_pass.white_fallback)
        ao_tex.use(location=4)

        ssgi_tex = ctx.resources.get("ssgi_texture", self.ssgi_pass.black_fallback)
        ssgi_tex.use(location=5)

        self.ibl_pass.lut_tex.use(location=6)
        self.ibl_pass.env_tex.use(location=7)

        ssr_tex = ctx.resources.get("ssr_texture", self.ssr_pass.black_fallback)
        ssr_tex.use(location=8)

        lpv_vol = ctx.resources.get("lpv_volume", self.lpv_pass.black_fallback)
        lpv_vol.use(location=9)

        self.sky_atmosphere_pass.sky_view_lut.use(location=10)
        self.sky_atmosphere_pass.transmittance_lut.use(location=11)
        self.hiz_pass.hiz_texture.use(location=14)

        # Set resolve uniforms
        if self._u_pcf_samples is not None:
            self._u_pcf_samples.value = self.config.pcf_samples
        if self._u_shadow_mode is not None:
            mode_val = 0 if self.config.shadow_mode == "HARD" else (1 if self.config.shadow_mode == "PCF" else 2)
            self._u_shadow_mode.value = mode_val
        if self._u_shadow_softness is not None:
            self._u_shadow_softness.value = float(self.config.shadow_softness)
        if self._u_shadow_bias is not None:
            self._u_shadow_bias.value = float(self.config.shadow_bias)
        if self._u_shadow_normal_bias is not None:
            self._u_shadow_normal_bias.value = float(getattr(self.config, "shadow_normal_bias", 0.0010))
        if self._u_shadow_offset is not None:
            self._u_shadow_offset.value = (
                float(getattr(self.config, "shadow_offset_x", 0.0)),
                float(getattr(self.config, "shadow_offset_y", 0.0)),
            )
        if self._u_cascade_count is not None:
            self._u_cascade_count.value = self.config.csm_cascades
        if self._u_gbuffer_debug is not None:
            self._u_gbuffer_debug.value = self.config.debug_gbuffer
        if self._u_hiz_debug_mip is not None:
            self._u_hiz_debug_mip.value = int(getattr(self.config, "hiz_debug_mip", 0))

        # Set Phase 5 Feature Toggles
        if self._u_ao_enabled is not None:
            self._u_ao_enabled.value = 1 if self.config.ao_mode != "OFF" else 0

        gi_enum_val = 0
        if self.config.gi_mode == "SSGI":
            gi_enum_val = 1
        elif self.config.gi_mode == "LPV":
            gi_enum_val = 2
        elif self.config.gi_mode == "HYBRID":
            gi_enum_val = 3
        if self._u_gi_enabled is not None:
            self._u_gi_enabled.value = gi_enum_val

        if self._u_ibl_enabled is not None:
            self._u_ibl_enabled.value = 1 if self.config.ibl_enabled else 0
        if self._u_ssr_enabled is not None:
            self._u_ssr_enabled.value = 1 if self.config.ssr_enabled else 0
        if self._u_point_light_count is not None:
            self._u_point_light_count.value = ctx.resources.get("point_light_count", 0)

        if self._u_lpv_min is not None and "lpv_min" in ctx.resources:
            self._u_lpv_min.value = ctx.resources["lpv_min"]
        if self._u_lpv_size is not None and "lpv_size" in ctx.resources:
            self._u_lpv_size.value = ctx.resources["lpv_size"]

        # Upload Parameterized Physical Atmosphere Uniforms
        atmo = self.atmosphere.config
        self.sky_atmosphere_pass._update_common_uniforms(self.resolve_prog, atmo)
        if self._u_rayleigh_beta is not None:
            self._u_rayleigh_beta.value = atmo.rayleigh_beta
        if self._u_mie_beta is not None:
            if atmo.mie_beta != self._prev_mie_beta:
                mb = float(atmo.mie_beta)
                self._mie_beta_tuple = (mb, mb, mb)
                self._prev_mie_beta = atmo.mie_beta
            self._u_mie_beta.value = self._mie_beta_tuple
        if self._u_mie_g is not None:
            self._u_mie_g.value = atmo.mie_asymmetry
        if self._u_ozone_beta is not None:
            self._u_ozone_beta.value = atmo.ozone_beta
        if self._u_atmo_ground_col is not None:
            self._u_atmo_ground_col.value = atmo.ground_color
        if self._u_night_zenith is not None:
            self._u_night_zenith.value = atmo.night_zenith
        if self._u_night_horizon is not None:
            self._u_night_horizon.value = atmo.night_horizon
        if self._u_sun_disc_size is not None:
            self._u_sun_disc_size.value = atmo.sun_disc_size
        if self._u_moon_disc_size is not None:
            self._u_moon_disc_size.value = atmo.moon_disc_size
        if self._u_moon_color is not None:
            self._u_moon_color.value = atmo.moon_color
        if self._u_star_intensity is not None:
            self._u_star_intensity.value = atmo.star_intensity
        if self._u_star_density is not None:
            self._u_star_density.value = atmo.star_density
        if self._u_atmo_turbidity is not None:
            self._u_atmo_turbidity.value = atmo.turbidity
        if self._u_atmo_sun_dir is not None:
            sx, sy, sz = float(true_sun_dir[0]), float(true_sun_dir[1]), float(true_sun_dir[2])
            if (sx, sy, sz) != self._atmo_sun_tuple:
                self._atmo_sun_tuple = (sx, sy, sz)
            self._u_atmo_sun_dir.value = self._atmo_sun_tuple

        self.resolve_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # ---- PASS 7.5: Immediate-Mode 3D Debug Wireframes ----
        active_debug = debug_draw if debug_draw is not None else self.debug
        if active_debug is not None and active_debug.vertex_count > 0:
            self.post_process.hdr_fbo.use()
            active_debug.render()
            active_debug.clear()

        # ---- PASS 7.6: Froxel Volumetric Fog & Atmospheric Scattering ----
        ctx.resources["csm"] = self.csm
        ctx.resources["g_buffer"] = self.g_buffer
        ctx.resources["hdr_fbo"] = self.post_process.hdr_fbo
        self.volumetric_fog_pass.execute(ctx)

        # ---- PASS 7.65: Dynamic Water Simulation & Screen-Space Refraction (Phase 4) ----
        if self.water_pass is not None and getattr(self.config, "water_enabled", True):
            self.water_pass.render_water(
                hdr_fbo=self.post_process.hdr_fbo,
                depth_texture=self.g_buffer.depth_texture,
                config=self.config,
                time_elapsed=time_elapsed,
            )

        # ---- PASS 7.7: GPU Compute Particles & Volumetric VFX (Phase 2) ----
        if self.particle_pass is not None:
            self.particle_pass.update(
                dt=calc_dt,
                camera_pos=(float(self._cam_pos[0]), float(self._cam_pos[1]), float(self._cam_pos[2])),
            )
            self.particle_pass.render(
                hdr_fbo=self.post_process.hdr_fbo,
                depth_texture=self.g_buffer.depth_texture,
            )

        # ---- PASS 7.8: Cinematic Camera Optics & Lens Flares (Phase 3) ----
        flare_tex = None
        if self.camera_optics_pass is not None:
            optics_needed = (
                getattr(self.config, "dof_enabled", True)
                or getattr(self.config, "motion_blur_enabled", True)
            )
            if optics_needed:
                scene_hdr_tex = self.post_process.hdr_fbo.color_attachments[0]
                self.camera_optics_pass.render_optics(
                    scene_hdr_texture=scene_hdr_tex,
                    depth_texture=self.g_buffer.depth_texture,
                    prev_vp_mat=self._prev_vp_mat,
                    config=self.config,
                )
                self.ctx.copy_framebuffer(self.post_process.hdr_fbo, self.camera_optics_pass.optics_fbo)

            if getattr(self.config, "lens_flare_enabled", True):
                scene_hdr_tex = self.post_process.hdr_fbo.color_attachments[0]
                flare_tex = self.camera_optics_pass.render_flare(
                    scene_hdr_texture=scene_hdr_tex,
                    config=self.config,
                )

        # ---- PASS 8: Post-Process & Mutually Exclusive Anti-Aliasing ----
        self.ctx.disable(moderngl.DEPTH_TEST)
        aa_mode = getattr(self.config, "aa_mode", "OFF")

        self.post_process.render(
            target_fbo=self.post_process.final_fbo,
            flare_texture=flare_tex,
            time_elapsed=time_elapsed,
        )
        ctx.resources["ldr_color"] = self.post_process.final_texture

        if aa_mode == "FXAA":
            self.fxaa_pass.execute(ctx)
            self.ctx.copy_framebuffer(self.post_process.final_fbo, self.fxaa_pass.fbo)
        elif aa_mode in ("SMAA_1X", "SMAA_2X", "SMAA_4X"):
            self.smaa_pass.execute(ctx)
            if aa_mode == "SMAA_1X":
                self.ctx.copy_framebuffer(self.post_process.final_fbo, self.smaa_pass.fbo_smaa)
            else:
                self.ctx.copy_framebuffer(self.post_process.final_fbo, self.smaa_pass._write_fbo)

        if not self.ctx_wrapper.is_headless:
            self.ctx.copy_framebuffer(self.ctx.screen, self.post_process.final_fbo)

        # Store current ViewProjection for next frame's motion vectors (zero-allocation)
        np.copyto(self._prev_vp_mat, self._vp_mat)

    @property
    def output_texture_id(self) -> int:
        """Exposes OpenGL texture handle for ImGui viewport dock."""
        return self.post_process.output_texture_id

    def destroy(self) -> None:
        unsubscribe_event(WindowResizeEvent, self._on_window_resize)
        self.debug.destroy()
        self.resources.close()
        self.frame_context.destroy()
        self.mega_buffer.destroy()
        self.mdi.destroy()
        self.g_buffer.destroy()
        self.csm.destroy()
        self.ibl_pass.destroy()
        self.ao_pass.destroy()
        self.ssgi_pass.destroy()
        self.lpv_pass.destroy()
        self.lights_pass.destroy()
        self.ssr_pass.destroy()
        self.taa_pass.destroy()
        self.fxaa_pass.destroy()
        self.smaa_pass.destroy()
        self.volumetric_fog_pass.destroy()
        if self.particle_pass is not None:
            self.particle_pass.destroy()
        if self.camera_optics_pass is not None:
            self.camera_optics_pass.destroy()
        self.post_process.destroy()
        self.ssbo_transforms.release()
        self.ssbo_materials.release()
        self.gbuffer_prog.release()
        self.csm_prog.release()
        self.resolve_prog.release()
        self.gbuffer_vao.release()
        self.csm_vao.release()
        self.texture_atlas.destroy()
        self.ssdm_pass.destroy()
        self.hiz_pass.destroy()
        self.tess_mdi.destroy()
        self.gbuffer_tess_prog.release()
        self.gbuffer_tess_vao.release()
        self.csm_tess_prog.release()
        self.csm_tess_vao.release()
        self.resolve_vao.release()
