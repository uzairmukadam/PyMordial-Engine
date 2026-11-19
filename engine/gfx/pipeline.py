"""Unified Master Render Graph Coordinator for PyMordial Engine.

Coordinates the full ModernGL 4.5 MDI geometry pass, Reversed-Z depth, Cascaded Shadow Maps,
Image-Based Lighting (IBL), GTAO, SSGI, Clustered Local Lighting, SSR, TAA, and Tone-Mapping.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import moderngl
import numpy as np

from engine.assets.resource_cache import ResourceCache
from engine.core.ecs import EntityManager
from engine.core.math_utils import mat4_inv, mat4_mul, matrix_look_at, matrix_perspective
from engine.debug.debug_draw import DebugDraw
from engine.events import WindowResizeEvent, subscribe_event, unsubscribe_event
from engine.gfx.context import RenderContext
from engine.gfx.culling import FrustumCuller
from engine.gfx.environment import create_default_environment
from engine.gfx.frame_context import FrameContext
from engine.gfx.g_buffer import GBuffer
from engine.gfx.gpu_profiler import GpuPassProfiler
from engine.gfx.material_registry import MaterialRegistry
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.passes.ao_pass import AmbientOcclusionPass
from engine.gfx.passes.bloom_pass import BloomPass
from engine.gfx.passes.camera_optics_pass import CameraOpticsPass
from engine.gfx.passes.clustered_lights import ClusteredLightingPass, PointLight, SpotLight
from engine.gfx.passes.decal_pass import DecalPass
from engine.gfx.passes.direct_radiance_pass import DirectRadiancePass
from engine.gfx.passes.exposure_pass import ExposurePass
from engine.gfx.passes.fxaa_pass import FXAAPass
from engine.gfx.passes.hiz_pass import HiZPass
from engine.gfx.passes.ibl import IBLPass
from engine.gfx.passes.motion_blur_pass import MotionBlurPass
from engine.gfx.passes.particle_pass import ParticleSystemPass
from engine.gfx.passes.radiance_cascades_pass import RadianceCascadesPass
from engine.gfx.passes.smaa_pass import SMAAPass
from engine.gfx.passes.solar_flare_pass import SolarFlarePass
from engine.gfx.passes.ssdm_pass import SSDMPass
from engine.gfx.passes.ssgi_pass import SSGIPass
from engine.gfx.passes.ssr_pass import SSRPass
from engine.gfx.passes.taa_pass import TAAPass
from engine.gfx.passes.volumetric_fog_pass import VolumetricFogPass
from engine.gfx.passes.water_pass import WaterPass
from engine.gfx.post_process import PostProcessPipeline
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.render_feature import RenderFeature, RenderFeatureRegistry
from engine.gfx.render_graph import RenderGraph, RenderGraphContext, RenderPass
from engine.gfx.render_services import RenderServices
from engine.gfx.shadow_csm import CascadedShadowMap
from engine.gfx.spot_shadows import SpotLightShadowMap
from engine.gfx.texture_atlas import TextureArrayAtlas

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()
ROOT_DIR = SHADER_DIR.parent
_load_shader = load_shader


class _BuiltinPass(RenderPass):
    """Declarative placeholder describing a built-in pipeline stage.

    Built-in stages are executed by ``RenderPipeline.render_frame`` in explicit
    order; these instances exist so the render graph can validate ordering and
    resource contracts, and so module features can anchor to them by name.
    """

    def execute(self, context: RenderGraphContext) -> None:
        pass


class RenderPipeline:
    """Coordinates the full ModernGL 4.5 MDI, Reversed-Z, CSM, and deferred resolve pipeline."""

    __slots__ = (
        "ctx_wrapper",
        "ctx",
        "config",
        "frame_context",
        "mega_buffer",
        "mdi",
        "shadow_mdi",
        "cascade_shadow_mdis",
        "g_buffer",
        "csm",
        "ibl_pass",
        "ao_pass",
        "ssgi_pass",
        "rc_pass",
        "lights_pass",
        "ssr_pass",
        "taa_pass",
        "fxaa_pass",
        "smaa_pass",
        "volumetric_fog_pass",
        "bloom_pass",
        "exposure_pass",
        "_prev_vp_mat",
        "_prev_view_mat",
        "_prev_transforms_seeded",
        "post_process",
        "resources",
        "debug",
        "ssbo_transforms",
        "ssbo_materials",
        "ssbo_prev_transforms",
        "gbuffer_prog",
        "csm_prog",
        "resolve_prog",
        "resolve_vao",
        "gbuffer_vao",
        "csm_vao",
        "spot_shadow_map",
        "spot_depth_prog",
        "spot_depth_vao",
        "texture_atlas",
        "material_registry",
        "ssdm_pass",
        "decal_pass",
        "hiz_pass",
        "direct_radiance_pass",
        "culler",
        "stats_visible_entities",
        "stats_culled_entities",
        "stats_visible_commands",
        "stats_culled_commands",
        "stats_shadow_commands",
        "stats_shadow_commands_unculled",
        "_caster_batches_scratch",
        "tess_mdi",
        "gbuffer_tess_prog",
        "gbuffer_tess_vao",
        "csm_tess_prog",
        "csm_tess_vao",
        "_view_mat",
        "_proj_mat",
        "_vp_mat",
        "_unjittered_proj_mat",
        "_unjittered_vp_mat",
        "_inv_proj",
        "_inv_view",
        "_cam_pos",
        "_cam_fwd",
        "_sun_v",
        "_screen_size_tuple",
        "_zero_jitter",
        "_mie_beta_tuple",
        "_prev_mie_beta",
        "_csm_cascade_idx_uniform",
        "_u_shadow_bias",
        "_u_shadow_normal_bias",
        "_u_shadow_bias_slope",
        "_u_shadow_bias_min",
        "_u_shadow_bias_max",
        "_u_shadow_filter_mode",
        "_u_pcf_cascade_limit",
        "_u_pcf_samples",
        "_u_pcf_radius",
        "_u_pcf_falloff",
        "_u_pcss_cascade_limit",
        "_u_pcss_search_samples",
        "_u_pcss_search_radius",
        "_u_pcss_light_size",
        "_u_pcss_samples",
        "_u_pcss_falloff",
        "_u_pcss_max_radius",
        "_u_pcss_closest_weight",
        "_u_cascade_atlas_rect",
        "_u_cascade_count",
        "_u_gbuffer_debug",
        "_u_hiz_debug_mip",
        "_u_gbuffer_prev_vp",
        "_u_gbuffer_object_motion",
        "_u_gbuffer_tess_object_motion",
        "_u_ao_enabled",
        "_u_gi_enabled",
        "_u_ibl_enabled",
        "_u_ssr_enabled",
        "_u_energy_compensation_enabled",
        "_u_specular_aa_enabled",
        "_u_specular_aa_strength",
        "_u_point_light_count",
        "_u_spot_light_count",
        "_u_spot_shadow_enabled",
        "_u_spot_vp",
        "_u_spot_depth_vp",
        "_spot_matrices_bytes",
        "_u_gbuffer_pom_enabled",
        "_u_gbuffer_pom_min_samples",
        "_u_gbuffer_pom_max_samples",
        "_u_gbuffer_pom_height_scale",
        "_u_gbuffer_pom_scale_mult",
        "_u_gbuffer_pom_self_shadow",
        "_u_gbuffer_disp_near_radius",
        "_u_gbuffer_disp_mid_radius",
        "_u_gbuffer_car_paint_layer",
        "_u_gbuffer_car_glass_layer",
        "_u_gbuffer_car_rim_layer",
        "_u_gbuffer_car_tire_layer",
        "_u_gbuffer_car_chrome_layer",
        "_u_gbuffer_car_taillight_layer",
        "_u_gbuffer_car_indicator_layer",
        "_u_gbuffer_car_interior_layer",
        "_u_gbuffer_brick_wall_layer",
        "_u_gbuffer_concrete_floor_layer",
        "_u_gbuffer_tess_brick_wall_layer",
        "_u_gbuffer_tess_concrete_floor_layer",
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
        "environment",
        "_u_rayleigh_beta",
        "_u_mie_beta",
        "_u_mie_g",
        "_u_ozone_beta",
        "_u_atmo_ground_col",
        "_u_night_zenith",
        "_u_night_horizon",
        "_u_sky_mode",
        "_u_space_zenith",
        "_u_nebula_color",
        "_u_nebula_intensity",
        "_u_sun_disc_size",
        "_u_moon_disc_size",
        "_u_moon_color",
        "_u_moon_intensity",
        "_u_star_intensity",
        "_u_star_density",
        "_u_atmo_turbidity",
        "_u_atmo_sun_dir",
        "particle_pass",
        "camera_optics_pass",
        "solar_flare_pass",
        "motion_blur_pass",
        "water_pass",
        "_prev_time",
        "_graph_context",
        "render_graph",
        "render_features",
        "_render_services",
        "gpu_profiler",
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
        self.shadow_mdi = MultiDrawIndirect(self.ctx)
        self.cascade_shadow_mdis = [MultiDrawIndirect(self.ctx) for _ in range(4)]

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
            split_lambda=getattr(self.config, "csm_split_lambda", 0.85),
        )
        self.csm.update_splits(
            self.config.shadow_distance,
            self.config.csm_cascades,
            getattr(self.config, "csm_split_lambda", 0.85),
            getattr(self.config, "csm_cascade_distances", None),
        )

        # 5. Initialize SSBO 1 (Transforms) & SSBO 2 (Materials)
        max_entities = int(getattr(self.config, "max_entities", 100_000))
        self.ssbo_transforms = self.ctx.buffer(reserve=max_entities * 16 * 4)
        self.ssbo_transforms.bind_to_storage_buffer(binding=1)

        self.ssbo_materials = self.ctx.buffer(reserve=max_entities * 8 * 4)
        self.ssbo_materials.bind_to_storage_buffer(binding=2)

        # SSBO 7: Previous-frame world transforms, consumed by the G-buffer
        # vertex shader to build per-object motion vectors. Zero-initialised, so
        # the first frame falls back to the current transform (see gbuffer.vert).
        self.ssbo_prev_transforms = self.ctx.buffer(reserve=max_entities * 16 * 4)
        self.ssbo_prev_transforms.bind_to_storage_buffer(binding=7)

        # 6. Initialize Phase 5 High-End Graphics Passes
        self.ibl_pass = IBLPass(self.ctx)
        self.ao_pass = AmbientOcclusionPass(
            self.ctx,
            w,
            h,
            resolution_scale=getattr(self.config, "ao_resolution_scale", 1.0),
        )
        self.ssgi_pass = SSGIPass(
            self.ctx,
            w,
            h,
            resolution_scale=getattr(self.config, "ssgi_resolution_scale", 1.0),
        )
        self.rc_pass = RadianceCascadesPass(
            self.ctx,
            w,
            h,
            resolution_scale=getattr(self.config, "rc_resolution_scale", 1.0),
        )
        self.lights_pass = ClusteredLightingPass(
            self.ctx,
            max_lights=self.config.max_point_lights,
            max_spot_lights=getattr(self.config, "max_spot_lights", 64),
        )
        self.spot_shadow_map = SpotLightShadowMap(
            self.ctx,
            atlas_size=getattr(self.config, "spot_shadow_resolution", 2048),
        )
        self.ssr_pass = SSRPass(
            self.ctx,
            w,
            h,
            resolution_scale=getattr(self.config, "ssr_resolution_scale", 1.0),
        )
        self.taa_pass = TAAPass(self.ctx, w, h)
        self.fxaa_pass = FXAAPass(self.ctx, w, h)
        self.smaa_pass = SMAAPass(self.ctx, w, h)
        self.volumetric_fog_pass = VolumetricFogPass(self.ctx, w, h)
        self._prev_vp_mat = np.identity(4, dtype=np.float32).flatten()
        self._prev_view_mat = np.identity(4, dtype=np.float32).flatten()
        self._prev_transforms_seeded = False

        # Phase 3: Frustum Culling & MDI Batch Filter
        self.culler = FrustumCuller(max_entities=max_entities)
        self.stats_visible_entities = 0
        self.stats_culled_entities = 0
        self.stats_visible_commands = 0
        self.stats_culled_commands = 0
        self.stats_shadow_commands = 0
        self.stats_shadow_commands_unculled = 0
        # Reused every frame so shadow caster filtering allocates nothing.
        self._caster_batches_scratch: list[tuple] = []

        # Pre-allocated scratch buffers for zero-allocation frame rendering
        self._view_mat = np.zeros(16, dtype=np.float32)
        self._proj_mat = np.zeros(16, dtype=np.float32)
        self._vp_mat = np.zeros(16, dtype=np.float32)
        self._unjittered_proj_mat = np.zeros(16, dtype=np.float32)
        self._unjittered_vp_mat = np.zeros(16, dtype=np.float32)
        self._inv_proj = np.zeros(16, dtype=np.float32)
        self._inv_view = np.zeros(16, dtype=np.float32)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_fwd = np.zeros(3, dtype=np.float32)
        self._sun_v = np.zeros(3, dtype=np.float32)
        self._screen_size_tuple = (float(w), float(h))
        self._zero_jitter = (0.0, 0.0)
        self._mie_beta_tuple = (0.0, 0.0, 0.0)
        self._prev_mie_beta = -1.0

        # 7. Compile Core Programs
        quad_vert = _load_shader("fullscreen_quad.vert")
        gbuffer_vert = _load_shader("gbuffer.vert")
        gbuffer_frag = _load_shader("gbuffer.frag")
        csm_vert = _load_shader("csm_depth.vert")
        csm_frag = _load_shader("csm_depth.frag")
        spot_depth_vert = _load_shader("spot_depth.vert")
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

        self.spot_depth_prog = self.ctx.program(
            vertex_shader=spot_depth_vert,
            fragment_shader=csm_frag,
        )
        self.spot_depth_vao = self.mega_buffer.get_vao(self.spot_depth_prog)

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

        # Automatically rebuild VAOs whenever MegaBuffer bakes/reallocates
        self.mega_buffer.on_bake_callbacks.append(self.rebuild_vaos)

        # Texture Array Atlas for PBR Materials & Central Material Registry
        self.texture_atlas = TextureArrayAtlas(self.ctx, width=2048, height=2048, max_layers=32)
        self.material_registry = MaterialRegistry()
        self.material_registry.sync_with_atlas(self.texture_atlas)

        # SSDM Post-G-Buffer Pass
        self.ssdm_pass = SSDMPass(self.ctx, w, h)

        # Screen-Space Deferred Decal Pass (Phase 9)
        self.decal_pass = DecalPass(self.ctx, w, h)

        # Hierarchical-Z (Hi-Z) Depth Pyramid Pass
        self.hiz_pass = HiZPass(self.ctx, w, h, reverse_z=self.config.reverse_z)

        # Current-frame direct radiance (GI source for SSGI / future RC)
        self.direct_radiance_pass = DirectRadiancePass(self.ctx, w, h)

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
            "u_SkyViewLUT": 10,
            "u_TransmittanceLUT": 11,
            "u_SpotShadowAtlas": 12,
            "u_IrradianceMap": 13,
            "u_HiZTexture": 14,
            "u_RCTexture": 15,
        }
        for name, unit in tex_uniforms.items():
            if name in self.resolve_prog:
                self.resolve_prog[name].value = unit

        self._u_spot_light_count = self.resolve_prog.get("u_SpotLightCount", None)
        self._u_spot_shadow_enabled = self.resolve_prog.get("u_SpotShadowEnabled", None)
        self._u_spot_vp = self.resolve_prog.get("u_SpotLightViewProjection", None)
        self._u_spot_depth_vp = self.spot_depth_prog.get("u_SpotLightViewProjection", None)
        self._spot_matrices_bytes = bytearray(4 * 16 * 4)

        # Pre-cache mesh uniforms
        self._csm_cascade_idx_uniform = self.csm_prog.get("u_CascadeIndex", None)

        self._u_shadow_bias = self.resolve_prog.get("u_ShadowBias", None)
        self._u_shadow_normal_bias = self.resolve_prog.get("u_ShadowNormalBias", None)
        self._u_shadow_bias_slope = self.resolve_prog.get("u_ShadowBiasSlopeScale", None)
        self._u_shadow_bias_min = self.resolve_prog.get("u_ShadowBiasMinScale", None)
        self._u_shadow_bias_max = self.resolve_prog.get("u_ShadowBiasMaxScale", None)
        self._u_shadow_filter_mode = self.resolve_prog.get("u_ShadowFilterMode", None)
        self._u_pcf_cascade_limit = self.resolve_prog.get("u_PCFCascadeLimit", None)
        self._u_pcf_samples = self.resolve_prog.get("u_PCFSamples", None)
        self._u_pcf_radius = self.resolve_prog.get("u_PCFRadius", None)
        self._u_pcf_falloff = self.resolve_prog.get("u_PCFFalloff", None)
        self._u_pcss_cascade_limit = self.resolve_prog.get("u_PCSS_CascadeLimit", None)
        self._u_pcss_search_samples = self.resolve_prog.get("u_PCSSSearchSamples", None)
        self._u_pcss_search_radius = self.resolve_prog.get("u_PCSSSearchRadius", None)
        self._u_pcss_light_size = self.resolve_prog.get("u_PCSSLightSize", None)
        self._u_pcss_samples = self.resolve_prog.get("u_PCSSSamples", None)
        self._u_pcss_falloff = self.resolve_prog.get("u_PCSSFalloff", None)
        self._u_pcss_max_radius = self.resolve_prog.get("u_PCSSMaxRadius", None)
        self._u_pcss_closest_weight = self.resolve_prog.get("u_PCSSClosestBlockerWeight", None)
        self._u_cascade_atlas_rect = self.resolve_prog.get("u_CascadeAtlasRect", None)
        self._u_cascade_count = self.resolve_prog.get("u_CascadeCount", None)
        self._u_gbuffer_debug = self.resolve_prog.get("u_GBufferDebug", None)
        self._u_hiz_debug_mip = self.resolve_prog.get("u_HiZDebugMip", None)
        self._u_gbuffer_prev_vp = self.gbuffer_prog.get("u_PrevViewProjection", None)
        self._u_gbuffer_object_motion = self.gbuffer_prog.get("u_ObjectMotionVectors", None)
        self._u_gbuffer_tess_object_motion = self.gbuffer_tess_prog.get("u_ObjectMotionVectors", None)

        self._u_ao_enabled = self.resolve_prog.get("u_AOEnabled", None)
        self._u_gi_enabled = self.resolve_prog.get("u_GIEnabled", None)
        self._u_ibl_enabled = self.resolve_prog.get("u_IBLEnabled", None)
        self._u_ssr_enabled = self.resolve_prog.get("u_SSREnabled", None)
        self._u_energy_compensation_enabled = self.resolve_prog.get("u_EnergyCompensationEnabled", None)
        self._u_specular_aa_enabled = self.gbuffer_prog.get("u_SpecularAAEnabled", None)
        self._u_specular_aa_strength = self.gbuffer_prog.get("u_SpecularAAStrength", None)
        self._u_point_light_count = self.resolve_prog.get("u_PointLightCount", None)
        self._u_spot_light_count = self.resolve_prog.get("u_SpotLightCount", None)
        self._u_spot_shadow_enabled = self.resolve_prog.get("u_SpotShadowEnabled", None)
        self._u_spot_vp = self.resolve_prog.get("u_SpotLightViewProjection", None)
        self._u_spot_depth_vp = self.spot_depth_prog.get("u_SpotLightViewProjection", None)
        self._spot_matrices_bytes = bytearray(4 * 64)

        # Pre-cache POM and micro-geometry uniform handles
        self._u_gbuffer_pom_enabled = self.gbuffer_prog.get("u_POMEnabled", None)
        self._u_gbuffer_pom_min_samples = self.gbuffer_prog.get("u_POMMinSamples", None)
        self._u_gbuffer_pom_max_samples = self.gbuffer_prog.get("u_POMMaxSamples", None)
        self._u_gbuffer_pom_height_scale = self.gbuffer_prog.get("u_POMHeightScale", None)
        self._u_gbuffer_pom_scale_mult = self.gbuffer_prog.get("u_POMScaleMultiplier", None)
        self._u_gbuffer_pom_self_shadow = self.gbuffer_prog.get("u_POMSelfShadow", None)
        self._u_gbuffer_disp_near_radius = self.gbuffer_prog.get("u_DispNearRadius", None)
        self._u_gbuffer_disp_mid_radius = self.gbuffer_prog.get("u_DispMidRadius", None)
        self._u_gbuffer_car_paint_layer = self.gbuffer_prog.get("u_CarPaintLayer", None)
        self._u_gbuffer_car_glass_layer = self.gbuffer_prog.get("u_CarGlassLayer", None)
        self._u_gbuffer_car_rim_layer = self.gbuffer_prog.get("u_CarRimLayer", None)
        self._u_gbuffer_car_tire_layer = self.gbuffer_prog.get("u_CarTireLayer", None)
        self._u_gbuffer_car_chrome_layer = self.gbuffer_prog.get("u_CarChromeLayer", None)
        self._u_gbuffer_car_taillight_layer = self.gbuffer_prog.get("u_CarTaillightLayer", None)
        self._u_gbuffer_car_indicator_layer = self.gbuffer_prog.get("u_CarIndicatorLayer", None)
        self._u_gbuffer_car_interior_layer = self.gbuffer_prog.get("u_CarInteriorLayer", None)
        self._u_gbuffer_brick_wall_layer = self.gbuffer_prog.get("u_BrickWallLayer", None)
        self._u_gbuffer_concrete_floor_layer = self.gbuffer_prog.get("u_ConcreteFloorLayer", None)
        self._u_gbuffer_tess_brick_wall_layer = self.gbuffer_tess_prog.get("u_BrickWallLayer", None)
        self._u_gbuffer_tess_concrete_floor_layer = self.gbuffer_tess_prog.get("u_ConcreteFloorLayer", None)

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

        # 8. Environment Provider (sky, atmosphere, clouds)
        # The pipeline consumes the EnvironmentProvider contract and never owns a
        # specific atmosphere implementation. The default provider is resolved
        # lazily so a disabled module is never imported.
        self.environment = create_default_environment(self.ctx, self.config)
        self.atmosphere = self.environment
        self._u_rayleigh_beta = self.resolve_prog.get("u_RayleighBeta", None)
        self._u_mie_beta = self.resolve_prog.get("u_MieBeta", None)
        self._u_mie_g = self.resolve_prog.get("u_MieG", None)
        self._u_ozone_beta = self.resolve_prog.get("u_OzoneBeta", None)
        self._u_atmo_ground_col = self.resolve_prog.get("u_AtmosphereGroundColor", None)
        self._u_night_zenith = self.resolve_prog.get("u_NightZenithColor", None)
        self._u_night_horizon = self.resolve_prog.get("u_NightHorizonColor", None)
        self._u_sky_mode = self.resolve_prog.get("u_SkyMode", None)
        self._u_space_zenith = self.resolve_prog.get("u_SpaceZenithColor", None)
        self._u_nebula_color = self.resolve_prog.get("u_NebulaColor", None)
        self._u_nebula_intensity = self.resolve_prog.get("u_NebulaIntensity", None)
        self._u_sun_disc_size = self.resolve_prog.get("u_SunDiscSize", None)
        self._u_moon_disc_size = self.resolve_prog.get("u_MoonDiscSize", None)
        self._u_moon_color = self.resolve_prog.get("u_MoonColor", None)
        self._u_moon_intensity = self.resolve_prog.get("u_MoonIntensity", None)
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
        self.camera_optics_pass = CameraOpticsPass(
            self.ctx,
            w,
            h,
            flare_resolution_scale=getattr(self.config, "lens_flare_resolution_scale", 1.0),
        )
        # 8.6b. Sun-anchored physically-based solar flare (lens_flare_mode == "SOLAR")
        self.solar_flare_pass = SolarFlarePass(
            self.ctx,
            w,
            h,
            resolution_scale=getattr(self.config, "lens_flare_resolution_scale", 1.0),
        )
        # 8.6c. Velocity-buffer motion blur (executes after TAA)
        self.motion_blur_pass = MotionBlurPass(self.ctx, w, h)

        # 8.7. HDR Bloom Pyramid
        self.bloom_pass = BloomPass(self.ctx, w, h)

        # 8.75. GPU Auto-Exposure (histogram + temporal adaptation)
        self.exposure_pass = ExposurePass(self.ctx)

        # 8.8. Dynamic Water Simulation & Screen-Space Refraction (Phase 4)
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

        # Render extension graph: describes the built-in frame so ordering and
        # resource declarations are validated, and hosts module render features.
        self.render_features = RenderFeatureRegistry()
        self._render_services = RenderServices(
            ctx=self.ctx,
            config=self.config,
            width=w,
            height=h,
            frame_context=self.frame_context,
            g_buffer=self.g_buffer,
            csm=self.csm,
            hdr_fbo=self.post_process.hdr_fbo,
            hdr_texture=self.post_process.hdr_texture,
            depth_texture=self.g_buffer.depth_texture,
            mega_buffer=self.mega_buffer,
            texture_atlas=self.texture_atlas,
            material_registry=self.material_registry,
        )
        self.render_graph = self._build_render_graph()

        # Non-blocking per-pass GPU timing. Timestamp queries are read from a
        # ring of older frames, so this never stalls the pipeline.
        self.gpu_profiler = GpuPassProfiler(self.ctx)

        # Automatically synchronize framebuffers with window resolution changes
        subscribe_event(WindowResizeEvent, self._on_window_resize, priority=90)

    # ---------------- Render Extension Graph ----------------

    def _build_render_graph(self) -> RenderGraph:
        """Describes the built-in frame as a validated render graph.

        The built-in passes are declared with their resource contracts so the
        graph can prove the frame is schedulable. Execution still follows the
        explicit order in ``render_frame``; this graph is the contract that
        module render features are validated against.
        """
        graph = RenderGraph()
        graph.declare_external_input(
            "g_buffer",
            "csm",
            "hdr_fbo",
            "scene_color",
            "prev_view",
            "hdr_color",
            "ldr_color",
            "prev_unjittered_vp",
        )

        builtin_passes: list[RenderPass] = [
            _BuiltinPass("ClusteredLights", outputs=("point_light_count", "spot_light_count", "shadow_spot_lights")),
            _BuiltinPass("CascadedShadows", inputs=("csm",), outputs=("shadow_atlas",)),
            _BuiltinPass("SpotShadows", inputs=("shadow_spot_lights",), outputs=("spot_shadow_atlas",)),
            _BuiltinPass("GBuffer", inputs=("g_buffer",), outputs=("gbuffer_depth",)),
            _BuiltinPass("SSDM", inputs=("g_buffer",), outputs=("ssdm_output",)),
            _BuiltinPass("Decals", inputs=("g_buffer",), outputs=("decal_output",)),
            _BuiltinPass("HiZ", inputs=("g_buffer",), outputs=("hiz_texture",)),
            _BuiltinPass("AmbientOcclusion", inputs=("g_buffer",), outputs=("ao_texture",)),
            _BuiltinPass("GlobalIllumination", inputs=("g_buffer", "scene_color"), outputs=("gi_texture",)),
            _BuiltinPass("IBL", outputs=("ibl_lut", "ibl_env_map", "ibl_irradiance")),
            _BuiltinPass("SSR", inputs=("g_buffer", "scene_color"), outputs=("ssr_texture",)),
            _BuiltinPass("SkyAtmosphere", outputs=("sky_view_lut", "transmittance_lut", "multiscattering_lut")),
            _BuiltinPass(
                "DeferredResolve",
                inputs=(
                    "g_buffer",
                    "shadow_atlas",
                    "ao_texture",
                    "gi_texture",
                    "ibl_lut",
                    "ibl_env_map",
                    "ssr_texture",
                    "sky_view_lut",
                    "transmittance_lut",
                    "hiz_texture",
                    "hdr_fbo",
                ),
                outputs=("hdr_color",),
            ),
            _BuiltinPass("VolumetricFog", inputs=("g_buffer", "csm", "hdr_fbo"), outputs=("fog_output",)),
            _BuiltinPass("Water", inputs=("g_buffer", "hdr_fbo"), outputs=("water_output",)),
            _BuiltinPass("Particles", inputs=("g_buffer", "hdr_fbo"), outputs=("particle_output",)),
            _BuiltinPass("CameraOptics", inputs=("g_buffer", "hdr_fbo"), outputs=("lens_flare_texture",)),
            _BuiltinPass("Exposure", inputs=("hdr_color",), outputs=("exposure_ev",)),
            _BuiltinPass("TAA", inputs=("hdr_color", "g_buffer"), outputs=("taa_output",)),
            _BuiltinPass("PostProcess", inputs=("hdr_color",), outputs=("ldr_color",)),
        ]
        for render_pass in builtin_passes:
            graph.add_pass(render_pass)

        return graph

    def register_feature(self, feature: RenderFeature) -> RenderFeature:
        """Registers a render feature and validates the resulting graph.

        Raises:
            RenderGraphError: if the feature's passes break ordering or resource
                contracts. The feature is not retained on failure.
        """
        self.render_features.register(feature)
        report = self.render_features.apply_to(self.render_graph, self._render_services)
        if not report.is_valid:
            self.render_features.unregister(feature)
            self.render_graph = self._build_render_graph()
            report.raise_for_errors()
        return feature

    def unregister_feature(self, feature: RenderFeature) -> bool:
        """Removes a render feature and rebuilds the validated graph."""
        removed = self.render_features.unregister(feature)
        if removed:
            try:
                feature.destroy()
            except Exception:
                pass
            self.render_graph = self._build_render_graph()
        return removed

    def add_point_light(
        self,
        position: tuple[float, float, float],
        radius: float = 10.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 2.0,
    ) -> PointLight:
        """Adds a dynamic local point light to the active frame."""
        return self.lights_pass.add_light(position, radius, color, intensity)

    def add_spot_light(
        self,
        position: tuple[float, float, float],
        direction: tuple[float, float, float],
        radius: float = 25.0,
        inner_cone_angle: float = math.radians(18.0),
        outer_cone_angle: float = math.radians(32.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 4.5,
        cast_shadow: bool = False,
        shadow_bias: float = 0.0015,
    ) -> SpotLight:
        """Adds a dynamic local spot light with cone falloff and optional perspective shadow casting."""
        return self.lights_pass.add_spot_light(
            position=position,
            direction=direction,
            radius=radius,
            inner_cone_angle=inner_cone_angle,
            outer_cone_angle=outer_cone_angle,
            color=color,
            intensity=intensity,
            cast_shadow=cast_shadow,
            shadow_bias=shadow_bias,
        )

    def clear_point_lights(self) -> None:
        """Clears all active dynamic point lights."""
        self.lights_pass.clear_point_lights()

    def clear_spot_lights(self) -> None:
        """Clears all active dynamic spot lights."""
        self.lights_pass.clear_spot_lights()

    def clear_lights(self) -> None:
        """Clears all active dynamic point and spot lights."""
        self.lights_pass.clear()

    def add_decal(
        self,
        position: tuple[float, float, float],
        rotation_quat: tuple[float, float, float, float] | np.ndarray,
        size: tuple[float, float, float],
        color: tuple[float, float, float] = (0.04, 0.04, 0.05),
        opacity: float = 0.85,
        roughness: float = 0.95,
        decal_type: int = 0,
    ) -> int:
        """Adds or updates an oriented bounding box decal into the deferred decal pass."""
        return self.decal_pass.add_decal(
            position=position,
            rotation_quat=rotation_quat,
            size=size,
            color=color,
            opacity=opacity,
            roughness=roughness,
            decal_type=decal_type,
        )

    def clear_decals(self) -> None:
        """Clears all active deferred decals."""
        self.decal_pass.clear()

    def apply_config(self, new_config: RenderConfig) -> None:
        """Applies dynamic graphics quality configuration changes."""
        self.config = new_config
        self.csm.resize_atlas(new_config.shadow_resolution)
        self.spot_shadow_map.resize_atlas(getattr(new_config, "spot_shadow_resolution", 2048))
        self.csm.update_splits(
            new_config.shadow_distance,
            new_config.csm_cascades,
            getattr(new_config, "csm_split_lambda", 0.85),
            getattr(new_config, "csm_cascade_distances", None),
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

        if self.ao_pass is not None:
            self.ao_pass.enabled = getattr(new_config, "ao_mode", "GTAO") != "OFF"
            self.ao_pass.set_resolution_scale(
                getattr(new_config, "ao_resolution_scale", 1.0),
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
            )
        if self.ssr_pass is not None:
            self.ssr_pass.enabled = getattr(new_config, "ssr_enabled", True)
            self.ssr_pass.resize(
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
                getattr(new_config, "ssr_resolution_scale", 1.0),
            )
        if self.ssgi_pass is not None:
            self.ssgi_pass.resize(
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
                getattr(new_config, "ssgi_resolution_scale", 1.0),
            )
        if self.rc_pass is not None:
            self.rc_pass.resize(
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
                getattr(new_config, "rc_resolution_scale", 1.0),
            )
        if self.camera_optics_pass is not None:
            self.camera_optics_pass.resize(
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
                getattr(new_config, "lens_flare_resolution_scale", 1.0),
            )
        if self.solar_flare_pass is not None:
            self.solar_flare_pass.resize(
                self.ctx_wrapper.width,
                self.ctx_wrapper.height,
                getattr(new_config, "lens_flare_resolution_scale", 1.0),
            )
        if self.volumetric_fog_pass is not None:
            self.volumetric_fog_pass.enabled = getattr(new_config, "volumetric_fog_enabled", True)

    def resize(self, width: int, height: int) -> None:
        """Resizes MRT G-Buffer and all post-processing/intermediate pass buffers."""
        self.ctx_wrapper.width = width
        self.ctx_wrapper.height = height
        self._graph_context.width = width
        self._graph_context.height = height
        self._screen_size_tuple = (float(width), float(height))

        self.g_buffer.resize(width, height)
        self.ssdm_pass.resize(width, height)
        self.decal_pass.resize(width, height)
        self.hiz_pass.resize(width, height)
        self.direct_radiance_pass.resize(width, height)
        self.ao_pass.resize(width, height)
        self.ssgi_pass.resize(
            width,
            height,
            getattr(self.config, "ssgi_resolution_scale", 1.0),
        )
        self.rc_pass.resize(
            width,
            height,
            getattr(self.config, "rc_resolution_scale", 1.0),
        )
        self.ssr_pass.resize(
            width,
            height,
            getattr(self.config, "ssr_resolution_scale", 1.0),
        )
        self.taa_pass.resize(width, height)
        self.fxaa_pass.resize(width, height)
        self.smaa_pass.resize(width, height)
        self.volumetric_fog_pass.resize(width, height)
        self.camera_optics_pass.resize(
            width,
            height,
            getattr(self.config, "lens_flare_resolution_scale", 1.0),
        )
        self.solar_flare_pass.resize(
            width,
            height,
            getattr(self.config, "lens_flare_resolution_scale", 1.0),
        )
        self.motion_blur_pass.resize(width, height)
        self.bloom_pass.resize(width, height)
        self.water_pass.resize(width, height)
        self.post_process.resize(width, height)
        self.gpu_profiler.resize(width, height)

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
        self.rebuild_vaos()
        return alloc

    def rebuild_vaos(self) -> None:
        """Rebuilds all pipeline VAOs bound to the MegaBuffer."""
        self.gbuffer_vao = self.mega_buffer.get_vao(self.gbuffer_prog)
        self.csm_vao = self.mega_buffer.get_vao(self.csm_prog)
        self.spot_depth_vao = self.mega_buffer.get_vao(self.spot_depth_prog)
        self.gbuffer_tess_vao = self.mega_buffer.get_vao(self.gbuffer_tess_prog, mode=self.ctx.PATCHES)
        self.csm_tess_vao = self.mega_buffer.get_vao(self.csm_tess_prog, mode=self.ctx.PATCHES)

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
        res = self.texture_atlas.load_materials_from_folder(textures_dir, progress_callback=progress_callback)
        self.material_registry.sync_with_atlas(self.texture_atlas)
        return res

    def _compute_sun_screen_position(self) -> tuple[tuple[float, float], bool]:
        """Projects the sun direction to UV space for the solar flare.

        Returns ``((u, v), on_screen)``. ``on_screen`` is False when the sun is
        behind the camera, in which case the flare must not draw.
        """
        sx = float(self._sun_v[0])
        sy = float(self._sun_v[1])
        sz = float(self._sun_v[2])
        # The sun sits at infinity along -sun_dir (sun_dir points into the scene).
        dx, dy, dz = -sx, -sy, -sz

        vp = self._vp_mat
        # Column-major mat4 * vec4: clip = VP * (dir, 0) for a direction at infinity.
        cx = vp[0] * dx + vp[4] * dy + vp[8] * dz
        cy = vp[1] * dx + vp[5] * dy + vp[9] * dz
        cw = vp[3] * dx + vp[7] * dy + vp[11] * dz

        if cw <= 1e-6:
            return (0.5, 0.5), False

        inv_w = 1.0 / cw
        u = (cx * inv_w) * 0.5 + 0.5
        v = (cy * inv_w) * 0.5 + 0.5
        on_screen = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
        return (u, v), on_screen

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

        profiler = self.gpu_profiler
        profiler.begin_frame()

        w, h = self.ctx_wrapper.width, self.ctx_wrapper.height
        aspect = w / (h if h > 0 else 1)
        if w != self.hiz_pass.width or h != self.hiz_pass.height:
            self.hiz_pass.resize(w, h)

        # 1. Update SSBO 1 & SSBO 2 from ECS contiguous memory tables
        # Phase 4: Static / Dynamic Entity Upload Optimization
        s_count = ecs.static_count
        valid_partition = s_count > 0 and s_count <= active_count and not np.any(~ecs.is_static[:s_count])
        if valid_partition:
            if ecs.is_static_dirty:
                self.ssbo_transforms.write(ecs.get_static_transforms_view().tobytes(), offset=0)
                self.ssbo_materials.write(ecs.get_static_materials_view().tobytes(), offset=0)
                ecs.clear_static_dirty()

            # Upload dynamic entity slice every frame
            if active_count > s_count:
                dyn_offset_t = s_count * 64  # 16 floats * 4 bytes
                dyn_offset_m = s_count * 32  # 8 floats * 4 bytes
                self.ssbo_transforms.write(ecs.get_dynamic_transforms_view().tobytes(), offset=dyn_offset_t)
                self.ssbo_materials.write(ecs.get_dynamic_materials_view().tobytes(), offset=dyn_offset_m)
        else:
            # Full upload: guarantees all dynamic entities are refreshed even after map rebuilding
            self.ssbo_transforms.write(ecs.get_active_transforms_view().tobytes(), offset=0)
            self.ssbo_materials.write(ecs.get_active_materials_view().tobytes(), offset=0)
            ecs.clear_static_dirty()

        # 2. Camera Matrices & Subpixel TAA Jitter
        self._cam_pos[0] = float(camera_pos[0])
        self._cam_pos[1] = float(camera_pos[1])
        self._cam_pos[2] = float(camera_pos[2])

        matrix_look_at(camera_pos, camera_target, up=(0.0, 1.0, 0.0), out=self._view_mat)

        # Synchronize CSM splits with the active cascade distances and count.
        cascade_distances = getattr(self.config, "csm_cascade_distances", None)
        if (
            self.csm.cascade_count != self.config.csm_cascades
            or self.csm.split_lambda != getattr(self.config, "csm_split_lambda", 0.85)
            or self.csm.cascade_distances != cascade_distances
            or (cascade_distances is None and self.csm.max_distance != self.config.shadow_distance)
        ):
            self.csm.update_splits(
                self.config.shadow_distance,
                self.config.csm_cascades,
                getattr(self.config, "csm_split_lambda", 0.85),
                cascade_distances,
            )

        matrix_perspective(
            math.radians(fovy_deg),
            aspect,
            near=0.1,
            far=max(2000.0, self.config.shadow_distance * 3.0),
            reverse_z=self.config.reverse_z,
            out=self._proj_mat,
        )

        np.copyto(self._unjittered_proj_mat, self._proj_mat)

        jitter_x, jitter_y = 0.0, 0.0
        aa_mode = getattr(self.config, "aa_mode", "OFF")
        if aa_mode == "TAA" or getattr(self.config, "taa_enabled", False):
            jitter_x, jitter_y = self.taa_pass.get_jitter(w, h)
        elif aa_mode in ("SMAA_2X", "SMAA_4X"):
            jitter_x, jitter_y = self.smaa_pass.get_jitter(w, h, aa_mode)

        if jitter_x != 0.0 or jitter_y != 0.0:
            self._proj_mat[8] += jitter_x * 2.0
            self._proj_mat[9] += jitter_y * 2.0

        mat4_mul(self._proj_mat, self._view_mat, out=self._vp_mat)
        mat4_mul(self._unjittered_proj_mat, self._view_mat, out=self._unjittered_vp_mat)
        mat4_inv(self._proj_mat, out=self._inv_proj)
        mat4_inv(self._view_mat, out=self._inv_view)

        fx = float(camera_target[0] - camera_pos[0])
        fy = float(camera_target[1] - camera_pos[1])
        fz = float(camera_target[2] - camera_pos[2])
        inv_fwd_len = 1.0 / (math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-6)
        self._cam_fwd[0] = fx * inv_fwd_len
        self._cam_fwd[1] = fy * inv_fwd_len
        self._cam_fwd[2] = fz * inv_fwd_len

        # 3. Advance environment simulation and synchronise lighting
        if dt is None:
            calc_dt = max(0.0, min(0.1, time_elapsed - self._prev_time)) if self._prev_time > 0.0 else 0.016
        else:
            calc_dt = dt
        self._prev_time = time_elapsed

        env = self.environment
        if calc_dt > 0.0:
            env.update(calc_dt)

        atmo_sun_dir = env.compute_sun_vector()
        atmo_dir, atmo_lux, atmo_col, atmo_ambient = env.compute_lighting()
        env_config = getattr(env, "config", None)
        day_speed = float(getattr(env_config, "day_speed", 0.0)) if env_config is not None else 0.0
        is_custom_sun = sun_dir != (0.35, -0.85, 0.40) and day_speed == 0.0
        true_sun_dir = sun_dir if is_custom_sun else atmo_sun_dir
        active_sun_dir = sun_dir if is_custom_sun else atmo_dir
        active_sun_lux = sun_lux if (sun_lux != 4.0 and is_custom_sun) else atmo_lux
        active_sun_col = atmo_col
        active_ambient = atmo_ambient

        self._sun_v[0] = float(active_sun_dir[0])
        self._sun_v[1] = float(active_sun_dir[1])
        self._sun_v[2] = float(active_sun_dir[2])
        ecs.sync_dynamic_aabbs()
        csm_matrices = self.csm.compute_cascade_matrices(
            self._cam_pos,
            self._cam_fwd,
            self._sun_v,
            fovy_deg=fovy_deg,
            aspect=aspect,
            aabbs=ecs.get_active_aabbs_view(),
            active_count=active_count,
        )

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

        # Prepare MDI batch commands (camera view, tessellated, and shadow passes)
        self.mdi.begin_frame()
        self.tess_mdi.begin_frame()
        self.shadow_mdi.begin_frame()
        if draw_batches is None:
            draw_batches = ecs.mesh_draw_batches

        # Phase 1: Per-Cascade Frustum Culling for Shadow MDI Commands
        for m in self.cascade_shadow_mdis:
            m.begin_frame()

        if draw_batches is not None:
            active_aabbs = ecs.get_active_aabbs_view()
            # Filter shadow caster batches per active cascade. The scratch list is
            # reused across frames to keep the inner loop allocation-free.
            caster_batches = self._caster_batches_scratch
            caster_batches.clear()
            for b in draw_batches:
                if b[1] > 0 and (not b[3] if len(b) >= 4 else True) and (b[4] if len(b) >= 5 else True):
                    caster_batches.append(b)
            culled_cascades = self.csm.cull_cascades(
                aabbs=active_aabbs,
                active_count=active_count,
                draw_batches=caster_batches,
                cascade_count=self.config.csm_cascades,
            )
            shadow_commands = 0
            for c_idx in range(self.config.csm_cascades):
                casc_mdi = self.cascade_shadow_mdis[c_idx]
                for item in culled_cascades[c_idx]:
                    mesh_item, count, base_inst = item[0], item[1], item[2]
                    alloc = self.mega_buffer.allocations[mesh_item] if isinstance(mesh_item, str) else mesh_item
                    casc_mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
                    shadow_commands += 1
            self.stats_shadow_commands = shadow_commands
            self.stats_shadow_commands_unculled = len(caster_batches) * self.config.csm_cascades

            # Spot shadow maps are rendered once per spot light, so they reuse a single
            # command list holding every shadow caster (no per-light frustum culling).
            if getattr(self.config, "spot_shadows_enabled", True):
                for item in caster_batches:
                    mesh_item, count, base_inst = item[0], item[1], item[2]
                    alloc = self.mega_buffer.allocations[mesh_item] if isinstance(mesh_item, str) else mesh_item
                    self.shadow_mdi.add_command(alloc, instance_count=count, base_instance=base_inst)

        # Phase 3: Culling for Camera MDI Draw Calls (Vectorized CPU Frustum + GPU Hi-Z Occlusion)
        frustum_cull_active = getattr(self.config, "frustum_cull_enabled", True)
        hiz_cull_active = getattr(self.config, "hiz_cull_enabled", False)

        if (frustum_cull_active or hiz_cull_active) and draw_batches:
            active_aabbs = ecs.get_active_aabbs_view()
            if hiz_cull_active and hasattr(self, "hiz_pass") and self.hiz_pass is not None:
                visible_mask = self.hiz_pass.cull_aabbs_gpu(
                    aabbs=active_aabbs,
                    active_count=active_count,
                    vp_matrix=self._vp_mat,
                    screen_size=(w, h),
                )
            else:
                self.culler.extract_planes(self._vp_mat, reverse_z=self.config.reverse_z)
                visible_mask = self.culler.cull_aabbs(active_aabbs, active_count)

            batches_to_render = self.culler.filter_draw_batches(draw_batches, visible_mask)
            self.stats_visible_entities = self.culler.stats_visible_entities
            self.stats_culled_entities = self.culler.stats_culled_entities
            self.stats_visible_commands = self.culler.stats_visible_commands
            self.stats_culled_commands = self.culler.stats_culled_commands
        else:
            batches_to_render = draw_batches
            self.stats_visible_entities = active_count
            self.stats_culled_entities = 0
            self.stats_visible_commands = len(draw_batches) if draw_batches else 0
            self.stats_culled_commands = 0

        if batches_to_render is not None:
            for item in batches_to_render:
                mesh_item = item[0]
                count = item[1]
                base_inst = item[2]
                is_tess = item[3] if len(item) >= 4 else False

                if count > 0:
                    alloc = self.mega_buffer.allocations[mesh_item] if isinstance(mesh_item, str) else mesh_item
                    if is_tess:
                        self.tess_mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
                    else:
                        self.mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
        # Setup Graph Context Blackboard
        ctx = self._graph_context
        ctx.camera_pos = (self._cam_pos[0], self._cam_pos[1], self._cam_pos[2])
        ctx.sun_dir = sun_dir
        ctx.sun_lux = sun_lux
        ctx.time_elapsed = time_elapsed
        ctx.resources["g_buffer"] = self.g_buffer
        ctx.resources["csm"] = self.csm

        # Pre-pass: Dynamic Clustered Local Point & Spot Lights (SSBO 3)
        self.lights_pass.execute(ctx)

        # ---- PASS 1: Cascaded Shadow Maps Pass ----
        self.csm.fbo.use()
        self.csm.clear()
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        # Push shadow casters away from the light by a depth-dependent amount.
        # This removes most self-shadow acne at the source, which lets the
        # receiver-side bias stay small enough that shadows stay welded to their
        # casters instead of peter-panning.
        self.ctx.polygon_offset = (1.5, 2.0)

        # Bind PBR texture array atlases to units 10..13 (essential for csm_tess_prog displacement sampling)
        self.texture_atlas.bind(10, 11, 12, 13)

        for c in range(self.config.csm_cascades):
            self.csm.begin_cascade(c)
            if self._csm_cascade_idx_uniform is not None:
                self._csm_cascade_idx_uniform.value = c
            # Submit only meshes that are within this specific cascade's bounds
            casc_mdi = self.cascade_shadow_mdis[c]
            if casc_mdi.command_count > 0:
                casc_mdi.submit(self.csm_vao, self.csm_prog)
            if getattr(self.config, "tess_enabled", True) and self.tess_mdi.command_count > 0:
                self.ctx.patch_vertices = 3
                if self._u_csm_tess_cascade_idx is not None:
                    self._u_csm_tess_cascade_idx.value = c
                if self._u_csm_tess_near_radius is not None:
                    self._u_csm_tess_near_radius.value = getattr(self.config, "disp_near_radius", 120.0)
                if self._u_csm_tess_mid_radius is not None:
                    self._u_csm_tess_mid_radius.value = getattr(self.config, "disp_mid_radius", 300.0)
                if self._u_csm_tess_max_level is not None:
                    self._u_csm_tess_max_level.value = getattr(self.config, "tess_max_level", 24.0)
                if self._u_csm_tess_med_level is not None:
                    self._u_csm_tess_med_level.value = getattr(self.config, "tess_med_level", 8.0)
                if self._u_csm_tess_disp_scale is not None:
                    self._u_csm_tess_disp_scale.value = getattr(self.config, "tess_displacement_scale", 1.0)
                self.texture_atlas.upload_depths(self.csm_tess_prog)
                self.tess_mdi.submit(self.csm_tess_vao, self.csm_tess_prog)

        # ---- PASS 1.5: Spot Light Shadow Maps Pass ----
        shadow_spots = ctx.resources.get("shadow_spot_lights", [])
        spot_shadows_active = getattr(self.config, "spot_shadows_enabled", True) and len(shadow_spots) > 0
        if spot_shadows_active:
            self.spot_shadow_map.clear()
            self.ctx.depth_func = "<"
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.polygon_offset = (1.5, 2.0)
            for idx, spot in enumerate(shadow_spots[:4]):
                vp = self.spot_shadow_map.compute_spot_matrix(
                    (spot.x, spot.y, spot.z),
                    (spot.dir_x, spot.dir_y, spot.dir_z),
                    spot.outer_cone_angle,
                    spot.radius,
                    spot_idx=idx,
                )
                self.spot_shadow_map.begin_spot(idx)
                if self._u_spot_depth_vp is not None:
                    self._u_spot_depth_vp.write(vp.tobytes())
                self.shadow_mdi.submit(self.spot_depth_vao, self.spot_depth_prog)

        # ---- PASS 2: G-Buffer Pass (Reversed-Z) ----
        self.g_buffer.clear()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.depth_func = ">" if self.config.reverse_z else "<"
        self.ctx.enable(moderngl.CULL_FACE)
        # Shadow passes above used a polygon offset; the G-Buffer must not.
        self.ctx.polygon_offset = (0.0, 0.0)

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
            self._u_gbuffer_disp_near_radius.value = getattr(self.config, "disp_near_radius", 120.0)
        if self._u_gbuffer_disp_mid_radius is not None:
            self._u_gbuffer_disp_mid_radius.value = getattr(self.config, "disp_mid_radius", 300.0)
        object_motion = 1 if getattr(self.config, "object_motion_vectors_enabled", True) else 0
        if self._u_gbuffer_object_motion is not None:
            self._u_gbuffer_object_motion.value = object_motion
        if self._u_gbuffer_tess_object_motion is not None:
            self._u_gbuffer_tess_object_motion.value = object_motion
        if self._u_specular_aa_enabled is not None:
            self._u_specular_aa_enabled.value = 1 if getattr(self.config, "specular_aa_enabled", True) else 0
        if self._u_specular_aa_strength is not None:
            self._u_specular_aa_strength.value = float(getattr(self.config, "specular_aa_strength", 1.0))
        if self._u_gbuffer_car_paint_layer is not None:
            self._u_gbuffer_car_paint_layer.value = float(self.texture_atlas.name_to_layer.get("car_body_paint", 2))
        if self._u_gbuffer_car_glass_layer is not None:
            self._u_gbuffer_car_glass_layer.value = float(self.texture_atlas.name_to_layer.get("car_window_glass", 8))
        if self._u_gbuffer_car_rim_layer is not None:
            self._u_gbuffer_car_rim_layer.value = float(self.texture_atlas.name_to_layer.get("car_alloy_rim", 1))
        if self._u_gbuffer_car_tire_layer is not None:
            self._u_gbuffer_car_tire_layer.value = float(self.texture_atlas.name_to_layer.get("car_tire_rubber", 7))
        if self._u_gbuffer_car_chrome_layer is not None:
            self._u_gbuffer_car_chrome_layer.value = float(self.texture_atlas.name_to_layer.get("car_chrome_trim", 3))
        if self._u_gbuffer_car_taillight_layer is not None:
            self._u_gbuffer_car_taillight_layer.value = float(
                self.texture_atlas.name_to_layer.get("car_taillight_red", 6)
            )
        if self._u_gbuffer_car_indicator_layer is not None:
            self._u_gbuffer_car_indicator_layer.value = float(
                self.texture_atlas.name_to_layer.get("car_indicator_amber", 4)
            )
        if self._u_gbuffer_car_interior_layer is not None:
            self._u_gbuffer_car_interior_layer.value = float(self.texture_atlas.name_to_layer.get("car_interior", 5))
        if self._u_gbuffer_brick_wall_layer is not None:
            self._u_gbuffer_brick_wall_layer.value = float(self.texture_atlas.name_to_layer.get("brick_wall", 0))
        if self._u_gbuffer_concrete_floor_layer is not None:
            self._u_gbuffer_concrete_floor_layer.value = float(
                self.texture_atlas.name_to_layer.get("concrete_floor", 0)
            )
        if self._u_gbuffer_tess_brick_wall_layer is not None:
            self._u_gbuffer_tess_brick_wall_layer.value = float(self.texture_atlas.name_to_layer.get("brick_wall", 0))
        if self._u_gbuffer_tess_concrete_floor_layer is not None:
            self._u_gbuffer_tess_concrete_floor_layer.value = float(
                self.texture_atlas.name_to_layer.get("concrete_floor", 0)
            )
        self.texture_atlas.upload_depths(self.gbuffer_prog)

        is_wireframe = getattr(self.config, "wireframe", False)
        if is_wireframe:
            self.ctx.wireframe = True
        profiler.begin("G-buffer")
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
                    self._u_tess_near_radius.value = getattr(self.config, "disp_near_radius", 120.0)
                if self._u_tess_mid_radius is not None:
                    self._u_tess_mid_radius.value = getattr(self.config, "disp_mid_radius", 300.0)
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
            profiler.end()
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
                disp_mid_radius=getattr(self.config, "disp_mid_radius", 300.0),
                material_depths=self.texture_atlas.material_depths,
            )

        # ---- PASS 2.6: Screen-Space Deferred Decals (Phase 9) ----
        self.decal_pass.execute(ctx, self.g_buffer)

        # ---- PASS 2.7: Hierarchical-Z (Hi-Z) Depth Pyramid Pass ----
        with profiler.region("Hi-Z"):
            self.hiz_pass.execute(ctx)

        # ---- PASS 2.8: Current-Frame Direct Radiance (GI source) ----
        # Runs after shadows and Hi-Z so SSGI samples this frame's direct
        # diffuse + emissive instead of a stale composited HDR frame.
        ctx.resources["spot_shadow_map"] = self.spot_shadow_map
        with profiler.region("Direct radiance"):
            self.direct_radiance_pass.execute(ctx)

        # ---- PASS 3: Ambient Occlusion Pass (GTAO / SSAO) ----
        with profiler.region("AO"):
            self.ao_pass.execute(ctx)

        # ---- PASS 4: Dynamic Local Lights (SSBO 3 already updated at pre-pass) ----
        # lights_pass.execute(ctx) executed prior to shadow passes to allocate spot shadow casters
        # ---- PASS 5: Global Illumination (Mutually Exclusive: SSGI vs Radiance Cascades) ----
        ctx.resources["scene_color"] = (
            getattr(self.post_process, "hdr_texture", None) or self.g_buffer.albedo_roughness_texture
        )
        ctx.resources["sky_view_lut"] = env.sky_view_lut
        if self.config.gi_mode == "RADIANCE_CASCADES":
            ctx.resources["prev_view"] = self._prev_view_mat
            self.ssgi_pass.invalidate_history()
            with profiler.region("RC total"):
                self.rc_pass.execute(ctx)
        elif self.config.gi_mode == "SSGI":
            self.rc_pass.invalidate_history()
            ctx.resources["prev_view"] = self._prev_view_mat
            with profiler.region("SSGI total"):
                self.ssgi_pass.execute(ctx)
        else:
            self.rc_pass.invalidate_history()
            self.ssgi_pass.invalidate_history()

        # ---- PASS 6: Image-Based Lighting & Screen-Space Reflections (SSR) ----
        self.ibl_pass.execute(ctx)
        with profiler.region("SSR"):
            self.ssr_pass.execute(ctx)

        # ---- PASS 6.5: Environment Sky & Atmosphere (provider-owned) ----
        env.render_sky(
            camera_pos=(float(self._cam_pos[0]), float(self._cam_pos[1]), float(self._cam_pos[2])),
            sun_dir=true_sun_dir,
            sun_color=active_sun_col,
            sun_intensity=float(getattr(env_config, "sun_intensity", 4.0)) if env_config is not None else 4.0,
        )

        # ---- PASS 7: Consolidated Deferred Resolve ----
        self.post_process.hdr_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        # Bind all required texture units
        self.g_buffer.bind_textures(base_unit=0)  # 0: AlbedoRough, 1: NormalMetal, 2: Depth
        self.csm.depth_texture.use(location=3)  # 3: ShadowAtlas (raw depth)
        # 9: ShadowAtlas through a hardware depth-comparison sampler. Same
        # texture, different sampler state, so the filter gets bilinear PCF for
        # free while the blocker search keeps raw depth access.
        self.csm.compare_sampler.use(location=9)

        ao_tex = ctx.resources.get("ao_texture", self.ao_pass.white_fallback)
        ao_tex.use(location=4)

        ssgi_tex = ctx.resources.get("ssgi_texture", self.ssgi_pass.black_fallback)
        ssgi_tex.use(location=5)

        self.ibl_pass.lut_tex.use(location=6)
        self.ibl_pass.env_tex.use(location=7)

        ssr_tex = ctx.resources.get("ssr_texture", self.ssr_pass.black_fallback)
        ssr_tex.use(location=8)

        env.bind_luts(10, 11)
        self.spot_shadow_map.depth_texture.use(location=12)
        self.spot_shadow_map.compare_sampler.use(location=16)
        self.ibl_pass.irradiance_tex.use(location=13)
        self.hiz_pass.hiz_texture.use(location=14)

        rc_tex = ctx.resources.get("rc_texture", self.rc_pass.black_fallback)
        rc_tex.use(location=15)

        # RG32F linear-view-depth bounds hierarchy (debug visualization only).
        hiz_bounds_tex = ctx.resources.get("hiz_bounds_texture", None)
        if hiz_bounds_tex is not None:
            hiz_bounds_tex.use(location=17)

        # Current-frame direct radiance (debug visualization).
        direct_radiance_tex = ctx.resources.get("direct_radiance_texture", None)
        if direct_radiance_tex is not None:
            direct_radiance_tex.use(location=18)

        # Set resolve uniforms
        if self._u_shadow_bias is not None:
            self._u_shadow_bias.value = float(self.config.shadow_bias)
        if self._u_shadow_normal_bias is not None:
            self._u_shadow_normal_bias.value = float(getattr(self.config, "shadow_normal_bias", 0.0010))
        if self._u_shadow_bias_slope is not None:
            self._u_shadow_bias_slope.value = float(getattr(self.config, "shadow_bias_slope_scale", 2.0))
        if self._u_shadow_bias_min is not None:
            self._u_shadow_bias_min.value = float(getattr(self.config, "shadow_bias_min_scale", 0.5))
        if self._u_shadow_bias_max is not None:
            self._u_shadow_bias_max.value = float(getattr(self.config, "shadow_bias_max_scale", 6.0))
        if self._u_shadow_filter_mode is not None:
            mode = str(getattr(self.config, "shadow_filter_mode", "PCF")).upper()
            self._u_shadow_filter_mode.value = {"HARD": 0, "PCF": 1, "PCSS": 2}.get(mode, 1)
        if self._u_pcf_cascade_limit is not None:
            self._u_pcf_cascade_limit.value = int(getattr(self.config, "pcf_cascade_limit", 3))
        if self._u_pcf_samples is not None:
            self._u_pcf_samples.value = int(getattr(self.config, "pcf_samples", 16))
        if self._u_pcf_radius is not None:
            self._u_pcf_radius.value = float(getattr(self.config, "pcf_radius", 1.5))
        if self._u_pcf_falloff is not None:
            self._u_pcf_falloff.value = float(getattr(self.config, "pcf_falloff", 2.0))
        if self._u_pcss_cascade_limit is not None:
            self._u_pcss_cascade_limit.value = int(getattr(self.config, "pcss_cascade_limit", 1))
        if self._u_pcss_search_samples is not None:
            self._u_pcss_search_samples.value = int(getattr(self.config, "pcss_search_samples", 16))
        if self._u_pcss_search_radius is not None:
            self._u_pcss_search_radius.value = float(getattr(self.config, "pcss_search_radius", 8.0))
        if self._u_pcss_light_size is not None:
            self._u_pcss_light_size.value = float(getattr(self.config, "pcss_light_size", 0.02))
        if self._u_pcss_samples is not None:
            self._u_pcss_samples.value = int(getattr(self.config, "pcss_samples", 24))
        if self._u_pcss_falloff is not None:
            self._u_pcss_falloff.value = float(getattr(self.config, "pcss_falloff", 2.0))
        if self._u_pcss_max_radius is not None:
            self._u_pcss_max_radius.value = float(getattr(self.config, "pcss_max_radius", 96.0))
        if self._u_pcss_closest_weight is not None:
            self._u_pcss_closest_weight.value = float(getattr(self.config, "pcss_closest_blocker_weight", 1.0))
        if self._u_cascade_count is not None:
            self._u_cascade_count.value = self.config.csm_cascades
        if self._u_cascade_atlas_rect is not None:
            self._u_cascade_atlas_rect.write(self.csm.cascade_rect_uv_array.tobytes())
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
        elif self.config.gi_mode == "RADIANCE_CASCADES":
            gi_enum_val = 4
        if self._u_gi_enabled is not None:
            self._u_gi_enabled.value = gi_enum_val

        if self._u_ibl_enabled is not None:
            self._u_ibl_enabled.value = 1 if self.config.ibl_enabled else 0
        if self._u_ssr_enabled is not None:
            self._u_ssr_enabled.value = 1 if self.config.ssr_enabled else 0
        if self._u_energy_compensation_enabled is not None:
            self._u_energy_compensation_enabled.value = (
                1 if getattr(self.config, "energy_compensation_enabled", True) else 0
            )
        if self._u_point_light_count is not None:
            self._u_point_light_count.value = ctx.resources.get("point_light_count", 0)
        if self._u_spot_light_count is not None:
            self._u_spot_light_count.value = ctx.resources.get("spot_light_count", 0)
        if self._u_spot_shadow_enabled is not None:
            self._u_spot_shadow_enabled.value = 1 if getattr(self.config, "spot_shadows_enabled", True) else 0
        if self._u_spot_vp is not None:
            for i, mat in enumerate(self.spot_shadow_map.spot_matrices):
                self._spot_matrices_bytes[i * 64 : (i + 1) * 64] = mat.tobytes()
            self._u_spot_vp.write(self._spot_matrices_bytes)

        # Upload environment uniforms (provider-owned, internally cached)
        env.apply_resolve_uniforms(self.resolve_prog)
        env.apply_sun_direction(self.resolve_prog, true_sun_dir)

        profiler.begin("Deferred resolve")
        self.resolve_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        profiler.end()

        env.render_clouds(
            hdr_fbo=self.post_process.hdr_fbo,
            depth_texture=self.g_buffer.depth_texture,
            inverse_projection=self._inv_proj,
            inverse_view=self._inv_view,
            camera_position=(float(self._cam_pos[0]), float(self._cam_pos[1]), float(self._cam_pos[2])),
            sun_direction=true_sun_dir,
            sun_color=active_sun_col,
            time_elapsed=time_elapsed,
        )

        # ---- PASS 7.1: GPU Auto-Exposure (histogram + temporal adaptation) ----
        # Runs on the resolved HDR scene before any additive effect, so the
        # histogram measures scene radiance rather than lens flare or bloom.
        ctx.resources["hdr_color"] = self.post_process.hdr_fbo.color_attachments[0]
        ctx.delta_time = calc_dt
        self.exposure_pass.execute(ctx)

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
        if self.water_pass is not None:
            self.water_pass.render_water(
                hdr_fbo=self.post_process.hdr_fbo,
                depth_texture=self.g_buffer.depth_texture,
                time_elapsed=time_elapsed,
                ecs=ecs,
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
            optics_needed = getattr(self.config, "dof_enabled", False)
            if optics_needed:
                scene_hdr_tex = self.post_process.hdr_fbo.color_attachments[0]
                self.camera_optics_pass.render_optics(
                    scene_hdr_texture=scene_hdr_tex,
                    depth_texture=self.g_buffer.depth_texture,
                    config=self.config,
                )
                self.ctx.copy_framebuffer(self.post_process.hdr_fbo, self.camera_optics_pass.optics_fbo)

            flare_mode = str(getattr(self.config, "lens_flare_mode", "ANAMORPHIC")).upper()
            flare_enabled = getattr(self.config, "lens_flare_enabled", True) and flare_mode != "OFF"
            if flare_enabled:
                scene_hdr_tex = self.post_process.hdr_fbo.color_attachments[0]
                if flare_mode == "SOLAR" and self.solar_flare_pass is not None:
                    sun_uv, sun_visible = self._compute_sun_screen_position()
                    flare_tex = self.solar_flare_pass.render(
                        scene_hdr_texture=scene_hdr_tex,
                        depth_texture=self.g_buffer.depth_texture,
                        sun_screen_pos=sun_uv,
                        sun_on_screen=sun_visible,
                        config=self.config,
                    )
                else:
                    flare_tex = self.camera_optics_pass.render_flare(
                        scene_hdr_texture=scene_hdr_tex,
                        config=self.config,
                    )

        # ---- PASS 7.9: Temporal Anti-Aliasing (TAA in linear HDR space before Tonemapping) ----
        aa_mode = getattr(self.config, "aa_mode", "OFF")
        taa_active = aa_mode == "TAA" or getattr(self.config, "taa_enabled", False)
        if taa_active:
            ctx.resources["hdr_color"] = self.post_process.hdr_fbo.color_attachments[0]
            ctx.resources["g_buffer"] = self.g_buffer
            ctx.resources["prev_unjittered_vp"] = self._prev_vp_mat
            self.taa_pass.execute(ctx)
            self.ctx.copy_framebuffer(self.post_process.hdr_fbo, self.taa_pass.output_fbo)

        # ---- PASS 7.95: Velocity-Buffer Motion Blur (after TAA, before bloom) ----
        if getattr(self.config, "motion_blur_enabled", False) and self.motion_blur_pass is not None:
            ctx.resources["hdr_color"] = self.post_process.hdr_fbo.color_attachments[0]
            ctx.resources["g_buffer"] = self.g_buffer
            self.motion_blur_pass.execute(ctx)
            self.ctx.copy_framebuffer(self.post_process.hdr_fbo, self.motion_blur_pass.output_fbo)

        bloom_tex = None
        if self.config.bloom_enabled:
            bloom_tex = self.bloom_pass.render(
                self.post_process.hdr_fbo.color_attachments[0],
                threshold=self.config.bloom_threshold,
                iterations=self.config.bloom_iterations,
            )

        # ---- PASS 8: Post-Process & Mutually Exclusive Anti-Aliasing ----
        self.ctx.disable(moderngl.DEPTH_TEST)
        aa_mode = getattr(self.config, "aa_mode", "OFF")

        self.post_process.render(
            target_fbo=self.post_process.final_fbo,
            flare_texture=flare_tex,
            bloom_texture=bloom_tex,
            time_elapsed=time_elapsed,
            exposure_ev=ctx.resources.get("exposure_ev"),
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
                self.ctx.copy_framebuffer(self.post_process.final_fbo, self.smaa_pass.output_fbo)

        if not self.ctx_wrapper.is_headless:
            self.ctx.copy_framebuffer(self.ctx.screen, self.post_process.final_fbo)

        # Store current unjittered ViewProjection for next frame's motion vectors (zero-allocation)
        np.copyto(self._prev_vp_mat, self._unjittered_vp_mat)
        np.copyto(self._prev_view_mat, self._view_mat)

        # Snapshot this frame's transforms as next frame's "previous" transforms.
        # Only the dynamic slice can change, so static entities are copied once
        # (on the first frame) and never again.
        if getattr(self.config, "object_motion_vectors_enabled", True):
            if not self._prev_transforms_seeded:
                self.ssbo_prev_transforms.write(ecs.get_active_transforms_view().tobytes(), offset=0)
                self._prev_transforms_seeded = True
            elif active_count > s_count:
                dyn_offset_t = s_count * 64
                self.ssbo_prev_transforms.write(ecs.get_dynamic_transforms_view().tobytes(), offset=dyn_offset_t)

        # Clear dynamic point lights for the next frame
        self.lights_pass.clear()

        profiler.end_frame()

    @property
    def output_texture_id(self) -> int:
        """Exposes OpenGL texture handle for ImGui viewport dock."""
        return self.post_process.output_texture_id

    def destroy(self) -> None:
        unsubscribe_event(WindowResizeEvent, self._on_window_resize)
        self.gpu_profiler.destroy()
        self.environment.destroy()
        self.debug.destroy()
        self.resources.close()
        self.frame_context.destroy()
        self.mega_buffer.destroy()
        self.mdi.destroy()
        self.shadow_mdi.destroy()
        self.g_buffer.destroy()
        self.csm.destroy()
        self.ibl_pass.destroy()
        self.ao_pass.destroy()
        self.ssgi_pass.destroy()
        self.rc_pass.destroy()
        self.lights_pass.destroy()
        self.ssr_pass.destroy()
        self.taa_pass.destroy()
        self.fxaa_pass.destroy()
        self.smaa_pass.destroy()
        self.volumetric_fog_pass.destroy()
        self.exposure_pass.destroy()
        if self.particle_pass is not None:
            self.particle_pass.destroy()
        if self.camera_optics_pass is not None:
            self.camera_optics_pass.destroy()
        if self.solar_flare_pass is not None:
            self.solar_flare_pass.destroy()
        if self.motion_blur_pass is not None:
            self.motion_blur_pass.destroy()
        self.bloom_pass.destroy()
        self.post_process.destroy()
        self.ssbo_transforms.release()
        self.ssbo_materials.release()
        self.ssbo_prev_transforms.release()
        self.gbuffer_prog.release()
        self.csm_prog.release()
        self.resolve_prog.release()
        self.gbuffer_vao.release()
        self.csm_vao.release()
        self.texture_atlas.destroy()
        self.ssdm_pass.destroy()
        self.decal_pass.destroy()
        self.hiz_pass.destroy()
        self.direct_radiance_pass.destroy()
        self.tess_mdi.destroy()
        self.gbuffer_tess_prog.release()
        self.gbuffer_tess_vao.release()
        self.csm_tess_prog.release()
        self.csm_tess_vao.release()
        self.resolve_vao.release()
