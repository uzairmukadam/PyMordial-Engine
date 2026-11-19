"""Current-frame direct radiance pass for PyMordial Engine.

Produces the GI source radiance texture consumed by SSGI (and, later, Radiance
Cascades). It exists so screen-space GI never samples a stale or previously
composited HDR frame.

Texture contract (``direct_radiance_texture``):

    rgb = directional diffuse (with CSM shadows)
        + clustered point-light diffuse
        + clustered spot-light diffuse (with spot shadows)
        + emissive
    a   = 1.0

Explicitly excluded: indirect diffuse (SSGI/RC/IBL irradiance), all specular
terms (direct specular, SSR, environment specular), and any previous-frame data.
Because indirect and specular are absent, sampling this texture from SSGI cannot
create a recursive feedback loop through the HDR history.
"""

from __future__ import annotations

import moderngl
import numpy as np

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class DirectRadiancePass(RenderPass):
    """Renders current-frame direct diffuse + emissive radiance."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="DirectRadiancePass", enabled=True)
        self.ctx = ctx
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)

        self._create_targets()

        quad_vert = load_shader("fullscreen_quad.vert")
        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=load_shader("direct_radiance.frag"))
        self.vao = self.ctx.vertex_array(self.prog, [])

        # Shadow uniforms are shared with the deferred resolve; the pipeline
        # writes the same values into both programs.
        self._u_shadow_bias = self.prog.get("u_ShadowBias", None)
        self._u_shadow_normal_bias = self.prog.get("u_ShadowNormalBias", None)
        self._u_shadow_bias_slope = self.prog.get("u_ShadowBiasSlopeScale", None)
        self._u_shadow_bias_min = self.prog.get("u_ShadowBiasMinScale", None)
        self._u_shadow_bias_max = self.prog.get("u_ShadowBiasMaxScale", None)
        self._u_shadow_filter_mode = self.prog.get("u_ShadowFilterMode", None)
        self._u_pcf_cascade_limit = self.prog.get("u_PCFCascadeLimit", None)
        self._u_pcf_samples = self.prog.get("u_PCFSamples", None)
        self._u_pcf_radius = self.prog.get("u_PCFRadius", None)
        self._u_pcf_falloff = self.prog.get("u_PCFFalloff", None)
        self._u_pcss_cascade_limit = self.prog.get("u_PCSS_CascadeLimit", None)
        self._u_pcss_search_samples = self.prog.get("u_PCSSSearchSamples", None)
        self._u_pcss_search_radius = self.prog.get("u_PCSSSearchRadius", None)
        self._u_pcss_light_size = self.prog.get("u_PCSSLightSize", None)
        self._u_pcss_samples = self.prog.get("u_PCSSSamples", None)
        self._u_pcss_falloff = self.prog.get("u_PCSSFalloff", None)
        self._u_pcss_max_radius = self.prog.get("u_PCSSMaxRadius", None)
        self._u_pcss_closest_weight = self.prog.get("u_PCSSClosestBlockerWeight", None)
        self._u_sun_angular_radius = self.prog.get("u_SunAngularRadius", None)
        self._u_cascade_count = self.prog.get("u_CascadeCount", None)
        self._u_cascade_atlas_rect = self.prog.get("u_CascadeAtlasRect", None)
        self._u_point_light_count = self.prog.get("u_PointLightCount", None)
        self._u_spot_light_count = self.prog.get("u_SpotLightCount", None)
        self._u_spot_shadow_enabled = self.prog.get("u_SpotShadowEnabled", None)
        self._u_spot_vp = self.prog.get("u_SpotLightViewProjection", None)

    def _create_targets(self) -> None:
        self.radiance_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.radiance_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.radiance_tex.repeat_x = False
        self.radiance_tex.repeat_y = False
        self.radiance_fbo = self.ctx.framebuffer(color_attachments=[self.radiance_tex])

    def resize(self, width: int, height: int) -> None:
        new_w = max(int(width), 1)
        new_h = max(int(height), 1)
        if new_w == self.width and new_h == self.height:
            return
        self.width = new_w
        self.height = new_h
        self.radiance_fbo.release()
        self.radiance_tex.release()
        self._create_targets()

    def execute(self, context: RenderGraphContext) -> None:
        g_buffer = context.resources.get("g_buffer")
        csm = context.resources.get("csm")
        if g_buffer is None or csm is None:
            context.resources["direct_radiance_texture"] = None
            return

        self.radiance_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        g_buffer.bind_textures(base_unit=0)  # 0: AlbedoRough, 1: NormalMetal, 2: Depth
        csm.depth_texture.use(location=3)
        csm.compare_sampler.use(location=9)
        context.resources["spot_shadow_map"].depth_texture.use(location=12)
        context.resources["spot_shadow_map"].compare_sampler.use(location=16)

        config = context.config
        if self._u_shadow_bias is not None:
            self._u_shadow_bias.value = float(config.shadow_bias)
        if self._u_shadow_normal_bias is not None:
            self._u_shadow_normal_bias.value = float(getattr(config, "shadow_normal_bias", 0.0010))
        if self._u_shadow_bias_slope is not None:
            self._u_shadow_bias_slope.value = float(getattr(config, "shadow_bias_slope_scale", 2.0))
        if self._u_shadow_bias_min is not None:
            self._u_shadow_bias_min.value = float(getattr(config, "shadow_bias_min_scale", 0.5))
        if self._u_shadow_bias_max is not None:
            self._u_shadow_bias_max.value = float(getattr(config, "shadow_bias_max_scale", 6.0))
        if self._u_shadow_filter_mode is not None:
            mode = str(getattr(config, "shadow_filter_mode", "PCF")).upper()
            self._u_shadow_filter_mode.value = {"HARD": 0, "PCF": 1, "PCSS": 2}.get(mode, 1)
        if self._u_pcf_cascade_limit is not None:
            self._u_pcf_cascade_limit.value = int(getattr(config, "pcf_cascade_limit", 3))
        if self._u_pcf_samples is not None:
            self._u_pcf_samples.value = int(getattr(config, "pcf_samples", 16))
        if self._u_pcf_radius is not None:
            self._u_pcf_radius.value = float(getattr(config, "pcf_radius", 1.5))
        if self._u_pcf_falloff is not None:
            self._u_pcf_falloff.value = float(getattr(config, "pcf_falloff", 2.0))
        if self._u_pcss_cascade_limit is not None:
            self._u_pcss_cascade_limit.value = int(getattr(config, "pcss_cascade_limit", 1))
        if self._u_pcss_search_samples is not None:
            self._u_pcss_search_samples.value = int(getattr(config, "pcss_search_samples", 16))
        if self._u_pcss_search_radius is not None:
            self._u_pcss_search_radius.value = float(getattr(config, "pcss_search_radius", 8.0))
        if self._u_pcss_light_size is not None:
            self._u_pcss_light_size.value = float(getattr(config, "pcss_light_size", 0.02))
        if self._u_pcss_samples is not None:
            self._u_pcss_samples.value = int(getattr(config, "pcss_samples", 24))
        if self._u_pcss_falloff is not None:
            self._u_pcss_falloff.value = float(getattr(config, "pcss_falloff", 2.0))
        if self._u_pcss_max_radius is not None:
            self._u_pcss_max_radius.value = float(getattr(config, "pcss_max_radius", 96.0))
        if self._u_pcss_closest_weight is not None:
            self._u_pcss_closest_weight.value = float(getattr(config, "pcss_closest_blocker_weight", 1.0))
        if self._u_sun_angular_radius is not None:
            self._u_sun_angular_radius.value = float(getattr(config, "sun_angular_radius", 0.0))
        if self._u_cascade_count is not None:
            self._u_cascade_count.value = config.csm_cascades
        if self._u_cascade_atlas_rect is not None:
            self._u_cascade_atlas_rect.write(csm.cascade_rect_uv_array.tobytes())
        if self._u_point_light_count is not None:
            self._u_point_light_count.value = context.resources.get("point_light_count", 0)
        if self._u_spot_light_count is not None:
            self._u_spot_light_count.value = context.resources.get("spot_light_count", 0)
        if self._u_spot_shadow_enabled is not None:
            self._u_spot_shadow_enabled.value = 1 if getattr(config, "spot_shadows_enabled", True) else 0
        if self._u_spot_vp is not None:
            spot_map = context.resources["spot_shadow_map"]
            self._u_spot_vp.write(np.asarray(spot_map.spot_matrices, dtype=np.float32).tobytes())
        self.vao.render(moderngl.TRIANGLES, vertices=3)

        context.resources["direct_radiance_texture"] = self.radiance_tex

    def destroy(self) -> None:
        self.radiance_fbo.release()
        self.radiance_tex.release()
        self.vao.release()
        self.prog.release()
