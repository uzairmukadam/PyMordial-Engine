"""Screen-Space Global Illumination (SSGI) Pass for PyMordial Engine.

Executes cosine-weighted raymarching over the G-Buffer to compute near-field
diffuse indirect bounces and contact color bleeding.

Reconstruction is split into three independent responsibilities:

1. **Spatial filter** (``ssgi_blur.frag``) — run twice, horizontally then
   vertically. It never reads history or motion vectors.
2. **Temporal resolve** (``ssgi_temporal.frag``) — run exactly once per frame.
   It is the only stage that reads history and motion vectors.
3. **History update** — the resolved result is copied into the history target.

Resource roles:

    raw_ssgi_tex       ssgi_raw        current trace result
    spatial_x_tex      ssgi_spatial_x  horizontal filtered result
    spatial_y_tex      ssgi_spatial_y  vertical filtered result (current spatial)
    resolved_ssgi_tex  ssgi_resolved   final temporal result (published)
    history_ssgi_tex   ssgi_history    previous resolved result

History contract: ``rgb`` = resolved indirect radiance, ``a`` = positive linear
view depth of the pixel that produced ``rgb``.
"""

from __future__ import annotations

import moderngl

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class SSGIPass(RenderPass):
    """Computes screen-space indirect diffuse bounces and bilateral reconstruction."""

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        resolution_scale: float = 1.0,
    ) -> None:
        super().__init__(name="SSGIPass", enabled=True)
        self.ctx = ctx
        # Observable backend state: what was requested vs what actually ran.
        self.requested_backend = "LEGACY"
        self.active_backend = "LEGACY"
        self.fallback_reason: str | None = None
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(width * self.resolution_scale)), 1)
        self.height = max(int(round(height * self.resolution_scale)), 1)

        self._create_targets()

        # 1x1 black fallback
        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # Compile Shaders
        quad_vert = load_shader("fullscreen_quad.vert")
        self.ssgi_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=load_shader("ssgi.frag"))
        self.blur_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=load_shader("ssgi_blur.frag"))
        self.temporal_prog = self.ctx.program(
            vertex_shader=quad_vert, fragment_shader=load_shader("ssgi_temporal.frag")
        )

        self.ssgi_vao = self.ctx.vertex_array(self.ssgi_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])
        self.temporal_vao = self.ctx.vertex_array(self.temporal_prog, [])

        # Uniform handles — trace
        self._u_steps = self.ssgi_prog.get("u_SSGI_Steps", None)
        self._u_rays = self.ssgi_prog.get("u_SSGI_RayCount", None)
        self._u_dist = self.ssgi_prog.get("u_SSGI_RayDistance", None)
        self._u_thickness = self.ssgi_prog.get("u_SSGI_Thickness", None)
        self._u_intensity = self.ssgi_prog.get("u_SSGI_Intensity", None)

        # Uniform handles — spatial filter
        self._u_blur_dir = self.blur_prog.get("u_BlurDirection", None)

        # Uniform handles — temporal resolve
        self._u_target_res = self.temporal_prog.get("u_TargetResolution", None)
        self._u_prev_view = self.temporal_prog.get("u_PrevView", None)
        self._u_prev_jitter = self.temporal_prog.get("u_PrevJitter", None)
        self._u_temporal_blend = self.temporal_prog.get("u_TemporalBlend", None)
        self._u_history_valid = self.temporal_prog.get("u_HistoryValid", None)
        self._u_temporal_enabled = self.temporal_prog.get("u_TemporalEnabled", None)

        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)
        self._target_res_tuple = (float(self.width), float(self.height))
        self._first_frame = True

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _create_targets(self) -> None:
        def _tex() -> moderngl.Texture:
            tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            return tex

        self.raw_ssgi_tex = _tex()
        self.spatial_x_tex = _tex()
        self.spatial_y_tex = _tex()
        self.resolved_ssgi_tex = _tex()
        self.history_ssgi_tex = _tex()

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssgi_tex])
        self.spatial_x_fbo = self.ctx.framebuffer(color_attachments=[self.spatial_x_tex])
        self.spatial_y_fbo = self.ctx.framebuffer(color_attachments=[self.spatial_y_tex])
        self.resolved_fbo = self.ctx.framebuffer(color_attachments=[self.resolved_ssgi_tex])
        self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_ssgi_tex])

    def _release_targets(self) -> None:
        for fbo in (
            self.raw_fbo,
            self.spatial_x_fbo,
            self.spatial_y_fbo,
            self.resolved_fbo,
            self.history_fbo,
        ):
            fbo.release()
        for tex in (
            self.raw_ssgi_tex,
            self.spatial_x_tex,
            self.spatial_y_tex,
            self.resolved_ssgi_tex,
            self.history_ssgi_tex,
        ):
            tex.release()

    def invalidate_history(self) -> None:
        """Forces the next frame to discard temporal history."""
        self._first_frame = True

    def resize(self, width: int, height: int, resolution_scale: float | None = None) -> None:
        new_scale = self.resolution_scale if resolution_scale is None else min(max(float(resolution_scale), 0.25), 1.0)
        new_w = max(int(round(width * new_scale)), 1)
        new_h = max(int(round(height * new_scale)), 1)
        if new_w == self.width and new_h == self.height and new_scale == self.resolution_scale:
            return

        self.resolution_scale = new_scale
        self.width = new_w
        self.height = new_h

        self._release_targets()
        self._create_targets()

        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)
        self._target_res_tuple = (float(self.width), float(self.height))
        # A resized target invalidates the history: the old samples no longer
        # correspond to the same pixels.
        self._first_frame = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "SSGI")
        self.requested_backend = "LEGACY"
        self.fallback_reason = None
        if gi_mode != "SSGI":
            self.active_backend = "OFF"
            context.resources["ssgi_texture"] = self.black_fallback
            # Leaving SSGI invalidates the history so a later switch back does
            # not blend against samples from an unrelated frame.
            self.invalidate_history()
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            self.active_backend = "OFF"
            self.fallback_reason = "no G-buffer"
            context.resources["ssgi_texture"] = self.black_fallback
            self.invalidate_history()
            return

        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        # 1. Trace -> raw_ssgi_tex
        self._trace(context, g_buffer)

        # 2. Horizontal spatial filter: raw -> spatial_x
        self._spatial_filter(g_buffer, self.raw_ssgi_tex, self.spatial_x_fbo, self._blur_dir_h)

        # 3. Vertical spatial filter: spatial_x -> spatial_y
        self._spatial_filter(g_buffer, self.spatial_x_tex, self.spatial_y_fbo, self._blur_dir_v)

        # 4. Temporal resolve: spatial_y + history -> resolved
        self._temporal_resolve(context, g_buffer)

        # 5. Publish the resolved result.
        context.resources["ssgi_texture"] = self.resolved_ssgi_tex

        # 6. Update history for the next frame.
        self.ctx.copy_framebuffer(self.history_fbo, self.resolved_fbo)
        self._first_frame = False

    def _trace(self, context: RenderGraphContext, g_buffer) -> None:
        """Runs the fixed-step marcher into ``raw_ssgi_tex``."""
        self.raw_fbo.use()

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)

        # Current-frame direct radiance. Falls back to the G-Buffer albedo when
        # the direct-radiance pass has not published a texture yet.
        radiance_tex = context.resources.get("direct_radiance_texture", None)
        if radiance_tex is None:
            radiance_tex = g_buffer.albedo_roughness_texture
        radiance_tex.use(location=3)

        self.active_backend = "LEGACY"

        if self._u_steps is not None:
            self._u_steps.value = getattr(context.config, "ssgi_steps", 16)
        if self._u_rays is not None:
            self._u_rays.value = getattr(context.config, "ssgi_rays", 8)
        if self._u_intensity is not None:
            self._u_intensity.value = getattr(context.config, "ssgi_intensity", 1.5)
        if self._u_dist is not None:
            self._u_dist.value = getattr(context.config, "ssgi_ray_distance", 3.0)
        if self._u_thickness is not None:
            self._u_thickness.value = getattr(context.config, "ssgi_thickness", 0.35)

        self.ssgi_vao.render(moderngl.TRIANGLES, vertices=3)

    def _spatial_filter(self, g_buffer, source_tex, target_fbo, direction) -> None:
        """Runs one separable direction of the edge-aware spatial filter."""
        target_fbo.use()
        source_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = direction
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

    def _temporal_resolve(self, context: RenderGraphContext, g_buffer) -> None:
        """Runs the single per-frame temporal accumulation stage."""
        prev_view = context.resources.get("prev_view", None)
        history_valid = (not self._first_frame) and prev_view is not None
        temporal_enabled = bool(getattr(context.config, "ssgi_temporal_enabled", True))
        temporal_blend = float(getattr(context.config, "ssgi_temporal_blend", 0.90))

        if self._u_target_res is not None:
            self._u_target_res.value = self._target_res_tuple
        if self._u_temporal_blend is not None:
            self._u_temporal_blend.value = temporal_blend
        if self._u_history_valid is not None:
            self._u_history_valid.value = 1 if history_valid else 0
        if self._u_temporal_enabled is not None:
            self._u_temporal_enabled.value = 1 if temporal_enabled else 0
        if self._u_prev_view is not None and prev_view is not None:
            self._u_prev_view.write(prev_view.tobytes())
        if self._u_prev_jitter is not None:
            prev_jitter = context.resources.get("prev_jitter", (0.0, 0.0))
            self._u_prev_jitter.value = (float(prev_jitter[0]), float(prev_jitter[1]))

        self.resolved_fbo.use()
        self.spatial_y_tex.use(location=0)
        self.history_ssgi_tex.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.velocity_texture.use(location=3)
        self.temporal_vao.render(moderngl.TRIANGLES, vertices=3)

    def destroy(self) -> None:
        self._release_targets()
        self.black_fallback.release()
        self.ssgi_vao.release()
        self.blur_vao.release()
        self.temporal_vao.release()
        self.ssgi_prog.release()
        self.blur_prog.release()
        self.temporal_prog.release()
