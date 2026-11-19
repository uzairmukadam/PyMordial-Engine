"""Screen-Space Global Illumination (SSGI) Pass for PyMordial Engine.

Executes cosine-weighted raymarching over the G-Buffer to compute near-field
diffuse indirect bounces and contact color bleeding, followed by normal- and
depth-aware edge-preserving filtering.
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
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(width * self.resolution_scale)), 1)
        self.height = max(int(round(height * self.resolution_scale)), 1)

        # RGBA16F ping-pong buffers at the configured render scale
        self.raw_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssgi_tex.repeat_x = False
        self.raw_ssgi_tex.repeat_y = False

        self.blur_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssgi_tex.repeat_x = False
        self.blur_ssgi_tex.repeat_y = False

        # Temporal history. Holds the previous frame's resolved result, with the
        # view depth in alpha so the next frame can validate the reprojection.
        self.history_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.history_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.history_ssgi_tex.repeat_x = False
        self.history_ssgi_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssgi_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssgi_tex])
        self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_ssgi_tex])

        # 1x1 black fallback
        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # Compile Shaders
        quad_vert = load_shader("fullscreen_quad.vert")
        ssgi_frag = load_shader("ssgi.frag")
        blur_frag = load_shader("ssgi_blur.frag")

        self.ssgi_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssgi_frag)
        self.blur_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=blur_frag)

        self.ssgi_vao = self.ctx.vertex_array(self.ssgi_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])

        # Uniform handles
        self._u_steps = self.ssgi_prog.get("u_SSGI_Steps", None)
        self._u_rays = self.ssgi_prog.get("u_SSGI_RayCount", None)
        self._u_dist = self.ssgi_prog.get("u_SSGI_RayDistance", None)
        self._u_thickness = self.ssgi_prog.get("u_SSGI_Thickness", None)
        self._u_hiz_mips = self.ssgi_prog.get("u_HiZMipCount", None)
        self._u_intensity = self.ssgi_prog.get("u_SSGI_Intensity", None)
        self._u_blur_dir = self.blur_prog.get("u_BlurDirection", None)
        self._u_target_res = self.blur_prog.get("u_TargetResolution", None)
        self._u_prev_view = self.blur_prog.get("u_PrevView", None)
        self._u_temporal_blend = self.blur_prog.get("u_TemporalBlend", None)
        self._u_history_valid = self.blur_prog.get("u_HistoryValid", None)
        self._u_temporal_enabled = self.blur_prog.get("u_TemporalEnabled", None)
        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)
        self._target_res_tuple = (float(self.width), float(self.height))
        self._first_frame = True

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

        self.raw_fbo.release()
        self.blur_fbo.release()
        self.history_fbo.release()
        self.raw_ssgi_tex.release()
        self.blur_ssgi_tex.release()
        self.history_ssgi_tex.release()

        self.raw_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssgi_tex.repeat_x = False
        self.raw_ssgi_tex.repeat_y = False

        self.blur_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssgi_tex.repeat_x = False
        self.blur_ssgi_tex.repeat_y = False

        self.history_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.history_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.history_ssgi_tex.repeat_x = False
        self.history_ssgi_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssgi_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssgi_tex])
        self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_ssgi_tex])

        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)
        self._target_res_tuple = (float(self.width), float(self.height))
        # A resized target invalidates the history: the old samples no longer
        # correspond to the same pixels.
        self._first_frame = True

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "SSGI")
        if gi_mode != "SSGI":
            context.resources["ssgi_texture"] = self.black_fallback
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            context.resources["ssgi_texture"] = self.black_fallback
            return

        # 1. Execute SSGI Raymarch
        self.raw_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)
        scene_color = context.resources.get("scene_color", g_buffer.albedo_roughness_texture)
        scene_color.use(location=3)

        # Hi-Z pyramid (unit 4) for empty-space skipping. Falls back to the depth
        # texture with a mip count of 1, which disables the skip in the shader.
        hiz_tex = context.resources.get("hiz_texture", None)
        hiz_mips = int(context.resources.get("hiz_mip_count", 1))
        if hiz_tex is not None:
            hiz_tex.use(location=4)
        else:
            g_buffer.depth_texture.use(location=4)
            hiz_mips = 1
        if self._u_hiz_mips is not None:
            self._u_hiz_mips.value = hiz_mips

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

        # Temporal state. The history is only valid if the previous frame wrote
        # one at the same resolution and the camera did not teleport.
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

        # 2. Horizontal Bilateral Blur + temporal resolve: raw -> blur_fbo
        self.blur_fbo.use()
        self.raw_ssgi_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        self.history_ssgi_tex.use(location=3)
        g_buffer.velocity_texture.use(location=4)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_h
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # 3. Vertical Bilateral Blur: blur -> raw_fbo
        self.raw_fbo.use()
        self.blur_ssgi_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        self.history_ssgi_tex.use(location=3)
        g_buffer.velocity_texture.use(location=4)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_v
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # 4. Copy the resolved result into the history for the next frame.
        self.ctx.copy_framebuffer(self.history_fbo, self.raw_fbo)
        self._first_frame = False

        context.resources["ssgi_texture"] = self.raw_ssgi_tex

    def destroy(self) -> None:
        self.raw_fbo.release()
        self.blur_fbo.release()
        self.history_fbo.release()
        self.raw_ssgi_tex.release()
        self.blur_ssgi_tex.release()
        self.history_ssgi_tex.release()
        self.black_fallback.release()
        self.ssgi_vao.release()
        self.blur_vao.release()
        self.ssgi_prog.release()
        self.blur_prog.release()
