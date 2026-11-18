"""Ambient Occlusion Pass with Bilateral Filter for PyMordial Engine.

Supports SSAO (Hemisphere sampling) and GTAO (Ground-Truth Ambient Occlusion)
with an edge-preserving metric view-depth bilateral filter.
"""

from __future__ import annotations

import moderngl

from engine.gfx.passes.blur_pass import BlurPass
from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class AmbientOcclusionPass(RenderPass):
    """Computes SSAO or GTAO and applies bilateral edge-preserving blur."""

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        resolution_scale: float = 1.0,
    ) -> None:
        super().__init__(name="AmbientOcclusionPass", enabled=True)
        self.ctx = ctx
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(width * self.resolution_scale)), 1)
        self.height = max(int(round(height * self.resolution_scale)), 1)

        # 1. R8 raw AO target; the shared BlurPass owns the ping-pong pair
        self.raw_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.raw_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ao_tex.repeat_x = False
        self.raw_ao_tex.repeat_y = False
        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ao_tex])

        self.blur_pass = BlurPass(ctx, self.width, self.height, shader="bilateral_blur.frag", components=1, dtype="f1")

        # 1x1 fallback white texture when AO is disabled
        self.white_fallback = self.ctx.texture((1, 1), 1, data=b"\xff", dtype="f1")

        # 2. Compile Shaders
        quad_vert = load_shader("fullscreen_quad.vert")
        ssao_frag = load_shader("ssao.frag")
        gtao_frag = load_shader("gtao.frag")

        self.ssao_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssao_frag)
        self.gtao_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=gtao_frag)

        self.ssao_vao = self.ctx.vertex_array(self.ssao_prog, [])
        self.gtao_vao = self.ctx.vertex_array(self.gtao_prog, [])

        # Pre-cache uniform handles
        self._u_ssao_radius = self.ssao_prog.get("u_Radius", None)
        self._u_ssao_intensity = self.ssao_prog.get("u_Intensity", None)

        self._u_gtao_radius = self.gtao_prog.get("u_Radius", None)
        self._u_gtao_intensity = self.gtao_prog.get("u_Intensity", None)

    @property
    def blur_ao_tex(self) -> moderngl.Texture:
        """Second ping-pong target holding the vertically blurred result."""
        return self.blur_pass.textures[1]

    @property
    def blur_fbo(self) -> moderngl.Framebuffer:
        """Framebuffer of the first ping-pong target (horizontal pass)."""
        return self.blur_pass.fbos[0]

    @property
    def blur_prog(self) -> moderngl.Program:
        """The shared bilateral blur program."""
        return self.blur_pass.prog

    @property
    def blur_vao(self) -> moderngl.VertexArray:
        """Fullscreen VAO bound to the shared bilateral blur program."""
        return self.blur_pass.vao

    def resize(self, width: int, height: int) -> None:
        new_w = max(int(round(width * self.resolution_scale)), 1)
        new_h = max(int(round(height * self.resolution_scale)), 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h

        self.raw_fbo.release()
        self.raw_ao_tex.release()

        self.raw_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.raw_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ao_tex.repeat_x = False
        self.raw_ao_tex.repeat_y = False
        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ao_tex])

        self.blur_pass.resize(self.width, self.height)

    def set_resolution_scale(self, scale: float, full_width: int, full_height: int) -> None:
        """Changes the AO render scale and reallocates the targets.

        The AO shader derives its screen-space radius from the *full* render
        resolution, so a downscaled AO target must still be told the full size
        or the radius clamp silently shrinks with the buffer.
        """
        new_scale = min(max(float(scale), 0.25), 1.0)
        if new_scale == self.resolution_scale:
            return
        self.resolution_scale = new_scale
        self.resize(full_width, full_height)

    def execute(self, context: RenderGraphContext) -> None:
        ao_mode = getattr(context.config, "ao_mode", "GTAO")
        if ao_mode == "OFF":
            context.resources["ao_texture"] = self.white_fallback
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            context.resources["ao_texture"] = self.white_fallback
            return

        radius = float(getattr(context.config, "ao_radius", 0.35))
        intensity = float(getattr(context.config, "ao_intensity", 1.0))

        # 1. Select AO program and VAO
        if ao_mode == "SSAO":
            vao = self.ssao_vao
            u_rad = self._u_ssao_radius
            u_int = self._u_ssao_intensity
        else:  # "GTAO"
            vao = self.gtao_vao
            u_rad = self._u_gtao_radius
            u_int = self._u_gtao_intensity

        # 2. Render Raw AO
        self.raw_fbo.use()
        prev_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        g_buffer.depth_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)

        if u_rad is not None:
            u_rad.value = radius
        if u_int is not None:
            u_int.value = intensity

        vao.render(moderngl.TRIANGLES, vertices=3)

        # 3. Horizontal then vertical bilateral blur through the shared helper
        context.resources["ao_texture"] = self.blur_pass.blur_hv(self.raw_ao_tex, g_buffer.depth_texture)

        self.ctx.viewport = prev_viewport

    def destroy(self) -> None:
        self.raw_fbo.release()
        self.raw_ao_tex.release()
        self.blur_pass.destroy()
        self.white_fallback.release()
        self.ssao_vao.release()
        self.gtao_vao.release()
        self.ssao_prog.release()
        self.gtao_prog.release()
