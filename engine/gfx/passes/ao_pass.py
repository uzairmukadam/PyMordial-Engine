"""Ambient Occlusion Pass with Bilateral Filter for PyMordial Engine.

Supports SSAO (Hemisphere sampling), HBAO (Horizon-Based Ambient Occlusion), and GTAO
(Ground-Truth Ambient Occlusion) with an edge-preserving metric view-depth bilateral filter.
"""

from __future__ import annotations
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class AmbientOcclusionPass(RenderPass):
    """Computes SSAO, HBAO, or GTAO and applies bilateral edge-preserving blur."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="AmbientOcclusionPass", enabled=True)
        self.ctx = ctx
        self.width = max(width // 2, 1)
        self.height = max(height // 2, 1)

        # 1. Ping-pong half-resolution R8 textures for AO computation & blur
        self.raw_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.raw_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ao_tex.repeat_x = False
        self.raw_ao_tex.repeat_y = False

        self.blur_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.blur_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ao_tex.repeat_x = False
        self.blur_ao_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ao_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ao_tex])

        # 1x1 fallback white texture when AO is disabled
        self.white_fallback = self.ctx.texture((1, 1), 1, data=b"\xff", dtype="f1")

        # 2. Compile Shaders
        quad_vert = load_shader("fullscreen_quad.vert")
        ssao_frag = load_shader("ssao.frag")
        hbao_frag = load_shader("hbao.frag")
        gtao_frag = load_shader("gtao.frag")
        blur_frag = load_shader("bilateral_blur.frag")

        self.ssao_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssao_frag)
        self.hbao_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=hbao_frag)
        self.gtao_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=gtao_frag)
        self.blur_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=blur_frag)

        self.ssao_vao = self.ctx.vertex_array(self.ssao_prog, [])
        self.hbao_vao = self.ctx.vertex_array(self.hbao_prog, [])
        self.gtao_vao = self.ctx.vertex_array(self.gtao_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])

        # Pre-cache uniform handles
        self._u_ssao_radius = self.ssao_prog.get("u_Radius", None)
        self._u_ssao_intensity = self.ssao_prog.get("u_Intensity", None)

        self._u_hbao_radius = self.hbao_prog.get("u_Radius", None)
        self._u_hbao_intensity = self.hbao_prog.get("u_Intensity", None)

        self._u_gtao_radius = self.gtao_prog.get("u_Radius", None)
        self._u_gtao_intensity = self.gtao_prog.get("u_Intensity", None)

        self._u_blur_dir = self.blur_prog.get("u_BlurDirection", None)
        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)

    def resize(self, width: int, height: int) -> None:
        new_w = max(width // 2, 1)
        new_h = max(height // 2, 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h

        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ao_tex.release()
        self.blur_ao_tex.release()

        self.raw_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.raw_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ao_tex.repeat_x = False
        self.raw_ao_tex.repeat_y = False

        self.blur_ao_tex = self.ctx.texture((self.width, self.height), 1, dtype="f1")
        self.blur_ao_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ao_tex.repeat_x = False
        self.blur_ao_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ao_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ao_tex])

        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)

    def execute(self, context: RenderGraphContext) -> None:
        ao_mode = getattr(context.config, "ao_mode", "GTAO")
        if ao_mode == "OFF":
            context.resources["ao_texture"] = self.white_fallback
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            context.resources["ao_texture"] = self.white_fallback
            return

        radius = float(getattr(context.config, "ao_radius", 0.75))
        intensity = float(getattr(context.config, "ao_intensity", 1.2))

        # 1. Select AO program and VAO
        if ao_mode == "SSAO":
            vao = self.ssao_vao
            u_rad = self._u_ssao_radius
            u_int = self._u_ssao_intensity
        elif ao_mode == "HBAO":
            vao = self.hbao_vao
            u_rad = self._u_hbao_radius
            u_int = self._u_hbao_intensity
        else:  # "GTAO"
            vao = self.gtao_vao
            u_rad = self._u_gtao_radius
            u_int = self._u_gtao_intensity

        # 2. Render Raw AO
        self.raw_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        g_buffer.depth_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)

        if u_rad is not None:
            u_rad.value = radius
        if u_int is not None:
            u_int.value = intensity

        vao.render(moderngl.TRIANGLES, vertices=3)

        # 3. Horizontal Bilateral Blur: raw -> blur_fbo
        self.blur_fbo.use()
        self.raw_ao_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_h
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # 4. Vertical Bilateral Blur: blur -> raw_fbo
        self.raw_fbo.use()
        self.blur_ao_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_v
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # Published result is in raw_ao_tex (after 2 passes)
        context.resources["ao_texture"] = self.raw_ao_tex

    def destroy(self) -> None:
        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ao_tex.release()
        self.blur_ao_tex.release()
        self.white_fallback.release()
        self.ssao_vao.release()
        self.hbao_vao.release()
        self.gtao_vao.release()
        self.blur_vao.release()
        self.ssao_prog.release()
        self.hbao_prog.release()
        self.gtao_prog.release()
        self.blur_prog.release()
