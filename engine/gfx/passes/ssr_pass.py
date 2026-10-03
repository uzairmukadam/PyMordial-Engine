"""Screen-Space Reflections (SSR) Pass for PyMordial Engine.

Ray-marches reflected view vectors against the G-Buffer depth buffer in view space
with sub-texel IGN jitter, adaptive thickness, binary search refinement, and
roughness-guided edge-preserving bilateral blur.
"""

from __future__ import annotations
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class SSRPass(RenderPass):
    """Computes screen-space specular reflections with edge-preserving bilateral blur."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SSRPass", enabled=True)
        self.ctx = ctx
        self.width = max(width // 2, 1)
        self.height = max(height // 2, 1)

        # 1. Ping-pong textures for raw SSR raymarching & bilateral filtering (RGBA16F)
        self.raw_ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssr_tex.repeat_x = False
        self.raw_ssr_tex.repeat_y = False

        self.blur_ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssr_tex.repeat_x = False
        self.blur_ssr_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssr_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssr_tex])

        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # 2. Compile Shaders
        quad_vert = load_shader("fullscreen_quad.vert")
        ssr_frag = load_shader("ssr.frag")
        blur_frag = load_shader("ssr_blur.frag")

        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssr_frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        self.blur_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=blur_frag)
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])

        self._u_steps = self.prog.get("u_SSR_Steps", None)
        self._u_max_dist = self.prog.get("u_SSR_MaxDistance", None)
        self._u_thickness = self.prog.get("u_SSR_Thickness", None)
        self._u_max_rough = self.prog.get("u_SSR_MaxRoughness", None)

        self._u_blur_dir = self.blur_prog.get("u_BlurDirection", None)
        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)

    @property
    def ssr_tex(self) -> moderngl.Texture:
        """Exposes primary SSR texture for backward-compatibility with tests."""
        return self.raw_ssr_tex

    def resize(self, width: int, height: int) -> None:
        new_w = max(width // 2, 1)
        new_h = max(height // 2, 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h

        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ssr_tex.release()
        self.blur_ssr_tex.release()

        self.raw_ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssr_tex.repeat_x = False
        self.raw_ssr_tex.repeat_y = False

        self.blur_ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssr_tex.repeat_x = False
        self.blur_ssr_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssr_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssr_tex])

        self._blur_dir_h = (1.0 / self.width, 0.0)
        self._blur_dir_v = (0.0, 1.0 / self.height)

    def execute(self, context: RenderGraphContext) -> None:
        if not getattr(context.config, "ssr_enabled", True):
            context.resources["ssr_texture"] = self.black_fallback
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            context.resources["ssr_texture"] = self.black_fallback
            return

        # Scene color source: can use resolved lit frame or albedo fallback
        scene_tex = context.resources.get("scene_color", g_buffer.albedo_roughness_texture)

        # 1. Render Raw Ray-marched SSR
        self.raw_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)
        scene_tex.use(location=3)

        if self._u_steps is not None:
            self._u_steps.value = int(getattr(context.config, "ssr_steps", 32))
        if self._u_max_dist is not None:
            self._u_max_dist.value = float(getattr(context.config, "ssr_max_distance", 20.0))
        if self._u_thickness is not None:
            self._u_thickness.value = float(getattr(context.config, "ssr_thickness", 0.40))
        if self._u_max_rough is not None:
            self._u_max_rough.value = float(getattr(context.config, "ssr_max_roughness", 0.65))

        self.vao.render(moderngl.TRIANGLES, vertices=3)

        # 2. Horizontal Bilateral Denoising Blur: raw -> blur_fbo
        self.blur_fbo.use()
        self.raw_ssr_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.albedo_roughness_texture.use(location=2)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_h
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # 3. Vertical Bilateral Denoising Blur: blur -> raw_fbo
        self.raw_fbo.use()
        self.blur_ssr_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.albedo_roughness_texture.use(location=2)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = self._blur_dir_v
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        context.resources["ssr_texture"] = self.raw_ssr_tex

    def destroy(self) -> None:
        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ssr_tex.release()
        self.blur_ssr_tex.release()
        self.black_fallback.release()
        self.vao.release()
        self.blur_vao.release()
        self.prog.release()
        self.blur_prog.release()
