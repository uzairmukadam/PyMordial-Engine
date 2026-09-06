"""Screen-Space Reflections (SSR) Pass for PyMordial Engine.

Ray-marches reflected view vectors against the G-Buffer depth buffer with
binary search refinement, screen-edge attenuation, and roughness-based falloff.
"""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class SSRPass(RenderPass):
    """Computes screen-space specular reflections with edge and roughness fading."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SSRPass", enabled=True)
        self.ctx = ctx
        self.width = max(width // 2, 1)
        self.height = max(height // 2, 1)

        self.ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ssr_tex.repeat_x = False
        self.ssr_tex.repeat_y = False
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ssr_tex])

        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        quad_vert = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
        ssr_frag = (SHADER_DIR / "ssr.frag").read_text(encoding="utf-8")

        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssr_frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        self._u_steps = self.prog.get("u_SSR_Steps", None)
        self._u_max_dist = self.prog.get("u_SSR_MaxDistance", None)
        self._u_thickness = self.prog.get("u_SSR_Thickness", None)
        self._u_max_rough = self.prog.get("u_SSR_MaxRoughness", None)

    def resize(self, width: int, height: int) -> None:
        new_w = max(width // 2, 1)
        new_h = max(height // 2, 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h
        self.fbo.release()
        self.ssr_tex.release()

        self.ssr_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.ssr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ssr_tex.repeat_x = False
        self.ssr_tex.repeat_y = False
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ssr_tex])

    def execute(self, context: RenderGraphContext) -> None:
        if not getattr(context.config, "ssr_enabled", True):
            context.resources["ssr_texture"] = self.black_fallback
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            context.resources["ssr_texture"] = self.black_fallback
            return

        # Scene color source: can use albedo or resolved frame
        scene_tex = context.resources.get("scene_color", g_buffer.albedo_roughness_texture)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)
        scene_tex.use(location=3)

        if self._u_steps is not None:
            self._u_steps.value = getattr(context.config, "ssr_steps", 24)
        if self._u_max_rough is not None:
            self._u_max_rough.value = getattr(context.config, "ssr_max_roughness", 0.65)

        self.vao.render(moderngl.TRIANGLES, vertices=3)
        context.resources["ssr_texture"] = self.ssr_tex

    def destroy(self) -> None:
        self.fbo.release()
        self.ssr_tex.release()
        self.black_fallback.release()
        self.vao.release()
        self.prog.release()
