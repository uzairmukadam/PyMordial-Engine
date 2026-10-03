"""Fast Approximate Anti-Aliasing (FXAA 3.11 Quality) Pass."""

from __future__ import annotations
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


class FXAAPass(RenderPass):
    """Executes high-quality single-pass FXAA 3.11 post-processing."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="FXAAPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height

        self.output_texture = self.ctx.texture((width, height), 4, dtype="f1")
        self.output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.output_texture.repeat_x = False
        self.output_texture.repeat_y = False

        self.fbo = self.ctx.framebuffer(color_attachments=[self.output_texture])

        quad_vert = load_shader("fullscreen_quad.vert")
        fxaa_frag = load_shader("fxaa.frag")

        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=fxaa_frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        self._u_inv_screen = self.prog.get("u_InverseScreenSize", None)
        self._u_subpixel = self.prog.get("u_SubpixelQuality", None)
        self._u_threshold = self.prog.get("u_EdgeThreshold", None)
        self._u_threshold_min = self.prog.get("u_EdgeThresholdMin", None)

        self._inv_screen = (1.0 / max(width, 1), 1.0 / max(height, 1))
        if self._u_inv_screen is not None:
            self._u_inv_screen.value = self._inv_screen

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        self.fbo.release()
        self.output_texture.release()

        self.output_texture = self.ctx.texture((width, height), 4, dtype="f1")
        self.output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.output_texture.repeat_x = False
        self.output_texture.repeat_y = False

        self.fbo = self.ctx.framebuffer(color_attachments=[self.output_texture])

        self._inv_screen = (1.0 / max(width, 1), 1.0 / max(height, 1))
        if self._u_inv_screen is not None:
            self._u_inv_screen.value = self._inv_screen

    def execute(self, context: RenderGraphContext) -> None:
        input_tex = context.resources.get("ldr_color")
        if input_tex is None:
            return

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        input_tex.use(location=0)

        if self._u_inv_screen is not None:
            self._u_inv_screen.value = self._inv_screen
        if self._u_subpixel is not None:
            self._u_subpixel.value = float(getattr(context.config, "fxaa_subpixel", 0.75))
        if self._u_threshold is not None:
            self._u_threshold.value = float(getattr(context.config, "fxaa_edge_threshold", 0.125))

        self.vao.render(moderngl.TRIANGLES, vertices=3)
        context.resources["fxaa_output"] = self.output_texture

    def destroy(self) -> None:
        self.fbo.release()
        self.output_texture.release()
        self.vao.release()
        self.prog.release()
