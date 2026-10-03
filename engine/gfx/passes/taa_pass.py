"""Temporal Anti-Aliasing (TAA) Pass for PyMordial Engine.

Implements subpixel Halton (2, 3) projection jittering, history reprojection,
and tonemapped YCoCg neighborhood variance bounding-box clamping to eliminate specular shimmer
and geometric aliasing without color shifting or temporal wobbling.
"""

from __future__ import annotations
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()


def _halton(index: int, base: int) -> float:
    """Computes the index-th value of the radical inverse sequence for a prime base."""
    f = 1.0
    r = 0.0
    curr = index
    while curr > 0:
        f = f / base
        r = r + f * (curr % base)
        curr = curr // base
    return r


class TAAPass(RenderPass):
    """Sub-pixel jittering and temporal reprojection accumulation pass."""

    # Precomputed 8-phase Halton(2, 3) normalized offsets (-0.5 .. +0.5 pixels)
    HALTON_8 = tuple(
        (_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5)
        for i in range(8)
    )

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="TAAPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.frame_idx = 0

        # Ping-pong history textures (RGBA16F)
        self.hist_a = self.ctx.texture((width, height), 4, dtype="f2")
        self.hist_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_a.repeat_x = False
        self.hist_a.repeat_y = False

        self.hist_b = self.ctx.texture((width, height), 4, dtype="f2")
        self.hist_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_b.repeat_x = False
        self.hist_b.repeat_y = False

        self.fbo_a = self.ctx.framebuffer(color_attachments=[self.hist_a])
        self.fbo_b = self.ctx.framebuffer(color_attachments=[self.hist_b])

        self._read_hist = self.hist_a
        self._write_fbo = self.fbo_b
        self._write_hist = self.hist_b

        self._prev_vp = np.identity(4, dtype=np.float32).flatten()

        quad_vert = load_shader("fullscreen_quad.vert")
        taa_frag = load_shader("taa.frag")

        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=taa_frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        self._u_prev_vp = self.prog.get("u_PrevViewProjection", None)
        self._u_feedback = self.prog.get("u_Feedback", None)
        self._u_sharpness = self.prog.get("u_Sharpness", None)
        self._u_gamma = self.prog.get("u_Gamma", None)

    def get_jitter(self, width: int, height: int) -> tuple[float, float]:
        """Calculates subpixel projection offset for the current frame in UV coordinates."""
        hx, hy = self.HALTON_8[self.frame_idx % 8]
        return hx / max(width, 1), hy / max(height, 1)

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        self.fbo_a.release()
        self.fbo_b.release()
        self.hist_a.release()
        self.hist_b.release()

        self.hist_a = self.ctx.texture((width, height), 4, dtype="f2")
        self.hist_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_a.repeat_x = False
        self.hist_a.repeat_y = False

        self.hist_b = self.ctx.texture((width, height), 4, dtype="f2")
        self.hist_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_b.repeat_x = False
        self.hist_b.repeat_y = False

        self.fbo_a = self.ctx.framebuffer(color_attachments=[self.hist_a])
        self.fbo_b = self.ctx.framebuffer(color_attachments=[self.hist_b])

        self._read_hist = self.hist_a
        self._write_fbo = self.fbo_b
        self._write_hist = self.hist_b
        self.frame_idx = 0

    def execute(self, context: RenderGraphContext) -> None:
        self.frame_idx += 1
        current_tex = context.resources.get("hdr_color")
        g_buffer = context.resources.get("g_buffer")

        aa_mode = getattr(context.config, "aa_mode", "OFF")
        taa_active = (aa_mode == "TAA" or getattr(context.config, "taa_enabled", False))

        if not taa_active or current_tex is None or g_buffer is None:
            context.resources["taa_output"] = current_tex
            return

        self._write_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        current_tex.use(location=0)
        self._read_hist.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.velocity_texture.use(location=3)

        # On frame 1, seed previous unjittered VP from current to avoid initialization flash
        prev_vp_candidate = context.resources.get("prev_unjittered_vp")
        if prev_vp_candidate is not None:
            self._prev_vp = prev_vp_candidate
        elif hasattr(context.frame_context, "view_proj_mat") and self.frame_idx <= 1:
            self._prev_vp = np.copy(context.frame_context.view_proj_mat)

        if self._u_prev_vp is not None:
            self._u_prev_vp.write(self._prev_vp.tobytes())
        if self._u_feedback is not None:
            self._u_feedback.value = float(getattr(context.config, "taa_feedback", 0.95))
        if self._u_sharpness is not None:
            self._u_sharpness.value = float(getattr(context.config, "taa_sharpness", 0.35))
        if self._u_gamma is not None:
            self._u_gamma.value = float(getattr(context.config, "taa_gamma", 1.25))

        self.vao.render(moderngl.TRIANGLES, vertices=3)

        # Ping-pong swap
        context.resources["taa_output"] = self._write_hist
        if self._read_hist is self.hist_a:
            self._read_hist = self.hist_b
            self._write_fbo = self.fbo_a
            self._write_hist = self.hist_a
        else:
            self._read_hist = self.hist_a
            self._write_fbo = self.fbo_b
            self._write_hist = self.hist_b

    def destroy(self) -> None:
        self.fbo_a.release()
        self.fbo_b.release()
        self.hist_a.release()
        self.hist_b.release()
        self.vao.release()
        self.prog.release()
