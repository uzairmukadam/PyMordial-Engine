"""Enhanced Subpixel Morphological Anti-Aliasing (SMAA 1x, 2x, and 4x) Pass."""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class SMAAPass(RenderPass):
    """Executes multi-stage SMAA 1x, 2x, or 4x anti-aliasing."""

    # Standard SMAA 2x (T2x) subpixel jitter offsets (normalized)
    JITTER_2X = [
        (-0.25, 0.25),
        (0.25, -0.25),
    ]

    # Standard SMAA 4x subpixel jitter offsets (normalized)
    JITTER_4X = [
        (-0.125, -0.375),
        (0.375, -0.125),
        (0.125, 0.375),
        (-0.375, 0.125),
    ]

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SMAAPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.frame_idx = 0

        quad_vert = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
        edge_frag = (SHADER_DIR / "smaa_edge.frag").read_text(encoding="utf-8")
        blend_frag = (SHADER_DIR / "smaa_blend.frag").read_text(encoding="utf-8")
        neigh_frag = (SHADER_DIR / "smaa_neighborhood.frag").read_text(encoding="utf-8")
        resolve_frag = (SHADER_DIR / "smaa_resolve.frag").read_text(encoding="utf-8")

        self.prog_edge = self.ctx.program(vertex_shader=quad_vert, fragment_shader=edge_frag)
        self.prog_blend = self.ctx.program(vertex_shader=quad_vert, fragment_shader=blend_frag)
        self.prog_neigh = self.ctx.program(vertex_shader=quad_vert, fragment_shader=neigh_frag)
        self.prog_resolve = self.ctx.program(vertex_shader=quad_vert, fragment_shader=resolve_frag)

        self.vao_edge = self.ctx.vertex_array(self.prog_edge, [])
        self.vao_blend = self.ctx.vertex_array(self.prog_blend, [])
        self.vao_neigh = self.ctx.vertex_array(self.prog_neigh, [])
        self.vao_resolve = self.ctx.vertex_array(self.prog_resolve, [])

        self._create_textures()

    def _create_textures(self) -> None:
        w, h = self.width, self.height

        # Pass 1: Edges (RG8)
        self.tex_edge = self.ctx.texture((w, h), 2, dtype="f1")
        self.tex_edge.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_edge.repeat_x = False
        self.tex_edge.repeat_y = False
        self.fbo_edge = self.ctx.framebuffer(color_attachments=[self.tex_edge])

        # Pass 2: Blending Weights (RGBA8)
        self.tex_blend = self.ctx.texture((w, h), 4, dtype="f1")
        self.tex_blend.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_blend.repeat_x = False
        self.tex_blend.repeat_y = False
        self.fbo_blend = self.ctx.framebuffer(color_attachments=[self.tex_blend])

        # Pass 3: Neighborhood Blended SMAA 1x output (RGBA8)
        self.tex_smaa = self.ctx.texture((w, h), 4, dtype="f1")
        self.tex_smaa.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_smaa.repeat_x = False
        self.tex_smaa.repeat_y = False
        self.fbo_smaa = self.ctx.framebuffer(color_attachments=[self.tex_smaa])

        # Pass 4 (2x/4x): Ping-pong temporal history textures (RGBA8)
        self.hist_a = self.ctx.texture((w, h), 4, dtype="f1")
        self.hist_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_a.repeat_x = False
        self.hist_a.repeat_y = False
        self.fbo_hist_a = self.ctx.framebuffer(color_attachments=[self.hist_a])

        self.hist_b = self.ctx.texture((w, h), 4, dtype="f1")
        self.hist_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hist_b.repeat_x = False
        self.hist_b.repeat_y = False
        self.fbo_hist_b = self.ctx.framebuffer(color_attachments=[self.hist_b])

        self._read_hist = self.hist_a
        self._write_fbo = self.fbo_hist_b
        self._write_hist = self.hist_b

    def get_jitter(self, width: int, height: int, mode: str, frame_idx: int | None = None) -> tuple[float, float]:
        """Returns subpixel projection jitter for temporal SMAA 2x / 4x."""
        idx = self.frame_idx if frame_idx is None else frame_idx
        if mode == "SMAA_2X":
            offset = self.JITTER_2X[idx % len(self.JITTER_2X)]
            return offset[0] / max(width, 1), offset[1] / max(height, 1)
        elif mode == "SMAA_4X":
            offset = self.JITTER_4X[idx % len(self.JITTER_4X)]
            return offset[0] / max(width, 1), offset[1] / max(height, 1)
        return 0.0, 0.0

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        self.fbo_edge.release()
        self.tex_edge.release()
        self.fbo_blend.release()
        self.tex_blend.release()
        self.fbo_smaa.release()
        self.tex_smaa.release()
        self.fbo_hist_a.release()
        self.hist_a.release()
        self.fbo_hist_b.release()
        self.hist_b.release()

        self._create_textures()

    def execute(self, context: RenderGraphContext) -> None:
        self.frame_idx += 1
        input_tex = context.resources.get("ldr_color")
        g_buffer = context.resources.get("g_buffer")
        if input_tex is None:
            return

        aa_mode = getattr(context.config, "aa_mode", "SMAA_1X")
        inv_screen = (1.0 / max(self.width, 1), 1.0 / max(self.height, 1))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.viewport = (0, 0, self.width, self.height)

        # ---- PASS 1: Edge Detection ----
        self.fbo_edge.use()
        self.fbo_edge.clear(0.0, 0.0, 0.0, 0.0)
        input_tex.use(location=0)
        if "u_InverseScreenSize" in self.prog_edge:
            self.prog_edge["u_InverseScreenSize"].value = inv_screen
        if "u_EdgeThreshold" in self.prog_edge:
            self.prog_edge["u_EdgeThreshold"].value = float(getattr(context.config, "smaa_threshold", 0.08))
        self.vao_edge.render(moderngl.TRIANGLES, vertices=3)

        # ---- PASS 2: Blending Weight Calculation ----
        self.fbo_blend.use()
        self.fbo_blend.clear(0.0, 0.0, 0.0, 0.0)
        self.tex_edge.use(location=0)
        if "u_InverseScreenSize" in self.prog_blend:
            self.prog_blend["u_InverseScreenSize"].value = inv_screen
        if "u_MaxSearchSteps" in self.prog_blend:
            steps = 32 if aa_mode == "SMAA_4X" else 16
            self.prog_blend["u_MaxSearchSteps"].value = steps
        self.vao_blend.render(moderngl.TRIANGLES, vertices=3)

        # ---- PASS 3: Neighborhood Blending ----
        self.fbo_smaa.use()
        input_tex.use(location=0)
        self.tex_blend.use(location=1)
        if "u_InverseScreenSize" in self.prog_neigh:
            self.prog_neigh["u_InverseScreenSize"].value = inv_screen
        self.vao_neigh.render(moderngl.TRIANGLES, vertices=3)

        # If SMAA 1x, we are done
        if aa_mode == "SMAA_1X" or g_buffer is None:
            context.resources["smaa_output"] = self.tex_smaa
            return

        # ---- PASS 4: Temporal Subpixel Resolve (SMAA 2x / 4x) ----
        self._write_fbo.use()
        self.tex_smaa.use(location=0)
        self._read_hist.use(location=1)
        g_buffer.velocity_texture.use(location=2)
        g_buffer.depth_texture.use(location=3)

        if "u_InverseScreenSize" in self.prog_resolve:
            self.prog_resolve["u_InverseScreenSize"].value = inv_screen
        if "u_TemporalWeight" in self.prog_resolve:
            weight = 0.75 if aa_mode == "SMAA_4X" else 0.50
            self.prog_resolve["u_TemporalWeight"].value = weight

        self.vao_resolve.render(moderngl.TRIANGLES, vertices=3)

        context.resources["smaa_output"] = self._write_hist

        # Ping-pong swap
        if self._read_hist is self.hist_a:
            self._read_hist = self.hist_b
            self._write_fbo = self.fbo_hist_a
            self._write_hist = self.hist_a
        else:
            self._read_hist = self.hist_a
            self._write_fbo = self.fbo_hist_b
            self._write_hist = self.hist_b

    def destroy(self) -> None:
        self.fbo_edge.release()
        self.tex_edge.release()
        self.fbo_blend.release()
        self.tex_blend.release()
        self.fbo_smaa.release()
        self.tex_smaa.release()
        self.fbo_hist_a.release()
        self.hist_a.release()
        self.fbo_hist_b.release()
        self.hist_b.release()

        self.vao_edge.release()
        self.prog_edge.release()
        self.vao_blend.release()
        self.prog_blend.release()
        self.vao_neigh.release()
        self.prog_neigh.release()
        self.vao_resolve.release()
        self.prog_resolve.release()
