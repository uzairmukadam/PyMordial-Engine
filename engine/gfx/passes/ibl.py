"""Image-Based Lighting (IBL) and Split-Sum BRDF Approximation for PyMordial Engine.

Generates precomputed 2D split-sum Cook-Torrance BRDF integration LUT texture (256x256)
and manages HDR environment irradiance and prefiltered specular radiance maps.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class IBLPass(RenderPass):
    """Generates BRDF LUT and manages HDR environment specular/diffuse radiance."""

    def __init__(self, ctx: moderngl.Context) -> None:
        super().__init__(name="IBLPass", enabled=True)
        self.ctx = ctx

        # 1. Generate Split-Sum BRDF LUT (256x256 RG16F)
        self.lut_tex = self.ctx.texture((256, 256), 2, dtype="f2")
        self.lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lut_tex.repeat_x = False
        self.lut_tex.repeat_y = False

        self._generate_brdf_lut()

        # 2. Procedural HDR Sky / Specular Environment Radiance Map (512x256 RGBA16F with mips)
        self.env_tex = self._generate_procedural_env_map()

    def _generate_brdf_lut(self) -> None:
        quad_vert = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
        brdf_frag = (SHADER_DIR / "brdf_lut.frag").read_text(encoding="utf-8")

        prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=brdf_frag)
        vao = self.ctx.vertex_array(prog, [])
        fbo = self.ctx.framebuffer(color_attachments=[self.lut_tex])

        prev_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, 256, 256)
        fbo.use()
        vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.viewport = prev_viewport

        fbo.release()
        vao.release()
        prog.release()

    def _generate_procedural_env_map(self) -> moderngl.Texture:
        """Generates a procedural HDR physical sky panoramic texture with mipmaps."""
        w, h = 512, 256
        # Generate physical sky gradient in float16
        u = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
        v = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # Horizon to zenith gradient
        zenith = np.array([0.15, 0.35, 0.75, 1.0], dtype=np.float32)
        horizon = np.array([0.70, 0.80, 0.95, 1.0], dtype=np.float32)
        ground = np.array([0.15, 0.18, 0.16, 1.0], dtype=np.float32)

        data = np.zeros((h, w, 4), dtype=np.float32)
        sky_mask = vv >= 0.5
        sky_t = ((vv[sky_mask] - 0.5) * 2.0)[..., np.newaxis]
        data[sky_mask] = horizon * (1.0 - sky_t) + zenith * sky_t

        ground_mask = vv < 0.5
        ground_t = (vv[ground_mask] * 2.0)[..., np.newaxis]
        data[ground_mask] = ground * (1.0 - ground_t) + horizon * ground_t

        # Add HDR sun disc in the sky
        sun_u, sun_v = 0.65, 0.75
        dist_sq = (uu - sun_u) ** 2 + (vv - sun_v) ** 2
        sun_mask = dist_sq < 0.005
        data[sun_mask, :3] += (
            np.array([12.0, 10.0, 7.0], dtype=np.float32) * (1.0 - dist_sq[sun_mask] / 0.005)[..., np.newaxis]
        )

        tex = self.ctx.texture((w, h), 4, data=data.astype(np.float16).tobytes(), dtype="f2")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = False
        tex.build_mipmaps()
        return tex

    def execute(self, context: RenderGraphContext) -> None:
        # Publish IBL resources to blackboard
        context.resources["ibl_lut"] = self.lut_tex
        context.resources["ibl_env_map"] = self.env_tex

    def destroy(self) -> None:
        self.lut_tex.release()
        self.env_tex.release()
