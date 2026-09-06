"""Image-Based Lighting (IBL) and Split-Sum BRDF Approximation for PyMordial Engine.

Generates precomputed 2D split-sum Cook-Torrance BRDF integration LUT texture (256x256)
and manages HDR environment irradiance and prefiltered specular radiance maps dynamically
synchronized with the active directional sun vector.
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class IBLPass(RenderPass):
    """Generates BRDF LUT and manages HDR environment specular/diffuse radiance."""

    def __init__(self, ctx: moderngl.Context) -> None:
        super().__init__(name="IBLPass", enabled=True)
        self.ctx = ctx
        self._last_sun_dir: tuple[float, float, float] | None = None

        # 1. Generate Split-Sum BRDF LUT (256x256 RG16F)
        self.lut_tex = self.ctx.texture((256, 256), 2, dtype="f2")
        self.lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lut_tex.repeat_x = False
        self.lut_tex.repeat_y = False

        self._generate_brdf_lut()

        # 2. Procedural HDR Sky / Specular Environment Radiance Map (512x256 RGBA16F with mips)
        self.w, self.h = 512, 256
        self.env_tex = self.ctx.texture((self.w, self.h), 4, dtype="f2")
        self.env_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.env_tex.repeat_x = True
        self.env_tex.repeat_y = False

        self._update_procedural_env_map((0.5, -0.7, 0.4))

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

    def _update_procedural_env_map(self, sun_dir: tuple[float, float, float]) -> None:
        """Updates the procedural HDR physical sky panoramic texture aligned with the sun."""
        w, h = self.w, self.h
        u = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
        v = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # Horizon to zenith gradient (physical atmospheric tones)
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

        # Dynamic sun position in panoramic spherical coordinates
        # sun_dir points from sun to scene, so -sun_dir points to the sun in the sky
        lx, ly, lz = -sun_dir[0], -sun_dir[1], -sun_dir[2]
        l_len = math.sqrt(lx * lx + ly * ly + lz * lz) + 1e-6
        lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

        sun_u = (math.atan2(lz, lx) / (2.0 * math.pi) + 0.5) % 1.0
        sun_v = max(0.001, min(0.999, math.asin(max(-0.999, min(0.999, ly))) / math.pi + 0.5))

        # Add HDR sun disc in the sky
        dist_sq = (uu - sun_u) ** 2 + (vv - sun_v) ** 2
        sun_mask = dist_sq < 0.006
        data[sun_mask, :3] += (
            np.array([16.0, 14.0, 10.0], dtype=np.float32) * (1.0 - dist_sq[sun_mask] / 0.006)[..., np.newaxis]
        )

        self.env_tex.write(data.astype(np.float16).tobytes())
        self.env_tex.build_mipmaps()
        self._last_sun_dir = sun_dir

    def execute(self, context: RenderGraphContext) -> None:
        sun_dir = getattr(context, "sun_dir", None)
        if sun_dir is not None:
            if self._last_sun_dir is None:
                self._update_procedural_env_map(sun_dir)
            else:
                # Update only if sun direction shifted noticeably
                dx = sun_dir[0] - self._last_sun_dir[0]
                dy = sun_dir[1] - self._last_sun_dir[1]
                dz = sun_dir[2] - self._last_sun_dir[2]
                d_sq = dx * dx + dy * dy + dz * dz
                if d_sq > 0.0005:
                    self._update_procedural_env_map(sun_dir)

        context.resources["ibl_lut"] = self.lut_tex
        context.resources["ibl_env_map"] = self.env_tex

    def destroy(self) -> None:
        self.lut_tex.release()
        self.env_tex.release()
