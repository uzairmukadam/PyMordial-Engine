"""Screen-Space Global Illumination (SSGI) Pass for PyMordial Engine.

Executes cosine-weighted raymarching over the G-Buffer to compute near-field
diffuse indirect bounces and contact color bleeding, followed by normal- and
depth-aware edge-preserving filtering.
"""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class SSGIPass(RenderPass):
    """Computes screen-space indirect diffuse bounces and bilateral reconstruction."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SSGIPass", enabled=True)
        self.ctx = ctx
        self.width = max(width // 2, 1)
        self.height = max(height // 2, 1)

        # Half-resolution RGBA16F ping-pong buffers
        self.raw_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssgi_tex.repeat_x = False
        self.raw_ssgi_tex.repeat_y = False

        self.blur_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssgi_tex.repeat_x = False
        self.blur_ssgi_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssgi_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssgi_tex])

        # 1x1 black fallback
        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # Compile Shaders
        quad_vert = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
        ssgi_frag = (SHADER_DIR / "ssgi.frag").read_text(encoding="utf-8")
        blur_frag = (SHADER_DIR / "ssgi_blur.frag").read_text(encoding="utf-8")

        self.ssgi_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=ssgi_frag)
        self.blur_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=blur_frag)

        self.ssgi_vao = self.ctx.vertex_array(self.ssgi_prog, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [])

        # Uniform handles
        self._u_steps = self.ssgi_prog.get("u_SSGI_Steps", None)
        self._u_rays = self.ssgi_prog.get("u_SSGI_RayCount", None)
        self._u_dist = self.ssgi_prog.get("u_SSGI_RayDistance", None)
        self._u_thickness = self.ssgi_prog.get("u_SSGI_Thickness", None)
        self._u_intensity = self.ssgi_prog.get("u_SSGI_Intensity", None)
        self._u_blur_dir = self.blur_prog.get("u_BlurDirection", None)

    def resize(self, width: int, height: int) -> None:
        new_w = max(width // 2, 1)
        new_h = max(height // 2, 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h

        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ssgi_tex.release()
        self.blur_ssgi_tex.release()

        self.raw_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_ssgi_tex.repeat_x = False
        self.raw_ssgi_tex.repeat_y = False

        self.blur_ssgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.blur_ssgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_ssgi_tex.repeat_x = False
        self.blur_ssgi_tex.repeat_y = False

        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ssgi_tex])
        self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_ssgi_tex])

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "HYBRID")
        if gi_mode not in ("SSGI", "HYBRID"):
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

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)
        scene_color = context.resources.get("scene_color", g_buffer.albedo_roughness_texture)
        scene_color.use(location=3)

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

        # 2. Horizontal Bilateral Blur: raw -> blur_fbo
        self.blur_fbo.use()
        self.raw_ssgi_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = (1.0 / self.width, 0.0)
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # 3. Vertical Bilateral Blur: blur -> raw_fbo
        self.raw_fbo.use()
        self.blur_ssgi_tex.use(location=0)
        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        if self._u_blur_dir is not None:
            self._u_blur_dir.value = (0.0, 1.0 / self.height)
        self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        context.resources["ssgi_texture"] = self.raw_ssgi_tex

    def destroy(self) -> None:
        self.raw_fbo.release()
        self.blur_fbo.release()
        self.raw_ssgi_tex.release()
        self.blur_ssgi_tex.release()
        self.black_fallback.release()
        self.ssgi_vao.release()
        self.blur_vao.release()
        self.ssgi_prog.release()
        self.blur_prog.release()
