"""Velocity-Buffer Motion Blur Pass for PyMordial Engine.

Implements the two-pass reconstruction filter of McGuire et al. 2012,
"A Reconstruction Filter for Plausible Motion Blur":

1. ``motion_blur_tile.frag`` reduces the G-buffer velocity buffer (RT2, RG16F,
   UV-space velocity) into a small per-tile maximum-velocity texture.
2. ``motion_blur_gather.frag`` performs a directional gather whose per-pixel
   sample count is derived from that tile maximum, with 3x3 velocity dilation
   and depth-aware weighting so background never bleeds over foreground.

Because the pass reads the real velocity buffer it blurs object motion as well
as camera motion, and it runs *after* TAA so the temporal resolve never
accumulates an already-blurred image.
"""

from __future__ import annotations

import math

import moderngl

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader


class MotionBlurPass(RenderPass):
    """Two-pass tile-max velocity motion blur."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(
            name="MotionBlurPass",
            enabled=True,
            inputs=("hdr_color", "g_buffer"),
            outputs=("motion_blur_output",),
            after=("TAA",),
            before=("PostProcess",),
        )
        self.ctx = ctx
        self.width = max(1, width)
        self.height = max(1, height)

        quad_vert = load_shader("fullscreen_quad.vert")
        tile_frag = load_shader("motion_blur_tile.frag")
        gather_frag = load_shader("motion_blur_gather.frag")

        self.tile_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=tile_frag)
        self.gather_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=gather_frag)

        self.tile_vao = self.ctx.vertex_array(self.tile_prog, [])
        self.gather_vao = self.ctx.vertex_array(self.gather_prog, [])

        # Cached uniform handles (never looked up inside the frame loop)
        self._u_tile_velocity = self.tile_prog.get("u_VelocityTexture", None)
        self._u_tile_size = self.tile_prog.get("u_TileSize", None)
        self._u_tile_screen = self.tile_prog.get("u_ScreenSize", None)

        self._u_g_scene = self.gather_prog.get("u_SceneHDR", None)
        self._u_g_velocity = self.gather_prog.get("u_VelocityTexture", None)
        self._u_g_depth = self.gather_prog.get("u_DepthTexture", None)
        self._u_g_tilemax = self.gather_prog.get("u_TileMaxTexture", None)
        self._u_g_tile_size = self.gather_prog.get("u_TileSize", None)
        self._u_g_max_samples = self.gather_prog.get("u_MaxSamples", None)
        self._u_g_intensity = self.gather_prog.get("u_Intensity", None)
        self._u_g_max_radius = self.gather_prog.get("u_MaxRadius", None)

        if self._u_tile_velocity is not None:
            self._u_tile_velocity.value = 0
        if self._u_g_scene is not None:
            self._u_g_scene.value = 0
        if self._u_g_velocity is not None:
            self._u_g_velocity.value = 1
        if self._u_g_depth is not None:
            self._u_g_depth.value = 2
        if self._u_g_tilemax is not None:
            self._u_g_tilemax.value = 3

        self.tile_size = 16
        self.tile_texture: moderngl.Texture | None = None
        self.tile_fbo: moderngl.Framebuffer | None = None
        self.output_texture: moderngl.Texture | None = None
        self.output_fbo: moderngl.Framebuffer | None = None

        self._create_buffers()

    # ---------------- Allocation ----------------

    def _tile_dims(self, tile_size: int) -> tuple[int, int]:
        return (
            max(1, math.ceil(self.width / max(1, tile_size))),
            max(1, math.ceil(self.height / max(1, tile_size))),
        )

    def _create_buffers(self) -> None:
        """Allocates the tile-max texture and the full-resolution output target."""
        tw, th = self._tile_dims(self.tile_size)
        self.tile_texture = self.ctx.texture((tw, th), 4, dtype="f2")
        self.tile_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tile_texture.repeat_x = False
        self.tile_texture.repeat_y = False
        self.tile_fbo = self.ctx.framebuffer(color_attachments=[self.tile_texture])

        self.output_texture = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.output_texture.repeat_x = False
        self.output_texture.repeat_y = False
        self.output_fbo = self.ctx.framebuffer(color_attachments=[self.output_texture])

    def _destroy_buffers(self) -> None:
        if self.tile_fbo is not None:
            self.tile_fbo.release()
            self.tile_fbo = None
        if self.tile_texture is not None:
            self.tile_texture.release()
            self.tile_texture = None
        if self.output_fbo is not None:
            self.output_fbo.release()
            self.output_fbo = None
        if self.output_texture is not None:
            self.output_texture.release()
            self.output_texture = None

    def _ensure_tile_size(self, tile_size: int) -> None:
        """Reallocates the tile texture only when the configured tile size changes."""
        tile_size = max(1, int(tile_size))
        if tile_size == self.tile_size and self.tile_texture is not None:
            return
        self.tile_size = tile_size
        if self.tile_fbo is not None:
            self.tile_fbo.release()
            self.tile_fbo = None
        if self.tile_texture is not None:
            self.tile_texture.release()
            self.tile_texture = None
        tw, th = self._tile_dims(tile_size)
        self.tile_texture = self.ctx.texture((tw, th), 4, dtype="f2")
        self.tile_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tile_texture.repeat_x = False
        self.tile_texture.repeat_y = False
        self.tile_fbo = self.ctx.framebuffer(color_attachments=[self.tile_texture])

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if width == self.width and height == self.height:
            return
        self._destroy_buffers()
        self.width = width
        self.height = height
        self._create_buffers()

    # ---------------- Execution ----------------

    def execute(self, context: RenderGraphContext) -> None:
        resources = context.resources
        g_buffer = resources.get("g_buffer")
        config = context.config

        source_tex = resources.get("hdr_color")
        if source_tex is None:
            hdr_fbo = resources.get("hdr_fbo")
            if hdr_fbo is not None and hdr_fbo.color_attachments:
                source_tex = hdr_fbo.color_attachments[0]

        enabled = bool(getattr(config, "motion_blur_enabled", False))
        if not enabled or g_buffer is None or source_tex is None:
            resources["motion_blur_output"] = source_tex
            return

        self._ensure_tile_size(getattr(config, "motion_blur_tile_size", 16))

        velocity_tex = g_buffer.velocity_texture
        depth_tex = g_buffer.depth_texture
        max_samples = max(1, int(getattr(config, "motion_blur_max_samples", 32)))
        intensity = float(getattr(config, "motion_blur_intensity", 1.5))
        max_radius = float(getattr(config, "motion_blur_max_radius", 48.0))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # ---- Pass 1: per-tile maximum velocity ----
        tw, th = self.tile_texture.size
        self.tile_fbo.use()
        self.ctx.viewport = (0, 0, tw, th)
        velocity_tex.use(location=0)
        if self._u_tile_size is not None:
            self._u_tile_size.value = (self.tile_size, self.tile_size)
        if self._u_tile_screen is not None:
            self._u_tile_screen.value = (float(self.width), float(self.height))
        self.tile_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # ---- Pass 2: directional gather ----
        self.output_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        source_tex.use(location=0)
        velocity_tex.use(location=1)
        depth_tex.use(location=2)
        self.tile_texture.use(location=3)
        if self._u_g_tile_size is not None:
            self._u_g_tile_size.value = (self.tile_size, self.tile_size)
        if self._u_g_max_samples is not None:
            self._u_g_max_samples.value = max_samples
        if self._u_g_intensity is not None:
            self._u_g_intensity.value = intensity
        if self._u_g_max_radius is not None:
            self._u_g_max_radius.value = max_radius
        self.gather_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        resources["motion_blur_output"] = self.output_texture

    def destroy(self) -> None:
        self._destroy_buffers()
        if self.tile_vao is not None:
            self.tile_vao.release()
            self.tile_vao = None
        if self.gather_vao is not None:
            self.gather_vao.release()
            self.gather_vao = None
        if self.tile_prog is not None:
            self.tile_prog.release()
            self.tile_prog = None
        if self.gather_prog is not None:
            self.gather_prog.release()
            self.gather_prog = None
