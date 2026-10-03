"""Native ModernGL UI Batch Renderer for PyMordial Engine.

Draws screen-space 2D SDF rounded rectangles, gradients, borders,
images, and cached font textures using ModernGL.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import moderngl
import pygame

from engine.ui.shaders import UI_VERT_GLSL, UI_FRAG_GLSL
from engine.ui.theme import UIStyle

if TYPE_CHECKING:
    pass


class UIRenderer:
    """High-performance ModernGL 2D batch renderer for engine UI."""

    __slots__ = (
        "ctx",
        "screen_width",
        "screen_height",
        "prog",
        "vao",
        "vbo",
        "max_quads",
        "quad_count",
        "_vertex_data",
        "_font_cache",
        "_text_texture_cache",
        "_white_texture",
        "_u_screen_size",
        "_u_texture",
    )

    # 24 floats per quad:
    # 0..3: in_Position [x, y, w, h]
    # 4..7: in_UV [u, v, uw, vh]
    # 8..11: in_ColorTop [r, g, b, a]
    # 12..15: in_ColorBottom [r, g, b, a]
    # 16..19: in_BorderColor [r, g, b, a]
    # 20..23: in_Params [corner_radius, border_width, tex_mode, glow_radius]
    FLOATS_PER_QUAD = 24

    def __init__(self, ctx: moderngl.Context, screen_width: int = 1920, screen_height: int = 1080, max_quads: int = 2048) -> None:
        self.ctx = ctx
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_quads = max_quads
        self.quad_count = 0

        # Pre-allocate zero-allocation batch vertex buffer
        self._vertex_data = np.zeros(self.max_quads * self.FLOATS_PER_QUAD, dtype=np.float32)

        # 1x1 default white texture for flat/gradient quads
        self._white_texture = self.ctx.texture((1, 1), 4, data=b"\xff\xff\xff\xff")
        self._white_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Fonts and textures cache
        self._font_cache: dict[tuple[str, int, bool], pygame.font.Font] = {}
        self._text_texture_cache: dict[str, tuple[moderngl.Texture, int, int]] = {}

        self._init_gl()

    def _init_gl(self) -> None:
        """Compiles UI shaders and builds instanced VAO."""
        self.prog = self.ctx.program(vertex_shader=UI_VERT_GLSL, fragment_shader=UI_FRAG_GLSL)
        self._u_screen_size = self.prog["u_ScreenSize"]
        self._u_texture = self.prog["u_Texture"]
        self._u_texture.value = 0

        self.vbo = self.ctx.buffer(reserve=self.max_quads * self.FLOATS_PER_QUAD * 4, dynamic=True)

        # Layout: 4f 4f 4f 4f 4f 4f /i (per instance quad)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (
                    self.vbo,
                    "4f 4f 4f 4f 4f 4f/i",
                    "in_Position",
                    "in_UV",
                    "in_ColorTop",
                    "in_ColorBottom",
                    "in_BorderColor",
                    "in_Params",
                )
            ],
        )

    def set_screen_size(self, width: int, height: int) -> None:
        """Updates display resolution."""
        self.screen_width = max(1, width)
        self.screen_height = max(1, height)

    def begin_frame(self) -> None:
        """Resets the quad batch buffer for a new frame."""
        self.quad_count = 0

    def draw_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        style: UIStyle,
    ) -> None:
        """Enqueues a styled rounded rectangle."""
        if self.quad_count >= self.max_quads:
            self.flush()

        idx = self.quad_count * self.FLOATS_PER_QUAD
        vd = self._vertex_data

        # in_Position
        vd[idx + 0] = x
        vd[idx + 1] = y
        vd[idx + 2] = w
        vd[idx + 3] = h

        # in_UV (default full quad)
        vd[idx + 4] = 0.0
        vd[idx + 5] = 0.0
        vd[idx + 6] = 1.0
        vd[idx + 7] = 1.0

        # in_ColorTop
        vd[idx + 8] = style.bg_color_top[0]
        vd[idx + 9] = style.bg_color_top[1]
        vd[idx + 10] = style.bg_color_top[2]
        vd[idx + 11] = style.bg_color_top[3]

        # in_ColorBottom
        vd[idx + 12] = style.bg_color_bottom[0]
        vd[idx + 13] = style.bg_color_bottom[1]
        vd[idx + 14] = style.bg_color_bottom[2]
        vd[idx + 15] = style.bg_color_bottom[3]

        # in_BorderColor
        vd[idx + 16] = style.border_color[0]
        vd[idx + 17] = style.border_color[1]
        vd[idx + 18] = style.border_color[2]
        vd[idx + 19] = style.border_color[3]

        # in_Params: [corner_radius, border_width, tex_mode=0, glow_radius]
        vd[idx + 20] = style.corner_radius
        vd[idx + 21] = style.border_width
        vd[idx + 22] = 0.0  # Flat / Gradient
        vd[idx + 23] = style.glow_radius

        self.quad_count += 1

    def draw_textured_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        texture: moderngl.Texture,
        uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        corner_radius: float = 0.0,
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        border_width: float = 0.0,
    ) -> None:
        """Draws a textured quad by flushing existing quads and binding texture."""
        self.flush()

        texture.use(location=0)

        idx = 0
        vd = self._vertex_data

        vd[idx + 0] = x
        vd[idx + 1] = y
        vd[idx + 2] = w
        vd[idx + 3] = h

        vd[idx + 4] = uv[0]
        vd[idx + 5] = uv[1]
        vd[idx + 6] = uv[2]
        vd[idx + 7] = uv[3]

        for offset in (8, 12):
            vd[idx + offset + 0] = tint[0]
            vd[idx + offset + 1] = tint[1]
            vd[idx + offset + 2] = tint[2]
            vd[idx + offset + 3] = tint[3]

        vd[idx + 16] = border_color[0]
        vd[idx + 17] = border_color[1]
        vd[idx + 18] = border_color[2]
        vd[idx + 19] = border_color[3]

        vd[idx + 20] = corner_radius
        vd[idx + 21] = border_width
        vd[idx + 22] = 1.0  # Texture mode
        vd[idx + 23] = 0.0

        self.quad_count = 1
        self.flush(custom_texture=texture)

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        font_size: int = 18,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        align: str = "left",
        bold: bool = False,
    ) -> tuple[int, int]:
        """Draws cached high-definition text glyphs to the screen."""
        if not text:
            return (0, 0)

        tex, tw, th = self._get_or_create_text_texture(text, font_size, bold)

        draw_x = x
        if align == "center":
            draw_x = x - tw * 0.5
        elif align == "right":
            draw_x = x - tw

        self.draw_textured_rect(
            x=draw_x,
            y=y,
            w=float(tw),
            h=float(th),
            texture=tex,
            tint=color,
            corner_radius=0.0,
        )
        return (tw, th)

    def _get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = ("default", size, bold)
        f = self._font_cache.get(key)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            f = pygame.font.SysFont("Segoe UI, Arial, sans-serif", size, bold=bold)
            self._font_cache[key] = f
        return f

    def _get_or_create_text_texture(self, text: str, size: int, bold: bool) -> tuple[moderngl.Texture, int, int]:
        cache_key = f"{text}_{size}_{bold}"
        entry = self._text_texture_cache.get(cache_key)
        if entry is not None:
            return entry

        font = self._get_font(size, bold)
        surf = font.render(text, True, (255, 255, 255))
        w, h = surf.get_size()
        w = max(1, w)
        h = max(1, h)

        raw_data = pygame.image.tobytes(surf, "RGBA", False)  # False = top-down matching top-down UI quad UVs
        tex = self.ctx.texture((w, h), 4, data=raw_data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        result = (tex, w, h)
        # Limit cache size to 256 text textures
        if len(self._text_texture_cache) > 256:
            # Purge older half
            old_keys = list(self._text_texture_cache.keys())[:128]
            for k in old_keys:
                t, _, _ = self._text_texture_cache.pop(k)
                t.release()

        self._text_texture_cache[cache_key] = result
        return result

    def flush(self, custom_texture: moderngl.Texture | None = None) -> None:
        """Submits all queued quads to OpenGL with alpha blending."""
        if self.quad_count == 0:
            return

        # Bind texture: either custom texture or default 1x1 white
        active_tex = custom_texture if custom_texture is not None else self._white_texture
        active_tex.use(location=0)

        # Update uniforms
        self._u_screen_size.value = (float(self.screen_width), float(self.screen_height))

        # Write vertex data
        self.vbo.write(self._vertex_data[: self.quad_count * self.FLOATS_PER_QUAD].tobytes())

        # Enable blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Render 4 vertices per quad (gl_VertexID triangle strip) instanced
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=self.quad_count)

        self.quad_count = 0

    def destroy(self) -> None:
        """Releases all ModernGL resources."""
        if self._white_texture:
            self._white_texture.release()
            self._white_texture = None
        for tex, _, _ in self._text_texture_cache.values():
            tex.release()
        self._text_texture_cache.clear()
        if self.vbo:
            self.vbo.release()
            self.vbo = None
        if self.vao:
            self.vao.release()
            self.vao = None
        if self.prog:
            self.prog.release()
            self.prog = None
