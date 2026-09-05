"""ModernGL 4.5 Core Context and Display Manager for PyMordial Engine."""

from __future__ import annotations
from typing import Optional
import pygame
import moderngl

from engine.gfx.quality_presets import RenderConfig, get_quality_preset, GraphicsQuality


class RenderContext:
    """Manages the ModernGL 4.5 Core Profile OpenGL context and display window."""

    __slots__ = (
        "width",
        "height",
        "screen",
        "ctx",
        "config",
        "is_headless",
        "depth_func_name",
    )

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "PyMordial Engine 3D",
        hidden: bool = False,
        config: Optional[RenderConfig] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.config = config if config is not None else get_quality_preset(GraphicsQuality.HIGH)
        self.is_headless = hidden

        if not pygame.get_init():
            pygame.init()

        # Request OpenGL 4.5 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 5)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if hidden:
            flags |= pygame.HIDDEN

        self.screen = pygame.display.set_mode((width, height), flags)
        if not hidden:
            pygame.display.set_caption(title)

        # Create ModernGL Core context
        try:
            self.ctx = moderngl.create_context(require=450)
        except Exception:
            # Fallback if driver requires auto-detect
            self.ctx = moderngl.create_context()

        # Configure standard 3D state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"

        # Apply Reversed-Z depth function if enabled
        if self.config.reverse_z:
            # In Reversed-Z: near = 1.0, far = 0.0, so closer geometry has GREATER depth
            self.ctx.depth_func = ">"
            self.depth_func_name = ">"
            try:
                import ctypes
                opengl32 = ctypes.windll.opengl32
                wglGetProcAddress = opengl32.wglGetProcAddress
                wglGetProcAddress.restype = ctypes.c_void_p
                wglGetProcAddress.argtypes = [ctypes.c_char_p]
                proc = wglGetProcAddress(b"glClipControl")
                if proc:
                    clip_func = ctypes.WINFUNCTYPE(None, ctypes.c_uint, ctypes.c_uint)(proc)
                    # GL_LOWER_LEFT = 0x8CA1, GL_ZERO_TO_ONE = 0x935F
                    clip_func(0x8CA1, 0x935F)
            except Exception:
                pass
        else:
            self.ctx.depth_func = "<"
            self.depth_func_name = "<"

    @classmethod
    def create_headless(cls, width: int = 800, height: int = 600) -> RenderContext:
        """Creates a headless hidden OpenGL 4.5 context suitable for automated unit tests."""
        return cls(width=width, height=height, hidden=True)

    def resize(self, new_width: int, new_height: int) -> None:
        """Updates internal viewport resolution."""
        if new_width <= 0 or new_height <= 0:
            return
        self.width = new_width
        self.height = new_height
        self.ctx.viewport = (0, 0, new_width, new_height)

    def clear(
        self,
        color: tuple[float, float, float, float] = (0.05, 0.06, 0.08, 1.0),
        depth: Optional[float] = None,
    ) -> None:
        """Clears the default framebuffer with proper Reversed-Z depth defaults."""
        if depth is None:
            # In Reversed-Z, clear depth to 0.0 (farthest distance)
            depth = 0.0 if self.config.reverse_z else 1.0
        self.ctx.clear(
            color[0],
            color[1],
            color[2],
            color[3],
            depth=depth,
        )

    def swap_buffers(self) -> None:
        """Swaps front and back buffers."""
        if not self.is_headless:
            pygame.display.flip()

    def destroy(self) -> None:
        """Releases context resources."""
        if self.ctx:
            self.ctx.release()
