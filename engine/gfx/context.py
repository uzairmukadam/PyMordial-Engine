"""ModernGL 4.5 Core Context and Display Manager for PyMordial Engine."""

from __future__ import annotations
from typing import Optional
import moderngl

from engine.gfx.quality_presets import RenderConfig, get_quality_preset, GraphicsQuality
from engine.window import Window, WindowConfig
from engine.events import subscribe_event, WindowResizeEvent


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
        "window",
        "pipeline",
    )

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "PyMordial Engine 3D",
        hidden: bool = False,
        config: Optional[RenderConfig] = None,
        window: Optional[Window] = None,
        resizable: bool = False,
    ) -> None:
        self.config = config if config is not None else get_quality_preset(GraphicsQuality.HIGH)
        self.is_headless = hidden
        self.pipeline = None

        if window is not None:
            self.window = window
            self.width = window.width
            self.height = window.height
            self.is_headless = window.is_headless
        else:
            win_cfg = WindowConfig(
                width=width,
                height=height,
                title=title,
                hidden=hidden,
                depth_bits=24,
                resizable=resizable,
            )
            self.window = Window(win_cfg)
            self.width = self.window.width
            self.height = self.window.height

        self.screen = self.window.surface

        # Create ModernGL Core context
        try:
            self.ctx = moderngl.create_context(require=450)
        except Exception:
            try:
                # Fallback if driver requires auto-detect
                self.ctx = moderngl.create_context()
            except Exception:
                if self.is_headless:
                    self.ctx = moderngl.create_context(standalone=True)
                else:
                    raise

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

        # Auto-subscribe to window resize events
        subscribe_event(WindowResizeEvent, self._on_window_resize, priority=100)

    def _on_window_resize(self, event: WindowResizeEvent) -> None:
        """Synchronizes OpenGL viewport when window resolution changes."""
        self.resize(event.width, event.height)

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
        if self.ctx.screen is not None:
            try:
                self.ctx.screen.viewport = (0, 0, new_width, new_height)
            except Exception:
                pass

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
        self.window.swap_buffers()

    def save_screenshot(self, filepath: str) -> None:
        """Reads backbuffer pixels and saves image to disk."""
        from PIL import Image
        fbo = self.ctx.screen
        if hasattr(self, "pipeline") and self.pipeline is not None:
            pp = getattr(self.pipeline, "post_process", None)
            if pp is not None and getattr(pp, "final_fbo", None) is not None:
                fbo = pp.final_fbo
        components = 4 if fbo is self.ctx.screen else 3
        mode = "RGBA" if components == 4 else "RGB"
        raw = fbo.read(components=components, dtype="f1")
        try:
            from OpenGL import GL
            while GL.glGetError() != 0:
                pass
        except Exception:
            pass
        img = Image.frombytes(mode, (self.width, self.height), raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img.save(filepath)

    def destroy(self) -> None:
        """Releases context resources."""
        if self.ctx:
            try:
                self.ctx.release()
            except Exception:
                pass
            self.ctx = None
        if hasattr(self, "window") and self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
