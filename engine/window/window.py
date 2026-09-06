"""AAA Windowing System for PyMordial Engine.

Supports runtime dynamic display mode switching (Windowed, Borderless, Exclusive),
resolution changes, VSync toggling, DPI/monitor queries, and lifecycle event dispatch.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import sys
import pygame

from engine.events import (
    publish_event,
    WindowResizeEvent,
    WindowModeChangedEvent,
    WindowFocusEvent,
    WindowCloseEvent,
)
from engine.logging import log_info, LogChannel


class WindowMode(IntEnum):
    """Supported display modes."""
    WINDOWED = 0
    BORDERLESS_FULLSCREEN = 1
    EXCLUSIVE_FULLSCREEN = 2


class VSyncMode(IntEnum):
    """VSync synchronization modes."""
    OFF = 0
    ON = 1
    ADAPTIVE = -1


@dataclass(slots=True)
class DisplayMonitor:
    """Metadata representing a connected physical display monitor."""
    index: int
    name: str
    width: int
    height: int
    refresh_rate: int
    is_primary: bool = True


@dataclass(slots=True)
class WindowConfig:
    """Configuration blueprint for window initialization."""
    width: int = 1280
    height: int = 720
    title: str = "PyMordial Engine 3D"
    mode: WindowMode = WindowMode.WINDOWED
    vsync: VSyncMode = VSyncMode.OFF
    resizable: bool = False
    hidden: bool = False
    depth_bits: int = 24


class Window:
    """Platform window and display surface coordinator."""

    __slots__ = (
        "config",
        "width",
        "height",
        "title",
        "mode",
        "vsync",
        "is_headless",
        "has_focus",
        "is_minimized",
        "surface",
        "_desktop_width",
        "_desktop_height",
        "_windowed_width",
        "_windowed_height",
    )

    def __init__(self, config: WindowConfig | None = None) -> None:
        self.config = config if config is not None else WindowConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.title = self.config.title
        self.mode = self.config.mode
        self.vsync = self.config.vsync
        self.is_headless = self.config.hidden
        self.has_focus = True
        self.is_minimized = False

        self._windowed_width = self.width
        self._windowed_height = self.height

        # Enable Per-Monitor DPI Awareness on Windows to prevent DWM scaling distortion
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

        if not pygame.get_init():
            pygame.init()

        # Query desktop resolution
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes and len(sizes) > 0:
                self._desktop_width, self._desktop_height = sizes[0]
            else:
                display_info = pygame.display.Info()
                self._desktop_width = display_info.current_w if display_info.current_w > 0 else 1920
                self._desktop_height = display_info.current_h if display_info.current_h > 0 else 1080
        except Exception:
            self._desktop_width = 1920
            self._desktop_height = 1080

        # Configure OpenGL attributes
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 5)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, self.config.depth_bits)

        self.surface = self._create_surface()
        if not self.is_headless:
            pygame.display.set_caption(self.title)
            self.set_vsync(self.vsync)

        log_info(
            LogChannel.WINDOW,
            f"Window initialized: {self.width}x{self.height} mode={self.mode.name} vsync={self.vsync.name}",
        )

    def _create_surface(self) -> pygame.Surface | None:
        """Constructs or re-creates the pygame display surface according to current mode."""
        flags = pygame.OPENGL | pygame.DOUBLEBUF

        if self.is_headless:
            flags |= pygame.HIDDEN
            return pygame.display.set_mode((self.width, self.height), flags)

        vsync_val = 1 if self.vsync == VSyncMode.ON else 0

        if self.mode == WindowMode.EXCLUSIVE_FULLSCREEN:
            flags |= pygame.FULLSCREEN
            w, h = self.width, self.height
        elif self.mode == WindowMode.BORDERLESS_FULLSCREEN:
            flags |= pygame.NOFRAME
            w, h = self._desktop_width, self._desktop_height
            self.width, self.height = w, h
        else:  # WINDOWED
            if self.config.resizable:
                flags |= pygame.RESIZABLE
            w, h = self.width, self.height

        try:
            surf = pygame.display.set_mode((w, h), flags, vsync=vsync_val)
        except TypeError:
            # Fallback if SDL2 build doesn't accept vsync kwarg
            surf = pygame.display.set_mode((w, h), flags)

        return surf

    def set_mode(self, mode: WindowMode) -> None:
        """Transitions display mode on the fly (Windowed <-> Borderless <-> Exclusive)."""
        if mode == self.mode:
            return

        old_mode = self.mode
        self.mode = mode

        if old_mode == WindowMode.WINDOWED:
            # Save windowed dimensions to restore later
            self._windowed_width = self.width
            self._windowed_height = self.height

        if mode == WindowMode.WINDOWED:
            self.width = self._windowed_width
            self.height = self._windowed_height
        elif mode == WindowMode.BORDERLESS_FULLSCREEN:
            self.width = self._desktop_width
            self.height = self._desktop_height

        if hasattr(pygame, "Window") and not self.is_headless:
            try:
                win = pygame.Window.from_display_module()
                if mode == WindowMode.BORDERLESS_FULLSCREEN:
                    win.borderless = True
                    win.size = (self._desktop_width, self._desktop_height)
                    win.position = (0, 0)
                elif mode == WindowMode.EXCLUSIVE_FULLSCREEN:
                    win.set_fullscreen(desktop=False)
                else:  # WINDOWED
                    win.borderless = False
                    win.size = (self.width, self.height)
                    win.position = pygame.WINDOWPOS_CENTERED
            except Exception:
                self.surface = self._create_surface()
        elif not self.is_headless:
            self.surface = self._create_surface()

        if not self.is_headless:
            pygame.display.set_caption(self.title)

        log_info(LogChannel.WINDOW, f"Display mode switched: {old_mode.name} -> {mode.name} ({self.width}x{self.height})")
        publish_event(WindowModeChangedEvent(mode=int(mode), width=self.width, height=self.height))
        publish_event(WindowResizeEvent(width=self.width, height=self.height))

    def set_resolution(self, width: int, height: int) -> None:
        """Resizes the display window at runtime."""
        if width <= 0 or height <= 0 or (width == self.width and height == self.height):
            return

        self.width = width
        self.height = height
        if self.mode == WindowMode.WINDOWED:
            self._windowed_width = width
            self._windowed_height = height

        if not self.is_headless:
            if hasattr(pygame, "Window"):
                try:
                    win = pygame.Window.from_display_module()
                    win.size = (width, height)
                    if self.mode == WindowMode.WINDOWED:
                        win.position = pygame.WINDOWPOS_CENTERED
                except Exception:
                    self.surface = self._create_surface()
            else:
                self.surface = self._create_surface()
            pygame.display.set_caption(self.title)

        log_info(LogChannel.WINDOW, f"Resolution changed to {width}x{height}")
        publish_event(WindowResizeEvent(width=width, height=height))

    def set_vsync(self, vsync: VSyncMode) -> None:
        """Toggles vertical synchronization at runtime without surface recreation."""
        self.vsync = vsync
        interval = 1 if vsync == VSyncMode.ON else 0
        applied = False
        if sys.platform.startswith("win") and not self.is_headless:
            try:
                import ctypes
                lib = ctypes.WinDLL("opengl32.dll")
                wgl = lib.wglGetProcAddress
                wgl.restype = ctypes.c_void_p
                wgl.argtypes = [ctypes.c_char_p]
                proc = wgl(b"wglSwapIntervalEXT")
                if proc:
                    fn = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int)(proc)
                    applied = bool(fn(interval))
            except Exception:
                applied = False
        log_info(LogChannel.WINDOW, f"VSync set to {vsync.name} (wgl={applied})")

    def toggle_fullscreen(self) -> None:
        """Toggles between Windowed and Borderless Fullscreen."""
        if self.mode == WindowMode.WINDOWED:
            self.set_mode(WindowMode.BORDERLESS_FULLSCREEN)
        else:
            self.set_mode(WindowMode.WINDOWED)

    def set_cursor_visible(self, visible: bool) -> None:
        """Shows or hides the system mouse cursor."""
        if not self.is_headless:
            pygame.mouse.set_visible(visible)

    def set_cursor_grab(self, grabbed: bool) -> None:
        """Locks the mouse cursor to the window bounds (relative mode)."""
        if not self.is_headless:
            pygame.event.set_grab(grabbed)

    def is_cursor_grabbed(self) -> bool:
        """Queries if mouse grab is currently active."""
        return pygame.event.get_grab() if not self.is_headless else False

    def swap_buffers(self) -> None:
        """Presents backbuffer to the display surface."""
        if not self.is_headless:
            pygame.display.flip()

    def get_supported_resolutions(self) -> list[tuple[int, int]]:
        """Queries standard resolutions up to current desktop size."""
        candidates = [
            (800, 600),
            (1024, 768),
            (1280, 720),
            (1600, 900),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ]
        max_w = max(self._desktop_width, self.width)
        max_h = max(self._desktop_height, self.height)
        res = [r for r in candidates if r[0] <= max_w and r[1] <= max_h]
        if not res:
            res.append((self.width, self.height))
        return res

    def poll_window_events(self, event: pygame.event.Event) -> None:
        """Processes OS window events and dispatches PyMordial events."""
        if event.type == pygame.VIDEORESIZE:
            new_w, new_h = event.w, event.h
            if new_w > 0 and new_h > 0 and (new_w != self.width or new_h != self.height):
                self.width = new_w
                self.height = new_h
                self._windowed_width = new_w
                self._windowed_height = new_h
                publish_event(WindowResizeEvent(width=new_w, height=new_h))

        elif event.type == pygame.WINDOWFOCUSGAINED:
            self.has_focus = True
            publish_event(WindowFocusEvent(has_focus=True))

        elif event.type == pygame.WINDOWFOCUSLOST:
            self.has_focus = False
            publish_event(WindowFocusEvent(has_focus=False))

        elif event.type == pygame.WINDOWMINIMIZED:
            self.is_minimized = True

        elif event.type == pygame.WINDOWRESTORED:
            self.is_minimized = False

        elif event.type == pygame.QUIT:
            publish_event(WindowCloseEvent())
