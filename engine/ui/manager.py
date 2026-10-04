"""Central UI Manager Subsystem for PyMordial Engine.

Coordinates native ModernGL UI screens, widget hierarchies,
input event dispatching, and per-frame rendering.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import moderngl
import pygame

from engine.events import subscribe_event, unsubscribe_event, WindowResizeEvent
from engine.ui.renderer import UIRenderer
from engine.ui.theme import UITheme
from engine.ui.elements import UIElement

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class UIScreen(UIElement):
    """Represents a fullscreen UI page or menu."""

    def __init__(self, name: str = "Screen", is_modal: bool = True) -> None:
        super().__init__(x=0.0, y=0.0, w=1920.0, h=1080.0)
        self.name = name
        self.is_modal = is_modal

    def on_enter(self, app: ProjectApp) -> None:
        """Called when this screen becomes active."""
        pass

    def on_exit(self, app: ProjectApp) -> None:
        """Called when this screen is replaced or closed."""
        pass

    def update(self, dt: float) -> None:
        """Per-frame animation or state update."""
        pass

    def on_mouse_move(self, mx: float, my: float) -> bool:
        if not self.visible or not self.enabled:
            return False
        handled = False
        for c in reversed(self.children):
            if c.on_mouse_move(mx, my):
                handled = True
                break
        self.is_hovered = self.hit_test(mx, my)
        return handled or (self.is_modal and self.is_hovered)

    def on_mouse_down(self, mx: float, my: float, button: int) -> bool:
        if not self.visible or not self.enabled:
            return False
        for c in reversed(self.children):
            if c.on_mouse_down(mx, my, button):
                return True
        return self.is_modal and self.hit_test(mx, my)

    def on_mouse_up(self, mx: float, my: float, button: int) -> bool:
        if not self.visible or not self.enabled:
            return False
        for c in reversed(self.children):
            if c.on_mouse_up(mx, my, button):
                return True
        return self.is_modal and self.hit_test(mx, my)


class UIManager:
    """Master UI Subsystem coordinating OpenGL UI rendering, captured AAA cursor, and interaction."""

    __slots__ = (
        "ctx",
        "renderer",
        "theme",
        "screen_stack",
        "width",
        "height",
        "cursor_pos",
        "cursor_visible",
        "_app",
        "_mouse_pos",
        "_is_clicking",
        "_cursor_texture",
    )

    def __init__(self, ctx: moderngl.Context | None = None, width: int = 1920, height: int = 1080, theme: UITheme | None = None) -> None:
        self.ctx = ctx
        self.width = max(1, width)
        self.height = max(1, height)
        self.renderer: UIRenderer | None = UIRenderer(ctx, width, height) if ctx is not None else None
        self.theme = theme if theme is not None else UITheme()
        self.screen_stack: list[UIScreen] = []
        self._app: ProjectApp | None = None
        self.cursor_pos: list[float] = [float(width) * 0.5, float(height) * 0.5]
        self.cursor_visible: bool = False
        self._mouse_pos: tuple[float, float] = (self.cursor_pos[0], self.cursor_pos[1])
        self._is_clicking: bool = False
        self._cursor_texture: moderngl.Texture | None = None

    def attach_app(self, app: ProjectApp) -> None:
        self._app = app
        self.width = app.config.width
        self.height = app.config.height
        if self.renderer is not None:
            self.renderer.set_screen_size(self.width, self.height)
        subscribe_event(WindowResizeEvent, self._on_window_resize, priority=85)

    def _on_window_resize(self, event: WindowResizeEvent) -> None:
        self.width = event.width
        self.height = event.height
        if self.renderer is not None:
            self.renderer.set_screen_size(event.width, event.height)
        for screen in self.screen_stack:
            screen.w = float(event.width)
            screen.h = float(event.height)

    @property
    def active_screen(self) -> UIScreen | None:
        return self.screen_stack[-1] if self.screen_stack else None

    def set_screen(self, screen: UIScreen | None) -> None:
        """Replaces the active screen."""
        while self.screen_stack:
            prev = self.screen_stack.pop()
            if self._app:
                prev.on_exit(self._app)

        if screen is not None:
            self.screen_stack.append(screen)
            if self._app:
                screen.w = float(self.width)
                screen.h = float(self.height)
                screen.on_enter(self._app)

    def push_screen(self, screen: UIScreen) -> None:
        """Pushes an overlay screen onto the stack."""
        self.screen_stack.append(screen)
        if self._app:
            screen.w = float(self.width)
            screen.h = float(self.height)
            screen.on_enter(self._app)

    def pop_screen(self) -> UIScreen | None:
        """Pops the top screen from the stack."""
        if not self.screen_stack:
            return None
        popped = self.screen_stack.pop()
        if self._app:
            popped.on_exit(self._app)
        return popped

    def _get_or_create_cursor_texture(self) -> moderngl.Texture | None:
        """Generates a crisp AAA hardware-accelerated pointer cursor with dark outline and cyan core."""
        if self._cursor_texture is not None:
            return self._cursor_texture
        if self.ctx is None:
            return None

        size = 32
        surf = pygame.Surface((size, size), pygame.SRCALPHA)

        # 1. Dark obsidian drop-shadow outline for high contrast
        shadow_pts = [(2, 2), (2, 23), (7, 18), (12, 24), (16, 20), (10, 14), (18, 14)]
        pygame.draw.polygon(surf, (10, 15, 26, 180), shadow_pts)

        # 2. Outer crisp dark border
        outline_pts = [(0, 0), (0, 21), (5, 16), (10, 22), (14, 18), (9, 13), (17, 13)]
        pygame.draw.polygon(surf, (15, 23, 42, 255), outline_pts)

        # 3. Inner clean bright silver-white body
        inner_pts = [(1, 2), (1, 18), (5, 14), (9, 19), (12, 17), (7, 12), (14, 12)]
        pygame.draw.polygon(surf, (245, 248, 255, 255), inner_pts)

        # 4. Futuristic glowing cyan accent core
        core_pts = [(2, 4), (2, 11), (5, 11)]
        pygame.draw.polygon(surf, (56, 189, 248, 255), core_pts)

        raw_data = pygame.image.tobytes(surf, "RGBA", False)
        tex = self.ctx.texture((size, size), 4, data=raw_data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._cursor_texture = tex
        return self._cursor_texture

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Dispatches SDL2/PyGame events to the top UI screen using virtual cursor coordinates."""
        top = self.active_screen
        if top is None or not top.visible or not top.enabled:
            return False

        if not self.cursor_visible and not top.is_modal:
            return False

        if event.type == pygame.MOUSEMOTION:
            rel = getattr(event, "rel", (0, 0))
            is_grabbed = bool(pygame.display.get_init() and pygame.event.get_grab())
            if is_grabbed or (self.cursor_visible and rel != (0, 0)):
                self.cursor_pos[0] = max(0.0, min(float(self.width), self.cursor_pos[0] + float(rel[0])))
                self.cursor_pos[1] = max(0.0, min(float(self.height), self.cursor_pos[1] + float(rel[1])))
            else:
                self.cursor_pos[0] = float(event.pos[0])
                self.cursor_pos[1] = float(event.pos[1])

            self._mouse_pos = (self.cursor_pos[0], self.cursor_pos[1])
            return top.on_mouse_move(self.cursor_pos[0], self.cursor_pos[1])

        elif event.type == pygame.MOUSEBUTTONDOWN:
            is_grabbed = bool(pygame.display.get_init() and pygame.event.get_grab())
            if not is_grabbed and not self.cursor_visible:
                self.cursor_pos[0] = float(event.pos[0])
                self.cursor_pos[1] = float(event.pos[1])
            self._mouse_pos = (self.cursor_pos[0], self.cursor_pos[1])
            self._is_clicking = True
            return top.on_mouse_down(self.cursor_pos[0], self.cursor_pos[1], event.button)

        elif event.type == pygame.MOUSEBUTTONUP:
            is_grabbed = bool(pygame.display.get_init() and pygame.event.get_grab())
            if not is_grabbed and not self.cursor_visible:
                self.cursor_pos[0] = float(event.pos[0])
                self.cursor_pos[1] = float(event.pos[1])
            self._mouse_pos = (self.cursor_pos[0], self.cursor_pos[1])
            self._is_clicking = False
            return top.on_mouse_up(self.cursor_pos[0], self.cursor_pos[1], event.button)

        return False

    def update(self, dt: float) -> None:
        """Updates active screen animations and handles gamepad virtual cursor navigation."""
        top = self.active_screen
        if top is not None and top.visible:
            top.update(dt)

            # Gamepad Left Analog Stick virtual cursor movement
            if self.cursor_visible and self._app is not None and hasattr(self._app, "input_manager"):
                pad_move = self._app.input_manager.get_vector2("move")
                if abs(pad_move[0]) > 0.08 or abs(pad_move[1]) > 0.08:
                    speed = 950.0  # Pixels per second
                    self.cursor_pos[0] = max(0.0, min(float(self.width), self.cursor_pos[0] + pad_move[0] * speed * dt))
                    self.cursor_pos[1] = max(0.0, min(float(self.height), self.cursor_pos[1] - pad_move[1] * speed * dt))
                    self._mouse_pos = (self.cursor_pos[0], self.cursor_pos[1])
                    top.on_mouse_move(self.cursor_pos[0], self.cursor_pos[1])

    def render(self) -> None:
        """Renders active UI screens and in-engine captured cursor on top of the ModernGL backbuffer."""
        top = self.active_screen
        if top is None or not top.visible or self.renderer is None:
            return

        self.renderer.begin_frame()
        top.render(self.renderer, self.theme)

        # Draw sleek AAA in-engine captured cursor if visible
        if self.cursor_visible:
            tex = self._get_or_create_cursor_texture()
            if tex is not None:
                offset = 1.0 if self._is_clicking else 0.0
                self.renderer.draw_textured_rect(
                    x=self.cursor_pos[0] + offset,
                    y=self.cursor_pos[1] + offset,
                    w=24.0,
                    h=24.0,
                    texture=tex,
                    tint=(1.0, 1.0, 1.0, 1.0),
                    corner_radius=0.0,
                )

        self.renderer.flush()

    def destroy(self) -> None:
        """Releases GPU buffers and cached textures."""
        unsubscribe_event(WindowResizeEvent, self._on_window_resize)
        self.set_screen(None)
        if self._cursor_texture is not None:
            try:
                self._cursor_texture.release()
            except Exception:
                pass
            self._cursor_texture = None
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
