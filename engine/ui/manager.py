"""Central UI Manager Subsystem for PyMordial Engine.

Coordinates native ModernGL UI screens, widget hierarchies,
input event dispatching, and per-frame rendering.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import moderngl
import pygame

from engine.ui.renderer import UIRenderer
from engine.ui.theme import UITheme
from engine.ui.elements import UIElement

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class UIScreen(UIElement):
    """Represents a fullscreen UI page or menu."""

    def __init__(self, name: str = "Screen") -> None:
        super().__init__(x=0.0, y=0.0, w=1920.0, h=1080.0)
        self.name = name

    def on_enter(self, app: ProjectApp) -> None:
        """Called when this screen becomes active."""
        pass

    def on_exit(self, app: ProjectApp) -> None:
        """Called when this screen is replaced or closed."""
        pass

    def update(self, dt: float) -> None:
        """Per-frame animation or state update."""
        pass


class UIManager:
    """Master UI Subsystem coordinating OpenGL UI rendering and interaction."""

    __slots__ = (
        "ctx",
        "renderer",
        "theme",
        "screen_stack",
        "_app",
        "_mouse_pos",
    )

    def __init__(self, ctx: moderngl.Context | None = None, width: int = 1920, height: int = 1080, theme: UITheme | None = None) -> None:
        self.ctx = ctx
        self.renderer: UIRenderer | None = UIRenderer(ctx, width, height) if ctx is not None else None
        self.theme = theme if theme is not None else UITheme()
        self.screen_stack: list[UIScreen] = []
        self._app: ProjectApp | None = None
        self._mouse_pos: tuple[float, float] = (0.0, 0.0)

    def attach_app(self, app: ProjectApp) -> None:
        self._app = app
        if self.renderer is not None:
            self.renderer.set_screen_size(app.config.width, app.config.height)

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
                screen.w = float(self._app.config.width)
                screen.h = float(self._app.config.height)
                screen.on_enter(self._app)

    def push_screen(self, screen: UIScreen) -> None:
        """Pushes an overlay screen onto the stack."""
        self.screen_stack.append(screen)
        if self._app:
            screen.w = float(self._app.config.width)
            screen.h = float(self._app.config.height)
            screen.on_enter(self._app)

    def pop_screen(self) -> UIScreen | None:
        """Pops the top screen from the stack."""
        if not self.screen_stack:
            return None
        popped = self.screen_stack.pop()
        if self._app:
            popped.on_exit(self._app)
        return popped

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Dispatches SDL2/PyGame events to the top UI screen. Returns True if consumed."""
        top = self.active_screen
        if top is None or not top.visible or not top.enabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            mx, my = float(event.pos[0]), float(event.pos[1])
            self._mouse_pos = (mx, my)
            return top.on_mouse_move(mx, my)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = float(event.pos[0]), float(event.pos[1])
            self._mouse_pos = (mx, my)
            return top.on_mouse_down(mx, my, event.button)

        elif event.type == pygame.MOUSEBUTTONUP:
            mx, my = float(event.pos[0]), float(event.pos[1])
            self._mouse_pos = (mx, my)
            return top.on_mouse_up(mx, my, event.button)

        return False

    def update(self, dt: float) -> None:
        """Updates active screen animations."""
        top = self.active_screen
        if top is not None and top.visible:
            top.update(dt)

    def render(self) -> None:
        """Renders active UI screens on top of the ModernGL backbuffer."""
        top = self.active_screen
        if top is None or not top.visible or self.renderer is None:
            return

        self.renderer.begin_frame()
        top.render(self.renderer, self.theme)
        self.renderer.flush()

    def destroy(self) -> None:
        """Releases GPU buffers and cached textures."""
        self.set_screen(None)
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
