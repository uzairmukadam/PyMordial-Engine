"""Project Module Protocol and Base Class.

Defines the standard lifecycle contract for pluggable, self-contained game features.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pygame

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class ProjectModule:
    """Base class for all modular game features and plugins."""

    name: str = "Module"
    enabled: bool = True

    def on_attach(self, app: ProjectApp) -> None:
        """Called once when the module is attached to the active project application."""
        pass

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        """Called during fixed 60 Hz simulation ticks (physics, motor, kinematics)."""
        pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        """Called during variable render frames (input polling, camera, animations)."""
        pass

    def on_ui(self, app: ProjectApp) -> None:
        """Called during the Dear ImGui or HUD drawing pass."""
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        """Called for polled window/input events. Returns True if consumed."""
        return False

    def on_detach(self, app: ProjectApp) -> None:
        """Called when the module is detached or project is shutting down."""
        pass
