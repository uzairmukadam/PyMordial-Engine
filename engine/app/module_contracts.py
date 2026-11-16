"""Role-Specific Module Contracts for PyMordial Engine.

``ProjectModule`` is the gameplay callback shape. Render features and tools need
different lifecycle hooks, so they get their own base classes here. All three
share the same metadata contract (``MODULE_MANIFEST``) and are hosted by the
same ``ModuleHost``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.app.module import ProjectModule
from engine.app.module_manifest import ModuleManifest, ModuleRole

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.gfx.render_services import RenderServices


class GameplayModule(ProjectModule):
    """Gameplay/system module: event handling, fixed update, variable update, UI.

    This is the existing ``ProjectModule`` contract with an explicit role. It
    exists so that ``ModuleRole.GAMEPLAY`` has a named base class; subclassing
    ``ProjectModule`` directly remains fully supported.
    """

    role = ModuleRole.GAMEPLAY


class RenderFeatureModule:
    """Render feature module: declares passes and the resources they consume/produce.

    A render feature never touches ``RenderPipeline`` internals. It receives a
    ``RenderServices`` handle during ``on_register`` and returns the passes it
    owns. The pipeline schedules those passes through the validated render graph.
    """

    role = ModuleRole.RENDER_FEATURE
    MODULE_MANIFEST: ModuleManifest | None = None

    def on_register(self, services: RenderServices) -> list[Any]:
        """Returns the render passes this feature contributes.

        Args:
            services: Stable renderer surface (context, config, frame context,
                G-buffer, CSM, HDR target, texture-unit reservations).
        """
        return []

    def on_unregister(self) -> None:
        """Releases resources owned by this feature. Called before pass destruction."""
        pass


class ToolModule:
    """Developer tool module: debug panels, inspectors, authoring helpers.

    Tool modules are stripped from release builds by the packager.
    """

    role = ModuleRole.TOOL
    MODULE_MANIFEST: ModuleManifest | None = None

    def on_attach(self, app: ProjectApp) -> None:
        pass

    def on_ui(self, app: ProjectApp) -> None:
        pass

    def on_detach(self, app: ProjectApp) -> None:
        pass


__all__ = [
    "GameplayModule",
    "ProjectModule",
    "RenderFeatureModule",
    "ToolModule",
]
