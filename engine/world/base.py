"""Base World and Map Builder Interface.

Encapsulates map generation, terrain construction, static colliders, and level loading.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class BaseWorldBuilder:
    """Interface for world generation and map building."""

    def build_world(self, app: ProjectApp) -> None:
        """Constructs geometry, static colliders, lighting, and spawn environments.

        Args:
            app: The active ProjectApp instance.
        """
        raise NotImplementedError

    def teardown_world(self, app: ProjectApp) -> None:
        """Cleans up static colliders and frees mesh allocations.

        Args:
            app: The active ProjectApp instance.
        """
        pass
