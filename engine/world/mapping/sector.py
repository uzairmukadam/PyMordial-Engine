"""Sector Container and 6-Stage Lifecycle State Machine.

Encapsulates spatial chunks/sectors with stateful transitions:
UNLOADED <-> LOADING <-> LOADED <-> ACTIVATING <-> ACTIVE <-> DEACTIVATING
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.prefab_manager import PrefabManager
from engine.world.mapping.schema import SectorDef, load_sector_file

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class SectorState(Enum):
    """Lifecycle states of a map sector/chunk."""

    UNLOADED = auto()      # Only metadata/path in memory; zero runtime allocations.
    LOADING = auto()       # Asynchronously reading file/assets.
    LOADED = auto()        # SectorDef parsed in RAM, ready for instant activation.
    ACTIVATING = auto()    # Instantiating entities into ECS and Rapier within frame budget.
    ACTIVE = auto()        # Fully simulated in physics and rendered in MDI batches.
    DEACTIVATING = auto()  # Removing physics bodies and returning ECS entities to pool.


class SectorInstance:
    """Manages the runtime lifecycle and entity pool of an isolated map sector."""

    __slots__ = (
        "sector_def",
        "file_path",
        "prefab_manager",
        "state",
        "root_nodes",
        "entity_ids",
        "_activation_index",
        "_on_activate_callbacks",
        "_on_deactivate_callbacks",
    )

    def __init__(
        self,
        sector_def: SectorDef | None = None,
        file_path: str | Path | None = None,
        prefab_manager: PrefabManager | None = None,
    ) -> None:
        self.sector_def = sector_def
        self.file_path = Path(file_path) if file_path is not None else None
        self.prefab_manager = prefab_manager or PrefabManager()
        self.state: SectorState = SectorState.LOADED if sector_def is not None else SectorState.UNLOADED
        self.root_nodes: list[MapEntityNode] = []
        self.entity_ids: list[int] = []
        self._activation_index: int = 0
        self._on_activate_callbacks: list[Callable[[SectorInstance], None]] = []
        self._on_deactivate_callbacks: list[Callable[[SectorInstance], None]] = []

    @property
    def name(self) -> str:
        if self.sector_def is not None:
            return self.sector_def.name
        if self.file_path is not None:
            return self.file_path.stem
        return "unnamed_sector"

    @property
    def is_active(self) -> bool:
        return self.state == SectorState.ACTIVE

    @property
    def is_loaded(self) -> bool:
        return self.state in (SectorState.LOADED, SectorState.ACTIVATING, SectorState.ACTIVE)

    @property
    def is_unloaded(self) -> bool:
        return self.state == SectorState.UNLOADED

    @property
    def active_entity_count(self) -> int:
        return len(self.entity_ids)

    def contains_point(self, point: tuple[float, float, float]) -> bool:
        """Evaluates whether a 3D point is within this sector's bounding volume."""
        if self.sector_def is not None and self.sector_def.bounds is not None:
            return self.sector_def.bounds.contains_point(point)
        return True

    def on_activate(self, callback: Callable[[SectorInstance], None]) -> None:
        """Registers a callback fired when the sector transitions to ACTIVE."""
        self._on_activate_callbacks.append(callback)

    def on_deactivate(self, callback: Callable[[SectorInstance], None]) -> None:
        """Registers a callback fired when the sector transitions out of ACTIVE."""
        self._on_deactivate_callbacks.append(callback)

    # --------------------------------------------------------------------------
    # Lifecycle Transitions
    # --------------------------------------------------------------------------

    def load(self, app: ProjectApp | None = None) -> None:
        """Loads and parses the sector definition from disk into memory."""
        if self.state != SectorState.UNLOADED:
            return

        self.state = SectorState.LOADING
        if self.sector_def is None and self.file_path is not None:
            self.sector_def = load_sector_file(self.file_path)

        self.state = SectorState.LOADED

    def activate(self, app: ProjectApp, budget: int | None = None) -> bool:
        """Instantiates entities into ECS and Rapier simulation.

        Args:
            app: The active ProjectApp instance.
            budget: Maximum entities to spawn this call. If omitted or large enough,
                    completes immediately and returns True. If partitioned, returns
                    False until all entities have been instantiated.

        Returns:
            True if activation is complete (ACTIVE); False if still ACTIVATING.
        """
        # Ensure sector data is loaded in RAM
        if self.state == SectorState.UNLOADED:
            self.load(app)

        if self.state == SectorState.ACTIVE:
            return True

        if self.sector_def is None:
            self.state = SectorState.ACTIVE
            return True

        if self.state == SectorState.LOADED:
            self.state = SectorState.ACTIVATING
            self._activation_index = 0
            self.root_nodes.clear()
            self.entity_ids.clear()

        # Staged or instantaneous entity spawning
        entities = self.sector_def.entities
        total = len(entities)
        max_to_spawn = budget if budget is not None and budget > 0 else (total - self._activation_index)
        spawned_this_frame = 0

        while self._activation_index < total and spawned_this_frame < max_to_spawn:
            ent_def = entities[self._activation_index]
            node = MapEntityNode(ent_def)
            node.spawn(app, prefab_mgr=self.prefab_manager)
            self.root_nodes.append(node)
            self.entity_ids.extend(node.collect_all_ecs_ids())

            self._activation_index += 1
            spawned_this_frame += 1

        # Check if finished
        if self._activation_index >= total:
            self.state = SectorState.ACTIVE
            for cb in self._on_activate_callbacks:
                cb(self)
            return True

        return False

    def deactivate(self, app: ProjectApp) -> None:
        """Tears down live ECS entities and Rapier bodies, preserving parsed definitions."""
        if self.state not in (SectorState.ACTIVE, SectorState.ACTIVATING):
            return

        self.state = SectorState.DEACTIVATING

        for cb in self._on_deactivate_callbacks:
            cb(self)

        for node in self.root_nodes:
            node.teardown(app)
        self.root_nodes.clear()
        self.entity_ids.clear()
        self._activation_index = 0

        self.state = SectorState.LOADED

    def unload(self, app: ProjectApp) -> None:
        """Completely flushes this sector from memory, deactivating it first if active."""
        if self.state in (SectorState.ACTIVE, SectorState.ACTIVATING):
            self.deactivate(app)

        # Clear cached definition if we can reload from disk later
        if self.file_path is not None:
            self.sector_def = None

        self.state = SectorState.UNLOADED

    # --------------------------------------------------------------------------
    # Entity Queries
    # --------------------------------------------------------------------------

    def find_entity(self, target_id: str) -> MapEntityNode | None:
        """Searches for a root entity or child entity by string ID."""
        for node in self.root_nodes:
            found = node.find_by_id(target_id)
            if found is not None:
                return found
        return None
