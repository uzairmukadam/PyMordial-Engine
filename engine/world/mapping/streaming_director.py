"""Streaming Director for Linear Levels and Dynamic Multi-Map Portals.

Coordinates staged sector prefetching, activation, and deactivation driven by
triggers and observer position.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from engine.world.mapping.schema import SectorDef, load_sector_file
from engine.world.mapping.sector import SectorInstance, SectorState

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.map_manager import MapManager


class StreamingDirector:
    """Manages sector activation and streaming transitions."""

    __slots__ = ("map_manager", "_sectors", "_staged_queue", "activation_budget")

    def __init__(self, map_manager: MapManager, activation_budget: int = 150) -> None:
        self.map_manager = map_manager
        self._sectors: dict[str, SectorInstance] = {}
        self._staged_queue: list[SectorInstance] = []
        self.activation_budget = activation_budget

    def register_sector(self, sector: SectorInstance) -> None:
        """Registers a sector instance with the director."""
        self._sectors[sector.name] = sector
        if sector.file_path is not None:
            self._sectors[sector.file_path.name] = sector
            self._sectors[str(sector.file_path)] = sector

    def request_stream(self, target_identifier: str, mode: str = "ACTIVATE") -> SectorInstance | None:
        """Requests a sector to be streamed in (ACTIVATED), DEACTIVATED, or UNLOADED."""
        sector = self._sectors.get(target_identifier)
        if sector is None:
            # Check if target is a file path
            path = Path(target_identifier)
            if not path.is_file():
                # Check relative to app or assets/maps
                candidates = [
                    Path("projects/playground/assets/maps") / path.name,
                    Path("assets/maps") / path.name,
                ]
                for c in candidates:
                    if c.is_file():
                        path = c
                        break

            if path.is_file():
                sector = SectorInstance(file_path=path, prefab_manager=self.map_manager.prefab_manager)
                self.register_sector(sector)

        if sector is None:
            return None

        mode_upper = mode.upper()
        if mode_upper in ("ACTIVATE", "LOAD"):
            if sector.state != SectorState.ACTIVE and sector not in self._staged_queue:
                self._staged_queue.append(sector)
        elif mode_upper == "DEACTIVATE":
            if sector in self._staged_queue:
                self._staged_queue.remove(sector)
            sector.deactivate(self.map_manager.app)
        elif mode_upper == "UNLOAD":
            if sector in self._staged_queue:
                self._staged_queue.remove(sector)
            sector.unload(self.map_manager.app)

        return sector

    def on_fixed_update(self, dt: float) -> None:
        """Processes queued staged sector activations without frame hitches."""
        if not self._staged_queue:
            return

        sector = self._staged_queue[0]
        done = sector.activate(self.map_manager.app, budget=self.activation_budget)
        if done:
            self._staged_queue.pop(0)
            # Register newly activated trigger zones & scripts
            for node in sector.root_nodes:
                self._register_node_components(node)

    def _register_node_components(self, node: Any) -> None:
        # Register triggers
        if node.entity_def.trigger is not None:
            self.map_manager.trigger_system.register_trigger(node, node.entity_def.trigger)
        # Register scripts
        for script in node.scripts:
            self.map_manager.script_runner.attach_script(script)
        # Children
        for child in node.children:
            self._register_node_components(child)
