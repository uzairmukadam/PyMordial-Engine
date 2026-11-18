"""Map Manager Subsystem Host Interface.

Integrates the mapping system, script runner, trigger system, and streaming
director into PyMordial's ProjectApp runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.world.mapping.prefab_manager import PrefabManager
from engine.world.mapping.schema import (
    MapDef,
    SectorDef,
    load_map_file,
    load_sector_file,
)
from engine.world.mapping.script_runner import ScriptRunner
from engine.world.mapping.sector import SectorInstance
from engine.world.mapping.streaming_director import StreamingDirector
from engine.world.mapping.trigger_system import TriggerZoneSystem

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


class MapManager:
    """Central mapping coordinator mounted on ProjectApp."""

    __slots__ = (
        "app",
        "script_runner",
        "trigger_system",
        "streaming_director",
        "prefab_manager",
        "active_sectors",
        "active_map",
        "_observer_position",
    )

    def __init__(self, app: ProjectApp) -> None:
        self.app = app
        self.script_runner = ScriptRunner()
        self.trigger_system = TriggerZoneSystem(app)
        self.streaming_director = StreamingDirector(self)
        self.prefab_manager = PrefabManager()
        self.active_sectors: dict[str, SectorInstance] = {}
        self.active_map: MapDef | None = None
        self._observer_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def set_observer_position(self, pos: tuple[float, float, float]) -> None:
        """Updates the primary observer position (player/camera) for trigger and streaming checks."""
        self._observer_position = (float(pos[0]), float(pos[1]), float(pos[2]))

    def register_sector(self, sector: SectorInstance) -> None:
        """Registers a sector instance and hooks it into streaming and triggers."""
        self.active_sectors[sector.name] = sector
        self.streaming_director.register_sector(sector)

        # If already active, register its triggers and scripts
        if sector.is_active:
            for node in sector.root_nodes:
                self._register_node_components(node)

    def _register_node_components(self, node: MapEntityNode) -> None:
        if node.entity_def.trigger is not None:
            self.trigger_system.register_trigger(node, node.entity_def.trigger)
        for script in node.scripts:
            self.script_runner.attach_script(script)
        for child in node.children:
            self._register_node_components(child)

    def load_map(self, map_source: str | Path | MapDef) -> MapDef:
        """Loads and activates a MapDef (monolithic, linear triggers, or multi-sector)."""
        self.teardown()

        if isinstance(map_source, MapDef):
            self.active_map = map_source
        else:
            self.active_map = load_map_file(map_source)

        # Instantiate declared sectors
        for sector_entry in self.active_map.sectors:
            if isinstance(sector_entry, SectorDef):
                sector_inst = SectorInstance(sector_def=sector_entry, prefab_manager=self.prefab_manager)
            else:
                sector_inst = SectorInstance(file_path=sector_entry, prefab_manager=self.prefab_manager)
            self.register_sector(sector_inst)

        # Activate initial sector (or all sectors if monolithic)
        if self.active_map.streaming_mode == "monolithic":
            for sector in self.active_sectors.values():
                sector.activate(self.app)
                for node in sector.root_nodes:
                    self._register_node_components(node)
        else:
            init_target = self.active_map.initial_sector or (
                list(self.active_sectors.keys())[0] if self.active_sectors else None
            )
            if init_target:
                self.streaming_director.request_stream(init_target, mode="ACTIVATE")

        return self.active_map

    def find_entity(self, target_id: str) -> MapEntityNode | None:
        """Searches across all active sectors for an entity by ID."""
        for sector in self.active_sectors.values():
            found = sector.find_entity(target_id)
            if found is not None:
                return found
        return None

    def signal_entity(self, target_id: str, signal: str, payload: Any = None) -> int:
        """Delivers a named signal to all scripts attached to a target entity."""
        return self.script_runner.signal_entity(target_id, signal, payload)

    def trigger_interaction(self, origin: tuple[float, float, float], radius: float = 3.5) -> int:
        """Performs proximity-based interaction, triggering on_interact for nearby entities.

        Returns:
            Number of entities triggered.
        """
        ox, oy, oz = origin
        r_sq = radius * radius
        triggered = 0

        for sector in self.active_sectors.values():
            if not sector.is_active:
                continue
            for node in sector.root_nodes:
                triggered += self._interact_node_recursive(node, ox, oy, oz, r_sq)

        return triggered

    def _interact_node_recursive(
        self,
        node: MapEntityNode,
        ox: float,
        oy: float,
        oz: float,
        r_sq: float,
    ) -> int:
        count = 0
        px, py, pz = node.world_transform.position
        dx = px - ox
        dy = py - oy
        dz = pz - oz
        if (dx * dx + dy * dy + dz * dz) <= r_sq:
            for script in node.scripts:
                if script.is_active:
                    script.on_interact()
                    count += 1

        for child in node.children:
            count += self._interact_node_recursive(child, ox, oy, oz, r_sq)

        return count

    def on_fixed_update(self, dt: float) -> None:
        """Fixed simulation step: updates kinematic scripts, triggers, and streaming."""
        # 1. Update attached kinematic and behavioral scripts
        self.script_runner.on_fixed_update(dt)

        # 2. Evaluate trigger zones against observer position
        self.trigger_system.on_fixed_update(self._observer_position)

        # 3. Process queued staged sector activations
        self.streaming_director.on_fixed_update(dt)

    def teardown(self) -> None:
        """Cleans up all active sectors, scripts, triggers, and streaming state."""
        self.script_runner.clear()
        self.trigger_system.clear()
        for sector in self.active_sectors.values():
            sector.unload(self.app)
        self.active_sectors.clear()
        self.active_map = None
