"""Map-Driven World Builder.

Implements BaseWorldBuilder to construct and teardown scenes defined by
declarative SectorDef or MapDef structures, backed by SectorInstance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from engine.world.base import BaseWorldBuilder
from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.prefab_manager import PrefabManager
from engine.world.mapping.schema import (
    MapDef,
    SectorDef,
    load_map_file,
    load_sector_file,
)
from engine.world.mapping.sector import SectorInstance

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class MapWorldBuilder(BaseWorldBuilder):
    """World builder driven by declarative mapping sectors and SectorInstance."""

    __slots__ = (
        "sector_def",
        "map_def",
        "file_path",
        "prefab_manager",
        "sector_instance",
        "nodes",
        "entity_ids",
    )

    def __init__(
        self,
        sector_def: SectorDef | None = None,
        map_def: MapDef | None = None,
        file_path: str | Path | None = None,
        prefab_manager: PrefabManager | None = None,
    ) -> None:
        self.sector_def = sector_def
        self.map_def = map_def
        self.file_path = Path(file_path) if file_path is not None else None
        self.prefab_manager = prefab_manager or PrefabManager()
        self.sector_instance: SectorInstance | None = None
        self.nodes: list[MapEntityNode] = []
        self.entity_ids: list[int] = []

    def build_world(self, app: ProjectApp) -> None:
        """Constructs geometry, static colliders, lighting, and spawn environments."""
        self.teardown_world(app)

        # 1. Resolve SectorDef if file_path is given
        target_sector = self.sector_def
        if target_sector is None and self.file_path is not None:
            if str(self.file_path).endswith("map.json"):
                self.map_def = load_map_file(self.file_path)
            else:
                target_sector = load_sector_file(self.file_path)

        if target_sector is None and self.map_def is not None:
            # For monolithic or single initial sector
            if self.map_def.sectors:
                first = self.map_def.sectors[0]
                if isinstance(first, SectorDef):
                    target_sector = first
                elif isinstance(first, str):
                    target_sector = load_sector_file(first)

        if target_sector is None:
            return

        # 2. Instantiate and activate sector instance
        self.sector_instance = SectorInstance(
            sector_def=target_sector,
            prefab_manager=self.prefab_manager,
        )
        self.sector_instance.activate(app)

        # 3. Cache references for backwards compatibility
        self.nodes = list(self.sector_instance.root_nodes)
        self.entity_ids = list(self.sector_instance.entity_ids)

    def teardown_world(self, app: ProjectApp) -> None:
        """Cleans up static colliders and frees mesh allocations."""
        if self.sector_instance is not None:
            self.sector_instance.deactivate(app)
            self.sector_instance = None
        self.nodes.clear()
        self.entity_ids.clear()
