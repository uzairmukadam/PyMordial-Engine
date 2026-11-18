"""PyMordial Engine Mapping Subsystem.

Provides modular, data-driven mapping, sector lifecycle management,
hierarchical compound entity/prefab composition, interactive kinematics,
trigger zones, and seamless level streaming.
"""

from __future__ import annotations

from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.map_manager import MapManager
from engine.world.mapping.prefab_manager import PrefabManager
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    MapDef,
    ScriptDef,
    SectorBoundsDef,
    SectorDef,
    TransformDef,
    TriggerActionDef,
    TriggerDef,
    VisualDef,
    load_map_file,
    load_sector_file,
    save_map_file,
    save_sector_file,
)
from engine.world.mapping.script_runner import (
    EntityScript,
    ScriptRunner,
    get_script_class,
    register_script_type,
)
from engine.world.mapping.scripts import (
    InteractiveLeverScript,
    MovingPlatformScript,
    SlidingGateScript,
)
from engine.world.mapping.sector import SectorInstance, SectorState
from engine.world.mapping.streaming_director import StreamingDirector
from engine.world.mapping.transform_math import (
    combine_transforms,
    quat_multiply,
    rotate_vec3_by_quat,
)
from engine.world.mapping.trigger_system import TriggerZone, TriggerZoneSystem
from engine.world.mapping.world_builder import MapWorldBuilder

__all__ = [
    "TransformDef",
    "VisualDef",
    "CollisionDef",
    "TriggerActionDef",
    "TriggerDef",
    "ScriptDef",
    "EntityDef",
    "SectorBoundsDef",
    "SectorDef",
    "MapDef",
    "load_sector_file",
    "save_sector_file",
    "load_map_file",
    "save_map_file",
    "combine_transforms",
    "quat_multiply",
    "rotate_vec3_by_quat",
    "MapEntityNode",
    "PrefabManager",
    "SectorState",
    "SectorInstance",
    "MapWorldBuilder",
    "EntityScript",
    "ScriptRunner",
    "register_script_type",
    "get_script_class",
    "MovingPlatformScript",
    "InteractiveLeverScript",
    "SlidingGateScript",
    "TriggerZone",
    "TriggerZoneSystem",
    "StreamingDirector",
    "MapManager",
]
