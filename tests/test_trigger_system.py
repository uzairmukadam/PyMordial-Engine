"""Unit Tests for Trigger Zone Subsystem and Event Dispatching."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    ScriptDef,
    SectorDef,
    TransformDef,
    TriggerActionDef,
    TriggerDef,
    VisualDef,
)
from engine.world.mapping.sector import SectorInstance


@pytest.fixture
def headless_app() -> ProjectApp:
    cfg = ProjectConfig(
        headless=True,
        enable_debug=False,
        max_entities=1000,
        atmosphere_enabled=False,
        particles_enabled=False,
    )
    return ProjectApp(cfg)


def test_trigger_zone_overlap_and_signal_dispatch(headless_app: ProjectApp) -> None:
    app = headless_app

    # Gate entity
    gate_def = EntityDef(
        id="iron_door",
        transform=TransformDef(position=(0.0, 0.0, 20.0)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(body_type="kinematic_position", shape="box", dimensions=(1.0, 2.0, 0.1)),
        scripts=[ScriptDef("SlidingGateScript", {"open_offset": [0.0, 4.0, 0.0]})],
    )

    # Invisible trigger zone at (0, 1, 5) with box dimensions (2, 2, 2)
    trigger_def = EntityDef(
        id="proximity_sensor",
        transform=TransformDef(position=(0.0, 1.0, 5.0)),
        trigger=TriggerDef(
            shape="box",
            dimensions=(2.0, 2.0, 2.0),
            one_shot=False,
            on_enter=[TriggerActionDef("SIGNAL_ENTITY", target="iron_door", params={"signal": "OPEN"})],
            on_exit=[TriggerActionDef("SIGNAL_ENTITY", target="iron_door", params={"signal": "CLOSE"})],
        ),
    )

    sector = SectorInstance(sector_def=SectorDef(name="trigger_sector", entities=[gate_def, trigger_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    gate_node = sector.find_entity("iron_door")
    assert gate_node is not None
    gate_script = gate_node.scripts[0]
    assert gate_script.is_open is False

    # 1. Observer is outside (0, 1, 0)
    app.map_manager.set_observer_position((0.0, 1.0, 0.0))
    app.map_manager.on_fixed_update(1.0 / 60.0)
    assert gate_script.is_open is False

    # 2. Observer enters trigger (0, 1, 5)
    app.map_manager.set_observer_position((0.0, 1.0, 5.0))
    app.map_manager.on_fixed_update(1.0 / 60.0)
    assert gate_script.is_open is True  # on_enter dispatched OPEN!

    # 3. Observer leaves trigger back to (0, 1, 0)
    app.map_manager.set_observer_position((0.0, 1.0, 0.0))
    app.map_manager.on_fixed_update(1.0 / 60.0)
    assert gate_script.is_open is False  # on_exit dispatched CLOSE!
