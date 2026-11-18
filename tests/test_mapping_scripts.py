"""Unit Tests for Interactive Kinematics and Entity Scripting."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    ScriptDef,
    SectorDef,
    TransformDef,
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


def test_moving_platform_kinematic_oscillation(headless_app: ProjectApp) -> None:
    app = headless_app

    platform_def = EntityDef(
        id="elevator_platform",
        transform=TransformDef(position=(0.0, 0.0, 0.0), scale=(4.0, 0.4, 4.0)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(
            body_type="kinematic_position",
            shape="box",
            dimensions=(2.0, 0.2, 2.0),
        ),
        scripts=[
            ScriptDef("MovingPlatformScript", {"axis": [0.0, 1.0, 0.0], "distance": 10.0, "speed": 2.0})
        ],
    )

    sector = SectorInstance(sector_def=SectorDef(name="platform_sector", entities=[platform_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    node = sector.find_entity("elevator_platform")
    assert node is not None
    assert len(node.scripts) == 1
    script = node.scripts[0]
    assert script.is_active is True

    initial_y = node.world_transform.position[1]
    assert initial_y == 0.0

    # Simulate 30 fixed ticks (0.5 seconds at dt = 1/60)
    dt = 1.0 / 60.0
    for _ in range(30):
        app.map_manager.on_fixed_update(dt)

    # Position should have moved upward along Y
    dense_idx = app.ecs.pool.get_dense_index(node.ecs_id)
    current_ecs_y = float(app.ecs.rigid_body_state[1, dense_idx, 1])
    assert current_ecs_y > 0.1  # Elevator has risen!


def test_interactive_lever_and_sliding_gate(headless_app: ProjectApp) -> None:
    app = headless_app

    # Gate
    gate_def = EntityDef(
        id="security_gate",
        transform=TransformDef(position=(10.0, 0.0, 0.0), scale=(4.0, 6.0, 0.4)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(
            body_type="kinematic_position",
            shape="box",
            dimensions=(2.0, 3.0, 0.2),
        ),
        scripts=[ScriptDef("SlidingGateScript", {"open_offset": [0.0, 5.0, 0.0], "speed": 10.0})],
    )

    # Lever
    lever_def = EntityDef(
        id="gate_lever",
        transform=TransformDef(position=(5.0, 1.0, 0.0), scale=(0.2, 0.6, 0.2)),
        visual=VisualDef(shape="cylinder"),
        collision=CollisionDef(
            body_type="kinematic_position",
            shape="cylinder",
            dimensions=(0.3, 0.1),
        ),
        scripts=[
            ScriptDef(
                "InteractiveLeverScript",
                {"target_id": "security_gate", "pull_angle_deg": 60.0, "speed": 20.0},
            )
        ],
    )

    sector = SectorInstance(sector_def=SectorDef(name="interactive_sector", entities=[gate_def, lever_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    gate_node = sector.find_entity("security_gate")
    lever_node = sector.find_entity("gate_lever")
    assert gate_node is not None and lever_node is not None

    gate_script = gate_node.scripts[0]
    assert gate_script.is_open is False

    # Player interacts with lever from proximity
    triggered_count = app.map_manager.trigger_interaction(origin=(5.0, 1.0, 0.0), radius=2.0)
    assert triggered_count == 1
    assert lever_node.scripts[0].is_pulled is True
    assert gate_script.is_open is True

    # Simulate fixed steps for gate and lever animations
    dt = 1.0 / 60.0
    for _ in range(60):
        app.map_manager.on_fixed_update(dt)

    # Gate should now be fully open at y ~ 5.0
    gate_dense = app.ecs.pool.get_dense_index(gate_node.ecs_id)
    gate_y = float(app.ecs.rigid_body_state[1, gate_dense, 1])
    assert gate_y > 4.5  # Gate has fully opened!
