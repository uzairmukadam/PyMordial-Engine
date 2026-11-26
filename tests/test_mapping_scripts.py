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


def test_game_directory_script_discovery(headless_app: ProjectApp) -> None:
    """Verifies that scripts residing in projects/<project>/scripts/ are discovered and spawned."""
    app = headless_app
    beacon_def = EntityDef(
        id="test_beacon",
        transform=TransformDef(position=(0.0, 5.0, 0.0), scale=(1.0, 1.0, 1.0)),
        visual=VisualDef(shape="sphere"),
        scripts=[ScriptDef("ArenaBeaconScript", {"pulse_speed": 5.0, "rotation_speed": 2.5})],
    )

    sector = SectorInstance(sector_def=SectorDef(name="beacon_sector", entities=[beacon_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    node = sector.find_entity("test_beacon")
    assert node is not None
    assert len(node.scripts) == 1
    script = node.scripts[0]
    assert script.__class__.__name__ == "ArenaBeaconScript"
    assert script.is_active is True

    # Advance a tick to verify execution
    app.map_manager.on_fixed_update(1.0 / 60.0)


def test_character_motor_moving_platform_carrier_up_down(headless_app: ProjectApp) -> None:
    """Verifies AAA platform relative carriage during vertical descent (zero floating/hover)."""
    from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig

    app = headless_app
    dt = 1.0 / 60.0

    # Moving platform oscillating vertically (descending from y=10.0)
    platform_def = EntityDef(
        id="elevator",
        transform=TransformDef(position=(0.0, 10.0, 0.0), scale=(4.0, 0.4, 4.0)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(
            body_type="kinematic_position",
            shape="box",
            dimensions=(2.0, 0.2, 2.0),
        ),
        scripts=[
            ScriptDef("MovingPlatformScript", {"axis": [0.0, -1.0, 0.0], "distance": 6.0, "speed": 1.5})
        ],
    )
    sector = SectorInstance(sector_def=SectorDef(name="elevator_sector", entities=[platform_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    plat_node = sector.find_entity("elevator")
    assert plat_node is not None

    # Place player on top of elevator: top of box is at y = 10.0 + 0.2 = 10.2.
    # Capsule half_height 0.55, radius 0.35 -> feet offset 0.90 -> player center at 10.2 + 0.901 = 11.101
    player_id = app.ecs.create_entity(shape="capsule", position=(0.0, 11.101, 0.0))
    motor = CharacterMotor(
        player_id,
        app.physics,
        app.ecs,
        config=CharacterMotorConfig(capsule_half_height=0.55, capsule_radius=0.35),
        initial_position=(0.0, 11.101, 0.0),
    )

    # Initial tick to ground player
    app.map_manager.on_fixed_update(dt)
    motor.update(dt, (0.0, 0.0))
    app.physics.step_simulation(dt)
    app.physics.sync_to_ecs(app.ecs)
    assert motor.state.is_grounded is True

    # Simulate 30 frames of downward movement
    for _ in range(30):
        app.map_manager.on_fixed_update(dt)
        motor.update(dt, (0.0, 0.0))
        app.physics.step_simulation(dt)
        app.physics.sync_to_ecs(app.ecs)
        # Player must remain grounded throughout downward motion (no floating in midair)
        assert motor.state.is_grounded is True


def test_character_motor_moving_platform_carrier_horizontal(headless_app: ProjectApp) -> None:
    """Verifies relative base carriage: horizontal platform moves player without flinging or velocity fighting."""
    from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig

    app = headless_app
    dt = 1.0 / 60.0

    platform_def = EntityDef(
        id="shuttle",
        transform=TransformDef(position=(0.0, 2.0, 0.0), scale=(4.0, 0.4, 4.0)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(
            body_type="kinematic_position",
            shape="box",
            dimensions=(2.0, 0.2, 2.0),
        ),
        scripts=[
            ScriptDef("MovingPlatformScript", {"axis": [1.0, 0.0, 0.0], "distance": 10.0, "speed": 1.5})
        ],
    )
    sector = SectorInstance(sector_def=SectorDef(name="shuttle_sector", entities=[platform_def]))
    app.map_manager.register_sector(sector)
    sector.activate(app)

    plat_node = sector.find_entity("shuttle")
    assert plat_node is not None

    player_id = app.ecs.create_entity(shape="capsule", position=(0.0, 3.101, 0.0))
    motor = CharacterMotor(
        player_id,
        app.physics,
        app.ecs,
        config=CharacterMotorConfig(capsule_half_height=0.55, capsule_radius=0.35),
        initial_position=(0.0, 3.101, 0.0),
    )

    # Initial settle
    app.map_manager.on_fixed_update(dt)
    motor.update(dt, (0.0, 0.0))
    app.physics.step_simulation(dt)
    app.physics.sync_to_ecs(app.ecs)
    assert motor.state.is_grounded is True

    # Simulate 30 frames of horizontal movement while player has zero input
    for _ in range(30):
        app.map_manager.on_fixed_update(dt)
        motor.update(dt, (0.0, 0.0))
        app.physics.step_simulation(dt)
        app.physics.sync_to_ecs(app.ecs)
        assert motor.state.is_grounded is True

    # Player position must have moved along with platform (x > 0.2)
    assert motor.position[0] > 0.2
    # Player's relative walking velocity must NOT have accumulated or fought the platform
    assert abs(motor.state.velocity[0]) < 0.1
    assert abs(motor.state.velocity[2]) < 0.1
