"""Unit Tests for Physics Auto-Derivation and Dynamic Prop Registration in Mapping."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.schema import (
    EntityDef,
    PhysicsDef,
    PhysicsRole,
    TransformDef,
    TriggerDef,
    VisualDef,
)


@pytest.fixture
def app() -> ProjectApp:
    cfg = ProjectConfig(
        headless=True,
        enable_debug=False,
        max_entities=500,
        atmosphere_enabled=False,
        particles_enabled=False,
    )
    return ProjectApp(cfg)


def test_auto_derivation_static_scenery(app: ProjectApp) -> None:
    # Static stone cube: no manual dimensions or collision block
    wall_def = EntityDef(
        id="castle_wall",
        transform=TransformDef(position=(0.0, 2.0, 0.0), scale=(8.0, 4.0, 1.0)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.STATIC, material="stone"),
    )

    node = MapEntityNode(wall_def)
    node.spawn(app)

    assert node.ecs_id is not None
    assert node.has_rigid_body is True

    # Rapier body should be fixed
    transform = app.physics.world.get_transform(node.ecs_id)
    assert transform[0] == 0.0
    assert transform[1] == 2.0
    assert transform[2] == 0.0

    node.teardown(app)
    assert node.ecs_id is None
    assert node.has_rigid_body is False


def test_auto_derivation_dynamic_wooden_crate(app: ProjectApp) -> None:
    # Dynamic wooden crate: auto-derives box half-extents, wood friction, registers with app.props
    crate_def = EntityDef(
        id="wooden_crate_alpha",
        transform=TransformDef(position=(10.0, 1.0, 0.0), scale=(1.5, 1.5, 1.5)),
        visual=VisualDef(shape="cube", material="wood"),
        physics=PhysicsDef(role=PhysicsRole.DYNAMIC, material="wood"),
    )

    node = MapEntityNode(crate_def)
    node.spawn(app)

    assert node.ecs_id is not None
    assert node.has_rigid_body is True

    # Verify registered with InteractivePropManager
    assert hasattr(app, "props") and app.props is not None
    assert app.props.is_prop(node.ecs_id) is True
    prop = app.props.get_prop(node.ecs_id)
    assert prop is not None
    assert prop.scale == (1.5, 1.5, 1.5)

    # Teardown unregisters from props
    node.teardown(app)
    assert app.props.is_prop(node.ecs_id) is False


def test_auto_derivation_dynamic_rubber_sphere(app: ProjectApp) -> None:
    sphere_def = EntityDef(
        id="bouncy_ball",
        transform=TransformDef(position=(0.0, 10.0, 0.0), scale=(2.0, 2.0, 2.0)),
        visual=VisualDef(shape="sphere"),
        physics=PhysicsDef(role=PhysicsRole.DYNAMIC, material="rubber"),
    )

    node = MapEntityNode(sphere_def)
    node.spawn(app)

    assert node.ecs_id is not None
    assert node.has_rigid_body is True
    assert app.props.is_prop(node.ecs_id) is True

    node.teardown(app)


def test_auto_derivation_pure_sensor_no_ecs_mesh(app: ProjectApp) -> None:
    # Sensor volume without visual: must NOT create an ECS entity or mesh
    sensor_def = EntityDef(
        id="room_trigger",
        transform=TransformDef(position=(15.0, 0.0, 0.0), scale=(5.0, 5.0, 5.0)),
        trigger=TriggerDef(shape="box", dimensions=(2.5, 2.5, 2.5)),
        physics=PhysicsDef(role=PhysicsRole.SENSOR),
    )

    node = MapEntityNode(sensor_def)
    node.spawn(app)

    assert node.ecs_id is None
    assert node.has_rigid_body is False

    # Trigger zone is still registered with map manager
    assert len(app.map_manager.trigger_system._triggers) == 1
    zone = app.map_manager.trigger_system._triggers[0]
    assert zone.test_overlap((15.0, 0.0, 0.0)) is True
    assert zone.test_overlap((0.0, 0.0, 0.0)) is False

    node.teardown(app)
    assert len(app.map_manager.trigger_system._triggers) == 0


def test_auto_derivation_visual_only_no_physics(app: ProjectApp) -> None:
    # None role: visual only, no Rapier rigid body
    decal_def = EntityDef(
        id="decorative_sign",
        transform=TransformDef(position=(0.0, 1.0, 0.0), scale=(1.0, 1.0, 0.1)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.NONE),
    )

    node = MapEntityNode(decal_def)
    node.spawn(app)

    assert node.ecs_id is not None
    assert node.has_rigid_body is False

    node.teardown(app)
    assert node.ecs_id is None
