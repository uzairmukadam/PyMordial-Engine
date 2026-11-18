"""Unit tests for Hierarchical Entity & Prefab Instantiation."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.entity_node import MapEntityNode
from engine.world.mapping.prefab_manager import PrefabManager
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    SectorDef,
    TransformDef,
    VisualDef,
)
from engine.world.mapping.transform_math import (
    combine_transforms,
    quat_multiply,
    rotate_vec3_by_quat,
)
from engine.world.mapping.world_builder import MapWorldBuilder


def test_transform_math_combination() -> None:
    # Parent positioned at (10, 0, 0), rotated 90 degrees around Y: [0, sin(45°), 0, cos(45°)] = [0, 0.7071068, 0, 0.7071068]
    parent_tf = TransformDef(
        position=(10.0, 0.0, 0.0),
        rotation=(0.0, 0.7071068, 0.0, 0.7071068),
        scale=(2.0, 2.0, 2.0),
    )
    # Child positioned at local (1, 0, 0)
    child_tf = TransformDef(
        position=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
    )

    combined = combine_transforms(parent_tf, child_tf)
    # Child scaled by parent: local pos becomes (2, 0, 0)
    # Rotated 90 deg around Y: (2, 0, 0) rotated by 90 deg Y becomes approx (0, 0, -2) in right-handed coords
    # Added to parent (10, 0, 0): (10, 0, -2)
    assert abs(combined.position[0] - 10.0) < 1e-4
    assert abs(combined.position[1] - 0.0) < 1e-4
    assert abs(combined.position[2] - (-2.0)) < 1e-4

    # Scale: 2 * 1 = 2
    assert combined.scale == (2.0, 2.0, 2.0)


def test_prefab_manager_registration_and_overrides() -> None:
    mgr = PrefabManager()
    lamp_post = EntityDef(
        id="lamp_post_template",
        visual=VisualDef(shape="cylinder", color=(0.2, 0.2, 0.2)),
        collision=CollisionDef(shape="cylinder", dimensions=(2.0, 0.2)),
        properties={"light_color": "white", "is_lit": True},
        children=[
            EntityDef(
                id="bulb",
                transform=TransformDef(position=(0.0, 2.0, 0.0)),
                visual=VisualDef(shape="sphere", color=(1.0, 1.0, 0.8)),
            )
        ],
    )
    mgr.register_template("lamp_post", lamp_post)

    instance = mgr.instantiate(
        "lamp_post",
        instance_id="street_lamp_42",
        property_overrides={"is_lit": False, "bulb.intensity": 2.5},
    )
    assert instance.id == "street_lamp_42"
    assert instance.properties["is_lit"] is False
    assert instance.children[0].properties["intensity"] == 2.5


@pytest.fixture
def headless_app() -> ProjectApp:
    cfg = ProjectConfig(
        headless=True,
        enable_debug=False,
        max_entities=1000,
        atmosphere_enabled=False,
        particles_enabled=False,
    )
    app = ProjectApp(cfg)
    return app


def test_compound_entity_spawn_and_teardown(headless_app: ProjectApp) -> None:
    app = headless_app

    # Build compound table: 1 tabletop + 2 legs
    tabletop = EntityDef(
        id="tabletop",
        transform=TransformDef(position=(0.0, 1.0, 0.0), scale=(3.0, 0.1, 1.5)),
        visual=VisualDef(shape="cube", color=(0.6, 0.4, 0.2)),
        collision=CollisionDef(body_type="fixed", shape="box", dimensions=(1.5, 0.05, 0.75)),
    )
    leg1 = EntityDef(
        id="leg_left",
        transform=TransformDef(position=(-1.2, -0.5, 0.0), scale=(0.1, 1.0, 0.1)),
        visual=VisualDef(shape="cylinder", color=(0.1, 0.1, 0.1)),
        collision=CollisionDef(body_type="fixed", shape="cylinder", dimensions=(0.5, 0.05)),
    )
    leg2 = EntityDef(
        id="leg_right",
        transform=TransformDef(position=(1.2, -0.5, 0.0), scale=(0.1, 1.0, 0.1)),
        visual=VisualDef(shape="cylinder", color=(0.1, 0.1, 0.1)),
        collision=CollisionDef(body_type="fixed", shape="cylinder", dimensions=(0.5, 0.05)),
    )

    table_compound = EntityDef(
        id="table_group",
        transform=TransformDef(position=(5.0, 0.0, 10.0)),
        children=[tabletop, leg1, leg2],
    )

    root_node = MapEntityNode(table_compound)
    root_node.spawn(app)

    # 3 child entities spawned in ECS
    all_ecs_ids = root_node.collect_all_ecs_ids()
    assert len(all_ecs_ids) == 3
    for eid in all_ecs_ids:
        assert app.ecs.is_alive(eid)

    # Check world position of tabletop: parent (5, 0, 10) + local (0, 1, 0) = (5, 1, 10)
    top_node = root_node.find_by_id("tabletop")
    assert top_node is not None
    assert top_node.world_transform.position == (5.0, 1.0, 10.0)

    # Teardown
    root_node.teardown(app)
    assert len(root_node.children) == 0
    for eid in all_ecs_ids:
        assert not app.ecs.is_alive(eid)


def test_map_world_builder_lifecycle(headless_app: ProjectApp) -> None:
    app = headless_app

    sector = SectorDef(
        name="test_arena",
        entities=[
            EntityDef(
                id="ground",
                transform=TransformDef(position=(0.0, -0.5, 0.0), scale=(50.0, 1.0, 50.0)),
                visual=VisualDef(shape="cube"),
                collision=CollisionDef(body_type="fixed", shape="box", dimensions=(25.0, 0.5, 25.0)),
            ),
            EntityDef(
                id="crate",
                transform=TransformDef(position=(0.0, 1.0, 0.0), scale=(2.0, 2.0, 2.0)),
                visual=VisualDef(shape="cube"),
                collision=CollisionDef(body_type="dynamic", shape="box", dimensions=(1.0, 1.0, 1.0)),
            ),
        ],
    )

    builder = MapWorldBuilder(sector_def=sector)
    app.set_world_builder(builder)

    assert len(builder.entity_ids) == 2
    for eid in builder.entity_ids:
        assert app.ecs.is_alive(eid)

    # Teardown through builder
    builder.teardown_world(app)
    assert len(builder.entity_ids) == 0
