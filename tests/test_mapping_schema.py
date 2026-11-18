"""Unit Tests for Mapping System Schemas and Data Contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_transform_def_defaults_and_validation() -> None:
    tf = TransformDef()
    assert tf.position == (0.0, 0.0, 0.0)
    assert tf.rotation == (0.0, 0.0, 0.0, 1.0)
    assert tf.scale == (1.0, 1.0, 1.0)

    # Custom valid values
    tf2 = TransformDef(position=(1.5, 2.0, -3.5), rotation=(0.0, 0.7071, 0.0, 0.7071), scale=(2.0, 2.0, 2.0))
    d = tf2.to_dict()
    assert d["position"] == [1.5, 2.0, -3.5]
    assert d["rotation"] == [0.0, 0.7071, 0.0, 0.7071]
    assert d["scale"] == [2.0, 2.0, 2.0]

    # Deserialization
    tf3 = TransformDef.from_dict(d)
    assert tf3.position == (1.5, 2.0, -3.5)
    assert tf3.rotation == (0.0, 0.7071, 0.0, 0.7071)
    assert tf3.scale == (2.0, 2.0, 2.0)

    # Invalid lengths
    with pytest.raises(ValueError):
        TransformDef(position=(1.0, 2.0))  # requires 3 floats


def test_visual_def_primitives_and_options() -> None:
    vis = VisualDef(shape="Cube", color=(0.8, 0.2, 0.1), roughness=0.3, metallic=0.9)
    assert vis.shape == "cube"
    assert vis.color == (0.8, 0.2, 0.1)
    assert vis.roughness == 0.3
    assert vis.metallic == 0.9
    assert vis.cast_shadow is True
    assert vis.is_visible is True

    # Phase 2 mesh upgrade field
    vis_mesh = VisualDef(shape="custom", mesh_path="assets/models/pillar.pm_mesh")
    assert vis_mesh.mesh_path == "assets/models/pillar.pm_mesh"

    # Invalid primitive shape without mesh_path
    with pytest.raises(ValueError):
        VisualDef(shape="invalid_shape_name")


def test_collision_def_body_types_and_shapes() -> None:
    # Static box
    col_box = CollisionDef(body_type="fixed", shape="box", dimensions=(2.0, 0.5, 2.0))
    assert col_box.body_type == "fixed"
    assert col_box.shape == "box"
    assert col_box.dimensions == (2.0, 0.5, 2.0)
    assert col_box.is_sensor is False

    # Kinematic position for moving platform or rotating lever
    col_kin = CollisionDef(
        body_type="kinematic_position",
        shape="cylinder",
        dimensions=(1.0, 0.2),
        friction=0.95,
        restitution=0.05,
    )
    assert col_kin.body_type == "kinematic_position"
    assert col_kin.shape == "cylinder"

    # Kinematic velocity
    col_vel = CollisionDef(body_type="kinematic_velocity", shape="sphere", dimensions=(0.5,))
    assert col_vel.body_type == "kinematic_velocity"

    # Sensor
    col_sensor = CollisionDef(body_type="fixed", shape="box", dimensions=(1.0, 1.0, 1.0), is_sensor=True)
    assert col_sensor.is_sensor is True

    # Invalid body type
    with pytest.raises(ValueError):
        CollisionDef(body_type="invalid_type")

    # Invalid negative dimensions
    with pytest.raises(ValueError):
        CollisionDef(dimensions=(-1.0, 0.5, 0.5))


def test_trigger_def_and_actions() -> None:
    # String shorthand action
    act1 = TriggerActionDef.from_dict("PLAY_ALARM")
    assert act1.action_type == "PLAY_ALARM"

    # Detailed action
    act2 = TriggerActionDef.from_dict({
        "action_type": "STREAM_SECTOR",
        "target": "sectors/corridor_b.json",
        "params": {"fade_duration": 0.5},
    })
    assert act2.action_type == "STREAM_SECTOR"
    assert act2.target == "sectors/corridor_b.json"
    assert act2.params["fade_duration"] == 0.5

    trig = TriggerDef(
        shape="box",
        dimensions=(4.0, 2.0, 4.0),
        filter_tag="player",
        one_shot=True,
        on_enter=[act2],
        on_exit=[act1],
    )
    d = trig.to_dict()
    assert d["shape"] == "box"
    assert d["dimensions"] == [4.0, 2.0, 4.0]
    assert d["one_shot"] is True
    assert len(d["on_enter"]) == 1
    assert len(d["on_exit"]) == 1

    # Roundtrip
    trig2 = TriggerDef.from_dict(d)
    assert trig2.one_shot is True
    assert trig2.on_enter[0].action_type == "STREAM_SECTOR"
    assert trig2.on_exit[0].action_type == "PLAY_ALARM"


def test_script_def_parsing() -> None:
    # String shorthand
    s1 = ScriptDef.from_dict("OscillatingPlatformScript")
    assert s1.script_type == "OscillatingPlatformScript"
    assert s1.params == {}

    # Dict format
    s2 = ScriptDef.from_dict({
        "script_type": "InteractiveLeverScript",
        "params": {"target_gate_id": "gate_01", "angle_deg": 60.0},
    })
    assert s2.script_type == "InteractiveLeverScript"
    assert s2.params["target_gate_id"] == "gate_01"


def test_entity_def_atomic_and_resolved_collision() -> None:
    ent = EntityDef(
        id="floor_tile",
        transform=TransformDef(position=(0.0, -0.5, 0.0), scale=(10.0, 1.0, 10.0)),
        visual=VisualDef(shape="cube", color=(0.5, 0.5, 0.5)),
    )
    assert not ent.is_compound
    # Collision was omitted; resolved_collision should auto-derive a fixed box
    derived = ent.resolved_collision()
    assert derived is not None
    assert derived.shape == "box"
    assert derived.body_type == "fixed"
    # Half-extents are half scale: 10*0.5=5.0, 1*0.5=0.5, 10*0.5=5.0
    assert derived.dimensions == (5.0, 0.5, 5.0)

    # Properties
    ent.properties["destructible"] = True
    assert ent.get_property("destructible") is True
    assert ent.get_property("missing_prop", "default_val") == "default_val"


def test_compound_entity_hierarchy_and_prefabs() -> None:
    lever_arm = EntityDef(
        id="lever_arm",
        transform=TransformDef(position=(0.0, 0.5, 0.0)),
        visual=VisualDef(shape="cylinder"),
        collision=CollisionDef(body_type="kinematic_position", shape="cylinder", dimensions=(0.5, 0.05)),
        scripts=[ScriptDef("InteractiveLeverScript", {"target": "main_gate"})],
    )
    lever_base = EntityDef(
        id="lever_base",
        transform=TransformDef(position=(0.0, 0.0, 0.0)),
        visual=VisualDef(shape="cube"),
        collision=CollisionDef(body_type="fixed", shape="box", dimensions=(0.2, 0.2, 0.1)),
    )
    lever_compound = EntityDef(
        id="wall_lever",
        transform=TransformDef(position=(5.0, 1.2, -3.0)),
        children=[lever_base, lever_arm],
    )

    assert lever_compound.is_compound is True
    assert len(lever_compound.children) == 2

    d = lever_compound.to_dict()
    assert "children" in d
    assert len(d["children"]) == 2

    # Roundtrip
    deserialized = EntityDef.from_dict(d)
    assert deserialized.is_compound is True
    assert len(deserialized.children) == 2
    assert deserialized.children[1].scripts[0].script_type == "InteractiveLeverScript"


def test_sector_bounds_contains_point() -> None:
    bounds = SectorBoundsDef(center=(0.0, 0.0, 0.0), half_extents=(10.0, 5.0, 10.0))
    # Inside
    assert bounds.contains_point((0.0, 0.0, 0.0)) is True
    assert bounds.contains_point((5.0, 2.0, -5.0)) is True
    assert bounds.contains_point((10.0, 5.0, 10.0)) is True  # on boundary
    # Outside
    assert bounds.contains_point((10.1, 0.0, 0.0)) is False
    assert bounds.contains_point((0.0, 5.1, 0.0)) is False
    assert bounds.contains_point((0.0, 0.0, -10.1)) is False


def test_sector_and_map_file_io_roundtrip(tmp_path: Path) -> None:
    # 1. Build Sector
    sector = SectorDef(
        name="test_dungeon_sector_a",
        bounds=SectorBoundsDef(center=(0.0, 5.0, 0.0), half_extents=(25.0, 10.0, 25.0)),
        entities=[
            EntityDef(
                id="platform_01",
                transform=TransformDef(position=(0.0, 2.0, 0.0)),
                visual=VisualDef(shape="cube", color=(0.2, 0.8, 0.2)),
                collision=CollisionDef(body_type="kinematic_position", shape="box", dimensions=(2.0, 0.2, 2.0)),
                scripts=[ScriptDef("OscillatingPlatformScript", {"distance": 5.0, "speed": 1.5})],
            ),
            EntityDef(
                id="exit_trigger",
                transform=TransformDef(position=(0.0, 1.0, 20.0)),
                trigger=TriggerDef(
                    shape="box",
                    dimensions=(2.0, 2.0, 1.0),
                    one_shot=True,
                    on_enter=[TriggerActionDef("STREAM_SECTOR", target="sector_b.json")],
                ),
            ),
        ],
        metadata={"author": "Antigravity", "lighting": "dungeon_dark"},
    )

    sector_file = tmp_path / "sector_a.json"
    save_sector_file(sector, sector_file)
    assert sector_file.is_file()

    loaded_sector = load_sector_file(sector_file)
    assert loaded_sector.name == "test_dungeon_sector_a"
    assert loaded_sector.bounds is not None
    assert loaded_sector.bounds.half_extents == (25.0, 10.0, 25.0)
    assert len(loaded_sector.entities) == 2
    assert loaded_sector.entities[0].id == "platform_01"
    assert loaded_sector.entities[0].scripts[0].script_type == "OscillatingPlatformScript"
    assert loaded_sector.entities[1].trigger is not None
    assert loaded_sector.entities[1].trigger.one_shot is True

    # 2. Build Map
    map_def = MapDef(
        name="dungeon_level_01",
        sectors=[str(sector_file), "sectors/sector_b.json"],
        initial_sector=str(sector_file),
        streaming_mode="linear_triggers",
        metadata={"difficulty": "normal"},
    )

    map_file = tmp_path / "dungeon_map.json"
    save_map_file(map_def, map_file)
    assert map_file.is_file()

    loaded_map = load_map_file(map_file)
    assert loaded_map.name == "dungeon_level_01"
    assert loaded_map.streaming_mode == "linear_triggers"
    assert len(loaded_map.sectors) == 2
    assert loaded_map.initial_sector == str(sector_file)
