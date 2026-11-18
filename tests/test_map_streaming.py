"""Unit Tests for Map Loading and Linear Trigger Streaming."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    MapDef,
    SectorDef,
    TransformDef,
    VisualDef,
)


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


def test_map_manager_monolithic_load(headless_app: ProjectApp) -> None:
    app = headless_app

    sec1 = SectorDef(
        name="sec_1",
        entities=[
            EntityDef(id="cube_1", visual=VisualDef(shape="cube"), collision=CollisionDef(shape="box", dimensions=(1, 1, 1)))
        ],
    )
    sec2 = SectorDef(
        name="sec_2",
        entities=[
            EntityDef(id="cube_2", visual=VisualDef(shape="cube"), collision=CollisionDef(shape="box", dimensions=(1, 1, 1)))
        ],
    )

    map_def = MapDef(name="monolithic_level", sectors=[sec1, sec2], streaming_mode="monolithic")
    app.map_manager.load_map(map_def)

    assert app.map_manager.find_entity("cube_1") is not None
    assert app.map_manager.find_entity("cube_2") is not None
    assert len(app.map_manager.active_sectors) == 2
    for s in app.map_manager.active_sectors.values():
        assert s.is_active is True


def test_map_manager_linear_streaming(headless_app: ProjectApp) -> None:
    app = headless_app

    sec_a = SectorDef(
        name="room_a",
        entities=[
            EntityDef(id="chair_a", visual=VisualDef(shape="cube"), collision=CollisionDef(shape="box", dimensions=(1, 1, 1)))
        ],
    )
    sec_b = SectorDef(
        name="room_b",
        entities=[
            EntityDef(id="table_b", visual=VisualDef(shape="cube"), collision=CollisionDef(shape="box", dimensions=(2, 1, 2)))
        ],
    )

    map_def = MapDef(
        name="linear_stream_level",
        sectors=[sec_a, sec_b],
        initial_sector="room_a",
        streaming_mode="linear_triggers",
    )
    app.map_manager.load_map(map_def)

    # Initially room_a is active, room_b is not yet active
    sec_a_inst = app.map_manager.active_sectors["room_a"]
    sec_b_inst = app.map_manager.active_sectors["room_b"]

    # Process staged queue
    app.map_manager.on_fixed_update(1.0 / 60.0)
    assert sec_a_inst.is_active is True
    assert sec_b_inst.is_active is False

    # Trigger request to stream room_b
    app.map_manager.streaming_director.request_stream("room_b", mode="ACTIVATE")
    app.map_manager.on_fixed_update(1.0 / 60.0)
    assert sec_b_inst.is_active is True

    # Deactivate room_a
    app.map_manager.streaming_director.request_stream("room_a", mode="DEACTIVATE")
    assert sec_a_inst.is_active is False
    assert sec_b_inst.is_active is True
