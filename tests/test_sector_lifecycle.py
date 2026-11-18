"""Unit Tests for Sector Lifecycle State Machine and Staged Activation."""

from __future__ import annotations

import pytest

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.mapping.schema import (
    CollisionDef,
    EntityDef,
    SectorBoundsDef,
    SectorDef,
    TransformDef,
    VisualDef,
)
from engine.world.mapping.sector import SectorInstance, SectorState


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


def test_sector_lifecycle_state_transitions(headless_app: ProjectApp) -> None:
    app = headless_app
    initial_ecs_count = app.ecs.pool.active_count

    sector_def = SectorDef(
        name="sector_alpha",
        bounds=SectorBoundsDef(center=(0.0, 0.0, 0.0), half_extents=(20.0, 10.0, 20.0)),
        entities=[
            EntityDef(
                id="pillar_1",
                transform=TransformDef(position=(2.0, 1.0, 2.0)),
                visual=VisualDef(shape="cylinder"),
                collision=CollisionDef(shape="cylinder", dimensions=(1.0, 0.3)),
            ),
            EntityDef(
                id="pillar_2",
                transform=TransformDef(position=(-2.0, 1.0, -2.0)),
                visual=VisualDef(shape="cylinder"),
                collision=CollisionDef(shape="cylinder", dimensions=(1.0, 0.3)),
            ),
        ],
    )

    sector = SectorInstance(sector_def=sector_def)
    assert sector.state == SectorState.LOADED
    assert sector.is_loaded is True
    assert sector.is_active is False

    # 1. Activate
    activated_events: list[str] = []
    deactivated_events: list[str] = []
    sector.on_activate(lambda s: activated_events.append(s.name))
    sector.on_deactivate(lambda s: deactivated_events.append(s.name))

    success = sector.activate(app)
    assert success is True
    assert sector.state == SectorState.ACTIVE
    assert sector.is_active is True
    assert sector.active_entity_count == 2
    assert app.ecs.pool.active_count == initial_ecs_count + 2
    assert len(activated_events) == 1

    # Check entity lookup
    node = sector.find_entity("pillar_1")
    assert node is not None
    assert node.ecs_id is not None
    assert app.ecs.is_alive(node.ecs_id)

    # 2. Deactivate (preserves SectorDef in RAM, destroys ECS & Rapier instances)
    sector.deactivate(app)
    assert sector.state == SectorState.LOADED
    assert sector.is_active is False
    assert sector.active_entity_count == 0
    assert app.ecs.pool.active_count == initial_ecs_count  # Zero leak!
    assert len(deactivated_events) == 1

    # 3. Reactivate instantly from RAM
    success_re = sector.activate(app)
    assert success_re is True
    assert sector.state == SectorState.ACTIVE
    assert sector.active_entity_count == 2
    assert app.ecs.pool.active_count == initial_ecs_count + 2

    # 4. Unload
    sector.unload(app)
    assert sector.state == SectorState.UNLOADED
    assert sector.is_unloaded is True
    assert app.ecs.pool.active_count == initial_ecs_count


def test_sector_staged_activation_budget(headless_app: ProjectApp) -> None:
    app = headless_app
    initial_ecs_count = app.ecs.pool.active_count

    # Create sector with 8 entities
    entities = [
        EntityDef(
            id=f"crate_{i}",
            transform=TransformDef(position=(float(i * 2), 0.5, 0.0)),
            visual=VisualDef(shape="cube"),
            collision=CollisionDef(shape="box", dimensions=(0.5, 0.5, 0.5)),
        )
        for i in range(8)
    ]
    sector = SectorInstance(sector_def=SectorDef(name="staged_sector", entities=entities))

    # Activate with budget = 3 entities per frame
    # Frame 1: spawns 3
    done1 = sector.activate(app, budget=3)
    assert done1 is False
    assert sector.state == SectorState.ACTIVATING
    assert sector.active_entity_count == 3
    assert app.ecs.pool.active_count == initial_ecs_count + 3

    # Frame 2: spawns 3 (total 6)
    done2 = sector.activate(app, budget=3)
    assert done2 is False
    assert sector.state == SectorState.ACTIVATING
    assert sector.active_entity_count == 6
    assert app.ecs.pool.active_count == initial_ecs_count + 6

    # Frame 3: spawns remaining 2 (total 8)
    done3 = sector.activate(app, budget=3)
    assert done3 is True
    assert sector.state == SectorState.ACTIVE
    assert sector.active_entity_count == 8
    assert app.ecs.pool.active_count == initial_ecs_count + 8

    # Deactivate and verify clean leak-free state
    sector.deactivate(app)
    assert sector.state == SectorState.LOADED
    assert app.ecs.pool.active_count == initial_ecs_count
