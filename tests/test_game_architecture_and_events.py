"""Unit tests for Game Architecture, Event Bus, World Context, and Story Manager."""

import math
import numpy as np
import pytest

from engine.app.project_app import ProjectApp
from projects.shotgun_escape_the_heat.core.events import GameEventBus, GameEvents
from projects.shotgun_escape_the_heat.core.math_utils import (
    heading_deg_to_quat,
    quat_multiply,
    quat_rotate_vector,
    quat_to_heading_deg,
)
from projects.shotgun_escape_the_heat.world.world_interface import (
    PoiStationDefinition,
    WorldDefinition,
)
from projects.shotgun_escape_the_heat.world.test_circuit_world import TestCircuitWorldBuilder
from projects.shotgun_escape_the_heat.world.city_world import CityWorldBuilder
from projects.shotgun_escape_the_heat.modules.station_manager import StationManagerModule
from projects.shotgun_escape_the_heat.modules.story_manager import StoryManagerModule
from projects.shotgun_escape_the_heat.modules.story_manager.mission_types import MissionObjectiveType


def test_event_bus_publish_subscribe():
    """Verifies that GameEventBus dispatches typed events to subscribers."""
    bus = GameEventBus.get()
    bus.clear()

    received_events: list[dict] = []

    def on_damage(car_id: str = "", damage: float = 0.0, delta: float = 0.0, **kwargs):
        received_events.append({"event": "damage", "car_id": car_id, "damage": damage, "delta": delta})

    def on_toast(message: str = "", category: str = "", **kwargs):
        received_events.append({"event": "toast", "message": message, "category": category})

    bus.subscribe(GameEvents.CAR_DAMAGED, on_damage)
    bus.subscribe(GameEvents.TOAST, on_toast)

    # Publish events
    bus.publish(GameEvents.CAR_DAMAGED, car_id="starter_sedan", damage=25.0, delta=5.0)
    bus.publish(GameEvents.TOAST, message="Engine Upgraded", category="TUNING")

    assert len(received_events) == 2
    assert received_events[0] == {"event": "damage", "car_id": "starter_sedan", "damage": 25.0, "delta": 5.0}
    assert received_events[1] == {"event": "toast", "message": "Engine Upgraded", "category": "TUNING"}

    # Unsubscribe
    bus.unsubscribe(GameEvents.CAR_DAMAGED, on_damage)
    bus.publish(GameEvents.CAR_DAMAGED, car_id="starter_sedan", damage=30.0, delta=5.0)
    assert len(received_events) == 2  # No new event received

    bus.clear()


def test_shared_math_utils():
    """Verifies centralized vector and quaternion math operations."""
    # 1. Heading to Quat & back
    q = heading_deg_to_quat(90.0)
    assert abs(q[0]) < 1e-6
    assert abs(q[2]) < 1e-6
    deg = quat_to_heading_deg(q)
    assert pytest.approx(deg, 0.1) == 90.0

    # 2. Vector rotation by 90 degree yaw
    # Vector (0, 0, -1) rotated by 90 deg yaw counter-clockwise about +Y points along (-1, 0, 0)
    v = (0.0, 0.0, -1.0)
    out = np.zeros(3, dtype=np.float32)
    quat_rotate_vector(q, v, out)
    assert pytest.approx(out[0], 0.01) == -1.0
    assert pytest.approx(out[1], 0.01) == 0.0
    assert pytest.approx(out[2], 0.01) == 0.0

    # 3. Quaternion multiplication
    q_ident = (0.0, 0.0, 0.0, 1.0)
    q_res = quat_multiply(q, q_ident)
    assert pytest.approx(q_res[1], 0.01) == q[1]
    assert pytest.approx(q_res[3], 0.01) == q[3]


def test_world_definition_protocol_circuit():
    """Verifies that TestCircuitWorldBuilder fulfills WorldDefinition protocol."""
    circuit = TestCircuitWorldBuilder()
    assert isinstance(circuit, WorldDefinition)
    assert circuit.name == "TestCircuit"

    # POI stations
    stations = circuit.get_stations()
    assert len(stations) >= 4
    station_types = {s.station_type for s in stations}
    assert "safehouse" in station_types or "garage" in station_types
    assert "respray" in station_types
    assert "chop_shop" in station_types
    assert "mission" in station_types

    # Pedestrian routes
    routes = circuit.get_pedestrian_routes()
    assert len(routes) >= 5
    for r in routes:
        assert len(r) >= 3  # Closed waypoint loop

    # Road waypoints
    road_pts = circuit.get_road_waypoints()
    assert len(road_pts) >= 10


def test_world_definition_protocol_city():
    """Verifies that CityWorldBuilder fulfills WorldDefinition protocol."""
    city = CityWorldBuilder()
    assert isinstance(city, WorldDefinition)
    assert city.name == "City"
    assert city.map_size > 0.0

    # Stations
    stations = city.get_stations()
    assert len(stations) >= 3

    # Pedestrian routes
    routes = city.get_pedestrian_routes()
    assert len(routes) >= 1

    # Road waypoints
    road_pts = city.get_road_waypoints()
    assert len(road_pts) >= 10


def test_station_handlers_modular_dispatch():
    """Verifies StationManagerModule initializes and maps dedicated handlers."""
    sm = StationManagerModule()
    assert "garage" in sm.handlers
    assert "respray" in sm.handlers
    assert "chop_shop" in sm.handlers
    assert "mission" in sm.handlers

    # Verify event bus toast listener
    bus = GameEventBus.get()
    initial_count = len(sm.toasts)
    sm.on_attach(None)
    bus.publish(GameEvents.TOAST, message="Architectural Refactor Success", category="ENGINE")
    assert len(sm.toasts) == initial_count + 1
    assert sm.toasts[-1].message == "Architectural Refactor Success"
    sm.on_detach(None)


def test_story_manager_lifecycle_and_missions():
    """Verifies StoryManagerModule contract activation, progress, and completion."""
    story = StoryManagerModule()
    assert story.current_chapter == 1
    assert len(story.get_available_missions()) >= 2

    # Start Chapter 1 Getaway Audition
    ok = story.start_mission("contract_the_audition")
    assert ok is True
    assert story.active_mission is not None
    assert story.active_mission.mission_id == "contract_the_audition"
    assert story.active_mission.is_active is True

    # Advance time: 45.0s is required to complete the audition
    story.on_fixed_update(None, 20.0)
    assert story.active_mission.is_active is True
    assert story.active_mission.elapsed_time == 20.0

    # Reaching 45.0s completes the survival getaway!
    story.on_fixed_update(None, 26.0)
    assert story.active_mission is None
    assert "contract_the_audition" in story.completed_missions
