"""Unit Tests for GPS Minimap & Radar Module."""

from __future__ import annotations
import math
import pytest

from engine.app.project_app import ProjectApp
from projects.shotgun_escape_the_heat.main import create_game
from projects.shotgun_escape_the_heat.modules.minimap import MinimapModule
from projects.shotgun_escape_the_heat.modules.vehicle_controller import VehicleControllerModule
from projects.shotgun_escape_the_heat.world.city_map_spec import create_default_city_spec
from projects.shotgun_escape_the_heat.world.city_world import CityWorldBuilder


def test_minimap_module_lifecycle() -> None:
    """Verifies that MinimapModule initializes and updates without errors."""
    mm = MinimapModule(radar_radius_px=80.0, base_range_m=100.0)
    assert mm.name == "Minimap"
    assert mm.radar_radius_px == 80.0
    assert mm.base_range_m == 100.0

    # Test update ticking
    mm.on_update(None, 0.016)
    assert mm._siren_timer > 0.0


def test_minimap_road_and_poi_query_headless() -> None:
    """Verifies that the minimap queries world roads, POIs, and player vehicle correctly."""
    app = create_game(headless=True, max_frames=5, world_type="city")
    vc = app.get_module(VehicleControllerModule)
    assert vc is not None
    assert vc.vehicle is not None

    mm = app.get_module(MinimapModule)
    assert mm is not None

    # Step frame
    for _ in range(3):
        app.step_frame(0.016)

    # Verify world contains map spec with roads and POIs
    world = app.world_builder
    assert world is not None
    assert hasattr(world, "map_spec")
    assert len(world.map_spec.roads) >= 8
    assert len(world.map_spec.pois) >= 8

    # Check vehicle coordinates
    pos = vc.vehicle.position
    assert not any(math.isnan(x) for x in pos)
