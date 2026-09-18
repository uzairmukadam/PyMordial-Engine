"""Unit Tests for City Map Data & City World Builder."""

from __future__ import annotations
import pytest
from projects.shotgun_escape_the_heat.world.city_map_data import (
    create_default_city_map,
    CityMapData,
    RoadDef,
    BuildingDef,
    POIDef,
)
from projects.shotgun_escape_the_heat.world.city_world import CityWorldBuilder


def test_city_map_data_generation() -> None:
    """Verifies that the default city map defines roads, alleys, buildings, and POIs."""
    city = create_default_city_map()

    assert city.map_size >= 200.0
    assert len(city.roads) >= 6
    assert len(city.buildings) >= 15
    assert len(city.decorations) >= 20
    assert len(city.pois) >= 5

    # Verify road types
    alleys = [r for r in city.roads if r.is_alley]
    assert len(alleys) >= 3
    avenues = [r for r in city.roads if r.lanes >= 4]
    assert len(avenues) >= 2

    # Verify required mission POIs
    poi_ids = {p.id for p in city.pois}
    assert "poi_safehouse" in poi_ids
    assert "poi_bank" in poi_ids
    assert "poi_warehouse" in poi_ids
    assert "poi_respray" in poi_ids
    assert "poi_jewelry" in poi_ids


def test_city_world_builder_navigation_graph() -> None:
    """Tests that CityWorldBuilder constructs a connected WaypointGraph with POIs."""
    builder = CityWorldBuilder()
    builder._build_waypoint_graph()

    graph = builder.waypoint_graph
    assert len(graph.nodes) > 50
    assert len(graph.edges) > 50
    assert len(graph.pois) >= 5

    # Test POI lookup
    safehouse = graph.get_poi("Safehouse Garage")
    assert safehouse is not None
    assert safehouse.name == "Safehouse Garage"
    assert safehouse.poi_type == "safehouse"

    # Nearest POI query
    sh = builder.map_data.get_safehouse()
    nearest = graph.find_nearest_poi((sh.position[0] - 2.0, 0.0, sh.position[2] - 5.0))
    assert nearest is not None
    assert nearest.name == "Safehouse Garage"

    # POI by type query
    banks = graph.get_pois_by_type("heist_target")
    assert len(banks) == 1
    assert banks[0].name == "Metropolitan Bank"

    # Spawn position query
    spawn_pos, heading = builder.get_poi_spawn("Safehouse Garage")
    assert spawn_pos[0] == pytest.approx(sh.position[0])
    assert heading == pytest.approx(sh.heading_deg)




def test_city_world_builder_pathfinding() -> None:
    """Tests that A* pathfinding routes across the city road network."""
    builder = CityWorldBuilder()
    builder._build_waypoint_graph()

    graph = builder.waypoint_graph

    # Find path between safehouse vicinity and bank vicinity
    sh = builder.map_data.get_safehouse()
    bank = graph.get_pois_by_type("heist_target")[0]
    start_node = graph.find_nearest_node(sh.position)
    goal_node = graph.find_nearest_node(bank.position)
    assert start_node is not None
    assert goal_node is not None

    path = graph.find_path(start_node, goal_node)
    assert len(path) >= 2
