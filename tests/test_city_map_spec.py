"""Unit Tests for Unified Single-Source CityMapSpec & Parametrized Facilities."""

from __future__ import annotations
import json
from pathlib import Path
import pytest

from projects.shotgun_escape_the_heat.world.city_map_spec import (
    CityMapSpec,
    POISpec,
    POIType,
    RoadDef,
    create_default_city_spec,
)
from projects.shotgun_escape_the_heat.world.city_world import CityWorldBuilder


def test_city_map_spec_serialization_roundtrip(tmp_path: Path) -> None:
    """Verifies that CityMapSpec serializes to JSON and deserializes identically."""
    spec = create_default_city_spec()
    assert spec.map_size >= 250.0
    assert len(spec.roads) >= 8
    assert len(spec.pois) >= 8
    assert len(spec.buildings) >= 20

    json_file = tmp_path / "test_city.json"
    spec.save_json(json_file)
    assert json_file.is_file()

    loaded = CityMapSpec.load_json(json_file)
    assert loaded.map_id == spec.map_id
    assert loaded.map_size == spec.map_size
    assert loaded.safehouse_poi_id == spec.safehouse_poi_id
    assert len(loaded.roads) == len(spec.roads)
    assert len(loaded.pois) == len(spec.pois)
    assert len(loaded.buildings) == len(spec.buildings)


def test_parametrized_safehouse_moving() -> None:
    """Verifies that the safehouse can be dynamically relocated without code changes."""
    spec = create_default_city_spec()
    initial_sh = spec.get_safehouse()
    assert initial_sh.id == "poi_safehouse"
    orig_pos, orig_heading = spec.get_spawn_point()
    assert orig_pos[0] == initial_sh.position[0]

    # Relocate safehouse in the data model
    new_sh = POISpec(
        id="poi_safehouse_docks",
        name="Harbor Safehouse Loft",
        poi_type=POIType.SAFEHOUSE.value,
        position=(85.0, 0.0, 95.0),
        heading_deg=180.0,
        radius=12.0,
    )
    spec.pois.append(new_sh)
    spec.safehouse_poi_id = "poi_safehouse_docks"

    # Verify query reflects new safehouse
    curr_sh = spec.get_safehouse()
    assert curr_sh.id == "poi_safehouse_docks"
    new_spawn, new_heading = spec.get_spawn_point()
    assert new_spawn[0] == 85.0
    assert new_spawn[2] == 95.0
    assert new_heading == 180.0

    # Test CityWorldBuilder uses the relocated safehouse
    builder = CityWorldBuilder(map_data=spec)
    spawn_pos, spawn_h = builder.get_safehouse_spawn()
    assert spawn_pos[0] == 85.0
    assert spawn_pos[2] == 95.0
    assert spawn_h == 180.0


def test_dedicated_garage_facilities_registration() -> None:
    """Verifies that Respray, Chop Shop, Dealership, and Bribe stations are properly declared."""
    spec = create_default_city_spec()

    # Find dedicated stations
    resprays = spec.get_pois_by_type(POIType.RESPRAY)
    chop_shops = spec.get_pois_by_type(POIType.CHOP_SHOP)
    dealerships = spec.get_pois_by_type(POIType.DEALERSHIP)
    bribes = spec.get_pois_by_type(POIType.BRIBE_CONTACT)

    assert len(resprays) >= 1
    assert len(chop_shops) >= 1
    assert len(dealerships) >= 1
    assert len(bribes) >= 1

    # Verify StationManager translation in CityWorldBuilder
    builder = CityWorldBuilder(map_data=spec)
    stations = builder.get_stations()
    st_types = {s.station_type for s in stations}

    assert "garage" in st_types
    assert "respray" in st_types
    assert "chop_shop" in st_types
    assert "dealership" in st_types
    assert "bribe_contact" in st_types


def test_single_source_waypoint_and_route_generation() -> None:
    """Verifies that navigation waypoints and pedestrian routes derive directly from roads."""
    spec = create_default_city_spec()
    builder = CityWorldBuilder(map_data=spec)

    waypoints = builder.get_road_waypoints()
    assert len(waypoints) > 50

    routes = builder.get_pedestrian_routes()
    assert len(routes) >= 6
    for route in routes:
        assert len(route) == 4  # Closed rectangular sidewalk loop
