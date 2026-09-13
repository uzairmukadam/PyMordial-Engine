"""Unit Tests for Phase 1: Car Classes, Career Fleet Persistence, and Test Circuit POIs."""

import tempfile
from pathlib import Path
import pytest

from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_presets import (
    CarClass,
    CLASS_SPECS,
    VEHICLE_CATALOG,
    build_tuned_vehicle_config,
)
from projects.shotgun_escape_the_heat.modules.career_manager import CareerManagerModule, OwnedCarData
from projects.shotgun_escape_the_heat.modules.heat_system import HeatSystemModule
from projects.shotgun_escape_the_heat.world.test_circuit_world import TestCircuitWorldBuilder


def test_car_classes_and_specs():
    """Verify all 5 vehicle classes have complete physics specs and 4-stage multipliers."""
    expected_classes = {
        CarClass.STARTER_SEDAN,
        CarClass.CLASSIC_MUSCLE,
        CarClass.COMPACT_TUNER,
        CarClass.ENFORCER_SUV,
        CarClass.EXOTIC_COUPE,
    }
    assert set(CLASS_SPECS.keys()) == expected_classes

    for c_cls, spec in CLASS_SPECS.items():
        assert len(spec.engine_multipliers) == 4
        assert spec.engine_multipliers[0] == 1.0
        assert spec.engine_multipliers[3] > spec.engine_multipliers[0]

        assert len(spec.armor_damage_factors) == 4
        assert spec.armor_damage_factors[0] == 1.0
        assert spec.armor_damage_factors[3] < spec.armor_damage_factors[0]  # Armor reduces damage

        assert len(spec.handling_multipliers) == 4
        assert len(spec.brake_multipliers) == 4


def test_vehicle_catalog():
    """Verify vehicle catalog contains valid pricing, defaults, and classes."""
    assert "starter_sedan" in VEHICLE_CATALOG
    assert "classic_muscle" in VEHICLE_CATALOG
    assert "compact_tuner" in VEHICLE_CATALOG
    assert "enforcer_suv" in VEHICLE_CATALOG
    assert "exotic_coupe" in VEHICLE_CATALOG

    starter = VEHICLE_CATALOG["starter_sedan"]
    assert starter.price == 0  # Starter is free
    assert starter.car_class == CarClass.STARTER_SEDAN

    exotic = VEHICLE_CATALOG["exotic_coupe"]
    assert exotic.price > 30000
    assert exotic.car_class == CarClass.EXOTIC_COUPE


def test_build_tuned_vehicle_config():
    """Verify build_tuned_vehicle_config properly scales torque and grip with stages."""
    stock = build_tuned_vehicle_config("classic_muscle", engine_stage=0, handling_stage=0)
    tuned = build_tuned_vehicle_config("classic_muscle", engine_stage=3, handling_stage=3)

    assert tuned.engine_torque > stock.engine_torque
    assert tuned.top_speed_mps > stock.top_speed_mps
    assert tuned.tire_grip > stock.tire_grip


def test_career_manager_lifecycle():
    """Test purchasing, upgrading, repairing, and respraying vehicles."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_save.json"
        career = CareerManagerModule(save_file_path=save_path, initial_money=15000)

        # 1. Verify default starter sedan
        assert career.active_car_id == "starter_sedan"
        assert "starter_sedan" in career.owned_cars
        assert career.money == 15000

        # 2. Upgrade active car engine
        ok, msg = career.upgrade_spec("engine")
        assert ok is True
        assert career.active_car.engine_stage == 1
        assert career.money < 15000

        # 3. Respray active car
        new_color = (0.12, 0.45, 0.88)
        ok, msg = career.respray_car(new_color)
        assert ok is True
        assert career.get_active_car_color() == new_color

        # 4. Purchase new vehicle from catalog
        ok, msg = career.purchase_car("classic_muscle")
        assert ok is True
        assert "classic_muscle" in career.owned_cars

        # 5. Switch active car
        ok, msg = career.select_active_car("classic_muscle")
        assert ok is True
        assert career.active_car_id == "classic_muscle"

        # 6. Damage & Repair
        career.add_damage(40.0)
        assert career.active_car.damage == 40.0
        ok, msg = career.repair_car()
        assert ok is True
        assert career.active_car.damage == 0.0

        # 7. Heat & Bribe
        career.add_heat(50.0)
        assert career.active_car.heat == 50.0
        ok, msg = career.pay_bribe()
        assert ok is True
        assert career.active_car.heat == 0.0

        # 8. Persistence roundtrip
        career.save_to_disk()
        assert save_path.is_file()

        # Load fresh career instance from disk
        loaded_career = CareerManagerModule(save_file_path=save_path)
        assert loaded_career.active_car_id == "classic_muscle"
        assert "classic_muscle" in loaded_career.owned_cars
        assert "starter_sedan" in loaded_career.owned_cars
        assert loaded_career.owned_cars["starter_sedan"].engine_stage == 1
        assert loaded_career.owned_cars["starter_sedan"].color == new_color


def test_heat_system_tiers():
    """Verify police heat system tiers and status labeling."""
    hs = HeatSystemModule()
    assert hs.cop_tier == 0
    assert hs.status_label == "CLEAN"

    hs.add_heat(20.0)
    assert hs.cop_tier == 1
    assert hs.status_label == "PATROL ALERT"

    hs.add_heat(30.0)
    assert hs.cop_tier == 2
    assert hs.status_label == "HEAVY PURSUIT"

    hs.add_heat(35.0)
    assert hs.cop_tier == 3
    assert hs.status_label == "FULL APB & ROADBLOCKS"

    hs.reduce_heat(50.0)
    assert hs.cop_tier == 1


def test_test_circuit_poi_detection():
    """Verify testing circuit registers all 4 POIs and provides spatial detection."""
    builder = TestCircuitWorldBuilder(map_size=320.0)

    # Waypoint graph POI registry check
    assert len(builder.POI_DEFINITIONS) == 4
    for name, pdef in builder.POI_DEFINITIONS.items():
        assert "pos" in pdef
        assert "radius" in pdef
        assert "type" in pdef

    # Simulate building graph registry (add pois)
    from engine.world.waypoint_graph import PointOfInterest
    for name, pdef in builder.POI_DEFINITIONS.items():
        builder.waypoint_graph.add_poi(
            PointOfInterest(
                name=name,
                poi_type=pdef["type"],
                position=(pdef["pos"][0], 0.0, pdef["pos"][2]),
                radius=pdef["radius"],
                heading_deg=pdef["heading"],
            )
        )

    # Spawn query
    spawn_pos, spawn_heading = builder.get_poi_spawn("Safehouse Garage")
    assert spawn_pos[0] == 0.0
    assert spawn_pos[2] == 20.0

    # Proximity detection when parked inside Chop Shop
    chop_poi = builder.get_active_poi_at_pos((55.0, 0.0, -30.0))
    assert chop_poi is not None
    assert chop_poi.name == "Chop Shop"

    # Proximity detection when driving in middle of nowhere
    empty_poi = builder.get_active_poi_at_pos((120.0, 0.0, 50.0))
    assert empty_poi is None
