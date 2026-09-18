"""Automated Test Suite for Vehicle Model Registry, Baked 3D Meshes, and Dynamic Coloring."""

import json
from pathlib import Path
import pytest
import numpy as np

from engine.assets.mesh_format import PMMesh, VERTEX_DTYPE
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_presets import CarClass
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_model_registry import (
    VehicleModelRegistry,
    CIVILIAN_CAR_COLORS,
)


def test_catalog_integrity():
    """Verifies that catalog.json exists and all 11 baked vehicles have valid assets."""
    catalog_path = Path("projects/shotgun_escape_the_heat/assets/models/cars/catalog.json")
    assert catalog_path.is_file(), f"Missing catalog at {catalog_path}"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert len(catalog) == 11, f"Expected 11 vehicles in catalog, found {len(catalog)}"

    required_keys = ["starter_sedan", "classic_muscle", "compact_tuner", "enforcer_suv", "exotic_coupe",
                     "police_cruiser", "police_interceptor",
                     "civilian_sedan_1", "civilian_sedan_2", "civilian_compact", "civilian_suv"]
    for k in required_keys:
        assert k in catalog, f"Missing {k} in catalog"

    base_dir = catalog_path.parent.parent.parent.parent
    for car_id, entry in catalog.items():
        chassis_file = base_dir / entry["chassis_mesh_path"]
        wheel_file = base_dir / entry["wheel_mesh_path"]
        assert chassis_file.is_file(), f"Missing chassis mesh for {car_id}: {chassis_file}"
        assert wheel_file.is_file(), f"Missing wheel mesh for {car_id}: {wheel_file}"

        attach = entry["wheel_attachments"]
        for corner in ("FL", "FR", "RL", "RR"):
            assert corner in attach, f"Missing corner {corner} in {car_id}"
            pos = attach[corner]
            assert len(pos) == 3
            # Front wheels must have Z < 0, rear wheels Z > 0 in PyMordial coordinates
            if "F" in corner:
                assert pos[2] < 0.0, f"Front wheel {corner} must have Z < 0: {pos[2]}"
            else:
                assert pos[2] > 0.0, f"Rear wheel {corner} must have Z > 0: {pos[2]}"


def test_pm_mesh_vertex_stride_and_paint_mask():
    """Verifies that .pm_mesh files comply with 32-byte cache alignment and body paint mask."""
    catalog_path = Path("projects/shotgun_escape_the_heat/assets/models/cars/catalog.json")
    base_dir = catalog_path.parent.parent.parent.parent

    for car_id in ("starter_sedan", "classic_muscle", "police_cruiser"):
        chassis_file = base_dir / f"assets/models/cars/{car_id}/chassis.pm_mesh"
        wheel_file = base_dir / f"assets/models/cars/{car_id}/wheel.pm_mesh"

        chassis = PMMesh.load(chassis_file)
        wheel = PMMesh.load(wheel_file)

        assert chassis.vertices.dtype == VERTEX_DTYPE
        assert chassis.vertices.itemsize == 32
        assert len(chassis.indices) > 0
        assert len(chassis.indices) % 3 == 0

        # Chassis must have paintable vertices (U < 0.0)
        u_vals = chassis.vertices["uv"][:, 0]
        paintable_count = np.count_nonzero(u_vals < 0.0)
        unpainted_count = np.count_nonzero(u_vals >= 0.0)
        assert paintable_count > 0, f"{car_id} chassis must contain paintable body vertices (U < 0.0)"
        assert unpainted_count > 0, f"{car_id} chassis must contain unpainted detail vertices (glass, lights, chrome)"

        # Wheel must NOT have negative U (all tire rubber and rim textures)
        w_u_vals = wheel.vertices["uv"][:, 0]
        assert np.all(w_u_vals >= 0.0), f"{car_id} wheel should not have negative U"


def test_vehicle_model_registry_queries():
    """Verifies that VehicleModelRegistry helper methods provide correct models and colors."""
    # Test random civilian colors
    for _ in range(20):
        c = VehicleModelRegistry.get_random_npc_color()
        assert len(c) == 3
        assert c in CIVILIAN_CAR_COLORS

    # Test police color schemes
    p1 = VehicleModelRegistry.get_police_color(1)
    p2 = VehicleModelRegistry.get_police_color(2)
    assert p1 == (0.95, 0.95, 0.95)  # Cruiser White
    assert p2 == (0.08, 0.08, 0.08)  # Interceptor Black
