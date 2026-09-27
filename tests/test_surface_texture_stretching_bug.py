"""Test suite verifying fix for the surface texture stretching bug.

Ensures that world surfaces (ground planes, roads, walls, curbs, buildings) never
collide with vehicle multi-material shader branches even when assigned texture layers
1, 2, or 7, or when using tiled/negative UV coordinates.
"""

import pytest
import numpy as np

from engine.core.ecs import EntityManager
from engine.core.materials import (
    DisplacementMode,
    MAT_FLAG_HAS_TEXTURE,
    MAT_FLAG_DISP_SHIFT,
    MAT_FLAG_DISP_MASK,
    MAT_FLAG_VEHICLE_CHASSIS,
    MAT_FLAG_VEHICLE_WHEEL,
    encode_mat_flags,
    decode_mat_flags,
    MaterialRegistry,
)


def test_encode_decode_material_flags_isolation():
    """Verifies that vehicle chassis and wheel flags are isolated from has_texture and disp_mode."""
    # Standard surface with POM displacement and texture
    surface_flag = encode_mat_flags(
        has_texture=True,
        disp_mode=DisplacementMode.POM,
        is_vehicle_chassis=False,
        is_vehicle_wheel=False,
    )
    raw_int = int(surface_flag)
    assert (raw_int & MAT_FLAG_HAS_TEXTURE) != 0
    assert ((raw_int >> MAT_FLAG_DISP_SHIFT) & MAT_FLAG_DISP_MASK) == DisplacementMode.POM
    assert (raw_int & MAT_FLAG_VEHICLE_CHASSIS) == 0, "Surface must not have vehicle chassis flag"
    assert (raw_int & MAT_FLAG_VEHICLE_WHEEL) == 0, "Surface must not have vehicle wheel flag"

    # Vehicle chassis
    chassis_flag = encode_mat_flags(
        has_texture=True,
        disp_mode=DisplacementMode.NONE,
        is_vehicle_chassis=True,
        is_vehicle_wheel=False,
    )
    chassis_int = int(chassis_flag)
    assert (chassis_int & MAT_FLAG_VEHICLE_CHASSIS) != 0
    assert (chassis_int & MAT_FLAG_VEHICLE_WHEEL) == 0

    # Vehicle wheel
    wheel_flag = encode_mat_flags(
        has_texture=True,
        disp_mode=DisplacementMode.NONE,
        is_vehicle_chassis=False,
        is_vehicle_wheel=True,
    )
    wheel_int = int(wheel_flag)
    assert (wheel_int & MAT_FLAG_VEHICLE_CHASSIS) == 0
    assert (wheel_int & MAT_FLAG_VEHICLE_WHEEL) != 0

    # Backwards compatible decode_mat_flags
    has_tex, disp = decode_mat_flags(surface_flag)
    assert has_tex is True
    assert disp == DisplacementMode.POM


def test_surface_entity_creation_layer_collision_immunity():
    """Verifies that creating surface entities with layer 1, 2, or 7 does NOT set vehicle flags."""
    ecs = EntityManager(max_entities=100)

    # Road surface using layer 1 (historically collided with car_alloy_rim)
    road_id = ecs.create_entity(
        position=(0.0, 0.0, 0.0),
        scale=(14.0, 0.4, 60.0),
        layer_idx=1,
        disp_mode=DisplacementMode.NONE,
        is_static=True,
    )
    mat = ecs.get_entity_material(road_id)
    assert mat["layer_idx"] == 1
    assert mat["has_texture"] is True
    assert mat["is_vehicle_chassis"] is False, "Road on layer 1 must not be marked as vehicle chassis"
    assert mat["is_vehicle_wheel"] is False, "Road on layer 1 must not be marked as vehicle wheel"

    # Building surface using layer 2 (historically collided with car_body_paint)
    building_id = ecs.create_entity(
        position=(20.0, 10.0, 0.0),
        scale=(20.0, 20.0, 20.0),
        layer_idx=2,
        disp_mode=DisplacementMode.POM,
        is_static=True,
    )
    mat_b = ecs.get_entity_material(building_id)
    assert mat_b["layer_idx"] == 2
    assert mat_b["is_vehicle_chassis"] is False, "Building on layer 2 must not be marked as vehicle chassis"
    assert mat_b["is_vehicle_wheel"] is False, "Building on layer 2 must not be marked as vehicle wheel"

    # Ground plane using layer 7 (historically collided with car_tire_rubber)
    ground_id = ecs.create_entity(
        position=(0.0, 0.0, 0.0),
        scale=(1000.0, 1.0, 1000.0),
        layer_idx=7,
        is_static=True,
    )
    mat_g = ecs.get_entity_material(ground_id)
    assert mat_g["layer_idx"] == 7
    assert mat_g["is_vehicle_chassis"] is False
    assert mat_g["is_vehicle_wheel"] is False


def test_vehicle_entity_creation_flags():
    """Verifies that vehicle chassis and wheel entities correctly receive their respective flags."""
    ecs = EntityManager(max_entities=100)

    # Player chassis
    chassis_id = ecs.create_entity(
        position=(0.0, 1.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        layer_idx=2,
        is_vehicle_chassis=True,
    )
    ch_mat = ecs.get_entity_material(chassis_id)
    assert ch_mat["is_vehicle_chassis"] is True
    assert ch_mat["is_vehicle_wheel"] is False

    # Player wheel
    wheel_id = ecs.create_entity(
        position=(1.0, 0.5, 1.0),
        scale=(1.0, 1.0, 1.0),
        layer_idx=1,
        is_vehicle_wheel=True,
    )
    wh_mat = ecs.get_entity_material(wheel_id)
    assert wh_mat["is_vehicle_chassis"] is False
    assert wh_mat["is_vehicle_wheel"] is True


def test_material_registry_sync_vehicle_material_classification():
    """Verifies that MaterialRegistry accurately classifies vehicle materials without affecting scene materials."""
    reg = MaterialRegistry()

    class MockAtlas:
        name_to_layer = {
            "car_body_paint": 2,
            "car_alloy_rim": 1,
            "car_tire_rubber": 7,
            "asphalt_worn": 3,
            "castle_brick_02_red": 4,
            "concrete_floor_worn_02": 5,
        }
        material_depths = np.zeros(16, dtype=np.float32)

    reg.sync_with_atlas(MockAtlas())

    paint = reg.get("car_body_paint")
    assert paint is not None
    assert paint.is_vehicle_chassis is True
    assert paint.is_vehicle_wheel is False

    rim = reg.get("car_alloy_rim")
    assert rim is not None
    assert rim.is_vehicle_chassis is False
    assert rim.is_vehicle_wheel is True

    asphalt = reg.get("asphalt_worn")
    assert asphalt is not None
    assert asphalt.is_vehicle_chassis is False
    assert asphalt.is_vehicle_wheel is False

    brick = reg.get("castle_brick_02_red")
    assert brick is not None
    assert brick.is_vehicle_chassis is False
    assert brick.is_vehicle_wheel is False


def test_gbuffer_shader_no_horizontal_banding():
    """Verifies that shaders/gbuffer.frag has no unbandlimited micro_grain and uses geometric UV derivatives."""
    from pathlib import Path
    shader_path = Path(__file__).resolve().parent.parent / "shaders" / "gbuffer.frag"
    code = shader_path.read_text(encoding="utf-8")

    assert "micro_grain" not in code, "Unbandlimited procedural micro_grain causes severe horizontal Moiré banding"
    assert "dFdx(v_UV)" in code, "Mesh texture sampling must use geometric v_UV derivatives to eliminate POM step banding"
    assert "dFdy(v_UV)" in code

