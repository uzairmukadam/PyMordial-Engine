"""Unit Tests for Mapping Subsystem High-Level Physics Schemas and Material Presets."""

from __future__ import annotations

import pytest

from engine.world.mapping.schema import (
    PHYSICAL_MATERIALS,
    CollisionDef,
    EntityDef,
    PhysicalMaterial,
    PhysicsDef,
    PhysicsRole,
    TransformDef,
    VisualDef,
    get_physical_material,
)


def test_physics_role_enum_and_validation() -> None:
    assert PhysicsRole.STATIC == "static"
    assert PhysicsRole.KINEMATIC == "kinematic"
    assert PhysicsRole.DYNAMIC == "dynamic"
    assert PhysicsRole.SENSOR == "sensor"
    assert PhysicsRole.NONE == "none"

    # Valid string roles in PhysicsDef
    p_stat = PhysicsDef(role="STATIC")
    assert p_stat.role == PhysicsRole.STATIC

    p_dyn = PhysicsDef(role="dynamic")
    assert p_dyn.role == PhysicsRole.DYNAMIC

    # Invalid role raises ValueError
    with pytest.raises(ValueError, match="Unknown physics role 'invalid'"):
        PhysicsDef(role="invalid")


def test_physical_materials_presets_and_lookup() -> None:
    assert "wood" in PHYSICAL_MATERIALS
    assert "metal" in PHYSICAL_MATERIALS
    assert "concrete" in PHYSICAL_MATERIALS
    assert "rubber" in PHYSICAL_MATERIALS
    assert "ice" in PHYSICAL_MATERIALS
    assert "cardboard" in PHYSICAL_MATERIALS

    wood = get_physical_material("wood")
    assert wood.friction == 0.65
    assert wood.restitution == 0.15
    assert wood.density == 650.0

    metal = get_physical_material("METAL")  # case-insensitive
    assert metal.friction == 0.50
    assert metal.restitution == 0.20
    assert metal.density == 7800.0

    rubber = get_physical_material("rubber")
    assert rubber.restitution == 0.80

    ice = get_physical_material("ice")
    assert ice.friction == 0.05

    # Unknown or None falls back to default
    default_mat = get_physical_material(None)
    assert default_mat.name == "default"
    assert default_mat.friction == 0.70

    unknown_mat = get_physical_material("unobtainium")
    assert unknown_mat.name == "default"


def test_physics_def_material_inheritance_and_overrides() -> None:
    # Wood inherits wood friction & restitution
    p_wood = PhysicsDef(role=PhysicsRole.DYNAMIC, material="wood")
    assert p_wood.friction == 0.65
    assert p_wood.restitution == 0.15
    assert p_wood.density == 650.0

    # Explicit override takes precedence
    p_custom = PhysicsDef(role=PhysicsRole.DYNAMIC, material="wood", friction=0.99, restitution=0.42)
    assert p_custom.friction == 0.99
    assert p_custom.restitution == 0.42
    assert p_custom.density == 650.0  # density remains inherited from wood


def test_physics_def_serialization_roundtrip() -> None:
    p = PhysicsDef(
        role=PhysicsRole.DYNAMIC,
        material="metal",
        mass=45.0,
        ccd=True,
        can_sleep=False,
        dimensions=(1.0, 2.0, 1.0),
    )
    d = p.to_dict()
    assert d["role"] == "dynamic"
    assert d["material"] == "metal"
    assert d["mass"] == 45.0
    assert d["ccd"] is True
    assert d["can_sleep"] is False
    assert d["dimensions"] == [1.0, 2.0, 1.0]

    reconstructed = PhysicsDef.from_dict(d)
    assert reconstructed.role == PhysicsRole.DYNAMIC
    assert reconstructed.material == "metal"
    assert reconstructed.mass == 45.0
    assert reconstructed.ccd is True
    assert reconstructed.can_sleep is False
    assert reconstructed.dimensions == (1.0, 2.0, 1.0)


def test_entity_def_resolved_collision_from_physics_archetypes() -> None:
    # 1. Static Scenery with auto-derived box dimensions
    e_static = EntityDef(
        id="stone_wall",
        transform=TransformDef(position=(0.0, 1.0, 0.0), scale=(10.0, 4.0, 1.0)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.STATIC, material="stone"),
    )
    col_static = e_static.resolved_collision()
    assert col_static is not None
    assert col_static.enabled is True
    assert col_static.body_type == "fixed"
    assert col_static.shape == "box"
    assert col_static.dimensions == (5.0, 2.0, 0.5)  # half of (10, 4, 1)
    assert col_static.friction == 0.85

    # 2. Kinematic Moving Platform
    e_kinematic = EntityDef(
        id="lift",
        transform=TransformDef(position=(0.0, 0.0, 0.0), scale=(4.0, 0.5, 4.0)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.KINEMATIC, material="metal"),
    )
    col_kin = e_kinematic.resolved_collision()
    assert col_kin is not None
    assert col_kin.body_type == "kinematic_velocity"
    assert col_kin.dimensions == (2.0, 0.25, 2.0)
    assert col_kin.friction == 0.50

    # 3. Dynamic Pushable Wooden Crate
    e_crate = EntityDef(
        id="crate_01",
        transform=TransformDef(position=(5.0, 0.5, 0.0), scale=(1.2, 1.2, 1.2)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.DYNAMIC, material="wood", mass=20.0),
    )
    col_dyn = e_crate.resolved_collision()
    assert col_dyn is not None
    assert col_dyn.body_type == "dynamic"
    assert col_dyn.shape == "box"
    assert col_dyn.dimensions == (0.6, 0.6, 0.6)
    assert col_dyn.friction == 0.65
    assert col_dyn.restitution == 0.15

    # 4. Dynamic Rolling Sphere with derived radius
    e_sphere = EntityDef(
        id="rubber_ball",
        transform=TransformDef(position=(0.0, 5.0, 0.0), scale=(2.0, 2.0, 2.0)),
        visual=VisualDef(shape="sphere"),
        physics=PhysicsDef(role=PhysicsRole.DYNAMIC, material="rubber"),
    )
    col_sphere = e_sphere.resolved_collision()
    assert col_sphere is not None
    assert col_sphere.body_type == "dynamic"
    assert col_sphere.shape == "sphere"
    assert col_sphere.dimensions == (1.0,)  # radius 1.0 (half of 2.0)
    assert col_sphere.restitution == 0.80

    # 5. Cylinder auto-derivation (height = sy * 0.5, radius = sx * 1.0)
    e_cylinder = EntityDef(
        id="pedestal",
        transform=TransformDef(position=(0.0, 0.5, 0.0), scale=(1.5, 2.0, 1.5)),
        visual=VisualDef(shape="cylinder"),
        physics=PhysicsDef(role=PhysicsRole.STATIC, material="concrete"),
    )
    col_cyl = e_cylinder.resolved_collision()
    assert col_cyl is not None
    assert col_cyl.shape == "cylinder"
    assert col_cyl.dimensions == (1.0, 1.5)  # half_height 1.0, radius 1.5

    # 6. Sensor Zone (Zero physical collision resistance)
    e_sensor = EntityDef(
        id="zone_portal",
        transform=TransformDef(position=(0.0, 0.0, 0.0), scale=(3.0, 3.0, 3.0)),
        physics=PhysicsDef(role=PhysicsRole.SENSOR),
    )
    assert e_sensor.resolved_collision() is None

    # 7. None Role
    e_none = EntityDef(
        id="sky_dome_decal",
        transform=TransformDef(position=(0.0, 100.0, 0.0)),
        visual=VisualDef(shape="cube"),
        physics=PhysicsDef(role=PhysicsRole.NONE),
    )
    assert e_none.resolved_collision() is None


def test_backward_compatibility_with_legacy_collision_and_json() -> None:
    # Explicit legacy collision block takes priority if provided
    raw_json = {
        "id": "legacy_box",
        "transform": {
            "position": [1.0, 2.0, 3.0],
            "scale": [2.0, 2.0, 2.0],
        },
        "visual": {
            "shape": "cube",
        },
        "collision": {
            "enabled": True,
            "body_type": "dynamic",
            "shape": "box",
            "dimensions": [1.0, 1.0, 1.0],
            "friction": 0.44,
            "restitution": 0.11,
        },
    }
    entity = EntityDef.from_dict(raw_json)
    assert entity.collision is not None
    assert entity.collision.friction == 0.44
    col = entity.resolved_collision()
    assert col is not None
    assert col.friction == 0.44

    # Synthesizes PhysicsDef cleanly
    phys = entity.resolved_physics()
    assert phys is not None
    assert phys.role == PhysicsRole.DYNAMIC
    assert phys.friction == 0.44
