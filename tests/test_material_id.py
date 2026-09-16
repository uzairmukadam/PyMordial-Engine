"""Unit tests for Phase 2: Per-Entity Material ID and MaterialRegistry System."""

import pytest
import numpy as np

from engine.core.ecs import EntityManager, TransformProxy
from engine.gfx.material_registry import MaterialDef, MaterialRegistry
from engine.gfx.texture_atlas import DisplacementMode, encode_mat_flags, decode_mat_flags


def test_material_registry_builtins():
    """Verifies that standard city map and asset materials are pre-registered with correct properties."""
    registry = MaterialRegistry()

    # Asphalt Road
    asphalt = registry.get("asphalt_road")
    assert asphalt is not None
    assert asphalt.name == "asphalt_road"
    assert asphalt.friction == 1.00
    assert asphalt.roughness >= 0.8
    assert asphalt.metallic <= 0.05

    # Concrete Sidewalk
    concrete = registry.get("concrete_sidewalk")
    assert concrete is not None
    assert concrete.friction == 0.85
    assert concrete.disp_depth == 0.005

    # Brick Wall
    brick = registry.get("brick_wall")
    assert brick is not None
    assert brick.friction == 0.80
    assert brick.disp_depth == 0.035

    # Metal Guardrail
    metal = registry.get("metal_guardrail")
    assert metal is not None
    assert metal.metallic >= 0.85
    assert metal.roughness <= 0.35

    # Glass
    glass = registry.get("glass")
    assert glass is not None
    assert glass.roughness <= 0.15


def test_material_registry_aliases():
    """Verifies that aliases resolve to their canonical material definitions."""
    registry = MaterialRegistry()

    assert registry.get("asphalt") is registry.get("asphalt_road")
    assert registry.get("road") is registry.get("asphalt_road")
    assert registry.get("sidewalk") is registry.get("concrete_sidewalk")
    assert registry.get("concrete") is registry.get("concrete_sidewalk")
    assert registry.get("brick") is registry.get("brick_wall")
    assert registry.get("metal") is registry.get("metal_guardrail")
    assert registry.get("guardrail") is registry.get("metal_guardrail")
    assert registry.get("window") is registry.get("glass")
    assert registry.get("mud") is registry.get("mud_cracked_dry_03")
    assert registry.get("roof") is registry.get("clay_roof_tiles_02")


def test_material_registry_custom_registration():
    """Verifies registering new custom materials with unique IDs."""
    registry = MaterialRegistry()
    initial_count = len(registry.list_materials())

    custom = registry.register_material(
        name="neon_cyber_panel",
        layer_idx=5,
        has_texture=True,
        disp_mode=DisplacementMode.POM,
        color=(0.1, 0.9, 0.8),
        roughness=0.15,
        metallic=0.70,
        ao=1.0,
        disp_depth=0.020,
        friction=0.75,
        restitution=0.10,
        description="Glow panel surface",
    )

    assert custom.name == "neon_cyber_panel"
    assert custom.material_id >= 0
    assert registry.get("neon_cyber_panel") is custom
    assert registry.get_by_id(custom.material_id) is custom
    assert len(registry.list_materials()) == initial_count + 1


def test_material_registry_sync_with_mock_atlas():
    """Verifies synchronizing texture atlas layers into material definitions."""
    registry = MaterialRegistry()

    class MockAtlas:
        def __init__(self):
            self.name_to_layer = {
                "identity": 0,
                "mud_cracked_dry_03": 1,
                "castle_brick_02_red": 2,
                "concrete_floor_worn_02": 3,
                "custom_asphalt_layer": 4,
            }
            self.material_depths = np.array([0.0, 0.045, 0.035, 0.005, 0.010], dtype=np.float32)

    atlas = MockAtlas()
    registry.sync_with_atlas(atlas)

    # Mud cracked
    mud = registry.get("mud_cracked_dry_03")
    assert mud.layer_idx == 1
    assert mud.has_texture is True
    assert mud.disp_depth == pytest.approx(0.045)
    assert mud.disp_mode == DisplacementMode.POM

    # Castle brick
    brick = registry.get("castle_brick_02_red")
    assert brick.layer_idx == 2
    assert brick.has_texture is True
    assert brick.disp_depth == pytest.approx(0.035)

    # Aliased city materials updated
    brick_wall = registry.get("brick_wall")
    assert brick_wall.layer_idx == 2
    assert brick_wall.has_texture is True

    sidewalk = registry.get("concrete_sidewalk")
    assert sidewalk.layer_idx == 3
    assert sidewalk.has_texture is True


def test_ecs_create_entity_with_material_id():
    """Verifies that ecs.create_entity(material_id=...) sets the SSBO 2 material slot correctly."""
    ecs = EntityManager(max_entities=100)
    registry = MaterialRegistry()
    ecs.set_material_registry(registry)

    # 1. Untextured Asphalt Road
    ent_road = ecs.create_entity(material_id="asphalt_road")
    mat_road = ecs.get_entity_material(ent_road)
    assert mat_road["roughness"] == pytest.approx(0.85)
    assert mat_road["metallic"] == pytest.approx(0.02)
    assert mat_road["layer_idx"] == 0
    assert mat_road["has_texture"] is False

    # 2. Textured Mud Cracked with POM
    ent_mud = ecs.create_entity(
        material_id="mud_cracked_dry_03",
        layer_idx=1,
        disp_mode=DisplacementMode.POM,
    )
    mat_mud = ecs.get_entity_material(ent_mud)
    assert mat_mud["layer_idx"] == 1
    assert mat_mud["has_texture"] is True
    assert mat_mud["disp_mode"] == DisplacementMode.POM

    # 3. Custom Color Override on Metal Guardrail
    ent_metal = ecs.create_entity(
        material_id="metal_guardrail",
        color=(1.0, 0.0, 0.0),  # Red painted guardrail
    )
    mat_metal = ecs.get_entity_material(ent_metal)
    assert mat_metal["color"] == (1.0, 0.0, 0.0)
    assert mat_metal["metallic"] == pytest.approx(0.90)
    assert mat_metal["roughness"] == pytest.approx(0.25)


def test_ecs_set_entity_material_and_proxy():
    """Verifies updating an entity's material at runtime via ecs and TransformProxy."""
    ecs = EntityManager(max_entities=100)
    registry = MaterialRegistry()
    ecs.set_material_registry(registry)

    ent = ecs.create_entity()
    # Default is matte white
    mat_before = ecs.get_entity_material(ent)
    assert mat_before["color"] == (1.0, 1.0, 1.0)
    assert mat_before["roughness"] == pytest.approx(0.5)

    # Change to glass
    ecs.set_entity_material(ent, material_id="glass")
    mat_after = ecs.get_entity_material(ent)
    assert mat_after["roughness"] == pytest.approx(0.08)
    assert mat_after["metallic"] == pytest.approx(0.10)

    # Modify via TransformProxy
    proxy = ecs.get_transform_proxy(ent)
    proxy.set_material(material_id="concrete_sidewalk", color=(0.8, 0.8, 0.8))
    assert proxy.roughness == pytest.approx(0.75)
    np.testing.assert_allclose(proxy.color, [0.8, 0.8, 0.8], atol=1e-5)


def test_ecs_multi_entity_ssbo_material_layout():
    """Verifies that multiple entities with distinct materials pack into contiguous SSBO 2 layout without crosstalk."""
    ecs = EntityManager(max_entities=100)
    registry = MaterialRegistry()
    ecs.set_material_registry(registry)

    e1 = ecs.create_entity(material_id="asphalt_road")
    e2 = ecs.create_entity(material_id="concrete_sidewalk", layer_idx=3, disp_mode=DisplacementMode.POM)
    e3 = ecs.create_entity(material_id="metal_guardrail", layer_idx=4)
    e4 = ecs.create_entity(color=(0.2, 0.4, 0.8), roughness=0.1, metallic=0.9)

    raw_mat_view = ecs.get_active_materials_view()
    assert raw_mat_view.shape == (4, 8)
    assert raw_mat_view.flags["C_CONTIGUOUS"]
    assert raw_mat_view.dtype == np.float32

    # Entity 1: Asphalt (layer 0)
    assert raw_mat_view[0, 6] == 0.0  # layer_idx
    has_tex_1, disp_1 = decode_mat_flags(raw_mat_view[0, 7])
    assert has_tex_1 is False

    # Entity 2: Concrete (layer 3, POM)
    assert raw_mat_view[1, 6] == 3.0  # layer_idx
    has_tex_2, disp_2 = decode_mat_flags(raw_mat_view[1, 7])
    assert has_tex_2 is True
    assert disp_2 == DisplacementMode.POM

    # Entity 3: Metal Guardrail (layer 4, None)
    assert raw_mat_view[2, 6] == 4.0  # layer_idx
    has_tex_3, disp_3 = decode_mat_flags(raw_mat_view[2, 7])
    assert has_tex_3 is True
    assert disp_3 == DisplacementMode.NONE

    # Entity 4: Blue shiny metal
    np.testing.assert_allclose(raw_mat_view[3, 0:3], [0.2, 0.4, 0.8], atol=1e-5)
    assert raw_mat_view[3, 3] == pytest.approx(0.1, abs=1e-5)
    assert raw_mat_view[3, 4] == pytest.approx(0.9, abs=1e-5)
    assert raw_mat_view[3, 6] == 0.0
