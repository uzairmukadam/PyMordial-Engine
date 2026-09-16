"""PBR Material Definition and Registry Subsystem.

Defines PBR material properties, texture displacement modes, and the central
MaterialRegistry for mapping material IDs to atlas layers and PBR attributes.
Pure CPU data structures with zero dependencies on OpenGL or render pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from engine.gfx.texture_atlas import TextureArrayAtlas


class DisplacementMode(IntEnum):
    """Mutually exclusive displacement modes per entity."""
    NONE = 0          # Standard PBR (normal mapping only, no displacement)
    POM = 1           # Parallax Occlusion Mapping (fragment shader raymarch + self-shadowing)
    SSDM = 2          # Screen-Space Displacement Mapping (post-G-buffer depth extrusion)
    TESSELLATION = 3  # Hardware Tessellation (GPU TCS+TES vertex displacement along normal)


# Material flag bitfield layout stored in material_data[entity, 7]:
# Bit 0: MAT_FLAG_HAS_TEXTURE (1 = sample texture arrays)
# Bits 1..2: Displacement Mode (2 bits: 0=None, 1=POM, 2=SSDM, 3=Tessellation)
MAT_FLAG_HAS_TEXTURE = 1 << 0
MAT_FLAG_DISP_SHIFT = 1
MAT_FLAG_DISP_MASK = 0x3


def encode_mat_flags(has_texture: bool, disp_mode: DisplacementMode | int = DisplacementMode.NONE) -> float:
    """Encodes material flags and displacement mode into a float for material_data[entity, 7]."""
    mode_val = int(disp_mode)
    flags = (1 if has_texture else 0) | ((mode_val & MAT_FLAG_DISP_MASK) << MAT_FLAG_DISP_SHIFT)
    return float(flags)


def decode_mat_flags(flag_val: float | int) -> tuple[bool, DisplacementMode]:
    """Decodes material flags and displacement mode from material_data[entity, 7]."""
    val = int(flag_val)
    has_texture = bool(val & MAT_FLAG_HAS_TEXTURE)
    disp_mode = DisplacementMode((val >> MAT_FLAG_DISP_SHIFT) & MAT_FLAG_DISP_MASK)
    return has_texture, disp_mode


DEFAULT_MATERIAL_DEPTHS: dict[str, float] = {
    "identity": 0.0,
    "clay_roof_tiles_02": 0.040,      # Terracotta roof tiles (4.0 cm physical depth)
    "metal_grate_rusty": 0.015,       # Metal floor grate (1.5 cm physical depth)
    "castle_brick_02_red": 0.035,     # Exterior masonry brick (3.5 cm depth)
    "mud_cracked_dry_03": 0.045,      # Dry cracked ground fissures (4.5 cm depth)
    "floor_pattern_02": 0.025,        # Decorative stone floor tiles (2.5 cm depth)
    "ribbed_corduroy": 0.008,         # Fabric corduroy micro-ribs (0.8 cm depth)
    "concrete_floor_worn_02": 0.005,  # Worn industrial concrete (0.5 cm depth)
}


@dataclass
class MaterialDef:
    """Complete PBR material definition for an entity.

    Mirrors the std430 SSBO 2 MaterialBuffer layout:
    - [0:3]: Base color / tint (R, G, B)
    - [3]: Roughness (0.0 = mirror smooth, 1.0 = completely rough)
    - [4]: Metallic (0.0 = dielectric, 1.0 = pure metal)
    - [5]: Ambient Occlusion multiplier (1.0 = unoccluded)
    - [6]: Texture array atlas layer index (0 = identity / untextured)
    - [7]: Bitfield flags (bit 0: has_texture, bits 1..2: DisplacementMode)
    """

    name: str
    material_id: int
    layer_idx: int = 0
    has_texture: bool = False
    disp_mode: DisplacementMode = DisplacementMode.NONE
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    ao: float = 1.0
    disp_depth: float = 0.030
    friction: float = 0.80       # Surface friction coefficient for vehicle tires
    restitution: float = 0.05    # Surface restitution (bounciness)
    description: str = ""

    def encode_flags(self) -> float:
        """Encodes texture usage and displacement mode into a float bitfield."""
        return encode_mat_flags(
            has_texture=(self.has_texture and self.layer_idx > 0),
            disp_mode=self.disp_mode,
        )

    def to_material_floats(self) -> np.ndarray:
        """Returns 8 float32 values matching the MaterialData layout."""
        return np.array(
            [
                float(self.color[0]),
                float(self.color[1]),
                float(self.color[2]),
                float(self.roughness),
                float(self.metallic),
                float(self.ao),
                float(self.layer_idx),
                self.encode_flags(),
            ],
            dtype=np.float32,
        )


class MaterialRegistry:
    """Central registry mapping material names and numeric IDs to PBR definitions and atlas layers."""

    __slots__ = (
        "_materials_by_name",
        "_materials_by_id",
        "_alias_to_name",
        "_next_id",
    )

    def __init__(self) -> None:
        self._materials_by_name: dict[str, MaterialDef] = {}
        self._materials_by_id: dict[int, MaterialDef] = {}
        self._alias_to_name: dict[str, str] = {}
        self._next_id = 0
        self._register_default_materials()

    def _register_default_materials(self) -> None:
        """Initializes built-in materials for city environments and standard assets."""
        # 0. Default Identity (Layer 0, untextured clean PBR)
        self.register_material(
            name="default",
            color=(1.0, 1.0, 1.0),
            roughness=0.5,
            metallic=0.0,
            ao=1.0,
            description="Default white matte PBR surface",
        )
        self.add_alias("identity", "default")
        self.add_alias("none", "default")

        # 1. City Road Asphalt
        self.register_material(
            name="asphalt_road",
            color=(0.18, 0.18, 0.20),
            roughness=0.85,
            metallic=0.02,
            ao=1.0,
            friction=1.00,
            restitution=0.02,
            description="High-friction worn city road asphalt",
        )
        self.add_alias("asphalt", "asphalt_road")
        self.add_alias("road", "asphalt_road")

        # 2. Concrete Sidewalk & Curbs
        self.register_material(
            name="concrete_sidewalk",
            color=(0.58, 0.58, 0.56),
            roughness=0.75,
            metallic=0.02,
            ao=1.0,
            disp_depth=0.005,
            friction=0.85,
            restitution=0.05,
            description="Light gray cast concrete sidewalk and street curbs",
        )
        self.add_alias("concrete", "concrete_sidewalk")
        self.add_alias("sidewalk", "concrete_sidewalk")

        # 3. Exterior Brick Wall (Masonry)
        self.register_material(
            name="brick_wall",
            color=(0.62, 0.24, 0.18),
            roughness=0.80,
            metallic=0.0,
            ao=1.0,
            disp_depth=0.035,
            friction=0.80,
            restitution=0.05,
            description="Red clay brick masonry exterior building wall",
        )
        self.add_alias("brick", "brick_wall")

        # 4. Metal Guardrail & Street Fixtures
        self.register_material(
            name="metal_guardrail",
            color=(0.78, 0.80, 0.82),
            roughness=0.25,
            metallic=0.90,
            ao=1.0,
            disp_depth=0.015,
            friction=0.60,
            restitution=0.15,
            description="Galvanized steel barrier and metal fixtures",
        )
        self.add_alias("metal", "metal_guardrail")
        self.add_alias("guardrail", "metal_guardrail")

        # 5. Architectural Glass
        self.register_material(
            name="glass",
            color=(0.65, 0.85, 0.90),
            roughness=0.08,
            metallic=0.10,
            ao=1.0,
            friction=0.50,
            restitution=0.10,
            description="Smooth reflective window glass",
        )
        self.add_alias("window", "glass")

        # 6. Grass & Green Spaces
        self.register_material(
            name="grass",
            color=(0.20, 0.42, 0.15),
            roughness=0.92,
            metallic=0.0,
            ao=1.0,
            friction=0.55,
            restitution=0.02,
            description="Vegetation lawn and grassy roadside shoulder",
        )

        # 7. Loose Gravel & Dirt
        self.register_material(
            name="gravel",
            color=(0.42, 0.39, 0.35),
            roughness=0.95,
            metallic=0.0,
            ao=1.0,
            friction=0.45,
            restitution=0.02,
            description="Loose unpaved gravel road and construction fill",
        )
        self.add_alias("dirt", "gravel")

        # 8. Mud Cracked Dry (Matches asset 'mud_cracked_dry_03')
        self.register_material(
            name="mud_cracked_dry_03",
            color=(0.65, 0.55, 0.42),
            roughness=0.88,
            metallic=0.02,
            ao=1.0,
            disp_depth=0.045,
            disp_mode=DisplacementMode.POM,
            friction=0.65,
            restitution=0.02,
            description="Dry cracked mud terrain with POM fissures",
        )
        self.add_alias("mud", "mud_cracked_dry_03")

        # 9. Castle Brick Red (Matches asset 'castle_brick_02_red')
        self.register_material(
            name="castle_brick_02_red",
            color=(0.65, 0.25, 0.18),
            roughness=0.80,
            metallic=0.0,
            ao=1.0,
            disp_depth=0.035,
            disp_mode=DisplacementMode.POM,
            friction=0.80,
            restitution=0.05,
            description="Exterior masonry castle brick with POM depth",
        )

        # 10. Clay Roof Tiles (Matches asset 'clay_roof_tiles_02')
        self.register_material(
            name="clay_roof_tiles_02",
            color=(0.70, 0.35, 0.20),
            roughness=0.75,
            metallic=0.0,
            ao=1.0,
            disp_depth=0.040,
            disp_mode=DisplacementMode.POM,
            description="Terracotta clay roof tiles with POM depth",
        )
        self.add_alias("roof", "clay_roof_tiles_02")

        # 11. Concrete Floor Worn (Matches asset 'concrete_floor_worn_02')
        self.register_material(
            name="concrete_floor_worn_02",
            color=(0.55, 0.55, 0.54),
            roughness=0.70,
            metallic=0.02,
            ao=1.0,
            disp_depth=0.005,
            disp_mode=DisplacementMode.POM,
            description="Worn industrial concrete surface",
        )

        # 12. Floor Pattern (Matches asset 'floor_pattern_02')
        self.register_material(
            name="floor_pattern_02",
            color=(0.60, 0.58, 0.54),
            roughness=0.65,
            metallic=0.05,
            ao=1.0,
            disp_depth=0.025,
            disp_mode=DisplacementMode.POM,
            description="Decorative paved stone floor",
        )

        # 13. Metal Grate Rusty (Matches asset 'metal_grate_rusty')
        self.register_material(
            name="metal_grate_rusty",
            color=(0.50, 0.45, 0.40),
            roughness=0.55,
            metallic=0.80,
            ao=1.0,
            disp_depth=0.015,
            disp_mode=DisplacementMode.POM,
            description="Industrial rusty metal drainage grate",
        )

        # 14. Ribbed Corduroy (Matches asset 'ribbed_corduroy')
        self.register_material(
            name="ribbed_corduroy",
            color=(0.40, 0.35, 0.30),
            roughness=0.90,
            metallic=0.0,
            ao=1.0,
            disp_depth=0.008,
            disp_mode=DisplacementMode.POM,
            description="Textile corduroy ribbed fabric",
        )

    def register_material(
        self,
        name: str,
        layer_idx: int = 0,
        has_texture: bool = False,
        disp_mode: DisplacementMode | int = DisplacementMode.NONE,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        roughness: float = 0.5,
        metallic: float = 0.0,
        ao: float = 1.0,
        disp_depth: float = 0.030,
        friction: float = 0.80,
        restitution: float = 0.05,
        description: str = "",
    ) -> MaterialDef:
        """Registers or updates a material definition in the registry."""
        norm_name = name.strip().lower()
        d_mode = DisplacementMode(disp_mode)
        mat_id = self._next_id
        if norm_name in self._materials_by_name:
            mat = self._materials_by_name[norm_name]
            mat.layer_idx = layer_idx
            mat.has_texture = has_texture or (layer_idx > 0)
            mat.disp_mode = d_mode
            mat.color = color
            mat.roughness = roughness
            mat.metallic = metallic
            mat.ao = ao
            mat.disp_depth = disp_depth
            mat.friction = friction
            mat.restitution = restitution
            if description:
                mat.description = description
            return mat

        mat = MaterialDef(
            name=norm_name,
            material_id=mat_id,
            layer_idx=layer_idx,
            has_texture=has_texture or (layer_idx > 0),
            disp_mode=d_mode,
            color=color,
            roughness=roughness,
            metallic=metallic,
            ao=ao,
            disp_depth=disp_depth,
            friction=friction,
            restitution=restitution,
            description=description,
        )
        self._next_id += 1
        self._materials_by_name[norm_name] = mat
        self._materials_by_id[mat_id] = mat
        return mat

    def add_alias(self, alias: str, canonical_name: str) -> None:
        """Associates a convenient alias name with an existing registered material."""
        self._alias_to_name[alias.strip().lower()] = canonical_name.strip().lower()

    def get(self, key: int | str | None) -> MaterialDef | None:
        """Looks up a MaterialDef by numeric ID or string name (including aliases)."""
        if key is None:
            return None
        if isinstance(key, int):
            return self._materials_by_id.get(key)
        name = str(key).strip().lower()
        canonical = self._alias_to_name.get(name, name)
        return self._materials_by_name.get(canonical)

    def get_by_name(self, name: str) -> MaterialDef | None:
        return self.get(name)

    def get_by_id(self, mat_id: int) -> MaterialDef | None:
        return self._materials_by_id.get(mat_id)

    def get_layer_idx(self, key: int | str) -> int:
        """Returns the texture atlas layer index for the specified material (default 0)."""
        mat = self.get(key)
        return mat.layer_idx if mat is not None else 0

    def sync_with_atlas(self, atlas: TextureArrayAtlas) -> None:
        """Synchronizes atlas layer indices and displacement depths with registered materials."""
        for name, layer_idx in atlas.name_to_layer.items():
            norm_name = name.strip().lower()
            depth = (
                float(atlas.material_depths[layer_idx])
                if layer_idx < len(atlas.material_depths)
                else DEFAULT_MATERIAL_DEPTHS.get(norm_name, 0.030)
            )
            mat = self.get(norm_name)
            if mat is not None:
                mat.layer_idx = layer_idx
                mat.has_texture = (layer_idx > 0)
                mat.disp_depth = depth
                if depth > 0.0 and mat.disp_mode == DisplacementMode.NONE:
                    mat.disp_mode = DisplacementMode.POM
            else:
                self.register_material(
                    name=norm_name,
                    layer_idx=layer_idx,
                    has_texture=(layer_idx > 0),
                    disp_mode=DisplacementMode.POM if depth > 0.0 else DisplacementMode.NONE,
                    disp_depth=depth,
                    description=f"Auto-registered from texture atlas layer {layer_idx}",
                )

        # Wire aliases for canonical city materials if an atlas layer matched
        if "concrete_floor_worn_02" in atlas.name_to_layer:
            layer = atlas.name_to_layer["concrete_floor_worn_02"]
            c_mat = self.get("concrete_sidewalk")
            if c_mat and c_mat.layer_idx == 0:
                c_mat.layer_idx = layer
                c_mat.has_texture = True
                c_mat.disp_depth = float(atlas.material_depths[layer])

        if "castle_brick_02_red" in atlas.name_to_layer:
            layer = atlas.name_to_layer["castle_brick_02_red"]
            b_mat = self.get("brick_wall")
            if b_mat and b_mat.layer_idx == 0:
                b_mat.layer_idx = layer
                b_mat.has_texture = True
                b_mat.disp_depth = float(atlas.material_depths[layer])

        if "metal_grate_rusty" in atlas.name_to_layer:
            layer = atlas.name_to_layer["metal_grate_rusty"]
            m_mat = self.get("metal_guardrail")
            if m_mat and m_mat.layer_idx == 0:
                m_mat.layer_idx = layer
                m_mat.has_texture = True
                m_mat.disp_depth = float(atlas.material_depths[layer])

    def list_materials(self) -> list[MaterialDef]:
        """Returns all registered unique MaterialDefs."""
        return list(self._materials_by_name.values())
