"""Declarative Data Contracts and Serialization Schemas for the Mapping Subsystem.

Defines typed dataclasses for transforms, visual models, Rapier colliders,
trigger zones, scripts, compound entity hierarchies, sectors, and maps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

VALID_BODY_TYPES = frozenset({"fixed", "dynamic", "kinematic_position", "kinematic_velocity"})
VALID_PRIMITIVE_SHAPES = frozenset(
    {"box", "cube", "sphere", "capsule", "cylinder", "cone", "plane", "round_box", "water"}
)
VALID_STREAMING_MODES = frozenset({"monolithic", "linear_triggers", "spatial_grid"})


def _to_tuple_floats(val: Any, expected_len: int, name: str) -> tuple[float, ...]:
    """Helper to validate and cast sequence to a tuple of floats."""
    if not isinstance(val, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of {expected_len} floats, got {type(val).__name__}")
    if len(val) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(val)}")
    return tuple(float(x) for x in val)


@dataclass(slots=True)
class TransformDef:
    """Local or world spatial transform definition."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion [x, y, z, w]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _to_tuple_floats(self.position, 3, "position"))
        object.__setattr__(self, "rotation", _to_tuple_floats(self.rotation, 4, "rotation"))
        object.__setattr__(self, "scale", _to_tuple_floats(self.scale, 3, "scale"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformDef:
        pos = data.get("position", (0.0, 0.0, 0.0))
        rot = data.get("rotation", (0.0, 0.0, 0.0, 1.0))
        scl = data.get("scale", (1.0, 1.0, 1.0))
        return cls(position=pos, rotation=rot, scale=scl)


@dataclass(slots=True)
class VisualDef:
    """Visual rendering descriptor."""

    shape: str = "cube"
    material: str | None = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    ao: float = 1.0
    emissive_intensity: float = 0.0
    layer_idx: int = 0
    cast_shadow: bool = True
    is_visible: bool = True
    # Future Phase 2 upgrade fields
    mesh_path: str | None = None
    submesh_index: int = 0

    def __post_init__(self) -> None:
        shape_norm = self.shape.lower()
        if shape_norm not in VALID_PRIMITIVE_SHAPES and self.mesh_path is None:
            raise ValueError(f"Unknown visual shape '{self.shape}'. Expected one of {sorted(VALID_PRIMITIVE_SHAPES)}")
        object.__setattr__(self, "shape", shape_norm)
        object.__setattr__(self, "color", _to_tuple_floats(self.color, 3, "color"))
        object.__setattr__(self, "roughness", float(self.roughness))
        object.__setattr__(self, "metallic", float(self.metallic))
        object.__setattr__(self, "ao", float(self.ao))
        object.__setattr__(self, "emissive_intensity", float(self.emissive_intensity))
        object.__setattr__(self, "layer_idx", int(self.layer_idx))
        object.__setattr__(self, "cast_shadow", bool(self.cast_shadow))
        object.__setattr__(self, "is_visible", bool(self.is_visible))
        object.__setattr__(self, "submesh_index", int(self.submesh_index))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "shape": self.shape,
            "color": list(self.color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "ao": self.ao,
            "emissive_intensity": self.emissive_intensity,
            "layer_idx": self.layer_idx,
            "cast_shadow": self.cast_shadow,
            "is_visible": self.is_visible,
        }
        if self.material is not None:
            d["material"] = self.material
        if self.mesh_path is not None:
            d["mesh_path"] = self.mesh_path
            d["submesh_index"] = self.submesh_index
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualDef:
        return cls(
            shape=data.get("shape", "cube"),
            material=data.get("material"),
            color=data.get("color", (1.0, 1.0, 1.0)),
            roughness=data.get("roughness", 0.5),
            metallic=data.get("metallic", 0.0),
            ao=data.get("ao", 1.0),
            emissive_intensity=data.get("emissive_intensity", 0.0),
            layer_idx=data.get("layer_idx", 0),
            cast_shadow=data.get("cast_shadow", True),
            is_visible=data.get("is_visible", True),
            mesh_path=data.get("mesh_path"),
            submesh_index=data.get("submesh_index", 0),
        )


@dataclass(slots=True)
class CollisionDef:
    """Rapier3D physics collider descriptor."""

    enabled: bool = True
    body_type: str = "fixed"  # "fixed", "dynamic", "kinematic_position", "kinematic_velocity"
    shape: str = "box"  # "box", "cube", "sphere", "capsule", "cylinder", "cone", "plane", "round_box"
    dimensions: tuple[float, ...] = (0.5, 0.5, 0.5)  # Half-extents or (radius,) or (half_h, radius)
    friction: float = 0.8
    restitution: float = 0.0
    density: float | None = None
    is_sensor: bool = False

    def __post_init__(self) -> None:
        body_norm = self.body_type.lower()
        if body_norm not in VALID_BODY_TYPES:
            raise ValueError(f"Unknown body_type '{self.body_type}'. Expected one of {sorted(VALID_BODY_TYPES)}")
        object.__setattr__(self, "body_type", body_norm)

        shape_norm = self.shape.lower()
        if shape_norm not in VALID_PRIMITIVE_SHAPES:
            raise ValueError(f"Unknown collision shape '{self.shape}'. Expected one of {sorted(VALID_PRIMITIVE_SHAPES)}")
        object.__setattr__(self, "shape", shape_norm)

        if not isinstance(self.dimensions, (list, tuple)) or len(self.dimensions) == 0:
            raise ValueError(f"dimensions must be a non-empty sequence of positive floats, got {self.dimensions}")
        dim_floats = tuple(float(x) for x in self.dimensions)
        if any(x < 0.0 for x in dim_floats):
            raise ValueError(f"Collider dimensions must be non-negative: {dim_floats}")
        object.__setattr__(self, "dimensions", dim_floats)

        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "friction", float(self.friction))
        object.__setattr__(self, "restitution", float(self.restitution))
        if self.density is not None:
            object.__setattr__(self, "density", float(self.density))
        object.__setattr__(self, "is_sensor", bool(self.is_sensor))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "enabled": self.enabled,
            "body_type": self.body_type,
            "shape": self.shape,
            "dimensions": list(self.dimensions),
            "friction": self.friction,
            "restitution": self.restitution,
            "is_sensor": self.is_sensor,
        }
        if self.density is not None:
            d["density"] = self.density
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollisionDef:
        return cls(
            enabled=data.get("enabled", True),
            body_type=data.get("body_type", "fixed"),
            shape=data.get("shape", "box"),
            dimensions=data.get("dimensions", (0.5, 0.5, 0.5)),
            friction=data.get("friction", 0.8),
            restitution=data.get("restitution", 0.0),
            density=data.get("density"),
            is_sensor=data.get("is_sensor", False),
        )


class PhysicsRole(str, Enum):
    """High-level semantic physics role for map entities."""

    STATIC = "static"        # Immovable scenery, ground, walls (zero gravity, fixed body)
    KINEMATIC = "kinematic"  # Animated / scripted velocity body (elevators, gates; unstoppable)
    DYNAMIC = "dynamic"      # Interactive rigid body (crates, barrels, balls; reacts to gravity & impulses)
    SENSOR = "sensor"        # Trigger volume only (zero physical resistance, zero draw overhead)
    NONE = "none"            # No physics simulation (visual only)


@dataclass(slots=True)
class PhysicalMaterial:
    """Preset physical material properties for friction, restitution, and density."""

    name: str
    friction: float = 0.70
    restitution: float = 0.05
    density: float = 1000.0  # kg/m^3


PHYSICAL_MATERIALS: dict[str, PhysicalMaterial] = {
    "concrete": PhysicalMaterial(name="concrete", friction=0.85, restitution=0.02, density=2400.0),
    "stone": PhysicalMaterial(name="stone", friction=0.85, restitution=0.02, density=2400.0),
    "wood": PhysicalMaterial(name="wood", friction=0.65, restitution=0.15, density=650.0),
    "metal": PhysicalMaterial(name="metal", friction=0.50, restitution=0.20, density=7800.0),
    "rubber": PhysicalMaterial(name="rubber", friction=0.90, restitution=0.80, density=1100.0),
    "ice": PhysicalMaterial(name="ice", friction=0.05, restitution=0.02, density=920.0),
    "cardboard": PhysicalMaterial(name="cardboard", friction=0.55, restitution=0.05, density=200.0),
    "default": PhysicalMaterial(name="default", friction=0.70, restitution=0.05, density=1000.0),
}


def get_physical_material(name: str | None) -> PhysicalMaterial:
    """Resolves a physical material preset by name, or returns the default preset."""
    if name is None:
        return PHYSICAL_MATERIALS["default"]
    return PHYSICAL_MATERIALS.get(str(name).lower(), PHYSICAL_MATERIALS["default"])


@dataclass(slots=True)
class PhysicsDef:
    """High-level, designer-friendly physics descriptor for map entities."""

    role: PhysicsRole | str = PhysicsRole.STATIC
    material: str | None = None
    mass: float | None = None
    friction: float | None = None
    restitution: float | None = None
    density: float | None = None
    ccd: bool = False
    can_sleep: bool = True
    linear_damping: float = 0.05
    angular_damping: float = 0.1
    shape: str | None = None
    dimensions: tuple[float, ...] | None = None
    is_sensor: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            role_norm = self.role.lower()
            valid_roles = {r.value for r in PhysicsRole}
            if role_norm not in valid_roles:
                raise ValueError(f"Unknown physics role '{self.role}'. Expected one of {sorted(valid_roles)}")
            object.__setattr__(self, "role", PhysicsRole(role_norm))

        mat = get_physical_material(self.material)
        if self.friction is None:
            object.__setattr__(self, "friction", mat.friction)
        else:
            object.__setattr__(self, "friction", float(self.friction))

        if self.restitution is None:
            object.__setattr__(self, "restitution", mat.restitution)
        else:
            object.__setattr__(self, "restitution", float(self.restitution))

        if self.density is None:
            object.__setattr__(self, "density", mat.density)
        else:
            object.__setattr__(self, "density", float(self.density))

        if self.dimensions is not None:
            if not isinstance(self.dimensions, (list, tuple)) or len(self.dimensions) == 0:
                raise ValueError(f"dimensions must be a non-empty sequence of floats, got {self.dimensions}")
            dim_floats = tuple(float(x) for x in self.dimensions)
            if any(x < 0.0 for x in dim_floats):
                raise ValueError(f"Physics dimensions must be non-negative: {dim_floats}")
            object.__setattr__(self, "dimensions", dim_floats)

        if self.mass is not None:
            object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "ccd", bool(self.ccd))
        object.__setattr__(self, "can_sleep", bool(self.can_sleep))
        object.__setattr__(self, "linear_damping", float(self.linear_damping))
        object.__setattr__(self, "angular_damping", float(self.angular_damping))
        object.__setattr__(self, "is_sensor", bool(self.is_sensor or self.role == PhysicsRole.SENSOR))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, PhysicsRole) else str(self.role),
            "friction": self.friction,
            "restitution": self.restitution,
            "density": self.density,
            "ccd": self.ccd,
            "can_sleep": self.can_sleep,
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
        }
        if self.material is not None:
            d["material"] = self.material
        if self.mass is not None:
            d["mass"] = self.mass
        if self.shape is not None:
            d["shape"] = self.shape
        if self.dimensions is not None:
            d["dimensions"] = list(self.dimensions)
        if self.is_sensor:
            d["is_sensor"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysicsDef:
        return cls(
            role=data.get("role", PhysicsRole.STATIC),
            material=data.get("material"),
            mass=float(data["mass"]) if "mass" in data and data["mass"] is not None else None,
            friction=float(data["friction"]) if "friction" in data and data["friction"] is not None else None,
            restitution=float(data["restitution"]) if "restitution" in data and data["restitution"] is not None else None,
            density=float(data["density"]) if "density" in data and data["density"] is not None else None,
            ccd=bool(data.get("ccd", False)),
            can_sleep=bool(data.get("can_sleep", True)),
            linear_damping=float(data.get("linear_damping", 0.05)),
            angular_damping=float(data.get("angular_damping", 0.1)),
            shape=data.get("shape"),
            dimensions=tuple(float(x) for x in data["dimensions"]) if "dimensions" in data and data["dimensions"] is not None else None,
            is_sensor=bool(data.get("is_sensor", False)),
        )


@dataclass(slots=True)
class TriggerActionDef:
    """Action dispatched when an observer interacts with a trigger zone."""

    action_type: str  # e.g., "STREAM_SECTOR", "PLAY_SOUND", "CLOSE_GATE", "CUSTOM_EVENT"
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action_type": self.action_type}
        if self.target is not None:
            d["target"] = self.target
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> TriggerActionDef:
        if isinstance(data, str):
            return cls(action_type=data)
        action = data.get("action_type") or data.get("action")
        if not action:
            raise ValueError("TriggerActionDef requires 'action_type' or 'action' field.")
        return cls(
            action_type=str(action),
            target=data.get("target"),
            params=dict(data.get("params", {})),
        )


@dataclass(slots=True)
class TriggerDef:
    """Spatial sensor/trigger volume definition."""

    shape: str = "box"  # "box", "sphere"
    dimensions: tuple[float, ...] = (1.0, 1.0, 1.0)  # half_extents or (radius,)
    filter_tag: str | None = "player"
    one_shot: bool = False
    on_enter: list[TriggerActionDef] = field(default_factory=list)
    on_exit: list[TriggerActionDef] = field(default_factory=list)

    def __post_init__(self) -> None:
        shape_norm = self.shape.lower()
        if shape_norm not in ("box", "cube", "sphere"):
            raise ValueError(f"Trigger shape must be 'box' or 'sphere', got '{self.shape}'")
        object.__setattr__(self, "shape", shape_norm)
        if not isinstance(self.dimensions, (list, tuple)) or len(self.dimensions) == 0:
            raise ValueError(f"Trigger dimensions must be a non-empty sequence: {self.dimensions}")
        object.__setattr__(self, "dimensions", tuple(float(x) for x in self.dimensions))
        object.__setattr__(self, "one_shot", bool(self.one_shot))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dimensions": list(self.dimensions),
            "filter_tag": self.filter_tag,
            "one_shot": self.one_shot,
            "on_enter": [a.to_dict() for a in self.on_enter],
            "on_exit": [a.to_dict() for a in self.on_exit],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriggerDef:
        def parse_actions(raw: Any) -> list[TriggerActionDef]:
            if raw is None:
                return []
            if isinstance(raw, (list, tuple)):
                return [TriggerActionDef.from_dict(item) for item in raw]
            return [TriggerActionDef.from_dict(raw)]

        return cls(
            shape=data.get("shape", "box"),
            dimensions=data.get("dimensions", (1.0, 1.0, 1.0)),
            filter_tag=data.get("filter_tag", "player"),
            one_shot=data.get("one_shot", False),
            on_enter=parse_actions(data.get("on_enter")),
            on_exit=parse_actions(data.get("on_exit")),
        )


@dataclass(slots=True)
class ScriptDef:
    """Specification of an attached behavioral script."""

    script_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"script_type": self.script_type}
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> ScriptDef:
        if isinstance(data, str):
            return cls(script_type=data)
        stype = data.get("script_type") or data.get("type")
        if not stype:
            raise ValueError("ScriptDef requires 'script_type' or 'type' field.")
        return cls(script_type=str(stype), params=dict(data.get("params", {})))


@dataclass(slots=True)
class EntityDef:
    """Atomic or Compound Map Entity Definition.

    Can define an individual visual/physics object, a trigger zone, or a compound
    parent grouping multiple child entities (e.g. doors, levers, building blocks).
    """

    id: str
    transform: TransformDef = field(default_factory=TransformDef)
    visual: VisualDef | None = None
    collision: CollisionDef | None = None
    physics: PhysicsDef | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    scripts: list[ScriptDef] = field(default_factory=list)
    trigger: TriggerDef | None = None
    prefab: str | None = None
    property_overrides: dict[str, Any] = field(default_factory=dict)
    children: list[EntityDef] = field(default_factory=list)

    @property
    def is_compound(self) -> bool:
        """True if this entity aggregates child entities or references a prefab."""
        return bool(self.children or self.prefab)

    def get_property(self, key: str, default: Any = None) -> Any:
        """Retrieves a metadata property from properties, falling back to default."""
        return self.properties.get(key, default)

    def resolved_collision(self) -> CollisionDef | None:
        """Returns the declared collision, or auto-derives a matching collider if collision is omitted."""
        if self.collision is not None:
            return self.collision

        # If high-level physics is declared, synthesize a CollisionDef from it
        if self.physics is not None:
            role = self.physics.role
            if role in (PhysicsRole.NONE, PhysicsRole.SENSOR):
                return None

            # Map role to Rapier body_type
            if role == PhysicsRole.DYNAMIC:
                body_type = "dynamic"
            elif role == PhysicsRole.KINEMATIC:
                body_type = "kinematic_velocity"
            else:
                body_type = "fixed"

            # Derive shape
            raw_shape = self.physics.shape
            if not raw_shape and self.visual is not None:
                raw_shape = self.visual.shape
            shape = "box" if (not raw_shape or raw_shape in ("box", "cube")) else raw_shape.lower()

            # Derive dimensions if not explicitly set
            if self.physics.dimensions is not None:
                dim = self.physics.dimensions
            elif self.visual is not None and self.visual.shape in VALID_PRIMITIVE_SHAPES:
                sx, sy, sz = self.transform.scale
                if shape in ("box", "cube"):
                    dim = (sx * 0.5, sy * 0.5, sz * 0.5)
                elif shape == "sphere":
                    dim = (0.5 * max(sx, sy, sz),)
                elif shape == "cylinder":
                    dim = (sy * 0.5, sx * 1.0)
                elif shape == "capsule":
                    dim = (sy * 0.5, sx * 0.5)
                elif shape == "cone":
                    dim = (sy * 0.5, sx * 0.5)
                elif shape == "plane":
                    dim = (sx * 0.5, sz * 0.5, 0.05)
                else:
                    dim = (sx * 0.5, sy * 0.5, sz * 0.5)
            else:
                dim = (0.5, 0.5, 0.5)

            return CollisionDef(
                enabled=True,
                body_type=body_type,
                shape=shape,
                dimensions=dim,
                friction=self.physics.friction if self.physics.friction is not None else 0.70,
                restitution=self.physics.restitution if self.physics.restitution is not None else 0.05,
                density=self.physics.density,
                is_sensor=self.physics.is_sensor or (role == PhysicsRole.SENSOR),
            )

        # Fallback: if visual exists without explicit collision/physics, auto-derive default fixed collider matching visual
        if self.visual is not None and self.visual.shape in VALID_PRIMITIVE_SHAPES:
            sx, sy, sz = self.transform.scale
            v_shape = self.visual.shape.lower()
            if v_shape in ("box", "cube"):
                dim = (sx * 0.5, sy * 0.5, sz * 0.5)
            elif v_shape == "sphere":
                dim = (0.5 * max(sx, sy, sz),)
            elif v_shape == "cylinder":
                dim = (sy * 0.5, sx * 1.0)
            elif v_shape == "capsule":
                dim = (sy * 0.5, sx * 0.5)
            elif v_shape == "cone":
                dim = (sy * 0.5, sx * 0.5)
            elif v_shape == "plane":
                dim = (sx * 0.5, sz * 0.5, 0.05)
            else:
                dim = (sx * 0.5, sy * 0.5, sz * 0.5)

            return CollisionDef(
                enabled=True,
                body_type="fixed",
                shape="box" if v_shape in ("box", "cube") else v_shape,
                dimensions=dim,
            )
        return None

    def resolved_physics(self) -> PhysicsDef | None:
        """Returns the declared PhysicsDef, or synthesizes one from CollisionDef or VisualDef."""
        if self.physics is not None:
            return self.physics
        if self.collision is not None:
            if not self.collision.enabled:
                role = PhysicsRole.NONE
            elif self.collision.is_sensor:
                role = PhysicsRole.SENSOR
            elif self.collision.body_type == "dynamic":
                role = PhysicsRole.DYNAMIC
            elif self.collision.body_type in ("kinematic_velocity", "kinematic_position"):
                role = PhysicsRole.KINEMATIC
            else:
                role = PhysicsRole.STATIC
            return PhysicsDef(
                role=role,
                friction=self.collision.friction,
                restitution=self.collision.restitution,
                density=self.collision.density,
                shape=self.collision.shape,
                dimensions=self.collision.dimensions,
                is_sensor=self.collision.is_sensor,
            )
        if self.visual is not None and self.visual.shape in VALID_PRIMITIVE_SHAPES:
            return PhysicsDef(role=PhysicsRole.STATIC, shape=self.visual.shape)
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "transform": self.transform.to_dict(),
        }
        if self.visual is not None:
            d["visual"] = self.visual.to_dict()
        if self.collision is not None:
            d["collision"] = self.collision.to_dict()
        if self.physics is not None:
            d["physics"] = self.physics.to_dict()
        if self.properties:
            d["properties"] = self.properties
        if self.scripts:
            d["scripts"] = [s.to_dict() for s in self.scripts]
        if self.trigger is not None:
            d["trigger"] = self.trigger.to_dict()
        if self.prefab is not None:
            d["prefab"] = self.prefab
        if self.property_overrides:
            d["property_overrides"] = self.property_overrides
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityDef:
        ent_id = str(data.get("id", "entity"))

        # Transform
        tf_data = data.get("transform")
        transform = TransformDef.from_dict(tf_data) if isinstance(tf_data, dict) else TransformDef()

        # Visual
        vis_data = data.get("visual")
        visual = VisualDef.from_dict(vis_data) if isinstance(vis_data, dict) else None

        # Collision
        col_data = data.get("collision")
        collision = CollisionDef.from_dict(col_data) if isinstance(col_data, dict) else None

        # Physics (High-level)
        phys_data = data.get("physics")
        physics = PhysicsDef.from_dict(phys_data) if isinstance(phys_data, dict) else None

        # Properties
        properties = dict(data.get("properties", {}))

        # Scripts
        raw_scripts = data.get("scripts", [])
        scripts = [ScriptDef.from_dict(s) for s in raw_scripts]

        # Trigger
        trig_data = data.get("trigger")
        trigger = TriggerDef.from_dict(trig_data) if isinstance(trig_data, dict) else None

        # Prefab & overrides
        prefab = data.get("prefab")
        property_overrides = dict(data.get("property_overrides", {}))

        # Children
        raw_children = data.get("children", [])
        children = [EntityDef.from_dict(c) for c in raw_children if isinstance(c, dict)]

        return cls(
            id=ent_id,
            transform=transform,
            visual=visual,
            collision=collision,
            properties=properties,
            scripts=scripts,
            trigger=trigger,
            prefab=prefab,
            property_overrides=property_overrides,
            children=children,
        )


@dataclass(slots=True)
class SectorBoundsDef:
    """Axis-Aligned Bounding Box (AABB) defining the spatial bounds of a sector."""

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents: tuple[float, float, float] = (50.0, 50.0, 50.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _to_tuple_floats(self.center, 3, "bounds.center"))
        object.__setattr__(self, "half_extents", _to_tuple_floats(self.half_extents, 3, "bounds.half_extents"))

    def contains_point(self, point: tuple[float, float, float]) -> bool:
        """Evaluates whether a 3D point is inside this bounding box."""
        px, py, pz = point
        cx, cy, cz = self.center
        hx, hy, hz = self.half_extents
        return (
            (cx - hx) <= px <= (cx + hx)
            and (cy - hy) <= py <= (cy + hy)
            and (cz - hz) <= pz <= (cz + hz)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "half_extents": list(self.half_extents),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectorBoundsDef:
        return cls(
            center=data.get("center", (0.0, 0.0, 0.0)),
            half_extents=data.get("half_extents", (50.0, 50.0, 50.0)),
        )


@dataclass(slots=True)
class SectorDef:
    """Isolated spatial sector/chunk of a map."""

    name: str
    bounds: SectorBoundsDef | None = None
    entities: list[EntityDef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "entities": [e.to_dict() for e in self.entities],
        }
        if self.bounds is not None:
            d["bounds"] = self.bounds.to_dict()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectorDef:
        name = str(data.get("name", "sector"))
        bounds_data = data.get("bounds")
        bounds = SectorBoundsDef.from_dict(bounds_data) if isinstance(bounds_data, dict) else None
        entities_data = data.get("entities", [])
        entities = [EntityDef.from_dict(e) for e in entities_data if isinstance(e, dict)]
        metadata = dict(data.get("metadata", {}))
        return cls(name=name, bounds=bounds, entities=entities, metadata=metadata)


@dataclass(slots=True)
class MapDef:
    """Top-level map container (linear level, multi-map, or open world)."""

    name: str
    sectors: list[str | SectorDef] = field(default_factory=list)
    initial_sector: str | None = None
    streaming_mode: str = "monolithic"  # "monolithic", "linear_triggers", "spatial_grid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode_norm = self.streaming_mode.lower()
        if mode_norm not in VALID_STREAMING_MODES:
            raise ValueError(f"Unknown streaming_mode '{self.streaming_mode}'. Expected one of {sorted(VALID_STREAMING_MODES)}")
        object.__setattr__(self, "streaming_mode", mode_norm)

    def to_dict(self) -> dict[str, Any]:
        serialized_sectors: list[Any] = []
        for s in self.sectors:
            if isinstance(s, SectorDef):
                serialized_sectors.append(s.to_dict())
            else:
                serialized_sectors.append(str(s))
        d: dict[str, Any] = {
            "name": self.name,
            "sectors": serialized_sectors,
            "streaming_mode": self.streaming_mode,
        }
        if self.initial_sector is not None:
            d["initial_sector"] = self.initial_sector
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapDef:
        name = str(data.get("name", "map"))
        raw_sectors = data.get("sectors", [])
        sectors: list[str | SectorDef] = []
        for item in raw_sectors:
            if isinstance(item, dict):
                sectors.append(SectorDef.from_dict(item))
            elif isinstance(item, str):
                sectors.append(item)
        initial_sector = data.get("initial_sector")
        streaming_mode = str(data.get("streaming_mode", "monolithic"))
        metadata = dict(data.get("metadata", {}))
        return cls(
            name=name,
            sectors=sectors,
            initial_sector=initial_sector,
            streaming_mode=streaming_mode,
            metadata=metadata,
        )


# ==============================================================================
# File I/O Helpers
# ==============================================================================


def save_sector_file(sector: SectorDef, file_path: str | Path, indent: int = 2) -> None:
    """Serializes a SectorDef to a UTF-8 JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sector.to_dict(), f, indent=indent)


def load_sector_file(file_path: str | Path) -> SectorDef:
    """Loads and validates a SectorDef from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sector file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Root of sector JSON must be an object, got {type(data).__name__}")
    return SectorDef.from_dict(data)


def save_map_file(map_def: MapDef, file_path: str | Path, indent: int = 2) -> None:
    """Serializes a MapDef to a UTF-8 JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(map_def.to_dict(), f, indent=indent)


def load_map_file(file_path: str | Path) -> MapDef:
    """Loads and validates a MapDef from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Map file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Root of map JSON must be an object, got {type(data).__name__}")
    return MapDef.from_dict(data)
