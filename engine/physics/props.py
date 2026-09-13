"""Interactive Physics Prop & Destructibles Manager.

Coordinates dynamic rigid body props (crates, tumbling blocks, spheres, barrels),
impact damage thresholds, shattering into rigid debris fragments with radial explosions,
character velocity impulse transfer (shoving/kicking), and MDI batch rendering synchronization.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import TYPE_CHECKING
import numpy as np

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager

if TYPE_CHECKING:
    from engine.audio.audio_engine import AudioEngine
    from engine.gfx.mega_buffer import MeshAllocation


class PropType(str, Enum):
    """Semantic category of physics prop."""
    CRATE = "crate"
    SPHERE = "sphere"
    BARREL = "barrel"
    DEBRIS = "debris"


@dataclass(slots=True)
class PropInstance:
    """Tracks state and lifecycle of an active dynamic prop."""
    entity_id: int
    prop_type: PropType
    shape_name: str           # "cube", "sphere", "capsule"
    scale: tuple[float, float, float]
    is_destructible: bool = True
    health: float = 40.0
    max_health: float = 40.0
    lifetime: float | None = None  # None for permanent props, seconds for debris
    is_grabbed: bool = False


class InteractivePropManager:
    """Manages the full lifecycle of interactive physics props, debris, and rendering batches."""

    __slots__ = (
        "physics",
        "ecs",
        "audio",
        "_props",
        "_debris_entities",
        "last_impact_info",
    )

    def __init__(
        self,
        physics: PhysicsManager,
        ecs: EntityManager,
        audio: AudioEngine | None = None,
    ) -> None:
        self.physics = physics
        self.ecs = ecs
        self.audio = audio
        self._props: dict[int, PropInstance] = {}
        self._debris_entities: list[int] = []
        self.last_impact_info: str = "None"

    @property
    def active_props_count(self) -> int:
        """Returns the number of currently active dynamic props and debris."""
        return len(self._props)

    def is_prop(self, entity_id: int) -> bool:
        """Checks if a given entity ID belongs to a managed dynamic prop."""
        return entity_id in self._props

    def get_prop(self, entity_id: int) -> PropInstance | None:
        """Retrieves a prop instance by its entity ID."""
        return self._props.get(entity_id, None)

    # ---------------- Spawning Methods ----------------

    def spawn_crate(
        self,
        position: tuple[float, float, float],
        size: float = 1.0,
        color: tuple[float, float, float] = (0.76, 0.54, 0.32),
        is_destructible: bool = True,
        health: float = 45.0,
        roughness: float = 0.85,
        metallic: float = 0.05,
        density: float = 75.0,
    ) -> int:
        """Spawns a dynamic physical crate with realistic wood density, friction, and resting damping."""
        half = size * 0.5
        ent_id = self.ecs.create_entity(
            position=position,
            scale=(size, size, size),
            color=color,
            roughness=roughness,
            metallic=metallic,
        )

        self.physics.create_body(ent_id, body_type="dynamic", position=position)
        self.physics.attach_box_collider(ent_id, half, half, half, density=density, friction=0.70, restitution=0.05)
        self.physics.set_damping(ent_id, linear=1.2, angular=1.6)

        prop = PropInstance(
            entity_id=ent_id,
            prop_type=PropType.CRATE,
            shape_name="cube",
            scale=(size, size, size),
            is_destructible=is_destructible,
            health=health,
            max_health=health,
        )
        self._props[ent_id] = prop
        return ent_id

    def spawn_sphere(
        self,
        position: tuple[float, float, float],
        radius: float = 0.5,
        color: tuple[float, float, float] = (0.92, 0.28, 0.24),
        is_destructible: bool = False,
        health: float = 100.0,
        roughness: float = 0.25,
        metallic: float = 0.30,
        density: float = 120.0,
    ) -> int:
        """Spawns a dynamic bouncy sphere with unit-mesh scale aligned to collider radius."""
        # Note: unit sphere mesh has radius 1.0; scale (radius, radius, radius) produces exact collider radius
        ent_id = self.ecs.create_entity(
            position=position,
            scale=(radius, radius, radius),
            color=color,
            roughness=roughness,
            metallic=metallic,
        )

        self.physics.create_body(ent_id, body_type="dynamic", position=position)
        self.physics.attach_sphere_collider(ent_id, radius, density=density, friction=0.55, restitution=0.50)
        self.physics.set_damping(ent_id, linear=0.35, angular=0.45)

        prop = PropInstance(
            entity_id=ent_id,
            prop_type=PropType.SPHERE,
            shape_name="sphere",
            scale=(radius, radius, radius),
            is_destructible=is_destructible,
            health=health,
            max_health=health,
        )
        self._props[ent_id] = prop
        return ent_id

    def spawn_barrel(
        self,
        position: tuple[float, float, float],
        half_height: float = 0.5,
        radius: float = 0.35,
        color: tuple[float, float, float] = (0.28, 0.45, 0.72),
        is_destructible: bool = True,
        health: float = 60.0,
        density: float = 95.0,
    ) -> int:
        """Spawns a dynamic tumbling barrel with capsule mesh scaled to match collider bounds."""
        # Note: unit capsule mesh has radius 0.4 and half_height 0.5 (total height 1.8)
        scale_x = radius / 0.4
        scale_y = (half_height + radius) / 0.9
        scale_z = radius / 0.4
        ent_id = self.ecs.create_entity(
            position=position,
            scale=(scale_x, scale_y, scale_z),
            color=color,
            roughness=0.4,
            metallic=0.7,
        )

        self.physics.create_body(ent_id, body_type="dynamic", position=position)
        self.physics.attach_capsule_collider(ent_id, half_height, radius, density=density, friction=0.65, restitution=0.15)
        self.physics.set_damping(ent_id, linear=1.0, angular=1.2)

        prop = PropInstance(
            entity_id=ent_id,
            prop_type=PropType.BARREL,
            shape_name="capsule",
            scale=(scale_x, scale_y, scale_z),
            is_destructible=is_destructible,
            health=health,
            max_health=health,
        )
        self._props[ent_id] = prop
        return ent_id

    def spawn_pyramid(
        self,
        base_center: tuple[float, float, float] = (0.0, 0.5, 5.0),
        rows: int = 3,
        box_size: float = 0.8,
        color: tuple[float, float, float] = (0.75, 0.52, 0.30),
    ) -> list[int]:
        """Spawns a structured pyramid stack of tumbling crates."""
        spawned_ids: list[int] = []
        bx, by, bz = base_center

        for r in range(rows):
            count = rows - r
            row_y = by + r * (box_size + 0.02)
            start_x = bx - (count - 1) * box_size * 0.5
            for c in range(count):
                x = start_x + c * (box_size + 0.02)
                pos = (x, row_y, bz)
                crate_id = self.spawn_crate(
                    position=pos,
                    size=box_size,
                    color=color,
                    is_destructible=True,
                    health=35.0,
                )
                spawned_ids.append(crate_id)

        return spawned_ids

    # ---------------- Destruction & Debris ----------------

    def damage_prop(
        self,
        entity_id: int,
        damage: float,
        hit_point: tuple[float, float, float] | None = None,
        hit_direction: tuple[float, float, float] | None = None,
    ) -> bool:
        """Applies damage to a destructible prop.

        If health drops to 0, shatters the prop into mini debris fragments
        with explosive radial impulses.
        Returns True if the prop shattered/destroyed.
        """
        prop = self._props.get(entity_id)
        if prop is None or not prop.is_destructible:
            return False

        prop.health -= damage
        self.last_impact_info = f"Entity {entity_id} Took {damage:.1f} DMG ({prop.health:.1f}/{prop.max_health:.1f})"

        if prop.health <= 0.0:
            self.shatter_prop(entity_id, hit_point=hit_point, hit_direction=hit_direction)
            return True

        return False

    def shatter_prop(
        self,
        entity_id: int,
        hit_point: tuple[float, float, float] | None = None,
        hit_direction: tuple[float, float, float] | None = None,
        debris_count: int = 6,
    ) -> None:
        """Shatters a prop into dynamic rigid debris pieces with radial impulses."""
        prop = self._props.get(entity_id)
        if prop is None:
            return

        # Query position before removal
        try:
            trans = self.physics.world.get_transform(entity_id)
            center = (trans[0], trans[1], trans[2])
        except Exception:
            center = (0.0, 1.0, 0.0)

        origin = hit_point if hit_point is not None else center

        # Play 3D wood break sound
        if self.audio is not None:
            self.audio.play_sound("wood_break", position=origin, volume=0.85)

        # Remove original prop
        self.destroy_prop(entity_id)

        # Spawn mini debris fragments
        frag_size = max(0.18, min(prop.scale[0], prop.scale[1]) * 0.40)
        frag_half = frag_size * 0.5
        hdir = hit_direction if hit_direction is not None else (0.0, 0.0, 1.0)

        for _ in range(debris_count):
            ox = origin[0] + random.uniform(-frag_half, frag_half)
            oy = origin[1] + random.uniform(0.0, frag_half * 2.0)
            oz = origin[2] + random.uniform(-frag_half, frag_half)

            debris_id = self.ecs.create_entity(
                position=(ox, oy, oz),
                scale=(frag_size, frag_size, frag_size),
                color=(0.58, 0.40, 0.22),
                roughness=0.90,
                metallic=0.0,
            )

            self.physics.create_body(debris_id, body_type="dynamic", position=(ox, oy, oz))
            self.physics.attach_box_collider(debris_id, frag_half, frag_half, frag_half, density=65.0, friction=0.75, restitution=0.15)
            self.physics.set_damping(debris_id, linear=1.5, angular=2.0)

            # Radial explosion impulse biased along hit direction
            imp_x = (random.uniform(-1.0, 1.0) + hdir[0] * 0.8) * 3.5
            imp_y = random.uniform(2.5, 6.0)
            imp_z = (random.uniform(-1.0, 1.0) + hdir[2] * 0.8) * 3.5
            self.physics.apply_impulse(debris_id, (imp_x, imp_y, imp_z))

            # Random spin torque
            t_x = random.uniform(-1.5, 1.5)
            t_y = random.uniform(-1.5, 1.5)
            t_z = random.uniform(-1.5, 1.5)
            self.physics.apply_torque_impulse(debris_id, (t_x, t_y, t_z))

            frag = PropInstance(
                entity_id=debris_id,
                prop_type=PropType.DEBRIS,
                shape_name="cube",
                scale=(frag_size, frag_size, frag_size),
                is_destructible=False,
                health=10.0,
                max_health=10.0,
                lifetime=random.uniform(5.0, 7.5),
            )
            self._props[debris_id] = frag
            self._debris_entities.append(debris_id)

        self.last_impact_info = f"Shattered Entity {entity_id} -> {debris_count} fragments"

    def destroy_prop(self, entity_id: int) -> bool:
        """Removes a prop cleanly from physics simulation and ECS."""
        if entity_id in self._props:
            del self._props[entity_id]
        if entity_id in self._debris_entities:
            self._debris_entities.remove(entity_id)

        self.physics.remove_body(entity_id)
        return self.ecs.destroy_entity(entity_id)

    def clear_all_props(self) -> None:
        """Removes all managed dynamic props and debris."""
        ids = list(self._props.keys())
        for ent_id in ids:
            self.destroy_prop(ent_id)
        self._props.clear()
        self._debris_entities.clear()
        self.last_impact_info = "Cleared All Dynamic Props"

    # ---------------- Frame Lifecycle & Character Interaction ----------------

    def update(self, dt: float) -> None:
        """Steps prop lifecycles (debris countdowns and cleanup)."""
        if not self._debris_entities:
            return

        to_remove: list[int] = []
        for ent_id in self._debris_entities:
            prop = self._props.get(ent_id)
            if prop is None or prop.lifetime is None:
                to_remove.append(ent_id)
                continue
            prop.lifetime -= dt
            if prop.lifetime <= 0.0:
                to_remove.append(ent_id)

        for ent_id in to_remove:
            self.destroy_prop(ent_id)

    def push_props_near_character(
        self,
        character_pos: tuple[float, float, float] | np.ndarray,
        character_velocity: tuple[float, float, float] | np.ndarray,
        push_radius: float | None = None,
        push_force: float = 4.0,
        character_radius: float = 0.4,
        character_half_height: float = 0.9,
        contact_tolerance: float = 0.04,
    ) -> None:
        """Transfers character horizontal momentum to physically contacting dynamic rigid bodies.

        Respects physical touching boundaries (no air-gap phantom collisions),
        limits pushed velocity to character movement speed (no objects flying off like rockets),
        and scales impulses by realistic object mass and inertia.
        """
        vx = float(character_velocity[0])
        vz = float(character_velocity[2])
        speed_sq = vx * vx + vz * vz
        if speed_sq < 0.04:
            return

        cx, cy, cz = float(character_pos[0]), float(character_pos[1]), float(character_pos[2])

        for ent_id, prop in self._props.items():
            if prop.is_grabbed:
                continue

            try:
                t = self.physics.world.get_transform(ent_id)
                px, py, pz = t[0], t[1], t[2]
            except Exception:
                continue

            # Vertical overlap check: character [cy - half_h, cy + half_h] vs prop [py - half_h, py + half_h]
            prop_half_y = prop.scale[1] * 0.5
            if (py + prop_half_y < cy - character_half_height) or (py - prop_half_y > cy + character_half_height):
                continue

            # Horizontal distance between character cylinder axis and prop center
            dx = px - cx
            dz = pz - cz
            dist_sq = dx * dx + dz * dz

            # Effective horizontal radius of the prop
            prop_rad = max(prop.scale[0], prop.scale[2]) * 0.5
            touch_dist = character_radius + prop_rad

            # If push_radius is explicitly specified, use it as the maximum boundary;
            # otherwise strictly require contact within contact_tolerance (e.g. 4 cm).
            max_contact_dist = push_radius if push_radius is not None else (touch_dist + contact_tolerance)
            if dist_sq > max_contact_dist * max_contact_dist:
                continue

            dist = math.sqrt(dist_sq) if dist_sq > 1e-4 else 1.0
            nx = dx / dist
            nz = dz / dist

            # Component of character velocity directed along the push normal towards the prop
            v_char_normal = vx * nx + vz * nz
            if v_char_normal <= 0.05:  # Moving away or glancing/perpendicular
                continue

            # Current prop velocity along the push normal
            try:
                pvx, _pvy, pvz = self.physics.get_linvel(ent_id)
                v_prop_normal = pvx * nx + pvz * nz
            except Exception:
                v_prop_normal = 0.0

            # If the prop is already travelling away as fast as (or faster than) the character,
            # no further push force can be imparted.
            if v_prop_normal >= v_char_normal:
                continue

            # Deficit between player speed and prop speed along the push normal
            vel_deficit = v_char_normal - v_prop_normal

            # Push factor: scales with push_force (default 4.0 -> 0.40, sprint 7.0 -> 0.70)
            # Clamped so the prop accelerates smoothly towards character speed without overshooting
            push_ratio = min(0.85, max(0.15, push_force * 0.10))
            dv = vel_deficit * push_ratio

            # Query real mass from physics engine (or fallback based on scale)
            try:
                mass = self.physics.get_mass(ent_id)
            except Exception:
                mass = 35.0

            impulse_magnitude = mass * dv

            # Apply pure horizontal impulse along the contact normal (no upward launch)
            imp_x = nx * impulse_magnitude
            imp_z = nz * impulse_magnitude
            self.physics.apply_impulse(ent_id, (imp_x, 0.0, imp_z))

            if self.audio is not None and impulse_magnitude > 25.0:
                self.audio.play_sound("impact", position=(px, py, pz), volume=min(0.85, impulse_magnitude * 0.015))

    # ---------------- MDI Draw Batch Synchronization ----------------

    def get_draw_batches(
        self,
        alloc_cube: MeshAllocation,
        alloc_sphere: MeshAllocation | None = None,
        alloc_capsule: MeshAllocation | None = None,
    ) -> list[tuple[MeshAllocation, int, int, bool]]:
        """Generates compact MDI draw batch commands for all active props."""
        batches: list[tuple[MeshAllocation, int, int, bool]] = []
        pool = self.ecs.pool

        for ent_id, prop in self._props.items():
            d_idx = pool.get_dense_index(ent_id)
            if d_idx < 0:
                continue

            alloc: MeshAllocation
            if prop.shape_name == "sphere" and alloc_sphere is not None:
                alloc = alloc_sphere
            elif prop.shape_name == "capsule" and alloc_capsule is not None:
                alloc = alloc_capsule
            else:
                alloc = alloc_cube

            batches.append((alloc, 1, d_idx, False))

        return batches
