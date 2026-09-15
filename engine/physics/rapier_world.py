"""Synchronous in-thread Rapier3D physics manager.

Releases the Python GIL during simulation stepping and batch-syncs
transforms into ECS contiguous memory tables.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

import pymordial_rapier
from engine.core.ecs import EntityManager


class PhysicsManager:
    """Manages the Rapier3D physics world and synchronizes state with the ECS."""

    __slots__ = ("world", "_tracked_entities", "_dense_indices", "_has_direct_sync")

    def __init__(
        self,
        gravity_x: float = 0.0,
        gravity_y: float = -9.81,
        gravity_z: float = 0.0,
    ) -> None:
        self.world = pymordial_rapier.PyRapierWorld(gravity_x, gravity_y, gravity_z)
        self._tracked_entities: list[int] = []
        self._dense_indices: list[int] = []
        self._has_direct_sync: bool = hasattr(self.world, "sync_transforms_direct")

    def create_body(
        self,
        entity_id: int,
        body_type: str = "dynamic",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        """Creates a rigid body for the given entity ID."""
        self.world.create_rigid_body(
            entity_id,
            body_type,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(rotation[0]),
            float(rotation[1]),
            float(rotation[2]),
            float(rotation[3]),
        )
        if entity_id not in self._tracked_entities:
            self._tracked_entities.append(entity_id)

    def set_transform(
        self,
        entity_id: int,
        position: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float, float] | np.ndarray = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        """Sets an entity's rigid body transform."""
        self.world.set_transform(
            entity_id,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(rotation[0]),
            float(rotation[1]),
            float(rotation[2]),
            float(rotation[3]),
        )

    def apply_impulse(
        self,
        entity_id: int,
        impulse: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Applies an instantaneous linear impulse to a dynamic rigid body."""
        self.world.apply_impulse(
            entity_id,
            float(impulse[0]),
            float(impulse[1]),
            float(impulse[2]),
        )

    def apply_torque_impulse(
        self,
        entity_id: int,
        torque: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Applies an instantaneous rotational torque impulse to a dynamic rigid body."""
        self.world.apply_torque_impulse(
            entity_id,
            float(torque[0]),
            float(torque[1]),
            float(torque[2]),
        )

    def set_linvel(
        self,
        entity_id: int,
        velocity: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Sets linear velocity for a rigid body."""
        self.world.set_linvel(
            entity_id,
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
        )

    def get_linvel(self, entity_id: int) -> tuple[float, float, float]:
        """Gets current linear velocity (vx, vy, vz)."""
        return self.world.get_linvel(entity_id)

    def set_angvel(
        self,
        entity_id: int,
        angvel: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Sets angular velocity (wx, wy, wz)."""
        self.world.set_angvel(
            entity_id,
            float(angvel[0]),
            float(angvel[1]),
            float(angvel[2]),
        )

    def get_angvel(self, entity_id: int) -> tuple[float, float, float]:
        """Gets current angular velocity (wx, wy, wz)."""
        return self.world.get_angvel(entity_id)

    def set_damping(self, entity_id: int, linear: float, angular: float) -> None:
        """Sets linear and angular damping for a rigid body."""
        self.world.set_damping(entity_id, float(linear), float(angular))

    def set_gravity(self, gx: float, gy: float, gz: float) -> None:
        """Updates the global gravity vector in real time."""
        self.world.set_gravity(float(gx), float(gy), float(gz))

    def get_gravity(self) -> tuple[float, float, float]:
        """Returns the current global gravity vector (gx, gy, gz)."""
        return self.world.get_gravity()

    def attach_box_collider(
        self,
        entity_id: int,
        half_x: float,
        half_y: float,
        half_z: float,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches an oriented box collider to an entity's rigid body."""
        try:
            self.world.attach_box_collider(
                entity_id,
                float(half_x),
                float(half_y),
                float(half_z),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_box_collider(
                entity_id,
                float(half_x),
                float(half_y),
                float(half_z),
            )

    def attach_sphere_collider(
        self,
        entity_id: int,
        radius: float,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a sphere collider to an entity's rigid body."""
        try:
            self.world.attach_sphere_collider(
                entity_id,
                float(radius),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_sphere_collider(
                entity_id,
                float(radius),
            )

    def attach_capsule_collider(
        self,
        entity_id: int,
        half_height: float,
        radius: float,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a capsule collider (Y-axis aligned) to an entity's rigid body."""
        try:
            self.world.attach_capsule_collider(
                entity_id,
                float(half_height),
                float(radius),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_capsule_collider(
                entity_id,
                float(half_height),
                float(radius),
            )

    def attach_cylinder_collider(
        self,
        entity_id: int,
        half_height: float,
        radius: float,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a cylinder collider (Y-axis aligned) to an entity's rigid body."""
        try:
            self.world.attach_cylinder_collider(
                entity_id,
                float(half_height),
                float(radius),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_cylinder_collider(entity_id, float(half_height), float(radius))

    def attach_cone_collider(
        self,
        entity_id: int,
        half_height: float,
        radius: float,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a cone collider (Y-axis aligned, apex pointing +Y) to an entity's rigid body."""
        try:
            self.world.attach_cone_collider(
                entity_id,
                float(half_height),
                float(radius),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_cone_collider(entity_id, float(half_height), float(radius))

    def attach_plane_collider(
        self,
        entity_id: int,
        half_x: float,
        half_z: float,
        thickness: float = 0.05,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a planar slab collider (horizontal in X-Z plane) to an entity's rigid body."""
        try:
            self.world.attach_plane_collider(
                entity_id,
                float(half_x),
                float(half_z),
                thickness=float(thickness),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_plane_collider(entity_id, float(half_x), float(half_z))

    def attach_round_box_collider(
        self,
        entity_id: int,
        half_x: float,
        half_y: float,
        half_z: float,
        border_radius: float = 0.05,
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Attaches a beveled rounded box collider to an entity's rigid body."""
        try:
            self.world.attach_round_box_collider(
                entity_id,
                float(half_x),
                float(half_y),
                float(half_z),
                float(border_radius),
                density=float(density) if density is not None else None,
                friction=float(friction) if friction is not None else None,
                restitution=float(restitution) if restitution is not None else None,
            )
        except TypeError:
            self.world.attach_round_box_collider(
                entity_id, float(half_x), float(half_y), float(half_z), float(border_radius)
            )

    def attach_primitive_collider(
        self,
        entity_id: int,
        shape_type: str,
        dimensions: tuple[float, ...] = (1.0, 1.0, 1.0),
        density: float | None = None,
        friction: float | None = None,
        restitution: float | None = None,
    ) -> None:
        """Unified primitive shape collider dispatcher.

        Args:
            entity_id: Target entity ID with rigid body.
            shape_type: "box"|"cube", "sphere", "capsule", "cylinder", "cone", "plane".
            dimensions: Shape-specific extents:
                - "box" / "cube": (half_x, half_y, half_z)
                - "sphere": (radius,)
                - "capsule": (half_height, radius)
                - "cylinder": (half_height, radius)
                - "cone": (half_height, radius)
                - "plane": (half_x, half_z, thickness)
        """
        shape = shape_type.lower()
        if shape in ("box", "cube"):
            hx = dimensions[0]
            hy = dimensions[1] if len(dimensions) > 1 else dimensions[0]
            hz = dimensions[2] if len(dimensions) > 2 else dimensions[0]
            self.attach_box_collider(entity_id, hx, hy, hz, density, friction, restitution)
        elif shape == "sphere":
            r = dimensions[0]
            self.attach_sphere_collider(entity_id, r, density, friction, restitution)
        elif shape == "capsule":
            hh = dimensions[0]
            r = dimensions[1] if len(dimensions) > 1 else dimensions[0] * 0.5
            self.attach_capsule_collider(entity_id, hh, r, density, friction, restitution)
        elif shape == "cylinder":
            hh = dimensions[0]
            r = dimensions[1] if len(dimensions) > 1 else dimensions[0] * 0.5
            self.attach_cylinder_collider(entity_id, hh, r, density, friction, restitution)
        elif shape == "cone":
            hh = dimensions[0]
            r = dimensions[1] if len(dimensions) > 1 else dimensions[0] * 0.5
            self.attach_cone_collider(entity_id, hh, r, density, friction, restitution)
        elif shape == "plane":
            hx = dimensions[0]
            hz = dimensions[1] if len(dimensions) > 1 else dimensions[0]
            th = dimensions[2] if len(dimensions) > 2 else 0.05
            self.attach_plane_collider(entity_id, hx, hz, th, density, friction, restitution)
        else:
            raise ValueError(f"Unsupported primitive shape type: {shape_type}")

    def get_mass(self, entity_id: int) -> float:
        """Returns the mass of an entity's rigid body in kilograms."""
        return float(self.world.get_mass(entity_id))

    def create_character_controller(
        self,
        entity_id: int,
        half_height: float = 0.5,
        radius: float = 0.4,
        max_slope_deg: float = 45.0,
        step_height: float = 0.3,
        snap_to_ground: float = 0.2,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Creates a kinematic character controller with capsule collider for an entity."""
        self.world.create_character_controller(
            entity_id,
            float(half_height),
            float(radius),
            float(max_slope_deg),
            float(step_height),
            float(snap_to_ground),
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        if entity_id not in self._tracked_entities:
            self._tracked_entities.append(entity_id)

    def move_character(
        self,
        entity_id: int,
        desired_translation: tuple[float, float, float] | np.ndarray,
        dt: float,
    ) -> tuple[tuple[float, float, float], bool, bool]:
        """Moves character via native sweep & slide.

        Returns:
            ((effective_dx, effective_dy, effective_dz), is_grounded, is_sliding)
        """
        dx, dy, dz, grounded, sliding = self.world.move_character(
            entity_id,
            float(desired_translation[0]),
            float(desired_translation[1]),
            float(desired_translation[2]),
            float(dt),
        )
        return (dx, dy, dz), grounded, sliding

    def remove_body(self, entity_id: int) -> bool:
        """Removes a rigid body and its colliders from simulation."""
        if entity_id in self._tracked_entities:
            self._tracked_entities.remove(entity_id)
        return self.world.remove_rigid_body(entity_id)

    def step_simulation(self, dt: float) -> None:
        """Steps the Rapier physics world in-thread while releasing the Python GIL."""
        self.world.step(dt)

    def sync_to_ecs(self, ecs: EntityManager) -> None:
        """Batch-synchronizes updated Rapier body transforms into ECS RigidBodyState[1]."""
        n = len(self._tracked_entities)
        if n == 0:
            return

        if len(self._dense_indices) != n:
            self._dense_indices = [0] * n

        pool = ecs.pool
        dense_indices = self._dense_indices
        tracked = self._tracked_entities
        for i in range(n):
            dense_indices[i] = pool.get_dense_index(tracked[i])

        if self._has_direct_sync:
            self.world.sync_transforms_direct(
                tracked,
                dense_indices,
                ecs.rigid_body_state[1],
            )
        else:
            sync_results = self.world.sync_transforms(self._tracked_entities)
            curr_physics = ecs.rigid_body_state[1]
            for i, (ent_id, px, py, pz, qx, qy, qz, qw) in enumerate(sync_results):
                dense_idx = dense_indices[i] if i < len(dense_indices) else pool.get_dense_index(ent_id)
                if dense_idx >= 0:
                    curr_physics[dense_idx, 0] = px
                    curr_physics[dense_idx, 1] = py
                    curr_physics[dense_idx, 2] = pz
                    curr_physics[dense_idx, 3] = qx
                    curr_physics[dense_idx, 4] = qy
                    curr_physics[dense_idx, 5] = qz
                    curr_physics[dense_idx, 6] = qw

    def cast_ray_raw(
        self,
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        max_distance: float = 1000.0,
        solid: bool = True,
    ) -> Optional[tuple[int, float, float, float, float]]:
        """Casts a ray using scalar coordinates and returns (hit_entity_id, dist, nx, ny, nz) or None.

        Zero-allocation routine avoiding tuple creation on ray invocation.
        """
        return self.world.cast_ray(
            float(ox),
            float(oy),
            float(oz),
            float(dx),
            float(dy),
            float(dz),
            float(max_distance),
            solid,
        )

    def raycast(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        max_distance: float = 1000.0,
        solid: bool = True,
    ) -> Optional[tuple[int, float, tuple[float, float, float]]]:
        """Casts a ray and returns (hit_entity_id, hit_distance, normal_xyz) or None."""
        hit = self.world.cast_ray(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
            float(max_distance),
            solid,
        )
        if hit is not None:
            ent_id, dist, nx, ny, nz = hit
            return ent_id, dist, (nx, ny, nz)
        return None
