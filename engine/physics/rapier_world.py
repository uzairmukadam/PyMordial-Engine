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

    __slots__ = ("world", "_tracked_entities")

    def __init__(
        self,
        gravity_x: float = 0.0,
        gravity_y: float = -9.81,
        gravity_z: float = 0.0,
    ) -> None:
        self.world = pymordial_rapier.PyRapierWorld(gravity_x, gravity_y, gravity_z)
        self._tracked_entities: list[int] = []

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

    def attach_box_collider(
        self,
        entity_id: int,
        half_x: float,
        half_y: float,
        half_z: float,
    ) -> None:
        """Attaches an oriented box collider to an entity's rigid body."""
        self.world.attach_box_collider(entity_id, half_x, half_y, half_z)

    def attach_sphere_collider(self, entity_id: int, radius: float) -> None:
        """Attaches a sphere collider to an entity's rigid body."""
        self.world.attach_sphere_collider(entity_id, radius)

    def attach_capsule_collider(self, entity_id: int, half_height: float, radius: float) -> None:
        """Attaches a capsule collider (Y-axis aligned) to an entity's rigid body."""
        self.world.attach_capsule_collider(entity_id, half_height, radius)

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
        if not self._tracked_entities:
            return

        pool = ecs.pool
        dense_indices = [pool.get_dense_index(ent_id) for ent_id in self._tracked_entities]

        if hasattr(self.world, "sync_transforms_direct"):
            self.world.sync_transforms_direct(
                self._tracked_entities,
                dense_indices,
                ecs.rigid_body_state[1],
            )
        else:
            sync_results = self.world.sync_transforms(self._tracked_entities)
            curr_physics = ecs.rigid_body_state[1]
            for ent_id, px, py, pz, qx, qy, qz, qw in sync_results:
                dense_idx = pool.get_dense_index(ent_id)
                if dense_idx >= 0:
                    curr_physics[dense_idx, 0] = px
                    curr_physics[dense_idx, 1] = py
                    curr_physics[dense_idx, 2] = pz
                    curr_physics[dense_idx, 3] = qx
                    curr_physics[dense_idx, 4] = qy
                    curr_physics[dense_idx, 5] = qz
                    curr_physics[dense_idx, 6] = qw

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
