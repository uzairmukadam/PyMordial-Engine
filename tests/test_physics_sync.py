"""Unit tests for Rapier3D physics stepping, ECS sync, and raycasting."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager


class TestPhysicsSync:
    def test_physics_step_and_ecs_sync(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)

        # Entity 0: Fixed Ground Box at y = -1.0
        ground_ent = ecs.create_entity(position=(0.0, -1.0, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -1.0, 0.0))
        physics.attach_box_collider(ground_ent, half_x=50.0, half_y=1.0, half_z=50.0)

        # Entity 1: Dynamic Sphere at y = 10.0
        sphere_ent = ecs.create_entity(position=(0.0, 10.0, 0.0))
        physics.create_body(sphere_ent, body_type="dynamic", position=(0.0, 10.0, 0.0))
        physics.attach_sphere_collider(sphere_ent, radius=0.5)

        sphere_dense = ecs.pool.get_dense_index(sphere_ent)
        assert sphere_dense >= 0

        # Snapshot previous state in ECS
        ecs.cache_previous_physics_state()
        assert ecs.rigid_body_state[0, sphere_dense, 1] == 10.0

        # Step 10 frames (10 * 1/60s)
        for _ in range(10):
            physics.step_simulation(1.0 / 60.0)

        # Sync to ECS
        physics.sync_to_ecs(ecs)

        # Sphere should have fallen down under gravity
        new_y = ecs.rigid_body_state[1, sphere_dense, 1]
        assert new_y < 10.0
        assert new_y > 0.0  # Not reached ground yet

        # Interpolate transforms at alpha = 0.5
        ecs.interpolate_render_transforms(alpha=0.5)
        interpolated_y = ecs.world_transforms[sphere_dense, 13]  # Column 3 y
        assert interpolated_y == pytest.approx((10.0 + new_y) * 0.5, rel=1e-3)

    def test_raycast_query(self):
        physics = PhysicsManager()
        # Ground at y = 0
        physics.create_body(10, body_type="fixed", position=(0.0, 0.0, 0.0))
        physics.attach_box_collider(10, half_x=10.0, half_y=0.1, half_z=10.0)

        # Step once to register query pipeline
        physics.step_simulation(1.0 / 60.0)

        # Cast ray downward from (0, 5, 0)
        hit = physics.raycast(origin=(0.0, 5.0, 0.0), direction=(0.0, -1.0, 0.0), max_distance=10.0)
        assert hit is not None
        hit_id, hit_dist, normal = hit
        assert hit_id == 10
        assert hit_dist == pytest.approx(4.9, abs=0.1)  # 5.0 - 0.1 (half_y) = 4.9
        assert normal[1] > 0.9  # Upward normal (+Y)
