"""Unit and Integration Tests for Native C++ Pedestrian Acceleration."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from projects.shotgun_escape_the_heat.modules.pedestrians.pedestrian import Pedestrian
from projects.shotgun_escape_the_heat.modules.pedestrians.pedestrian_native import (
    is_native_available,
    update_pedestrians_native,
)


class DummyApp:
    def __init__(self, ecs):
        self.ecs = ecs


def test_native_pedestrian_library_loads():
    """Verifies that the compiled C++ DLL loads successfully via ctypes."""
    assert is_native_available(), "pedestrian_accel.dll should be compiled and available"


def test_native_pedestrian_batch_update():
    """Verifies that update_pedestrians_native correctly steps pedestrians and produces transforms."""
    ecs = EntityManager(max_entities=64)
    app = DummyApp(ecs)

    # Spawn 3 pedestrians along test waypoints
    route_a = [(0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (10.0, 0.0, 0.0)]
    route_b = [(-5.0, 0.0, -5.0), (-15.0, 0.0, -5.0)]

    peds = []
    for i in range(3):
        route = route_a if i < 2 else route_b
        t_id = ecs.create_entity(position=(0.0, 1.0, 0.0), scale=(0.45, 1.05, 0.45))
        h_id = ecs.create_entity(position=(0.0, 1.8, 0.0), scale=(0.36, 0.36, 0.36))
        ped = Pedestrian(app, waypoints=route, walk_speed=1.4, torso_id=t_id, head_id=h_id)
        peds.append(ped)

    dt = 1.0 / 60.0
    car_pos = (0.0, 0.0, 0.0)
    car_vel = (0.0, 0.0, 0.0)
    car_fwd = (0.0, 0.0, -1.0)
    car_right = (1.0, 0.0, 0.0)

    # Record initial positions
    init_pos = [np.copy(p.position) for p in peds]

    # Execute native batch update
    success = update_pedestrians_native(
        peds,
        dt,
        car_pos,
        car_vel,
        car_fwd,
        car_right,
        ecs,
    )
    assert success, "update_pedestrians_native should return True"

    # Pedestrians should have stepped along their paths
    for i, p in enumerate(peds):
        assert not np.array_equal(p.position, init_pos[i]), f"Pedestrian {i} should have moved"

        # Torso and head world_transforms in ECS should be non-zero valid matrices
        d_torso = ecs.pool.get_dense_index(p.torso_id)
        d_head = ecs.pool.get_dense_index(p.head_id)
        assert d_torso >= 0 and d_head >= 0

        torso_mat = ecs.world_transforms[d_torso]
        head_mat = ecs.world_transforms[d_head]

        # In column-major 4x4 flat 16-element matrix:
        # col 3 is [tx, ty, tz, 1.0] at indices 12, 13, 14, 15
        assert torso_mat[15] == 1.0
        assert head_mat[15] == 1.0
        # Translation X, Z should match pedestrian position
        assert abs(torso_mat[12] - p.position[0]) < 1e-4
        assert abs(torso_mat[14] - p.position[2]) < 1e-4
