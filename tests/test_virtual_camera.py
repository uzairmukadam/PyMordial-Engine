"""Unit tests for PyMordial Engine Virtual Camera Stack."""

from __future__ import annotations
import numpy as np
import pytest
from engine.camera import (
    VirtualCamera,
    FollowCamera,
    FollowCameraConfig,
    FreeFlyCamera,
    OrbitCamera,
    CameraManager,
)


def test_virtual_camera_matrices():
    cam = VirtualCamera(position=(0, 2, 5), target=(0, 0, 0), fov=60.0)
    view = cam.get_view_matrix()
    proj = cam.get_projection_matrix(reverse_z=True)

    assert view.size == 16
    assert proj.size == 16
    assert not np.isnan(view).any()
    assert not np.isnan(proj).any()

    fwd = cam.get_forward_vector()
    assert abs(np.linalg.norm(fwd) - 1.0) < 1e-5


def test_free_fly_camera_movement():
    free = FreeFlyCamera(position=(0, 0, 0), move_speed=10.0)
    # Move forward
    free.move(dt=0.1, forward_axis=1.0)
    assert free.position[2] < 0.0 or free.position[0] != 0.0 or free.position[1] != 0.0


def test_orbit_camera_angles():
    orbit = OrbitCamera(pivot=(0, 1, 0), radius=5.0)
    initial_pos = orbit.position.copy()
    orbit.handle_orbit(rel_x=20.0, rel_y=10.0)
    assert not np.allclose(orbit.position, initial_pos)


def test_camera_manager_blending():
    mgr = CameraManager()
    cam_a = VirtualCamera(position=(0, 0, 0), fov=60.0)
    cam_b = VirtualCamera(position=(10, 10, 10), fov=90.0)

    mgr.register_camera("cam_a", cam_a, make_active=True)
    mgr.register_camera("cam_b", cam_b)

    assert mgr.active_camera_name == "cam_a"
    assert mgr.active_camera is cam_a

    # Start blend
    assert mgr.blend_to("cam_b", duration_seconds=1.0)
    assert mgr._is_blending

    # Advance 0.5s (halfway)
    mgr.update(0.5)
    blended = mgr.active_camera
    assert 0.0 < blended.position[0] < 10.0
    assert 60.0 < blended.fov < 90.0

    # Finish blend
    mgr.update(0.6)
    assert not mgr._is_blending
    assert mgr.active_camera is cam_b
