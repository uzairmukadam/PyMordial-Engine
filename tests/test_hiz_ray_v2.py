"""V2.1 validation: incremental projected-ray representation vs direct projection.

Proves that ``setup_projected_ray`` + ``evaluate_projected_ray`` reproduce the
direct view-space projection of the same ray at arbitrary interior points.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math_utils import matrix_perspective
from engine.gfx.hiz_ray_v2 import evaluate_projected_ray, setup_projected_ray

FOVY = np.radians(60.0)
ASPECT = 16.0 / 9.0
NEAR = 0.1
FAR = 1000.0
PROJ = matrix_perspective(FOVY, ASPECT, NEAR, FAR, reverse_z=True).reshape((4, 4), order="F")


def _direct_project(p: np.ndarray) -> tuple[np.ndarray, float]:
    clip = PROJ @ np.array([p[0], p[1], p[2], 1.0])
    w = float(clip[3])
    uv = (clip[:2] / w) * 0.5 + 0.5
    return uv, w


def _screen_distance_of(ray, uv_target: np.ndarray) -> float:
    """Screen distance along the projected segment that lands on uv_target."""
    return float(np.dot(uv_target - ray.uv_start, ray.dir_uv))


# (name, origin, direction, t_start, t_end)
RAYS = [
    ("horizontal", np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]), 0.5, 4.0),
    ("vertical", np.array([0.0, 0.0, -5.0]), np.array([0.0, 1.0, 0.0]), 0.5, 4.0),
    ("diagonal", np.array([0.0, 0.0, -5.0]), np.array([0.7, 0.7, 0.0]), 0.5, 4.0),
    ("depth_change", np.array([0.0, 0.0, -3.0]), np.array([0.2, 0.1, -1.0]), 0.5, 6.0),
    ("small_depth", np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, -0.01]), 0.5, 4.0),
    ("toward_camera", np.array([0.0, 0.0, -8.0]), np.array([0.3, 0.0, 1.0]), 0.5, 5.0),
    ("away_camera", np.array([0.0, 0.0, -3.0]), np.array([0.3, 0.0, -1.0]), 0.5, 5.0),
    ("near_axis_x", np.array([0.0, 0.0, -5.0]), np.array([1.0, 1e-4, 0.0]), 0.5, 4.0),
    ("long", np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.2, 0.0]), 0.1, 20.0),
    ("short", np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]), 0.5, 0.6),
]


@pytest.mark.parametrize("name,origin,direction,t0,t1", RAYS, ids=[r[0] for r in RAYS])
def test_interior_points_match_direct_projection(name, origin, direction, t0, t1):
    ray = setup_projected_ray(origin, direction, t0, t1, PROJ)
    assert ray.valid, f"{name}: setup unexpectedly invalid"

    max_uv_err = 0.0
    max_t_err = 0.0
    max_w_err = 0.0
    prev_t = -np.inf

    for frac in np.linspace(0.0, 1.0, 17):
        t_direct = t0 + (t1 - t0) * frac
        p = origin + direction * t_direct
        uv_direct, w_direct = _direct_project(p)

        s = _screen_distance_of(ray, uv_direct)
        uv_v2, t_v2, w_v2 = evaluate_projected_ray(ray, s)

        max_uv_err = max(max_uv_err, float(np.max(np.abs(uv_v2 - uv_direct))))
        max_t_err = max(max_t_err, abs(t_v2 - t_direct))
        max_w_err = max(max_w_err, abs(w_v2 - w_direct))

        # Monotonic t along increasing screen distance.
        assert t_v2 >= prev_t - 1e-9, f"{name}: t decreased at frac={frac}"
        prev_t = t_v2

    assert max_uv_err < 1e-5, f"{name}: UV error {max_uv_err}"
    assert max_t_err < 1e-4, f"{name}: t error {max_t_err}"
    assert max_w_err < 1e-4, f"{name}: w error {max_w_err}"


@pytest.mark.parametrize("name,origin,direction,t0,t1", RAYS, ids=[r[0] for r in RAYS])
def test_endpoint_identities(name, origin, direction, t0, t1):
    ray = setup_projected_ray(origin, direction, t0, t1, PROJ)
    assert ray.valid

    uv0, t_at0, w_at0 = evaluate_projected_ray(ray, 0.0)
    uv1, t_at1, w_at1 = evaluate_projected_ray(ray, ray.screen_length)

    assert np.allclose(uv0, ray.uv_start, atol=1e-9)
    assert np.allclose(uv1, ray.uv_end, atol=1e-9)
    assert abs(t_at0 - t0) < 1e-6
    assert abs(t_at1 - t1) < 1e-6

    _, w_direct0 = _direct_project(origin + direction * t0)
    _, w_direct1 = _direct_project(origin + direction * t1)
    assert abs(w_at0 - w_direct0) < 1e-6
    assert abs(w_at1 - w_direct1) < 1e-6


def test_zero_screen_length_is_invalid():
    # Ray straight along the view axis: projects to a single point.
    ray = setup_projected_ray(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, -1.0]), 0.5, 4.0, PROJ)
    assert not ray.valid


def test_endpoint_behind_camera_is_invalid():
    # Ray travels toward and past the camera plane.
    ray = setup_projected_ray(np.array([0.0, 0.0, -2.0]), np.array([0.0, 0.0, 1.0]), 0.5, 5.0, PROJ)
    assert not ray.valid


def test_invalid_t_range():
    ray = setup_projected_ray(np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]), 4.0, 4.0, PROJ)
    assert not ray.valid


def test_near_axis_aligned_components():
    # Nearly zero dirUV.y must still produce a valid, accurate representation.
    ray = setup_projected_ray(np.array([0.0, 0.0, -5.0]), np.array([1.0, 1e-5, 0.0]), 0.5, 4.0, PROJ)
    assert ray.valid
    assert abs(ray.dir_uv[1]) < 1e-3
    uv, t, w = evaluate_projected_ray(ray, ray.screen_length * 0.5)
    p = np.array([0.0, 0.0, -5.0]) + np.array([1.0, 1e-5, 0.0]) * t
    uv_direct, w_direct = _direct_project(p)
    assert np.allclose(uv, uv_direct, atol=1e-5)
    assert abs(w - w_direct) < 1e-4
