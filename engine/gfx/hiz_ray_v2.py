"""TraceHiZ V2.1 — incremental perspective-correct projected-ray representation.

This module implements ONLY the projected-ray setup and evaluation math for the
V2 traversal design. It does not perform any Hi-Z traversal, sampling, or
hierarchy work. It exists to prove that the incremental screen-space
representation reproduces direct view-space projection of the same ray.

Engine conventions (confirmed from ``engine/core/math_utils.py`` and
``engine/camera/camera.py``):

* Right-handed view space, forward = -Z.
* Column-major perspective matrix; ``clip.w = -viewZ``.
* Positive linear view depth ``w = -viewPosition.z``.
* UV mapping ``uv = (clip.xy / clip.w) * 0.5 + 0.5`` (matches the shader's
  ``HiZProjectToUV``).

The projection of a 3D line is a straight 2D line in UV space, so the screen
direction is constant. Perspective-correct interpolation of ``1/w`` and
``t/w`` (both linear in screen space) recovers the ray parameter ``t`` and the
linear depth ``w`` at any screen position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this projected screen length the ray is treated as degenerate.
_MIN_SCREEN_LENGTH = 1e-6
# Below this clip w the point is behind (or on) the camera plane.
_MIN_CLIP_W = 1e-6


@dataclass
class ProjectedRay:
    """Incremental screen-space representation of a view-space ray segment."""

    valid: bool
    uv_start: np.ndarray
    uv_end: np.ndarray
    screen_length: float
    dir_uv: np.ndarray
    inv_w_start: float
    inv_w_end: float
    t_over_w_start: float
    t_over_w_end: float
    d_inv_w_ds: float
    d_t_over_w_ds: float
    t_start: float
    t_end: float


def _project_view_point(view_position: np.ndarray, projection: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Projects a view-space point to UV, returning (uv, clip_w) or None if behind."""
    clip = projection @ np.array(
        [view_position[0], view_position[1], view_position[2], 1.0], dtype=np.float64
    )
    w = float(clip[3])
    if w <= _MIN_CLIP_W:
        return None
    uv = (clip[:2] / w) * 0.5 + 0.5
    return uv.astype(np.float64), w


def setup_projected_ray(
    origin_view: np.ndarray,
    direction_view: np.ndarray,
    t_start: float,
    t_end: float,
    projection: np.ndarray,
) -> ProjectedRay:
    """Builds the incremental projected-ray representation for ``P(t)=O+tD``.

    Returns a ``ProjectedRay`` with ``valid=False`` when the segment cannot be
    represented (degenerate screen length, endpoint behind the camera, or
    non-finite values).
    """
    origin = np.asarray(origin_view, dtype=np.float64)
    direction = np.asarray(direction_view, dtype=np.float64)
    proj = np.asarray(projection, dtype=np.float64).reshape((4, 4), order="F")

    invalid = ProjectedRay(
        valid=False,
        uv_start=np.zeros(2),
        uv_end=np.zeros(2),
        screen_length=0.0,
        dir_uv=np.zeros(2),
        inv_w_start=0.0,
        inv_w_end=0.0,
        t_over_w_start=0.0,
        t_over_w_end=0.0,
        d_inv_w_ds=0.0,
        d_t_over_w_ds=0.0,
        t_start=float(t_start),
        t_end=float(t_end),
    )

    if not (np.isfinite(t_start) and np.isfinite(t_end)) or t_end <= t_start:
        return invalid

    p_start = origin + direction * t_start
    p_end = origin + direction * t_end

    proj_start = _project_view_point(p_start, proj)
    proj_end = _project_view_point(p_end, proj)
    if proj_start is None or proj_end is None:
        return invalid

    uv_start, w_start = proj_start
    uv_end, w_end = proj_end

    if not (np.all(np.isfinite(uv_start)) and np.all(np.isfinite(uv_end))):
        return invalid
    if w_start <= _MIN_CLIP_W or w_end <= _MIN_CLIP_W:
        return invalid

    delta_uv = uv_end - uv_start
    screen_length = float(np.linalg.norm(delta_uv))
    if not np.isfinite(screen_length) or screen_length < _MIN_SCREEN_LENGTH:
        return invalid

    dir_uv = delta_uv / screen_length

    inv_w_start = 1.0 / w_start
    inv_w_end = 1.0 / w_end
    t_over_w_start = t_start / w_start
    t_over_w_end = t_end / w_end

    d_inv_w_ds = (inv_w_end - inv_w_start) / screen_length
    d_t_over_w_ds = (t_over_w_end - t_over_w_start) / screen_length

    if not all(
        np.isfinite(v)
        for v in (inv_w_start, inv_w_end, t_over_w_start, t_over_w_end, d_inv_w_ds, d_t_over_w_ds)
    ):
        return invalid

    return ProjectedRay(
        valid=True,
        uv_start=uv_start,
        uv_end=uv_end,
        screen_length=screen_length,
        dir_uv=dir_uv,
        inv_w_start=inv_w_start,
        inv_w_end=inv_w_end,
        t_over_w_start=t_over_w_start,
        t_over_w_end=t_over_w_end,
        d_inv_w_ds=d_inv_w_ds,
        d_t_over_w_ds=d_t_over_w_ds,
        t_start=float(t_start),
        t_end=float(t_end),
    )


def evaluate_projected_ray(ray: ProjectedRay, screen_distance: float) -> tuple[np.ndarray, float, float]:
    """Evaluates the representation at a screen distance from the start.

    Returns ``(uv, t, w)`` where ``w`` is positive linear view depth.

    Uses the incremental linear quantities ``1/w`` and ``t/w`` (both linear in
    screen space) and recovers ``t`` and ``w`` with a single division.
    """
    if not ray.valid:
        raise ValueError("Cannot evaluate an invalid ProjectedRay")

    s = float(screen_distance)
    uv = ray.uv_start + ray.dir_uv * s
    inv_w = ray.inv_w_start + ray.d_inv_w_ds * s
    t_over_w = ray.t_over_w_start + ray.d_t_over_w_ds * s
    w = 1.0 / inv_w
    t = t_over_w / inv_w
    return uv, t, w


__all__ = ["ProjectedRay", "setup_projected_ray", "evaluate_projected_ray"]
