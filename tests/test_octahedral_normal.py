"""Regression tests for the octahedral normal encode/decode round-trip.

The G-buffer stores world normals octahedral-encoded (``OctahedralEncode`` in
``shaders/common/pbr_brdf.glsl``). The lower-hemisphere branch must use a sign
function that never returns zero: GLSL's ``sign(0.0) == 0.0``, so a normal with
an exactly-zero component (a cylinder barrel has ``n.y == 0``) would have that
component collapsed and decode back to a wrong axis-aligned normal, producing a
hard vertical seam on curved geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.gfx.shader_utils import load_shader


def _sign_not_zero(v: np.ndarray) -> np.ndarray:
    return np.where(v >= 0.0, 1.0, -1.0)


def _encode(n: np.ndarray) -> np.ndarray:
    """Mirror of OctahedralEncode in shaders/common/pbr_brdf.glsl."""
    n = n / (np.abs(n[..., 0]) + np.abs(n[..., 1]) + np.abs(n[..., 2]))[..., None]
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    upper = np.stack([x, y], -1)
    lower = (1.0 - np.abs(np.stack([y, x], -1))) * np.stack(
        [_sign_not_zero(x), _sign_not_zero(y)], -1
    )
    oct = np.where((z >= 0.0)[..., None], upper, lower)
    return oct * 0.5 + 0.5


def _decode(f: np.ndarray) -> np.ndarray:
    """Mirror of OctahedralDecode in shaders/common/pbr_brdf.glsl."""
    f = f * 2.0 - 1.0
    n = np.stack([f[..., 0], f[..., 1], 1.0 - np.abs(f[..., 0]) - np.abs(f[..., 1])], -1)
    t = np.clip(-n[..., 2], 0.0, 1.0)
    n[..., 0] += np.where(n[..., 0] >= 0.0, -t, t)
    n[..., 1] += np.where(n[..., 1] >= 0.0, -t, t)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _max_error_deg(normals: np.ndarray) -> float:
    decoded = _decode(_encode(normals))
    dots = np.clip(np.sum(decoded * normals, -1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def test_cylinder_barrel_round_trip_is_exact():
    """A cylinder barrel has n.y == 0 exactly; the round-trip must be lossless."""
    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    barrel = np.stack([-np.cos(t), np.zeros_like(t), np.sin(t)], -1)
    assert _max_error_deg(barrel) < 0.01


def test_sphere_round_trip_is_exact():
    u = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    v = np.linspace(-np.pi / 2, np.pi / 2, 16)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(vv) * np.cos(uu), np.sin(vv), np.cos(vv) * np.sin(uu)], -1
    ).reshape(-1, 3)
    assert _max_error_deg(sphere) < 0.01


def test_axis_aligned_normals_round_trip():
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    assert _max_error_deg(axes) < 0.01


def test_encoder_does_not_use_bare_sign():
    """Guard against reintroducing GLSL sign(), which returns 0 for 0 input."""
    source = load_shader("gbuffer.frag")
    assert "sign(n.xy)" not in source, (
        "OctahedralEncode must not use GLSL sign() on a possibly-zero component; "
        "use a sign-not-zero helper instead."
    )
