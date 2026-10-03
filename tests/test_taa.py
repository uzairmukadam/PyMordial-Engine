"""Unit tests for Phase 5: Temporal Anti-Aliasing (TAA) Mathematical Subsystems."""

import numpy as np
import pytest

from engine.gfx.passes.taa_pass import _halton, TAAPass


def rgb_to_ycocg(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Pure-Python reference matching shaders/taa.frag."""
    y = 0.25 * r + 0.5 * g + 0.25 * b
    co = 0.5 * r - 0.5 * b
    cg = -0.25 * r + 0.5 * g - 0.25 * b
    return y, co, cg


def ycocg_to_rgb(y: float, co: float, cg: float) -> tuple[float, float, float]:
    """Pure-Python reference matching shaders/taa.frag."""
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    return r, g, b


def tonemap_ycocg(y: float, co: float, cg: float) -> tuple[float, float, float]:
    w = 1.0 / (1.0 + max(y, 0.0))
    return y * w, co * w, cg * w


def untonemap_ycocg(y: float, co: float, cg: float) -> tuple[float, float, float]:
    w = 1.0 / max(1.0 - y, 1e-4)
    return y * w, co * w, cg * w


def clip_to_aabb(box_min: np.ndarray, box_max: np.ndarray, p: np.ndarray) -> np.ndarray:
    center = 0.5 * (box_max + box_min)
    half_size = 0.5 * (box_max - box_min) + 1e-6
    d = p - center
    d_norm = d / half_size
    max_comp = max(abs(d_norm[0]), abs(d_norm[1]), abs(d_norm[2]))
    if max_comp > 1.0:
        return center + d / max_comp
    return p


class TestHaltonJitterSequence:
    """Validates the mathematical properties of the subpixel Halton(2, 3) sequence."""

    def test_halton_range_and_distribution(self):
        for i in range(1, 64):
            h2 = _halton(i, 2)
            h3 = _halton(i, 3)
            assert 0.0 < h2 < 1.0
            assert 0.0 < h3 < 1.0

    def test_halton_8_normalized_offsets(self):
        assert len(TAAPass.HALTON_8) == 8
        xs = [pt[0] for pt in TAAPass.HALTON_8]
        ys = [pt[1] for pt in TAAPass.HALTON_8]

        # All offsets must be within [-0.5, +0.5] pixel bounds
        for x, y in zip(xs, ys):
            assert -0.5 <= x <= 0.5
            assert -0.5 <= y <= 0.5

        # Subpixel offsets must be roughly zero-centered across the 8 phases
        mean_x = sum(xs) / 8.0
        mean_y = sum(ys) / 8.0
        assert abs(mean_x) < 0.15
        assert abs(mean_y) < 0.15

    def test_jitter_resolution_scaling(self):
        class MockTAA:
            HALTON_8 = TAAPass.HALTON_8
            frame_idx = 0
            get_jitter = TAAPass.get_jitter

        taa = MockTAA()
        w, h = 1920, 1080
        jx, jy = taa.get_jitter(w, h)
        # Jitter in UV units must be strictly smaller than 0.5 / resolution
        assert abs(jx) <= 0.5 / w
        assert abs(jy) <= 0.5 / h


class TestColorSpaceTransforms:
    """Verifies invertible orthogonal YCoCg transforms and tonemapping."""

    @pytest.mark.parametrize("r, g, b", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.85, 0.12, 0.05),
        (0.1, 0.75, 0.35),
        (0.12, 0.45, 0.88),
        (12.5, 45.0, 8.2),   # HDR values
        (0.001, 0.002, 0.005), # Deep shadows
    ])
    def test_ycocg_roundtrip_precision(self, r, g, b):
        y, co, cg = rgb_to_ycocg(r, g, b)
        r_out, g_out, b_out = ycocg_to_rgb(y, co, cg)

        assert pytest.approx(r_out, abs=1e-5) == r
        assert pytest.approx(g_out, abs=1e-5) == g
        assert pytest.approx(b_out, abs=1e-5) == b

    @pytest.mark.parametrize("y, co, cg", [
        (0.0, 0.0, 0.0),
        (0.5, 0.1, -0.05),
        (2.0, 0.3, -0.1),
        (15.0, 0.8, -0.4),
    ])
    def test_tonemap_untonemap_roundtrip(self, y, co, cg):
        ty, tco, tcg = tonemap_ycocg(y, co, cg)
        # Tonemapped Y must be strictly in [0, 1)
        assert 0.0 <= ty < 1.0

        # Untonemap should recover exact original values
        uy, uco, ucg = untonemap_ycocg(ty, tco, tcg)
        assert pytest.approx(uy, rel=1e-3, abs=1e-4) == y
        assert pytest.approx(uco, rel=1e-3, abs=1e-4) == co
        assert pytest.approx(ucg, rel=1e-3, abs=1e-4) == cg


class TestAABBRayClipping:
    """Validates Playdead / Salvi variance AABB clipping."""

    def test_point_inside_aabb_remains_unchanged(self):
        bmin = np.array([0.2, -0.1, -0.1], dtype=np.float32)
        bmax = np.array([0.8, 0.1, 0.1], dtype=np.float32)
        p = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        res = clip_to_aabb(bmin, bmax, p)
        assert np.allclose(res, p, atol=1e-5)

    def test_point_outside_aabb_clipped_to_surface(self):
        bmin = np.array([0.2, -0.1, -0.1], dtype=np.float32)
        bmax = np.array([0.8, 0.1, 0.1], dtype=np.float32)
        p = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # Far to the right on Y axis

        res = clip_to_aabb(bmin, bmax, p)
        # Should be clamped to the right face of the AABB (res[0] ≈ 0.8)
        assert pytest.approx(res[0], abs=1e-3) == 0.8
        assert pytest.approx(res[1], abs=1e-3) == 0.0
        assert pytest.approx(res[2], abs=1e-3) == 0.0
