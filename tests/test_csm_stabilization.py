"""Unit and Integration Tests for Phase 6: CSM Shadow Map Stabilization & Texel Snapping.

Validates that sub-texel camera translations preserve exact texel alignment in the light
view-projection matrices, eliminating shadow shimmering and edge swimming.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.debug.engine_tweaks import EngineTweaks
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.shadow_csm import MAX_SPAN_RATIO, CascadedShadowMap


def _max_span_ratio(splits: tuple[float, ...], near: float = 0.1) -> float:
    """Worst adjacent world-space span ratio between cascade slices."""
    previous = near
    previous_span = 0.0
    worst = 1.0
    for split in splits:
        span = split - previous
        if previous_span > 0.0:
            worst = max(worst, span / previous_span)
        previous_span = span
        previous = split
    return worst


@pytest.fixture(scope="module")
def headless_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestCSMStabilization:
    """Tests subpixel texel snapping and cascade stability."""

    def test_texel_snapping_is_always_on(self, headless_ctx: RenderContext):
        """Texel snapping is unconditional now that the toggle is gone."""
        cfg = RenderConfig()
        assert not hasattr(cfg, "csm_stabilization")

        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        assert not hasattr(csm, "stabilize_cascades")
        csm.destroy()

    def test_csm_subpixel_camera_translation_invariant(self, headless_ctx: RenderContext):
        """Verifies that sub-texel camera translation produces identical snapped matrices."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([100.0, 15.0, -50.0], dtype=np.float32)
        cam_fwd = np.array([0.0, -0.3, -0.954], dtype=np.float32)
        cam_fwd /= np.linalg.norm(cam_fwd)
        sun_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)

        quad_size = csm.atlas_size // 2

        for c in range(4):
            matrix = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir)[c].reshape((4, 4), order="F")
            light_right = matrix[0, :3] / np.linalg.norm(matrix[0, :3])
            radius = 1.0 / float(np.linalg.norm(matrix[0, :3]))
            texel_size_world = (2.0 * radius) / float(quad_size)

            m_base = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir)[c].copy()

            # A shift far below one texel must not move the matrix at all. This
            # is the property that removes shimmer: the light-space centre is
            # quantised, so sub-texel camera motion is invisible to the map.
            cam_pos_subpixel = cam_pos + light_right * (texel_size_world * 1e-3)
            m_shifted = csm.compute_cascade_matrices(cam_pos_subpixel, cam_fwd, sun_dir)[c]
            np.testing.assert_array_almost_equal(
                m_base,
                m_shifted,
                decimal=5,
                err_msg=f"Cascade {c} matrix drifted under a sub-texel translation!",
            )

            # Crossing a full texel must step by exactly one NDC texel.
            cam_pos_next_bin = cam_pos + light_right * texel_size_world
            m_next = csm.compute_cascade_matrices(cam_pos_next_bin, cam_fwd, sun_dir)[c]
            ndc_step = abs(m_next[12] - m_base[12])
            expected_ndc_step = 2.0 / float(quad_size)
            assert math.isclose(ndc_step, expected_ndc_step, abs_tol=1e-5), (
                f"Expected discrete NDC step {expected_ndc_step}, got {ndc_step} for cascade {c}"
            )

        csm.destroy()

    def test_practical_splits_and_lambda_updates(self, headless_ctx: RenderContext):
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4, max_distance=600.0)
        logarithmic = csm.split_distances
        assert all(logarithmic[i] < logarithmic[i + 1] for i in range(3))
        assert logarithmic[-1] == pytest.approx(600.0)
        # The guard must not override the requested lambda. A cap below the
        # natural PSSM ratio silently flattens the cascades and destroys near
        # field texel density, so assert against the constant rather than a
        # literal that can drift out of sync with it.
        assert _max_span_ratio(logarithmic) <= MAX_SPAN_RATIO
        # The near cascade must stay genuinely near-field: a 600m shadow distance
        # with lambda 0.85 puts cascade 0 at ~23m, not at a flattened ~90m.
        assert logarithmic[0] < 40.0

        csm.update_splits(600.0, cascade_count=4, lambda_val=0.0)
        uniform = csm.split_distances
        assert uniform == pytest.approx((150.075, 300.05, 450.025, 600.0))
        assert uniform != logarithmic
        assert _max_span_ratio(uniform) == pytest.approx(1.0)
        csm.destroy()

    def test_cascade_depth_fit_includes_upstream_casters_only_in_footprint(self, headless_ctx: RenderContext):
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4, max_distance=500.0)
        cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        cam_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        sun_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        baseline = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir)[0].copy()
        upstream = np.array([[0.0, 100.0, -2.0, 0.5, 0.5, 0.5]], dtype=np.float32)
        relevant = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir, aabbs=upstream, active_count=1)[0].copy()
        outside_footprint = np.array([[1000.0, 100.0, -2.0, 0.5, 0.5, 0.5]], dtype=np.float32)
        outside = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir, aabbs=outside_footprint, active_count=1)[
            0
        ].copy()

        assert not np.allclose(relevant, baseline)
        np.testing.assert_array_equal(outside, baseline)
        csm.destroy()

    def test_cascade_bounds_cover_full_view_frustum(self, headless_ctx: RenderContext):
        """Checks that all corners remain inside each map, including wide/high-FOV views."""
        for cascade_count in (2, 4):
            csm = CascadedShadowMap(
                headless_ctx.ctx,
                atlas_size=2048,
                cascade_count=cascade_count,
                max_distance=600.0,
            )
            cam_pos = np.array([3.0, 7.0, -11.0], dtype=np.float32)
            cam_fwd = np.array([0.3, -0.15, -0.94], dtype=np.float32)
            cam_fwd /= np.linalg.norm(cam_fwd)
            sun_dir = np.array([0.45, 0.8, -0.25], dtype=np.float32)
            sun_dir /= np.linalg.norm(sun_dir)
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(cam_fwd, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, cam_fwd)
            fovy_deg = 82.0
            aspect = 21.0 / 9.0
            matrices = csm.compute_cascade_matrices(
                cam_pos,
                cam_fwd,
                sun_dir,
                fovy_deg=fovy_deg,
                aspect=aspect,
            )

            near = 0.1
            tan_half_y = math.tan(math.radians(fovy_deg) * 0.5)
            for cascade in range(cascade_count):
                far = csm.split_distances[cascade]
                matrix = matrices[cascade].reshape((4, 4), order="F")
                for depth in (near, far):
                    half_h = depth * tan_half_y
                    half_w = half_h * aspect
                    center = cam_pos + cam_fwd * depth
                    for x_sign in (-1.0, 1.0):
                        for y_sign in (-1.0, 1.0):
                            corner = center + right * (x_sign * half_w) + up * (y_sign * half_h)
                            light_clip = matrix @ np.append(corner, 1.0)
                            assert abs(float(light_clip[0])) <= 1.0
                            assert abs(float(light_clip[1])) <= 1.0
                            assert 0.0 <= float(light_clip[2]) <= 1.0
                near = far
            csm.destroy()

    def test_cascade_radii_do_not_breathe_with_camera_rotation(self, headless_ctx: RenderContext):
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([4.0, 3.0, -8.0], dtype=np.float32)
        sun_dir = np.array([0.3, 0.8, 0.2], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)
        forward_a = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward_b = np.array([0.75, 0.25, -0.61], dtype=np.float32)
        forward_b /= np.linalg.norm(forward_b)

        matrices_a = [matrix.copy() for matrix in csm.compute_cascade_matrices(cam_pos, forward_a, sun_dir)]
        matrices_b = csm.compute_cascade_matrices(cam_pos, forward_b, sun_dir)
        for cascade in range(csm.cascade_count):
            radius_a = 1.0 / np.linalg.norm(matrices_a[cascade].reshape((4, 4), order="F")[0, :3])
            radius_b = 1.0 / np.linalg.norm(matrices_b[cascade].reshape((4, 4), order="F")[0, :3])
            assert radius_a == pytest.approx(radius_b, rel=1e-6)
        csm.destroy()

    def test_csm_unstabilized_drifts_continuously(self, headless_ctx: RenderContext):
        """Verifies that disabling stabilization allows continuous sub-texel floating-point drift."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([100.0, 15.0, -50.0], dtype=np.float32)
        cam_fwd = np.array([0.0, -0.3, -0.954], dtype=np.float32)
        cam_fwd /= np.linalg.norm(cam_fwd)
        sun_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)

        m_base = [m.copy() for m in csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir)]

        # Small translation (0.05m)
        cam_pos_shift = cam_pos + np.array([0.05, 0.0, 0.0], dtype=np.float32)
        m_shifted = csm.compute_cascade_matrices(cam_pos_shift, cam_fwd, sun_dir)

        # Texel snapping quantises the light-space centre, so a 5cm shift must
        # not move the matrix at all: that is exactly what removes shimmer.
        for c in range(4):
            np.testing.assert_array_almost_equal(
                m_base[c],
                m_shifted[c],
                decimal=5,
                err_msg=f"Cascade {c} matrix drifted under a sub-texel translation",
            )

        csm.destroy()

    def test_csm_matrix_validity_and_determinant(self, headless_ctx: RenderContext):
        """Verifies that cascade projection matrices are valid, invertible, and finite."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        cam_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        sun_dir = np.array([0.4, 0.7, 0.2], dtype=np.float32)
        input_vectors = (cam_pos.copy(), cam_fwd.copy(), sun_dir.copy())

        matrices = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir)
        assert len(matrices) == 4
        np.testing.assert_array_equal(cam_pos, input_vectors[0])
        np.testing.assert_array_equal(cam_fwd, input_vectors[1])
        np.testing.assert_array_equal(sun_dir, input_vectors[2])

        for c, mat in enumerate(matrices):
            assert not np.isnan(mat).any()
            assert not np.isinf(mat).any()
            mat4x4 = mat.reshape((4, 4), order="F")
            det = np.linalg.det(mat4x4)
            # Orthographic determinant scale is 1 / (radius^2 * far), strictly non-zero
            assert abs(det) > 1e-15, f"Cascade {c} matrix is singular (det={det})"

        csm.destroy()

    def test_engine_tweaks_csm_cascade_distance_sync(self):
        """Verifies that EngineTweaks syncs and serializes the cascade distance."""
        tweaks = EngineTweaks()
        assert tweaks.shadow_distance == 500.0

        tweaks.shadow_distance = 900.0
        d = tweaks.to_dict()
        assert d["shadow_distance"] == 900.0

        new_tweaks = EngineTweaks()
        new_tweaks.from_dict(d)
        assert new_tweaks.shadow_distance == 900.0
