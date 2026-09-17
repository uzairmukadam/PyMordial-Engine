"""Unit and Integration Tests for Phase 6: CSM Shadow Map Stabilization & Texel Snapping.

Validates that sub-texel camera translations preserve exact texel alignment in the light
view-projection matrices, eliminating shadow shimmering and edge swimming.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig, GraphicsQuality, get_quality_preset
from engine.gfx.shadow_csm import CascadedShadowMap
from engine.debug.engine_tweaks import EngineTweaks


@pytest.fixture(scope="module")
def headless_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestCSMStabilization:
    """Tests subpixel texel snapping and cascade stability."""

    def test_csm_stabilization_defaults(self, headless_ctx: RenderContext):
        """Verifies default stabilization flags in config and CascadedShadowMap."""
        cfg = RenderConfig()
        assert cfg.csm_stabilization is True

        for quality in [GraphicsQuality.LOW, GraphicsQuality.MEDIUM, GraphicsQuality.HIGH, GraphicsQuality.ULTRA, GraphicsQuality.CINEMATIC]:
            preset = get_quality_preset(quality)
            assert preset.csm_stabilization is True

        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        assert csm.stabilize_cascades is True
        csm.destroy()

    def test_csm_subpixel_camera_translation_invariant(self, headless_ctx: RenderContext):
        """Verifies that sub-texel camera translation produces identical snapped matrices."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([100.0, 15.0, -50.0], dtype=np.float32)
        cam_fwd = np.array([0.0, -0.3, -0.954], dtype=np.float32)
        cam_fwd /= np.linalg.norm(cam_fwd)
        sun_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)

        # Light right vector s perpendicular to sun direction
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        s = np.cross(sun_dir, up)
        s /= np.linalg.norm(s)

        quad_size = csm.atlas_size // 2

        for c in range(4):
            near = 0.1 if c == 0 else csm.split_distances[c - 1]
            far = csm.split_distances[c]
            center_dist = (near + far) * 0.5
            center = cam_pos + cam_fwd * center_dist
            half_span = (far - near) * 0.5
            far_half_w = far * 0.65
            raw_radius = math.sqrt(half_span * half_span + far_half_w * far_half_w) * 1.15
            radius = math.ceil(raw_radius * 2.0) * 0.5
            texel_size_world = (2.0 * radius) / float(quad_size)

            # Center camera in discrete texel bin for cascade c
            val = float(np.dot(s, center))
            bin_frac = (val / texel_size_world) % 1.0
            cam_pos_centered = cam_pos + s * ((0.5 - bin_frac) * texel_size_world)

            m_base = csm.compute_cascade_matrices(cam_pos_centered, cam_fwd, sun_dir, stabilize=True)[c].copy()

            # Shift by 15% of this cascade's texel size
            cam_pos_subpixel = cam_pos_centered + s * (0.15 * texel_size_world)
            m_shifted = csm.compute_cascade_matrices(cam_pos_subpixel, cam_fwd, sun_dir, stabilize=True)[c]

            np.testing.assert_array_almost_equal(
                m_base,
                m_shifted,
                decimal=5,
                err_msg=f"Cascade {c} matrix drifted under sub-texel translation with stabilization enabled!",
            )

            # Verify that crossing a full texel boundary steps by an exact integer multiple of NDC texel size
            cam_pos_next_bin = cam_pos_centered + s * (1.0 * texel_size_world)
            m_next = csm.compute_cascade_matrices(cam_pos_next_bin, cam_fwd, sun_dir, stabilize=True)[c]
            ndc_step = abs(m_next[12] - m_base[12])
            expected_ndc_step = 2.0 / float(quad_size)
            assert math.isclose(ndc_step, expected_ndc_step, abs_tol=1e-5), (
                f"Expected discrete NDC step {expected_ndc_step}, got {ndc_step} for cascade {c}"
            )

        csm.destroy()

    def test_csm_unstabilized_drifts_continuously(self, headless_ctx: RenderContext):
        """Verifies that disabling stabilization allows continuous sub-texel floating-point drift."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([100.0, 15.0, -50.0], dtype=np.float32)
        cam_fwd = np.array([0.0, -0.3, -0.954], dtype=np.float32)
        cam_fwd /= np.linalg.norm(cam_fwd)
        sun_dir = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)

        m_base = [m.copy() for m in csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir, stabilize=False)]

        # Small translation (0.05m)
        cam_pos_shift = cam_pos + np.array([0.05, 0.0, 0.0], dtype=np.float32)
        m_shifted = csm.compute_cascade_matrices(cam_pos_shift, cam_fwd, sun_dir, stabilize=False)

        # Without stabilization, the light view translation changes immediately
        for c in range(4):
            assert not np.allclose(m_base[c], m_shifted[c], atol=1e-5), (
                f"Cascade {c} matrix should have drifted when stabilization was disabled"
            )

        csm.destroy()

    def test_csm_matrix_validity_and_determinant(self, headless_ctx: RenderContext):
        """Verifies that cascade projection matrices are valid, invertible, and finite."""
        csm = CascadedShadowMap(headless_ctx.ctx, atlas_size=2048, cascade_count=4)
        cam_pos = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        cam_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        sun_dir = np.array([0.4, 0.7, 0.2], dtype=np.float32)

        matrices = csm.compute_cascade_matrices(cam_pos, cam_fwd, sun_dir, stabilize=True)
        assert len(matrices) == 4

        for c, mat in enumerate(matrices):
            assert not np.isnan(mat).any()
            assert not np.isinf(mat).any()
            mat4x4 = mat.reshape((4, 4), order="F")
            det = np.linalg.det(mat4x4)
            # Orthographic determinant scale is 1 / (radius^2 * far), strictly non-zero
            assert abs(det) > 1e-15, f"Cascade {c} matrix is singular (det={det})"

        csm.destroy()

    def test_engine_tweaks_csm_stabilization_sync(self):
        """Verifies that EngineTweaks syncs and serializes csm_stabilization correctly."""
        tweaks = EngineTweaks()
        assert tweaks.csm_stabilization is True

        tweaks.csm_stabilization = False
        d = tweaks.to_dict()
        assert d["csm_stabilization"] is False

        new_tweaks = EngineTweaks()
        new_tweaks.from_dict(d)
        assert new_tweaks.csm_stabilization is False
