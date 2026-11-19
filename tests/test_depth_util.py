"""Validation tests for the canonical GLSL depth utility (shaders/common/depth.glsl).

Runs ``shaders/depth_util_test.comp`` against a synthetic reversed-Z depth
buffer and checks the depth -> view-position -> UV round trip.
"""

import numpy as np
import pytest

from engine.core.math_utils import mat4_inv, matrix_perspective
from engine.gfx.context import RenderContext
from engine.gfx.shader_utils import load_shader

NEAR = 0.1
FAR = 2000.0
FOVY = np.radians(60.0)


def _projection(width: int, height: int) -> np.ndarray:
    return matrix_perspective(FOVY, width / height, NEAR, FAR, reverse_z=True)


def _raw_depth_for_view_z(view_z: float, proj: np.ndarray) -> float:
    """Returns the reversed-Z raw depth a point at ``view_z`` would write."""
    clip = proj.reshape((4, 4), order="F") @ np.array([0.0, 0.0, view_z, 1.0])
    return float(clip[2] / clip[3])


class TestDepthUtility:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=64, height=64)
        yield ctx
        ctx.destroy()

    def _run(self, ctx, width, height, depth_values):
        """Uploads a depth image and returns (position, round_trip) float arrays."""
        proj = _projection(width, height)
        inv_proj = mat4_inv(proj)

        depth_tex = ctx.ctx.depth_texture((width, height))
        depth_tex.filter = (0, 0)
        depth_tex.write(np.asarray(depth_values, dtype=np.float32).tobytes())

        out_pos = ctx.ctx.texture((width, height), 4, dtype="f4")
        out_rt = ctx.ctx.texture((width, height), 4, dtype="f4")
        out_pos.bind_to_image(0, read=False, write=True)
        out_rt.bind_to_image(1, read=False, write=True)

        prog = ctx.ctx.compute_shader(load_shader("depth_util_test.comp"))
        prog["u_Projection"].write(proj.tobytes())
        prog["u_InvProjection"].write(inv_proj.tobytes())
        depth_tex.use(location=0)
        prog.run(group_x=(width + 7) // 8, group_y=(height + 7) // 8)

        pos = np.frombuffer(out_pos.read(), dtype=np.float32).reshape((height, width, 4))
        rt = np.frombuffer(out_rt.read(), dtype=np.float32).reshape((height, width, 4))

        prog.release()
        out_pos.release()
        out_rt.release()
        depth_tex.release()
        return pos, rt

    def test_round_trip_center_and_edges(self, render_ctx: RenderContext):
        width, height = 64, 64
        proj = _projection(width, height)
        view_z = -12.5
        raw = _raw_depth_for_view_z(view_z, proj)
        depths = np.full((height, width), raw, dtype=np.float32)

        pos, rt = self._run(render_ctx, width, height, depths)

        # Every pixel reconstructs the same view-space plane.
        assert np.allclose(pos[..., 2], view_z, atol=1e-3)
        assert np.allclose(pos[..., 3], -view_z, atol=1e-3)

        # Round trip: reconstructed UV must match the source pixel centre.
        u = (np.arange(width) + 0.5) / width
        v = (np.arange(height) + 0.5) / height
        expected_u = np.tile(u, (height, 1))
        expected_v = np.tile(v[:, None], (1, width))
        assert np.allclose(rt[..., 0], expected_u, atol=1e-4)
        assert np.allclose(rt[..., 1], expected_v, atol=1e-4)
        assert np.all(rt[..., 3] == 1.0)

    def test_linear_depth_is_positive_and_ordered(self, render_ctx: RenderContext):
        width, height = 32, 32
        proj = _projection(width, height)
        near_raw = _raw_depth_for_view_z(-2.0, proj)
        far_raw = _raw_depth_for_view_z(-80.0, proj)

        depths = np.full((height, width), far_raw, dtype=np.float32)
        depths[:, : width // 2] = near_raw

        pos, _ = self._run(render_ctx, width, height, depths)

        near_linear = pos[:, : width // 2, 3]
        far_linear = pos[:, width // 2 :, 3]
        assert np.all(near_linear > 0.0)
        assert np.all(far_linear > 0.0)
        # Smaller linear depth means nearer.
        assert np.all(near_linear < far_linear)
        assert np.allclose(near_linear, 2.0, atol=1e-3)
        assert np.allclose(far_linear, 80.0, atol=1e-2)

    def test_background_detection(self, render_ctx: RenderContext):
        width, height = 16, 16
        proj = _projection(width, height)
        raw = _raw_depth_for_view_z(-5.0, proj)

        depths = np.full((height, width), raw, dtype=np.float32)
        depths[: height // 2, :] = 0.0  # sky

        pos, rt = self._run(render_ctx, width, height, depths)

        assert np.all(rt[: height // 2, :, 2] == 1.0)  # background flag
        assert np.all(rt[height // 2 :, :, 2] == 0.0)
        # Background reconstructs to a finite zero, never NaN/Inf.
        assert np.all(np.isfinite(pos))
        assert np.all(pos[: height // 2, :, 3] == 0.0)

    def test_no_nan_or_inf_for_valid_geometry(self, render_ctx: RenderContext):
        width, height = 24, 24
        proj = _projection(width, height)
        depths = np.zeros((height, width), dtype=np.float32)
        for row in range(height):
            depths[row, :] = _raw_depth_for_view_z(-(1.0 + row * 3.0), proj)

        pos, rt = self._run(render_ctx, width, height, depths)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(rt))
        assert np.all(pos[..., 3] > 0.0)
