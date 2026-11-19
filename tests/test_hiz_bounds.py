"""Validation tests for the RG32F linear-view-depth Hi-Z bounds hierarchy.

Builds a synthetic reversed-Z depth image, runs the real GPU mip chain, and
compares every level against an independent CPU reference implementation.
"""

import numpy as np
import pytest

from engine.core.math_utils import mat4_inv, matrix_perspective
from engine.gfx.context import RenderContext
from engine.gfx.passes.hiz_pass import HiZPass

EMPTY_NEAREST = 1e30
EMPTY_FARTHEST = 0.0
NEAR = 0.1
FAR = 2000.0
FOVY = np.radians(60.0)


def _raw_depth_for_view_z(view_z: float, proj: np.ndarray) -> float:
    clip = proj.reshape((4, 4), order="F") @ np.array([0.0, 0.0, view_z, 1.0])
    return float(clip[2] / clip[3])


def _cpu_mip_chain(level0: np.ndarray) -> list[np.ndarray]:
    """Reference conservative min/max reduction with clamped edge reads."""
    chain = [level0]
    current = level0
    while current.shape[0] > 1 or current.shape[1] > 1:
        h, w = current.shape[0], current.shape[1]
        nh, nw = max(1, h // 2), max(1, w // 2)
        parent = np.empty((nh, nw, 2), dtype=np.float64)
        for y in range(nh):
            for x in range(nw):
                ys = [min(y * 2 + dy, h - 1) for dy in (0, 1)]
                xs = [min(x * 2 + dx, w - 1) for dx in (0, 1)]
                nearest = EMPTY_NEAREST
                farthest = EMPTY_FARTHEST
                for sy in ys:
                    for sx in xs:
                        n, f = current[sy, sx]
                        if n <= f:  # valid child
                            nearest = min(nearest, n)
                            farthest = max(farthest, f)
                parent[y, x] = (nearest, farthest)
        chain.append(parent)
        current = parent
    return chain


def _build_depth_image(width, height, occupied):
    """occupied[y, x] is a view-space Z (negative) or None for background."""
    proj = matrix_perspective(FOVY, width / height, NEAR, FAR, reverse_z=True)
    depths = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            z = occupied[y][x]
            if z is not None:
                depths[y, x] = _raw_depth_for_view_z(z, proj)
    return depths, proj


def _run_gpu(ctx, width, height, depths, proj):
    pass_instance = HiZPass(ctx.ctx, width=width, height=height, reverse_z=True)
    inv_proj = mat4_inv(proj)

    depth_tex = ctx.ctx.depth_texture((width, height))
    depth_tex.filter = (0, 0)
    depth_tex.write(depths.tobytes())

    class _FrameCtx:
        inv_proj_mat = inv_proj

    class _GraphCtx:
        resources = {}
        frame_context = _FrameCtx()

    pass_instance.execute(_GraphCtx(), depth_texture=depth_tex)

    levels = []
    for mip in range(pass_instance.num_mips):
        raw = pass_instance.hiz_bounds_texture.read(level=mip)
        mw, mh = pass_instance.mip_sizes[mip]
        levels.append(np.frombuffer(raw, dtype=np.float32).reshape((mh, mw, 2)).astype(np.float64))

    depth_tex.release()
    pass_instance.destroy()
    return levels


def _assert_chain_matches(gpu_levels, cpu_levels, tol=1e-3):
    assert len(gpu_levels) == len(cpu_levels)
    for mip, (gpu, cpu) in enumerate(zip(gpu_levels, cpu_levels)):
        assert gpu.shape == cpu.shape, f"mip {mip} shape {gpu.shape} != {cpu.shape}"
        gpu_n, gpu_f = gpu[..., 0], gpu[..., 1]
        cpu_n, cpu_f = cpu[..., 0], cpu[..., 1]

        # Empty cells must agree exactly on the sentinel pair.
        gpu_empty = gpu_n > gpu_f
        cpu_empty = cpu_n > cpu_f
        assert np.array_equal(gpu_empty, cpu_empty), f"mip {mip} empty mask mismatch"

        occupied = ~cpu_empty
        if np.any(occupied):
            assert np.allclose(gpu_n[occupied], cpu_n[occupied], rtol=1e-5, atol=tol), f"mip {mip} nearest"
            assert np.allclose(gpu_f[occupied], cpu_f[occupied], rtol=1e-5, atol=tol), f"mip {mip} farthest"

        # No negative, NaN or Inf bounds anywhere.
        assert np.all(np.isfinite(gpu_n))
        assert np.all(np.isfinite(gpu_f))
        assert np.all(gpu_n >= 0.0)
        assert np.all(gpu_f >= 0.0)


class TestHiZBoundsHierarchy:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=64, height=64)
        yield ctx
        ctx.destroy()

    def test_all_empty(self, render_ctx: RenderContext):
        width, height = 8, 8
        depths, proj = _build_depth_image(width, height, [[None] * width for _ in range(height)])
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        for mip, level in enumerate(gpu):
            assert np.all(level[..., 0] > level[..., 1]), f"mip {mip} should be empty"

    def test_all_occupied(self, render_ctx: RenderContext):
        width, height = 8, 8
        occupied = [[-(5.0 + x + y) for x in range(width)] for y in range(height)]
        depths, proj = _build_depth_image(width, height, occupied)
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        # Mip zero: nearest == farthest for every pixel.
        assert np.allclose(gpu[0][..., 0], gpu[0][..., 1], atol=1e-4)
        assert np.all(gpu[0][..., 0] > 0.0)

        cpu = _cpu_mip_chain(gpu[0])
        _assert_chain_matches(gpu, cpu)

    def test_mixed_occupied_and_empty(self, render_ctx: RenderContext):
        width, height = 8, 8
        occupied = [[None] * width for _ in range(height)]
        for y in range(4):
            for x in range(4):
                occupied[y][x] = -(3.0 + x)
        depths, proj = _build_depth_image(width, height, occupied)
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        cpu = _cpu_mip_chain(gpu[0])
        _assert_chain_matches(gpu, cpu)

        # A mixed cell that contains geometry stays occupied.
        assert gpu[1][0, 0, 0] <= gpu[1][0, 0, 1]
        # A cell with no geometry at all is empty.
        assert gpu[1][3, 3, 0] > gpu[1][3, 3, 1]

    def test_near_and_far_ordering(self, render_ctx: RenderContext):
        width, height = 8, 8
        occupied = [[None] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                occupied[y][x] = -2.0 if x < 4 else -60.0
        depths, proj = _build_depth_image(width, height, occupied)
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        near_vals = gpu[0][:, :4, 0]
        far_vals = gpu[0][:, 4:, 0]
        assert np.all(near_vals < far_vals)
        assert np.allclose(near_vals, 2.0, atol=1e-3)
        assert np.allclose(far_vals, 60.0, atol=1e-2)

    @pytest.mark.parametrize("width,height", [(8, 8), (7, 5), (1, 6), (6, 1), (1, 1)])
    def test_odd_and_degenerate_sizes(self, render_ctx: RenderContext, width, height):
        occupied = [[None] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if (x + y) % 3 != 0:
                    occupied[y][x] = -(4.0 + x * 2.0 + y * 3.0)
        depths, proj = _build_depth_image(width, height, occupied)
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        cpu = _cpu_mip_chain(gpu[0])
        _assert_chain_matches(gpu, cpu)

    def test_span_grows_with_mip_level(self, render_ctx: RenderContext):
        width, height = 8, 8
        occupied = [[None] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                occupied[y][x] = -(2.0 + x * 4.0)
        depths, proj = _build_depth_image(width, height, occupied)
        gpu = _run_gpu(render_ctx, width, height, depths, proj)

        # Mip zero span is zero for a single surface per pixel.
        assert np.allclose(gpu[0][..., 1] - gpu[0][..., 0], 0.0, atol=1e-4)
        # Coarser levels must be at least as wide as their children.
        for mip in range(1, len(gpu)):
            assert np.all(gpu[mip][..., 1] - gpu[mip][..., 0] >= -1e-4)
        # The root spans the whole scene.
        root = gpu[-1][0, 0]
        assert root[1] - root[0] > 20.0
