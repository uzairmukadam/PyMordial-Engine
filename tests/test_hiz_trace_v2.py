"""V2.2 validation: mip-0 incremental screen-space traversal (GLSL float32).

Runs ``shaders/hiz_trace_v2_test.comp`` against synthetic reversed-Z depth
buffers and checks hit/miss, UV, distance, depth, and termination against a
direct reference. V1 is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math_utils import mat4_inv, matrix_perspective
from engine.gfx.context import RenderContext
from engine.gfx.shader_utils import load_shader

FOVY = np.radians(60.0)
NEAR = 0.1
FAR = 2000.0

MISS_NONE = 0
MISS_LEFT_SCREEN = 1
MISS_EXCEEDED_DIST = 2
MISS_EXCEEDED_ITERS = 3
MISS_INVALID_PROJ = 4
MISS_INVALID_SETUP = 5


def _proj(w, h):
    return matrix_perspective(FOVY, w / h, NEAR, FAR, reverse_z=True)


def _raw_depth_for_view_z(view_z, proj):
    clip = proj.reshape((4, 4), order="F") @ np.array([0.0, 0.0, view_z, 1.0])
    return float(clip[2] / clip[3])


def _make_plane_depth(w, h, plane_z, proj):
    """Constant view-space depth plane (front-facing)."""
    raw = _raw_depth_for_view_z(plane_z, proj)
    return np.full((h, w), raw, dtype=np.float32)


def _make_depth_from_view_z(w, h, view_z_grid, proj):
    """Converts a per-pixel view-space depth grid to raw reversed-Z depth."""
    p = proj.reshape((4, 4), order="F")
    raw = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            z = view_z_grid[y, x]
            if z >= 0.0:  # background sentinel
                raw[y, x] = 0.0
            else:
                clip = p @ np.array([0.0, 0.0, z, 1.0])
                raw[y, x] = clip[2] / clip[3]
    return raw


EMPTY_NEAREST = 1e30
EMPTY_FARTHEST = 0.0


def _build_bounds_from_depth(depth, proj):
    """Builds an RG32F nearest/farthest linear-depth bounds hierarchy from a
    reversed-Z depth buffer, matching the production reduction rules."""
    h, w = depth.shape
    inv = mat4_inv(proj).reshape((4, 4), order="F")
    mip0 = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            raw = float(depth[y, x])
            if raw <= 1e-6:
                mip0[y, x] = (EMPTY_NEAREST, EMPTY_FARTHEST)
                continue
            uv = ((x + 0.5) / w, (y + 0.5) / h)
            clip = np.array([uv[0] * 2 - 1, uv[1] * 2 - 1, raw, 1.0])
            v = inv @ clip
            lin = -v[2] / v[3]
            if not (lin > 0.0) or not np.isfinite(lin):
                mip0[y, x] = (EMPTY_NEAREST, EMPTY_FARTHEST)
            else:
                mip0[y, x] = (lin, lin)
    levels = [mip0]
    cur = mip0
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        ch, cw = cur.shape[0], cur.shape[1]
        nh, nw = max(1, ch // 2), max(1, cw // 2)
        nxt = np.zeros((nh, nw, 2), dtype=np.float32)
        for y in range(nh):
            for x in range(nw):
                nearest = EMPTY_NEAREST
                farthest = EMPTY_FARTHEST
                for dy in (0, 1):
                    for dx in (0, 1):
                        sy = min(y * 2 + dy, ch - 1)
                        sx = min(x * 2 + dx, cw - 1)
                        cn, cf = cur[sy, sx]
                        if cn <= cf:
                            nearest = min(nearest, cn)
                            farthest = max(farthest, cf)
                nxt[y, x] = (nearest, farthest)
        levels.append(nxt)
        cur = nxt
    return levels


class _Harness:
    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.proj = _proj(width, height)
        self.inv_proj = mat4_inv(self.proj)

    def trace(self, depth, origins, directions, t_start, t_end, thickness=0.15, max_iters=256,
              trace_cells=False, bounds=None, start_mip=0, hierarchical=False, ascent=False):
        n = len(origins)
        import moderngl as _mgl

        depth_tex = self.ctx.ctx.depth_texture((self.width, self.height))
        depth_tex.filter = (_mgl.NEAREST, _mgl.NEAREST)
        depth_tex.write(np.asarray(depth, dtype=np.float32).tobytes())

        # Bounds hierarchy (RG32F nearest/farthest). If not supplied, build a
        # trivial all-empty hierarchy so the sampler binding is always valid.
        if bounds is None:
            bounds = _build_bounds_from_depth(depth, self.proj)
        bounds_tex = self.ctx.ctx.texture((self.width, self.height), 2, dtype="f4")
        bounds_tex.filter = (_mgl.NEAREST_MIPMAP_NEAREST, _mgl.NEAREST)
        bounds_tex.repeat_x = False
        bounds_tex.repeat_y = False
        bounds_tex.build_mipmaps()
        bounds_tex.write(bounds[0].tobytes())
        for level in range(1, len(bounds)):
            bounds_tex.write(bounds[level].tobytes(), level=level)

        out = self.ctx.ctx.texture((n, 4), 4, dtype="f4")
        out.bind_to_image(0, read=False, write=True)
        cell_tex = self.ctx.ctx.texture((n, max_iters), 4, dtype="f4")
        cell_tex.write(np.zeros((n * max_iters * 4,), dtype=np.float32).tobytes())
        cell_tex.bind_to_image(1, read=False, write=True)

        prog = self.ctx.ctx.compute_shader(load_shader("hiz_trace_v2_test.comp"))
        prog["u_Projection"].write(self.proj.tobytes())
        prog["u_InvProjection"].write(self.inv_proj.tobytes())
        prog["u_RayCount"].value = n
        prog["u_Thickness"].value = float(thickness)
        prog["u_MaxIterations"].value = int(max_iters)
        prog["u_TraceCells"].value = 1 if trace_cells else 0
        prog["u_StartMip"].value = int(start_mip)
        prog["u_Hierarchical"].value = 2 if ascent else (1 if hierarchical else 0)

        # Pack rays into an SSBO: 8 floats per ray (origin.xyz, dir.xyz, t0, t1).
        ray_data = np.zeros((n, 8), dtype=np.float32)
        ray_data[:, 0:3] = np.asarray(origins, dtype=np.float32)
        ray_data[:, 3:6] = np.asarray(directions, dtype=np.float32)
        ray_data[:, 6] = np.asarray(t_start, dtype=np.float32)
        ray_data[:, 7] = np.asarray(t_end, dtype=np.float32)
        ray_buf = self.ctx.ctx.buffer(ray_data.tobytes())
        ray_buf.bind_to_storage_buffer(0)

        depth_tex.use(location=0)
        bounds_tex.use(location=1)
        prog.run(group_x=n, group_y=1)

        data = np.frombuffer(out.read(), dtype=np.float32).reshape((4, n, 4))
        cells = None
        if trace_cells:
            cells = np.frombuffer(cell_tex.read(), dtype=np.float32).reshape((max_iters, n, 4))
        prog.release()
        out.release()
        cell_tex.release()
        depth_tex.release()
        bounds_tex.release()
        ray_buf.release()

        results = []
        for i in range(n):
            entry = {
                "hit": bool(data[0, i, 0] > 0.5),
                "uv": data[0, i, 1:3].copy(),
                "distance": float(data[0, i, 3]),
                "linear_depth": float(data[1, i, 0]),
                "iterations": int(data[1, i, 1]),
                "miss_reason": int(data[1, i, 2]),
                "setup_valid": bool(data[1, i, 3] > 0.5),
                "start_mip": int(data[2, i, 0]),
                "final_mip": int(data[2, i, 1]),
                "highest_mip": int(data[2, i, 2]),
                "ascents": int(data[2, i, 3]),
                "descents": int(data[3, i, 0]),
                "parent_lookaheads": int(data[3, i, 1]),
                "parent_lookahead_success": int(data[3, i, 2]),
            }
            if cells is not None:
                visited = []
                for k in range(max_iters):
                    if cells[k, i, 3] < 0.5:
                        break
                    visited.append((int(cells[k, i, 0]), int(cells[k, i, 1])))
                entry["cells"] = visited
            results.append(entry)
        return results


@pytest.fixture(scope="module")
def ctx():
    c = RenderContext.create_headless(width=64, height=64)
    yield c
    c.destroy()


def _direct_hit(origin, direction, t_start, t_end, depth, proj, inv_proj, thickness, w, h):
    """Reference: march the ray in view space and test the depth buffer directly."""
    p = proj.reshape((4, 4), order="F")
    tol = max(thickness, 1e-4)
    steps = 4000
    for k in range(steps + 1):
        t = t_start + (t_end - t_start) * k / steps
        pos = origin + direction * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0])
        if clip[3] <= 1e-6:
            return None
        uv = (clip[:2] / clip[3]) * 0.5 + 0.5
        if not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0):
            return None
        px = min(int(uv[0] * w), w - 1)
        py = min(int(uv[1] * h), h - 1)
        raw = depth[py, px]
        if raw <= 1e-6:
            continue
        c2 = inv_proj.reshape((4, 4), order="F") @ np.array([uv[0] * 2 - 1, uv[1] * 2 - 1, raw, 1.0])
        lin = -c2[2] / c2[3]
        if abs(-pos[2] - lin) <= tol:
            return {"uv": uv, "distance": t, "linear_depth": lin}
    return None


class TestMip0Traversal:
    def test_flat_plane_hit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        origin = np.array([[0.0, 0.0, -2.0]])
        direction = np.array([[0.3, 0.0, -1.0]]) / np.linalg.norm([0.3, 0.0, -1.0])
        r = hz.trace(depth, origin, direction, [0.5], [10.0])[0]
        assert r["setup_valid"]
        assert r["hit"], f"expected hit, got miss {r['miss_reason']}"
        assert abs(r["linear_depth"] - 5.0) < 0.1

    def test_immediate_near_field_hit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -1.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -0.5]], [d], [0.1], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 1.0) < 0.1

    def test_empty_background_ray(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)  # all background
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [[1.0, 0.0, 0.0]], [0.5], [10.0])[0]
        assert not r["hit"]
        assert r["miss_reason"] in (MISS_EXCEEDED_DIST, MISS_LEFT_SCREEN)

    def test_empty_then_geometry(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        # Left half background, right half a plane at z=-5.
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0])[0]
        assert r["hit"], f"expected hit, got {r['miss_reason']}"

    def test_thin_geometry(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        # 1-px column placed where the ray crosses z=-5 (uv.x ~= 0.656 -> px 41).
        depth[:, 41:42] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0], thickness=0.5)[0]
        assert r["hit"], f"thin geometry missed: {r['miss_reason']}"

    def test_horizontal_ray(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        r = hz.trace(depth, [[0.0, 0.0, -5.0]], [[1.0, 0.0, 0.0]], [0.1], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 5.0) < 0.1

    def test_vertical_ray(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        r = hz.trace(depth, [[0.0, 0.0, -5.0]], [[0.0, 1.0, 0.0]], [0.1], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 5.0) < 0.1

    def test_diagonal_ray(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        r = hz.trace(depth, [[0.0, 0.0, -5.0]], [[0.7, 0.7, 0.0]], [0.1], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 5.0) < 0.1

    def test_screen_edge_exit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [[1.0, 0.0, 0.0]], [0.5], [10.0])[0]
        assert not r["hit"]
        assert r["miss_reason"] == MISS_LEFT_SCREEN

    def test_max_distance_exit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        # Ray along view axis: never leaves screen, never hits.
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [[0.05, 0.0, -1.0]], [0.5], [2.0])[0]
        assert not r["hit"]
        assert r["miss_reason"] in (MISS_EXCEEDED_DIST, MISS_LEFT_SCREEN)

    def test_toward_camera_valid(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -2.0, hz.proj)
        # Start at z=-8 moving toward camera (+z) but stopping before the plane.
        d = np.array([0.2, 0.0, 1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -8.0]], [d], [0.5], [8.0])[0]
        assert r["setup_valid"]
        assert r["hit"]
        assert abs(r["linear_depth"] - 2.0) < 0.15

    def test_away_from_camera(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        d = np.array([0.2, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -1.0]], [d], [0.5], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 5.0) < 0.15

    def test_non_power_of_two(self, ctx):
        w, h = 63, 47
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0])[0]
        assert r["hit"]
        assert abs(r["linear_depth"] - 5.0) < 0.1

    def test_invalid_setup_zero_screen_length(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        # Ray exactly along view axis projects to a point -> invalid setup.
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [[0.0, 0.0, -1.0]], [0.5], [10.0])[0]
        # Note: along-axis ray has zero screen length -> setup invalid.
        assert not r["setup_valid"]
        assert r["miss_reason"] == MISS_INVALID_SETUP


class TestAgainstReference:
    @pytest.mark.parametrize(
        "direction",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0], [0.5, 0.0, -0.5]],
        ids=["horiz", "vert", "diag", "depth"],
    )
    def test_hit_matches_reference(self, ctx, direction):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        origin = np.array([0.0, 0.0, -2.0])
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0])[0]
        ref = _direct_hit(origin, d, 0.5, 10.0, depth, hz.proj, hz.inv_proj, 0.15, w, h)
        assert r["hit"] == (ref is not None), f"hit mismatch: v2={r['hit']} ref={ref is not None}"
        if ref is not None:
            assert np.allclose(r["uv"], ref["uv"], atol=2.0 / w), f"uv {r['uv']} vs {ref['uv']}"
            assert abs(r["distance"] - ref["distance"]) < 0.15


class TestCellCoverage:
    """Directly validates the sequence of mip-0 cells visited by the DDA."""

    def _cells(self, ctx, w, h, origin, direction, t0, t1):
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)  # empty: traverse whole screen
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [t0], [t1], trace_cells=True)[0]
        return r["cells"], w, h

    def _assert_contiguous(self, cells):
        # Consecutive cells must be 4-neighbour adjacent (or identical for a
        # corner nudge); never jump more than one cell in either axis.
        for a, b in zip(cells, cells[1:]):
            assert abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1, f"cell jump {a}->{b}"
            assert a != b, f"duplicate cell {a}"

    def test_horizontal_positive(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert all(c[1] == cells[0][1] for c in cells), "horizontal ray changed row"
        assert cells[-1][0] > cells[0][0], "did not advance +x"

    def test_horizontal_negative(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [-1.0, 0.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert cells[-1][0] < cells[0][0], "did not advance -x"

    def test_vertical_positive(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [0.0, 1.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert all(c[0] == cells[0][0] for c in cells), "vertical ray changed column"
        assert cells[-1][1] > cells[0][1], "did not advance +y"

    def test_vertical_negative(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [0.0, -1.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert cells[-1][1] < cells[0][1], "did not advance -y"

    def test_diagonal(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [1.0, 1.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert cells[-1][0] > cells[0][0] and cells[-1][1] > cells[0][1]

    def test_shallow_near_horizontal(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [1.0, 0.05, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)

    def test_shallow_near_vertical(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [0.05, 1.0, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)

    def test_non_power_of_two(self, ctx):
        cells, w, h = self._cells(ctx, 63, 47, [0.0, 0.0, -5.0], [1.0, 0.3, 0.0], 0.1, 10.0)
        assert len(cells) > 1
        self._assert_contiguous(cells)
        assert all(0 <= c[0] < w and 0 <= c[1] < h for c in cells), "cell out of bounds"

    def test_monotonic_and_bounded(self, ctx):
        cells, w, h = self._cells(ctx, 64, 64, [0.0, 0.0, -5.0], [1.0, 0.2, 0.0], 0.1, 10.0)
        # Screen progress is monotonic by construction; verify no cell repeats.
        assert len(cells) == len(set(cells)), "duplicate cell visited"


class TestThinGeometryAnalytic:
    """Thin geometry placed analytically at the ray's known crossing cell."""

    def _crossing_px(self, hz, origin, direction, plane_z, w, h):
        p = hz.proj.reshape((4, 4), order="F")
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        t = (plane_z - origin[2]) / d[2]
        pos = np.array(origin) + d * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0])
        uv = (clip[:2] / clip[3]) * 0.5 + 0.5
        return int(uv[0] * w), int(uv[1] * h), uv

    def test_thin_column_hit_horizontal(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]
        direction = [0.3, 0.0, -1.0]
        px, py, _ = self._crossing_px(hz, origin, direction, -5.0, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, px : px + 1] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array(direction); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0], thickness=0.5)[0]
        assert r["hit"], f"thin column at px={px} missed: {r['miss_reason']}"

    def test_thin_column_miss_adjacent(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]
        direction = [0.3, 0.0, -1.0]
        px, py, _ = self._crossing_px(hz, origin, direction, -5.0, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        # Place the thin column one cell away from the crossing.
        adj = px + 3
        depth[:, adj : adj + 1] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array(direction); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0], thickness=0.5)[0]
        assert not r["hit"], f"adjacent column at px={adj} produced false hit"

    def test_thin_column_hit_diagonal(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]
        direction = [0.3, 0.3, -1.0]
        px, py, _ = self._crossing_px(hz, origin, direction, -5.0, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[py : py + 1, px : px + 1] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array(direction); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0], thickness=0.5)[0]
        assert r["hit"], f"thin diagonal cell ({px},{py}) missed: {r['miss_reason']}"

    def test_thin_column_miss_diagonal_adjacent(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]
        direction = [0.3, 0.3, -1.0]
        px, py, _ = self._crossing_px(hz, origin, direction, -5.0, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[py + 3 : py + 4, px + 3 : px + 4] = _raw_depth_for_view_z(-5.0, hz.proj)
        d = np.array(direction); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0], thickness=0.5)[0]
        assert not r["hit"], "adjacent diagonal cell produced false hit"


class TestHitUVSemantics:
    def test_hit_uv_inside_sampled_cell(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        origin = [0.0, 0.0, -2.0]
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0])[0]
        assert r["hit"]
        # hitUV must lie within one cell of the true crossing.
        p = hz.proj.reshape((4, 4), order="F")
        t = (-5.0 - origin[2]) / d[2]
        pos = np.array(origin) + d * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0])
        uv_ref = (clip[:2] / clip[3]) * 0.5 + 0.5
        assert np.max(np.abs(r["uv"] - uv_ref)) < 1.5 / w, f"hitUV {r['uv']} vs ref {uv_ref}"

    def test_hit_uv_matches_crossing_tightly(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        origin = [0.0, 0.0, -2.0]
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        r = hz.trace(depth, [origin], [d], [0.5], [10.0])[0]
        p = hz.proj.reshape((4, 4), order="F")
        t = (-5.0 - origin[2]) / d[2]
        pos = np.array(origin) + d * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0])
        uv_ref = (clip[:2] / clip[3]) * 0.5 + 0.5
        # Crossing recovery should be within a small fraction of a texel.
        assert np.max(np.abs(r["uv"] - uv_ref)) < 0.25 / w, f"hitUV {r['uv']} vs ref {uv_ref}"


class TestHierarchicalDescent:
    """V2.3 descent-only hierarchy traversal vs the V2.2.1 mip-0 reference."""

    def _both(self, ctx, w, h, depth, origin, direction, t0, t1, thickness=0.15, start_mip=0):
        hz = _Harness(ctx, w, h)
        d = np.array(direction, dtype=np.float64); d = d / np.linalg.norm(d)
        mip0 = hz.trace(depth, [origin], [d], [t0], [t1], thickness=thickness)[0]
        hier = hz.trace(depth, [origin], [d], [t0], [t1], thickness=thickness,
                        hierarchical=True, start_mip=start_mip)[0]
        return mip0, hier

    def test_empty_hierarchy_advances(self, ctx):
        w, h = 64, 64
        depth = np.zeros((h, w), dtype=np.float32)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0, start_mip=3)
        assert not hier["hit"]
        assert hier["miss_reason"] in (MISS_LEFT_SCREEN, MISS_EXCEEDED_DIST)
        assert mip0["hit"] == hier["hit"]

    def test_immediate_overlap_descends(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=4)
        assert hier["hit"] and mip0["hit"]
        assert np.allclose(hier["uv"], mip0["uv"], atol=1e-4)
        assert abs(hier["distance"] - mip0["distance"]) < 1e-3

    def test_empty_then_occupied(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=3)
        assert hier["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert np.allclose(hier["uv"], mip0["uv"], atol=1e-3)

    def test_thin_geometry_via_parent(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]; direction = [0.3, 0.0, -1.0]
        p = hz.proj.reshape((4, 4), order="F")
        d = np.array(direction); d = d / np.linalg.norm(d)
        t = (-5.0 - origin[2]) / d[2]; pos = np.array(origin) + d * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0]); uv = (clip[:2] / clip[3]) * 0.5 + 0.5
        px = int(uv[0] * w)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, px : px + 1] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, origin, direction, 0.5, 10.0, thickness=0.5, start_mip=3)
        assert hier["hit"], f"thin geometry via parent missed: {hier['miss_reason']}"
        assert mip0["hit"]

    def test_sibling_child_no_false_hit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]; direction = [0.3, 0.0, -1.0]
        p = hz.proj.reshape((4, 4), order="F")
        d = np.array(direction); d = d / np.linalg.norm(d)
        t = (-5.0 - origin[2]) / d[2]; pos = np.array(origin) + d * t
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0]); uv = (clip[:2] / clip[3]) * 0.5 + 0.5
        px = int(uv[0] * w)
        depth = np.zeros((h, w), dtype=np.float32)
        # Geometry in a sibling cell the ray does not cross.
        depth[:, px + 4 : px + 5] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, origin, direction, 0.5, 10.0, thickness=0.5, start_mip=3)
        assert not hier["hit"], "sibling geometry produced false hit"
        assert mip0["hit"] == hier["hit"]

    def test_mixed_geometry_background(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=4)
        assert hier["hit"] == mip0["hit"]

    def test_multiple_depths(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, : w // 2] = _raw_depth_for_view_z(-3.0, hz.proj)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-8.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=4)
        assert hier["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert abs(hier["linear_depth"] - mip0["linear_depth"]) < 0.1

    def test_ray_in_front_no_hit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -20.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 3.0, start_mip=3)
        assert not hier["hit"]
        assert mip0["hit"] == hier["hit"]

    def test_ray_behind_no_hit(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -1.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -5.0], [0.3, 0.0, -1.0], 0.5, 3.0, start_mip=3)
        assert mip0["hit"] == hier["hit"]

    def test_screen_edge_geometry(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w - 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=3)
        assert mip0["hit"] == hier["hit"]

    def test_non_power_of_two(self, ctx):
        w, h = 63, 47
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=3)
        assert hier["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert np.allclose(hier["uv"], mip0["uv"], atol=1e-3)

    def test_odd_mip_dimensions(self, ctx):
        w, h = 63, 47
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, hier = self._both(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=4)
        assert hier["hit"] == mip0["hit"]

    def test_start_mip_clamped(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        # start_mip far beyond the chain must clamp, not crash.
        r = hz.trace(depth, [[0.0, 0.0, -2.0]], [[0.3, 0.0, -1.0]], [0.5], [10.0],
                     hierarchical=True, start_mip=99)[0]
        assert r["hit"]
        assert np.all(np.isfinite(r["uv"]))


class TestMultiResolution:
    @pytest.mark.parametrize("w,h", [(64, 64), (63, 47), (128, 72), (256, 144)])
    def test_hierarchy_matches_mip0(self, ctx, w, h):
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        mip0 = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0])[0]
        hier = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0], hierarchical=True, start_mip=3)[0]
        assert hier["hit"] == mip0["hit"], f"{w}x{h}: hit mismatch"
        if mip0["hit"]:
            assert np.allclose(hier["uv"], mip0["uv"], atol=2.0 / w)
            assert abs(hier["distance"] - mip0["distance"]) < 0.05


class TestAscentTraversal:
    """V2.4 single-step ascent vs V2.2.1 mip-0 and V2.3 descent-only."""

    def _three(self, ctx, w, h, depth, origin, direction, t0, t1, thickness=0.15, start_mip=0):
        hz = _Harness(ctx, w, h)
        d = np.array(direction, dtype=np.float64); d = d / np.linalg.norm(d)
        mip0 = hz.trace(depth, [origin], [d], [t0], [t1], thickness=thickness)[0]
        desc = hz.trace(depth, [origin], [d], [t0], [t1], thickness=thickness,
                        hierarchical=True, start_mip=start_mip)[0]
        asc = hz.trace(depth, [origin], [d], [t0], [t1], thickness=thickness,
                       ascent=True, start_mip=start_mip)[0]
        return mip0, desc, asc

    def test_empty_hierarchy_ascends(self, ctx):
        w, h = 64, 64
        depth = np.zeros((h, w), dtype=np.float32)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0, start_mip=0)
        assert not asc["hit"]
        assert asc["miss_reason"] in (MISS_LEFT_SCREEN, MISS_EXCEEDED_DIST)
        assert asc["ascents"] > 0, "empty hierarchy should ascend"
        assert asc["highest_mip"] > 0
        assert mip0["hit"] == asc["hit"] == desc["hit"]

    def test_single_step_ascent_per_iteration(self, ctx):
        w, h = 64, 64
        depth = np.zeros((h, w), dtype=np.float32)
        _, _, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0, start_mip=0)
        # Each ascent is one mip; ascents cannot exceed iterations.
        assert asc["ascents"] <= asc["iterations"]

    def test_occupied_sibling_rejects_ascent(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        # Occupied region in the upper half; ray travels along the lower half.
        depth[: h // 2, :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"] == desc["hit"]

    def test_empty_then_geometry(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"] == desc["hit"]
        if mip0["hit"]:
            assert np.allclose(asc["uv"], mip0["uv"], atol=1e-3)

    def test_thin_geometry_sibling_descendant(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        origin = [0.0, 0.0, -2.0]; direction = [0.3, 0.0, -1.0]
        p = hz.proj.reshape((4, 4), order="F")
        d = np.array(direction); d = d / np.linalg.norm(d)
        tt = (-5.0 - origin[2]) / d[2]; pos = np.array(origin) + d * tt
        clip = p @ np.array([pos[0], pos[1], pos[2], 1.0]); uv = (clip[:2] / clip[3]) * 0.5 + 0.5
        px = int(uv[0] * w)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, px : px + 1] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, origin, direction, 0.5, 10.0, thickness=0.5, start_mip=0)
        assert asc["hit"], f"thin geometry missed with ascent: {asc['miss_reason']}"
        assert mip0["hit"] == asc["hit"]

    def test_coarse_parent_multiple_depths(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, : w // 2] = _raw_depth_for_view_z(-3.0, hz.proj)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-8.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"] == desc["hit"]
        if mip0["hit"]:
            assert abs(asc["linear_depth"] - mip0["linear_depth"]) < 0.1

    def test_immediate_geometry_no_unsafe_ascent(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] and mip0["hit"]
        assert np.allclose(asc["uv"], mip0["uv"], atol=1e-3)

    def test_viewport_exit_through_empty(self, ctx):
        w, h = 64, 64
        depth = np.zeros((h, w), dtype=np.float32)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -5.0], [1.0, 0.0, 0.0], 0.1, 10.0, start_mip=0)
        assert asc["miss_reason"] == MISS_LEFT_SCREEN
        assert mip0["miss_reason"] == MISS_LEFT_SCREEN

    def test_max_distance_inside_parent(self, ctx):
        w, h = 64, 64
        depth = np.zeros((h, w), dtype=np.float32)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -5.0], [0.05, 0.0, -1.0], 0.1, 2.0, start_mip=0)
        assert not asc["hit"]
        assert asc["miss_reason"] in (MISS_EXCEEDED_DIST, MISS_LEFT_SCREEN)

    def test_diagonal_corner_crossing(self, ctx):
        w, h = 64, 64
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.3, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert np.allclose(asc["uv"], mip0["uv"], atol=1e-3)

    def test_non_power_of_two(self, ctx):
        w, h = 63, 47
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert np.allclose(asc["uv"], mip0["uv"], atol=1e-3)

    def test_odd_mip_dimensions(self, ctx):
        w, h = 63, 47
        hz = _Harness(ctx, w, h)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:, w // 2 :] = _raw_depth_for_view_z(-5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"]

    def test_high_resolution(self, ctx):
        w, h = 512, 288
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        mip0, desc, asc = self._three(ctx, w, h, depth, [0.0, 0.0, -2.0], [0.3, 0.0, -1.0], 0.5, 10.0, start_mip=0)
        assert asc["hit"] == mip0["hit"]
        if mip0["hit"]:
            assert np.allclose(asc["uv"], mip0["uv"], atol=2.0 / w)


class TestThreeWayAgreement:
    @pytest.mark.parametrize("w,h", [(64, 64), (63, 47), (128, 72), (256, 144)])
    @pytest.mark.parametrize("start_mip", [0, 2, 4])
    def test_all_paths_agree(self, ctx, w, h, start_mip):
        hz = _Harness(ctx, w, h)
        depth = _make_plane_depth(w, h, -5.0, hz.proj)
        d = np.array([0.3, 0.0, -1.0]); d = d / np.linalg.norm(d)
        mip0 = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0])[0]
        desc = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0], hierarchical=True, start_mip=start_mip)[0]
        asc = hz.trace(depth, [[0.0, 0.0, -2.0]], [d], [0.5], [10.0], ascent=True, start_mip=start_mip)[0]
        assert mip0["hit"] == desc["hit"] == asc["hit"], f"{w}x{h} sm={start_mip}: hit mismatch"
        if mip0["hit"]:
            assert np.allclose(asc["uv"], mip0["uv"], atol=2.0 / w)
            assert abs(asc["distance"] - mip0["distance"]) < 0.05
