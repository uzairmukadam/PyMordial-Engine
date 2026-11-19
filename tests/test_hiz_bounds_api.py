"""Validation tests for the shared Hi-Z bounds API (shaders/common/hiz_bounds.glsl)."""

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.shader_utils import load_shader

EMPTY_NEAREST = 1e30
EMPTY_FARTHEST = 0.0


class TestHiZBoundsAPI:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=64, height=64)
        yield ctx
        ctx.destroy()

    def _evaluate(self, ctx, cell_bounds, ray_intervals, thickness=0.0):
        """Runs the bounds test shader for one cell and a list of ray intervals."""
        hiz = ctx.ctx.texture((4, 4), 2, dtype="f4")
        hiz.filter = (0, 0)
        data = np.zeros((4, 4, 2), dtype=np.float32)
        data[0, 0] = cell_bounds
        hiz.write(data.tobytes())

        count = len(ray_intervals)
        out = ctx.ctx.texture((count, 1), 4, dtype="f4")
        out.bind_to_image(0, read=False, write=True)

        prog = ctx.ctx.compute_shader(load_shader("hiz_bounds_test.comp"))
        prog["u_TestCount"].value = count
        prog["u_Thickness"].value = float(thickness)
        prog["u_Cell"].value = (0, 0)
        prog["u_Mip"].value = 0
        intervals = np.zeros((64, 2), dtype=np.float32)
        intervals[:count] = np.asarray(ray_intervals, dtype=np.float32)
        prog["u_RayIntervals"].write(intervals.tobytes())

        hiz.use(location=0)
        prog.run(group_x=count, group_y=1)

        result = np.frombuffer(out.read(), dtype=np.float32).reshape((count, 4))
        prog.release()
        out.release()
        hiz.release()
        return result

    def test_empty_cell_never_overlaps(self, render_ctx: RenderContext):
        rays = [(0.0, 1000.0), (5.0, 5.0), (-10.0, 10.0), (1e29, 1e30)]
        res = self._evaluate(render_ctx, (EMPTY_NEAREST, EMPTY_FARTHEST), rays)
        assert np.all(res[:, 0] == 0.0)
        assert np.all(res[:, 3] == 1.0)

    def test_ray_entirely_in_front(self, render_ctx: RenderContext):
        # Geometry at 10..12; ray at 1..5 is nearer than the geometry.
        res = self._evaluate(render_ctx, (10.0, 12.0), [(1.0, 5.0)])
        assert res[0, 0] == 0.0

    def test_ray_entirely_behind(self, render_ctx: RenderContext):
        res = self._evaluate(render_ctx, (10.0, 12.0), [(20.0, 30.0)])
        assert res[0, 0] == 0.0

    def test_partial_overlap(self, render_ctx: RenderContext):
        res = self._evaluate(render_ctx, (10.0, 12.0), [(8.0, 11.0), (11.0, 20.0)])
        assert res[0, 0] == 1.0
        assert res[1, 0] == 1.0

    def test_exact_boundary_contact(self, render_ctx: RenderContext):
        # Closed intervals: touching an endpoint counts as overlap.
        res = self._evaluate(render_ctx, (10.0, 12.0), [(5.0, 10.0), (12.0, 20.0)])
        assert res[0, 0] == 1.0
        assert res[1, 0] == 1.0

    def test_reversed_ray_interval(self, render_ctx: RenderContext):
        res = self._evaluate(render_ctx, (10.0, 12.0), [(11.0, 8.0)])
        assert res[0, 0] == 1.0

    def test_thickness_only_overlap(self, render_ctx: RenderContext):
        # Ray ends 0.5 in front of the geometry; only thickness can bridge it.
        no_thickness = self._evaluate(render_ctx, (10.0, 12.0), [(5.0, 9.5)], thickness=0.0)
        with_thickness = self._evaluate(render_ctx, (10.0, 12.0), [(5.0, 9.5)], thickness=1.0)
        assert no_thickness[0, 0] == 0.0
        assert with_thickness[0, 0] == 1.0

    def test_loaded_bounds_round_trip(self, render_ctx: RenderContext):
        res = self._evaluate(render_ctx, (3.5, 7.25), [(0.0, 1.0)])
        assert res[0, 1] == pytest.approx(3.5, abs=1e-5)
        assert res[0, 2] == pytest.approx(7.25, abs=1e-5)
        assert res[0, 3] == 0.0
