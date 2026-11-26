"""Unit tests for the non-blocking GPU pass profiler."""

import pytest

from engine.gfx.context import RenderContext
from engine.gfx.gpu_profiler import GpuPassProfiler


class TestGpuPassProfiler:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=64, height=64)
        yield ctx
        ctx.destroy()

    def test_reports_milliseconds_after_ring_delay(self, render_ctx: RenderContext):
        profiler = GpuPassProfiler(render_ctx.ctx, frame_lag=3)
        assert profiler.enabled and profiler.supported

        for _ in range(6):
            profiler.begin_frame()
            with profiler.region("Work"):
                render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
            profiler.end_frame()

        value = profiler.get_ms("Work")
        assert value is not None
        assert value >= 0.0
        profiler.destroy()

    def test_disabled_region_reports_zero(self, render_ctx: RenderContext):
        profiler = GpuPassProfiler(render_ctx.ctx, frame_lag=2)

        for _ in range(4):
            profiler.begin_frame()
            with profiler.region("Always"):
                render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
            profiler.end_frame()

        # Now stop recording the region, as a disabled pass would.
        for _ in range(4):
            profiler.begin_frame()
            profiler.end_frame()

        assert profiler.get_ms("Always") == 0.0
        profiler.destroy()

    def test_disabled_profiler_is_noop(self, render_ctx: RenderContext):
        profiler = GpuPassProfiler(render_ctx.ctx, enabled=False)
        profiler.begin_frame()
        with profiler.region("Work"):
            render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
        profiler.end_frame()
        assert profiler.results() == {}
        profiler.destroy()

    def test_resize_does_not_reallocate_queries(self, render_ctx: RenderContext):
        profiler = GpuPassProfiler(render_ctx.ctx, frame_lag=2)
        profiler.begin_frame()
        with profiler.region("Work"):
            render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
        profiler.end_frame()

        before = [id(pair) for slot in profiler._slots for pair in slot.values()]
        profiler.resize(128, 96)
        after = [id(pair) for slot in profiler._slots for pair in slot.values()]
        assert before == after
        profiler.destroy()

    def test_unbalanced_begin_is_closed_by_end_frame(self, render_ctx: RenderContext):
        profiler = GpuPassProfiler(render_ctx.ctx, frame_lag=2)
        profiler.begin_frame()
        profiler.begin("Dangling")
        profiler.end_frame()
        assert profiler._open_names == []
        profiler.destroy()

    def test_nested_regions_report_independently(self, render_ctx: RenderContext):
        """Stage 1: an enclosing region (e.g. "AMRGI total") may contain finer
        regions ("AMRGI trace"/"AMRGI reconstruct") and both report timings."""
        profiler = GpuPassProfiler(render_ctx.ctx, frame_lag=3)

        for _ in range(6):
            profiler.begin_frame()
            with profiler.region("AMRGI total"):
                with profiler.region("AMRGI trace"):
                    render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
                with profiler.region("AMRGI reconstruct"):
                    render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
            profiler.end_frame()

        for name in ("AMRGI total", "AMRGI trace", "AMRGI reconstruct"):
            value = profiler.get_ms(name)
            assert value is not None
            assert value >= 0.0
        profiler.destroy()
