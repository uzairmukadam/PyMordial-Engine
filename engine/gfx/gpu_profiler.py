"""Non-blocking GPU pass profiler for PyMordial Engine.

Wraps OpenGL ``GL_TIMESTAMP`` query counters around named render regions and
reports the elapsed GPU time in milliseconds.

Why timestamps instead of ``GL_TIME_ELAPSED``: elapsed-time queries cannot be
nested or overlapped (only one may be active at a time), while timestamp
counters are independent query objects that simply record a point in the GPU
command stream. That lets a coarse region such as ``SSGI`` contain finer
regions such as ``SSGI trace`` without any special casing.

Results are harvested from a small ring of query frames: a query is never read
in the frame that created it, so the CPU never blocks on the GPU and no
``glFinish`` is required. A region that is not recorded in a frame (because the
pass was disabled) reports ``0.0`` instead of a stale value.

Usage::

    profiler.begin_frame()

    with profiler.region("HiZ"):
        hiz_pass.execute(ctx)

    profiler.begin("SSR")
    ssr_pass.execute(ctx)
    profiler.end()

    profiler.end_frame()

    ms = profiler.get_ms("HiZ")          # None until the ring has wrapped
    for name, ms in profiler.results().items():
        ...

Query objects are allocated once per region and reused for the lifetime of the
profiler; nothing is created or destroyed per frame.
"""

from __future__ import annotations

import ctypes

import moderngl

from engine.logging import log_warn

try:
    import OpenGL.GL as gl
except ImportError:  # pragma: no cover - PyOpenGL is a hard runtime dependency
    gl = None


class _GpuRegion:
    """Reusable context manager for one named GPU timing region."""

    __slots__ = ("_profiler", "_name")

    def __init__(self, profiler: GpuPassProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name

    def __enter__(self) -> _GpuRegion:
        self._profiler.begin(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._profiler.end()
        return False


class GpuPassProfiler:
    """Collects per-region GPU timings from a ring of timestamp query frames.

    Args:
        ctx: ModernGL context the queries belong to.
        frame_lag: Number of query frames in the ring. Must be >= 2 so a query
            is always read at least one frame after it was issued.
        enabled: Master switch. When False every call is a no-op and no query
            objects are ever created.
    """

    __slots__ = (
        "ctx",
        "enabled",
        "supported",
        "frame_lag",
        "_gl",
        "_slots",
        "_recorded",
        "_write_slot",
        "_frame_open",
        "_open_names",
        "_open_pairs",
        "_results",
        "_region_objects",
        "_u64_ok",
        "disabled_reason",
    )

    def __init__(self, ctx: moderngl.Context, frame_lag: int = 3, enabled: bool = True) -> None:
        self.ctx = ctx
        self.enabled = bool(enabled)
        self.supported = gl is not None
        self.frame_lag = max(2, int(frame_lag))
        self._gl = gl
        # One dict per ring slot: region name -> [start_query_id, end_query_id].
        self._slots: list[dict[str, list[int]]] = [{} for _ in range(self.frame_lag)]
        # Regions actually recorded in each slot this frame.
        self._recorded: list[set[str]] = [set() for _ in range(self.frame_lag)]
        self._write_slot = 0
        self._frame_open = False
        self._open_names: list[str] = []
        self._open_pairs: list[list[int]] = []
        self._results: dict[str, float] = {}
        self._region_objects: dict[str, _GpuRegion] = {}
        # Some PyOpenGL builds ship a broken glGetQueryObjectui64v converter, so
        # the 64-bit read is attempted once and then latched to whichever path
        # actually works on this driver.
        self._u64_ok: bool | None = None
        self.disabled_reason: str | None = None

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Harvests the oldest ring slot and opens a new query frame."""
        if not self.enabled or not self.supported:
            return
        # A previous frame that raised before end_frame() must not leave a
        # dangling open region behind.
        if self._frame_open:
            self.end_frame()
        self._harvest(self._write_slot)
        self._recorded[self._write_slot].clear()
        self._frame_open = True

    def end_frame(self) -> None:
        """Closes any still-open region and advances the ring cursor."""
        if not self._frame_open:
            return
        while self._open_names:
            self.end()
        self._frame_open = False
        self._write_slot = (self._write_slot + 1) % self.frame_lag

    # ------------------------------------------------------------------
    # Region API
    # ------------------------------------------------------------------

    def begin(self, name: str) -> None:
        """Records the start timestamp of a named GPU region."""
        if not self._frame_open or not self.enabled or not self.supported:
            return
        slot = self._slots[self._write_slot]
        pair = slot.get(name)
        if pair is None:
            pair = self._allocate_pair()
            if pair is None:
                return
            slot[name] = pair
            if name not in self._results:
                self._results[name] = 0.0
        try:
            self._gl.glQueryCounter(pair[0], self._gl.GL_TIMESTAMP)
        except Exception as exc:
            self._disable(f"GPU timestamp queries unavailable: {exc}")
            return
        self._open_names.append(name)
        self._open_pairs.append(pair)
        self._recorded[self._write_slot].add(name)

    def end(self) -> None:
        """Records the end timestamp of the most recently begun region."""
        if not self._open_names:
            return
        self._open_names.pop()
        pair = self._open_pairs.pop()
        try:
            self._gl.glQueryCounter(pair[1], self._gl.GL_TIMESTAMP)
        except Exception as exc:
            self._disable(f"GPU timestamp query failed: {exc}")

    def region(self, name: str) -> _GpuRegion:
        """Returns a cached, allocation-free context manager for ``name``."""
        obj = self._region_objects.get(name)
        if obj is None:
            obj = _GpuRegion(self, name)
            self._region_objects[name] = obj
        return obj

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_ms(self, name: str) -> float | None:
        """Returns the last harvested GPU time for ``name`` in milliseconds."""
        return self._results.get(name)

    def results(self) -> dict[str, float]:
        """Returns the live ``region name -> milliseconds`` mapping."""
        return self._results

    def total_ms(self) -> float:
        """Sums every reported region (top-level regions do not overlap)."""
        total = 0.0
        for value in self._results.values():
            total += value
        return total

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        """No-op: timestamp queries are resolution independent.

        Present so the pipeline can forward resize events without special
        casing, and so a future implementation that caches per-size state has a
        hook. Query objects are never reallocated on resize, so nothing leaks.
        """

    def destroy(self) -> None:
        """Releases every query object while the GL context is still current."""
        if self._gl is None:
            return
        ids: list[int] = []
        for slot in self._slots:
            for pair in slot.values():
                ids.append(pair[0])
                ids.append(pair[1])
            slot.clear()
        for recorded in self._recorded:
            recorded.clear()
        self._open_names.clear()
        self._open_pairs.clear()
        self._results.clear()
        self._region_objects.clear()
        self._frame_open = False
        if ids:
            try:
                self._gl.glDeleteQueries(len(ids), ids)
            except Exception as exc:  # pragma: no cover - driver teardown
                log_warn("GpuPassProfiler", f"Failed to release GPU timestamp queries: {exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _allocate_pair(self) -> list[int] | None:
        try:
            generated = self._gl.glGenQueries(2)
            return [int(generated[0]), int(generated[1])]
        except Exception as exc:
            self._disable(f"GPU timestamp queries unavailable: {exc}")
            return None

    def _harvest(self, slot_index: int) -> None:
        """Reads finished results from one ring slot without blocking."""
        slot = self._slots[slot_index]
        if not slot:
            return
        recorded = self._recorded[slot_index]
        gl_api = self._gl
        for name, pair in slot.items():
            if name not in recorded:
                # The pass was disabled this frame: report zero, not a stale value.
                self._results[name] = 0.0
                continue
            try:
                if not gl_api.glGetQueryObjectuiv(pair[1], gl_api.GL_QUERY_RESULT_AVAILABLE):
                    # Not ready yet; keep the previous value and try again next wrap.
                    continue
                start_ns = self._query_result(pair[0])
                end_ns = self._query_result(pair[1])
            except Exception as exc:
                self._disable(f"GPU timestamp query read failed: {exc}")
                return
            self._results[name] = max(0.0, (end_ns - start_ns) / 1_000_000.0)

    def _query_result(self, query_id: int) -> int:
        """Reads a finished timestamp counter, preferring the 64-bit path."""
        if self._u64_ok is not False:
            try:
                out = ctypes.c_uint64(0)
                self._gl.glGetQueryObjectui64v(int(query_id), self._gl.GL_QUERY_RESULT, ctypes.byref(out))
                self._u64_ok = True
                return int(out.value)
            except Exception:
                self._u64_ok = False
        # 32-bit fallback wraps after ~4.29 s, which is far beyond any region.
        return int(self._gl.glGetQueryObjectuiv(int(query_id), self._gl.GL_QUERY_RESULT))

    def _disable(self, reason: str) -> None:
        self.enabled = False
        self.supported = False
        self.disabled_reason = reason
        self._open_names.clear()
        self._open_pairs.clear()
        self._frame_open = False
        log_warn("GpuPassProfiler", reason)


__all__ = ["GpuPassProfiler"]
