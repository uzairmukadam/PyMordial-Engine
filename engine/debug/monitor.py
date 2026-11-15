"""System monitoring, CPU/GPU profiling, and renderer resource diagnostics."""

from __future__ import annotations
import time
from typing import TYPE_CHECKING, Any
import numpy as np

from engine.logging import log_warn

if TYPE_CHECKING:
    from engine.core.ecs import EntityManager
    from engine.gfx.pipeline import RenderPipeline
    from engine.gfx.mega_buffer import MegaBuffer
    from engine.assets.vfs import VFS


class ProfileScope:
    """Reusable CPU stage timer."""

    __slots__ = ("name", "monitor", "start_ns")

    def __init__(self, name: str, monitor: SystemMonitor) -> None:
        self.name = name
        self.monitor = monitor
        self.start_ns = 0

    def __enter__(self) -> ProfileScope:
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_us = (time.perf_counter_ns() - self.start_ns) / 1000.0
        self.monitor.record_stage(self.name, elapsed_us)


class SystemMonitor:
    """Collects rolling frame metrics, stage timings, and live renderer statistics."""

    GPU_QUERY_COUNT = 4
    STUTTER_THRESHOLD_MS = 50.0

    __slots__ = (
        "history_size",
        "frame_times_ms",
        "_ordered_frame_times",
        "_sorted_frame_times",
        "frame_idx",
        "total_frames",
        "last_frame_time",
        "frame_budget_ms",
        "stage_times_us",
        "stage_averages_us",
        "stage_max_us",
        "fps",
        "avg_fps",
        "one_percent_low_fps",
        "point_one_percent_low_fps",
        "avg_frame_time_ms",
        "p95_frame_time_ms",
        "p99_frame_time_ms",
        "min_frame_time_ms",
        "max_frame_time_ms",
        "over_budget_percent",
        "stutter_count",
        "gpu_frame_time_ms",
        "gpu_timing_supported",
        "_gpu_gl",
        "_gpu_queries",
        "_gpu_pending",
        "_gpu_query_cursor",
        "_gpu_active_query",
        "_gpu_timing_disabled",
        "_scope_instances",
        "runtime_stats",
    )

    def __init__(self, history_size: int = 1000, target_fps: float = 60.0) -> None:
        self.history_size = max(1, int(history_size))
        self.frame_times_ms = np.zeros(self.history_size, dtype=np.float32)
        self._ordered_frame_times = np.zeros(self.history_size, dtype=np.float32)
        self._sorted_frame_times = np.zeros(self.history_size, dtype=np.float32)
        self.frame_idx = 0
        self.total_frames = 0
        self.last_frame_time = time.perf_counter()
        self.frame_budget_ms = 1000.0 / target_fps if target_fps > 0 else 16.6667

        self.stage_times_us: dict[str, float] = {}
        self.stage_averages_us: dict[str, float] = {}
        self.stage_max_us: dict[str, float] = {}

        self.fps = 60.0
        self.avg_fps = 60.0
        self.one_percent_low_fps = 60.0
        self.point_one_percent_low_fps = 60.0
        self.avg_frame_time_ms = self.frame_budget_ms
        self.p95_frame_time_ms = self.frame_budget_ms
        self.p99_frame_time_ms = self.frame_budget_ms
        self.min_frame_time_ms = self.frame_budget_ms
        self.max_frame_time_ms = self.frame_budget_ms
        self.over_budget_percent = 0.0
        self.stutter_count = 0

        self.gpu_frame_time_ms: float | None = None
        self.gpu_timing_supported: bool | None = None
        self._gpu_gl: Any | None = None
        self._gpu_queries: list[int] = [0] * self.GPU_QUERY_COUNT
        self._gpu_pending: list[bool] = [False] * self.GPU_QUERY_COUNT
        self._gpu_query_cursor = 0
        self._gpu_active_query: int | None = None
        self._gpu_timing_disabled = False

        self._scope_instances: dict[str, ProfileScope] = {}
        self.runtime_stats: dict[str, int | float | str] = {
            "active_entities": 0,
            "entity_capacity": 0,
            "visible_entities": 0,
            "culled_entities": 0,
            "draw_batches": 0,
            "draw_commands": 0,
            "triangles": 0,
            "point_lights": 0,
            "spot_lights": 0,
            "mesh_vertices": 0,
            "mesh_indices": 0,
            "mesh_mib": 0.0,
            "material_layers": 0,
            "material_capacity": 0,
            "material_resolution": 0,
            "material_mib": 0.0,
            "tracked_gpu_mib": 0.0,
            "render_width": 0,
            "render_height": 0,
            "quality": "",
            "gi_mode": "",
            "shadow_resolution": 0,
            "shadow_cascades": 0,
        }

    def scope(self, stage_name: str) -> ProfileScope:
        """Returns a reusable timing scope for a named CPU stage."""
        if stage_name not in self._scope_instances:
            self._scope_instances[stage_name] = ProfileScope(stage_name, self)
        return self._scope_instances[stage_name]

    def record_stage(self, stage_name: str, elapsed_us: float) -> None:
        """Records the latest, smoothed average, and peak CPU time for a stage."""
        elapsed = max(0.0, float(elapsed_us))
        self.stage_times_us[stage_name] = elapsed
        previous = self.stage_averages_us.get(stage_name, elapsed)
        self.stage_averages_us[stage_name] = previous * 0.90 + elapsed * 0.10
        self.stage_max_us[stage_name] = max(self.stage_max_us.get(stage_name, 0.0), elapsed)

    def update(self, dt: float = 0.0) -> None:
        """Samples the wall-clock interval between rendered frames."""
        self.begin_frame()

    def record_fps(self, fps: float) -> None:
        """Records an externally measured FPS value."""
        if fps > 0.0:
            self.fps = fps

    def set_target_fps(self, target_fps: float) -> None:
        """Updates the frame budget used for hitch and budget-overrun metrics."""
        if target_fps > 0.0:
            self.frame_budget_ms = 1000.0 / float(target_fps)

    def begin_frame(self) -> None:
        """Samples frame time and periodically refreshes rolling percentiles."""
        now = time.perf_counter()
        dt_ms = max(0.001, min(1000.0, (now - self.last_frame_time) * 1000.0))
        self.last_frame_time = now
        self.record_frame_time(dt_ms)

    def record_frame_time(self, frame_time_ms: float) -> None:
        """Adds a deterministic frame-time sample to the rolling window."""
        dt_ms = max(0.001, min(1000.0, float(frame_time_ms)))
        self.frame_times_ms[self.frame_idx] = dt_ms
        self.frame_idx = (self.frame_idx + 1) % self.history_size
        self.total_frames += 1
        self.fps = 1000.0 / dt_ms

        if self.total_frames < 10 or self.total_frames % 10 == 0:
            active_count = min(self.total_frames, self.history_size)
            active_window = self.frame_history_view()
            self.avg_frame_time_ms = float(np.mean(active_window))
            self.min_frame_time_ms = float(np.min(active_window))
            self.max_frame_time_ms = float(np.max(active_window))
            sorted_window = self._sorted_frame_times[:active_count]
            np.copyto(sorted_window, active_window)
            sorted_window.sort()
            self.p95_frame_time_ms = self._percentile(sorted_window, active_count, 0.95)
            self.p99_frame_time_ms = self._percentile(sorted_window, active_count, 0.99)
            p99_9_time = self._percentile(sorted_window, active_count, 0.999)
            self.avg_fps = 1000.0 / self.avg_frame_time_ms
            self.one_percent_low_fps = 1000.0 / max(self.p99_frame_time_ms, 0.001)
            self.point_one_percent_low_fps = 1000.0 / max(p99_9_time, 0.001)
            self.over_budget_percent = float(np.count_nonzero(active_window > self.frame_budget_ms)) * 100.0 / active_count
            self.stutter_count = int(np.count_nonzero(active_window >= self.STUTTER_THRESHOLD_MS))

    @staticmethod
    def _percentile(sorted_values: np.ndarray, count: int, percentile: float) -> float:
        index = (count - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, count - 1)
        fraction = index - lower
        return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)

    def frame_history_view(self) -> np.ndarray:
        """Returns the active frame-time ring in chronological order using preallocated storage."""
        count = min(self.total_frames, self.history_size)
        if count == 0:
            return self._ordered_frame_times[:0]
        start = (self.frame_idx - count) % self.history_size
        first_count = min(count, self.history_size - start)
        self._ordered_frame_times[:first_count] = self.frame_times_ms[start:start + first_count]
        if first_count < count:
            self._ordered_frame_times[first_count:count] = self.frame_times_ms[:count - first_count]
        return self._ordered_frame_times[:count]

    def begin_gpu_frame(self) -> bool:
        """Starts an asynchronous GPU timer query, if available and a query slot is free."""
        if self._gpu_timing_disabled:
            return False
        try:
            if self._gpu_gl is None:
                from OpenGL import GL

                self._gpu_gl = GL
                generated = GL.glGenQueries(self.GPU_QUERY_COUNT)
                self._gpu_queries = [int(query_id) for query_id in generated]
                self.gpu_timing_supported = True

            gl = self._gpu_gl
            for index, query_id in enumerate(self._gpu_queries):
                if self._gpu_pending[index] and gl.glGetQueryObjectiv(
                    query_id, gl.GL_QUERY_RESULT_AVAILABLE
                ):
                    elapsed_ns = int(gl.glGetQueryObjectuiv(query_id, gl.GL_QUERY_RESULT))
                    self.gpu_frame_time_ms = elapsed_ns / 1_000_000.0
                    self._gpu_pending[index] = False

            for offset in range(self.GPU_QUERY_COUNT):
                index = (self._gpu_query_cursor + offset) % self.GPU_QUERY_COUNT
                if not self._gpu_pending[index]:
                    gl.glBeginQuery(gl.GL_TIME_ELAPSED, self._gpu_queries[index])
                    self._gpu_active_query = index
                    return True
            return False
        except Exception as exc:
            self._disable_gpu_timing(f"GPU timer queries unavailable: {exc}")
            return False

    def end_gpu_frame(self) -> None:
        """Ends the active GPU timer query without waiting for its result."""
        if self._gpu_active_query is None:
            return
        try:
            self._gpu_gl.glEndQuery(self._gpu_gl.GL_TIME_ELAPSED)
            self._gpu_pending[self._gpu_active_query] = True
            self._gpu_query_cursor = (self._gpu_active_query + 1) % self.GPU_QUERY_COUNT
            self._gpu_active_query = None
        except Exception as exc:
            self._disable_gpu_timing(f"GPU timer query failed: {exc}")

    def _disable_gpu_timing(self, reason: str) -> None:
        if self._gpu_active_query is not None and self._gpu_gl is not None:
            try:
                self._gpu_gl.glEndQuery(self._gpu_gl.GL_TIME_ELAPSED)
            except Exception:
                pass
        self._gpu_active_query = None
        self._gpu_timing_disabled = True
        self.gpu_timing_supported = False
        log_warn("SystemMonitor", reason)

    def update_runtime_stats(
        self,
        ecs: EntityManager,
        pipeline: RenderPipeline,
        point_lights: int = 0,
        spot_lights: int = 0,
        draw_batches: int = 0,
        quality: str = "custom",
    ) -> None:
        """Snapshots scene, visibility, draw, and known GPU resource statistics."""
        stats = self.runtime_stats
        stats["active_entities"] = ecs.active_count
        stats["entity_capacity"] = ecs.max_entities
        stats["visible_entities"] = pipeline.stats_visible_entities
        stats["culled_entities"] = pipeline.stats_culled_entities
        stats["draw_batches"] = draw_batches
        stats["draw_commands"] = pipeline.mdi.command_count + pipeline.tess_mdi.command_count
        stats["point_lights"] = point_lights
        stats["spot_lights"] = spot_lights

        triangles = 0
        for batcher in (pipeline.mdi, pipeline.tess_mdi):
            commands = batcher.command_buffer[:batcher.command_count]
            for command in commands:
                triangles += int(command["count"]) * int(command["instance_count"]) // 3
        stats["triangles"] = triangles

        mega_buffer = pipeline.mega_buffer
        stats["mesh_vertices"] = mega_buffer.total_vertices
        stats["mesh_indices"] = mega_buffer.total_indices
        mesh_bytes = mega_buffer.total_vertices * 32 + mega_buffer.total_indices * 4
        stats["mesh_mib"] = mesh_bytes / (1024.0 * 1024.0)

        atlas = pipeline.texture_atlas
        stats["material_layers"] = atlas.layer_count
        stats["material_capacity"] = atlas.max_layers
        stats["material_resolution"] = atlas.width
        atlas_bytes = atlas.width * atlas.height * atlas.max_layers * 13 * 4 // 3
        stats["material_mib"] = atlas_bytes / (1024.0 * 1024.0)

        width, height = pipeline.ctx_wrapper.width, pipeline.ctx_wrapper.height
        stats["render_width"] = width
        stats["render_height"] = height
        stats["quality"] = quality.upper()
        stats["gi_mode"] = str(pipeline.config.gi_mode)
        stats["shadow_resolution"] = pipeline.csm.atlas_size
        stats["shadow_cascades"] = pipeline.csm.cascade_count

        gbuffer_bytes = width * height * 36
        shadow_bytes = pipeline.csm.atlas_size**2 * 4
        spot_shadow_bytes = pipeline.spot_shadow_map.atlas_size**2 * 4
        stats["tracked_gpu_mib"] = (
            mesh_bytes + atlas_bytes + gbuffer_bytes + shadow_bytes + spot_shadow_bytes
        ) / (1024.0 * 1024.0)

    def get_resource_summary(
        self,
        ecs: EntityManager | None = None,
        mega_buffer: MegaBuffer | None = None,
        vfs: VFS | None = None,
        mdi_batches: int = 0,
    ) -> dict[str, str | int | float]:
        """Returns a compact resource summary suitable for logs and debug UI."""
        summary: dict[str, str | int | float] = {
            "fps": round(self.fps, 1),
            "avg_fps": round(self.avg_fps, 1),
            "one_percent_low": round(self.one_percent_low_fps, 1),
            "frame_time_ms": round(self.avg_frame_time_ms, 2),
            "mdi_batches": mdi_batches,
        }
        if ecs is not None:
            summary["entities_active"] = ecs.active_count
            summary["entities_capacity"] = ecs.max_entities
        if mega_buffer is not None:
            summary["mega_verts"] = mega_buffer.total_vertices
            summary["mega_indices"] = mega_buffer.total_indices
            summary["mega_vram_kb"] = round(
                (mega_buffer.total_vertices * 32 + mega_buffer.total_indices * 4) / 1024.0, 1
            )
        if vfs is not None:
            summary["vfs_files"] = len(vfs.list_files())
        return summary

    def destroy(self) -> None:
        """Releases OpenGL timer queries while their context is current."""
        self.end_gpu_frame()
        if self._gpu_gl is not None and any(self._gpu_queries):
            try:
                self._gpu_gl.glDeleteQueries(len(self._gpu_queries), self._gpu_queries)
            except Exception as exc:
                log_warn("SystemMonitor", f"Failed to release GPU timer queries: {exc}")
            finally:
                self._gpu_queries = [0] * self.GPU_QUERY_COUNT
