"""Tier 1 Debug Subsystem: System Monitoring, Profiler & Resource Diagnostics.

Provides zero-allocation hierarchical CPU stage profiling, 120-frame rolling
frame-time histograms, 1% low calculations, and live resource gauges.
"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from engine.core.ecs import EntityManager
    from engine.gfx.mega_buffer import MegaBuffer
    from engine.assets.vfs import VFS


class ProfileScope:
    """Zero-overhead microsecond context manager for hierarchical CPU profiling."""

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
    """Coordinates performance telemetry, stage profiling, and engine resource gauges."""

    __slots__ = (
        "history_size",
        "frame_times_ms",
        "frame_idx",
        "total_frames",
        "last_frame_time",
        "stage_times_us",
        "stage_averages_us",
        "fps",
        "avg_fps",
        "one_percent_low_fps",
        "avg_frame_time_ms",
        "min_frame_time_ms",
        "max_frame_time_ms",
        "_scope_instances",
    )

    def __init__(self, history_size: int = 120) -> None:
        self.history_size = history_size
        self.frame_times_ms = np.full(history_size, 16.67, dtype=np.float32)
        self.frame_idx = 0
        self.total_frames = 0
        self.last_frame_time = time.perf_counter()

        self.stage_times_us: dict[str, float] = {}
        self.stage_averages_us: dict[str, float] = {}

        self.fps = 60.0
        self.avg_fps = 60.0
        self.one_percent_low_fps = 60.0
        self.avg_frame_time_ms = 16.67
        self.min_frame_time_ms = 16.67
        self.max_frame_time_ms = 16.67

        # Pre-allocate scope instances for common stages to eliminate runtime garbage
        self._scope_instances: dict[str, ProfileScope] = {}

    def scope(self, stage_name: str) -> ProfileScope:
        """Returns a reusable ProfileScope for the named execution block."""
        if stage_name not in self._scope_instances:
            self._scope_instances[stage_name] = ProfileScope(stage_name, self)
        return self._scope_instances[stage_name]

    def record_stage(self, stage_name: str, elapsed_us: float) -> None:
        """Records the microsecond duration for a named stage and updates smoothed averages."""
        self.stage_times_us[stage_name] = elapsed_us
        prev_avg = self.stage_averages_us.get(stage_name, elapsed_us)
        # Exponential moving average (alpha = 0.05)
        self.stage_averages_us[stage_name] = prev_avg * 0.95 + elapsed_us * 0.05

    def begin_frame(self) -> None:
        """Marks the start of a frame and updates rolling metrics."""
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now

        dt_ms = max(0.001, min(100.0, dt * 1000.0))
        self.frame_times_ms[self.frame_idx] = dt_ms
        self.frame_idx = (self.frame_idx + 1) % self.history_size
        self.total_frames += 1

        # Instant FPS
        self.fps = 1000.0 / dt_ms if dt_ms > 0.0 else 60.0

        # Compute summary statistics over rolling history window periodically (every 10 frames
        # or startup frames) to eliminate redundant percentile sorting overhead at high frame rates
        if self.total_frames < 10 or self.total_frames % 10 == 0:
            active_count = min(self.total_frames, self.history_size)
            active_window = self.frame_times_ms[:active_count]

            self.avg_frame_time_ms = float(np.mean(active_window))
            self.min_frame_time_ms = float(np.min(active_window))
            self.max_frame_time_ms = float(np.max(active_window))
            self.avg_fps = 1000.0 / self.avg_frame_time_ms if self.avg_frame_time_ms > 0 else 60.0

            # 1% Low FPS (99th percentile of frame time latency)
            p99_time = float(np.percentile(active_window, 99))
            self.one_percent_low_fps = 1000.0 / p99_time if p99_time > 0 else self.avg_fps

    def get_resource_summary(
        self,
        ecs: EntityManager | None = None,
        mega_buffer: MegaBuffer | None = None,
        vfs: VFS | None = None,
        mdi_batches: int = 0,
    ) -> dict[str, str | int | float]:
        """Collects current engine resource utilization metrics."""
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
