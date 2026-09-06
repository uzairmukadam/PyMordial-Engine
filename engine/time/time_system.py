"""High-Precision Multi-Domain Time & Clock System for PyMordial Engine."""

from __future__ import annotations
import time
from engine.events import publish_event, TimeScaleChangedEvent


class TimeSystem:
    """Manages independent time domains: RealTime, GameTime, and PhysicsTime."""

    __slots__ = (
        "real_time",
        "real_delta_time",
        "game_time",
        "game_delta_time",
        "fixed_dt",
        "fixed_time",
        "time_scale",
        "is_paused",
        "max_delta_time",
        "frame_count",
        "fixed_tick_count",
        "fps",
        "_fps_timer",
        "_fps_counter",
    )

    def __init__(
        self,
        fixed_dt: float = 1.0 / 60.0,
        max_delta_time: float = 0.20,
    ) -> None:
        self.real_time: float = 0.0
        self.real_delta_time: float = 0.0
        self.game_time: float = 0.0
        self.game_delta_time: float = 0.0
        self.fixed_dt: float = fixed_dt
        self.fixed_time: float = 0.0
        self.time_scale: float = 1.0
        self.is_paused: bool = False
        self.max_delta_time: float = max_delta_time

        self.frame_count: int = 0
        self.fixed_tick_count: int = 0
        self.fps: float = 0.0
        self._fps_timer: float = time.perf_counter()
        self._fps_counter: int = 0

    def tick(self, raw_dt: float) -> tuple[float, float]:
        """Advances real and game time domains.

        Args:
            raw_dt: Raw wall-clock delta time in seconds.

        Returns:
            (clamped_real_dt, effective_game_dt)
        """
        # Clamp frame time to prevent accumulator death-spiral on hitches
        clamped_dt = max(0.0, min(raw_dt, self.max_delta_time))
        self.real_delta_time = clamped_dt
        self.real_time += clamped_dt
        self.frame_count += 1

        if self.is_paused:
            self.game_delta_time = 0.0
        else:
            self.game_delta_time = clamped_dt * self.time_scale
            self.game_time += self.game_delta_time

        # Update FPS estimation every 0.5 seconds
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self.fps = self._fps_counter / elapsed
            self._fps_timer = now
            self._fps_counter = 0

        return (self.real_delta_time, self.game_delta_time)

    def step_fixed_tick(self) -> float:
        """Advances physics time by fixed_dt."""
        self.fixed_time += self.fixed_dt
        self.fixed_tick_count += 1
        return self.fixed_dt

    def set_time_scale(self, scale: float) -> None:
        """Modulates game time dilation factor (e.g., 0.2 for slow-motion)."""
        new_scale = max(0.0, min(10.0, float(scale)))
        if new_scale != self.time_scale:
            old_scale = self.time_scale
            self.time_scale = new_scale
            publish_event(TimeScaleChangedEvent(old_scale=old_scale, new_scale=new_scale))

    def pause(self) -> None:
        """Freezes gameplay simulation while real-time UI/editor continues."""
        self.is_paused = True

    def resume(self) -> None:
        """Resumes gameplay simulation."""
        self.is_paused = False

    def toggle_pause(self) -> bool:
        """Toggles pause state and returns the new state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def reset(self) -> None:
        """Resets all clocks to zero."""
        self.real_time = 0.0
        self.real_delta_time = 0.0
        self.game_time = 0.0
        self.game_delta_time = 0.0
        self.fixed_time = 0.0
        self.frame_count = 0
        self.fixed_tick_count = 0


class Stopwatch:
    """Measures high-resolution elapsed time."""

    __slots__ = ("_start_time", "_elapsed_accum", "_running")

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._elapsed_accum: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        if not self._running:
            self._start_time = time.perf_counter()
            self._running = True

    def stop(self) -> float:
        if self._running:
            self._elapsed_accum += time.perf_counter() - self._start_time
            self._running = False
        return self._elapsed_accum

    def reset(self) -> None:
        self._start_time = 0.0
        self._elapsed_accum = 0.0
        self._running = False

    def restart(self) -> float:
        elapsed = self.elapsed_seconds
        self.reset()
        self.start()
        return elapsed

    @property
    def elapsed_seconds(self) -> float:
        if self._running:
            return self._elapsed_accum + (time.perf_counter() - self._start_time)
        return self._elapsed_accum

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000.0


class Cooldown:
    """Lightweight timer utility for rate-limiting actions."""

    __slots__ = ("duration", "_last_trigger")

    def __init__(self, duration_seconds: float) -> None:
        self.duration = duration_seconds
        self._last_trigger = -duration_seconds

    def is_ready(self, current_time: float) -> bool:
        return (current_time - self._last_trigger) >= self.duration

    def trigger(self, current_time: float) -> bool:
        if self.is_ready(current_time):
            self._last_trigger = current_time
            return True
        return False


class FPSLimiter:
    """High-precision frame pacing limiter utilizing hybrid sleep and spin-locking."""

    __slots__ = ("target_fps", "target_frame_time", "_last_frame_time")

    def __init__(self, target_fps: int = 0) -> None:
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_frame_time = time.perf_counter()

    def set_target_fps(self, fps: int) -> None:
        self.target_fps = max(0, fps)
        self.target_frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

    def wait(self) -> float:
        """Paces the frame rate to match target_fps. Returns actual elapsed delta."""
        if self.target_frame_time <= 0.0:
            now = time.perf_counter()
            dt = now - self._last_frame_time
            self._last_frame_time = now
            return dt

        target_time = self._last_frame_time + self.target_frame_time
        now = time.perf_counter()

        # Coarse sleep if more than 2ms remaining
        remaining = target_time - now
        if remaining > 0.002:
            time.sleep(remaining - 0.0015)

        # Fine busy-wait for sub-millisecond precision
        while time.perf_counter() < target_time:
            pass

        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        return dt


# Global default TimeSystem instance
_GLOBAL_TIME_SYSTEM: TimeSystem = TimeSystem()


def get_time_system() -> TimeSystem:
    return _GLOBAL_TIME_SYSTEM
