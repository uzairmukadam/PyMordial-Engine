"""Deterministic Fixed-Accumulator Engine Tick Loop.

Orchestrates input polling, fixed-frequency simulation (60 Hz), sub-frame
alpha interpolation, and variable-rate render/UI phases.
"""

from __future__ import annotations
import time
from typing import Callable, Optional
import pygame

from engine.core.ecs import EntityManager
from engine.input import InputManager


class EngineLoop:
    """Manages deterministic fixed simulation and variable render stepping."""

    __slots__ = (
        "fixed_dt",
        "max_accumulator",
        "accumulator",
        "is_running",
        "ecs",
        "input_manager",
        "fixed_tick_count",
        "total_frames",
        "elapsed_time",
        "fps",
        "frame_time_ms",
        "on_fixed_update",
        "on_update",
        "on_render",
        "on_ui",
    )

    def __init__(
        self,
        ecs: Optional[EntityManager] = None,
        input_manager: Optional[InputManager] = None,
        fixed_dt: float = 1.0 / 60.0,
        max_accumulator: float = 0.20,
    ) -> None:
        self.fixed_dt = fixed_dt
        self.max_accumulator = max_accumulator
        self.accumulator = 0.0
        self.is_running = False

        self.ecs = ecs if ecs is not None else EntityManager()
        self.input_manager = input_manager if input_manager is not None else InputManager()

        self.fixed_tick_count = 0
        self.total_frames = 0
        self.elapsed_time = 0.0
        self.fps = 0.0
        self.frame_time_ms = 0.0

        # Optional user hooks
        self.on_fixed_update: Optional[Callable[[float], None]] = None
        self.on_update: Optional[Callable[[float, float], None]] = None
        self.on_render: Optional[Callable[[float], None]] = None
        self.on_ui: Optional[Callable[[], None]] = None

    def step_frame(self, frame_time: float) -> int:
        """Executes a single deterministic frame step.

        Useful for unit testing and headless simulation with exact delta times.

        Args:
            frame_time: Elapsed seconds since the last frame.

        Returns:
            The number of fixed update sub-ticks executed during this frame.
        """
        clamped_dt = min(frame_time, self.max_accumulator)
        self.accumulator += clamped_dt
        self.elapsed_time += clamped_dt
        self.total_frames += 1

        sub_ticks = 0

        # Phase 1: Fixed Timestep Phase (Physics & Native Simulation)
        while self.accumulator >= self.fixed_dt:
            self.ecs.cache_previous_physics_state()

            if self.on_fixed_update:
                self.on_fixed_update(self.fixed_dt)

            self.fixed_tick_count += 1
            sub_ticks += 1
            self.accumulator -= self.fixed_dt

        # Phase 2: Variable Timestep Phase (Sub-Frame Interpolation & Gameplay)
        alpha = self.accumulator / self.fixed_dt
        self.ecs.interpolate_render_transforms(alpha)

        if self.on_update:
            self.on_update(clamped_dt, alpha)

        # Phase 3: Render Phase
        if self.on_render:
            self.on_render(alpha)

        # Phase 4: UI Phase
        if self.on_ui:
            self.on_ui()

        return sub_ticks

    def run(self, max_frames: Optional[int] = None) -> None:
        """Starts the real-time execution loop with high-resolution clock timing."""
        self.is_running = True
        clock = pygame.time.Clock()
        fps_timer = time.perf_counter()
        frame_counter = 0

        while self.is_running:
            raw_dt_ms = clock.tick()
            frame_time = raw_dt_ms * 0.001
            self.frame_time_ms = raw_dt_ms

            # 1. Input Processing
            self.input_manager.poll_events()
            if self.input_manager.should_quit or self.input_manager.is_action_just_pressed("pause"):
                self.is_running = False
                break

            # 2-4. Step simulation, interpolation, and rendering
            self.step_frame(frame_time)

            # FPS counter update (once per second)
            frame_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                self.fps = frame_counter / (now - fps_timer)
                fps_timer = now
                frame_counter = 0

            if max_frames and self.total_frames >= max_frames:
                self.is_running = False
                break


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("PyMordial Engine - Loop Test")

    loop = EngineLoop()
    font = pygame.font.SysFont("Consolas", 18)

    # Spawn test entities
    for i in range(10):
        loop.ecs.create_entity(position=(i * 1.5, 0.0, 0.0))

    def on_fixed_tick(dt: float) -> None:
        pass

    def on_render_frame(alpha: float) -> None:
        screen.fill((25, 28, 35))
        fps_text = font.render(
            f"PyMordial Engine Phase 1 | FPS: {loop.fps:.1f} | Frame: {loop.frame_time_ms:.2f}ms | Ticks: {loop.fixed_tick_count}",
            True,
            (180, 220, 255),
        )
        screen.blit(fps_text, (20, 20))
        pygame.display.flip()

    loop.on_fixed_update = on_fixed_tick
    loop.on_render = on_render_frame

    print("Running PyMordial Engine loop... (Press ESC or close window to exit)")
    loop.run(max_frames=180)
    pygame.quit()
    print("Loop execution completed successfully.")
