"""Repeatable graphics stress scene with live FPS and pipeline mode controls."""

from __future__ import annotations

import math

import pygame

from _common import FlyCameraControls, add_box, create_app, create_parser
from engine.app.module import ProjectModule
from engine.camera import FreeFlyCamera
from engine.input.codes import GamepadButton
from engine.ui import UILabel, UIPanel, UIScreen


class BenchmarkOverlay(ProjectModule):
    """Tracks observed frame pacing and changes renderer workload on demand."""

    name = "GraphicsBenchmarkOverlay"

    def __init__(self, entity_count: int) -> None:
        self.entity_count = entity_count
        self.fps_label: UILabel | None = None
        self.mode_label: UILabel | None = None
        self._fps_average = 0.0
        self._fps_min = float("inf")
        self._fps_max = 0.0
        self._sample_count = 0
        self._display_elapsed = 0.0
        self._gpu_mode = "HYBRID"

    def on_attach(self, app) -> None:
        app.input_manager.bind_action(
            "benchmark_next_gi",
            gamepad_buttons=[GamepadButton.RIGHT_BUMPER],
        )
        app.input_manager.bind_action(
            "benchmark_toggle_ssr",
            gamepad_buttons=[GamepadButton.Y],
        )
        screen = UIScreen(name="BenchmarkHUD", is_modal=False)
        panel = screen.add_child(UIPanel(x=20.0, y=20.0, w=510.0, h=144.0))
        panel.add_child(UILabel("PYMORDIAL GRAPHICS BENCHMARK", x=18.0, y=12.0, w=474.0, h=28.0, font_size=18))
        self.fps_label = panel.add_child(
            UILabel("FPS: waiting for samples", x=18.0, y=52.0, w=474.0, h=26.0, font_size=16)
        )
        self.mode_label = panel.add_child(
            UILabel("GI: HYBRID | SSR: ON", x=18.0, y=86.0, w=474.0, h=26.0, font_size=15)
        )
        app.ui.set_screen(screen)

    def on_update(self, app, dt: float) -> None:
        if app.input_manager.is_action_pressed("benchmark_next_gi"):
            modes = ("OFF", "HYBRID", "RADIANCE_CASCADES")
            self._gpu_mode = modes[(modes.index(self._gpu_mode) + 1) % len(modes)]
            app.pipeline.config.gi_mode = self._gpu_mode
            app.pipeline.apply_config(app.pipeline.config)
        if app.input_manager.is_action_pressed("benchmark_toggle_ssr"):
            app.pipeline.config.ssr_enabled = not app.pipeline.config.ssr_enabled
            app.pipeline.apply_config(app.pipeline.config)

        if dt > 1e-6:
            fps = 1.0 / dt
            self._sample_count += 1
            self._fps_average += (fps - self._fps_average) / self._sample_count
            self._fps_min = min(self._fps_min, fps)
            self._fps_max = max(self._fps_max, fps)
        self._display_elapsed += dt
        if self._display_elapsed < 0.25:
            return
        self._display_elapsed = 0.0
        if self.fps_label is not None:
            self.fps_label.text = (
                f"FPS now/avg/min/max: {1.0 / max(dt, 1e-6):.0f} / "
                f"{self._fps_average:.0f} / {self._fps_min:.0f} / {self._fps_max:.0f}  "
                f"| {dt * 1000.0:.2f} ms"
            )
        if self.mode_label is not None:
            self.mode_label.text = (
                f"GI: {self._gpu_mode} [1-3 / RB] | SSR: {'ON' if app.pipeline.config.ssr_enabled else 'OFF'} [4 / Y] "
                f"| Entities: {self.entity_count} | Frames: {app.engine_loop.total_frames}"
            )

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False
        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            modes = {pygame.K_1: "OFF", pygame.K_2: "HYBRID", pygame.K_3: "RADIANCE_CASCADES"}
            self._gpu_mode = modes[event.key]
            config = app.pipeline.config
            config.gi_mode = self._gpu_mode
            app.pipeline.apply_config(config)
            return True
        if event.key == pygame.K_4:
            config = app.pipeline.config
            config.ssr_enabled = not config.ssr_enabled
            app.pipeline.apply_config(config)
            return True
        return False


def build_scene(app, entity_count: int) -> None:
    app.pipeline.config.gi_mode = "HYBRID"
    app.pipeline.config.ao_mode = "GTAO"
    app.pipeline.config.ssr_enabled = True
    app.pipeline.config.ssr_resolution_scale = 1.0
    app.pipeline.config.volumetric_fog_enabled = False
    app.pipeline.apply_config(app.pipeline.config)

    side = math.ceil(math.sqrt(entity_count))
    spacing = 2.1
    half_span = (side - 1) * spacing * 0.5
    add_box(
        app,
        (0.0, -0.35, 0.0),
        (side * spacing + 8.0, 0.7, side * spacing + 8.0),
        (0.13, 0.16, 0.2),
        0.92,
    )
    shape_cycle = ("cube", "sphere", "cylinder", "cone")
    for index in range(entity_count):
        row, column = divmod(index, side)
        position = (column * spacing - half_span, 0.85, row * spacing - half_span)
        color = (
            0.22 + 0.68 * (column / max(1, side - 1)),
            0.2 + 0.6 * (row / max(1, side - 1)),
            0.72 if index % 2 else 0.32,
        )
        shape = shape_cycle[index % len(shape_cycle)]
        app.ecs.create_entity(
            shape=shape,
            position=position,
            dimensions=(1.35, 1.7 if shape in ("cylinder", "cone") else 1.35, 1.35),
            color=color,
            roughness=0.12 + (index % 8) * 0.11,
            metallic=(index % 5) * 0.18,
            is_static=True,
        )

    app.pipeline.lights_pass.clear()
    app.pipeline.add_point_light((-half_span, 5.0, -half_span), radius=side * 1.6, color=(1.0, 0.35, 0.18), intensity=5.0)
    app.pipeline.add_point_light((half_span, 5.0, -half_span), radius=side * 1.6, color=(0.18, 0.5, 1.0), intensity=5.0)
    app.pipeline.add_point_light((-half_span, 5.0, half_span), radius=side * 1.6, color=(0.25, 1.0, 0.42), intensity=5.0)
    app.pipeline.add_point_light((half_span, 5.0, half_span), radius=side * 1.6, color=(1.0, 0.72, 0.2), intensity=5.0)

    camera = FreeFlyCamera(
        position=(half_span + 12.0, side * 0.85, half_span + 18.0),
        yaw_deg=-135.0,
        pitch_deg=-25.0,
        move_speed=12.0,
    )
    app.camera_manager.register_camera("benchmark_flycam", camera, make_active=True)
    app.add_module(FlyCameraControls(camera))
    app.add_module(BenchmarkOverlay(entity_count))


def _positive_entity_count(value: str) -> int:
    count = int(value)
    if not 16 <= count <= 10000:
        raise ValueError("entity count must be between 16 and 10000")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = create_parser(__doc__ or "Graphics benchmark")
    parser.add_argument("--entities", type=_positive_entity_count, default=625)
    args = parser.parse_args(argv)
    app = create_app(args, "PyMordial | Graphics Benchmark")
    build_scene(app, args.entities)
    app.run()


if __name__ == "__main__":
    main()
