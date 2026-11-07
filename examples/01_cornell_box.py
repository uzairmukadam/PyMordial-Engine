"""PyMordial Engine Example: Cornell Box GI & AO Validation Sandbox (Vibrant Bounce & Fixed Floor).

Constructs the classic Cornell Box featuring:
- Red left wall and green right wall for color bleeding[cite: 9].
- White floor, ceiling, and back wall[cite: 9].
- Two central pillars (tall and short) for contact AO and shadow validation[cite: 9].
- Optimized Lighting Setup:
    - Directional sun relocated to fall squarely on the RED wall.
    - Sun intensity boosted for rich, vibrant bounce.
    - High-resolution concrete floor texture tiling fixed.
- Live GI mode switching via keyboard keys [1-5].

Usage:
    python examples/07_vibrant_gi_box.py
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any
import pygame

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import (
    ProjectApp,
    ProjectConfig,
    ProjectModule,
    GameState,
    CharacterMotor,
    CharacterMotorConfig,
    FollowCamera,
    FollowCameraConfig,
)
from engine.world.default_world import DefaultWorldBuilder
from engine.input.codes import GamepadButton


# ---------------- Cornell Box Module ----------------

class CornellBoxModule(ProjectModule):
    """Constructs the classic closed Cornell Box scene with red/green walls and central pillars."""

    name = "CornellBoxModule"

    def __init__(self) -> None:
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self._first_prop_id = -1
        self._total_props_count = 0
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._app_ref: ProjectApp | None = None

    def on_attach(self, app: ProjectApp) -> None:
        self._app_ref = app
        app.pipeline.config.gi_mode = "RADIANCE_CASCADES"
        app.pipeline.config.ao_mode = "GTAO"
        app.pipeline.config.bloom_enabled = True
        app.pipeline.config.ssr_enabled = True
        app.pipeline.config.volumetric_fog_enabled = False
        app.pipeline.config.dof_enabled = False

        if app.engine_tweaks is not None:
            app.engine_tweaks.gi_mode = "RADIANCE_CASCADES"
            app.engine_tweaks.dof_enabled = False

        # Floor texture tiling configuration
        app.set_world_builder(DefaultWorldBuilder(size=200.0, uv_tiles=8.0, material_name="concrete_clean"))

        prop_specs = []

        # Room Dimensions: Width = 12.0, Height = 10.0, Depth = 14.0
        # 1. Floor (White)[cite: 9]
        prop_specs.append((0.0, 0.0, -7.0, 12.0, 0.5, 14.0, (0.85, 0.85, 0.85, 1.0), 0.7, 0.1))
        # 2. Ceiling (White)[cite: 9]
        prop_specs.append((0.0, 10.0, -7.0, 12.0, 0.5, 14.0, (0.85, 0.85, 0.85, 1.0), 0.7, 0.1))
        # 3. Back Wall (White)[cite: 9]
        prop_specs.append((0.0, 5.0, -14.0, 12.0, 10.0, 0.5, (0.85, 0.85, 0.85, 1.0), 0.7, 0.1))
        
        # 4. Left Wall (Vibrant Red)[cite: 9]
        prop_specs.append((-6.0, 5.0, -7.0, 0.5, 10.0, 14.0, (0.85, 0.12, 0.12, 1.0), 0.5, 0.1))
        # 5. Right Wall (Vibrant Green)[cite: 9]
        prop_specs.append((6.0, 5.0, -7.0, 0.5, 10.0, 14.0, (0.12, 0.75, 0.20, 1.0), 0.5, 0.1))

        # --- CENTRAL PILLARS / BOXES ---
        # Tall Pillar[cite: 9]
        prop_specs.append((-2.2, 3.2, -6.5, 3.2, 6.4, 3.2, (0.82, 0.82, 0.82, 1.0), 0.4, 0.1))
        # Short Pillar[cite: 9]
        prop_specs.append((2.0, 1.8, -8.5, 3.2, 3.6, 3.2, (0.82, 0.82, 0.82, 1.0), 0.4, 0.1))

        self._first_prop_id = -1
        self._total_props_count = len(prop_specs)
        for px, py, pz, sx, sy, sz, col, rough, metal in prop_specs:
            eid = app.ecs.create_entity(
                position=(px, py, pz),
                scale=(sx, sy, sz),
                color=col,
                roughness=rough,
                metallic=metal,
                is_static=True,
            )
            if self._first_prop_id == -1:
                self._first_prop_id = eid
            app.physics.create_body(eid, body_type="fixed", position=(px, py, pz))
            app.physics.attach_box_collider(eid, half_x=sx * 0.5, half_y=sy * 0.5, half_z=sz * 0.5)

        # Player Spawn
        self.player_id = app.ecs.create_entity(
            position=(0.0, 1.0, 3.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.90, 0.92, 0.95, 1.0),
            roughness=0.25,
            metallic=0.5,
            is_static=False,
        )
        motor_cfg = CharacterMotorConfig(walk_speed=4.0, run_speed=7.0, jump_force=6.0, step_height=0.30)
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=motor_cfg, initial_position=(0.0, 1.0, 3.0))

        cam_cfg = FollowCameraConfig(distance=4.5, target_offset_y=0.6)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=0.0, initial_pitch_deg=10.0)
        self.camera.update_follow(0.016, (0.0, 1.0, 3.0), physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("cornell_cam", self.camera, make_active=True)

    def set_gi_mode(self, mode_str: str) -> None:
        if self._app_ref is None:
            return
        self._app_ref.pipeline.config.gi_mode = mode_str
        if self._app_ref.engine_tweaks is not None:
            self._app_ref.engine_tweaks.gi_mode = mode_str
        print(f"[GI Mode Switched]: {mode_str}")

    def get_draw_batches(self, app: ProjectApp) -> list[tuple]:
        alloc_cube = app.pipeline.mega_buffer.allocations["cube"]
        alloc_capsule = app.pipeline.mega_buffer.allocations["capsule"]
        d_prop = app.ecs.pool.get_dense_index(self._first_prop_id)
        d_capsule = app.ecs.pool.get_dense_index(self.player_id)
        return [
            (alloc_cube, self._total_props_count, d_prop, False, True),
            (alloc_capsule, 1, d_capsule, False, True),
        ]

    def poll_inputs(self, app: ProjectApp, dt: float) -> None:
        keys = pygame.key.get_pressed()
        fwd = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        strafe = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)

        self._move_forward = fwd
        self._move_strafe = strafe

        look_yaw = app.input_manager.get_axis("look_yaw")
        look_pitch = app.input_manager.get_axis("look_pitch")
        if (abs(look_yaw) > 0.02 or abs(look_pitch) > 0.02) and self.camera is not None:
            self.camera.yaw_deg += look_yaw * 120.0 * dt
            self.camera.pitch_deg = max(
                self.camera.config.min_pitch_deg,
                min(self.camera.config.max_pitch_deg, self.camera.pitch_deg - look_pitch * 90.0 * dt),
            )
            self.camera.mark_dirty()

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("CornellPlay"):
            return
        self.motor.update(
            dt=dt,
            move_input=(self._move_forward, self._move_strafe),
            is_running=False,
            jump_requested=False,
            camera_yaw_deg=self.camera.yaw_deg,
        )

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("CornellPlay"):
            return
        self.poll_inputs(app, dt)
        self.camera.update_follow(dt, self.motor.position, physics=app.physics, exclude_entity_id=self.player_id)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if self.camera is None or not app.state_manager.is_in_state("CornellPlay"):
            return False

        if event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = float(event.rel[0]), float(event.rel[1])
            self.camera.handle_mouse_orbit(-rel_x, -rel_y)
            return True
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.handle_zoom(float(event.y))
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.set_gi_mode("OFF")
                return True
            elif event.key == pygame.K_2:
                self.set_gi_mode("SSGI")
                return True
            elif event.key == pygame.K_3:
                self.set_gi_mode("LPV")
                return True
            elif event.key == pygame.K_4:
                self.set_gi_mode("HYBRID")
                return True
            elif event.key == pygame.K_5:
                self.set_gi_mode("RADIANCE_CASCADES")
                return True
        return False


# ---------------- Game States ----------------

class CornellPlayState(GameState):
    name = "CornellPlay"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        # Clear screen of any HUD elements upon entering play
        app.ui.set_screen(None)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if app.input_manager.is_action_pressed("pause") or app.input_manager.is_gamepad_pressed(GamepadButton.START):
            app.stop()
            return
        mod = app.get_module(CornellBoxModule)
        if mod:
            mod.poll_inputs(app, dt)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.stop()
            return True
        mod = app.get_module(CornellBoxModule)
        if mod:
            return mod.on_event(app, event)
        return False


# ---------------- Main Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Cornell Box GI Test")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--frames", type=int, default=None, help="Max frames")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Cornell Box GI & AO Validation (Vibrant Red Bounce Setup)",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(CornellBoxModule())

    app.state_manager.register(CornellPlayState())
    app.state_manager.set_state("CornellPlay")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()