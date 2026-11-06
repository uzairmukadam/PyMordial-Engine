"""PyMordial Engine Example: GI Test Sandbox (Radiance Cascades, SSGI & LPV Evaluation).

Demonstrates:
1. A multi-tiered architectural courtyard with colored bounce panels, columns, and overhangs designed to stress-test global illumination.
2. Dynamic sun angle and high-contrast shadow casting for evaluating diffuse light propagation.
3. Live toggle controls in the UI to switch between GI modes (OFF, SSGI, LPV, HYBRID, RADIANCE_CASCADES).
4. Authoritative 60 Hz Kinematic Character Motor for free exploration.

Usage:
    python examples/07_gi_test_sandbox.py
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
    UIScreen,
    UIPanel,
    UILabel,
    UIButton,
    UIStyle,
)
from engine.world.default_world import DefaultWorldBuilder
from engine.input.codes import GamepadButton
from engine.gfx.quality_presets import GIMode


# ---------------- GI Test Sandbox HUD & Controls ----------------

class GITestHUD(UIScreen):
    """Telemetry and GI Mode Selector HUD."""

    def __init__(self, on_mode_change: Any) -> None:
        super().__init__(name="GITestHUD", is_modal=False)

        # Top-Left Info Card
        card = UIPanel(x=24, y=20, w=420, h=140)
        card.style = UIStyle(
            bg_color_top=(0.06, 0.08, 0.14, 0.94),
            bg_color_bottom=(0.02, 0.03, 0.06, 0.98),
            border_color=(0.30, 0.70, 1.0, 0.6),
            border_width=1.5,
            corner_radius=10.0,
        )
        self.add_child(card)

        self.title_lbl = UILabel("GLOBAL ILLUMINATION TEST SANDBOX", x=18, y=12, w=384, h=22, font_size=12, bold=True, color=(0.30, 0.85, 1.0, 1.0))
        card.add_child(self.title_lbl)

        self.gi_status_lbl = UILabel("Active GI Mode: RADIANCE_CASCADES", x=18, y=38, w=384, h=20, font_size=12, bold=True, color=(0.20, 0.95, 0.60, 0.95))
        card.add_child(self.gi_status_lbl)

        desc = UILabel("Observe color bleeding under overhangs & indirect bounce", x=18, y=62, w=384, h=20, font_size=11, color=(0.80, 0.85, 0.90, 0.8))
        card.add_child(desc)

        hint = UILabel("• [1-5]: Switch GI Mode  •  WASD: Move  •  ESC: Menu", x=18, y=96, w=384, h=18, font_size=10, color=(0.60, 0.70, 0.80, 0.75))
        card.add_child(hint)

        # Mode Selector Buttons Panel
        btn_panel = UIPanel(x=24, y=170, w=420, h=175)
        btn_panel.style = UIStyle(
            bg_color_top=(0.06, 0.08, 0.14, 0.94),
            bg_color_bottom=(0.02, 0.03, 0.06, 0.98),
            border_color=(0.30, 0.70, 1.0, 0.4),
            border_width=1.0,
            corner_radius=10.0,
        )
        self.add_child(btn_panel)

        UILabel("QUICK GI MODE SELECTOR:", x=18, y=12, w=384, h=20, font_size=11, bold=True, color=(0.7, 0.8, 0.9, 0.9)).parent = btn_panel

        btn_panel.add_child(UILabel("QUICK GI MODE SELECTOR:", x=18, y=12, w=384, h=20, font_size=11, bold=True, color=(0.7, 0.8, 0.9, 0.9)))
        btn_panel.add_child(UIButton(text="[1] GI OFF", x=18, y=38, w=185, h=32, variant="neutral", font_size=11, on_click=lambda: on_mode_change("OFF")))
        btn_panel.add_child(UIButton(text="[2] SSGI (Screen-Space)", x=215, y=38, w=185, h=32, variant="neutral", font_size=11, on_click=lambda: on_mode_change("SSGI")))
        btn_panel.add_child(UIButton(text="[3] LPV (Volumetric)", x=18, y=78, w=185, h=32, variant="neutral", font_size=11, on_click=lambda: on_mode_change("LPV")))
        btn_panel.add_child(UIButton(text="[4] HYBRID (SSGI+LPV)", x=215, y=78, w=185, h=32, variant="neutral", font_size=11, on_click=lambda: on_mode_change("HYBRID")))
        btn_panel.add_child(UIButton(text="[5] RADIANCE CASCADES", x=18, y=118, w=382, h=38, variant="primary", font_size=12, on_click=lambda: on_mode_change("RADIANCE_CASCADES")))

    def update_mode_label(self, mode: str) -> None:
        self.gi_status_lbl.text = f"Active GI Mode: {mode}"


# ---------------- GI Test Sandbox Module ----------------

class GITestSandboxModule(ProjectModule):
    """Constructs a courtyard test environment optimized for GI color bleeding and bounce validation."""

    name = "GITestSandboxModule"

    def __init__(self) -> None:
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self.hud: GITestHUD | None = None
        self._first_prop_id = -1
        self._total_props_count = 0
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._is_sprinting = False
        self._app_ref: ProjectApp | None = None

    def on_attach(self, app: ProjectApp) -> None:
        self._app_ref = app
        # Enable high-end GI settings and disable conflicting post-process effects
        app.pipeline.config.gi_mode = "RADIANCE_CASCADES"
        app.pipeline.config.ao_mode = "GTAO"
        app.pipeline.config.bloom_enabled = True
        app.pipeline.config.ssr_enabled = True
        app.pipeline.config.volumetric_fog_enabled = True
        app.pipeline.config.dof_enabled = False

        if app.engine_tweaks is not None:
            app.engine_tweaks.gi_mode = "RADIANCE_CASCADES"
            app.engine_tweaks.dof_enabled = False

        # 1. Neutral Ground Plane
        app.set_world_builder(DefaultWorldBuilder(size=140.0, uv_tiles=40.0, material_name="concrete_clean"))

        prop_specs = []

        # --- ARCHITECTURAL COURTYARD & COLOR BLEEDING BOUNCE WALLS ---
        # Back wall with vibrant color panels to test indirect color bleeding
        prop_specs.append((0.0, 6.0, -25.0, 30.0, 12.0, 2.0, (0.85, 0.85, 0.85, 1.0), 0.6, 0.1))  # Main back wall
        prop_specs.append((-9.0, 5.0, -24.0, 6.0, 10.0, 0.6, (0.90, 0.15, 0.15, 1.0), 0.4, 0.1))   # Vivid Red Bounce Panel
        prop_specs.append((9.0, 5.0, -24.0, 6.0, 10.0, 0.6, (0.15, 0.30, 0.90, 1.0), 0.4, 0.1))    # Vivid Blue Bounce Panel

        # Overhanging roof / balcony (Tests indirect shadow occlusion & ambient darkening underneath)
        prop_specs.append((0.0, 10.0, -15.0, 24.0, 1.5, 12.0, (0.35, 0.38, 0.42, 1.0), 0.5, 0.4))

        # Supporting columns
        prop_specs.append((-7.0, 5.0, -10.0, 1.5, 10.0, 1.5, (0.45, 0.48, 0.52, 1.0), 0.3, 0.7))
        prop_specs.append((7.0, 5.0, -10.0, 1.5, 10.0, 1.5, (0.45, 0.48, 0.52, 1.0), 0.3, 0.7))

        # Central monolith to cast sharp directional sun shadows
        prop_specs.append((0.0, 3.5, -6.0, 3.0, 7.0, 3.0, (0.70, 0.72, 0.75, 1.0), 0.2, 0.8))

        # Side corridor enclosing walls
        for z_coord in [-5.0, -15.0, -25.0]:
            prop_specs.append((-15.0, 6.0, z_coord, 2.0, 12.0, 10.0, (0.30, 0.32, 0.35, 1.0), 0.6, 0.2))
            prop_specs.append((15.0, 6.0, z_coord, 2.0, 12.0, 10.0, (0.30, 0.32, 0.35, 1.0), 0.6, 0.2))

        # Instantiate static architecture
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

        # 2. Spawn Player Entity and Kinematic Character Controller
        self.player_id = app.ecs.create_entity(
            position=(0.0, 1.0, 6.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.90, 0.92, 0.95, 1.0),
            roughness=0.25,
            metallic=0.5,
            is_static=False,
        )
        motor_cfg = CharacterMotorConfig(walk_speed=5.0, run_speed=9.0, jump_force=8.0, step_height=0.40)
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=motor_cfg, initial_position=(0.0, 1.0, 6.0))

        # 3. Attach Follow Camera
        cam_cfg = FollowCameraConfig(distance=5.0, target_offset_y=0.7)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=0.0, initial_pitch_deg=20.0)
        self.camera.update_follow(0.016, (0.0, 1.0, 6.0), physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("sandbox_cam", self.camera, make_active=True)

        # 4. Initialize HUD with Mode Switch Callback
        self.hud = GITestHUD(on_mode_change=self.set_gi_mode)

    def set_gi_mode(self, mode_str: str) -> None:
        if self._app_ref is None:
            return
        self._app_ref.pipeline.config.gi_mode = mode_str
        if self._app_ref.engine_tweaks is not None:
            self._app_ref.engine_tweaks.gi_mode = mode_str
        if self.hud is not None:
            self.hud.update_mode_label(mode_str)

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

        if app.input_manager.is_action_pressed("jump") or keys[pygame.K_SPACE]:
            self._jump_requested = True

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
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("SandboxPlay"):
            return
        self.motor.update(
            dt=dt,
            move_input=(self._move_forward, self._move_strafe),
            is_running=False,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("SandboxPlay"):
            return
        self.poll_inputs(app, dt)
        self.camera.update_follow(dt, self.motor.position, physics=app.physics, exclude_entity_id=self.player_id)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if self.camera is None or not app.state_manager.is_in_state("SandboxPlay"):
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

class SandboxPlayState(GameState):
    name = "SandboxPlay"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        mod = app.get_module(GITestSandboxModule)
        if mod and mod.hud:
            app.ui.set_screen(mod.hud)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if app.input_manager.is_action_pressed("pause") or app.input_manager.is_gamepad_pressed(GamepadButton.START):
            app.stop()
            return
        mod = app.get_module(GITestSandboxModule)
        if mod:
            mod.poll_inputs(app, dt)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.stop()
            return True
        mod = app.get_module(GITestSandboxModule)
        if mod:
            return mod.on_event(app, event)
        return False


# ---------------- Main Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial GI Test Sandbox")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--frames", type=int, default=None, help="Max frames")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Global Illumination Test Sandbox",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(GITestSandboxModule())

    app.state_manager.register(SandboxPlayState())
    app.state_manager.set_state("SandboxPlay")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()