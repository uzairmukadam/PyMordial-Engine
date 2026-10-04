"""PyMordial Engine Example 02: Kinematic Player Movement, Physics & Cubes.

Demonstrates authoritative 60 Hz Kinematic Character Motor physics, WASD movement,
jumping, sprinting, dynamic Rapier3D physics cubes, curb/stair step-climbing,
a real-time telemetry HUD, and in-game pause state management.

Zero debug menu dependency.

Usage:
    python examples/02_player_physics_and_cubes.py
    python examples/02_player_physics_and_cubes.py --headless --frames 60
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pygame

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import (
    ProjectApp,
    ProjectConfig,
    ProjectModule,
    GameState,
    FollowCamera,
    FollowCameraConfig,
    CharacterMotor,
    CharacterMotorConfig,
    UIScreen,
    UIPanel,
    UILabel,
    UIButton,
    UIStyle,
)
from engine.world.default_world import DefaultWorldBuilder
from engine.input.codes import GamepadButton


# ---------------- Modern HUD & Menus ----------------

def _create_hud_badge(x: float, y: float, w: float, h: float, border_col=(0.18, 0.65, 0.95, 0.5)) -> UIPanel:
    panel = UIPanel(x=x, y=y, w=w, h=h)
    panel.style = UIStyle(
        bg_color_top=(0.04, 0.06, 0.10, 0.85),
        bg_color_bottom=(0.02, 0.03, 0.05, 0.92),
        border_color=border_col,
        border_width=1.0,
        corner_radius=8.0,
    )
    return panel


class GameplayHUD(UIScreen):
    """In-game telemetry HUD displaying speed, grounded state, and controls."""

    def __init__(self) -> None:
        super().__init__(name="GameplayHUD", is_modal=False)

        # Top-Left Telemetry Card
        tele_card = _create_hud_badge(x=30, y=30, w=320, h=100)
        self.add_child(tele_card)

        self.speed_lbl = UILabel("SPEED: 0.0 KM/H", x=20, y=14, w=280, h=28, font_size=18, bold=True, color=(0.2, 0.9, 1.0, 1.0))
        tele_card.add_child(self.speed_lbl)

        self.status_lbl = UILabel("STATUS: Grounded | Walking", x=20, y=44, w=280, h=22, font_size=13, color=(0.8, 0.85, 0.9, 0.9))
        tele_card.add_child(self.status_lbl)

        self.pos_lbl = UILabel("POS: (0.0, 1.0, 0.0)", x=20, y=68, w=280, h=20, font_size=11, color=(0.55, 0.65, 0.75, 0.8))
        tele_card.add_child(self.pos_lbl)

        # Bottom-Right Controls Legend Card
        ctrl_card = _create_hud_badge(x=880, y=550, w=370, h=140)
        self.add_child(ctrl_card)

        c_title = UILabel("CONTROLS (KBM & GAMEPAD)", x=20, y=8, w=330, h=22, font_size=12, bold=True, color=(0.95, 0.75, 0.20, 1.0))
        ctrl_card.add_child(c_title)

        c_txt = UILabel(
            "• Move: WASD / Left Stick\n"
            "• Look: Mouse / Right Stick | Scroll: Zoom\n"
            "• Jump: Space / Button (A)\n"
            "• Sprint: Left Shift / (L3) / (LB)\n"
            "• Pause: ESC / Button Start",
            x=20,
            y=30,
            w=330,
            h=95,
            font_size=11,
            color=(0.7, 0.75, 0.85, 0.85),
        )
        ctrl_card.add_child(c_txt)

    def update_telemetry(self, speed_mps: float, is_grounded: bool, is_sprinting: bool, pos: tuple[float, float, float]) -> None:
        kmh = speed_mps * 3.6
        self.speed_lbl.text = f"SPEED: {kmh:.1f} KM/H"

        if not is_grounded:
            stance = "Airborne"
            col = (1.0, 0.7, 0.2, 1.0)
        elif is_sprinting and speed_mps > 0.5:
            stance = "Sprinting"
            col = (0.2, 1.0, 0.5, 1.0)
        elif speed_mps > 0.5:
            stance = "Walking"
            col = (0.3, 0.8, 1.0, 1.0)
        else:
            stance = "Idle"
            col = (0.7, 0.8, 0.9, 0.9)

        ground_txt = "Grounded" if is_grounded else "In Air"
        self.status_lbl.text = f"STATUS: {ground_txt} | {stance}"
        self.status_lbl.color = col
        self.pos_lbl.text = f"POS: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"


class PlayerPauseMenuUI(UIScreen):
    """Modern dimmed modal pause menu."""

    def __init__(self, on_resume, on_reset_pos, on_main_menu) -> None:
        super().__init__(name="PlayerPauseMenu")

        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(
            bg_color_top=(0.0, 0.0, 0.0, 0.65),
            bg_color_bottom=(0.0, 0.0, 0.0, 0.80),
            corner_radius=0.0,
        )
        self.add_child(backdrop)

        modal = UIPanel(x=430, y=140, w=420, h=440)
        modal.style = UIStyle(
            bg_color_top=(0.07, 0.09, 0.14, 0.95),
            bg_color_bottom=(0.03, 0.04, 0.07, 0.98),
            border_color=(0.18, 0.75, 1.0, 0.7),
            border_width=1.5,
            corner_radius=12.0,
        )
        backdrop.add_child(modal)

        lbl = UILabel("SIMULATION PAUSED", x=0, y=25, w=420, h=34, font_size=23, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        modal.add_child(lbl)

        sub = UILabel("Kinematic Motor & Rapier Rigid Bodies Frozen", x=0, y=62, w=420, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.7))
        modal.add_child(sub)

        btn_resume = UIButton(text="RESUME", x=40, y=105, w=340, h=48, variant="primary", font_size=16, on_click=on_resume)
        modal.add_child(btn_resume)

        btn_reset = UIButton(text="RESET PLAYER POSITION", x=40, y=170, w=340, h=48, variant="secondary", font_size=15, on_click=on_reset_pos)
        modal.add_child(btn_reset)

        btn_menu = UIButton(text="QUIT TO MAIN MENU", x=40, y=235, w=340, h=48, variant="danger", font_size=16, on_click=on_main_menu)
        modal.add_child(btn_menu)

        hint = UILabel("Press ESC to Resume Instantly", x=0, y=380, w=420, h=20, font_size=12, align="center", color=(0.55, 0.65, 0.75, 0.6))
        modal.add_child(hint)


# ---------------- Player Controller & World Module ----------------

class PlayerWorldModule(ProjectModule):
    """Manages player character kinematics, Rapier colliders, dynamic cubes, and camera."""

    name = "PlayerWorld"

    def __init__(self) -> None:
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self.hud: GameplayHUD | None = None
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._is_sprinting = False

    def on_attach(self, app: ProjectApp) -> None:
        # Enforce fine 0.03 film grain, universal water, zero ghosting, and disable DOF
        app.pipeline.config.film_grain_intensity = 0.03
        app.pipeline.config.film_grain_enabled = True
        app.pipeline.config.water_enabled = True
        app.pipeline.config.water_height = 0.45
        app.pipeline.config.motion_blur_enabled = False
        app.pipeline.config.lens_flare_ghost_intensity = 0.0
        app.pipeline.config.dof_enabled = False
        if app.engine_tweaks is not None:
            app.engine_tweaks.film_grain_intensity = 0.03
            app.engine_tweaks.water_height = 0.45
            app.engine_tweaks.dof_enabled = False

        # 1. Spawn Fixed Ground Platform with cracked_dry_mud POM
        app.set_world_builder(DefaultWorldBuilder(size=80.0, uv_tiles=40.0, material_name="cracked_dry_mud"))

        # 2. Spawn Static Curbs / Stair Steps to test autostep
        self._first_cube_id = -1
        for idx, (height, z_pos) in enumerate([(0.15, 6.0), (0.25, 9.0)]):
            step_id = app.ecs.create_entity(
                position=(0.0, height * 0.5, z_pos),
                scale=(4.0, height, 2.0),
                color=(0.25, 0.45, 0.65, 1.0),
                roughness=0.4,
                metallic=0.2,
                is_static=True,
            )
            if self._first_cube_id == -1:
                self._first_cube_id = step_id
            app.physics.create_body(step_id, body_type="fixed", position=(0.0, height * 0.5, z_pos))
            app.physics.attach_box_collider(step_id, half_x=2.0, half_y=height * 0.5, half_z=1.0)

        # 3. Spawn Dynamic Physics Cubes
        cube_colors = [
            (0.95, 0.25, 0.25, 1.0),
            (0.20, 0.85, 0.35, 1.0),
            (0.95, 0.75, 0.15, 1.0),
            (0.65, 0.25, 0.95, 1.0),
            (0.15, 0.80, 0.95, 1.0),
            (0.95, 0.45, 0.15, 1.0),
        ]
        for i in range(8):
            px = -6.0 + (i % 4) * 2.5
            pz = -5.0 + (i // 4) * 3.0
            py = 1.0 + (i * 0.4)
            c_color = cube_colors[i % len(cube_colors)]
            cube_id = app.ecs.create_entity(
                position=(px, py, pz),
                scale=(1.0, 1.0, 1.0),
                color=c_color,
                roughness=0.3,
                metallic=0.5,
                is_static=False,
            )
            if self._first_cube_id == -1:
                self._first_cube_id = cube_id
            app.physics.create_body(cube_id, body_type="dynamic", position=(px, py, pz))
            app.physics.attach_box_collider(cube_id, half_x=0.5, half_y=0.5, half_z=0.5, density=1.5)

        # 4. Spawn Player Entity and Kinematic Character Controller
        # Primitive capsule already has 0.8m diameter and 1.8m height; scale (1,1,1) preserves perfect proportions
        self.player_id = app.ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.15, 0.95, 0.85, 1.0),
            roughness=0.2,
            metallic=0.7,
            is_static=False,
        )
        cfg = CharacterMotorConfig(
            walk_speed=6.0,
            run_speed=11.0,
            jump_force=8.0,
            step_height=0.30,
            snap_to_ground=0.25,
        )
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=cfg, initial_position=(0.0, 1.0, 0.0))

        # 5. Attach Follow Camera closer to player with AAA framing
        cam_cfg = FollowCameraConfig(distance=3.8, target_offset_y=0.65)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=45.0, initial_pitch_deg=18.0)
        self.camera.update_follow(0.016, (0.0, 1.0, 0.0), physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("player_cam", self.camera, make_active=True)

        # 6. Initialize HUD
        self.hud = GameplayHUD()

    def get_draw_batches(self, app: ProjectApp) -> list[tuple]:
        alloc_cube = app.pipeline.mega_buffer.allocations["cube"]
        alloc_capsule = app.pipeline.mega_buffer.allocations["capsule"]
        d_cube = app.ecs.pool.get_dense_index(self._first_cube_id)
        d_capsule = app.ecs.pool.get_dense_index(self.player_id)
        # 2 curbs + 8 dynamic cubes = 10 cubes
        return [
            (alloc_cube, 10, d_cube, False, True),
            (alloc_capsule, 1, d_capsule, False, True),
        ]

    def poll_inputs(self, app: ProjectApp, dt: float) -> None:
        """Polls dual keyboard and gamepad input with zero allocations."""
        # 1. Keyboard Movement
        keys = pygame.key.get_pressed()
        fwd = (1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0) - (1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0)
        strafe = (1.0 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 0.0) - (1.0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 0.0)
        sprint_key = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        # 2. Gamepad Left Analog Stick & Face Buttons
        pad_move = app.input_manager.get_vector2("move")
        if abs(pad_move[0]) > 0.05 or abs(pad_move[1]) > 0.05:
            strafe = pad_move[0]
            fwd = pad_move[1]

        pad_sprint = (
            app.input_manager.is_action_down("sprint")
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_STICK)
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_BUMPER)
            or app.input_manager.is_gamepad_down(GamepadButton.RIGHT_TRIGGER)
        )
        if app.input_manager.is_action_pressed("jump") or app.input_manager.is_gamepad_pressed(GamepadButton.A):
            self._jump_requested = True

        self._move_forward = fwd
        self._move_strafe = strafe
        self._is_sprinting = sprint_key or pad_sprint

        # 3. Gamepad Right Analog Stick Camera Look
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
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("Gameplay"):
            return

        # Continuous movement input vector: (forward, strafe)
        move_input = (self._move_forward, self._move_strafe)

        # Step 60 Hz authoritative character motor
        self.motor.update(
            dt=dt,
            move_input=move_input,
            is_running=self._is_sprinting,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("Gameplay"):
            return

        self.poll_inputs(app, dt)

        # Track player position with collision-aware follow camera
        pos = self.motor.position
        self.camera.update_follow(dt, pos, physics=app.physics, exclude_entity_id=self.player_id)

        # Update HUD
        if self.hud is not None and app.ui.active_screen is self.hud:
            spd = self.motor.state.horizontal_speed
            grounded = self.motor.state.is_grounded
            self.hud.update_telemetry(spd, grounded, self._is_sprinting, pos)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._jump_requested = True
                return True
            elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self._is_sprinting = True
                return True

        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self._is_sprinting = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self.camera is not None and app.state_manager.is_in_state("Gameplay"):
                rel_x, rel_y = float(event.rel[0]), float(event.rel[1])
                self.camera.handle_mouse_orbit(-rel_x, -rel_y)
                return True

        elif event.type == pygame.MOUSEWHEEL:
            if self.camera is not None and app.state_manager.is_in_state("Gameplay"):
                self.camera.handle_zoom(float(event.y))
                return True

        return False


# ---------------- Game State Implementations ----------------

class MainMenuState(GameState):
    """Main menu state: halts physics, displays UI, enables mouse cursor."""

    name = "MainMenu"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        def start_game():
            app.state_manager.set_state("Gameplay")

        def quit_app():
            app.stop()

        screen = UIScreen(name="StartScreen")
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(bg_color_top=(0.04, 0.05, 0.08, 0.95), bg_color_bottom=(0.02, 0.03, 0.05, 0.98), corner_radius=0.0)
        screen.add_child(bg)

        card = UIPanel(x=420, y=160, w=440, h=400)
        card.style = UIStyle(bg_color_top=(0.07, 0.09, 0.14, 0.92), bg_color_bottom=(0.03, 0.04, 0.07, 0.96), border_color=(0.2, 0.7, 1.0, 0.6), border_width=1.5, corner_radius=12.0)
        bg.add_child(card)

        title = UILabel("PHYSICS & PLAYER DEMO", x=0, y=30, w=440, h=36, font_size=24, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        card.add_child(title)

        sub = UILabel("Kinematic Motor, Sweep & Slide, Dynamic Cubes", x=0, y=70, w=440, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.8))
        card.add_child(sub)

        btn_play = UIButton(text="START SIMULATION", x=50, y=120, w=340, h=52, variant="primary", font_size=18, on_click=start_game)
        card.add_child(btn_play)

        btn_exit = UIButton(text="EXIT", x=50, y=190, w=340, h=52, variant="danger", font_size=18, on_click=quit_app)
        card.add_child(btn_exit)

        tip = UILabel("Features:\n• 60 Hz Decoupled Kinematics\n• Dynamic Rapier Cube Collisions\n• In-Game Telemetry HUD & Pause Menu", x=50, y=270, w=340, h=90, font_size=12, color=(0.6, 0.7, 0.8, 0.8))
        card.add_child(tip)

        app.ui.set_screen(screen)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class GameplayState(GameState):
    """Gameplay state: physics enabled, mouse grabbed, HUD visible."""

    name = "Gameplay"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        mod = app.get_module(PlayerWorldModule)
        if mod and mod.hud:
            app.ui.set_screen(mod.hud)

    def on_resume(self, app: ProjectApp) -> None:
        """Restores in-game HUD and re-asserts hardware mouse capture when unpausing."""
        app.input_manager.reset()
        mod = app.get_module(PlayerWorldModule)
        if mod:
            mod._move_forward = 0.0
            mod._move_strafe = 0.0
            mod._jump_requested = False
            if mod.hud:
                app.ui.set_screen(mod.hud)
        app.input_manager.set_mouse_grab(True)
        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
        except Exception:
            pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if app.input_manager.is_action_pressed("pause") or app.input_manager.is_gamepad_pressed(GamepadButton.START):
            app.state_manager.push_state("Paused")
            return
        mod = app.get_module(PlayerWorldModule)
        if mod:
            mod.poll_inputs(app, dt)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("Paused")
            return True
        mod = app.get_module(PlayerWorldModule)
        if mod:
            return mod.on_event(app, event)
        return False


class PausedState(GameState):
    """Paused state: freezes physics while displaying modal pause dialog."""

    name = "Paused"
    allow_fixed_update = False  # Freeze physics!
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        def resume():
            app.state_manager.pop_state()

        def reset_pos():
            mod = app.get_module(PlayerWorldModule)
            if mod and mod.motor:
                mod.motor.teleport((0.0, 1.0, 0.0), reset_velocity=True)
            app.state_manager.pop_state()

        def main_menu():
            app.state_manager.set_state("MainMenu")

        app.ui.set_screen(PlayerPauseMenuUI(on_resume=resume, on_reset_pos=reset_pos, on_main_menu=main_menu))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if (
            (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            or (event.type == pygame.CONTROLLERBUTTONDOWN and event.button in (getattr(pygame, "CONTROLLER_BUTTON_START", 6), getattr(pygame, "CONTROLLER_BUTTON_B", 1)))
            or (event.type == pygame.JOYBUTTONDOWN and event.button in (7, 1))
        ):
            app.state_manager.pop_state()
            return True
        return False


# ---------------- Application Main ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Kinematic Player & Physics Demo")
    parser.add_argument("--headless", action="store_true", help="Run in headless offscreen mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Kinematic Player & Rapier Physics",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,  # Central sacred debug subsystem available via F1/F2/F3
    )

    app = ProjectApp(config=config)
    app.add_module(PlayerWorldModule())

    # Register Game States
    app.state_manager.register(MainMenuState())
    app.state_manager.register(GameplayState())
    app.state_manager.register(PausedState())

    # Start in Main Menu
    app.state_manager.set_state("MainMenu")

    try:
        if args.headless and args.frames:
            step_count = max(1, args.frames // 3)
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.set_state("Gameplay")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.push_state("Paused")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
        else:
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
