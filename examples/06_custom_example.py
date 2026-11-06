"""PyMordial Engine Example 07: Cyberpunk Neon Alley & Wet Reflective PBR Showcase.

Demonstrates:
1. A dense, vertical cyberpunk alleyway featuring glowing neon signs, wet reflective ground, and overhanging steel architecture.
2. Custom procedural layout via WorldBuilder configured with high-roughness asphalt and rain-puddle screen-space reflections (SSR).
3. Authoritative 60 Hz Kinematic Character Motor with step climbing and smooth urban exploration controls.
4. Dynamic ModernGL PBR lighting, bloom, volumetric fog, and chromatic ambiance.
5. Glassmorphic cyberpunk HUD and telemetry tracking.

Zero debug menu dependency. All state and UI logic runs through public engine APIs.

Usage:
    python examples/07_cyberpunk_neon_alley.py
    python examples/07_cyberpunk_neon_alley.py --headless --frames 60
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


# ---------------- Cyberpunk Glassmorphic UI ----------------

def _create_cyber_card(
    x: float,
    y: float,
    w: float,
    h: float,
    border_accent: tuple[float, float, float, float] = (0.15, 0.85, 1.0, 0.6),
) -> UIPanel:
    """Creates a neon-cyan accented obsidian glass card for the cyberpunk HUD."""
    panel = UIPanel(x=x, y=y, w=w, h=h)
    panel.style = UIStyle(
        bg_color_top=(0.04, 0.06, 0.12, 0.94),
        bg_color_bottom=(0.01, 0.02, 0.05, 0.98),
        border_color=border_accent,
        border_width=1.5,
        corner_radius=10.0,
    )
    return panel


class CyberpunkHUD(UIScreen):
    """In-game cyberpunk telemetry HUD."""

    def __init__(self) -> None:
        super().__init__(name="CyberpunkHUD", is_modal=False)

        # Top-Left District Info Card
        card = _create_cyber_card(x=24, y=20, w=380, h=110, border_accent=(0.18, 0.88, 1.0, 0.6))
        self.add_child(card)

        self.dist_lbl = UILabel("DISTRICT: SECTOR 7 - NEON ALLEYWAY", x=18, y=12, w=345, h=22, font_size=12, bold=True, color=(0.18, 0.90, 1.0, 1.0))
        card.add_child(self.dist_lbl)

        self.obj_lbl = UILabel("OBJECTIVE: Navigate the wet reflective corridor", x=18, y=38, w=345, h=20, font_size=12, color=(0.90, 0.92, 0.96, 0.9))
        card.add_child(self.obj_lbl)

        self.status_lbl = UILabel("Weather: Heavy Rain / PBR SSR Active", x=18, y=62, w=345, h=20, font_size=11, color=(0.20, 0.95, 0.60, 0.95))
        card.add_child(self.status_lbl)

        hint = UILabel("• WASD / Stick: Move  •  Space: Jump  •  ESC: Pause", x=18, y=86, w=345, h=18, font_size=10, color=(0.65, 0.72, 0.82, 0.75))
        card.add_child(hint)

        # Bottom-Right Telemetry Card
        kin_card = _create_cyber_card(x=910, y=595, w=346, h=100, border_accent=(0.95, 0.25, 0.65, 0.6))
        self.add_child(kin_card)

        self.speed_lbl = UILabel("SPEED: 0.0 KM/H", x=16, y=12, w=310, h=24, font_size=15, bold=True, color=(0.95, 0.30, 0.75, 1.0))
        kin_card.add_child(self.speed_lbl)

        self.stance_lbl = UILabel("STATUS: Grounded | Urban Patrol", x=16, y=38, w=310, h=20, font_size=12, color=(0.8, 0.85, 0.9, 0.9))
        kin_card.add_child(self.stance_lbl)

        self.coords_lbl = UILabel("POS: X: 0.0, Y: 1.0, Z: 0.0", x=16, y=64, w=310, h=20, font_size=11, color=(0.65, 0.72, 0.82, 0.8))
        kin_card.add_child(self.coords_lbl)

    def update_telemetry(self, speed_mps: float, is_grounded: bool, pos: tuple[float, float, float]) -> None:
        kmh = speed_mps * 3.6
        self.speed_lbl.text = f"SPEED: {kmh:.1f} KM/H"
        ground_txt = "Grounded" if is_grounded else "Airborne"
        self.stance_lbl.text = f"STATUS: {ground_txt} | Urban Navigation"
        self.coords_lbl.text = f"POS: X: {pos[0]:.1f}, Y: {pos[1]:.1f}, Z: {pos[2]:.1f}"


class CyberPauseMenuUI(UIScreen):
    """Modal pause menu for the cyberpunk scene."""

    def __init__(self, on_resume: Any, on_reset: Any, on_menu: Any) -> None:
        super().__init__(name="CyberPauseMenu")

        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(bg_color_top=(0.0, 0.0, 0.0, 0.70), bg_color_bottom=(0.0, 0.0, 0.0, 0.88), corner_radius=0.0)
        self.add_child(backdrop)

        modal = _create_cyber_card(x=420, y=140, w=440, h=440, border_accent=(0.18, 0.88, 1.0, 0.7))
        backdrop.add_child(modal)

        lbl = UILabel("SYSTEM PAUSED", x=0, y=25, w=440, h=34, font_size=24, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        modal.add_child(lbl)

        sub = UILabel("Cyberpunk District Simulation Frozen", x=0, y=62, w=440, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.7))
        modal.add_child(sub)

        btn_resume = UIButton(text="RESUME SIMULATION", x=45, y=110, w=350, h=48, variant="primary", font_size=16, on_click=on_resume)
        modal.add_child(btn_resume)

        btn_reset = UIButton(text="RESET AGENT POSITION", x=45, y=175, w=350, h=48, variant="neutral", font_size=15, on_click=on_reset)
        modal.add_child(btn_reset)

        btn_menu = UIButton(text="QUIT TO MAIN MENU", x=45, y=240, w=350, h=48, variant="danger", font_size=16, on_click=on_menu)
        modal.add_child(btn_menu)

        hint = UILabel("Press [ESC] to Resume Instantly", x=0, y=385, w=440, h=20, font_size=12, align="center", color=(0.55, 0.65, 0.75, 0.6))
        modal.add_child(hint)


# ---------------- Cyberpunk Alley Module ----------------

class CyberpunkAlleyModule(ProjectModule):
    """Constructs a dense cyberpunk alleyway with towering megastructure walls and neon aesthetics."""

    name = "CyberpunkAlleyModule"

    def __init__(self) -> None:
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self.hud: CyberpunkHUD | None = None
        self._first_prop_id = -1
        self._total_props_count = 0
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._is_sprinting = False

    def on_attach(self, app: ProjectApp) -> None:
        # Enforce cyberpunk PBR atmosphere: wet reflective ground, neon bloom, and volumetric fog
        app.pipeline.config.water_enabled = True
        app.pipeline.config.water_height = 0.05  # Shallow rainwater puddles on asphalt
        app.pipeline.config.bloom_enabled = True
        app.pipeline.config.ssr_enabled = True
        app.pipeline.config.volumetric_fog_enabled = True
        app.pipeline.config.film_grain_intensity = 0.03
        app.pipeline.config.dof_enabled = False
        if app.engine_tweaks is not None:
            app.engine_tweaks.water_height = 0.05
            app.engine_tweaks.dof_enabled = False

        # 1. Asphalt Ground Plane
        app.set_world_builder(DefaultWorldBuilder(size=120.0, uv_tiles=60.0, material_name="cracked_dry_mud"))

        prop_specs = []

        # --- TOWERING MEGASTRUCTURE ALLEY WALLS ---
        # Left and Right walls forming a tight, vertical neon alleyway corridor
        for z_coord in [-5.0, -15.0, -25.0, -35.0, -45.0, -55.0]:
            # Left skyscraper wall (Dark metallic panels with neon accents)
            prop_specs.append((-8.0, 7.0, z_coord, 4.0, 14.0, 11.0, (0.12, 0.14, 0.18, 1.0), 0.4, 0.7))
            # Right skyscraper wall
            prop_specs.append((8.0, 7.0, z_coord, 4.0, 14.0, 11.0, (0.12, 0.14, 0.18, 1.0), 0.4, 0.7))

        # Overhanging bridges / AC ventilation units across the alley
        bridges = [
            (0.0, 9.0, -12.0, 12.0, 1.5, 3.0, (0.20, 0.22, 0.28, 1.0), 0.5, 0.5),
            (0.0, 11.0, -32.0, 12.0, 1.5, 3.0, (0.20, 0.22, 0.28, 1.0), 0.5, 0.5),
        ]
        for bx, by, bz, bsx, bsy, bsz, bcol, brough, bmetal in bridges:
            prop_specs.append((bx, by, bz, bsx, bsy, bsz, bcol, brough, bmetal))

        # Glowing Neon Sign Kiosks / Emissive Terminals lining the walls
        neon_signs = [
            (-6.2, 5.0, -8.0, 0.6, 2.5, 2.5, (0.95, 0.15, 0.65, 1.0), 0.1, 0.9),  # Neon Pink Sign
            (6.2, 6.0, -20.0, 0.6, 2.5, 2.5, (0.15, 0.85, 0.95, 1.0), 0.1, 0.9),  # Neon Cyan Sign
            (-6.2, 4.5, -38.0, 0.6, 2.5, 2.5, (0.95, 0.85, 0.15, 1.0), 0.1, 0.9), # Neon Yellow Sign
        ]
        for nx, ny, nz, nsx, nsy, nsz, ncol, nrough, nmetal in neon_signs:
            prop_specs.append((nx, ny, nz, nsx, nsy, nsz, ncol, nrough, nmetal))

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

        # 2. Spawn Dynamic PBR Cargo Crates & Drones in the alley
        dynamic_crates = [
            (-2.0, 0.6, -10.0, (0.85, 0.35, 0.15, 1.0)),
            (2.0, 0.6, -22.0, (0.15, 0.65, 0.85, 1.0)),
            (0.0, 0.6, -45.0, (0.75, 0.75, 0.80, 1.0)),
        ]
        self._total_props_count += len(dynamic_crates)
        for dx, dy, dz, dcol in dynamic_crates:
            did = app.ecs.create_entity(
                position=(dx, dy, dz),
                scale=(1.2, 1.2, 1.2),
                color=dcol,
                roughness=0.25,
                metallic=0.8,
                is_static=False,
            )
            app.physics.create_body(did, body_type="dynamic", position=(dx, dy, dz))
            app.physics.attach_box_collider(did, half_x=0.6, half_y=0.6, half_z=0.6, density=2.0)

        # 3. Spawn Player Entity and Kinematic Character Controller at Alley Entrance (Z: +8.0)
        self.player_id = app.ecs.create_entity(
            position=(0.0, 1.0, 8.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.18, 0.88, 1.0, 1.0),  # Cybernetic operative suit
            roughness=0.2,
            metallic=0.7,
            is_static=False,
        )
        motor_cfg = CharacterMotorConfig(
            walk_speed=6.0,
            run_speed=11.5,
            jump_force=8.5,
            step_height=0.40,
            snap_to_ground=0.30,
        )
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=motor_cfg, initial_position=(0.0, 1.0, 8.0))

        # 4. Attach Follow Camera
        cam_cfg = FollowCameraConfig(distance=4.0, target_offset_y=0.65)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=0.0, initial_pitch_deg=15.0)
        self.camera.update_follow(0.016, (0.0, 1.0, 8.0), physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("cyber_cam", self.camera, make_active=True)

        # 5. Initialize HUD
        self.hud = CyberpunkHUD()

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
        fwd = (1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0) - (1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0)
        strafe = (1.0 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 0.0) - (1.0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 0.0)
        sprint_key = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        pad_move = app.input_manager.get_vector2("move")
        if abs(pad_move[0]) > 0.05 or abs(pad_move[1]) > 0.05:
            strafe = pad_move[0]
            fwd = pad_move[1]

        pad_sprint = (
            app.input_manager.is_action_down("sprint")
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_STICK)
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_BUMPER)
        )
        if app.input_manager.is_action_pressed("jump") or app.input_manager.is_gamepad_pressed(GamepadButton.A):
            self._jump_requested = True

        self._move_forward = fwd
        self._move_strafe = strafe
        self._is_sprinting = sprint_key or pad_sprint

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
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("CyberPlay"):
            return

        move_input = (self._move_forward, self._move_strafe)
        self.motor.update(
            dt=dt,
            move_input=move_input,
            is_running=self._is_sprinting,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("CyberPlay"):
            return

        self.poll_inputs(app, dt)
        pos = self.motor.position
        self.camera.update_follow(dt, pos, physics=app.physics, exclude_entity_id=self.player_id)

        if self.hud is not None and app.ui.active_screen is self.hud:
            self.hud.update_telemetry(
                speed_mps=self.motor.state.horizontal_speed,
                is_grounded=self.motor.state.is_grounded,
                pos=pos,
            )

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if self.camera is None or not app.state_manager.is_in_state("CyberPlay"):
            return False

        if event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = float(event.rel[0]), float(event.rel[1])
            self.camera.handle_mouse_orbit(-rel_x, -rel_y)
            return True
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.handle_zoom(float(event.y))
            return True
        elif event.type == pygame.KEYDOWN:
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
        return False

    def reset_player_position(self) -> None:
        if self.motor is not None:
            self.motor.teleport((0.0, 1.0, 8.0), reset_velocity=True)


# ---------------- Game States ----------------

class CyberMenuState(GameState):
    """Fullscreen cyberpunk menu."""

    name = "CyberMenu"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        screen = UIScreen(name="CyberStartScreen")
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(bg_color_top=(0.02, 0.03, 0.06, 0.96), bg_color_bottom=(0.01, 0.01, 0.03, 0.99), corner_radius=0.0)
        screen.add_child(bg)

        card = _create_cyber_card(x=400, y=140, w=480, h=440, border_accent=(0.18, 0.88, 1.0, 0.7))
        bg.add_child(card)

        title = UILabel("SECTOR 7 - CYBERPUNK ALLEYWAY", x=0, y=30, w=480, h=36, font_size=19, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        card.add_child(title)

        sub = UILabel("Urban Neon & Wet Reflective PBR Showcase", x=0, y=68, w=480, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.8))
        card.add_child(sub)

        btn_start = UIButton(text="JACK INTO ALLEY", x=50, y=120, w=380, h=52, variant="primary", font_size=18, on_click=lambda: app.state_manager.set_state("CyberPlay"))
        card.add_child(btn_start)

        btn_exit = UIButton(text="RETURN TO SYSTEM", x=50, y=190, w=380, h=52, variant="danger", font_size=18, on_click=lambda: app.stop())
        card.add_child(btn_exit)

        tip = UILabel("Scene Highlights:\n• Towering vertical skyscraper alley walls\n• Glowing emissive neon signs & Bloom\n• Wet asphalt ground reflections (SSR)\n• 60 Hz Kinematic Character Motor", x=50, y=260, w=380, h=140, font_size=11, color=(0.65, 0.72, 0.82, 0.8))
        card.add_child(tip)

        app.ui.set_screen(screen)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class CyberPlayState(GameState):
    """Active cyberpunk simulation state."""

    name = "CyberPlay"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        mod = app.get_module(CyberpunkAlleyModule)
        if mod and mod.hud:
            app.ui.set_screen(mod.hud)

    def on_resume(self, app: ProjectApp) -> None:
        app.input_manager.reset()
        mod = app.get_module(CyberpunkAlleyModule)
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
            app.state_manager.push_state("CyberPaused")
            return
        mod = app.get_module(CyberpunkAlleyModule)
        if mod:
            mod.poll_inputs(app, dt)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("CyberPaused")
            return True
        mod = app.get_module(CyberpunkAlleyModule)
        if mod:
            return mod.on_event(app, event)
        return False


class CyberPausedState(GameState):
    """Modal pause state."""

    name = "CyberPaused"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        def resume():
            app.state_manager.pop_state()

        def reset_pos():
            mod = app.get_module(CyberpunkAlleyModule)
            if mod:
                mod.reset_player_position()
            app.state_manager.pop_state()

        def main_menu():
            app.state_manager.set_state("CyberMenu")

        app.ui.set_screen(CyberPauseMenuUI(on_resume=resume, on_reset=reset_pos, on_menu=main_menu))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if (
            (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            or (event.type == pygame.CONTROLLERBUTTONDOWN and event.button in (getattr(pygame, "CONTROLLER_BUTTON_START", 6), getattr(pygame, "CONTROLLER_BUTTON_B", 1)))
        ):
            app.state_manager.pop_state()
            return True
        return False


# ---------------- Application Main Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Cyberpunk Neon Alley Example")
    parser.add_argument("--headless", action="store_true", help="Run without opening display window")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Cyberpunk Neon Alley & SSR Showcase",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(CyberpunkAlleyModule())

    app.state_manager.register(CyberMenuState())
    app.state_manager.register(CyberPlayState())
    app.state_manager.register(CyberPausedState())

    app.state_manager.set_state("CyberMenu")

    try:
        if args.headless and args.frames:
            step_count = max(1, args.frames // 3)
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.set_state("CyberPlay")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.push_state("CyberPaused")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
        else:
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()