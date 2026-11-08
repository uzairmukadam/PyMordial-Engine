"""Shared command-line, camera, and player helpers for the engine examples."""

from __future__ import annotations

import argparse
import pygame

from engine.app.config import ProjectConfig
from engine.app.module import ProjectModule
from engine.app.project_app import ProjectApp
from engine.camera import FollowCamera, FollowCameraConfig, FreeFlyCamera
from engine.input.codes import GamepadButton
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig


class FlyCameraControls(ProjectModule):
    """WASD/QE free flight with automatically captured relative mouse look."""

    name = "ExampleFlyCameraControls"

    def __init__(self, camera: FreeFlyCamera) -> None:
        self.camera = camera
        self.start_position = camera.position.copy()
        self.start_yaw = camera.yaw_deg
        self.start_pitch = camera.pitch_deg
        self.enabled = True
        self._captured = False

    def on_attach(self, app) -> None:
        app.input_manager.bind_action(
            "example_fly_up",
            keys=[pygame.K_e],
            gamepad_buttons=[GamepadButton.A],
        )
        app.input_manager.bind_action(
            "example_fly_down",
            keys=[pygame.K_q],
            gamepad_buttons=[GamepadButton.B],
        )
        app.input_manager.bind_action(
            "example_reset",
            keys=[pygame.K_HOME],
            gamepad_buttons=[GamepadButton.X],
        )
        if self.enabled:
            self.capture(app)

    def on_detach(self, app) -> None:
        self.release(app)

    def set_active(self, app, active: bool) -> None:
        self.enabled = active
        if active:
            self.capture(app)
        else:
            self.release(app)

    def capture(self, app) -> None:
        if self.enabled and not app.config.headless:
            app.input_manager.set_mouse_grab(True)
            self._captured = True

    def release(self, app) -> None:
        if self._captured:
            app.input_manager.set_mouse_grab(False)
        self._captured = False

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.release(app)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and not self._captured:
            self.capture(app)
            return True
        if event.type == pygame.MOUSEMOTION and self._captured:
            self.camera.handle_mouse_look(-float(event.rel[0]), -float(event.rel[1]))
            return True
        if event.type == pygame.MOUSEWHEEL:
            self.camera.move_speed = max(1.0, min(80.0, self.camera.move_speed * (1.0 + event.y * 0.1)))
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_HOME:
            self.camera.position[:] = self.start_position
            self.camera.yaw_deg = self.start_yaw
            self.camera.pitch_deg = self.start_pitch
            self.camera._update_vectors()
            return True
        if event.type == pygame.WINDOWFOCUSLOST:
            self.release(app)
        return False

    def on_update(self, app, dt: float) -> None:
        if not self.enabled:
            return
        if self._captured and not app.input_manager.is_mouse_grabbed:
            self._captured = False
        input_mgr = app.input_manager
        forward = input_mgr.get_axis("move_forward")
        right = input_mgr.get_axis("move_right")
        up = float(input_mgr.is_action_down("example_fly_up")) - float(input_mgr.is_action_down("example_fly_down"))
        look_yaw, look_pitch = input_mgr.get_vector2("look")
        if look_yaw or look_pitch:
            self.camera.handle_mouse_look(look_yaw * 600.0 * dt, look_pitch * 600.0 * dt)
        if input_mgr.is_action_pressed("example_reset"):
            self.camera.position[:] = self.start_position
            self.camera.yaw_deg = self.start_yaw
            self.camera.pitch_deg = self.start_pitch
            self.camera._update_vectors()
        boost = input_mgr.is_action_down("sprint")
        self.camera.move(dt, forward, right, up, boost)


class CapsulePlayerControls(ProjectModule):
    """Third-person capsule motor driven by keyboard and mouse controls."""

    name = "ExampleCapsulePlayerControls"

    def __init__(
        self,
        spawn: tuple[float, float, float] = (0.0, 1.0, 0.0),
        camera_yaw: float = 0.0,
        fall_reset_y: float | None = None,
    ) -> None:
        self.spawn = spawn
        self.camera_yaw = camera_yaw
        self.player_id: int | None = None
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self._jump_requested = False
        self._reset_requested = False
        self._captured = False
        self.fall_reset_y = fall_reset_y

    def on_attach(self, app) -> None:
        app.input_manager.bind_action(
            "example_reset",
            keys=[pygame.K_r],
            gamepad_buttons=[GamepadButton.X],
        )
        self.player_id = app.ecs.create_entity(
            shape="capsule",
            position=self.spawn,
            dimensions=(0.8, 1.8, 0.8),
            color=(0.15, 0.72, 0.98),
            roughness=0.28,
            metallic=0.05,
        )
        self.motor = CharacterMotor(
            self.player_id,
            app.physics,
            app.ecs,
            CharacterMotorConfig(
                walk_speed=5.5,
                run_speed=9.0,
                jump_force=7.5,
                capsule_half_height=0.55,
                capsule_radius=0.35,
                step_height=0.35,
                snap_to_ground=0.25,
            ),
            initial_position=self.spawn,
        )
        self.camera = FollowCamera(
            config=FollowCameraConfig(distance=5.5, target_offset_y=0.65),
            initial_yaw_deg=self.camera_yaw,
            initial_pitch_deg=12.0,
        )
        self.camera.update_follow(0.0, self.spawn, app.physics, self.player_id)
        app.camera_manager.register_camera("player", self.camera, make_active=True)
        self.capture(app)

    def on_detach(self, app) -> None:
        self.release(app)

    def capture(self, app) -> None:
        if not app.config.headless:
            app.input_manager.set_mouse_grab(True)
            self._captured = True

    def release(self, app) -> None:
        if self._captured:
            app.input_manager.set_mouse_grab(False)
        self._captured = False

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self._jump_requested = True
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.release(app)
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self._reset_requested = True
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and not self._captured:
            self.capture(app)
            return True
        if event.type == pygame.MOUSEMOTION and self._captured and self.camera is not None:
            self.camera.handle_mouse_orbit(-float(event.rel[0]), -float(event.rel[1]))
            return True
        if event.type == pygame.MOUSEWHEEL and self.camera is not None:
            self.camera.handle_zoom(float(event.y))
            return True
        if event.type == pygame.WINDOWFOCUSLOST:
            self.release(app)
        return False

    def on_fixed_update(self, app, dt: float) -> None:
        if self.motor is None or self.camera is None:
            return
        dense_idx = app.ecs.pool.get_dense_index(self.player_id)
        fell_below_level = (
            self.fall_reset_y is not None
            and dense_idx >= 0
            and float(app.ecs.rigid_body_state[1, dense_idx, 1]) < self.fall_reset_y
        )
        if self._reset_requested or fell_below_level:
            self.motor.set_position(self.spawn)
            self._reset_requested = False
        input_mgr = getattr(app, "input_manager", None)
        if input_mgr is None:
            keys = pygame.key.get_pressed()
            forward = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN])
            right = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
            running = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            jump_requested = self._jump_requested
            reset_pressed = False
        else:
            forward = input_mgr.get_axis("move_forward")
            right = input_mgr.get_axis("move_right")
            running = input_mgr.is_action_down("sprint")
            jump_requested = self._jump_requested or input_mgr.is_action_pressed("jump")
            reset_pressed = input_mgr.is_action_pressed("example_reset")
        if reset_pressed:
            self.motor.set_position(self.spawn)
        if jump_requested:
            self._jump_requested = True
        self.motor.update(
            dt,
            (forward, right),
            is_running=running,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app, dt: float) -> None:
        if self.motor is not None and self.camera is not None:
            if self._captured and not app.input_manager.is_mouse_grabbed:
                self._captured = False
            look_yaw, look_pitch = app.input_manager.get_vector2("look")
            if look_yaw or look_pitch:
                self.camera.handle_mouse_orbit(look_yaw * 600.0 * dt, look_pitch * 600.0 * dt)
            self.camera.update_follow(dt, self.motor.position, app.physics, self.player_id)


def _positive_int(value: str) -> int:
    frames = int(value)
    if frames < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return frames


def create_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="Run without showing a window")
    parser.add_argument("--frames", type=_positive_int, help="Stop after this many frames")
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "ultra", "cinematic"),
        help="Override the example's default quality preset",
    )
    return parser


def create_app(args: argparse.Namespace, title: str) -> ProjectApp:
    return ProjectApp(
        ProjectConfig(
            title=title,
            width=1280,
            height=720,
            quality_preset=args.quality or ("low" if args.headless else "high"),
            enable_debug=not args.headless,
            particles_enabled=False,
            headless=args.headless,
            max_frames=args.frames,
        )
    )


def add_box(
    app: ProjectApp,
    position: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    color: tuple[float, float, float],
    roughness: float = 0.72,
    metallic: float = 0.0,
    *,
    physics: bool = False,
    fixed: bool = True,
) -> int:
    entity = app.ecs.create_entity(
        shape="cube",
        position=position,
        dimensions=dimensions,
        color=color,
        roughness=roughness,
        metallic=metallic,
        is_static=fixed,
    )
    if physics:
        half_x, half_y, half_z = (component * 0.5 for component in dimensions)
        body_type = "fixed" if fixed else "dynamic"
        app.physics.create_body(entity, body_type=body_type, position=position)
        app.physics.attach_box_collider(entity, half_x, half_y, half_z, friction=0.75)
    return entity
