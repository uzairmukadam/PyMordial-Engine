"""PyMordial Engine: Tactical Sci-Fi Facility Complex Level.

Loads and renders the multi-room facility map defined in facility_complex_map.json.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import pygame

# Ensure project root is in sys.path
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
    UIStyle,
)
from engine.world.default_world import DefaultWorldBuilder


class FacilityHUD(UIScreen):
    """In-Game Tactical HUD for Facility Complex."""

    def __init__(self) -> None:
        super().__init__(name="FacilityHUD", is_modal=False)

        panel = UIPanel(x=24, y=20, w=420, h=110)
        panel.style = UIStyle(
            bg_color_top=(0.04, 0.08, 0.14, 0.92),
            bg_color_bottom=(0.02, 0.04, 0.08, 0.96),
            border_color=(0.18, 0.75, 1.0, 0.6),
            border_width=1.5,
            corner_radius=8.0,
        )
        self.add_child(panel)

        t1 = UILabel("TACTICAL FACILITY COMPLEX", x=16, y=10, w=388, h=22, font_size=13, bold=True, color=(0.18, 0.88, 1.0, 1.0))
        panel.add_child(t1)

        self.status_lbl = UILabel("SECTOR: Helipad Spawn (+90m Z)", x=16, y=34, w=388, h=20, font_size=11, color=(0.20, 0.95, 0.60, 0.95))
        panel.add_child(self.status_lbl)

        ctrls = UILabel("• WASD: Walk  • Shift: Sprint  • Space: Jump  • ESC: Exit", x=16, y=60, w=388, h=20, font_size=10, color=(0.7, 0.8, 0.9, 0.85))
        panel.add_child(ctrls)

    def update_sector(self, z_pos: float) -> None:
        if z_pos > 75.0:
            sec = "Helipad LZ (North)"
        elif z_pos > 40.0:
            sec = "NW Staging Chamber"
        elif z_pos > 0.0:
            sec = "Central Control Hub & Courtyard"
        elif z_pos > -35.0:
            sec = "SW Research Lab & East Mezzanine"
        else:
            sec = "South Extraction Zone"
        self.status_lbl.text = f"SECTOR: {sec} ({z_pos:.1f}m Z)"


class FacilityLevelModule(ProjectModule):
    """Parses JSON facility data and handles player kinematics & lights."""

    name = "FacilityLevelModule"

    def __init__(self, map_path: str | Path) -> None:
        super().__init__()
        self.map_path = Path(map_path)
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self.hud: FacilityHUD | None = None

        self._start_pos = (0.0, 1.5, 90.0)
        self._first_prop_id = -1
        self._props_count = 0
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._is_sprinting = False

    def on_attach(self, app: ProjectApp) -> None:
        with open(self.map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        settings = data.get("settings", {})
        app.pipeline.config.gi_mode = settings.get("gi_mode", "RADIANCE_CASCADES")
        app.pipeline.config.ao_mode = settings.get("ao_mode", "GTAO")
        app.pipeline.config.bloom_enabled = settings.get("bloom", True)
        app.pipeline.config.ssr_enabled = settings.get("ssr", True)
        app.pipeline.config.dof_enabled = False

        app.set_world_builder(DefaultWorldBuilder(
            size=settings.get("world_size", 800.0),
            uv_tiles=settings.get("uv_tiles", 80.0),
            material_name=settings.get("material_name", "concrete_clean")
        ))

        # Parse Entities
        props = data.get("props", [])
        self._props_count = len(props)
        for i, p in enumerate(props):
            px, py, pz = p["position"]
            sx, sy, sz = p["scale"]
            col = tuple(p["color"])

            eid = app.ecs.create_entity(
                position=(px, py, pz),
                scale=(sx, sy, sz),
                color=col,
                roughness=p.get("roughness", 0.5),
                metallic=p.get("metallic", 0.4),
                is_static=True,
            )
            if self._first_prop_id == -1:
                self._first_prop_id = eid

            app.physics.create_body(eid, body_type="fixed", position=(px, py, pz))
            app.physics.attach_box_collider(eid, half_x=sx * 0.5, half_y=sy * 0.5, half_z=sz * 0.5)

        # Player Spawn
        p_spawn = data.get("player_spawn", {})
        self._start_pos = tuple(p_spawn.get("position", [0.0, 1.5, 90.0]))
        self.player_id = app.ecs.create_entity(
            position=self._start_pos,
            scale=(1.0, 1.0, 1.0),
            color=tuple(p_spawn.get("color", [0.2, 0.85, 1.0, 1.0])),
            roughness=0.2,
            metallic=0.7,
            is_static=False,
        )

        motor_cfg = CharacterMotorConfig(
            walk_speed=p_spawn.get("walk_speed", 6.5),
            run_speed=p_spawn.get("run_speed", 11.5),
            jump_force=p_spawn.get("jump_force", 8.5),
            step_height=0.40,
        )
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=motor_cfg, initial_position=self._start_pos)

        # Follow Camera
        cam_cfg = FollowCameraConfig(distance=4.8, target_offset_y=0.7)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=180.0, initial_pitch_deg=15.0)
        self.camera.update_follow(0.016, self._start_pos, physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("facility_cam", self.camera, make_active=True)

        # Clustered Lights
        pipe = app.pipeline
        if hasattr(pipe, "lights_pass"):
            pipe.lights_pass.clear()
            for l in data.get("point_lights", []):
                pipe.add_point_light(
                    position=tuple(l["position"]),
                    radius=l.get("radius", 30.0),
                    color=tuple(l["color"]),
                    intensity=l.get("intensity", 8.0)
                )

        self.hud = FacilityHUD()

    def get_draw_batches(self, app: ProjectApp) -> list[tuple]:
        alloc_cube = app.pipeline.mega_buffer.allocations["cube"]
        alloc_capsule = app.pipeline.mega_buffer.allocations["capsule"]
        d_prop = app.ecs.pool.get_dense_index(self._first_prop_id)
        d_capsule = app.ecs.pool.get_dense_index(self.player_id)
        return [
            (alloc_cube, self._props_count, d_prop, False, True),
            (alloc_capsule, 1, d_capsule, False, True),
        ]

    def poll_inputs(self, app: ProjectApp, dt: float) -> None:
        keys = pygame.key.get_pressed()
        fwd = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        strafe = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        if keys[pygame.K_SPACE] or app.input_manager.is_action_pressed("jump"):
            self._jump_requested = True
        self._move_forward = fwd
        self._move_strafe = strafe
        self._is_sprinting = bool(keys[pygame.K_LSHIFT])

        look_yaw = app.input_manager.get_axis("look_yaw")
        look_pitch = app.input_manager.get_axis("look_pitch")
        if (abs(look_yaw) > 0.02 or abs(look_pitch) > 0.02) and self.camera is not None:
            self.camera.yaw_deg += look_yaw * 120.0 * dt
            self.camera.pitch_deg = max(-80.0, min(80.0, self.camera.pitch_deg - look_pitch * 90.0 * dt))
            self.camera.mark_dirty()

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("FacilityPlay"):
            return
        self.motor.update(
            dt=dt,
            move_input=(self._move_forward, self._move_strafe),
            is_running=self._is_sprinting,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("FacilityPlay"):
            return
        self.poll_inputs(app, dt)
        pos = self.motor.position
        self.camera.update_follow(dt, pos, physics=app.physics, exclude_entity_id=self.player_id)
        if self.hud is not None and app.ui.active_screen is self.hud:
            self.hud.update_sector(pos[2])

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION and self.camera is not None:
            self.camera.handle_mouse_orbit(-float(event.rel[0]), -float(event.rel[1]))
            return True
        return False


class FacilityPlayState(GameState):
    name = "FacilityPlay"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        mod = app.get_module(FacilityLevelModule)
        if mod and mod.hud:
            app.ui.set_screen(mod.hud)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.stop()
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Facility Complex")
    parser.add_argument("--map", type=str, default="examples\platformer_map.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig(title="PyMordial - Tactical Facility Complex", width=1280, height=720, headless=args.headless, max_frames=args.frames)
    app = ProjectApp(config=config)
    app.add_module(FacilityLevelModule(map_path=args.map))
    app.state_manager.register(FacilityPlayState())
    app.state_manager.set_state("FacilityPlay")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()