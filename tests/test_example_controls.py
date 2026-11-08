"""Interaction tests for controls shared by the standalone examples."""

from __future__ import annotations

import importlib
import json
import numpy as np
import pygame
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from examples._common import FlyCameraControls
from engine.camera import FreeFlyCamera
from engine.input import InputManager
from engine.input.codes import GamepadButton
from engine.gfx.mega_buffer import MegaBuffer
from engine.ui import UIManager


def test_flycam_controls_look_speed_and_reset() -> None:
    camera = FreeFlyCamera(position=(0.0, 2.0, 10.0), yaw_deg=-90.0, pitch_deg=0.0)
    controls = FlyCameraControls(camera)
    class InputManager:
        is_mouse_grabbed = False

        def bind_action(self, *args, **kwargs) -> None:
            pass

        def get_axis(self, name: str) -> float:
            return 0.0

        def get_vector2(self, name: str) -> tuple[float, float]:
            return (0.0, 0.0)

        def is_action_down(self, name: str) -> bool:
            return False

        def is_action_pressed(self, name: str) -> bool:
            return False

        def set_mouse_grab(self, grabbed: bool) -> None:
            self.is_mouse_grabbed = grabbed

    app = SimpleNamespace(config=SimpleNamespace(headless=False), input_manager=InputManager())
    start_position = camera.position.copy()
    start_speed = camera.move_speed
    controls.on_attach(app)
    assert app.input_manager.is_mouse_grabbed
    controls.on_event(app, pygame.event.Event(pygame.MOUSEMOTION, rel=(100, 25), pos=(100, 25)))
    assert camera.yaw_deg < -90.0
    assert camera.pitch_deg > 0.0

    controls.on_event(app, pygame.event.Event(pygame.MOUSEWHEEL, y=1))
    assert camera.move_speed > start_speed

    controls.on_event(app, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert not app.input_manager.is_mouse_grabbed
    yaw_after_release = camera.yaw_deg
    controls.on_event(app, pygame.event.Event(pygame.MOUSEMOTION, rel=(100, 25), pos=(100, 25)))
    assert camera.yaw_deg == yaw_after_release

    controls.on_event(app, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
    assert app.input_manager.is_mouse_grabbed
    controls.on_event(app, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME))
    assert (camera.position == start_position).all()
    assert camera.yaw_deg == -90.0
    assert camera.pitch_deg == 0.0
    controls.on_detach(app)
    assert not app.input_manager.is_mouse_grabbed


def test_flycam_supports_gamepad_movement_look_and_vertical_flight() -> None:
    camera = FreeFlyCamera(position=(0.0, 0.0, 0.0), yaw_deg=-90.0, pitch_deg=0.0)
    controls = FlyCameraControls(camera)
    input_mgr = InputManager()
    app = SimpleNamespace(config=SimpleNamespace(headless=True), input_manager=input_mgr)
    controls.on_attach(app)

    input_mgr.process_event(pygame.event.Event(pygame.JOYAXISMOTION, axis=1, value=-32767))
    input_mgr.process_event(pygame.event.Event(pygame.JOYAXISMOTION, axis=2, value=32767))
    input_mgr.process_event(pygame.event.Event(pygame.JOYBUTTONDOWN, button=int(GamepadButton.A)))
    input_mgr.update_axes()
    start_yaw = camera.yaw_deg
    controls.on_update(app, 1.0 / 60.0)

    assert camera.position[1] > 0.0
    assert camera.yaw_deg > start_yaw
    assert camera.position[2] < 0.0


def test_menu_example_buttons_open_and_start_level() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))
    menu_example = importlib.import_module("05_menu_system")
    ui = UIManager(ctx=None)
    app = SimpleNamespace(ui=ui, stop=lambda: None)
    try:
        menu = menu_example.MenuLevel()
        menu._show_main_menu(app)
        panel = ui.active_screen.children[0]
        start_button = panel.children[2]
        assert start_button.text == "Start exploring"

        assert start_button.on_mouse_down(640.0, 360.0, 1)
        assert start_button.on_mouse_up(640.0, 360.0, 1)
        assert ui.active_screen is None
        assert ui.cursor_visible is False
    finally:
        sys.path.remove(str(examples_dir))
        sys.modules.pop("05_menu_system", None)


def test_menu_example_gamepad_moves_cursor_and_activates_button() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))
    menu_example = importlib.import_module("05_menu_system")

    class PadInput:
        def get_vector2(self, name: str) -> tuple[float, float]:
            return (1.0, 0.0)

        def is_gamepad_pressed(self, button) -> bool:
            return button == GamepadButton.A

    ui = UIManager(ctx=None)
    ui.cursor_pos = [630.0, 360.0]
    app = SimpleNamespace(
        config=SimpleNamespace(width=1280, height=720),
        ui=ui,
        input_manager=PadInput(),
        stop=lambda: None,
    )
    menu = menu_example.MenuLevel()
    try:
        ui.attach_app(app)
        menu._show_main_menu(app)
        ui.update(0.01)
        assert ui.cursor_pos[0] > 630.0
        assert ui.active_screen is None
        assert ui.cursor_visible is False
    finally:
        sys.path.remove(str(examples_dir))
        sys.modules.pop("05_menu_system", None)


def test_capsule_player_reads_gamepad_axes_and_actions() -> None:
    from examples._common import CapsulePlayerControls

    class PadInput:
        def get_axis(self, name: str) -> float:
            return {"move_forward": 1.0, "move_right": -0.5}[name]

        def is_action_down(self, name: str) -> bool:
            return name == "sprint"

        def is_action_pressed(self, name: str) -> bool:
            return name in ("jump", "example_reset")

    class Motor:
        def __init__(self) -> None:
            self.reset_position = None
            self.update_args = None

        def set_position(self, position) -> None:
            self.reset_position = position

        def update(self, *args, **kwargs) -> None:
            self.update_args = (args, kwargs)

    controls = CapsulePlayerControls(spawn=(1.0, 2.0, 3.0))
    controls.player_id = 4
    controls.motor = Motor()
    controls.camera = SimpleNamespace(yaw_deg=15.0)
    app = SimpleNamespace(
        input_manager=PadInput(),
        ecs=SimpleNamespace(pool=SimpleNamespace(get_dense_index=lambda entity_id: -1)),
    )
    controls.on_fixed_update(app, 1.0 / 60.0)

    assert controls.motor.reset_position == (1.0, 2.0, 3.0)
    args, kwargs = controls.motor.update_args
    assert args[1] == (1.0, -0.5)
    assert kwargs["is_running"] is True
    assert kwargs["jump_requested"] is True


def test_platformer_level_is_data_driven_and_detailed() -> None:
    level_path = Path(__file__).parent.parent / "examples" / "platformer_level.json"
    level = json.loads(level_path.read_text(encoding="utf-8"))
    assert len(level["platforms"]) >= 18
    assert len(level["decorations"]) >= 4
    assert len(level["lights"]) >= 2
    assert level["spawn"]
    assert level["fall_reset_y"] < level["spawn"][1]


def test_last_aqueduct_demo_has_detailed_traversal_and_art_direction() -> None:
    level_path = Path(__file__).parent.parent / "examples" / "last_aqueduct_level.json"
    level = json.loads(level_path.read_text(encoding="utf-8"))

    assert level["name"] == "The Last Aqueduct"
    assert len(level["platforms"]) >= 10
    assert len(level["bridges"]) >= 3
    assert len(level["ramps"]) >= 2
    assert len(level["props"]) >= 50
    assert len(level["water"]) == 1
    assert len(level["lights"]) >= 6
    assert level["fall_reset_y"] < level["spawn"][1]
    assert level["platforms"][-1]["position"][1] > level["platforms"][0]["position"][1]
    assert {prop["shape"] for prop in level["props"]} >= {"cube", "cylinder", "cone", "sphere"}
    assert all(len(platform["position"]) == len(platform["size"]) == 3 for platform in level["platforms"])
    assert all(max(prop["size"]) <= 3.0 for prop in level["props"] if prop["shape"] == "sphere")
    assert all(
        prop["position"][1] - prop["size"][1] * 0.5 >= 0.0
        for prop in level["props"]
        if prop["shape"] == "cone"
    )
    assert all(ramp["rotation_deg"][0] > 0.0 for ramp in level["ramps"])


def test_last_aqueduct_solids_have_physics_colliders() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))
    aqueduct = importlib.import_module("07_the_last_aqueduct")

    level_path = Path(__file__).parent.parent / "examples" / "last_aqueduct_level.json"
    level = json.loads(level_path.read_text(encoding="utf-8"))
    create_entity = Mock(side_effect=range(1, 1000))
    pipeline = SimpleNamespace(
        config=SimpleNamespace(),
        atmosphere=SimpleNamespace(config=SimpleNamespace()),
        lights_pass=SimpleNamespace(clear=Mock()),
        apply_config=Mock(),
        add_point_light=Mock(),
    )
    physics = SimpleNamespace(create_body=Mock(), attach_box_collider=Mock())
    app = SimpleNamespace(
        pipeline=pipeline,
        physics=physics,
        ecs=SimpleNamespace(create_entity=create_entity),
        add_module=Mock(),
    )

    try:
        aqueduct.build_level(app, level_path)

        solid_props = sum(prop.get("solid", prop["shape"] in {"cube", "cylinder"}) for prop in level["props"])
        expected_solids = len(level["platforms"]) + len(level["bridges"]) + len(level["ramps"]) + solid_props
        assert physics.create_body.call_count == expected_solids
        assert physics.attach_box_collider.call_count == expected_solids
    finally:
        sys.path.remove(str(examples_dir))
        sys.modules.pop("07_the_last_aqueduct", None)


def test_builtin_cylinder_mesh_is_y_aligned() -> None:
    mesh = object.__new__(MegaBuffer)
    mesh.ctx = None
    mesh.vbo = None
    mesh.ibo = None
    mesh.allocations = {}
    mesh.vertex_data = []
    mesh.index_data = []
    mesh.total_vertices = 0
    mesh.total_indices = 0
    mesh.on_bake_callbacks = []

    mesh._add_cylinder_primitive(sectors=8)

    positions = mesh.vertex_data[-1]["position"]
    assert np.isclose(positions[:, 0].min(), -1.0)
    assert np.isclose(positions[:, 0].max(), 1.0)
    assert np.isclose(positions[:, 1].min(), -0.5)
    assert np.isclose(positions[:, 1].max(), 0.5)
    assert np.isclose(positions[:, 2].min(), -1.0)
    assert np.isclose(positions[:, 2].max(), 1.0)


def test_capsule_player_resets_after_falling_below_level(monkeypatch) -> None:
    from collections import defaultdict

    from examples._common import CapsulePlayerControls

    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool))

    class Motor:
        def __init__(self) -> None:
            self.reset_position = None

        def set_position(self, position) -> None:
            self.reset_position = position

        def update(self, *args, **kwargs) -> None:
            pass

    controls = CapsulePlayerControls(spawn=(1.0, 2.0, 3.0), fall_reset_y=-10.0)
    controls.player_id = 4
    controls.motor = Motor()
    controls.camera = SimpleNamespace(yaw_deg=0.0)
    controls.fall_reset_y = -10.0
    app = SimpleNamespace(
        ecs=SimpleNamespace(
            pool=SimpleNamespace(get_dense_index=lambda entity_id: 0),
            rigid_body_state=np.asarray(
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.0, -11.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        )
    )

    controls.on_fixed_update(app, 1.0 / 60.0)
    assert controls.motor.reset_position == (1.0, 2.0, 3.0)


def test_capsule_player_manual_reset_runs_on_fixed_update(monkeypatch) -> None:
    from collections import defaultdict

    from examples._common import CapsulePlayerControls

    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool))

    class Motor:
        reset_position = None

        def set_position(self, position) -> None:
            self.reset_position = position

        def update(self, *args, **kwargs) -> None:
            pass

    controls = CapsulePlayerControls(spawn=(1.0, 2.0, 3.0))
    controls.motor = Motor()
    controls.camera = SimpleNamespace(yaw_deg=0.0)
    controls.on_event(None, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    assert controls.motor.reset_position is None

    app = SimpleNamespace(
        ecs=SimpleNamespace(
            pool=SimpleNamespace(get_dense_index=lambda entity_id: -1),
            rigid_body_state=np.zeros((2, 1, 3), dtype=np.float32),
        )
    )
    controls.on_fixed_update(app, 1.0 / 60.0)
    assert controls.motor.reset_position == (1.0, 2.0, 3.0)
