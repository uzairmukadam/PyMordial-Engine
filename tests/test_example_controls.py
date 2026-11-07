"""Interaction tests for controls shared by the standalone examples."""

from __future__ import annotations

import importlib
import json
import numpy as np
import pygame
import sys
from pathlib import Path
from types import SimpleNamespace

from examples._common import FlyCameraControls
from engine.camera import FreeFlyCamera
from engine.ui import UIManager


def test_flycam_controls_look_speed_and_reset() -> None:
    camera = FreeFlyCamera(position=(0.0, 2.0, 10.0), yaw_deg=-90.0, pitch_deg=0.0)
    controls = FlyCameraControls(camera)
    class InputManager:
        is_mouse_grabbed = False

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


def test_platformer_level_is_data_driven_and_detailed() -> None:
    level_path = Path(__file__).parent.parent / "examples" / "platformer_level.json"
    level = json.loads(level_path.read_text(encoding="utf-8"))
    assert len(level["platforms"]) >= 18
    assert len(level["decorations"]) >= 4
    assert len(level["lights"]) >= 2
    assert level["spawn"]
    assert level["fall_reset_y"] < level["spawn"][1]


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
