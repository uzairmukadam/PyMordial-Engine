"""Use a native main menu to enter and leave a small flycam level."""

from __future__ import annotations

import pygame

from _common import FlyCameraControls, add_box, create_app, create_parser
from engine.app.module import ProjectModule
from engine.camera import FreeFlyCamera
from engine.ui import UIButton, UILabel, UIPanel, UIScreen


class MenuLevel(ProjectModule):
    """Owns the menu/level flow and enables flight only during gameplay."""

    name = "MenuLevel"

    def __init__(self) -> None:
        self.fly_controls: FlyCameraControls | None = None

    def on_attach(self, app) -> None:
        add_box(app, (0.0, -0.3, 0.0), (32.0, 0.6, 28.0), (0.14, 0.19, 0.22), 0.9)
        add_box(app, (0.0, 2.5, -12.0), (28.0, 5.0, 0.7), (0.32, 0.36, 0.35), 0.8)
        add_box(app, (-8.0, 1.2, -2.0), (3.0, 2.4, 3.0), (0.78, 0.25, 0.13), 0.4, 0.1)
        add_box(app, (0.0, 1.6, -5.0), (2.0, 3.2, 2.0), (0.12, 0.57, 0.79), 0.18, 0.3)
        add_box(app, (8.0, 1.2, -2.0), (3.0, 2.4, 3.0), (0.78, 0.56, 0.16), 0.48, 0.2)
        camera = FreeFlyCamera(position=(0.0, 4.5, 14.0), yaw_deg=-90.0, pitch_deg=-10.0, move_speed=7.0)
        app.camera_manager.register_camera("menu_level", camera, make_active=True)
        self.fly_controls = FlyCameraControls(camera)
        self.fly_controls.set_active(app, False)
        app.add_module(self.fly_controls)
        self._show_main_menu(app)

    def _screen(self, name: str, title: str, subtitle: str, buttons: tuple[tuple[str, object], ...]) -> UIScreen:
        screen = UIScreen(name=name, is_modal=True)
        panel = screen.add_child(UIPanel(x=390.0, y=145.0, w=500.0, h=430.0))
        panel.add_child(UILabel(title, x=32.0, y=40.0, w=436.0, h=54.0, font_size=32, align="center"))
        panel.add_child(UILabel(subtitle, x=42.0, y=108.0, w=416.0, h=48.0, font_size=17, align="center"))
        for index, (text, callback) in enumerate(buttons):
            panel.add_child(
                UIButton(
                    text,
                    x=140.0,
                    y=190.0 + index * 64.0,
                    w=220.0,
                    h=48.0,
                    variant="accent" if index == 0 else "neutral",
                    on_click=callback,
                )
            )
        return screen

    def _show_main_menu(self, app) -> None:
        if self.fly_controls is not None:
            self.fly_controls.set_active(app, False)
        app.ui.cursor_visible = True
        app.ui.set_screen(
            self._screen(
                "MainMenu",
                "SUNLIT RUINS",
                "A small level with a real, interactive menu flow.",
                (
                    ("Start exploring", lambda: self._start_level(app)),
                    ("Controls", lambda: self._show_controls(app)),
                    ("Quit", app.stop),
                ),
            )
        )

    def _show_controls(self, app) -> None:
        app.ui.set_screen(
            self._screen(
                "Controls",
                "CONTROLS",
                "WASD / left stick move, Q/E or A/B fly. Right stick / mouse look, Shift / LB boost. Start pauses.",
                (("Back to menu", lambda: self._show_main_menu(app)),),
            )
        )

    def _start_level(self, app) -> None:
        app.ui.set_screen(None)
        app.ui.cursor_visible = False
        if self.fly_controls is not None:
            self.fly_controls.set_active(app, True)

    def _show_pause(self, app) -> None:
        if self.fly_controls is not None:
            self.fly_controls.set_active(app, False)
        app.ui.cursor_visible = True
        app.ui.set_screen(
            self._screen(
                "PauseMenu",
                "PAUSED",
                "Resume exploring or return to the title screen.",
                (
                    ("Resume", lambda: self._start_level(app)),
                    ("Main menu", lambda: self._show_main_menu(app)),
                ),
            )
        )

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.fly_controls is not None and self.fly_controls.enabled:
                self._show_pause(app)
            else:
                self._show_main_menu(app)
            return True
        return False

    def on_update(self, app, dt: float) -> None:
        if not app.input_manager.is_action_pressed("pause"):
            return
        screen = app.ui.active_screen
        if self.fly_controls is not None and self.fly_controls.enabled:
            self._show_pause(app)
        elif screen is not None and screen.name == "PauseMenu":
            self._start_level(app)
        elif screen is not None and screen.name == "MainMenu":
            self._start_level(app)


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Menu system").parse_args(argv)
    app = create_app(args, "PyMordial | Menu System")
    app.add_module(MenuLevel())
    app.run()


if __name__ == "__main__":
    main()
