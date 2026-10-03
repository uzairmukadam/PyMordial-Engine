"""PyMordial Engine Example 02: Game State Management & ModernGL UI.

Demonstrates how to build game menus, modal overlays, and manage transitions
between Main Menu, Gameplay, and Pause states using PyMordial's native
UIManager and GameStateManager.

Usage:
    python examples/02_game_states_and_ui.py
    python examples/02_game_states_and_ui.py --headless --frames 120
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
    GameState,
    FreeCamera,
    UIScreen,
    UIPanel,
    UILabel,
    UIButton,
    UISlider,
    UIStyle,
)


class MainMenuUI(UIScreen):
    """Fullscreen main menu screen created using native ModernGL UI widgets."""

    def __init__(self, on_start, on_quit) -> None:
        super().__init__(name="MainMenuScreen")

        # 1. Dark Gradient Background
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.04, 0.06, 0.09, 0.95),
            bg_color_bottom=(0.02, 0.03, 0.05, 0.98),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(bg)

        # 2. Cyan Header Accent Strip
        accent = UIPanel(x=0, y=0, w=1280, h=4)
        accent.style = UIStyle(
            bg_color_top=(0.15, 0.85, 1.0, 1.0),
            bg_color_bottom=(0.10, 0.65, 0.95, 0.9),
            corner_radius=0.0,
            border_width=0.0,
        )
        bg.add_child(accent)

        # 3. Center Container Card
        card = UIPanel(x=400, y=140, w=480, h=440)
        card.style = UIStyle(
            bg_color_top=(0.08, 0.10, 0.15, 0.90),
            bg_color_bottom=(0.05, 0.07, 0.10, 0.95),
            border_color=(0.18, 0.60, 0.90, 0.5),
            border_width=1.5,
            corner_radius=12.0,
        )
        bg.add_child(card)

        # Title Label
        title = UILabel(
            "PYMORDIAL ENGINE",
            x=0,
            y=30,
            w=480,
            h=40,
            font_size=32,
            bold=True,
            align="center",
            color=(0.15, 0.88, 1.0, 1.0),
        )
        card.add_child(title)

        subtitle = UILabel(
            "Native ModernGL UI & State Manager Demo",
            x=0,
            y=75,
            w=480,
            h=20,
            font_size=14,
            align="center",
            color=(0.7, 0.75, 0.85, 0.8),
        )
        card.add_child(subtitle)

        # Start Game Button
        btn_start = UIButton(
            text="START GAME",
            x=60,
            y=130,
            w=360,
            h=50,
            variant="primary",
            font_size=18,
            on_click=on_start,
        )
        card.add_child(btn_start)

        # Volume Slider
        slider = UISlider(
            label="Master Volume",
            min_val=0.0,
            max_val=100.0,
            val=80.0,
            step=5.0,
            x=60,
            y=210,
            w=360,
            h=50,
            on_change=lambda val: None,
        )
        card.add_child(slider)

        # Quit Button
        btn_quit = UIButton(
            text="QUIT",
            x=60,
            y=310,
            w=360,
            h=50,
            variant="danger",
            font_size=18,
            on_click=on_quit,
        )
        card.add_child(btn_quit)


class PauseMenuUI(UIScreen):
    """Modal pause dialog overlay."""

    def __init__(self, on_resume, on_quit_to_menu) -> None:
        super().__init__(name="PauseMenuScreen")

        # Dimmed backdrop
        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(
            bg_color_top=(0.0, 0.0, 0.0, 0.65),
            bg_color_bottom=(0.0, 0.0, 0.0, 0.75),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(backdrop)

        # Modal Window
        modal = UIPanel(x=440, y=200, w=400, h=300)
        modal.style = UIStyle(
            bg_color_top=(0.10, 0.12, 0.18, 0.95),
            bg_color_bottom=(0.06, 0.08, 0.12, 0.98),
            border_color=(1.0, 0.75, 0.20, 0.6),
            border_width=1.5,
            corner_radius=10.0,
        )
        backdrop.add_child(modal)

        lbl = UILabel("GAME PAUSED", x=0, y=25, w=400, h=35, font_size=26, bold=True, align="center", color=(1.0, 0.85, 0.3, 1.0))
        modal.add_child(lbl)

        btn_resume = UIButton(
            text="RESUME",
            x=50,
            y=90,
            w=300,
            h=46,
            variant="primary",
            font_size=16,
            on_click=on_resume,
        )
        modal.add_child(btn_resume)

        btn_menu = UIButton(
            text="MAIN MENU",
            x=50,
            y=160,
            w=300,
            h=46,
            variant="danger",
            font_size=16,
            on_click=on_quit_to_menu,
        )
        modal.add_child(btn_menu)

        hint = UILabel("Press ESC to Resume", x=0, y=235, w=400, h=20, font_size=13, align="center", color=(0.6, 0.65, 0.75, 0.7))
        modal.add_child(hint)


# ---------------- State Implementations ----------------

class MainMenuState(GameState):
    """Main menu state: halts physics, displays UI, enables mouse cursor."""

    name = "MainMenu"
    allow_fixed_update = False
    show_cursor = True
    grab_mouse = False

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        def on_start():
            app.state_manager.set_state("Gameplay")

        def on_quit():
            app.is_running = False

        app.ui.set_screen(MainMenuUI(on_start=on_start, on_quit=on_quit))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class GameplayState(GameState):
    """Active 3D gameplay state: runs simulation, locks mouse, listens for ESC to pause."""

    name = "Gameplay"
    allow_fixed_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        # Spawn some sample entities if none exist
        if app.ecs.active_count == 0:
            for x in (-3, 0, 3):
                app.ecs.create_entity(
                    position=(x * 2.0, 1.0, 0.0),
                    color=(0.1 + abs(x) * 0.2, 0.6, 0.9, 1.0),
                    roughness=0.2,
                    metallic=0.8,
                )

        cam = FreeCamera(position=(0.0, 4.0, 8.0), yaw_deg=-90.0, pitch_deg=-20.0)
        app.camera_manager.register_camera("game_cam", cam, make_active=True)

    def on_pause(self, app: ProjectApp) -> None:
        # Called when PauseState is pushed on top
        pass

    def on_resume(self, app: ProjectApp) -> None:
        # Called when returning from pause
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("Paused")
            return True
        return False


class PausedState(GameState):
    """Modal paused state: freezes physics while displaying interactive pause screen."""

    name = "Paused"
    allow_fixed_update = False  # Automatically halts physics simulation!
    show_cursor = True
    grab_mouse = False

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        def on_resume():
            app.state_manager.pop_state()

        def on_menu():
            app.state_manager.set_state("MainMenu")

        app.ui.set_screen(PauseMenuUI(on_resume=on_resume, on_quit_to_menu=on_menu))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.pop_state()
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial UI & State Manager Example")
    parser.add_argument("--headless", action="store_true", help="Run without opening a window")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to run before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - UI & State Manager",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)

    # Register states with the central GameStateManager
    app.state_manager.register(MainMenuState())
    app.state_manager.register(GameplayState())
    app.state_manager.register(PausedState())

    # Start in MainMenu
    app.state_manager.set_state("MainMenu")

    try:
        if args.headless and args.frames:
            # In headless test mode, test transitions across states
            for _ in range(args.frames // 3):
                app.step_frame(1.0 / 60.0)
            app.state_manager.set_state("Gameplay")
            for _ in range(args.frames // 3):
                app.step_frame(1.0 / 60.0)
            app.state_manager.push_state("Paused")
            for _ in range(args.frames // 3):
                app.step_frame(1.0 / 60.0)
        else:
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
