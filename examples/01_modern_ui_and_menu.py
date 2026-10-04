"""PyMordial Engine Example 01: Modern AAA UI, Menus & State Management.

Demonstrates:
1. Modern glassmorphic ModernGL UI screens with dual-tone obsidian gradients and glowing borders.
2. Interactive widgets (buttons, dragging numeric sliders, segmented option tabs).
3. Stack-based GameStateManager transitions (MainMenu -> Simulation -> Pause Overlay -> Settings).
4. Live 3D exhibition scene with free-fly 6-DOF camera and responsive gameplay HUD.

Zero debug menu dependency. All state and UI logic runs through public engine APIs.

Usage:
    python examples/01_modern_ui_and_menu.py
    python examples/01_modern_ui_and_menu.py --headless --frames 60
"""

from __future__ import annotations
import argparse
import math
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
    FreeCamera,
    UIScreen,
    UIPanel,
    UILabel,
    UIButton,
    UISlider,
    UISegmentGroup,
    UIStyle,
)
from engine.input.codes import GamepadButton


# ---------------- Modern AAA Glassmorphic UI Screens ----------------

def _create_glass_card(
    x: float,
    y: float,
    w: float,
    h: float,
    border_accent: tuple[float, float, float, float] = (0.18, 0.65, 0.95, 0.6),
) -> UIPanel:
    """Creates a reusable dark obsidian glass card with rounded corners and glowing border."""
    panel = UIPanel(x=x, y=y, w=w, h=h)
    panel.style = UIStyle(
        bg_color_top=(0.06, 0.08, 0.13, 0.92),
        bg_color_bottom=(0.02, 0.03, 0.06, 0.97),
        border_color=border_accent,
        border_width=1.5,
        corner_radius=12.0,
    )
    return panel


class MainMenuUI(UIScreen):
    """Fullscreen AAA Main Menu featuring glass panels, category badges, and widgets."""

    def __init__(self, on_start: Any, on_settings: Any, on_quit: Any) -> None:
        super().__init__(name="MainMenuScreen")

        # 1. Fullscreen subtle dark gradient backdrop
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.03, 0.04, 0.07, 0.94),
            bg_color_bottom=(0.01, 0.02, 0.04, 0.98),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(bg)

        # 2. Glowing Top Accent Bar
        top_accent = UIPanel(x=0, y=0, w=1280, h=4)
        top_accent.style = UIStyle(
            bg_color_top=(0.15, 0.85, 1.0, 1.0),
            bg_color_bottom=(0.10, 0.65, 0.95, 0.8),
            corner_radius=0.0,
        )
        bg.add_child(top_accent)

        # 3. Header Card
        header = _create_glass_card(x=80, y=60, w=1120, h=110, border_accent=(0.15, 0.80, 1.0, 0.4))
        bg.add_child(header)

        title = UILabel(
            "PYMORDIAL ENGINE",
            x=35,
            y=18,
            w=800,
            h=42,
            font_size=34,
            bold=True,
            color=(0.18, 0.88, 1.0, 1.0),
        )
        header.add_child(title)

        subtitle = UILabel(
            "MODERN AAA UI SUBSYSTEM  •  PBR SHADING  •  STATE MACHINE ARCHITECTURE",
            x=37,
            y=64,
            w=800,
            h=24,
            font_size=13,
            bold=True,
            color=(0.95, 0.75, 0.20, 1.0),
        )
        header.add_child(subtitle)

        version_badge = UILabel("v0.1.0 Beta", x=900, y=64, w=180, h=24, font_size=14, align="right", color=(0.6, 0.7, 0.8, 0.8))
        header.add_child(version_badge)

        # 4. Left Column: Action Buttons Card (x=80, y=190, w=480, h=470)
        menu_card = _create_glass_card(x=80, y=190, w=480, h=470)
        bg.add_child(menu_card)

        sec_title = UILabel("CAMPAIGN & ACTIONS", x=30, y=20, w=420, h=28, font_size=14, bold=True, color=(0.4, 0.8, 1.0, 0.9))
        menu_card.add_child(sec_title)

        btn_start = UIButton(
            text="ENTER SIMULATION",
            x=30,
            y=60,
            w=420,
            h=56,
            variant="primary",
            font_size=18,
            on_click=on_start,
        )
        menu_card.add_child(btn_start)

        btn_settings = UIButton(
            text="ENGINE SETTINGS",
            x=30,
            y=130,
            w=420,
            h=54,
            variant="neutral",
            font_size=16,
            on_click=on_settings,
        )
        menu_card.add_child(btn_settings)

        btn_quit = UIButton(
            text="EXIT ENGINE",
            x=30,
            y=200,
            w=420,
            h=54,
            variant="danger",
            font_size=16,
            on_click=on_quit,
        )
        menu_card.add_child(btn_quit)

        hint = UILabel("Press ESC during simulation to return to menu.", x=30, y=410, w=420, h=24, font_size=12, color=(0.55, 0.65, 0.75, 0.6))
        menu_card.add_child(hint)

        # 5. Right Column: Settings & Widgets Showcase (x=590, y=190, w=610, h=470)
        showcase_card = _create_glass_card(x=590, y=190, w=610, h=470, border_accent=(0.95, 0.70, 0.20, 0.4))
        bg.add_child(showcase_card)

        sh_title = UILabel("LIVE INTERACTIVE WIDGETS", x=30, y=20, w=550, h=28, font_size=14, bold=True, color=(1.0, 0.8, 0.25, 1.0))
        showcase_card.add_child(sh_title)

        # Slider 1: Master Volume
        self.slider_vol = UISlider(
            label="Master Audio Gain",
            min_val=0,
            max_val=100,
            val=85,
            x=30,
            y=65,
            w=550,
            h=44,
            format_str="{label}: {val:.0f}%",
        )
        showcase_card.add_child(self.slider_vol)

        # Slider 2: Camera FOV
        self.slider_fov = UISlider(
            label="Horizontal FOV",
            min_val=60,
            max_val=120,
            val=90,
            x=30,
            y=135,
            w=550,
            h=44,
            format_str="{label}: {val:.0f} deg",
        )
        showcase_card.add_child(self.slider_fov)

        # Segmented Group: Difficulty / Mode
        lbl_seg = UILabel("SIMULATION PRESET", x=30, y=210, w=550, h=22, font_size=13, bold=True, color=(0.7, 0.75, 0.85, 0.85))
        showcase_card.add_child(lbl_seg)

        self.seg_group = UISegmentGroup(
            options=[
                ("Standard", "std"),
                ("High Fidelity", "hi"),
                ("Competitive", "comp"),
                ("Cinematic", "cine"),
            ],
            selected_index=1,
            x=30,
            y=236,
            w=550,
            h=40,
        )
        showcase_card.add_child(self.seg_group)

        # Informational Badge
        info_panel = UIPanel(x=30, y=300, w=550, h=130)
        info_panel.style = UIStyle(
            bg_color_top=(0.08, 0.10, 0.16, 0.8),
            bg_color_bottom=(0.04, 0.05, 0.09, 0.9),
            border_color=(0.2, 0.5, 0.8, 0.3),
            border_width=1.0,
            corner_radius=8.0,
        )
        showcase_card.add_child(info_panel)

        badge_txt = UILabel(
            "Architecture Highlights:\n• Native ModernGL shaders with zero-allocation batch quad rendering\n• Complete separation between engine infrastructure and game UI\n• Stack-based GameStateManager coordinating transitions and pause lifecycles",
            x=18,
            y=15,
            w=514,
            h=100,
            font_size=12,
            color=(0.75, 0.82, 0.92, 0.85),
        )
        info_panel.add_child(badge_txt)


class SettingsModalUI(UIScreen):
    """Modern dimmed modal overlay for game settings."""

    def __init__(self, on_back: Any) -> None:
        super().__init__(name="SettingsModalScreen")

        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(
            bg_color_top=(0.0, 0.0, 0.0, 0.70),
            bg_color_bottom=(0.0, 0.0, 0.0, 0.85),
            corner_radius=0.0,
        )
        self.add_child(backdrop)

        modal = _create_glass_card(x=340, y=100, w=600, h=520, border_accent=(0.18, 0.85, 1.0, 0.7))
        backdrop.add_child(modal)

        m_title = UILabel("ENGINE CONFIGURATION", x=0, y=25, w=600, h=36, font_size=26, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        modal.add_child(m_title)

        m_sub = UILabel("TUNABLE RUNTIME PARAMETERS", x=0, y=65, w=600, h=20, font_size=12, align="center", color=(0.95, 0.75, 0.20, 0.9))
        modal.add_child(m_sub)

        # Settings Sliders
        sl_sfx = UISlider("Sound Effects Volume", min_val=0, max_val=100, val=90, x=50, y=110, w=500, h=44, format_str="{label}: {val:.0f}%")
        modal.add_child(sl_sfx)

        sl_music = UISlider("Music Stream Volume", min_val=0, max_val=100, val=75, x=50, y=180, w=500, h=44, format_str="{label}: {val:.0f}%")
        modal.add_child(sl_music)

        lbl_aa = UILabel("ANTI-ALIASING TECHNIQUE", x=50, y=255, w=500, h=20, font_size=12, bold=True, color=(0.7, 0.8, 0.9, 0.85))
        modal.add_child(lbl_aa)

        seg_aa = UISegmentGroup(
            options=[
                ("OFF", 0),
                ("FXAA", 1),
                ("SMAA 1X", 2),
                ("TAA 8X", 3),
            ],
            selected_index=3,
            x=50,
            y=280,
            w=500,
            h=38,
        )
        modal.add_child(seg_aa)

        btn_back = UIButton(
            text="APPLY & RETURN",
            x=50,
            y=430,
            w=500,
            h=50,
            variant="primary",
            font_size=16,
            on_click=on_back,
        )
        modal.add_child(btn_back)


class PauseMenuUI(UIScreen):
    """Interactive Pause Menu modal overlay."""

    def __init__(self, on_resume: Any, on_settings: Any, on_main_menu: Any) -> None:
        super().__init__(name="PauseMenuScreen")

        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(
            bg_color_top=(0.0, 0.0, 0.0, 0.65),
            bg_color_bottom=(0.0, 0.0, 0.0, 0.80),
            corner_radius=0.0,
        )
        self.add_child(backdrop)

        modal = _create_glass_card(x=430, y=140, w=420, h=440, border_accent=(0.18, 0.75, 1.0, 0.7))
        backdrop.add_child(modal)

        lbl = UILabel("GAME PAUSED", x=0, y=25, w=420, h=34, font_size=24, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        modal.add_child(lbl)

        sub = UILabel("Simulation and Physics Frozen", x=0, y=62, w=420, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.7))
        modal.add_child(sub)

        btn_resume = UIButton(text="RESUME", x=40, y=110, w=340, h=48, variant="primary", font_size=16, on_click=on_resume)
        modal.add_child(btn_resume)

        btn_settings = UIButton(text="SETTINGS", x=40, y=175, w=340, h=48, variant="neutral", font_size=16, on_click=on_settings)
        modal.add_child(btn_settings)

        btn_menu = UIButton(text="QUIT TO MAIN MENU", x=40, y=240, w=340, h=48, variant="danger", font_size=16, on_click=on_main_menu)
        modal.add_child(btn_menu)

        hint = UILabel("Press ESC to Resume Instantly", x=0, y=380, w=420, h=20, font_size=12, align="center", color=(0.55, 0.65, 0.75, 0.6))
        modal.add_child(hint)


class PlayingHUD(UIScreen):
    """In-game HUD shown during the 3D exhibition simulation."""

    def __init__(self) -> None:
        super().__init__(name="PlayingHUDScreen", is_modal=False)

        card = _create_glass_card(x=24, y=20, w=380, h=110, border_accent=(0.15, 0.85, 1.0, 0.45))
        self.add_child(card)

        t1 = UILabel("PYMORDIAL 3D ENVIRONMENT", x=16, y=12, w=350, h=22, font_size=14, bold=True, color=(0.18, 0.88, 1.0, 1.0))
        card.add_child(t1)

        t2 = UILabel("STATUS: Simulation Active • 6-DOF Free-Fly Camera", x=16, y=36, w=350, h=18, font_size=11, color=(0.20, 0.95, 0.60, 0.95))
        card.add_child(t2)

        t3 = UILabel("• WASD / Left Stick: Fly  •  E/Q: Up/Down  •  Shift: Boost\n• Mouse / Right Stick: 3D Look  •  [ESC] / Start: Pause", x=16, y=58, w=350, h=38, font_size=11, color=(0.75, 0.82, 0.92, 0.85))
        card.add_child(t3)


# ---------------- 3D Exhibition World Module ----------------

class ExhibitionWorldModule(ProjectModule):
    """Creates a visual 3D scene with ground, lighting, and animated metallic cubes."""

    name = "ExhibitionWorldModule"

    def __init__(self) -> None:
        self.cube_ids: list[int] = []
        self._rot_time = 0.0

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

        # 1. Ground Platform (60 x 1 x 60 meters)
        app.ecs.create_entity(
            position=(0.0, -0.5, 0.0),
            scale=(60.0, 1.0, 60.0),
            color=(0.10, 0.12, 0.16, 1.0),
            roughness=0.6,
            metallic=0.3,
            is_static=True,
        )

        # 2. Central Pedestal
        app.ecs.create_entity(
            position=(0.0, 0.25, 0.0),
            scale=(4.0, 0.5, 4.0),
            color=(0.95, 0.75, 0.20, 1.0),
            roughness=0.25,
            metallic=0.8,
            is_static=True,
        )

        # 3. Rotating Exhibition Metallic Cubes
        cube_configs = [
            ((0.0, 2.0, 0.0), (1.5, 1.5, 1.5), (0.15, 0.85, 1.0, 1.0), 0.15, 0.10),
            ((-3.0, 1.5, 0.0), (1.0, 1.0, 1.0), (0.95, 0.25, 0.35, 1.0), 0.3, 0.6),
            ((3.0, 1.5, 0.0), (1.0, 1.0, 1.0), (0.25, 0.95, 0.45, 1.0), 0.3, 0.6),
            ((0.0, 1.5, -3.0), (1.0, 1.0, 1.0), (0.85, 0.35, 0.95, 1.0), 0.3, 0.6),
        ]
        self.cube_ids.clear()
        for pos, scale, color, rough, metal in cube_configs:
            eid = app.ecs.create_entity(
                position=pos,
                scale=scale,
                color=color,
                roughness=rough,
                metallic=metal,
                is_static=False,
            )
            self.cube_ids.append(eid)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        self._rot_time += dt
        # Smoothly rotate the central floating cube
        if self.cube_ids:
            center_eid = self.cube_ids[0]
            d_idx = app.ecs.pool.get_dense_index(center_eid)
            if d_idx >= 0:
                angle = self._rot_time * 0.8
                qy = math.sin(angle * 0.5)
                qw = math.cos(angle * 0.5)
                app.ecs.rigid_body_state[1, d_idx, 3:7] = (0.0, qy, 0.0, qw)
                app.ecs.recompute_matrix(d_idx)


# ---------------- Game States ----------------

class MainMenuState(GameState):
    """Initial game state rendering the fullscreen AAA main menu."""

    name = "MainMenu"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        def start_game():
            app.state_manager.set_state("Playing")

        def open_settings():
            app.state_manager.push_state("Settings")

        def quit_app():
            app.stop()

        app.ui.set_screen(MainMenuUI(on_start=start_game, on_settings=open_settings, on_quit=quit_app))

    def on_resume(self, app: ProjectApp) -> None:
        """Restores main menu screen when returning from settings modal."""
        def start_game():
            app.state_manager.set_state("Playing")

        def open_settings():
            app.state_manager.push_state("Settings")

        def quit_app():
            app.stop()

        app.ui.set_screen(MainMenuUI(on_start=start_game, on_settings=open_settings, on_quit=quit_app))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class SettingsState(GameState):
    """Settings state: modal overlay pushed onto stack."""

    name = "Settings"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        def on_back():
            app.state_manager.pop_state()

        app.ui.set_screen(SettingsModalUI(on_back=on_back))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.pop_state()
            return True
        return False


class PlayingState(GameState):
    """Active gameplay state: runs simulation, camera navigation, listens for ESC to pause."""

    name = "Playing"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def __init__(self) -> None:
        self.cam: FreeCamera | None = None
        self.hud: PlayingHUD | None = None

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.cam is None:
            self.cam = FreeCamera(position=(0.0, 3.5, 9.0), yaw_deg=-90.0, pitch_deg=-12.0)
            app.camera_manager.register_camera("main_cam", self.cam, make_active=True)

        if self.hud is None:
            self.hud = PlayingHUD()
        app.ui.set_screen(self.hud)

    def on_resume(self, app: ProjectApp) -> None:
        """Restores in-game HUD and asserts hardware mouse capture when unpausing."""
        app.input_manager.reset()
        if self.hud is not None:
            app.ui.set_screen(self.hud)
        app.input_manager.set_mouse_grab(True)
        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
        except Exception:
            pass

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.cam is None:
            return

        if app.input_manager.is_action_pressed("pause") or app.input_manager.is_gamepad_pressed(GamepadButton.START):
            app.state_manager.push_state("Paused")
            return

        # Continuous 6-DOF free camera fly movement (KBM + Gamepad)
        keys = pygame.key.get_pressed()
        fwd = (1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0) - (1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0)
        strafe = (1.0 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 0.0) - (1.0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 0.0)
        up = (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)
        boost = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        pad_move = app.input_manager.get_vector2("move")
        if abs(pad_move[0]) > 0.05 or abs(pad_move[1]) > 0.05:
            strafe = pad_move[0]
            fwd = pad_move[1]

        if app.input_manager.is_action_down("sprint") or app.input_manager.is_gamepad_down(GamepadButton.LEFT_STICK):
            boost = True

        self.cam.move(dt, forward_axis=fwd, right_axis=strafe, up_axis=up, boost=boost)

        # Gamepad Right Stick Look
        look_yaw = app.input_manager.get_axis("look_yaw")
        look_pitch = app.input_manager.get_axis("look_pitch")
        if abs(look_yaw) > 0.02 or abs(look_pitch) > 0.02:
            self.cam.yaw_deg += look_yaw * 120.0 * dt
            self.cam.pitch_deg = max(-89.0, min(89.0, self.cam.pitch_deg - look_pitch * 90.0 * dt))
            self.cam.mark_dirty()

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("Paused")
            return True
        elif event.type == pygame.MOUSEMOTION:
            if self.cam is not None:
                self.cam.handle_mouse_look(float(event.rel[0]), float(event.rel[1]))
                return True
        return False


class PausedState(GameState):
    """Modal pause state: freezes physics while displaying interactive pause screen."""

    name = "Paused"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        def resume():
            app.state_manager.pop_state()

        def open_settings():
            app.state_manager.push_state("Settings")

        def return_to_menu():
            app.state_manager.set_state("MainMenu")

        app.ui.set_screen(PauseMenuUI(on_resume=resume, on_settings=open_settings, on_main_menu=return_to_menu))

    def on_resume(self, app: ProjectApp) -> None:
        """Restores pause menu screen when returning from settings modal."""
        def resume():
            app.state_manager.pop_state()

        def open_settings():
            app.state_manager.push_state("Settings")

        def return_to_menu():
            app.state_manager.set_state("MainMenu")

        app.ui.set_screen(PauseMenuUI(on_resume=resume, on_settings=open_settings, on_main_menu=return_to_menu))

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


# ---------------- Application Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Modern UI & State Demo")
    parser.add_argument("--headless", action="store_true", help="Run in headless offscreen mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Modern AAA UI & Game States",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,  # Central sacred debug subsystem available via F1/F2/F3
    )

    app = ProjectApp(config=config)

    # Attach 3D visual world module
    app.add_module(ExhibitionWorldModule())

    # Register states
    app.state_manager.register(MainMenuState())
    app.state_manager.register(SettingsState())
    app.state_manager.register(PlayingState())
    app.state_manager.register(PausedState())

    # Boot into MainMenu
    app.state_manager.set_state("MainMenu")

    try:
        if args.headless and args.frames:
            step_count = max(1, args.frames // 4)
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.push_state("Settings")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.pop_state()
            app.state_manager.set_state("Playing")
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
