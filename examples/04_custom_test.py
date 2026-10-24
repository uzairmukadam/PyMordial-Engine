from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Any
import pygame

# Ensure project root is in path
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
    UIStyle,
)


# ---------------- 1. UI Screens ----------------

class SimpleMainMenuUI(UIScreen):
    """Clean, high-tech menu screen for Global Illumination testbench."""

    def __init__(self, on_start: Any, on_quit: Any) -> None:
        super().__init__(name="SimpleMainMenuScreen")

        # Fullscreen dark backdrop
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.03, 0.04, 0.07, 0.96),
            bg_color_bottom=(0.01, 0.02, 0.03, 0.99),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(bg)

        # Center Menu Card
        card = UIPanel(x=420, y=190, w=440, h=340)
        card.style = UIStyle(
            bg_color_top=(0.07, 0.09, 0.15, 0.95),
            bg_color_bottom=(0.03, 0.04, 0.08, 0.98),
            border_color=(0.2, 0.7, 1.0, 0.6),
            border_width=1.5,
            corner_radius=12.0,
        )
        bg.add_child(card)

        title = UILabel(
            "PYMORDIAL GI BENCH",
            x=0,
            y=30,
            w=440,
            h=36,
            font_size=26,
            bold=True,
            align="center",
            color=(0.2, 0.85, 1.0, 1.0),
        )
        card.add_child(title)

        subtitle = UILabel(
            "Hi-Z Radiance Cascades (SSRC + FFPC)",
            x=0,
            y=70,
            w=440,
            h=24,
            font_size=13,
            align="center",
            color=(0.7, 0.8, 0.95, 0.85),
        )
        card.add_child(subtitle)

        btn_start = UIButton(
            text="START GI SIMULATION",
            x=50,
            y=125,
            w=340,
            h=50,
            variant="primary",
            font_size=16,
            on_click=on_start,
        )
        card.add_child(btn_start)

        btn_quit = UIButton(
            text="QUIT",
            x=50,
            y=200,
            w=340,
            h=50,
            variant="danger",
            font_size=16,
            on_click=on_quit,
        )
        card.add_child(btn_quit)

        hint = UILabel(
            "Press G in simulation to cycle GI pipelines",
            x=0,
            y=280,
            w=440,
            h=20,
            font_size=11,
            align="center",
            color=(0.55, 0.65, 0.75, 0.7),
        )
        card.add_child(hint)


class GIHUD(UIScreen):
    """In-game HUD displaying active GI pipeline metrics and controls."""

    def __init__(self) -> None:
        super().__init__(name="GIHUDScreen", is_modal=False)

        panel = UIPanel(x=24, y=24, w=380, h=130)
        panel.style = UIStyle(
            bg_color_top=(0.04, 0.06, 0.10, 0.92),
            bg_color_bottom=(0.02, 0.03, 0.06, 0.96),
            border_color=(0.2, 0.7, 1.0, 0.5),
            border_width=1.5,
            corner_radius=8.0,
        )
        self.add_child(panel)

        header = UILabel(
            "GLOBAL ILLUMINATION TEST BENCH",
            x=16,
            y=12,
            w=348,
            h=22,
            font_size=13,
            bold=True,
            color=(0.2, 0.85, 1.0, 1.0),
        )
        panel.add_child(header)

        self.lbl_mode = UILabel(
            "GI Mode: RADIANCE CASCADES",
            x=16,
            y=38,
            w=348,
            h=22,
            font_size=12,
            bold=True,
            color=(0.4, 1.0, 0.5, 1.0),
        )
        panel.add_child(self.lbl_mode)

        self.lbl_details = UILabel(
            "Pipeline: Hi-Z SSRC + FFPC Probes • Intensity: 1.35x",
            x=16,
            y=62,
            w=348,
            h=20,
            font_size=11,
            color=(0.85, 0.9, 0.95, 0.85),
        )
        panel.add_child(self.lbl_details)

        controls = UILabel(
            "[G]: Cycle GI • [1-4]: Intensity • [L]: Light • Arrows: Move Sun • ESC: Menu",
            x=16,
            y=90,
            w=348,
            h=24,
            font_size=10,
            color=(0.6, 0.7, 0.8, 0.8),
        )
        panel.add_child(controls)

    def update_gi_status(self, mode: str, intensity: float) -> None:
        color_map = {
            "RADIANCE_CASCADES": (0.2, 0.95, 0.45, 1.0),
            "HYBRID": (0.3, 0.8, 1.0, 1.0),
            "SSGI": (1.0, 0.85, 0.25, 1.0),
            "LPV": (1.0, 0.55, 0.2, 1.0),
            "OFF": (0.75, 0.75, 0.75, 0.8),
        }
        col = color_map.get(mode, (1.0, 1.0, 1.0, 1.0))
        self.lbl_mode.text = f"GI Mode: {mode}"
        self.lbl_mode.color = col

        if mode == "RADIANCE_CASCADES":
            sub = f"Pipeline: Hi-Z SSRC + FFPC Probes • Intensity: {intensity:.2f}x"
        elif mode == "HYBRID":
            sub = "Pipeline: Dual SSGI + 3D LPV Cascade • Intensity: 1.00x"
        elif mode == "SSGI":
            sub = "Pipeline: Screen-Space Raymarching • Intensity: 1.00x"
        elif mode == "LPV":
            sub = "Pipeline: Volumetric Light Propagation • Intensity: 1.00x"
        else:
            sub = "Direct Lighting Only (Global Illumination Disabled)"
        self.lbl_details.text = sub


# ---------------- 2. World Module (Cornell Box Scene) ----------------

class GIBoxWorldModule(ProjectModule):
    """Spawns an architectural Cornell Box test room optimized for GI bounce and color bleeding."""

    name = "GIBoxWorldModule"

    def __init__(self) -> None:
        super().__init__()
        self.spot_light_enabled = True

    def on_attach(self, app: ProjectApp) -> None:
        # Disable water rendering in indoor box
        app.pipeline.config.water_enabled = False
        if app.engine_tweaks is not None:
            app.engine_tweaks.water_enabled = False

        # Configure High-Performance Radiance Cascades as default GI pipeline
        app.pipeline.config.gi_mode = "RADIANCE_CASCADES"
        app.pipeline.config.rc_intensity = 1.00
        app.pipeline.config.rc_interval_c0 = 0.50
        app.pipeline.config.rc_interval_c1 = 14.00
        app.pipeline.config.rc_thickness = 0.20
        app.pipeline.config.rc_temporal_blend = 0.88
        app.pipeline.config.sun_intensity = 2.5
        app.pipeline.config.sun_direction = (0.20, -0.85, -0.45)

        if app.engine_tweaks is not None:
            app.engine_tweaks.gi_mode = "RADIANCE_CASCADES"
            app.engine_tweaks.rc_intensity = 1.00
            app.engine_tweaks.rc_interval_c0 = 0.50
            app.engine_tweaks.rc_interval_c1 = 14.00
            app.engine_tweaks.rc_thickness = 0.20
            app.engine_tweaks.rc_temporal_blend = 0.88

        # Color definitions for GI bleeding tests
        white_matte = (0.95, 0.95, 0.95, 1.0)
        vibrant_red = (0.92, 0.10, 0.10, 1.0)      # Left wall: red bleed
        vibrant_green = (0.10, 0.88, 0.15, 1.0)    # Right wall: green bleed
        cobalt_blue = (0.12, 0.45, 0.95, 1.0)      # Accent block
        gold_metal = (0.95, 0.78, 0.20, 1.0)       # Accent cylinder / block

        roughness_diffuse = 0.85
        metallic_dielectric = 0.05

        # 1. Floor (Bottom) - White Matte
        app.ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(20.0, 2.0, 18.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 2. Complete Solid Ceiling (Enclosed Room)
        app.ecs.create_entity(
            position=(0.0, 10.0, 0.0),
            scale=(20.0, 2.0, 18.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 3. Back Wall - White Matte
        app.ecs.create_entity(
            position=(0.0, 5.0, -8.0),
            scale=(20.0, 10.0, 2.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 3b. Front Wall Left & Right (Observation doorway in center for camera entry)
        app.ecs.create_entity(
            position=(-6.0, 5.0, 8.5),
            scale=(8.0, 10.0, 2.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )
        app.ecs.create_entity(
            position=(6.0, 5.0, 8.5),
            scale=(8.0, 10.0, 2.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 4. Left Wall - Vibrant Red (Causes intense diffuse color bleeding)
        app.ecs.create_entity(
            position=(-6.0, 5.0, 0.0),
            scale=(2.0, 10.0, 18.0),
            color=vibrant_red,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 5. Right Wall - Vibrant Green (Causes contrasting diffuse color bleeding)
        app.ecs.create_entity(
            position=(6.0, 5.0, 0.0),
            scale=(2.0, 10.0, 18.0),
            color=vibrant_green,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 6. Tall Pillar (Left side - close to red wall for intense red diffuse bleed)
        app.ecs.create_entity(
            position=(-3.4, 3.0, -2.5),
            scale=(2.0, 6.0, 2.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 7. Shorter Pillar (Right side - close to green wall for intense green diffuse bleed)
        app.ecs.create_entity(
            position=(3.4, 1.8, 0.5),
            scale=(2.0, 3.6, 2.0),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 8. Center Pedestal Block (Catches red and green bounces simultaneously)
        app.ecs.create_entity(
            position=(0.0, 0.6, -1.0),
            scale=(1.8, 1.2, 1.8),
            color=white_matte,
            roughness=roughness_diffuse,
            metallic=metallic_dielectric,
            is_static=True,
        )

        # 9. Cobalt Blue Accent Block
        app.ecs.create_entity(
            position=(-1.2, 0.75, 2.2),
            scale=(1.5, 1.5, 1.5),
            color=cobalt_blue,
            roughness=0.35,
            metallic=0.10,
            is_static=True,
        )

        # 10. Golden Metallic Cube
        app.ecs.create_entity(
            position=(1.5, 0.6, 2.5),
            scale=(1.2, 1.2, 1.2),
            color=gold_metal,
            roughness=0.20,
            metallic=0.90,
            is_static=True,
        )

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.spot_light_enabled:
            # Single overhead Cornell box ceiling light illuminating floor and walls
            app.pipeline.add_spot_light(
                position=(0.0, 8.8, -0.5),
                direction=(0.0, -1.0, 0.0),
                radius=25.0,
                inner_cone_angle=math.radians(35.0),
                outer_cone_angle=math.radians(72.0),
                color=(1.0, 0.98, 0.94),
                intensity=9.0,
                cast_shadow=True,
                shadow_bias=0.0015,
            )


# ---------------- 3. Game States ----------------

class MainMenuState(GameState):
    """State managing the start menu."""

    name = "MainMenu"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = False

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        def start_game() -> None:
            app.state_manager.set_state("Simulation")

        def quit_app() -> None:
            app.stop()

        app.ui.set_screen(SimpleMainMenuUI(on_start=start_game, on_quit=quit_app))

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class SimulationState(GameState):
    """Active simulation state with free camera flight and interactive GI toggling."""

    name = "Simulation"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    GI_MODES = ["RADIANCE_CASCADES", "HYBRID", "SSGI", "LPV", "OFF"]

    def __init__(self) -> None:
        self.cam: FreeCamera | None = None
        self.hud: GIHUD | None = None
        self._gi_idx = 0
        self._intensity = 1.75
        self._sun_azimuth = 0.85
        self._sun_elevation = -0.75

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.cam is None:
            self.cam = FreeCamera(position=(0.0, 3.5, 9.5), yaw_deg=-90.0, pitch_deg=-14.0)
            app.camera_manager.register_camera("main_cam", self.cam, make_active=True)

        if self.hud is None:
            self.hud = GIHUD()
        app.ui.set_screen(self.hud)
        self._update_hud_display(app)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.cam is None:
            return

        # Smooth camera flight controls
        keys = pygame.key.get_pressed()
        fwd = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        strafe = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        up = (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)
        boost = bool(keys[pygame.K_LSHIFT])

        self.cam.move(dt, forward_axis=fwd, right_axis=strafe, up_axis=up, boost=boost)

        # Interactive Sun Direction Controls (Arrow keys: Left/Right = Azimuth, Up/Down = Elevation)
        sun_changed = False
        sun_speed = 1.2 * dt
        if keys[pygame.K_LEFT]:
            self._sun_azimuth -= sun_speed
            sun_changed = True
        if keys[pygame.K_RIGHT]:
            self._sun_azimuth += sun_speed
            sun_changed = True
        if keys[pygame.K_UP]:
            self._sun_elevation = min(-0.15, self._sun_elevation + sun_speed * 0.75)
            sun_changed = True
        if keys[pygame.K_DOWN]:
            self._sun_elevation = max(-0.98, self._sun_elevation - sun_speed * 0.75)
            sun_changed = True

        if sun_changed:
            # Recompute directional sun vector from spherical coordinates
            cos_el = math.sqrt(max(0.001, 1.0 - self._sun_elevation * self._sun_elevation))
            sun_x = math.sin(self._sun_azimuth) * cos_el
            sun_y = self._sun_elevation
            sun_z = math.cos(self._sun_azimuth) * cos_el
            app.pipeline.config.sun_direction = (sun_x, sun_y, sun_z)
            if hasattr(app.pipeline, "atmosphere"):
                app.pipeline.atmosphere.config.day_speed = 0.0

    def _update_hud_display(self, app: ProjectApp) -> None:
        if self.hud is not None:
            curr_mode = getattr(app.pipeline.config, "gi_mode", "RADIANCE_CASCADES")
            self.hud.update_gi_status(curr_mode, self._intensity)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.state_manager.set_state("MainMenu")
                return True

            # G: Cycle Global Illumination pipelines
            elif event.key == pygame.K_g:
                self._gi_idx = (self._gi_idx + 1) % len(self.GI_MODES)
                new_mode = self.GI_MODES[self._gi_idx]
                app.pipeline.config.gi_mode = new_mode
                if app.engine_tweaks is not None:
                    app.engine_tweaks.gi_mode = new_mode
                self._update_hud_display(app)
                return True

            # L: Toggle overhead spotlight
            elif event.key == pygame.K_l:
                box_module = app.get_module(GIBoxWorldModule)
                if box_module is not None:
                    box_module.spot_light_enabled = not box_module.spot_light_enabled
                return True

            # 1-4: Quick-adjust Radiance Cascades GI Intensity
            elif event.key == pygame.K_1:
                self._intensity = 0.60
                app.pipeline.config.rc_intensity = self._intensity
                self._update_hud_display(app)
                return True
            elif event.key == pygame.K_2:
                self._intensity = 1.00
                app.pipeline.config.rc_intensity = self._intensity
                self._update_hud_display(app)
                return True
            elif event.key == pygame.K_3:
                self._intensity = 1.35
                app.pipeline.config.rc_intensity = self._intensity
                self._update_hud_display(app)
                return True
            elif event.key == pygame.K_4:
                self._intensity = 2.00
                app.pipeline.config.rc_intensity = self._intensity
                self._update_hud_display(app)
                return True

        elif event.type == pygame.MOUSEMOTION and self.cam is not None:
            self.cam.handle_mouse_look(float(event.rel[0]), float(event.rel[1]))
            return True

        return False


# ---------------- 4. Application Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Next-Gen GI & Radiance Cascades Demo")
    parser.add_argument("--headless", action="store_true", help="Run in headless offscreen mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Next-Gen Radiance Cascades GI",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)

    # Attach world module and register states
    app.add_module(GIBoxWorldModule())
    app.state_manager.register(MainMenuState())
    app.state_manager.register(SimulationState())

    # Boot straight into Main Menu
    app.state_manager.set_state("MainMenu")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()