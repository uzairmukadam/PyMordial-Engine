"""PyMordial Engine Example 09: Photorealistic IBL & Radiance Cascades GI Showcase.

Demonstrates:
1. GPU-driven procedural IBL environment radiance and Split-Sum BRDF specular reflections.
2. Hi-Z Radiance Cascades Global Illumination (SSRC + FFPC) with multi-bounce color bleeding.
3. Advanced PBR material matrix (rough metals, mirror chrome, polished dielectrics, matte Cornell walls).
4. Volumetric Froxel Fog, Bloom, Screen-Space Reflections (SSR), and Temporal Anti-Aliasing (TAA).
5. Real-time sun azimuth/elevation manipulation and live HUD diagnostics.

Usage:
    python examples/09_photoreal_ibl_gi_showcase.py
    python examples/09_photoreal_ibl_gi_showcase.py --headless --frames 60
"""

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
    UIStyle,
)


# ---------------- 1. Photoreal Telemetry & Options HUD ----------------

class PhotorealHUD(UIScreen):
    """High-tech glassmorphic telemetry HUD displaying live pipeline states."""

    def __init__(self) -> None:
        super().__init__(name="PhotorealHUD", is_modal=False)

        card = UIPanel(x=24, y=24, w=420, h=160)
        card.style = UIStyle(
            bg_color_top=(0.04, 0.06, 0.10, 0.94),
            bg_color_bottom=(0.02, 0.03, 0.06, 0.98),
            border_color=(0.20, 0.75, 1.0, 0.6),
            border_width=1.5,
            corner_radius=10.0,
        )
        self.add_child(card)

        title = UILabel(
            "PYMORDIAL PHOTOREALISTIC IBL & GI",
            x=16,
            y=12,
            w=388,
            h=22,
            font_size=13,
            bold=True,
            color=(0.20, 0.88, 1.0, 1.0),
        )
        card.add_child(title)

        self.lbl_gi = UILabel(
            "GI Pipeline: RADIANCE_CASCADES (Hi-Z SSRC)",
            x=16,
            y=38,
            w=388,
            h=22,
            font_size=12,
            bold=True,
            color=(0.25, 0.95, 0.55, 1.0),
        )
        card.add_child(self.lbl_gi)

        self.lbl_optics = UILabel(
            "IBL Sky: GPU-Driven • SSR: ON • TAA: Active • AO: GTAO",
            x=16,
            y=64,
            w=388,
            h=20,
            font_size=11,
            color=(0.85, 0.90, 0.95, 0.9),
        )
        card.add_child(self.lbl_optics)

        controls = UILabel(
            "[G]: Cycle GI Mode • [1-4]: RC Intensity • [Arrows]: Sun Angle • ESC: Exit",
            x=16,
            y=96,
            w=388,
            h=36,
            font_size=10,
            color=(0.60, 0.70, 0.80, 0.85),
        )
        card.add_child(controls)

    def update_status(self, gi_mode: str, intensity: float) -> None:
        self.lbl_gi.text = f"GI Pipeline: {gi_mode} (Intensity: {intensity:.2f}x)"
        if gi_mode == "RADIANCE_CASCADES":
            sub = "IBL Sky: GPU-Driven • SSR: ON • TAA: Active • AO: GTAO"
        elif gi_mode == "HYBRID":
            sub = "IBL Sky: GPU-Driven • SSGI + 3D LPV Cascade"
        elif gi_mode == "SSGI":
            sub = "IBL Sky: GPU-Driven • Screen-Space Raymarching Only"
        else:
            sub = "IBL Sky: GPU-Driven • Direct Lighting + IBL Ambient"
        self.lbl_optics.text = sub


# ---------------- 2. Photoreal Architectural Scene Module ----------------

class PhotorealWorldModule(ProjectModule):
    """Constructs an architectural gallery showcase designed to emphasize

    specular IBL reflections, shadow softness, and global illumination color bleeding.
    """

    name = "PhotorealWorldModule"

    def __init__(self) -> None:
        super().__init__()
        self.gi_modes = ["RADIANCE_CASCADES", "HYBRID", "SSGI", "OFF"]
        self.gi_index = 0
        self.rc_intensity = 1.25

    def on_attach(self, app: ProjectApp) -> None:
        # Configure High-End PBR, IBL, and Radiance Cascades Pipeline
        cfg = app.pipeline.config
        cfg.gi_mode = "RADIANCE_CASCADES"
        cfg.rc_intensity = self.rc_intensity
        cfg.rc_interval_c0 = 0.40
        cfg.rc_interval_c1 = 16.00
        cfg.rc_thickness = 0.25
        cfg.rc_temporal_blend = 0.90
        cfg.ao_mode = "GTAO"
        cfg.aa_mode = "TAA"
        cfg.bloom_enabled = True
        cfg.ssr_enabled = True
        cfg.volumetric_fog_enabled = True
        cfg.lens_flare_enabled = True
        cfg.sun_intensity = 3.2
        cfg.sun_direction = (0.35, -0.75, -0.55)

        if app.engine_tweaks is not None:
            app.engine_tweaks.gi_mode = "RADIANCE_CASCADES"
            app.engine_tweaks.rc_intensity = self.rc_intensity

        # --- Material Swatches ---
        white_marble = (0.92, 0.93, 0.95, 1.0)
        vibrant_red = (0.90, 0.12, 0.12, 1.0)      # Left wall color bleeding test
        vibrant_green = (0.12, 0.85, 0.18, 1.0)    # Right wall color bleeding test
        polished_chrome = (0.95, 0.96, 0.98, 1.0)  # Extreme mirror IBL reflection
        brushed_gold = (0.95, 0.78, 0.22, 1.0)     # Warm metallic specular
        matte_obsidian = (0.08, 0.09, 0.12, 1.0)

        # 1. Floor (Reflective Polished Marble)
        app.ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(24.0, 2.0, 22.0),
            color=white_marble,
            roughness=0.18,
            metallic=0.05,
            is_static=True,
        )

        # 2. Ceiling
        app.ecs.create_entity(
            position=(0.0, 11.0, 0.0),
            scale=(24.0, 2.0, 22.0),
            color=white_marble,
            roughness=0.80,
            metallic=0.02,
            is_static=True,
        )

        # 3. Back Wall
        app.ecs.create_entity(
            position=(0.0, 5.0, -10.0),
            scale=(24.0, 12.0, 2.0),
            color=white_marble,
            roughness=0.75,
            metallic=0.02,
            is_static=True,
        )

        # 4. Left Wall (Vibrant Red for diffuse color bleed)
        app.ecs.create_entity(
            position=(-11.0, 5.0, 0.0),
            scale=(2.0, 12.0, 22.0),
            color=vibrant_red,
            roughness=0.70,
            metallic=0.02,
            is_static=True,
        )

        # 5. Right Wall (Vibrant Green for diffuse color bleed)
        app.ecs.create_entity(
            position=(11.0, 5.0, 0.0),
            scale=(2.0, 12.0, 22.0),
            color=vibrant_green,
            roughness=0.70,
            metallic=0.02,
            is_static=True,
        )

        # 6. Central Polished Chrome Sculpture (Reflects IBL Sky & Environment)
        app.ecs.create_entity(
            position=(0.0, 1.6, -1.5),
            scale=(2.4, 3.2, 2.4),
            color=polished_chrome,
            roughness=0.03,
            metallic=1.00,
            is_static=True,
        )

        # 7. Left Brushed Gold Pillar
        app.ecs.create_entity(
            position=(-4.5, 2.0, -3.0),
            scale=(1.8, 5.0, 1.8),
            color=brushed_gold,
            roughness=0.28,
            metallic=0.92,
            is_static=True,
        )

        # 8. Right Matte Obsidian Pedestal Block
        app.ecs.create_entity(
            position=(4.5, 1.2, -1.0),
            scale=(2.2, 2.4, 2.2),
            color=matte_obsidian,
            roughness=0.45,
            metallic=0.10,
            is_static=True,
        )

        # 9. Overhead Soft Fill Point Lights for Clustered Shading
        pipe = app.pipeline
        if hasattr(pipe, "lights_pass"):
            pipe.lights_pass.clear()
            pipe.add_point_light(
                position=(0.0, 9.5, -2.0),
                radius=20.0,
                color=(1.0, 0.96, 0.90),
                intensity=7.5,
            )


# ---------------- 3. Game State & Controls ----------------

class PhotorealState(GameState):
    """Manages free-fly camera, sun direction tuning, and live GI mode switching."""

    name = "PhotorealState"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def __init__(self) -> None:
        self.cam: FreeCamera | None = None
        self.hud: PhotorealHUD | None = None
        self._sun_azimuth = 0.55
        self._sun_elevation = -0.72

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.cam is None:
            self.cam = FreeCamera(position=(0.0, 3.5, 10.5), yaw_deg=-90.0, pitch_deg=-12.0)
            app.camera_manager.register_camera("main_cam", self.cam, make_active=True)

        if self.hud is None:
            self.hud = PhotorealHUD()
        app.ui.set_screen(self.hud)
        self._sync_hud(app)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.cam is None:
            return

        # Smooth 6-DOF camera movement
        keys = pygame.key.get_pressed()
        fwd = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        strafe = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        up = (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)
        boost = bool(keys[pygame.K_LSHIFT])

        self.cam.move(dt, forward_axis=fwd, right_axis=strafe, up_axis=up, boost=boost)

        # Interactive Sun Angle Adjustment (Arrow Keys)
        sun_changed = False
        speed = 1.0 * dt
        if keys[pygame.K_LEFT]:
            self._sun_azimuth -= speed
            sun_changed = True
        if keys[pygame.K_RIGHT]:
            self._sun_azimuth += speed
            sun_changed = True
        if keys[pygame.K_UP]:
            self._sun_elevation = min(-0.15, self._sun_elevation + speed * 0.8)
            sun_changed = True
        if keys[pygame.K_DOWN]:
            self._sun_elevation = max(-0.95, self._sun_elevation - speed * 0.8)
            sun_changed = True

        if sun_changed:
            cos_el = math.sqrt(max(0.001, 1.0 - self._sun_elevation * self._sun_elevation))
            sx = math.sin(self._sun_azimuth) * cos_el
            sy = self._sun_elevation
            sz = math.cos(self._sun_azimuth) * cos_el
            app.pipeline.config.sun_direction = (sx, sy, sz)

    def _sync_hud(self, app: ProjectApp) -> None:
        if self.hud is not None:
            mode = getattr(app.pipeline.config, "gi_mode", "RADIANCE_CASCADES")
            intensity = getattr(app.pipeline.config, "rc_intensity", 1.25)
            self.hud.update_status(mode, intensity)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.stop()
                return True

            # [G]: Cycle Global Illumination Modes
            elif event.key == pygame.K_g:
                mod_world = app.get_module(PhotorealWorldModule)
                if mod_world is not None:
                    mod_world.gi_index = (mod_world.gi_index + 1) % len(mod_world.gi_modes)
                    new_mode = mod_world.gi_modes[mod_world.gi_index]
                    app.pipeline.config.gi_mode = new_mode
                    if app.engine_tweaks is not None:
                        app.engine_tweaks.gi_mode = new_mode
                    self._sync_hud(app)
                return True

            # [1-4]: Quick-adjust Radiance Cascades intensity
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                vals = {pygame.K_1: 0.5, pygame.K_2: 1.0, pygame.K_3: 1.5, pygame.K_4: 2.25}
                val = vals[event.key]
                app.pipeline.config.rc_intensity = val
                if app.engine_tweaks is not None:
                    app.engine_tweaks.rc_intensity = val
                self._sync_hud(app)
                return True

        elif event.type == pygame.MOUSEMOTION and self.cam is not None:
            self.cam.handle_mouse_look(float(event.rel[0]), float(event.rel[1]))
            return True

        return False


# ---------------- 4. Application Main Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Photorealistic IBL & GI Showcase")
    parser.add_argument("--headless", action="store_true", help="Run offscreen in headless mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate before exit")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial Engine - Photorealistic IBL & GI Showcase",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(PhotorealWorldModule())
    app.state_manager.register(PhotorealState())
    app.state_manager.set_state("PhotorealState")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
