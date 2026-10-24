"""Beta Upgrade Testbench & Hardware-Scaling Showcase (Example 05).

Provides a comprehensive manual test scene for evaluating Decima-inspired engine upgrades:
- Phase 1: Per-Cascade CSM Shadow Culling verification (telemetry readout & cascade inspection)
- Phase 1: Hardware-Scaled Preset switching (Low -> Steam Deck/Medium -> High -> Ultra)
- Real-time Sun Orbit & Elevation manipulation (Arrow keys)
- FreeCamera flight with HUD telemetry (FPS, frame times, visible vs culled entities/commands)
- Instant G-Buffer debug visualizer toggle (Albedo, Normals, Depth, Shadow Atlas, Hi-Z Mips)
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
    UIButton,
    UIStyle,
)
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset


# ---------------- 1. HUD & Diagnostic UI ----------------

class BetaShowcaseHUD(UIScreen):
    """High-tech responsive HUD displaying live telemetry, culling stats, and active presets."""

    def __init__(self) -> None:
        super().__init__(name="BetaShowcaseHUD", is_modal=False)

        # Main Telemetry Panel (Top-Left)
        panel = UIPanel(x=24, y=24, w=460, h=220)
        panel.style = UIStyle(
            bg_color_top=(0.04, 0.06, 0.10, 0.94),
            bg_color_bottom=(0.02, 0.03, 0.06, 0.98),
            border_color=(0.2, 0.75, 1.0, 0.6),
            border_width=1.5,
            corner_radius=8.0,
        )
        self.add_child(panel)

        header = UILabel(
            "PYMORDIAL BETA UPGRADE TESTBENCH",
            x=16,
            y=12,
            w=428,
            h=20,
            font_size=13,
            bold=True,
            color=(0.25, 0.85, 1.0, 1.0),
        )
        panel.add_child(header)

        self.lbl_preset = UILabel(
            "Quality Preset: HIGH (Mid-Range Target)",
            x=16,
            y=36,
            w=428,
            h=18,
            font_size=11,
            bold=True,
            color=(0.4, 1.0, 0.5, 1.0),
        )
        panel.add_child(self.lbl_preset)

        self.lbl_pipeline = UILabel(
            "Shadows: PCSS 24-tap • AO: GTAO • AA: TAA • GI: SSRC",
            x=16,
            y=58,
            w=428,
            h=18,
            font_size=10,
            color=(0.85, 0.90, 0.95, 0.9),
        )
        panel.add_child(self.lbl_pipeline)

        self.lbl_shadow_culling = UILabel(
            "CSM Per-Cascade Culling: Active (4 Cascades)",
            x=16,
            y=80,
            w=428,
            h=18,
            font_size=10,
            color=(1.0, 0.85, 0.3, 0.95),
        )
        panel.add_child(self.lbl_shadow_culling)

        self.lbl_cull_stats = UILabel(
            "Camera Visible: 0/0 Ents • 0 Draw Cmds",
            x=16,
            y=102,
            w=428,
            h=18,
            font_size=10,
            color=(0.7, 0.8, 0.9, 0.85),
        )
        panel.add_child(self.lbl_cull_stats)

        self.lbl_sun_info = UILabel(
            "Sun: Azimuth 0.85 rad • Elevation -0.75 rad",
            x=16,
            y=124,
            w=428,
            h=18,
            font_size=10,
            color=(1.0, 0.65, 0.35, 0.9),
        )
        panel.add_child(self.lbl_sun_info)

        self.lbl_debug_view = UILabel(
            "Debug View: Normal Shading (Press [V] to cycle G-Buffer)",
            x=16,
            y=146,
            w=428,
            h=18,
            font_size=10,
            color=(0.6, 0.75, 0.9, 0.85),
        )
        panel.add_child(self.lbl_debug_view)

        controls = UILabel(
            "[1-4]: Presets (Low/Deck/High/Ultra) • [Arrows]: Sun • [V]: G-Buffer • ESC: Exit",
            x=16,
            y=176,
            w=428,
            h=28,
            font_size=9,
            color=(0.6, 0.7, 0.8, 0.8),
        )
        panel.add_child(controls)

    def update_telemetry(
        self,
        preset_name: str,
        pipeline_info: str,
        shadow_info: str,
        vis_ents: int,
        total_ents: int,
        vis_cmds: int,
        sun_azimuth: float,
        sun_elevation: float,
        debug_mode_name: str,
    ) -> None:
        self.lbl_preset.text = f"Quality Preset: {preset_name}"
        self.lbl_pipeline.text = pipeline_info
        self.lbl_shadow_culling.text = shadow_info
        self.lbl_cull_stats.text = f"Camera Visible: {vis_ents}/{total_ents} Ents • {vis_cmds} MDI Cmds"
        self.lbl_sun_info.text = f"Sun: Azimuth {math.degrees(sun_azimuth):.1f}° • Elevation {math.degrees(sun_elevation):.1f}°"
        self.lbl_debug_view.text = f"Debug View: {debug_mode_name} ([V] to cycle)"


# ---------------- 2. World Module (Multi-Tier Test Arena) ----------------

class BetaShowcaseWorldModule(ProjectModule):
    """Spawns an extensive test arena with outdoor terrain, large occlusion walls,

    and dense foreground/background prop clusters to visually stress per-cascade
    shadow culling, GTAO, TAA, and multi-tier quality presets.
    """

    name = "BetaShowcaseWorldModule"

    def on_attach(self, app: ProjectApp) -> None:
        ecs = app.ecs

        # High-contrast materials for testing PBR, specular stability, and ambient bounce
        mat_ground = (0.75, 0.75, 0.75, 1.0)
        mat_red = (0.90, 0.15, 0.15, 1.0)
        mat_blue = (0.15, 0.45, 0.95, 1.0)
        mat_gold = (0.95, 0.78, 0.20, 1.0)
        mat_chrome = (0.90, 0.92, 0.95, 1.0)
        mat_dark = (0.12, 0.14, 0.18, 1.0)

        # 1. Main Vast Ground Plane (Stretches across all shadow cascades)
        ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(300.0, 2.0, 300.0),
            color=mat_ground,
            roughness=0.85,
            metallic=0.05,
            is_static=True,
        )

        # 2. Large Occlusion Wall (Tests whether objects behind it are culled in shadow & camera passes)
        ecs.create_entity(
            position=(0.0, 8.0, -25.0),
            scale=(60.0, 16.0, 4.0),
            color=mat_dark,
            roughness=0.70,
            metallic=0.10,
            is_static=True,
        )

        # 3. Dense Grid of Shadow Casters behind the wall (Distant cascade stress test)
        for x in range(-4, 5):
            for z in range(3):
                px = float(x * 6.0)
                pz = -35.0 - float(z * 8.0)
                ecs.create_entity(
                    position=(px, 4.0, pz),
                    scale=(2.0, 8.0, 2.0),
                    color=mat_red if (x % 2 == 0) else mat_blue,
                    roughness=0.60,
                    metallic=0.10,
                    is_static=True,
                )

        # 4. Foreground Close-Up Geometry (Cascade 0 Near Split Contact Shadows & GTAO)
        # Pedestal cluster
        ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            scale=(4.0, 2.0, 4.0),
            color=(0.95, 0.95, 0.95, 1.0),
            roughness=0.90,
            metallic=0.02,
            is_static=True,
        )
        # Metallic gold sphere/cube on top
        ecs.create_entity(
            position=(0.0, 2.8, 0.0),
            scale=(1.6, 1.6, 1.6),
            color=mat_gold,
            roughness=0.15,
            metallic=0.95,
            is_static=True,
        )
        # Chrome mirror block (Tests SSR & TAA specular shimmer)
        ecs.create_entity(
            position=(-3.5, 1.5, 1.5),
            scale=(1.8, 3.0, 1.8),
            color=mat_chrome,
            roughness=0.05,
            metallic=0.98,
            is_static=True,
        )
        # Rough dark cylinder / pillar
        ecs.create_entity(
            position=(3.5, 2.0, 1.5),
            scale=(1.8, 4.0, 1.8),
            color=mat_dark,
            roughness=0.95,
            metallic=0.05,
            is_static=True,
        )

        # 5. Row of Slanted Monoliths (Tests Sloped Depth Bias & Shadow Shimmering)
        for i in range(5):
            offset_x = -15.0 + float(i * 7.5)
            ecs.create_entity(
                position=(offset_x, 3.5, 10.0),
                scale=(1.5, 7.0, 1.5),
                color=(0.85, 0.85, 0.90, 1.0),
                roughness=0.45,
                metallic=0.20,
                is_static=True,
            )

        # Default sun direction and atmosphere setup
        app.pipeline.config.sun_intensity = 3.5
        app.pipeline.config.sun_direction = (0.45, -0.75, -0.48)
        if hasattr(app.pipeline, "atmosphere"):
            app.pipeline.atmosphere.config.day_speed = 0.0


# ---------------- 3. Simulation Game State ----------------

class BetaShowcaseState(GameState):
    """Interactive state controlling camera, sun position, and live preset switching."""

    name = "BetaShowcase"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    PRESET_NAMES = {
        1: (GraphicsQuality.LOW, "LOW (Integrated GPU Target)"),
        2: (GraphicsQuality.MEDIUM, "MEDIUM (Steam Deck / 15W Target)"),
        3: (GraphicsQuality.HIGH, "HIGH (Mid-to-High PC Target)"),
        4: (GraphicsQuality.ULTRA, "ULTRA (Cinematic Benchmark)"),
    }

    DEBUG_MODES = [
        (0, "Normal Shading"),
        (1, "Albedo Roughness"),
        (2, "World Normals"),
        (4, "Reversed-Z Depth"),
        (5, "CSM Shadow Atlas"),
    ]

    def __init__(self) -> None:
        self.cam: FreeCamera | None = None
        self.hud: BetaShowcaseHUD | None = None
        self._preset_id = 3  # Default to HIGH
        self._debug_idx = 0
        self._sun_azimuth = 0.75
        self._sun_elevation = -0.65

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.cam is None:
            self.cam = FreeCamera(position=(0.0, 6.0, 18.0), yaw_deg=-90.0, pitch_deg=-14.0)
            app.camera_manager.register_camera("main_cam", self.cam, make_active=True)

        if self.hud is None:
            self.hud = BetaShowcaseHUD()
        app.ui.set_screen(self.hud)
        self._apply_preset(app, self._preset_id)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def _apply_preset(self, app: ProjectApp, preset_id: int) -> None:
        self._preset_id = preset_id
        q_enum, label = self.PRESET_NAMES[preset_id]
        preset_cfg = get_quality_preset(q_enum)
        app.pipeline.apply_config(preset_cfg)
        if app.engine_tweaks is not None:
            app.engine_tweaks.set_quality_preset(q_enum)

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.cam is None:
            return

        # 1. Camera flight controls
        keys = pygame.key.get_pressed()
        fwd = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        strafe = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        up = (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)
        boost = bool(keys[pygame.K_LSHIFT])
        self.cam.move(dt, forward_axis=fwd, right_axis=strafe, up_axis=up, boost=boost)

        # 2. Interactive Sun Controls (Arrow keys)
        sun_changed = False
        sun_speed = 1.0 * dt
        if keys[pygame.K_LEFT]:
            self._sun_azimuth -= sun_speed
            sun_changed = True
        if keys[pygame.K_RIGHT]:
            self._sun_azimuth += sun_speed
            sun_changed = True
        if keys[pygame.K_UP]:
            self._sun_elevation = min(-0.10, self._sun_elevation + sun_speed * 0.75)
            sun_changed = True
        if keys[pygame.K_DOWN]:
            self._sun_elevation = max(-0.98, self._sun_elevation - sun_speed * 0.75)
            sun_changed = True

        if sun_changed:
            cos_el = math.sqrt(max(0.001, 1.0 - self._sun_elevation * self._sun_elevation))
            sun_x = math.sin(self._sun_azimuth) * cos_el
            sun_y = self._sun_elevation
            sun_z = math.cos(self._sun_azimuth) * cos_el
            app.pipeline.config.sun_direction = (sun_x, sun_y, sun_z)

        # 3. Update live HUD telemetry
        if self.hud is not None:
            cfg = app.pipeline.config
            _, preset_label = self.PRESET_NAMES[self._preset_id]
            pipeline_info = f"Shadows: {cfg.shadow_mode} ({cfg.pcf_samples}t) • AO: {cfg.ao_mode} • AA: {cfg.aa_mode} • GI: {cfg.gi_mode}"

            # Calculate total commands submitted across all active cascades
            casc_cmd_counts = [m.command_count for m in app.pipeline.cascade_shadow_mdis[:cfg.csm_cascades]]
            shadow_info = f"CSM Per-Cascade Commands: {casc_cmd_counts} (C0..C{cfg.csm_cascades-1})"

            debug_name = self.DEBUG_MODES[self._debug_idx][1]
            self.hud.update_telemetry(
                preset_name=preset_label,
                pipeline_info=pipeline_info,
                shadow_info=shadow_info,
                vis_ents=app.pipeline.stats_visible_entities,
                total_ents=app.ecs.active_count,
                vis_cmds=app.pipeline.stats_visible_commands,
                sun_azimuth=self._sun_azimuth,
                sun_elevation=self._sun_elevation,
                debug_mode_name=debug_name,
            )

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.stop()
                return True

            # Number keys [1-4]: Switch Hardware Scaled Presets
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                pid = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4}[event.key]
                self._apply_preset(app, pid)
                return True

            # [V]: Cycle G-Buffer / Shadow Atlas Debug Views
            elif event.key == pygame.K_v:
                self._debug_idx = (self._debug_idx + 1) % len(self.DEBUG_MODES)
                mode_val, _ = self.DEBUG_MODES[self._debug_idx]
                app.pipeline.config.debug_gbuffer = mode_val
                return True

        elif event.type == pygame.MOUSEMOTION and self.cam is not None:
            self.cam.handle_mouse_look(float(event.rel[0]), float(event.rel[1]))
            return True

        return False


# ---------------- 4. Application Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Beta Upgrade Testbench")
    parser.add_argument("--headless", action="store_true", help="Run in headless offscreen mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to simulate before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial Engine - Beta Upgrade Testbench & Showcase",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(BetaShowcaseWorldModule())
    app.state_manager.register(BetaShowcaseState())
    app.state_manager.set_state("BetaShowcase")

    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
