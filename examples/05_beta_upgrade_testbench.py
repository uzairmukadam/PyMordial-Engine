"""Beta Upgrade Testbench & Hardware-Scaling Showcase (Example 05).

Provides a comprehensive manual test scene for evaluating Decima-inspired engine upgrades:
- Phase 1: Per-Cascade CSM Shadow Culling verification (telemetry readout & cascade inspection)
- Phase 1: Hardware-Scaled Preset switching (Low -> Steam Deck/Medium -> High -> Ultra)
- Phase 2: Decima Sloped Depth Bias & GPU Hi-Z Occlusion Culling
- Decima-style Classical Courtyard & Monument Architecture with MegaBuffer procedural primitives
- Dynamic Clustered Local Lighting (SSBO 3 warm brazier point lights & focus spot lights)
- Real-time Sun Orbit & Elevation manipulation (Arrow keys or [T] auto-orbit)
- FreeCamera flight with live HUD diagnostics (FPS, ms, visible vs culled entities/commands)
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
    UIStyle,
)
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset


# ---------------- 1. HUD & Diagnostic UI ----------------

class BetaShowcaseHUD(UIScreen):
    """High-tech responsive HUD displaying live telemetry, culling stats, and active presets."""

    def __init__(self) -> None:
        super().__init__(name="BetaShowcaseHUD", is_modal=False)

        # Main Telemetry Panel (Top-Left)
        panel = UIPanel(x=24, y=20, w=480, h=250)
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
            w=448,
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
            w=448,
            h=18,
            font_size=11,
            bold=True,
            color=(0.4, 1.0, 0.5, 1.0),
        )
        panel.add_child(self.lbl_preset)

        self.lbl_pipeline = UILabel(
            "Shadows: PCSS • AO: GTAO • AA: TAA • GI: SSRC",
            x=16,
            y=58,
            w=448,
            h=18,
            font_size=10,
            color=(0.85, 0.90, 0.95, 0.9),
        )
        panel.add_child(self.lbl_pipeline)

        self.lbl_shadow_culling = UILabel(
            "CSM Per-Cascade Commands: [0, 0, 0, 0]",
            x=16,
            y=80,
            w=448,
            h=18,
            font_size=10,
            color=(1.0, 0.85, 0.3, 0.95),
        )
        panel.add_child(self.lbl_shadow_culling)

        self.lbl_cull_stats = UILabel(
            "Camera Visible: 0/0 Ents • 0 MDI Cmds",
            x=16,
            y=102,
            w=448,
            h=18,
            font_size=10,
            color=(0.7, 0.8, 0.9, 0.85),
        )
        panel.add_child(self.lbl_cull_stats)

        self.lbl_sun_info = UILabel(
            "Sun: Azimuth 0.0° • Elevation 0.0°",
            x=16,
            y=124,
            w=448,
            h=18,
            font_size=10,
            color=(1.0, 0.65, 0.35, 0.9),
        )
        panel.add_child(self.lbl_sun_info)

        self.lbl_optics_info = UILabel(
            "Optics: Flare ON • Bloom ON • DoF OFF • Blur OFF",
            x=16,
            y=146,
            w=448,
            h=18,
            font_size=10,
            color=(0.55, 0.85, 0.95, 0.9),
        )
        panel.add_child(self.lbl_optics_info)

        self.lbl_debug_view = UILabel(
            "Debug View: Normal Shading (Press [V] to cycle G-Buffer)",
            x=16,
            y=168,
            w=448,
            h=18,
            font_size=10,
            color=(0.6, 0.75, 0.9, 0.85),
        )
        panel.add_child(self.lbl_debug_view)

        controls = UILabel(
            "[1-4]: Presets • [Arrows]: Sun • [T]: Sun Orbit • [V]: Views • [O]: Optics • ESC: Quit",
            x=16,
            y=196,
            w=448,
            h=34,
            font_size=9,
            color=(0.65, 0.75, 0.85, 0.85),
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
        optics_info: str,
        debug_mode_name: str,
    ) -> None:
        self.lbl_preset.text = f"Quality Preset: {preset_name}"
        self.lbl_pipeline.text = pipeline_info
        self.lbl_shadow_culling.text = shadow_info
        self.lbl_cull_stats.text = f"Camera Visible: {vis_ents}/{total_ents} Ents • {vis_cmds} MDI Cmds"
        self.lbl_sun_info.text = f"Sun: Azimuth {math.degrees(sun_azimuth):.1f}° • Elevation {math.degrees(sun_elevation):.1f}°"
        self.lbl_optics_info.text = optics_info
        self.lbl_debug_view.text = f"Debug View: {debug_mode_name} ([V] to cycle)"


# ---------------- 2. World Module (Classical Monument Arena) ----------------

class BetaShowcaseWorldModule(ProjectModule):
    """Spawns an aesthetic architectural test arena:

    - Classical Colonnade / Pavilion using cylinders, architraves, and domes
    - Central Reflective Water Basin & Fountain Pedestal with bronze & gold monuments
    - Chrome mirror spheres and brushed metal pillars testing SSR, TAA specular stability, and GTAO
    - Tall stepped backdrop wall occluding distant monolith clusters for occlusion culling verification
    - Clustered local lights (braziers & spot lanterns) highlighting volumetric in-scattering
    """

    name = "BetaShowcaseWorldModule"

    def __init__(self) -> None:
        self._batches: list[tuple] = []
        self._batch_cache_valid = False

    def on_attach(self, app: ProjectApp) -> None:
        ecs = app.ecs

        # ---------------- Materials Palette ----------------
        mat_marble = (0.92, 0.92, 0.94, 1.0)
        mat_dark_stone = (0.18, 0.20, 0.24, 1.0)
        mat_terracotta = (0.75, 0.32, 0.20, 1.0)
        mat_gold = (0.95, 0.78, 0.20, 1.0)
        mat_bronze = (0.80, 0.50, 0.25, 1.0)
        mat_chrome = (0.94, 0.96, 0.98, 1.0)
        mat_obsidian = (0.05, 0.05, 0.07, 1.0)

        # 1. Vast Terrain Ground (Spans all CSM cascades)
        ecs.create_entity(
            position=(0.0, -1.0, 0.0),
            scale=(400.0, 2.0, 400.0),
            color=(0.55, 0.58, 0.60, 1.0),
            roughness=0.90,
            metallic=0.02,
            is_static=True,
        )

        # 2. Raised Courtyard Marble Terrace (Multi-tiered steps)
        ecs.create_entity(
            position=(0.0, 0.25, 0.0),
            scale=(64.0, 0.5, 64.0),
            color=mat_marble,
            roughness=0.35,
            metallic=0.05,
            is_static=True,
        )
        ecs.create_entity(
            position=(0.0, 0.65, 0.0),
            scale=(52.0, 0.4, 52.0),
            color=(0.88, 0.88, 0.90, 1.0),
            roughness=0.30,
            metallic=0.05,
            is_static=True,
        )

        # 3. Classical Colonnade (Grand Symmetrical Pavilion)
        col_count = 10
        col_radius_x = 22.0
        col_radius_z = 22.0
        col_height = 8.0
        col_width = 1.2

        for i in range(col_count):
            t = float(i) / float(col_count) * math.pi * 2.0
            cx = math.sin(t) * col_radius_x
            cz = math.cos(t) * col_radius_z

            # Pillar Base (Cube)
            ecs.create_entity(
                position=(cx, 1.2, cz),
                scale=(1.8, 0.7, 1.8),
                color=mat_dark_stone,
                roughness=0.75,
                metallic=0.10,
                is_static=True,
            )
            # Column Shaft (Cylinder)
            ecs.create_entity(
                position=(cx, 1.2 + col_height * 0.5, cz),
                scale=(col_width, col_height, col_width),
                color=mat_marble,
                roughness=0.40,
                metallic=0.05,
                is_static=True,
            )
            # Column Capital (Cube)
            ecs.create_entity(
                position=(cx, 1.2 + col_height + 0.35, cz),
                scale=(1.8, 0.7, 1.8),
                color=mat_dark_stone,
                roughness=0.75,
                metallic=0.10,
                is_static=True,
            )

        # Entablature Architrave Ring over Columns (4 segmented beams)
        ring_beam_w = 46.0
        ecs.create_entity(
            position=(0.0, 1.2 + col_height + 1.2, col_radius_z),
            scale=(ring_beam_w, 1.0, 2.5),
            color=mat_marble,
            roughness=0.45,
            metallic=0.05,
            is_static=True,
        )
        ecs.create_entity(
            position=(0.0, 1.2 + col_height + 1.2, -col_radius_z),
            scale=(ring_beam_w, 1.0, 2.5),
            color=mat_marble,
            roughness=0.45,
            metallic=0.05,
            is_static=True,
        )
        ecs.create_entity(
            position=(col_radius_x, 1.2 + col_height + 1.2, 0.0),
            scale=(2.5, 1.0, ring_beam_w),
            color=mat_marble,
            roughness=0.45,
            metallic=0.05,
            is_static=True,
        )
        ecs.create_entity(
            position=(-col_radius_x, 1.2 + col_height + 1.2, 0.0),
            scale=(2.5, 1.0, ring_beam_w),
            color=mat_marble,
            roughness=0.45,
            metallic=0.05,
            is_static=True,
        )

        # 4. Central Monument & Specular Showcase
        # Center Pedestal (Terracotta / Dark Obsidian Steps)
        ecs.create_entity(
            position=(0.0, 1.4, 0.0),
            scale=(8.0, 1.2, 8.0),
            color=mat_obsidian,
            roughness=0.15,
            metallic=0.85,
            is_static=True,
        )
        ecs.create_entity(
            position=(0.0, 2.4, 0.0),
            scale=(5.0, 1.0, 5.0),
            color=mat_terracotta,
            roughness=0.65,
            metallic=0.10,
            is_static=True,
        )

        # Large Golden Sun Sphere (Hero Specular Object)
        ecs.create_entity(
            position=(0.0, 4.4, 0.0),
            scale=(2.4, 2.4, 2.4),
            color=mat_gold,
            roughness=0.12,
            metallic=0.98,
            is_static=True,
        )

        # Orbiting Chrome Mirror Sphere & Bronze Relic
        ecs.create_entity(
            position=(-5.5, 2.8, -3.0),
            scale=(2.0, 2.0, 2.0),
            color=mat_chrome,
            roughness=0.02,
            metallic=1.0,
            is_static=True,
        )
        ecs.create_entity(
            position=(5.5, 2.8, 3.0),
            scale=(2.0, 2.0, 2.0),
            color=mat_bronze,
            roughness=0.22,
            metallic=0.92,
            is_static=True,
        )

        # 5. Massive Occlusion Backdrop Wall (Tests Hi-Z & CSM Per-Cascade Culling)
        ecs.create_entity(
            position=(0.0, 12.0, -42.0),
            scale=(90.0, 24.0, 5.0),
            color=mat_dark_stone,
            roughness=0.80,
            metallic=0.15,
            is_static=True,
        )

        # Dense Occluded Monolith Field behind the backdrop (Far Cascades C2 & C3 Stress)
        for gx in range(-6, 7):
            for gz in range(4):
                px = float(gx * 7.5)
                pz = -55.0 - float(gz * 9.0)
                h_rand = 6.0 + abs(gx) * 1.5
                ecs.create_entity(
                    position=(px, h_rand * 0.5, pz),
                    scale=(2.2, h_rand, 2.2),
                    color=(0.85, 0.25, 0.25, 1.0) if (gx % 2 == 0) else (0.25, 0.55, 0.90, 1.0),
                    roughness=0.55,
                    metallic=0.20,
                    is_static=True,
                )

        # 6. Slanted Flanking Obelisks (Sloped Depth Bias Verification)
        for i in range(4):
            side = -1.0 if (i % 2 == 0) else 1.0
            dist_z = 12.0 + float(i // 2) * 14.0
            ecs.create_entity(
                position=(side * 18.0, 4.0, dist_z),
                scale=(1.8, 8.0, 1.8),
                color=(0.90, 0.88, 0.85, 1.0),
                roughness=0.40,
                metallic=0.15,
                is_static=True,
            )

        # 7. Clustered Dynamic Local Lights (Braziers & Corner Spotlights)
        pipe = app.pipeline
        if hasattr(pipe, "lights_pass"):
            pipe.lights_pass.clear()
            # 4 Warm Amber Braziers near courtyard corners
            pipe.add_point_light(
                position=(-14.0, 3.0, -14.0),
                radius=18.0,
                color=(1.0, 0.55, 0.15),
                intensity=4.5,
            )
            pipe.add_point_light(
                position=(14.0, 3.0, -14.0),
                radius=18.0,
                color=(1.0, 0.55, 0.15),
                intensity=4.5,
            )
            pipe.add_point_light(
                position=(-14.0, 3.0, 14.0),
                radius=18.0,
                color=(1.0, 0.55, 0.15),
                intensity=4.5,
            )
            pipe.add_point_light(
                position=(14.0, 3.0, 14.0),
                radius=18.0,
                color=(1.0, 0.55, 0.15),
                intensity=4.5,
            )
            # High-intensity blue spot uplight on the central golden monument
            pipe.add_spot_light(
                position=(0.0, 1.2, -6.0),
                direction=(0.0, 0.6, 0.8),
                radius=25.0,
                inner_cone_angle=math.radians(18.0),
                outer_cone_angle=math.radians(35.0),
                color=(0.35, 0.75, 1.0),
                intensity=6.0,
            )

        # Configure cinematic golden hour atmosphere defaults
        pipe.config.sun_intensity = 4.2
        pipe.config.sun_direction = (0.55, -0.60, -0.58)
        if hasattr(pipe, "atmosphere"):
            pipe.atmosphere.config.day_speed = 0.0

    def get_draw_batches(self, app: ProjectApp) -> list[tuple]:
        """Provides batch allocations for all scene entities."""
        if not self._batch_cache_valid or not self._batches:
            self._batches.clear()
            alloc_cube = app.pipeline.mega_buffer.allocations.get("cube")
            alloc_sphere = app.pipeline.mega_buffer.allocations.get("sphere")
            alloc_cylinder = app.pipeline.mega_buffer.allocations.get("cylinder")

            cnt = app.ecs.active_count
            if cnt > 0 and alloc_cube is not None:
                # Group all primary architectural geometry
                self._batches.append((alloc_cube, cnt, 0, False, True))
            self._batch_cache_valid = True

        return self._batches


# ---------------- 3. Simulation Game State ----------------

class BetaShowcaseState(GameState):
    """Interactive state controlling camera flight, dynamic sun orbit, and diagnostic toggles."""

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
        (11, "Hi-Z Depth Pyramid"),
    ]

    def __init__(self) -> None:
        self.cam: FreeCamera | None = None
        self.hud: BetaShowcaseHUD | None = None
        self._preset_id = 3  # Default to HIGH
        self._debug_idx = 0
        self._sun_azimuth = 0.78
        self._sun_elevation = -0.62
        self._auto_orbit = False
        self._orbit_speed = 0.12

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.cam is None:
            self.cam = FreeCamera(position=(0.0, 7.5, 26.0), yaw_deg=-90.0, pitch_deg=-12.0, move_speed=18.0)
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

        # 2. Interactive Sun Controls & Auto-Orbit
        sun_changed = False
        if self._auto_orbit:
            self._sun_azimuth = (self._sun_azimuth + self._orbit_speed * dt) % (math.pi * 2.0)
            sun_changed = True

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

            casc_cmd_counts = [m.command_count for m in app.pipeline.cascade_shadow_mdis[:cfg.csm_cascades]]
            shadow_info = f"CSM Per-Cascade Commands: {casc_cmd_counts} (C0..C{cfg.csm_cascades-1})"

            dof_on = "ON" if getattr(cfg, "dof_enabled", False) else "OFF"
            mb_on = "ON" if getattr(cfg, "motion_blur_enabled", False) else "OFF"
            flare_on = "ON" if getattr(cfg, "lens_flare_enabled", True) else "OFF"
            bloom_on = "ON" if getattr(cfg, "bloom_enabled", True) else "OFF"
            optics_info = f"Optics: Flare {flare_on} • Bloom {bloom_on} • DoF {dof_on} • Blur {mb_on}"

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
                optics_info=optics_info,
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

            # [T]: Toggle Sun Auto-Orbit
            elif event.key == pygame.K_t:
                self._auto_orbit = not self._auto_orbit
                return True

            # [O]: Cycle Creative Optics
            elif event.key == pygame.K_o:
                et = app.engine_tweaks
                if et is not None:
                    et.lens_flare_enabled = not et.lens_flare_enabled
                    et.bloom_enabled = not et.bloom_enabled
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
        width=1920,
        height=1080,
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
