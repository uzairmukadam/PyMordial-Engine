"""PyMordial Engine Example 03: Full Graphics & Display Configuration In-Game Menu.

Demonstrates an empty 3D scene with:
1. Dynamic Rapier3D physics cubes, stepping curbs, and a controllable kinematic character motor.
2. In-game ModernGL AAA glassmorphic settings menu with staged changes and a dynamic Apply button:
   - Display: Resolution (720p, 900p, 1080p), Mode (Windowed, Borderless), VSync (Off, On).
   - Presets: Low, Medium, High, Ultra, Cinematic (via get_quality_preset).
   - Advanced: SSAO / GTAO, Bloom, Screen-Space Reflections (SSR), Volumetric Fog, and Anti-Aliasing (FXAA / TAA).
   - Staging: Changes are held in a pending state until confirmed via the activated [APPLY CHANGES] button.
3. Stack-based GameStateManager transitions (MainMenu -> Simulation -> Pause Overlay -> Settings).
4. Live Telemetry HUD reflecting real-time pipeline status and player kinematics.

Zero debug menu dependency. All configuration runs through public engine APIs.

Usage:
    python examples/03_full_graphics_and_display_menu.py
    python examples/03_full_graphics_and_display_menu.py --headless --frames 60
"""

from __future__ import annotations
import argparse
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
    CharacterMotor,
    CharacterMotorConfig,
    FollowCamera,
    FollowCameraConfig,
    UIScreen,
    UIPanel,
    UILabel,
    UIButton,
    UISegmentGroup,
    UIStyle,
    WindowMode,
    VSyncMode,
    get_quality_preset,
)
from engine.world.default_world import DefaultWorldBuilder
from engine.input.codes import GamepadButton


# ---------------- Modern Glassmorphic Helper ----------------

def _create_glass_card(
    x: float,
    y: float,
    w: float,
    h: float,
    border_accent: tuple[float, float, float, float] = (0.18, 0.65, 0.95, 0.5),
) -> UIPanel:
    """Creates a sleek dark obsidian glass card with rounded corners and glowing border."""
    panel = UIPanel(x=x, y=y, w=w, h=h)
    panel.style = UIStyle(
        bg_color_top=(0.06, 0.08, 0.13, 0.93),
        bg_color_bottom=(0.02, 0.03, 0.06, 0.97),
        border_color=border_accent,
        border_width=1.5,
        corner_radius=12.0,
    )
    return panel


# ---------------- Live Telemetry HUD ----------------

class TelemetryHUD(UIScreen):
    """Real-time HUD displaying active resolution, preset, rendering features, and kinematics."""

    def __init__(self) -> None:
        super().__init__(name="TelemetryHUD", is_modal=False)

        # 1. Top-Left Display & Performance Card
        disp_card = _create_glass_card(x=24, y=20, w=320, h=96, border_accent=(0.15, 0.85, 1.0, 0.4))
        self.add_child(disp_card)

        self.fps_lbl = UILabel("FPS: -- | -- ms", x=16, y=10, w=290, h=24, font_size=15, bold=True, color=(0.20, 0.95, 0.60, 1.0))
        disp_card.add_child(self.fps_lbl)

        self.res_lbl = UILabel("RES: 1280x720 (Windowed)", x=16, y=36, w=290, h=22, font_size=12, color=(0.85, 0.90, 0.95, 0.9))
        disp_card.add_child(self.res_lbl)

        self.vsync_lbl = UILabel("VSYNC: OFF", x=16, y=60, w=290, h=22, font_size=12, color=(0.60, 0.80, 1.0, 0.85))
        disp_card.add_child(self.vsync_lbl)

        # 2. Top-Right Active Pipeline Features Card
        gfx_card = _create_glass_card(x=936, y=20, w=320, h=118, border_accent=(0.95, 0.70, 0.20, 0.45))
        self.add_child(gfx_card)

        self.preset_lbl = UILabel("QUALITY: HIGH", x=16, y=10, w=290, h=24, font_size=15, bold=True, color=(1.0, 0.82, 0.25, 1.0))
        gfx_card.add_child(self.preset_lbl)

        self.features_line1 = UILabel("AO: GTAO  •  BLOOM: ON  •  SSR: ON", x=16, y=38, w=290, h=20, font_size=11, color=(0.8, 0.85, 0.9, 0.85))
        gfx_card.add_child(self.features_line1)

        self.features_line2 = UILabel("VOL FOG: ON  •  AA: FXAA", x=16, y=60, w=290, h=20, font_size=11, color=(0.8, 0.85, 0.9, 0.85))
        gfx_card.add_child(self.features_line2)

        self.cfg_hint = UILabel("Press [ESC] for Graphics Menu", x=16, y=86, w=290, h=18, font_size=11, bold=True, color=(0.18, 0.88, 1.0, 0.9))
        gfx_card.add_child(self.cfg_hint)

        # 3. Bottom-Left Character Telemetry Card
        kin_card = _create_glass_card(x=24, y=590, w=340, h=105, border_accent=(0.20, 0.75, 0.95, 0.35))
        self.add_child(kin_card)

        self.speed_lbl = UILabel("SPEED: 0.0 KM/H", x=16, y=10, w=300, h=24, font_size=15, bold=True, color=(0.18, 0.90, 1.0, 1.0))
        kin_card.add_child(self.speed_lbl)

        self.stance_lbl = UILabel("STATUS: Grounded | Idle", x=16, y=36, w=300, h=20, font_size=12, color=(0.3, 0.8, 1.0, 0.9))
        kin_card.add_child(self.stance_lbl)

        self.ctrl_lbl = UILabel("Move: WASD / Left Stick  •  Sprint: Shift / (L3)  •  Jump: Space / (A)", x=16, y=64, w=320, h=20, font_size=10, color=(0.65, 0.75, 0.85, 0.8))
        kin_card.add_child(self.ctrl_lbl)

    def update_telemetry(
        self,
        fps: float,
        dt_ms: float,
        width: int,
        height: int,
        mode_name: str,
        vsync_name: str,
        preset_name: str,
        ao_mode: str,
        bloom: bool,
        ssr: bool,
        fog: bool,
        aa_mode: str,
        speed_mps: float,
        is_grounded: bool,
        is_sprinting: bool,
        flare: bool = True,
    ) -> None:
        self.fps_lbl.text = f"FPS: {fps:.0f} | {dt_ms:.1f} ms"
        self.res_lbl.text = f"RES: {width}x{height} ({mode_name})"
        self.vsync_lbl.text = f"VSYNC: {vsync_name}"

        self.preset_lbl.text = f"QUALITY: {preset_name.upper()}"
        b_str = "ON" if bloom else "OFF"
        s_str = "ON" if ssr else "OFF"
        f_str = "ON" if fog else "OFF"
        fl_str = "ON" if flare else "OFF"
        self.features_line1.text = f"AO: {ao_mode}  •  BLOOM: {b_str}  •  SSR: {s_str}"
        self.features_line2.text = f"VOL FOG: {f_str}  •  FLARE: {fl_str}  •  AA: {aa_mode}"

        kmh = speed_mps * 3.6
        self.speed_lbl.text = f"SPEED: {kmh:.1f} KM/H"
        ground_str = "Grounded" if is_grounded else "In Air"
        stance = "Sprinting" if (is_sprinting and speed_mps > 0.5) else ("Walking" if speed_mps > 0.5 else "Idle")
        if not is_grounded:
            stance = "Airborne"
        self.stance_lbl.text = f"STATUS: {ground_str} | {stance}"


# ---------------- Full In-Game Graphics & Window Settings UI ----------------

class GraphicsSettingsUI(UIScreen):
    """In-game configuration screen with staged modifications and an active Apply button."""

    def __init__(
        self,
        app: ProjectApp,
        on_return: Any,
    ) -> None:
        super().__init__(name="GraphicsSettingsScreen")
        self.app = app
        self.on_return_cb = on_return

        # Settings state containers
        self.active_settings: dict[str, Any] = {}
        self.pending_settings: dict[str, Any] = {}

        # 1. Dark Glass Overlay Backdrop
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.04, 0.05, 0.08, 0.94),
            bg_color_bottom=(0.01, 0.02, 0.04, 0.98),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(bg)

        # 2. Glowing Header Bar
        header = _create_glass_card(x=60, y=30, w=1160, h=90, border_accent=(0.15, 0.85, 1.0, 0.5))
        bg.add_child(header)

        title = UILabel("GRAPHICS & DISPLAY CONFIGURATION", x=30, y=14, w=800, h=36, font_size=26, bold=True, color=(0.18, 0.88, 1.0, 1.0))
        header.add_child(title)

        subtitle = UILabel("LIVE MODERNGL PIPELINE ADJUSTMENT  •  STAGED SETTINGS  •  APPLY REQUIRED", x=32, y=52, w=800, h=22, font_size=12, color=(0.95, 0.75, 0.20, 0.95))
        header.add_child(subtitle)

        # 3. Left Panel: Display & Master Presets (x=60, y=135, w=565, h=495)
        left_card = _create_glass_card(x=60, y=135, w=565, h=495)
        bg.add_child(left_card)

        lbl_disp = UILabel("DISPLAY & WINDOW", x=24, y=16, w=500, h=24, font_size=14, bold=True, color=(0.3, 0.85, 1.0, 0.95))
        left_card.add_child(lbl_disp)

        # Resolution Segment
        lbl_res = UILabel("SCREEN RESOLUTION", x=24, y=48, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        left_card.add_child(lbl_res)

        self.seg_res = UISegmentGroup(
            options=[
                ("1280x720", (1280, 720)),
                ("1600x900", (1600, 900)),
                ("1920x1080", (1920, 1080)),
            ],
            selected_index=0,
            x=24,
            y=70,
            w=517,
            h=36,
            on_change=lambda val: self._on_staged_change("resolution", val),
        )
        left_card.add_child(self.seg_res)

        # Window Mode Segment
        lbl_mode = UILabel("WINDOW MODE", x=24, y=120, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        left_card.add_child(lbl_mode)

        self.seg_mode = UISegmentGroup(
            options=[
                ("Windowed", WindowMode.WINDOWED),
                ("Borderless Fullscreen", WindowMode.BORDERLESS_FULLSCREEN),
            ],
            selected_index=0,
            x=24,
            y=142,
            w=517,
            h=36,
            on_change=lambda val: self._on_staged_change("mode", val),
        )
        left_card.add_child(self.seg_mode)

        # VSync Segment
        lbl_vsync = UILabel("VERTICAL SYNCHRONIZATION (VSYNC)", x=24, y=192, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        left_card.add_child(lbl_vsync)

        self.seg_vsync = UISegmentGroup(
            options=[
                ("VSync OFF (Uncapped)", VSyncMode.OFF),
                ("VSync ON (Locked)", VSyncMode.ON),
            ],
            selected_index=0,
            x=24,
            y=214,
            w=517,
            h=36,
            on_change=lambda val: self._on_staged_change("vsync", val),
        )
        left_card.add_child(self.seg_vsync)

        # Master Presets Section
        lbl_pre = UILabel("MASTER QUALITY PRESETS", x=24, y=275, w=500, h=24, font_size=14, bold=True, color=(0.95, 0.75, 0.20, 0.95))
        left_card.add_child(lbl_pre)

        self.seg_presets = UISegmentGroup(
            options=[
                ("Low", "low"),
                ("Med", "medium"),
                ("High", "high"),
                ("Ultra", "ultra"),
                ("Cine", "cinematic"),
            ],
            selected_index=2,
            x=24,
            y=305,
            w=517,
            h=38,
            on_change=self._on_preset_staged,
        )
        left_card.add_child(self.seg_presets)

        preset_desc = UILabel(
            "Presets populate balanced shadow cascades, SSAO, volumetric fog,\nand SSR. Changes require clicking [APPLY CHANGES] below.",
            x=24,
            y=355,
            w=517,
            h=40,
            font_size=11,
            color=(0.65, 0.72, 0.82, 0.75),
        )
        left_card.add_child(preset_desc)

        # Restore defaults button
        btn_reset = UIButton(
            text="STAGE RECOMMENDED HIGH DEFAULTS",
            x=24,
            y=425,
            w=517,
            h=44,
            variant="neutral",
            font_size=13,
            on_click=self._on_stage_defaults,
        )
        left_card.add_child(btn_reset)

        # 4. Right Panel: Advanced Pipeline Toggles (x=655, y=135, w=565, h=495)
        right_card = _create_glass_card(x=655, y=135, w=565, h=495)
        bg.add_child(right_card)

        lbl_adv = UILabel("ADVANCED PIPELINE TOGGLES", x=24, y=14, w=500, h=24, font_size=14, bold=True, color=(0.3, 0.85, 1.0, 0.95))
        right_card.add_child(lbl_adv)

        # Ambient Occlusion (AO)
        lbl_ao = UILabel("AMBIENT OCCLUSION (AO)", x=24, y=44, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_ao)

        self.seg_ao = UISegmentGroup(
            options=[
                ("Disabled", "OFF"),
                ("SSAO (Fast)", "SSAO"),
                ("GTAO (Ground-Truth)", "GTAO"),
            ],
            selected_index=2,
            x=24,
            y=64,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("ao", val),
        )
        right_card.add_child(self.seg_ao)

        # Bloom & HDR
        lbl_bloom = UILabel("BLOOM & HDR GLOW", x=24, y=102, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_bloom)

        self.seg_bloom = UISegmentGroup(
            options=[
                ("Disabled", False),
                ("Enabled (Pyramid Bloom)", True),
            ],
            selected_index=1,
            x=24,
            y=122,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("bloom", val),
        )
        right_card.add_child(self.seg_bloom)

        # Screen-Space Reflections (SSR)
        lbl_ssr = UILabel("SCREEN-SPACE REFLECTIONS (SSR)", x=24, y=160, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_ssr)

        self.seg_ssr = UISegmentGroup(
            options=[
                ("Disabled", False),
                ("Enabled (Ray-Marched)", True),
            ],
            selected_index=1,
            x=24,
            y=180,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("ssr", val),
        )
        right_card.add_child(self.seg_ssr)

        # Volumetric Fog
        lbl_fog = UILabel("VOLUMETRIC FROXEL FOG", x=24, y=218, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_fog)

        self.seg_fog = UISegmentGroup(
            options=[
                ("Disabled", False),
                ("Enabled (Scattering 3D)", True),
            ],
            selected_index=1,
            x=24,
            y=238,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("fog", val),
        )
        right_card.add_child(self.seg_fog)

        # Anamorphic Lens Flare
        lbl_flare = UILabel("ANAMORPHIC LENS FLARE", x=24, y=276, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_flare)

        self.seg_flare = UISegmentGroup(
            options=[
                ("Disabled", False),
                ("Enabled (Streaks & Halo)", True),
            ],
            selected_index=1,
            x=24,
            y=296,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("lens_flare", val),
        )
        right_card.add_child(self.seg_flare)

        # Anti-Aliasing
        lbl_aa = UILabel("ANTI-ALIASING (AA)", x=24, y=334, w=500, h=18, font_size=11, bold=True, color=(0.7, 0.75, 0.85, 0.8))
        right_card.add_child(lbl_aa)

        self.seg_aa = UISegmentGroup(
            options=[
                ("Disabled", "OFF"),
                ("FXAA", "FXAA"),
                ("TAA (Temporal)", "TAA"),
            ],
            selected_index=1,
            x=24,
            y=354,
            w=517,
            h=32,
            on_change=lambda val: self._on_staged_change("aa", val),
        )
        right_card.add_child(self.seg_aa)

        lbl_note = UILabel(
            "Individual toggles override presets. Nothing is modified until [APPLY CHANGES] is clicked.",
            x=24,
            y=400,
            w=517,
            h=30,
            font_size=11,
            color=(0.55, 0.65, 0.75, 0.7),
        )
        right_card.add_child(lbl_note)

        # 5. Bottom Navigation Bar (x=60, y=642, w=1160, h=54)
        footer = _create_glass_card(x=60, y=642, w=1160, h=54, border_accent=(0.15, 0.85, 1.0, 0.3))
        bg.add_child(footer)

        # The Dynamic Apply Button
        self.btn_apply = UIButton(
            text="SETTINGS APPLIED",
            x=20,
            y=6,
            w=230,
            h=42,
            variant="neutral",
            font_size=14,
            on_click=self._on_apply_clicked,
        )
        footer.add_child(self.btn_apply)

        # Discard button
        self.btn_discard = UIButton(
            text="DISCARD CHANGES",
            x=260,
            y=6,
            w=190,
            h=42,
            variant="neutral",
            font_size=13,
            on_click=self._on_discard_clicked,
        )
        footer.add_child(self.btn_discard)

        # Return button
        btn_return = UIButton(
            text="RETURN",
            x=460,
            y=6,
            w=150,
            h=42,
            variant="neutral",
            font_size=14,
            on_click=self.on_return_cb,
        )
        footer.add_child(btn_return)

        self.status_msg = UILabel(
            "All settings synchronized with active ModernGL pipeline.",
            x=630,
            y=16,
            w=510,
            h=24,
            font_size=12,
            color=(0.20, 0.90, 0.60, 0.95),
        )
        footer.add_child(self.status_msg)

    def sync_from_pipeline(self) -> None:
        """Reads active engine state into active_settings and clones it into pending_settings."""
        win = self.app.render_ctx.window
        cfg = self.app.pipeline.config

        preset_val = getattr(cfg, "preset_name", "high").lower()
        if preset_val not in ("low", "medium", "high", "ultra", "cinematic"):
            preset_val = "high"

        self.active_settings = {
            "resolution": (win.width, win.height),
            "mode": win.mode,
            "vsync": win.vsync,
            "preset": preset_val,
            "ao": getattr(cfg, "ao_mode", "GTAO"),
            "bloom": bool(getattr(cfg, "bloom_enabled", True)),
            "ssr": bool(getattr(cfg, "ssr_enabled", True)),
            "fog": bool(getattr(cfg, "volumetric_fog_enabled", True)),
            "lens_flare": bool(getattr(cfg, "lens_flare_enabled", True)),
            "aa": getattr(cfg, "aa_mode", "FXAA"),
        }
        self.pending_settings = dict(self.active_settings)
        self._refresh_segments_from_pending()
        self._update_apply_button_state()

    def _refresh_segments_from_pending(self) -> None:
        # Resolution
        cur_res = self.pending_settings.get("resolution", (1280, 720))
        for i, opt in enumerate(self.seg_res.options):
            if opt[1] == cur_res:
                self.seg_res.selected_index = i
                break

        # Mode
        cur_mode = self.pending_settings.get("mode", WindowMode.WINDOWED)
        for i, opt in enumerate(self.seg_mode.options):
            if opt[1] == cur_mode:
                self.seg_mode.selected_index = i
                break

        # VSync
        cur_vsync = self.pending_settings.get("vsync", VSyncMode.OFF)
        for i, opt in enumerate(self.seg_vsync.options):
            if opt[1] == cur_vsync:
                self.seg_vsync.selected_index = i
                break

        # Preset
        cur_pre = self.pending_settings.get("preset", "high")
        for i, opt in enumerate(self.seg_presets.options):
            if opt[1] == cur_pre:
                self.seg_presets.selected_index = i
                break

        # AO
        cur_ao = self.pending_settings.get("ao", "GTAO")
        for i, opt in enumerate(self.seg_ao.options):
            if opt[1] == cur_ao:
                self.seg_ao.selected_index = i
                break

        # Bloom
        b_val = self.pending_settings.get("bloom", True)
        self.seg_bloom.selected_index = 1 if b_val else 0

        # SSR
        s_val = self.pending_settings.get("ssr", True)
        self.seg_ssr.selected_index = 1 if s_val else 0

        # Fog
        f_val = self.pending_settings.get("fog", True)
        self.seg_fog.selected_index = 1 if f_val else 0

        # Lens Flare
        fl_val = self.pending_settings.get("lens_flare", True)
        self.seg_flare.selected_index = 1 if fl_val else 0

        # AA
        cur_aa = self.pending_settings.get("aa", "FXAA")
        for i, opt in enumerate(self.seg_aa.options):
            if opt[1] == cur_aa:
                self.seg_aa.selected_index = i
                break

    def has_pending_changes(self) -> bool:
        return any(self.pending_settings[k] != self.active_settings[k] for k in self.active_settings)

    def _update_apply_button_state(self) -> None:
        changed_count = sum(1 for k in self.active_settings if self.pending_settings[k] != self.active_settings[k])
        if changed_count > 0:
            self.btn_apply.text = "APPLY CHANGES"
            self.btn_apply.variant = "primary"
            self.btn_discard.variant = "danger"
            self.status_msg.text = f"{changed_count} change{'s' if changed_count > 1 else ''} pending. Click [APPLY CHANGES] to activate."
            self.status_msg.color = (1.0, 0.85, 0.25, 1.0)
        else:
            self.btn_apply.text = "SETTINGS APPLIED"
            self.btn_apply.variant = "neutral"
            self.btn_discard.variant = "neutral"
            self.status_msg.text = "All settings synchronized with active ModernGL pipeline."
            self.status_msg.color = (0.20, 0.90, 0.60, 0.95)

    def _on_staged_change(self, key: str, val: Any) -> None:
        self.pending_settings[key] = val
        self._update_apply_button_state()

    def _on_preset_staged(self, preset_key: str) -> None:
        self.pending_settings["preset"] = preset_key
        p_cfg = get_quality_preset(preset_key)
        self.pending_settings["ao"] = p_cfg.ao_mode
        self.pending_settings["bloom"] = p_cfg.bloom_enabled
        self.pending_settings["ssr"] = p_cfg.ssr_enabled
        self.pending_settings["fog"] = p_cfg.volumetric_fog_enabled
        self.pending_settings["lens_flare"] = p_cfg.lens_flare_enabled
        self.pending_settings["aa"] = p_cfg.aa_mode
        self._refresh_segments_from_pending()
        self._update_apply_button_state()

    def _on_stage_defaults(self) -> None:
        p_cfg = get_quality_preset("high")
        self.pending_settings = {
            "resolution": (1280, 720),
            "mode": WindowMode.WINDOWED,
            "vsync": VSyncMode.OFF,
            "preset": "high",
            "ao": p_cfg.ao_mode,
            "bloom": p_cfg.bloom_enabled,
            "ssr": p_cfg.ssr_enabled,
            "fog": p_cfg.volumetric_fog_enabled,
            "lens_flare": p_cfg.lens_flare_enabled,
            "aa": p_cfg.aa_mode,
        }
        self._refresh_segments_from_pending()
        self._update_apply_button_state()

    def _on_discard_clicked(self) -> None:
        if self.has_pending_changes():
            self.pending_settings = dict(self.active_settings)
            self._refresh_segments_from_pending()
            self._update_apply_button_state()
            self.status_msg.text = "Discarded pending changes. Active configuration restored."

    def _on_apply_clicked(self) -> None:
        if not self.has_pending_changes():
            return

        # 1. Apply Window Settings
        if self.pending_settings["resolution"] != self.active_settings["resolution"]:
            w, h = self.pending_settings["resolution"]
            self.app.render_ctx.window.set_resolution(w, h)

        if self.pending_settings["mode"] != self.active_settings["mode"]:
            self.app.render_ctx.window.set_mode(self.pending_settings["mode"])

        if self.pending_settings["vsync"] != self.active_settings["vsync"]:
            self.app.render_ctx.window.set_vsync(self.pending_settings["vsync"])

        # 2. Apply Pipeline Graphics Configuration
        preset_key = self.pending_settings["preset"]
        new_cfg = get_quality_preset(preset_key)
        new_cfg.ao_mode = self.pending_settings["ao"]
        new_cfg.bloom_enabled = self.pending_settings["bloom"]
        new_cfg.ssr_enabled = self.pending_settings["ssr"]
        new_cfg.volumetric_fog_enabled = self.pending_settings["fog"]
        new_cfg.lens_flare_enabled = self.pending_settings["lens_flare"]
        new_cfg.aa_mode = self.pending_settings["aa"]
        new_cfg.dof_enabled = False  # Keep DOF disabled across all presets in examples!

        self.app.pipeline.apply_config(new_cfg)

        # Re-assert mouse grab & capture
        self.app.input_manager.set_mouse_grab(True)
        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        except Exception:
            pass

        # 3. Synchronize Active
        self.active_settings = dict(self.pending_settings)
        self._update_apply_button_state()
        self.status_msg.text = "Applied configuration changes successfully to active ModernGL pipeline!"
        self.status_msg.color = (0.20, 0.95, 0.60, 1.0)


# ---------------- Modal Pause Menu ----------------

class PauseMenuUI(UIScreen):
    """Modal pause menu with quick access to resume, graphics configuration, and main menu."""

    def __init__(self, on_resume: Any, on_settings: Any, on_quit_menu: Any) -> None:
        super().__init__(name="PauseMenuScreen")

        backdrop = UIPanel(x=0, y=0, w=1280, h=720)
        backdrop.style = UIStyle(
            bg_color_top=(0.0, 0.0, 0.0, 0.65),
            bg_color_bottom=(0.0, 0.0, 0.0, 0.85),
            corner_radius=0.0,
        )
        self.add_child(backdrop)

        modal = UIPanel(x=420, y=140, w=440, h=440)
        modal.style = UIStyle(
            bg_color_top=(0.07, 0.09, 0.15, 0.96),
            bg_color_bottom=(0.03, 0.04, 0.07, 0.98),
            border_color=(0.18, 0.75, 1.0, 0.7),
            border_width=1.5,
            corner_radius=12.0,
        )
        backdrop.add_child(modal)

        lbl = UILabel("SIMULATION PAUSED", x=0, y=25, w=440, h=34, font_size=24, bold=True, align="center", color=(0.18, 0.88, 1.0, 1.0))
        modal.add_child(lbl)

        sub = UILabel("Kinematics & Physics Simulation Frozen", x=0, y=62, w=440, h=20, font_size=12, align="center", color=(0.7, 0.75, 0.85, 0.7))
        modal.add_child(sub)

        btn_resume = UIButton(text="RESUME SIMULATION", x=45, y=105, w=350, h=48, variant="primary", font_size=16, on_click=on_resume)
        modal.add_child(btn_resume)

        btn_cfg = UIButton(text="GRAPHICS & DISPLAY CONFIG", x=45, y=170, w=350, h=48, variant="accent", font_size=15, on_click=on_settings)
        modal.add_child(btn_cfg)

        btn_menu = UIButton(text="QUIT TO MAIN MENU", x=45, y=235, w=350, h=48, variant="danger", font_size=16, on_click=on_quit_menu)
        modal.add_child(btn_menu)

        hint = UILabel("Press [ESC] to Resume Instantly", x=0, y=385, w=440, h=20, font_size=12, align="center", color=(0.55, 0.65, 0.75, 0.6))
        modal.add_child(hint)


# ---------------- Fullscreen Main Menu ----------------

class MainMenuUI(UIScreen):
    """Modern obsidian glass main menu."""

    def __init__(self, on_start: Any, on_settings: Any, on_exit: Any) -> None:
        super().__init__(name="MainMenuScreen")

        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.03, 0.04, 0.07, 0.95),
            bg_color_bottom=(0.01, 0.02, 0.04, 0.98),
            corner_radius=0.0,
            border_width=0.0,
        )
        self.add_child(bg)

        # Header Card
        header = _create_glass_card(x=80, y=60, w=1120, h=110, border_accent=(0.15, 0.80, 1.0, 0.45))
        bg.add_child(header)

        title = UILabel("PYMORDIAL ENGINE 3D", x=35, y=18, w=800, h=42, font_size=32, bold=True, color=(0.18, 0.88, 1.0, 1.0))
        header.add_child(title)

        subtitle = UILabel("GRAPHICS & DISPLAY CONFIGURATION DEMO  •  PBR PIPELINE  •  RAPIER3D", x=37, y=64, w=800, h=24, font_size=13, bold=True, color=(0.95, 0.75, 0.20, 1.0))
        header.add_child(subtitle)

        # Left Column: Action Buttons Card (x=80, y=190, w=480, h=470)
        menu_card = _create_glass_card(x=80, y=190, w=480, h=470)
        bg.add_child(menu_card)

        sec_title = UILabel("NAVIGATION & OPTIONS", x=30, y=20, w=420, h=28, font_size=14, bold=True, color=(0.4, 0.8, 1.0, 0.9))
        menu_card.add_child(sec_title)

        btn_start = UIButton(text="ENTER 3D SIMULATION", x=30, y=65, w=420, h=54, variant="primary", font_size=17, on_click=on_start)
        menu_card.add_child(btn_start)

        btn_gfx = UIButton(text="GRAPHICS & DISPLAY CONFIG", x=30, y=135, w=420, h=54, variant="accent", font_size=16, on_click=on_settings)
        menu_card.add_child(btn_gfx)

        btn_quit = UIButton(text="EXIT ENGINE", x=30, y=205, w=420, h=54, variant="danger", font_size=16, on_click=on_exit)
        menu_card.add_child(btn_quit)

        hint = UILabel("Press ESC during simulation to re-enter menu.", x=30, y=410, w=420, h=24, font_size=12, color=(0.55, 0.65, 0.75, 0.6))
        menu_card.add_child(hint)

        # Right Column: Overview Card (x=590, y=190, w=610, h=470)
        info_card = _create_glass_card(x=590, y=190, w=610, h=470, border_accent=(0.95, 0.70, 0.20, 0.35))
        bg.add_child(info_card)

        info_title = UILabel("DEMO HIGHLIGHTS", x=30, y=20, w=550, h=28, font_size=14, bold=True, color=(1.0, 0.8, 0.25, 1.0))
        info_card.add_child(info_title)

        bullets = [
            "• Live ModernGL Rendering Pipeline Reconfiguration",
            "• Staged Settings Architecture: Requires Click on [APPLY CHANGES]",
            "• Runtime Display Switching (Windowed <-> Borderless)",
            "• VSync Toggle (WGL / SDL2 synchronization)",
            "• Master Quality Presets (Low, Medium, High, Ultra, Cinematic)",
            "• Granular Toggles: GTAO / SSAO, Bloom, SSR, Fog, AA (FXAA/TAA)",
            "• Character Controller + Dynamic Rapier3D Physics Cubes",
            "• Zero Debug Menu Dependency: Centralized Native Engine UI",
        ]
        for i, b in enumerate(bullets):
            lbl_b = UILabel(b, x=30, y=65 + i * 36, w=550, h=24, font_size=13, color=(0.85, 0.90, 0.95, 0.9))
            info_card.add_child(lbl_b)


# ---------------- Player & Physics World Module ----------------

class WorldAndPlayerModule(ProjectModule):
    """Manages scene entities, kinematic character motor, Rapier dynamic cubes, and camera."""

    name = "WorldAndPlayerModule"

    def __init__(self) -> None:
        self.motor: CharacterMotor | None = None
        self.camera: FollowCamera | None = None
        self.player_id: int = -1
        self.hud: TelemetryHUD | None = None
        self._jump_requested = False
        self._move_forward = 0.0
        self._move_strafe = 0.0
        self._is_sprinting = False
        self._fps_accumulator = 0.0
        self._fps_count = 0
        self._current_fps = 60.0

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

        # 1. Spawn Fixed Ground Platform with cracked_dry_mud POM
        app.set_world_builder(DefaultWorldBuilder(size=80.0, uv_tiles=40.0, material_name="cracked_dry_mud"))

        # 2. Spawn Static Curbs / Steps
        self._first_cube_id = -1
        for height, z_pos in [(0.15, 6.0), (0.25, 9.0), (0.35, 12.0)]:
            step_id = app.ecs.create_entity(
                position=(0.0, height * 0.5, z_pos),
                scale=(5.0, height, 2.5),
                color=(0.25, 0.45, 0.65, 1.0),
                roughness=0.4,
                metallic=0.3,
                is_static=True,
            )
            if self._first_cube_id == -1:
                self._first_cube_id = step_id
            app.physics.create_body(step_id, body_type="fixed", position=(0.0, height * 0.5, z_pos))
            app.physics.attach_box_collider(step_id, half_x=2.5, half_y=height * 0.5, half_z=1.25)

        # 3. Spawn Dynamic Physics Cubes with vibrant colors and metallic properties
        cube_colors = [
            (0.95, 0.25, 0.25, 1.0),
            (0.20, 0.85, 0.35, 1.0),
            (0.95, 0.75, 0.15, 1.0),
            (0.65, 0.25, 0.95, 1.0),
            (0.15, 0.80, 0.95, 1.0),
            (0.95, 0.45, 0.15, 1.0),
        ]
        for i in range(10):
            px = -7.5 + (i % 5) * 3.0
            pz = -6.0 + (i // 5) * 4.0
            py = 1.0 + (i * 0.5)
            c_color = cube_colors[i % len(cube_colors)]
            cube_id = app.ecs.create_entity(
                position=(px, py, pz),
                scale=(1.2, 1.2, 1.2),
                color=c_color,
                roughness=0.25,
                metallic=0.6,
                is_static=False,
            )
            if self._first_cube_id == -1:
                self._first_cube_id = cube_id
            app.physics.create_body(cube_id, body_type="dynamic", position=(px, py, pz))
            app.physics.attach_box_collider(cube_id, half_x=0.6, half_y=0.6, half_z=0.6, density=2.0)

        # 4. Spawn Player Entity and Kinematic Character Controller
        # Primitive capsule already has 0.8m diameter and 1.8m height; scale (1,1,1) preserves perfect proportions
        self.player_id = app.ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.15, 0.95, 0.85, 1.0),
            roughness=0.2,
            metallic=0.7,
            is_static=False,
        )
        cfg = CharacterMotorConfig(
            walk_speed=6.0,
            run_speed=11.5,
            jump_force=8.5,
            step_height=0.35,
            snap_to_ground=0.25,
        )
        self.motor = CharacterMotor(self.player_id, app.physics, app.ecs, config=cfg, initial_position=(0.0, 1.0, 0.0))

        # 5. Follow Camera closer to player with AAA framing
        cam_cfg = FollowCameraConfig(distance=3.8, target_offset_y=0.65)
        self.camera = FollowCamera(config=cam_cfg, initial_yaw_deg=45.0, initial_pitch_deg=18.0)
        self.camera.update_follow(0.016, (0.0, 1.0, 0.0), physics=app.physics, exclude_entity_id=self.player_id)
        app.camera_manager.register_camera("player_cam", self.camera, make_active=True)

        # 6. Telemetry HUD
        self.hud = TelemetryHUD()

    def get_draw_batches(self, app: ProjectApp) -> list[tuple]:
        alloc_cube = app.pipeline.mega_buffer.allocations["cube"]
        alloc_capsule = app.pipeline.mega_buffer.allocations["capsule"]
        d_cube = app.ecs.pool.get_dense_index(self._first_cube_id)
        d_capsule = app.ecs.pool.get_dense_index(self.player_id)
        # 3 curbs + 10 cubes = 13 cubes
        return [
            (alloc_cube, 13, d_cube, False, True),
            (alloc_capsule, 1, d_capsule, False, True),
        ]

    def poll_inputs(self, app: ProjectApp, dt: float) -> None:
        """Polls dual keyboard and gamepad input with zero runtime allocations."""
        # 1. Keyboard Movement
        keys = pygame.key.get_pressed()
        fwd = (1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0) - (1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0)
        strafe = (1.0 if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) else 0.0) - (1.0 if (keys[pygame.K_a] or keys[pygame.K_LEFT]) else 0.0)
        sprint_key = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        # 2. Gamepad Left Analog Stick & Face Buttons
        pad_move = app.input_manager.get_vector2("move")
        if abs(pad_move[0]) > 0.05 or abs(pad_move[1]) > 0.05:
            strafe = pad_move[0]
            fwd = pad_move[1]

        pad_sprint = (
            app.input_manager.is_action_down("sprint")
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_STICK)
            or app.input_manager.is_gamepad_down(GamepadButton.LEFT_BUMPER)
            or app.input_manager.is_gamepad_down(GamepadButton.RIGHT_TRIGGER)
        )
        if app.input_manager.is_action_pressed("jump") or app.input_manager.is_gamepad_pressed(GamepadButton.A):
            self._jump_requested = True

        self._move_forward = fwd
        self._move_strafe = strafe
        self._is_sprinting = sprint_key or pad_sprint

        # 3. Gamepad Right Analog Stick Camera Look
        look_yaw = app.input_manager.get_axis("look_yaw")
        look_pitch = app.input_manager.get_axis("look_pitch")
        if (abs(look_yaw) > 0.02 or abs(look_pitch) > 0.02) and self.camera is not None:
            self.camera.yaw_deg += look_yaw * 120.0 * dt
            self.camera.pitch_deg = max(
                self.camera.config.min_pitch_deg,
                min(self.camera.config.max_pitch_deg, self.camera.pitch_deg - look_pitch * 90.0 * dt),
            )
            self.camera.mark_dirty()

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("Simulation"):
            return

        # Continuous movement input vector: (forward, strafe)
        move_input = (self._move_forward, self._move_strafe)
        self.motor.update(
            dt=dt,
            move_input=move_input,
            is_running=self._is_sprinting,
            jump_requested=self._jump_requested,
            camera_yaw_deg=self.camera.yaw_deg,
        )
        self._jump_requested = False

    def on_update(self, app: ProjectApp, dt: float) -> None:
        if self.motor is None or self.camera is None or not app.state_manager.is_in_state("Simulation"):
            return

        self.poll_inputs(app, dt)

        # Frame rate smoothing
        self._fps_accumulator += dt
        self._fps_count += 1
        if self._fps_accumulator >= 0.25:
            self._current_fps = self._fps_count / self._fps_accumulator
            self._fps_accumulator = 0.0
            self._fps_count = 0

        # Update follow camera tracking player position
        pos = self.motor.position
        self.camera.update_follow(dt, pos, physics=app.physics, exclude_entity_id=self.player_id)

        # Update HUD telemetry
        if self.hud is not None and app.ui.active_screen is self.hud:
            win = app.render_ctx.window
            cfg = app.pipeline.config
            preset_name = getattr(cfg, "preset_name", "Custom")
            self.hud.update_telemetry(
                fps=self._current_fps,
                dt_ms=dt * 1000.0,
                width=win.width,
                height=win.height,
                mode_name=win.mode.name,
                vsync_name=win.vsync.name,
                preset_name=preset_name,
                ao_mode=getattr(cfg, "ao_mode", "GTAO"),
                bloom=getattr(cfg, "bloom_enabled", True),
                ssr=getattr(cfg, "ssr_enabled", True),
                fog=getattr(cfg, "volumetric_fog_enabled", True),
                flare=getattr(cfg, "lens_flare_enabled", True),
                aa_mode=getattr(cfg, "aa_mode", "FXAA"),
                speed_mps=self.motor.state.horizontal_speed,
                is_grounded=self.motor.state.is_grounded,
                is_sprinting=self._is_sprinting,
            )

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if self.camera is None or not app.state_manager.is_in_state("Simulation"):
            return False

        if event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = float(event.rel[0]), float(event.rel[1])
            self.camera.handle_mouse_orbit(rel_x, rel_y)
            return True

        elif event.type == pygame.MOUSEWHEEL:
            self.camera.handle_zoom(float(event.y))
            return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._jump_requested = True
                return True
            elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self._is_sprinting = True
                return True

        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self._is_sprinting = False
                return True

        return False

    def reset_player_position(self) -> None:
        if self.motor is not None:
            self.motor.teleport((0.0, 2.0, 0.0), reset_velocity=True)


# ---------------- Game States ----------------

class MainMenuState(GameState):
    """Initial game state rendering the fullscreen AAA main menu."""

    name = "MainMenu"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        ui = MainMenuUI(
            on_start=lambda: app.state_manager.set_state("Simulation"),
            on_settings=lambda: app.state_manager.push_state("Settings"),
            on_exit=lambda: app.stop(),
        )
        app.ui.set_screen(ui)

    def on_resume(self, app: ProjectApp) -> None:
        """Restores main menu screen when returning from settings menu."""
        ui = MainMenuUI(
            on_start=lambda: app.state_manager.set_state("Simulation"),
            on_settings=lambda: app.state_manager.push_state("Settings"),
            on_exit=lambda: app.stop(),
        )
        app.ui.set_screen(ui)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)


class SimulationGameState(GameState):
    """Active 3D simulation state running player kinematics and physics."""

    name = "Simulation"
    allow_fixed_update = True
    allow_variable_update = True
    show_cursor = False
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        mod = app.get_module(WorldAndPlayerModule)
        if mod and mod.hud:
            app.ui.set_screen(mod.hud)

    def on_resume(self, app: ProjectApp) -> None:
        """Restores telemetry HUD and asserts hardware mouse capture when unpausing."""
        app.input_manager.reset()
        mod = app.get_module(WorldAndPlayerModule)
        if mod:
            mod._move_forward = 0.0
            mod._move_strafe = 0.0
            mod._jump_requested = False
            if mod.hud:
                app.ui.set_screen(mod.hud)
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
        if app.input_manager.is_action_pressed("pause") or app.input_manager.is_gamepad_pressed(GamepadButton.START):
            app.state_manager.push_state("Paused")
            return
        mod = app.get_module(WorldAndPlayerModule)
        if mod:
            mod.poll_inputs(app, dt)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("Paused")
            return True
        mod = app.get_module(WorldAndPlayerModule)
        if mod:
            return mod.on_event(app, event)
        return False


class PausedState(GameState):
    """Modal pause state freezing simulation and offering graphics options."""

    name = "Paused"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        ui = PauseMenuUI(
            on_resume=lambda: app.state_manager.pop_state(),
            on_settings=lambda: app.state_manager.push_state("Settings"),
            on_quit_menu=lambda: app.state_manager.set_state("MainMenu"),
        )
        app.ui.set_screen(ui)

    def on_resume(self, app: ProjectApp) -> None:
        """Restores pause menu screen when returning from settings menu."""
        ui = PauseMenuUI(
            on_resume=lambda: app.state_manager.pop_state(),
            on_settings=lambda: app.state_manager.push_state("Settings"),
            on_quit_menu=lambda: app.state_manager.set_state("MainMenu"),
        )
        app.ui.set_screen(ui)

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


class SettingsState(GameState):
    """Full-screen graphics and display configuration menu with staged apply."""

    name = "Settings"
    allow_fixed_update = False
    allow_variable_update = True
    show_cursor = True
    grab_mouse = True

    def __init__(self) -> None:
        self.settings_ui: GraphicsSettingsUI | None = None

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        if self.settings_ui is None:
            self.settings_ui = GraphicsSettingsUI(
                app=app,
                on_return=lambda: app.state_manager.pop_state(),
            )
        self.settings_ui.sync_from_pipeline()
        app.ui.set_screen(self.settings_ui)

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.pop_state()
            return True
        return False


# ---------------- Application Entry Point ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Engine Example 03: Full Graphics & Display Configuration")
    parser.add_argument("--headless", action="store_true", help="Run without opening an OS display window")
    parser.add_argument("--frames", type=int, default=None, help="Maximum number of frames to run before auto-exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial Engine - Example 03: Full Graphics & Display Menu",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,  # Central sacred debug subsystem available via F1/F2/F3
    )

    app = ProjectApp(config=config)

    # 1. Attach World & Player Module
    app.add_module(WorldAndPlayerModule())

    # 2. Register Game States
    app.state_manager.register(MainMenuState())
    app.state_manager.register(SimulationGameState())
    app.state_manager.register(PausedState())
    app.state_manager.register(SettingsState())

    # Start in Main Menu (or exercise state transitions in headless test)
    app.state_manager.set_state("MainMenu")

    try:
        if args.headless and args.frames:
            step_count = max(1, args.frames // 3)
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.set_state("Simulation")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
            app.state_manager.push_state("Settings")
            for _ in range(step_count):
                app.step_frame(1.0 / 60.0)
        else:
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
