"""Native Dear ImGui Debug System for PyMordial Engine.

Provides independent, non-intrusive debug panels:
- F1: Performance & Resource Profiler (Multi-Style: Compact HUD -> Expanded Profiler -> Closed)
- F2: Graphics Pipeline & Renderer Tweaks
- F3: Gameplay Developer Tweaks & Inspector
- F9: Mouse Release/Capture for seamless panel interaction without locking gameplay
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import moderngl
import pygame
from imgui_bundle import imgui

try:
    from imgui_bundle.python_backends.pygame_backend import PygameRenderer
except (ImportError, ModuleNotFoundError):
    PygameRenderer = None
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.debug.game_tweaks import GameTweaks, TweakType
from engine.debug.monitor import SystemMonitor
from engine.debug.toast import DebugToast
from engine.events import WindowResizeEvent, subscribe_event, unsubscribe_event
from engine.gfx.quality_presets import GraphicsQuality
from engine.input.input_manager import InputManager
from engine.logging import log_warn

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


# Hi-Z bounds debug modes (see GBufferDebugMode). These read the shared
# u_HiZDebugMip uniform, so the mip selector applies to them too.
_HIZ_BOUNDS_MODES = frozenset(
    {
        GBufferDebugMode.HIZ_BOUNDS_NEAREST,
        GBufferDebugMode.HIZ_BOUNDS_FARTHEST,
        GBufferDebugMode.HIZ_BOUNDS_SPAN,
        GBufferDebugMode.HIZ_BOUNDS_MASK,
        GBufferDebugMode.HIZ_BOUNDS_INVALID,
        GBufferDebugMode.HIZ_BOUNDS_RAW,
    }
)

# Tracing backend choices. LEGACY is the pre-existing implementation; the
# alternative selects the shared Hi-Z tracer / explicit cascade model.
_SSGI_BACKENDS = ("LEGACY", "HIZ_BOUNDS")
_SSR_BACKENDS = ("LEGACY", "HIZ_BOUNDS")
_AO_BACKENDS = ("LEGACY", "HIZ_BOUNDS")
_RC_BACKENDS = ("LEGACY", "CASCADES")


def _backend_index(value: str, choices: tuple[str, ...]) -> int:
    """Returns the combo index for a backend name, defaulting to LEGACY."""
    try:
        return choices.index(str(value).upper())
    except ValueError:
        return 0


class DebugMenu:
    """Coordinates independent Dear ImGui debug panels for F1, F2, and F3."""

    __slots__ = (
        "ctx",
        "monitor",
        "engine_tweaks",
        "game_tweaks",
        "toast",
        "input_mgr",
        "width",
        "height",
        "f1_style",
        "show_graphics",
        "show_game_tweaks",
        "imgui_ctx",
        "io",
        "renderer",
        "_warned_no_renderer",
        "config_filepath",
        "_custom_ui_callbacks",
        "_scroll_to_microgeom",
        "_scroll_to_culling",
        "app",
        "settings_export_path",
    )

    TABS = ["F1: Performance", "F2: Graphics Options", "F3: Game Options"]

    def __init__(
        self,
        ctx: moderngl.Context | None,
        monitor: SystemMonitor,
        engine_tweaks: EngineTweaks,
        game_tweaks: GameTweaks,
        toast: DebugToast,
        input_mgr: InputManager,
        screen_width: int = 1280,
        screen_height: int = 720,
        config_filepath: str | Path | None = None,
        app: Any | None = None,
        settings_export_path: str | Path | None = None,
    ) -> None:
        self.ctx = ctx
        self.monitor = monitor
        self.engine_tweaks = engine_tweaks
        self.game_tweaks = game_tweaks
        self.toast = toast
        self.input_mgr = input_mgr
        self.config_filepath = Path(config_filepath) if config_filepath is not None else None
        self.app = app
        # Optional project-declared INI export target. When set, the F2 panel
        # offers a "Save Settings" action that writes the live graphics config.
        self.settings_export_path = (
            Path(settings_export_path) if settings_export_path is not None else None
        )

        self.width = screen_width
        self.height = screen_height

        # Independent panel states
        # f1_style: 0 = Off, 1 = Basic HUD overlay, 2 = Expanded Profiler window
        self.f1_style: int = 0
        self.show_graphics: bool = False
        self.show_game_tweaks: bool = False
        self._custom_ui_callbacks: list[Callable[[], None]] = []
        self._scroll_to_microgeom: bool = False
        self._scroll_to_culling: bool = False

        # Initialize Dear ImGui context
        self.imgui_ctx = imgui.create_context()
        self.io = imgui.get_io()
        self.io.display_size = imgui.ImVec2(float(screen_width), float(screen_height))
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value

        # Initialize Pygame OpenGL backend if display surface and context are available
        self.renderer: PygameRenderer | None = None
        self._warned_no_renderer: bool = False
        if self.ctx is not None and pygame.display.get_surface() is not None:
            if PygameRenderer is not None:
                try:
                    self.renderer = PygameRenderer()
                except Exception as e:
                    log_warn("DebugMenu", f"PygameRenderer backend failed to initialize: {e}")
                    self.renderer = None
            else:
                log_warn(
                    "DebugMenu",
                    "PygameRenderer backend unavailable (PyOpenGL missing). "
                    "Install 'PyOpenGL' to enable Dear ImGui debug overlays.",
                )

        self._apply_theme()

        # Subscribe to runtime window resize events to keep ImGui viewport synchronized
        subscribe_event(WindowResizeEvent, self._on_window_resize, priority=80)

    def _apply_theme(self) -> None:
        """Applies a modern, sleek AAA dark theme with translucent slate & cyan accents."""
        style = imgui.get_style()
        style.display_window_padding = imgui.ImVec2(10.0, 10.0)
        style.display_safe_area_padding = imgui.ImVec2(10.0, 10.0)
        style.window_rounding = 8.0
        style.frame_rounding = 5.0
        style.grab_rounding = 4.0
        style.popup_rounding = 6.0
        style.scrollbar_rounding = 6.0
        style.frame_padding = imgui.ImVec2(8.0, 5.0)
        style.item_spacing = imgui.ImVec2(8.0, 6.0)

        style.set_color_(imgui.Col_.window_bg.value, imgui.ImVec4(0.06, 0.09, 0.14, 0.90))
        style.set_color_(imgui.Col_.border.value, imgui.ImVec4(0.22, 0.35, 0.48, 0.60))
        style.set_color_(imgui.Col_.title_bg.value, imgui.ImVec4(0.08, 0.13, 0.20, 0.95))
        style.set_color_(imgui.Col_.title_bg_active.value, imgui.ImVec4(0.12, 0.23, 0.40, 1.0))
        style.set_color_(imgui.Col_.header.value, imgui.ImVec4(0.12, 0.23, 0.38, 0.70))
        style.set_color_(imgui.Col_.header_hovered.value, imgui.ImVec4(0.18, 0.32, 0.52, 0.85))
        style.set_color_(imgui.Col_.header_active.value, imgui.ImVec4(0.22, 0.40, 0.65, 1.0))
        style.set_color_(imgui.Col_.button.value, imgui.ImVec4(0.12, 0.25, 0.42, 0.80))
        style.set_color_(imgui.Col_.button_hovered.value, imgui.ImVec4(0.20, 0.42, 0.70, 0.95))
        style.set_color_(imgui.Col_.button_active.value, imgui.ImVec4(0.22, 0.55, 0.92, 1.0))
        style.set_color_(imgui.Col_.frame_bg.value, imgui.ImVec4(0.10, 0.15, 0.22, 0.70))
        style.set_color_(imgui.Col_.frame_bg_hovered.value, imgui.ImVec4(0.15, 0.22, 0.32, 0.85))
        style.set_color_(imgui.Col_.frame_bg_active.value, imgui.ImVec4(0.20, 0.30, 0.45, 1.0))
        style.set_color_(imgui.Col_.slider_grab.value, imgui.ImVec4(0.22, 0.65, 0.95, 0.90))
        style.set_color_(imgui.Col_.slider_grab_active.value, imgui.ImVec4(0.35, 0.75, 1.0, 1.0))
        style.set_color_(imgui.Col_.check_mark.value, imgui.ImVec4(0.35, 0.85, 0.55, 1.0))

    # --------------------------------------------------------------------------
    # Panel Toggling & Multi-Style Lifecycle
    # --------------------------------------------------------------------------

    def cycle_f1(self) -> int:
        """Cycles F1 performance panel: 0 (Off) -> 1 (Basic HUD) -> 2 (Expanded Profiler) -> 0."""
        self.f1_style = (self.f1_style + 1) % 3
        return self.f1_style

    def toggle_graphics(self) -> bool:
        """Toggles independent F2 Graphics Options panel."""
        self.show_graphics = not self.show_graphics
        return self.show_graphics

    def toggle_game_tweaks(self) -> bool:
        """Toggles independent F3 Gameplay Developer Tweaks panel."""
        self.show_game_tweaks = not self.show_game_tweaks
        return self.show_game_tweaks

    @property
    def visible(self) -> bool:
        """Returns True if any debug panel or custom UI callback is currently active."""
        return self.f1_style > 0 or self.show_graphics or self.show_game_tweaks or len(self._custom_ui_callbacks) > 0

    @property
    def blocks_game_input(self) -> bool:
        """Whether interactive debug panels should temporarily release gameplay input."""
        return self.show_graphics or self.show_game_tweaks or bool(self._custom_ui_callbacks)

    def add_custom_ui(self, callback: Callable[[], None]) -> None:
        """Registers a custom UI rendering callback into the ImGui frame loop."""
        if callback not in self._custom_ui_callbacks:
            self._custom_ui_callbacks.append(callback)

    def remove_custom_ui(self, callback: Callable[[], None]) -> None:
        """Removes a registered custom UI rendering callback."""
        if callback in self._custom_ui_callbacks:
            self._custom_ui_callbacks.remove(callback)

    @visible.setter
    def visible(self, val: bool) -> None:
        if not val:
            self.f1_style = 0
            self.show_graphics = False
            self.show_game_tweaks = False
        else:
            if self.f1_style == 0 and not self.show_graphics and not self.show_game_tweaks:
                self.f1_style = 1

    @property
    def active_tab(self) -> int:
        """Compatibility property for legacy 3-tab queries."""
        if self.show_game_tweaks:
            return 2
        if self.show_graphics:
            return 1
        return 0

    @active_tab.setter
    def active_tab(self, tab_idx: int) -> None:
        """Compatibility setter for legacy 3-tab queries."""
        if tab_idx == 0:
            self.f1_style = 1 if self.f1_style == 0 else self.f1_style
        elif tab_idx == 1:
            self.show_graphics = True
        elif tab_idx == 2:
            self.show_game_tweaks = True

    def toggle(self) -> bool:
        """Toggles debug system visibility."""
        if self.visible:
            self.visible = False
            return False
        else:
            self.f1_style = 1
            return True

    def toggle_tab(self, tab_idx: int) -> bool:
        """Legacy tab toggle compatibility:
        - F1 (0): cycles basic -> expanded -> off
        - F2 (1): toggles graphics on/off
        - F3 (2): toggles game tweaks on/off
        """
        if tab_idx == 0:
            new_style = self.cycle_f1()
            return new_style > 0
        elif tab_idx == 1:
            return self.toggle_graphics()
        elif tab_idx == 2:
            return self.toggle_game_tweaks()
        return False

    def handle_input(self) -> None:
        """No-op: Debug menu uses mouse interaction via F9; keyboard is never hijacked."""
        pass

    def process_event(self, event: pygame.event.Event) -> bool:
        """Forwards Pygame events to ImGui (mouse clicks, motion, wheel when mouse is free)."""
        # Intercept window resize events to prevent PygameRenderer from destroying the OpenGL context
        if event.type in (
            pygame.VIDEORESIZE,
            getattr(pygame, "WINDOWRESIZED", -1),
            getattr(pygame, "WINDOWSIZECHANGED", -1),
        ):
            w = getattr(event, "w", getattr(event, "x", self.width))
            h = getattr(event, "h", getattr(event, "y", self.height))
            if w > 0 and h > 0:
                self.resize(w, h)
            return True

        if self.renderer is not None:
            if self.imgui_ctx is not None and imgui is not None:
                imgui.set_current_context(self.imgui_ctx)
            return bool(self.renderer.process_event(event))
        return False

    # --------------------------------------------------------------------------
    # ImGui Panel Renderers
    # --------------------------------------------------------------------------

    def _render_f1_basic_hud(self) -> None:
        """Renders minimal, unobtrusive HUD overlay showing FPS, Frame Time, and core indicators."""
        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.always.value)
        imgui.set_next_window_bg_alpha(0.70)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_inputs.value
        )
        imgui.begin("##F1_BasicHUD", flags=flags)

        fps = self.monitor.fps
        if fps >= 55.0:
            fps_col = imgui.ImVec4(0.29, 0.87, 0.50, 1.0)  # Green
        elif fps >= 30.0:
            fps_col = imgui.ImVec4(0.98, 0.80, 0.18, 1.0)  # Yellow
        else:
            fps_col = imgui.ImVec4(0.94, 0.27, 0.27, 1.0)  # Red

        imgui.text_colored(fps_col, f"FPS: {fps:5.1f}")
        imgui.same_line()
        imgui.text(f"|  {self.monitor.avg_frame_time_ms:4.1f} ms")

        # Core indicators: 1% Low FPS & Active Entities
        imgui.text_disabled(f"1% Low: {self.monitor.one_percent_low_fps:4.1f} FPS")

        stats = self.monitor.runtime_stats
        active_ent = stats["active_entities"]
        imgui.same_line()
        imgui.text_disabled(f"|  Visible: {stats['visible_entities']}/{active_ent}")
        gpu_ms = self.monitor.gpu_frame_time_ms
        imgui.text_disabled(
            f"CPU Render: {self.monitor.stage_times_us.get('Render Pipeline CPU', 0.0) / 1000.0:.2f} ms"
            + (f"  |  GPU Render: {gpu_ms:.2f} ms" if gpu_ms is not None else "  |  GPU timing pending")
        )

        imgui.separator()
        imgui.text_disabled("[F1] Cycle performance overlay  |  [F2] Engine controls")
        imgui.end()

    def _render_f1_expanded_profiler(self) -> None:
        """Renders performance, scene, renderer, and resource diagnostics."""
        imgui.set_next_window_size(imgui.ImVec2(620.0, 720.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_pos(imgui.ImVec2(20.0, 20.0), imgui.Cond_.first_use_ever.value)
        flags = imgui.WindowFlags_.no_inputs.value | imgui.WindowFlags_.no_focus_on_appearing.value
        expanded, p_open = imgui.begin("Performance & Resource Monitor [F1]", p_open=True, flags=flags)
        if not p_open:
            self.f1_style = 0

        if expanded:
            m = self.monitor
            fps = m.fps
            fps_col = (
                imgui.ImVec4(0.29, 0.87, 0.50, 1.0)
                if fps >= 55.0
                else (imgui.ImVec4(0.98, 0.80, 0.18, 1.0) if fps >= 30.0 else imgui.ImVec4(0.94, 0.27, 0.27, 1.0))
            )
            imgui.text("FPS:")
            imgui.same_line()
            imgui.text_colored(fps_col, f"{fps:5.1f}")
            imgui.same_line()
            imgui.text(
                f"Avg: {m.avg_fps:5.1f}  |  1%: {m.one_percent_low_fps:5.1f}  |  0.1%: {m.point_one_percent_low_fps:5.1f}"
            )

            gpu_ms = m.gpu_frame_time_ms
            gpu_text = (
                f"{gpu_ms:.2f} ms"
                if gpu_ms is not None
                else ("unavailable" if m.gpu_timing_supported is False else "warming up")
            )
            imgui.text(
                f"Frame: {m.avg_frame_time_ms:.2f} ms avg  |  P95: {m.p95_frame_time_ms:.2f}  "
                f"P99: {m.p99_frame_time_ms:.2f}  |  Budget: {m.frame_budget_ms:.2f} ms"
            )
            imgui.text(
                f"Over budget: {m.over_budget_percent:.1f}%  |  Hitches >= 50 ms: {m.stutter_count} "
                f"|  Min/Max: {m.min_frame_time_ms:.2f}/{m.max_frame_time_ms:.2f} ms"
            )
            imgui.text(
                f"CPU render submission: {m.stage_times_us.get('Render Pipeline CPU', 0.0) / 1000.0:.2f} ms"
                f"  |  GPU render (async): {gpu_text}"
            )

            imgui.separator()
            imgui.text("Frame Time History (ms)")
            ft_history = m.frame_history_view()
            if len(ft_history) > 0:
                overlay_str = f"{m.avg_frame_time_ms:.2f} ms avg | {m.frame_budget_ms:.2f} ms budget"
                imgui.plot_lines(
                    "##FT_Plot",
                    ft_history,
                    values_offset=0,
                    overlay_text=overlay_str,
                    scale_min=0.0,
                    scale_max=max(33.3, min(100.0, m.max_frame_time_ms)),
                    graph_size=imgui.ImVec2(580.0, 75.0),
                )

            imgui.separator()
            imgui.text("CPU Stage Timings (last / smoothed average / lifetime peak)")
            if imgui.begin_table(
                "Subsystems", 4, imgui.TableFlags_.borders_inner_h.value | imgui.TableFlags_.sizing_stretch_prop.value
            ):
                imgui.table_setup_column("Stage")
                imgui.table_setup_column("Last ms")
                imgui.table_setup_column("Average ms")
                imgui.table_setup_column("Peak ms")
                imgui.table_headers_row()

                for stage, us in m.stage_averages_us.items():
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(stage)
                    imgui.table_next_column()
                    last_ms = m.stage_times_us.get(stage, 0.0) / 1000.0
                    imgui.text(f"{last_ms:.2f}")
                    imgui.table_next_column()
                    ms_val = us / 1000.0
                    col = (
                        imgui.ImVec4(0.29, 0.87, 0.50, 1.0)
                        if ms_val <= 4.0
                        else (
                            imgui.ImVec4(0.98, 0.80, 0.18, 1.0)
                            if ms_val <= 8.0
                            else imgui.ImVec4(0.94, 0.27, 0.27, 1.0)
                        )
                    )
                    imgui.text_colored(col, f"{ms_val:.2f}")
                    imgui.table_next_column()
                    imgui.text(f"{m.stage_max_us.get(stage, 0.0) / 1000.0:.2f}")
                imgui.end_table()

            imgui.separator()
            imgui.text("GPU Pass Timings (async, ms)")
            profiler = getattr(getattr(self.app, "pipeline", None), "gpu_profiler", None)
            if profiler is None or not profiler.enabled:
                imgui.text_disabled("GPU pass timing unavailable")
            elif not profiler.results():
                imgui.text_disabled("GPU pass timing warming up")
            else:
                if imgui.begin_table(
                    "GpuPasses",
                    2,
                    imgui.TableFlags_.borders_inner_h.value | imgui.TableFlags_.sizing_stretch_prop.value,
                ):
                    imgui.table_setup_column("Region")
                    imgui.table_setup_column("GPU ms")
                    imgui.table_headers_row()
                    for region, ms in profiler.results().items():
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(region)
                        imgui.table_next_column()
                        col = (
                            imgui.ImVec4(0.29, 0.87, 0.50, 1.0)
                            if ms <= 2.0
                            else (imgui.ImVec4(0.98, 0.80, 0.18, 1.0) if ms <= 5.0 else imgui.ImVec4(0.94, 0.27, 0.27, 1.0))
                        )
                        imgui.text_colored(col, f"{ms:.3f}")
                    imgui.end_table()

            imgui.separator()
            stats = m.runtime_stats
            imgui.text(
                f"Scene: {stats['active_entities']:,}/{stats['entity_capacity']:,} entities  |  "
                f"Visible: {stats['visible_entities']:,}  |  Culled: {stats['culled_entities']:,}"
            )
            imgui.text(
                f"Draw batches: {stats['draw_batches']:,}  |  Commands: {stats['draw_commands']:,}  |  "
                f"Triangles submitted (estimate): {stats['triangles']:,}"
            )
            imgui.text(
                f"Lights: {stats['point_lights']} point / {stats['spot_lights']} spot  |  "
                f"Meshes: {stats['mesh_vertices']:,} verts / {stats['mesh_indices']:,} indices "
                f"({stats['mesh_mib']:.1f} MiB)"
            )
            imgui.text(
                f"View: {stats['render_width']}x{stats['render_height']}  |  "
                f"Materials: {stats['material_layers']}/{stats['material_capacity']} layers @ "
                f"{stats['material_resolution']}px ({stats['material_mib']:.0f} MiB allocated est.)"
            )
            imgui.text(
                f"Tracked GPU allocation estimate: {stats['tracked_gpu_mib']:.0f} MiB "
                "(partial; excludes driver and untracked pass resources)"
            )
            imgui.text(
                f"Graphics: {stats['quality']}  |  GI: {stats['gi_mode']}  |  "
                f"Shadows: {stats['shadow_resolution']}px, {stats['shadow_cascades']} cascades"
            )
            imgui.text_disabled("GPU time is asynchronous and may lag several frames. Peaks are lifetime maxima.")

            imgui.text_disabled("[F1] Cycle overlay  |  F2: Engine controls  |  F3: Game tweaks")
        imgui.end()

    def _save_settings_ini(self) -> None:
        """Writes the live graphics configuration to the project's INI file."""
        from engine.config.ini_export import write_ini

        if self.settings_export_path is None:
            return

        header = (
            "PyMordial Engine - user settings\n"
            "Generated by the in-game debug menu (F2 -> Save Settings).\n"
            "Edit freely; unknown keys are ignored on load."
        )
        if write_ini(self.settings_export_path, self.engine_tweaks.to_dict(), header=header):
            print(f"[INFO ] [DebugMenu] Saved settings to {self.settings_export_path}")
            self.toast.show(f"Settings saved to {self.settings_export_path.name}", duration=2.5, color=(56, 189, 248))
        else:
            self.toast.show("Failed to save settings", duration=2.5, color=(248, 113, 113))

    def _render_f2_graphics(self) -> None:
        """Renders independent F2 Graphics Pipeline & Renderer tweaks panel."""
        imgui.set_next_window_size(imgui.ImVec2(440.0, 480.0), imgui.Cond_.first_use_ever.value)
        f2_x = 20.0 + (500.0 if self.width >= 1600 else 300.0)
        imgui.set_next_window_pos(imgui.ImVec2(f2_x, 20.0), imgui.Cond_.first_use_ever.value)
        expanded, p_open = imgui.begin("Engine Controls & Diagnostics [F2]", p_open=True)
        if not p_open:
            self.show_graphics = False

        if expanded:
            et = self.engine_tweaks

            # Configuration Persistence
            if imgui.button("Save Configuration", imgui.ImVec2(160.0, 24.0)):
                if self.config_filepath is None:
                    self.toast.show("No configuration file path is available", duration=2.5, color=(250, 204, 21))
                elif self.engine_tweaks.save_to_file(self.config_filepath):
                    print(f"[INFO ] [DebugMenu] Saved configuration to {self.config_filepath}")
                    self.toast.show("Configuration saved to game directory", duration=2.5, color=(56, 189, 248))
                else:
                    self.toast.show("Failed to save configuration", duration=2.5, color=(248, 113, 113))

            # Project-declared INI export. Only shown when the project opted in
            # by setting ProjectConfig.settings_export_path.
            if self.settings_export_path is not None:
                imgui.same_line()
                if imgui.button("Save Settings (.ini)", imgui.ImVec2(160.0, 24.0)):
                    self._save_settings_ini()

            imgui.same_line()
            if imgui.button("Reset Defaults", imgui.ImVec2(130.0, 24.0)):
                self.engine_tweaks.reset_to_defaults()
                if self.config_filepath is None:
                    self.toast.show("Defaults restored (not saved)", duration=2.5, color=(250, 204, 21))
                elif self.engine_tweaks.save_to_file(self.config_filepath):
                    self.toast.show("Defaults restored and saved", duration=2.5, color=(250, 204, 21))
                else:
                    self.toast.show("Defaults restored; save failed", duration=2.5, color=(248, 113, 113))

            imgui.separator()

            if imgui.collapsing_header("Rendering: Quality & Diagnostics", imgui.TreeNodeFlags_.default_open.value):
                # 1. Quality Preset
                preset_values = [
                    GraphicsQuality.LOW,
                    GraphicsQuality.MEDIUM,
                    GraphicsQuality.HIGH,
                    GraphicsQuality.ULTRA,
                    GraphicsQuality.CUSTOM,
                ]
                preset_labels = ["LOW", "MID", "HIGH", "ULTRA", "CUSTOM"]
                current_preset = et.quality_preset
                current_idx = (
                    preset_values.index(current_preset)
                    if current_preset in preset_values
                    else preset_values.index(GraphicsQuality.CUSTOM)
                )
                changed, new_idx = imgui.combo("Quality Preset", current_idx, preset_labels)
                if changed and new_idx != current_idx:
                    et.set_quality_preset(preset_values[new_idx])
                    self.toast.show(f"Quality Preset: {preset_labels[new_idx]}", duration=2.0)

                # VSync Toggle
                vsync_changed, vsync_val = imgui.checkbox("VSync (Vertical Sync)", et.vsync_enabled)
                if vsync_changed:
                    et.set_vsync(vsync_val)
                    self.toast.show(f"VSync: {'ON' if vsync_val else 'OFF'}", duration=1.5)

                # 2. Tonemapper
                tonemap_options = ["ACES", "AgX", "Reinhard"]
                t_idx = tonemap_options.index(et.tonemap_mode) if et.tonemap_mode in tonemap_options else 0
                t_changed, new_t_idx = imgui.combo("Tonemapper", t_idx, tonemap_options)
                if t_changed and new_t_idx != t_idx:
                    et.tonemap_mode = tonemap_options[new_t_idx]
                    et.mark_custom()
                    self.toast.show(f"Tonemapper: {et.tonemap_mode}", duration=2.0)

                exposure_changed, exposure_value = imgui.slider_float("Exposure", et.exposure, 0.1, 4.0, "%.2f")
                if exposure_changed:
                    et.exposure = exposure_value
                    et.mark_custom()

                # 3. G-Buffer Debug Mode
                gbuf_names = [m.name for m in GBufferDebugMode]
                gbuf_idx = gbuf_names.index(et.gbuffer_debug.name) if et.gbuffer_debug.name in gbuf_names else 0
                g_changed, new_g_idx = imgui.combo("G-Buffer Debug", gbuf_idx, gbuf_names)
                if g_changed and new_g_idx != gbuf_idx:
                    et.gbuffer_debug = GBufferDebugMode[gbuf_names[new_g_idx]]
                    self.toast.show(f"G-Buffer Mode: {et.gbuffer_debug.name}", duration=2.0)

                if et.gbuffer_debug == GBufferDebugMode.HIZ:
                    hiz_changed, new_hiz_mip = imgui.slider_int("Hi-Z Mip Level", et.hiz_debug_mip, 0, 10)
                    if hiz_changed:
                        et.hiz_debug_mip = new_hiz_mip
                elif et.gbuffer_debug in _HIZ_BOUNDS_MODES:
                    # The bounds hierarchy debug views read the same mip uniform.
                    hiz_changed, new_hiz_mip = imgui.slider_int("Hi-Z Bounds Mip Level", et.hiz_debug_mip, 0, 10)
                    if hiz_changed:
                        et.hiz_debug_mip = new_hiz_mip

            if imgui.collapsing_header(
                "Lighting: Global Illumination & Occlusion", imgui.TreeNodeFlags_.default_open.value
            ):
                # GI Mode (OFF, IBL, SSGI, RADIANCE_CASCADES)
                gi_modes = ["OFF", "IBL", "SSGI", "RADIANCE_CASCADES"]
                curr_gi_idx = gi_modes.index(et.gi_mode) if et.gi_mode in gi_modes else 3
                gi_changed, new_gi_idx = imgui.combo("GI Mode", curr_gi_idx, gi_modes)
                if gi_changed and new_gi_idx != curr_gi_idx:
                    et.gi_mode = gi_modes[new_gi_idx]
                    et.mark_custom()
                    self.toast.show(f"GI Mode: {et.gi_mode}", duration=2.0)

                if et.gi_mode == "SSGI":
                    ssgi_backend_changed, ssgi_backend_idx = imgui.combo(
                        "SSGI Backend", _backend_index(et.ssgi_backend, _SSGI_BACKENDS), _SSGI_BACKENDS
                    )
                    if ssgi_backend_changed:
                        et.ssgi_backend = _SSGI_BACKENDS[ssgi_backend_idx]
                        et.mark_custom()
                        self.toast.show(f"SSGI Backend: {et.ssgi_backend}", duration=2.0)

                    ssgi_st_changed, ssgi_st_val = imgui.slider_int("SSGI Steps", et.ssgi_steps, 4, 32)
                    if ssgi_st_changed:
                        et.ssgi_steps = ssgi_st_val
                        et.mark_custom()

                    ssgi_ray_changed, ssgi_ray_val = imgui.slider_int("SSGI Rays", getattr(et, "ssgi_rays", 8), 1, 16)
                    if ssgi_ray_changed:
                        et.ssgi_rays = ssgi_ray_val
                        et.mark_custom()

                    ssgi_dist_changed, ssgi_dist_val = imgui.slider_float(
                        "SSGI Distance", getattr(et, "ssgi_ray_distance", 3.0), 0.5, 8.0, "%.1fm"
                    )
                    if ssgi_dist_changed:
                        et.ssgi_ray_distance = ssgi_dist_val
                        et.mark_custom()

                    ssgi_thick_changed, ssgi_thick_val = imgui.slider_float(
                        "SSGI Thickness", getattr(et, "ssgi_thickness", 0.35), 0.05, 1.0, "%.2fm"
                    )
                    if ssgi_thick_changed:
                        et.ssgi_thickness = ssgi_thick_val
                        et.mark_custom()

                    ssgi_in_changed, ssgi_in_val = imgui.slider_float(
                        "SSGI Intensity", et.ssgi_intensity, 0.1, 3.0, "%.2f"
                    )
                    if ssgi_in_changed:
                        et.ssgi_intensity = ssgi_in_val
                        et.mark_custom()

                    ssgi_res_changed, ssgi_res_val = imgui.slider_float(
                        "SSGI Resolution Scale",
                        float(getattr(et, "ssgi_resolution_scale", 1.0)),
                        0.25,
                        1.0,
                        "%.2f",
                    )
                    if ssgi_res_changed:
                        et.ssgi_resolution_scale = ssgi_res_val
                        et.mark_custom()

                    ssgi_t_changed, ssgi_t_val = imgui.checkbox(
                        "SSGI Temporal Denoise", bool(getattr(et, "ssgi_temporal_enabled", True))
                    )
                    if ssgi_t_changed:
                        et.ssgi_temporal_enabled = ssgi_t_val
                        et.mark_custom()

                    if getattr(et, "ssgi_temporal_enabled", True):
                        ssgi_tb_changed, ssgi_tb_val = imgui.slider_float(
                            "SSGI Temporal Blend",
                            float(getattr(et, "ssgi_temporal_blend", 0.90)),
                            0.0,
                            0.95,
                            "%.2f",
                        )
                        if ssgi_tb_changed:
                            et.ssgi_temporal_blend = ssgi_tb_val
                            et.mark_custom()

                if et.gi_mode == "RADIANCE_CASCADES":
                    rc_backend_changed, rc_backend_idx = imgui.combo(
                        "RC Backend", _backend_index(et.rc_backend, _RC_BACKENDS), _RC_BACKENDS
                    )
                    if rc_backend_changed:
                        et.rc_backend = _RC_BACKENDS[rc_backend_idx]
                        et.mark_custom()
                        self.toast.show(f"RC Backend: {et.rc_backend}", duration=2.0)

                    rc_ray_count_changed, rc_ray_count = imgui.slider_int("RC Ray Count", et.rc_ray_count, 4, 32)
                    if rc_ray_count_changed:
                        et.rc_ray_count = rc_ray_count
                        et.mark_custom()

                    rc_merge_changed, rc_merge_val = imgui.slider_int(
                        "RC Cascade Isolation", int(getattr(et, "rc_merge_max_cascade", 2)), 0, 2
                    )
                    if rc_merge_changed:
                        et.rc_merge_max_cascade = rc_merge_val
                        et.mark_custom()

                    rc_in_changed, rc_in_val = imgui.slider_float(
                        "RC Intensity", getattr(et, "rc_intensity", 1.0), 0.1, 3.0, "%.2f"
                    )
                    if rc_in_changed:
                        et.rc_intensity = rc_in_val
                        et.mark_custom()

                    rc_c0_changed, rc_c0_val = imgui.slider_float(
                        "RC C0 Range", getattr(et, "rc_interval_c0", 0.40), 0.1, 1.0, "%.2fm"
                    )
                    if rc_c0_changed:
                        et.rc_interval_c0 = rc_c0_val
                        et.mark_custom()

                    rc_c1_changed, rc_c1_val = imgui.slider_float(
                        "RC C1 Range", getattr(et, "rc_interval_c1", 2.50), 1.0, 6.0, "%.1fm"
                    )
                    if rc_c1_changed:
                        et.rc_interval_c1 = rc_c1_val
                        et.mark_custom()

                    rc_c2_changed, rc_c2_val = imgui.slider_float("RC C2 Range", et.rc_interval_c2, 6.0, 100.0, "%.1fm")
                    if rc_c2_changed:
                        et.rc_interval_c2 = rc_c2_val
                        et.mark_custom()

                    rc_steps_c0_changed, rc_steps_c0 = imgui.slider_int("RC C0 Steps", et.rc_steps_c0, 1, 32)
                    if rc_steps_c0_changed:
                        et.rc_steps_c0 = rc_steps_c0
                        et.mark_custom()

                    rc_steps_c1_changed, rc_steps_c1 = imgui.slider_int("RC C1 Steps", et.rc_steps_c1, 1, 32)
                    if rc_steps_c1_changed:
                        et.rc_steps_c1 = rc_steps_c1
                        et.mark_custom()

                    rc_thickness_changed, rc_thickness = imgui.slider_float(
                        "RC Thickness", et.rc_thickness, 0.01, 1.0, "%.2fm"
                    )
                    if rc_thickness_changed:
                        et.rc_thickness = rc_thickness
                        et.mark_custom()

                    rc_blend_changed, rc_blend_val = imgui.slider_float(
                        "RC Temporal Blend", getattr(et, "rc_temporal_blend", 0.96), 0.0, 0.98, "%.2f"
                    )
                    if rc_blend_changed:
                        et.rc_temporal_blend = rc_blend_val
                        et.mark_custom()

                    rc_res_changed, rc_res_val = imgui.slider_float(
                        "RC Resolution Scale",
                        float(getattr(et, "rc_resolution_scale", 1.0)),
                        0.25,
                        1.0,
                        "%.2f",
                    )
                    if rc_res_changed:
                        et.rc_resolution_scale = rc_res_val
                        et.mark_custom()

                # AO Mode (OFF, SSAO, GTAO)
                ao_modes = ["OFF", "SSAO", "GTAO"]
                curr_ao_idx = ao_modes.index(et.ao_mode) if et.ao_mode in ao_modes else 3
                ao_changed, new_ao_idx = imgui.combo("AO Mode", curr_ao_idx, ao_modes)
                if ao_changed and new_ao_idx != curr_ao_idx:
                    et.ao_mode = ao_modes[new_ao_idx]
                    et.mark_custom()
                    self.toast.show(f"AO Mode: {et.ao_mode}", duration=2.0)

                if et.ao_mode != "OFF":
                    if et.ao_mode == "GTAO":
                        ao_backend_changed, ao_backend_idx = imgui.combo(
                            "AO Backend", _backend_index(et.ao_backend, _AO_BACKENDS), _AO_BACKENDS
                        )
                        if ao_backend_changed:
                            et.ao_backend = _AO_BACKENDS[ao_backend_idx]
                            et.mark_custom()
                            self.toast.show(f"AO Backend: {et.ao_backend}", duration=2.0)

                    ao_in_changed, ao_in_val = imgui.slider_float("AO Intensity", et.ao_intensity, 0.1, 3.0, "%.2f")
                    if ao_in_changed:
                        et.ao_intensity = ao_in_val
                        et.mark_custom()

                    ao_rad_changed, ao_rad_val = imgui.slider_float("AO Radius", et.ao_radius, 0.1, 3.0, "%.2f")
                    if ao_rad_changed:
                        et.ao_radius = ao_rad_val
                        et.mark_custom()

                    ao_res_changed, ao_res_val = imgui.slider_float(
                        "AO Resolution Scale", float(getattr(et, "ao_resolution_scale", 1.0)), 0.25, 1.0, "%.2f"
                    )
                    if ao_res_changed:
                        et.ao_resolution_scale = ao_res_val
                        et.mark_custom()

            if imgui.collapsing_header("Image Quality, Reflections & Local Lights"):
                # IBL
                ibl_changed, ibl_val = imgui.checkbox("Image-Based Lighting (IBL)", et.ibl_enabled)
                if ibl_changed:
                    et.ibl_enabled = ibl_val
                    et.mark_custom()
                    self.toast.show(f"IBL: {'ON' if ibl_val else 'OFF'}", duration=1.5)

                # SSR
                ssr_changed, ssr_val = imgui.checkbox("Screen-Space Reflections (SSR)", et.ssr_enabled)
                if ssr_changed:
                    et.ssr_enabled = ssr_val
                    et.mark_custom()
                    self.toast.show(f"SSR: {'ON' if ssr_val else 'OFF'}", duration=1.5)

                if et.ssr_enabled:
                    ssr_backend_changed, ssr_backend_idx = imgui.combo(
                        "SSR Backend", _backend_index(et.ssr_backend, _SSR_BACKENDS), _SSR_BACKENDS
                    )
                    if ssr_backend_changed:
                        et.ssr_backend = _SSR_BACKENDS[ssr_backend_idx]
                        et.mark_custom()
                        self.toast.show(f"SSR Backend: {et.ssr_backend}", duration=2.0)

                    ssr_st_changed, ssr_st_val = imgui.slider_int("SSR Steps", et.ssr_steps, 24, 128)
                    if ssr_st_changed:
                        et.ssr_steps = ssr_st_val
                        et.mark_custom()

                    ssr_tk_changed, ssr_tk_val = imgui.slider_float(
                        "SSR Thickness", et.ssr_thickness, 0.03, 0.65, "%.2fm"
                    )
                    if ssr_tk_changed:
                        et.ssr_thickness = ssr_tk_val
                        et.mark_custom()

                    ssr_mr_changed, ssr_mr_val = imgui.slider_float(
                        "SSR Max Roughness", et.ssr_max_roughness, 0.1, 1.0, "%.2f"
                    )
                    if ssr_mr_changed:
                        et.ssr_max_roughness = ssr_mr_val
                        et.mark_custom()

                    ssr_dist_changed, ssr_dist_val = imgui.slider_float(
                        "SSR Max Distance", et.ssr_max_distance, 5.0, 100.0, "%.0f m"
                    )
                    if ssr_dist_changed:
                        et.ssr_max_distance = ssr_dist_val
                        et.mark_custom()

                    ssr_res_changed, ssr_res_val = imgui.slider_float(
                        "SSR Resolution Scale",
                        float(getattr(et, "ssr_resolution_scale", 1.0)),
                        0.25,
                        1.0,
                        "%.2f",
                    )
                    if ssr_res_changed:
                        et.ssr_resolution_scale = ssr_res_val
                        et.mark_custom()

                # Anti-Aliasing Mode (Mutually Exclusive: OFF, FXAA, SMAA 1x, SMAA 2x, SMAA 4x, TAA)
                aa_modes = ["OFF", "FXAA", "SMAA 1x", "SMAA 2x", "SMAA 4x", "TAA"]
                aa_internal_modes = ["OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X", "TAA"]
                curr_mode = getattr(et, "aa_mode", "OFF")
                curr_aa_idx = aa_internal_modes.index(curr_mode) if curr_mode in aa_internal_modes else 0

                aa_changed, aa_idx = imgui.combo("Anti-Aliasing", curr_aa_idx, aa_modes)
                if aa_changed:
                    chosen_internal = aa_internal_modes[aa_idx]
                    et.aa_mode = chosen_internal
                    et.taa_enabled = chosen_internal == "TAA"
                    et.mark_custom()
                    self.toast.show(f"Anti-Aliasing: {aa_modes[aa_idx]}", duration=1.5)

                if et.aa_mode == "TAA" or getattr(et, "taa_enabled", False):
                    taa_fb_c, taa_fb_v = imgui.slider_float(
                        "TAA Feedback", getattr(et, "taa_feedback", 0.95), 0.80, 0.98, "%.2f"
                    )
                    if taa_fb_c:
                        et.taa_feedback = taa_fb_v
                        et.mark_custom()

                    taa_sh_c, taa_sh_v = imgui.slider_float(
                        "TAA Sharpness", getattr(et, "taa_sharpness", 0.35), 0.0, 1.0, "%.2f"
                    )
                    if taa_sh_c:
                        et.taa_sharpness = taa_sh_v
                        et.mark_custom()

                    taa_gm_c, taa_gm_v = imgui.slider_float(
                        "TAA Variance Box (Gamma)", getattr(et, "taa_gamma", 1.25), 0.75, 2.50, "%.2f"
                    )
                    if taa_gm_c:
                        et.taa_gamma = taa_gm_v
                        et.mark_custom()

                    taa_maxfb_c, taa_maxfb_v = imgui.slider_float(
                        "TAA Max Feedback", getattr(et, "taa_max_feedback", 0.95), 0.80, 0.99, "%.2f"
                    )
                    if taa_maxfb_c:
                        et.taa_max_feedback = taa_maxfb_v
                        et.mark_custom()

                    taa_minfb_c, taa_minfb_v = imgui.slider_float(
                        "TAA Min Feedback", getattr(et, "taa_min_feedback", 0.78), 0.50, 0.95, "%.2f"
                    )
                    if taa_minfb_c:
                        et.taa_min_feedback = taa_minfb_v
                        et.mark_custom()

                    taa_vg_c, taa_vg_v = imgui.slider_float(
                        "TAA Variance Gamma", getattr(et, "taa_variance_gamma", 1.25), 0.75, 2.50, "%.2f"
                    )
                    if taa_vg_c:
                        et.taa_variance_gamma = taa_vg_v
                        et.mark_custom()
                elif et.aa_mode == "FXAA":
                    fxaa_subpixel_changed, fxaa_subpixel = imgui.slider_float(
                        "FXAA Subpixel Quality", et.fxaa_subpixel, 0.0, 1.0, "%.2f"
                    )
                    if fxaa_subpixel_changed:
                        et.fxaa_subpixel = fxaa_subpixel
                        et.mark_custom()

                    fxaa_threshold_changed, fxaa_threshold = imgui.slider_float(
                        "FXAA Edge Threshold", et.fxaa_edge_threshold, 0.01, 0.25, "%.3f"
                    )
                    if fxaa_threshold_changed:
                        et.fxaa_edge_threshold = fxaa_threshold
                        et.mark_custom()
                elif et.aa_mode.startswith("SMAA"):
                    smaa_threshold_changed, smaa_threshold = imgui.slider_float(
                        "SMAA Edge Threshold", et.smaa_threshold, 0.01, 0.2, "%.3f"
                    )
                    if smaa_threshold_changed:
                        et.smaa_threshold = smaa_threshold
                        et.mark_custom()

                # Dynamic Local Point Lights
                pl_changed, pl_val = imgui.checkbox("Dynamic Point Lights", et.point_lights_enabled)
                if pl_changed:
                    et.point_lights_enabled = pl_val
                    et.mark_custom()
                    self.toast.show(f"Point Lights: {'ON' if pl_val else 'OFF'}", duration=1.5)

                # Dynamic Local Spot Lights & Perspective Shadows
                sl_changed, sl_val = imgui.checkbox("Dynamic Spot Lights", getattr(et, "spot_lights_enabled", True))
                if sl_changed:
                    et.spot_lights_enabled = sl_val
                    et.mark_custom()
                    self.toast.show(f"Spot Lights: {'ON' if sl_val else 'OFF'}", duration=1.5)

                if getattr(et, "spot_lights_enabled", True):
                    ssh_changed, ssh_val = imgui.checkbox(
                        "Spot Shadows (2x2 Perspective Atlas)", getattr(et, "spot_shadows_enabled", True)
                    )
                    if ssh_changed:
                        et.spot_shadows_enabled = ssh_val
                        et.mark_custom()
                        self.toast.show(f"Spot Shadows: {'ON' if ssh_val else 'OFF'}", duration=1.5)

                    spot_res_options = ["1024", "2048"]
                    spot_res_str = str(et.spot_shadow_resolution)
                    spot_res_idx = spot_res_options.index(spot_res_str) if spot_res_str in spot_res_options else 1
                    spot_res_changed, spot_res_new_idx = imgui.combo(
                        "Spot Shadow Resolution", spot_res_idx, spot_res_options
                    )
                    if spot_res_changed:
                        et.spot_shadow_resolution = int(spot_res_options[spot_res_new_idx])
                        et.mark_custom()

            if imgui.collapsing_header("Environment: Atmosphere & Sun"):
                # Atmosphere Presets
                atmo_presets = [
                    "EARTH_DAY",
                    "EARTH_SUNSET",
                    "EARTH_NIGHT",
                    "ALIEN_CYAN_PURPLE",
                    "ALIEN_CRIMSON_MARS",
                    "SPACE",
                    "CUSTOM",
                ]
                preset_labels = [
                    "Earth Day (Noon)",
                    "Earth Sunset / Golden Hour",
                    "Earth Night / Starlight",
                    "Alien: Cyan & Purple",
                    "Alien: Crimson Mars",
                    "Outer Space",
                    "Custom",
                ]
                cur_p_name = getattr(et, "atmo_preset", "EARTH_DAY").upper()
                cur_p_idx = atmo_presets.index(cur_p_name) if cur_p_name in atmo_presets else 6
                ap_c, ap_idx = imgui.combo("Atmosphere Preset", cur_p_idx, preset_labels)
                if ap_c and ap_idx != cur_p_idx:
                    chosen_p = atmo_presets[ap_idx]
                    if chosen_p != "CUSTOM":
                        et.apply_atmo_preset(chosen_p)
                    else:
                        et.atmo_preset = "CUSTOM"
                    self.toast.show(f"Atmosphere: {preset_labels[ap_idx]}", duration=2.0)

                sky_modes = ["ATMOSPHERE", "SPACE"]
                sky_labels = ["Planetary Atmosphere", "Outer Space"]
                sky_mode = getattr(et, "sky_mode", "ATMOSPHERE").upper()
                sky_mode_idx = sky_modes.index(sky_mode) if sky_mode in sky_modes else 0
                sky_mode_changed, sky_mode_new_idx = imgui.combo("Sky Model", sky_mode_idx, sky_labels)
                if sky_mode_changed:
                    et.sky_mode = sky_modes[sky_mode_new_idx]
                    et.atmo_preset = "CUSTOM"

                # Time of Day (24h clock)
                tod_hours = int(et.time_of_day) % 24
                tod_mins = int((et.time_of_day % 1.0) * 60.0)
                time_str = f"{tod_hours:02d}:{tod_mins:02d}"
                imgui.set_next_item_width(-1.0)
                tod_c, tod_v = imgui.slider_float(
                    f"Time of Day ({time_str})##TimeOfDay",
                    et.time_of_day,
                    0.0,
                    24.0,
                    "%.2f h",
                )
                if tod_c:
                    et.day_speed = 0.0
                    et.time_of_day = min(tod_v, 23.999)
                    et.atmo_preset = "CUSTOM"

                sun_intensity_changed, sun_intensity = imgui.slider_float(
                    "Sun Intensity", et.sun_lux, 0.0, 20.0, "%.2f"
                )
                if sun_intensity_changed:
                    et.sun_lux = sun_intensity
                    et.mark_custom()

                # Day-Night Orbital Cycle Speed
                speed_labels = "Paused" if et.day_speed == 0.0 else f"{et.day_speed:.1f}x"
                ds_c, ds_v = imgui.slider_float(f"Cycle Speed ({speed_labels})", et.day_speed, 0.0, 10.0, "%.1fx")
                if ds_c:
                    et.day_speed = ds_v
                    et.atmo_preset = "CUSTOM"

                # Quick pause / play button
                if et.day_speed == 0.0:
                    if imgui.button("Play Day-Night Cycle (1.0x)"):
                        et.day_speed = 1.0
                        et.atmo_preset = "CUSTOM"
                else:
                    if imgui.button("Pause Cycle"):
                        et.day_speed = 0.0
                        et.atmo_preset = "CUSTOM"

                azimuth_c, azimuth_v = imgui.slider_float(
                    "Sun Orbit Azimuth", et.azimuth_deg, -180.0, 180.0, "%.1f deg"
                )
                if azimuth_c:
                    et.azimuth_deg = azimuth_v
                    et.atmo_preset = "CUSTOM"
                latitude_c, latitude_v = imgui.slider_float("Planet Latitude", et.latitude_deg, -90.0, 90.0, "%.1f deg")
                if latitude_c:
                    et.latitude_deg = latitude_v
                    et.atmo_preset = "CUSTOM"
                declination_c, declination_v = imgui.slider_float(
                    "Solar Declination", et.declination_deg, -23.44, 23.44, "%.2f deg"
                )
                if declination_c:
                    et.declination_deg = declination_v
                    et.atmo_preset = "CUSTOM"
                moon_offset_c, moon_offset_v = imgui.slider_float(
                    "Moon Orbit Offset", et.moon_orbit_offset_hours, 0.0, 24.0, "%.2f h"
                )
                if moon_offset_c:
                    et.moon_orbit_offset_hours = moon_offset_v
                    et.atmo_preset = "CUSTOM"
                moon_decl_c, moon_decl_v = imgui.slider_float(
                    "Moon Declination", et.moon_declination_deg, -28.5, 28.5, "%.2f deg"
                )
                if moon_decl_c:
                    et.moon_declination_deg = moon_decl_v
                    et.atmo_preset = "CUSTOM"
                moon_az_c, moon_az_v = imgui.slider_float(
                    "Moon Azimuth Offset", et.moon_azimuth_offset_deg, -180.0, 180.0, "%.1f deg"
                )
                if moon_az_c:
                    et.moon_azimuth_offset_deg = moon_az_v
                    et.atmo_preset = "CUSTOM"
                sky_exposure_c, sky_exposure_v = imgui.slider_float("Sky Exposure", et.sky_exposure, 0.1, 4.0, "%.2f")
                if sky_exposure_c:
                    et.sky_exposure = sky_exposure_v
                    et.atmo_preset = "CUSTOM"
                sky_saturation_c, sky_saturation_v = imgui.slider_float(
                    "Sky Saturation", et.sky_saturation, 0.0, 2.0, "%.2f"
                )
                if sky_saturation_c:
                    et.sky_saturation = sky_saturation_v
                    et.atmo_preset = "CUSTOM"

                # Collapsible Alien & Physical Scattering Tuning
                if imgui.tree_node("Atmospheric Scattering & Sky Colors"):
                    imgui.text_disabled("Rayleigh Wavelength Multipliers (Alien Sky Tuning):")
                    rr_c, rr_v = imgui.slider_float("Rayleigh Red (680nm)", et.rayleigh_r, 0.1, 50.0, "%.1f")
                    if rr_c:
                        et.rayleigh_r = rr_v
                        et.atmo_preset = "CUSTOM"

                    rg_c, rg_v = imgui.slider_float("Rayleigh Green (550nm)", et.rayleigh_g, 0.1, 50.0, "%.1f")
                    if rg_c:
                        et.rayleigh_g = rg_v
                        et.atmo_preset = "CUSTOM"

                    rb_c, rb_v = imgui.slider_float("Rayleigh Blue (440nm)", et.rayleigh_b, 0.1, 50.0, "%.1f")
                    if rb_c:
                        et.rayleigh_b = rb_v
                        et.atmo_preset = "CUSTOM"

                    mie_c, mie_v = imgui.slider_float("Mie Aerosol Scattering", et.mie_coeff, 1.0, 80.0, "%.1f")
                    if mie_c:
                        et.mie_coeff = mie_v
                        et.atmo_preset = "CUSTOM"

                    asymmetry_c, asymmetry_v = imgui.slider_float(
                        "Mie Forward Scattering", et.mie_asymmetry, 0.0, 0.95, "%.2f"
                    )
                    if asymmetry_c:
                        et.mie_asymmetry = asymmetry_v
                        et.atmo_preset = "CUSTOM"

                    turb_c, turb_v = imgui.slider_float("Atmospheric Turbidity (Haze)", et.turbidity, 1.0, 10.0, "%.2f")
                    if turb_c:
                        et.turbidity = turb_v
                        et.atmo_preset = "CUSTOM"

                    sun_color_c, sun_color_v = imgui.color_edit3("Sun Color", et.sun_color)
                    if sun_color_c:
                        et.sun_color = tuple(sun_color_v)
                        et.atmo_preset = "CUSTOM"
                    moon_color_c, moon_color_v = imgui.color_edit3("Moon Color", et.moon_color)
                    if moon_color_c:
                        et.moon_color = tuple(moon_color_v)
                        et.atmo_preset = "CUSTOM"

                    star_c, star_v = imgui.slider_float("Starfield Brightness", et.star_intensity, 0.0, 5.0, "%.2f")
                    if star_c:
                        et.star_intensity = star_v
                        et.atmo_preset = "CUSTOM"

                    density_c, density_v = imgui.slider_float("Star Density", et.star_density, 0.1, 4.0, "%.2f")
                    if density_c:
                        et.star_density = density_v
                        et.atmo_preset = "CUSTOM"

                    planet_radius_c, planet_radius_v = imgui.slider_float(
                        "Planet Radius (km)", et.planet_radius / 1000.0, 100.0, 10000.0, "%.0f km"
                    )
                    if planet_radius_c:
                        et.planet_radius = planet_radius_v * 1000.0
                        if et.atmosphere_radius <= et.planet_radius:
                            et.atmosphere_radius = et.planet_radius + 100000.0
                        et.atmo_preset = "CUSTOM"
                    atmo_height_c, atmo_height_v = imgui.slider_float(
                        "Atmosphere Thickness (km)",
                        (et.atmosphere_radius - et.planet_radius) / 1000.0,
                        1.0,
                        250.0,
                        "%.1f km",
                    )
                    if atmo_height_c:
                        et.atmosphere_radius = et.planet_radius + atmo_height_v * 1000.0
                        et.atmo_preset = "CUSTOM"
                    rayleigh_height_c, rayleigh_height_v = imgui.slider_float(
                        "Rayleigh Scale Height (km)", et.h_r / 1000.0, 1.0, 50.0, "%.1f km"
                    )
                    if rayleigh_height_c:
                        et.h_r = rayleigh_height_v * 1000.0
                        et.atmo_preset = "CUSTOM"
                    mie_height_c, mie_height_v = imgui.slider_float(
                        "Mie Scale Height (km)", et.h_m / 1000.0, 0.1, 25.0, "%.1f km"
                    )
                    if mie_height_c:
                        et.h_m = mie_height_v * 1000.0
                        et.atmo_preset = "CUSTOM"

                    if et.sky_mode == "SPACE":
                        nebula_c, nebula_v = imgui.slider_float(
                            "Nebula Intensity", et.nebula_intensity, 0.0, 2.0, "%.2f"
                        )
                        if nebula_c:
                            et.nebula_intensity = nebula_v
                            et.atmo_preset = "CUSTOM"
                        space_c, space_v = imgui.color_edit3("Space Background", et.space_zenith)
                        if space_c:
                            et.space_zenith = tuple(space_v)
                            et.atmo_preset = "CUSTOM"
                        nebula_color_c, nebula_color_v = imgui.color_edit3("Nebula Color", et.nebula_color)
                        if nebula_color_c:
                            et.nebula_color = tuple(nebula_color_v)
                            et.atmo_preset = "CUSTOM"
                    else:
                        clouds_c, clouds_v = imgui.checkbox("Procedural Cloud Deck", et.cloud_enabled)
                        if clouds_c:
                            et.cloud_enabled = clouds_v
                            et.atmo_preset = "CUSTOM"
                        if et.cloud_enabled:
                            coverage_c, coverage_v = imgui.slider_float(
                                "Cloud Coverage", et.cloud_coverage, 0.0, 1.0, "%.2f"
                            )
                            if coverage_c:
                                et.cloud_coverage = coverage_v
                                et.atmo_preset = "CUSTOM"
                            opacity_c, opacity_v = imgui.slider_float(
                                "Cloud Opacity", et.cloud_opacity, 0.0, 1.0, "%.2f"
                            )
                            if opacity_c:
                                et.cloud_opacity = opacity_v
                                et.atmo_preset = "CUSTOM"
                            altitude_c, altitude_v = imgui.slider_float(
                                "Cloud Deck Altitude (m)", et.cloud_altitude, 250.0, 12000.0, "%.0f"
                            )
                            if altitude_c:
                                et.cloud_altitude = altitude_v
                                et.atmo_preset = "CUSTOM"
                            scale_c, scale_v = imgui.slider_float(
                                "Cloud Size (m)", et.cloud_scale, 5000.0, 80000.0, "%.0f"
                            )
                            if scale_c:
                                et.cloud_scale = scale_v
                                et.atmo_preset = "CUSTOM"
                            layer_c, layer_v = imgui.slider_float(
                                "Upper Cloud Layer Separation (m)", et.cloud_layer_separation, 1000.0, 20000.0, "%.0f"
                            )
                            if layer_c:
                                et.cloud_layer_separation = layer_v
                                et.atmo_preset = "CUSTOM"
                            upper_opacity_c, upper_opacity_v = imgui.slider_float(
                                "Upper Cloud Layer Opacity", et.cloud_secondary_opacity, 0.0, 0.6, "%.2f"
                            )
                            if upper_opacity_c:
                                et.cloud_secondary_opacity = upper_opacity_v
                                et.atmo_preset = "CUSTOM"
                            wind_x_c, wind_x_v = imgui.slider_float(
                                "Cloud Wind X (m/s)", et.cloud_wind_x, -100.0, 100.0, "%.1f"
                            )
                            if wind_x_c:
                                et.cloud_wind_x = wind_x_v
                                et.atmo_preset = "CUSTOM"
                            wind_z_c, wind_z_v = imgui.slider_float(
                                "Cloud Wind Z (m/s)", et.cloud_wind_z, -100.0, 100.0, "%.1f"
                            )
                            if wind_z_c:
                                et.cloud_wind_z = wind_z_v
                                et.atmo_preset = "CUSTOM"

                    imgui.tree_pop()

            if imgui.collapsing_header("Environment: Volumetric Fog"):
                fog_c, fog_v = imgui.checkbox("Froxel Volumetric Fog", et.volumetric_fog_enabled)
                if fog_c:
                    et.volumetric_fog_enabled = fog_v
                    et.mark_custom()
                    self.toast.show(f"Volumetric Fog: {'ON' if fog_v else 'OFF'}", duration=1.5)

                if et.volumetric_fog_enabled:
                    res_names = [
                        "LOW (80x45x32)",
                        "MEDIUM (120x68x48)",
                        "HIGH (160x90x64)",
                        "ULTRA (200x112x80)",
                        "CINEMATIC (240x135x96)",
                    ]
                    res_keys = ["LOW", "MEDIUM", "HIGH", "ULTRA", "CINEMATIC"]
                    cur_res_str = getattr(et, "fog_resolution", "HIGH").upper()
                    cur_res_idx = res_keys.index(cur_res_str) if cur_res_str in res_keys else 2
                    res_c, res_idx = imgui.combo("Froxel Resolution", cur_res_idx, res_names)
                    if res_c:
                        et.fog_resolution = res_keys[res_idx]
                        et.mark_custom()
                        self.toast.show(f"Froxel Resolution: {res_names[res_idx]}", duration=1.5)

                    fpl_c, fpl_v = imgui.checkbox("Point Light Volumetrics", getattr(et, "fog_point_lights", True))
                    if fpl_c:
                        et.fog_point_lights = fpl_v
                        et.mark_custom()
                        self.toast.show(f"Point Light Volumetrics: {'ON' if fpl_v else 'OFF'}", duration=1.5)

                    fd_c, fd_v = imgui.slider_float("Fog Density", et.fog_density, 0.001, 0.10, "%.3f")
                    if fd_c:
                        et.fog_density = fd_v
                        et.mark_custom()

                    fh_c, fh_v = imgui.slider_float("Height Falloff", et.fog_height_falloff, 0.01, 0.50, "%.2f")
                    if fh_c:
                        et.fog_height_falloff = fh_v
                        et.mark_custom()

                    fa_c, fa_v = imgui.slider_float("Anisotropy (Phase g)", et.fog_anisotropy, 0.0, 0.95, "%.2f")
                    if fa_c:
                        et.fog_anisotropy = fa_v
                        et.mark_custom()

                    fdist_c, fdist_v = imgui.slider_float("Max Distance", et.fog_distance, 50.0, 1000.0, "%.0f m")
                    if fdist_c:
                        et.fog_distance = fdist_v
                        et.mark_custom()

                    famb_c, famb_v = imgui.slider_float("Ambient Light", et.fog_ambient, 0.0, 1.0, "%.2f")
                    if famb_c:
                        et.fog_ambient = famb_v
                        et.mark_custom()

                    debug_modes = ["Normal Composite", "In-Scattering Only", "Transmittance Only"]
                    cur_dbg_idx = max(0, min(2, et.fog_debug_mode))
                    dbg_c, dbg_idx = imgui.combo("Fog Visualizer", cur_dbg_idx, debug_modes)
                    if dbg_c:
                        et.fog_debug_mode = dbg_idx
                        et.mark_custom()
                        self.toast.show(f"Fog Visualizer: {debug_modes[dbg_idx]}", duration=1.5)

            if imgui.collapsing_header("Camera: Lens & Motion Effects"):
                # Depth of Field
                dof_c, dof_v = imgui.checkbox("Bokeh Depth of Field", getattr(et, "dof_enabled", True))
                if dof_c:
                    et.dof_enabled = dof_v
                    et.mark_custom()
                    self.toast.show(f"Depth of Field: {'ON' if dof_v else 'OFF'}", duration=1.5)

                if getattr(et, "dof_enabled", True):
                    fd_c, fd_v = imgui.slider_float("Focus Distance", et.dof_focus_distance, 0.5, 50.0, "%.1f m")
                    if fd_c:
                        et.dof_focus_distance = fd_v
                        et.mark_custom()

                    fl_c, fl_v = imgui.slider_float("Focal Length", et.dof_focal_length, 18.0, 135.0, "%.0f mm")
                    if fl_c:
                        et.dof_focal_length = fl_v
                        et.mark_custom()

                    ap_c, ap_v = imgui.slider_float("Aperture (f-stop)", et.dof_aperture, 1.2, 16.0, "f/%.1f")
                    if ap_c:
                        et.dof_aperture = ap_v
                        et.mark_custom()

                    shapes = ["CIRCULAR", "HEXAGONAL", "ANAMORPHIC"]
                    shape_labels = ["Circular Bokeh", "Hexagonal Diaphragm", "Anamorphic (2x Stretch)"]
                    cur_shape_str = getattr(et, "dof_bokeh_shape", "CIRCULAR").upper()
                    cur_s_idx = shapes.index(cur_shape_str) if cur_shape_str in shapes else 0
                    sh_c, sh_idx = imgui.combo("Bokeh Diaphragm", cur_s_idx, shape_labels)
                    if sh_c:
                        et.dof_bokeh_shape = shapes[sh_idx]
                        if et.dof_bokeh_shape == "ANAMORPHIC":
                            et.dof_anamorphic_ratio = 2.0
                        else:
                            et.dof_anamorphic_ratio = 1.0
                        et.mark_custom()

                    mc_c, mc_v = imgui.slider_float("Max Blur Radius", et.dof_max_coc, 4.0, 48.0, "%.0f px")
                    if mc_c:
                        et.dof_max_coc = mc_v
                        et.mark_custom()

                # Motion Blur
                mb_c, mb_v = imgui.checkbox("Velocity Motion Blur", getattr(et, "motion_blur_enabled", True))
                if mb_c:
                    et.motion_blur_enabled = mb_v
                    et.mark_custom()
                    self.toast.show(f"Motion Blur: {'ON' if mb_v else 'OFF'}", duration=1.5)

                if getattr(et, "motion_blur_enabled", True):
                    mbi_c, mbi_v = imgui.slider_float(
                        "Shutter Blur Intensity", et.motion_blur_intensity, 0.1, 3.0, "%.2fx"
                    )
                    if mbi_c:
                        et.motion_blur_intensity = mbi_v
                        et.mark_custom()

                    mbs_c, mbs_v = imgui.slider_int("Motion Blur Taps", et.motion_blur_samples, 4, 32)
                    if mbs_c:
                        et.motion_blur_samples = mbs_v
                        et.mark_custom()

                    mbr_c, mbr_v = imgui.slider_float(
                        "Motion Blur Max Radius", et.motion_blur_max_radius, 1.0, 64.0, "%.0f px"
                    )
                    if mbr_c:
                        et.motion_blur_max_radius = mbr_v
                        et.mark_custom()

                # Lens Flare Mode (OFF / ANAMORPHIC / SOLAR)
                flare_modes = ["OFF", "ANAMORPHIC", "SOLAR"]
                curr_flare = str(getattr(et, "lens_flare_mode", "ANAMORPHIC")).upper()
                fl_idx = flare_modes.index(curr_flare) if curr_flare in flare_modes else 1
                flm_c, flm_idx = imgui.combo("Lens Flare Mode", fl_idx, flare_modes)
                if flm_c and flm_idx != fl_idx:
                    et.lens_flare_mode = flare_modes[flm_idx]
                    et.lens_flare_enabled = flare_modes[flm_idx] != "OFF"
                    et.mark_custom()
                    self.toast.show(f"Lens Flare: {et.lens_flare_mode}", duration=1.5)

                if getattr(et, "lens_flare_enabled", True):
                    lft_c, lft_v = imgui.slider_float("Flare Threshold", et.lens_flare_threshold, 1.0, 5.0, "%.2f")
                    if lft_c:
                        et.lens_flare_threshold = lft_v
                        et.mark_custom()

                    # Solar-flare-only controls
                    if str(getattr(et, "lens_flare_mode", "ANAMORPHIC")).upper() == "SOLAR":
                        sfb_c, sfb_v = imgui.slider_int(
                            "Solar Blade Count", int(getattr(et, "solar_flare_blade_count", 6)), 2, 16
                        )
                        if sfb_c:
                            et.solar_flare_blade_count = sfb_v
                            et.mark_custom()

                        sbl_c, sbl_v = imgui.slider_float(
                            "Solar Bloom", float(getattr(et, "solar_flare_bloom_intensity", 1.0)), 0.0, 3.0, "%.2f"
                        )
                        if sbl_c:
                            et.solar_flare_bloom_intensity = sbl_v
                            et.mark_custom()

                        sst_c, sst_v = imgui.slider_float(
                            "Solar Starburst",
                            float(getattr(et, "solar_flare_starburst_intensity", 0.6)),
                            0.0,
                            3.0,
                            "%.2f",
                        )
                        if sst_c:
                            et.solar_flare_starburst_intensity = sst_v
                            et.mark_custom()

                        soc_c, soc_v = imgui.slider_float(
                            "Solar Occlusion Fade",
                            float(getattr(et, "solar_flare_occlusion_fade", 1.0)),
                            0.0,
                            1.0,
                            "%.2f",
                        )
                        if soc_c:
                            et.solar_flare_occlusion_fade = soc_v
                            et.mark_custom()

                    lfs_c, lfs_v = imgui.slider_float(
                        "Streak Intensity", et.lens_flare_streak_intensity, 0.0, 2.0, "%.2f"
                    )
                    if lfs_c:
                        et.lens_flare_streak_intensity = lfs_v
                        et.mark_custom()

                    lfw_c, lfw_v = imgui.slider_float("Streak Width", et.lens_flare_streak_width, 4.0, 96.0, "%.0f px")
                    if lfw_c:
                        et.lens_flare_streak_width = lfw_v
                        et.mark_custom()

                    lfg_c, lfg_v = imgui.slider_float(
                        "Ghost Reflections", et.lens_flare_ghost_intensity, 0.0, 1.5, "%.2f"
                    )
                    if lfg_c:
                        et.lens_flare_ghost_intensity = lfg_v
                        et.mark_custom()

                    lfh_c, lfh_v = imgui.slider_float("Flare Halo", et.lens_flare_halo_intensity, 0.0, 1.5, "%.2f")
                    if lfh_c:
                        et.lens_flare_halo_intensity = lfh_v
                        et.mark_custom()

                    lftint_c, lftint_v = imgui.color_edit3("Flare Tint", et.lens_flare_tint)
                    if lftint_c:
                        et.lens_flare_tint = tuple(lftint_v)
                        et.mark_custom()

                    lfr_c, lfr_v = imgui.slider_float(
                        "Flare Resolution Scale",
                        float(getattr(et, "lens_flare_resolution_scale", 1.0)),
                        0.25,
                        1.0,
                        "%.2f",
                    )
                    if lfr_c:
                        et.lens_flare_resolution_scale = lfr_v
                        et.mark_custom()

                # HDR Bloom
                bl_c, bl_v = imgui.checkbox("HDR Bloom", getattr(et, "bloom_enabled", True))
                if bl_c:
                    et.bloom_enabled = bl_v
                    et.mark_custom()
                if getattr(et, "bloom_enabled", True):
                    bli_c, bli_v = imgui.slider_float(
                        "Bloom Intensity", getattr(et, "bloom_intensity", 0.045), 0.0, 0.20, "%.3f"
                    )
                    if bli_c:
                        et.bloom_intensity = bli_v
                        et.mark_custom()

                # Lens Imperfections (Chromatic Aberration, Vignette, Film Grain)
                ca_c, ca_v = imgui.checkbox("Chromatic Aberration", getattr(et, "chromatic_aberration_enabled", True))
                if ca_c:
                    et.chromatic_aberration_enabled = ca_v
                    et.mark_custom()
                if getattr(et, "chromatic_aberration_enabled", True):
                    cai_c, cai_v = imgui.slider_float(
                        "Spectral Fringing", et.chromatic_aberration_intensity, 0.0, 0.02, "%.4f"
                    )
                    if cai_c:
                        et.chromatic_aberration_intensity = cai_v
                        et.mark_custom()

                vig_c, vig_v = imgui.checkbox("Physical Lens Vignette", getattr(et, "vignette_enabled", True))
                if vig_c:
                    et.vignette_enabled = vig_v
                    et.mark_custom()
                if getattr(et, "vignette_enabled", True):
                    vigi_c, vigi_v = imgui.slider_float("Vignette Intensity", et.vignette_intensity, 0.0, 1.0, "%.2f")
                    if vigi_c:
                        et.vignette_intensity = vigi_v
                        et.mark_custom()

                    vigr_c, vigr_v = imgui.slider_float("Vignette Roundness", et.vignette_roundness, 0.1, 1.0, "%.2f")
                    if vigr_c:
                        et.vignette_roundness = vigr_v
                        et.mark_custom()

                    vigs_c, vigs_v = imgui.slider_float(
                        "Vignette Smoothness", et.vignette_smoothness, 0.05, 1.0, "%.2f"
                    )
                    if vigs_c:
                        et.vignette_smoothness = vigs_v
                        et.mark_custom()

                fg_c, fg_v = imgui.checkbox("Filmic Grain (35mm)", getattr(et, "film_grain_enabled", True))
                if fg_c:
                    et.film_grain_enabled = fg_v
                    et.mark_custom()
                if getattr(et, "film_grain_enabled", True):
                    fgi_c, fgi_v = imgui.slider_float("Grain ISO", et.film_grain_intensity, 0.0, 0.15, "%.3f")
                    if fgi_c:
                        et.film_grain_intensity = fgi_v
                        et.mark_custom()

            if imgui.collapsing_header("Geometry: Displacement & Visibility"):
                if getattr(self, "_scroll_to_microgeom", False):
                    imgui.set_scroll_here_y(0.0)
                    self._scroll_to_microgeom = False

                # POM
                pom_c, pom_v = imgui.checkbox("Parallax Occlusion Mapping (POM)", et.pom_enabled)
                if pom_c:
                    et.pom_enabled = pom_v
                    et.mark_custom()
                if et.pom_enabled:
                    pmin_c, pmin_v = imgui.slider_int("POM Minimum Samples", et.pom_min_samples, 4, 64)
                    if pmin_c:
                        et.pom_min_samples = min(pmin_v, et.pom_max_samples)
                        et.mark_custom()
                    pmax_c, pmax_v = imgui.slider_int("POM Maximum Samples", et.pom_max_samples, 8, 128)
                    if pmax_c:
                        et.pom_max_samples = max(pmax_v, et.pom_min_samples)
                        et.mark_custom()
                    ps_c, ps_v = imgui.slider_float("POM Height Scale", et.pom_height_scale, 0.0, 2.0, "%.2f")
                    if ps_c:
                        et.pom_height_scale = ps_v
                        et.mark_custom()
                    pss_c, pss_v = imgui.checkbox("POM Sun Self-Shadowing", et.pom_self_shadow)
                    if pss_c:
                        et.pom_self_shadow = pss_v
                        et.mark_custom()
                    dnr_c, dnr_v = imgui.slider_float(
                        "POM Shadow Radius (m)", et.disp_near_radius, 10.0, 300.0, "%.1f m"
                    )
                    if dnr_c:
                        et.disp_near_radius = dnr_v
                        et.mark_custom()
                    dmr_c, dmr_v = imgui.slider_float(
                        "POM Max Cutoff Radius (m)", et.disp_mid_radius, 30.0, 500.0, "%.1f m"
                    )
                    if dmr_c:
                        et.disp_mid_radius = dmr_v
                        et.mark_custom()

                # Vectorized Frustum Culling (Phase 3) & Static/Dynamic Separation (Phase 4)
                if self._scroll_to_culling:
                    imgui.set_scroll_here_y(0.2)
                    self._scroll_to_culling = False
                fc_c, fc_v = imgui.checkbox("Vectorized Frustum Culling", et.frustum_cull_enabled)
                if fc_c:
                    et.frustum_cull_enabled = fc_v
                    et.mark_custom()

                hiz_c, hiz_v = imgui.checkbox("GPU Hi-Z Occlusion Culling", getattr(et, "hiz_cull_enabled", False))
                if hiz_c:
                    et.hiz_cull_enabled = hiz_v
                    et.mark_custom()

                pipe = getattr(self.app, "pipeline", None) if self.app is not None else None
                if pipe is not None:
                    vis_ent = getattr(pipe, "stats_visible_entities", 0)
                    cull_ent = getattr(pipe, "stats_culled_entities", 0)
                    vis_cmd = getattr(pipe, "stats_visible_commands", 0)
                    cull_cmd = getattr(pipe, "stats_culled_commands", 0)
                    imgui.text_colored(
                        imgui.ImVec4(0.4, 0.9, 0.4, 1.0), f"  Entities: {vis_ent} Visible | {cull_ent} Culled"
                    )
                    imgui.text_colored(
                        imgui.ImVec4(0.4, 0.8, 1.0, 1.0), f"  Commands: {vis_cmd} Visible | {cull_cmd} Culled"
                    )

                ecs_ref = getattr(self.app, "ecs", None) if self.app is not None else None
                if ecs_ref is not None and getattr(ecs_ref, "static_count", 0) > 0:
                    s_cnt = ecs_ref.static_count
                    d_cnt = max(0, ecs_ref.active_count - s_cnt)
                    saved_kb = (s_cnt * 96) / 1024.0
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                        f"  Static: {s_cnt} | Dynamic: {d_cnt} (Saved: {saved_kb:.1f} KB/frame)",
                    )

                # Hardware Tessellation
                tess_c, tess_v = imgui.checkbox("Hardware GPU Tessellation", et.tess_enabled)
                if tess_c:
                    et.tess_enabled = tess_v
                    et.mark_custom()
                if et.tess_enabled:
                    dnr_c, dnr_v = imgui.slider_float(
                        "Near Quality Radius (m)", et.disp_near_radius, 10.0, 300.0, "%.1f m"
                    )
                    if dnr_c:
                        et.disp_near_radius = dnr_v
                        et.mark_custom()
                    dmr_c, dmr_v = imgui.slider_float(
                        "Mid Radius Cutoff (m)", et.disp_mid_radius, 30.0, 500.0, "%.1f m"
                    )
                    if dmr_c:
                        et.disp_mid_radius = dmr_v
                        et.mark_custom()
                    tl_c, tl_v = imgui.slider_float("Near Tess Level", et.tess_max_level, 1.0, 32.0, "%.0f")
                    if tl_c:
                        et.tess_max_level = tl_v
                        et.mark_custom()
                    tml_c, tml_v = imgui.slider_float("Mid Tess Level", et.tess_med_level, 1.0, 16.0, "%.0f")
                    if tml_c:
                        et.tess_med_level = tml_v
                        et.mark_custom()
                    tds_c, tds_v = imgui.slider_float(
                        "Disp Depth Multiplier", et.tess_displacement_scale, 0.1, 3.0, "%.2fx"
                    )
                    if tds_c:
                        et.tess_displacement_scale = tds_v
                        et.mark_custom()

                # SSDM
                ssdm_c, ssdm_v = imgui.checkbox("Screen-Space Displacement (SSDM)", et.ssdm_enabled)
                if ssdm_c:
                    et.ssdm_enabled = ssdm_v
                    et.mark_custom()
                if et.ssdm_enabled:
                    sc_c, sc_v = imgui.slider_float("SSDM Scale", et.ssdm_scale, 0.0, 2.0, "%.2f")
                    if sc_c:
                        et.ssdm_scale = sc_v
                        et.mark_custom()

            if imgui.collapsing_header("Shadows: Cascades", imgui.TreeNodeFlags_.default_open.value):
                # 1. Shadow Map Resolution (1024, 2048, 4096, 8192)
                res_options = ["1024", "2048", "4096", "8192"]
                curr_res_str = str(et.shadow_resolution)
                res_idx = res_options.index(curr_res_str) if curr_res_str in res_options else 3
                res_changed, new_res_idx = imgui.combo("Shadow Resolution", res_idx, res_options)
                if res_changed and new_res_idx != res_idx:
                    et.shadow_resolution = int(res_options[new_res_idx])
                    et.mark_custom()
                    self.toast.show(f"Shadow Resolution: {et.shadow_resolution}x{et.shadow_resolution}", duration=2.0)

                # 2. Shadow Cascades (1 to 4)
                casc_options = ["1 Cascade", "2 Cascades", "3 Cascades", "4 Cascades"]
                curr_casc_idx = max(0, min(3, et.csm_cascades - 1))
                casc_changed, new_casc_idx = imgui.combo("Shadow Cascades", curr_casc_idx, casc_options)
                if casc_changed and new_casc_idx != curr_casc_idx:
                    et.csm_cascades = new_casc_idx + 1
                    et.mark_custom()
                    self.toast.show(f"Shadow Cascades: {et.csm_cascades}", duration=2.0)

                # 3. Per-cascade far distance. Cascade 0 is the tightest
                # near-field slice; each subsequent cascade reaches further out.
                # The sliders are clamped to stay strictly increasing so the
                # cascade selection in the shader remains valid.
                imgui.text_disabled("Cascade Far Distances")
                distances = list(et.csm_cascade_distances)
                changed = False
                for index in range(4):
                    lower = 1.0 if index == 0 else distances[index - 1] + 1.0
                    upper = 4000.0 if index == 3 else max(lower + 1.0, distances[index + 1] - 1.0)
                    c_changed, c_val = imgui.slider_float(
                        f"Cascade {index} End", distances[index], lower, upper, "%.0f m"
                    )
                    if c_changed:
                        distances[index] = c_val
                        changed = True
                if changed:
                    # Re-establish strict ordering after any edit.
                    for index in range(1, 4):
                        if distances[index] <= distances[index - 1]:
                            distances[index] = distances[index - 1] + 1.0
                    et.csm_cascade_distances = tuple(distances)
                    et.shadow_distance = distances[3]
                    et.mark_custom()

                # 4. Shadow Filtering. Tiers are limited to a prefix of the
                # cascades; the far cascades keep the cheaper path because their
                # texels already span several screen pixels.
                filter_modes = ["HARD", "PCF", "PCSS"]
                curr_filter = str(getattr(et, "shadow_filter_mode", "PCF")).upper()
                f_idx = filter_modes.index(curr_filter) if curr_filter in filter_modes else 1
                f_changed, new_f_idx = imgui.combo("Shadow Filter", f_idx, filter_modes)
                if f_changed and new_f_idx != f_idx:
                    et.shadow_filter_mode = filter_modes[new_f_idx]
                    et.mark_custom()
                    self.toast.show(f"Shadow Filter: {et.shadow_filter_mode}", duration=1.5)

                active_filter = str(getattr(et, "shadow_filter_mode", "PCF")).upper()

                if active_filter in ("PCF", "PCSS"):
                    lim_changed, lim_val = imgui.slider_int(
                        "PCF Cascade Limit", int(getattr(et, "pcf_cascade_limit", 3)), 1, 4
                    )
                    if lim_changed:
                        et.pcf_cascade_limit = lim_val
                        et.mark_custom()

                    pcf_s_changed, pcf_s_val = imgui.slider_int(
                        "PCF Samples", int(getattr(et, "pcf_samples", 16)), 4, 64
                    )
                    if pcf_s_changed:
                        et.pcf_samples = pcf_s_val
                        et.mark_custom()

                    pcf_r_changed, pcf_r_val = imgui.slider_float(
                        "PCF Radius", float(getattr(et, "pcf_radius", 1.5)), 0.5, 6.0, "%.2f texels"
                    )
                    if pcf_r_changed:
                        et.pcf_radius = pcf_r_val
                        et.mark_custom()

                    pcf_f_changed, pcf_f_val = imgui.slider_float(
                        "PCF Falloff", float(getattr(et, "pcf_falloff", 2.0)), 0.25, 6.0, "%.2f"
                    )
                    if pcf_f_changed:
                        et.pcf_falloff = pcf_f_val
                        et.mark_custom()

                if active_filter == "PCSS":
                    imgui.separator()
                    imgui.text_disabled("PCSS (variable penumbra)")

                    pcss_lim_changed, pcss_lim_val = imgui.slider_int(
                        "PCSS Cascade Limit", int(getattr(et, "pcss_cascade_limit", 1)), 1, 4
                    )
                    if pcss_lim_changed:
                        et.pcss_cascade_limit = pcss_lim_val
                        et.mark_custom()

                    pcss_ls_changed, pcss_ls_val = imgui.slider_float(
                        "PCSS Light Size",
                        float(getattr(et, "pcss_light_size", 0.02)),
                        0.001,
                        0.20,
                        "%.3f",
                    )
                    if pcss_ls_changed:
                        et.pcss_light_size = pcss_ls_val
                        et.mark_custom()

                    pcss_ss_changed, pcss_ss_val = imgui.slider_int(
                        "PCSS Search Samples", int(getattr(et, "pcss_search_samples", 16)), 4, 64
                    )
                    if pcss_ss_changed:
                        et.pcss_search_samples = pcss_ss_val
                        et.mark_custom()

                    pcss_sr_changed, pcss_sr_val = imgui.slider_float(
                        "PCSS Search Radius",
                        float(getattr(et, "pcss_search_radius", 8.0)),
                        1.0,
                        32.0,
                        "%.1f texels",
                    )
                    if pcss_sr_changed:
                        et.pcss_search_radius = pcss_sr_val
                        et.mark_custom()

                    pcss_s_changed, pcss_s_val = imgui.slider_int(
                        "PCSS Samples", int(getattr(et, "pcss_samples", 24)), 4, 64
                    )
                    if pcss_s_changed:
                        et.pcss_samples = pcss_s_val
                        et.mark_custom()

                    pcss_f_changed, pcss_f_val = imgui.slider_float(
                        "PCSS Falloff", float(getattr(et, "pcss_falloff", 2.0)), 0.25, 6.0, "%.2f"
                    )
                    if pcss_f_changed:
                        et.pcss_falloff = pcss_f_val
                        et.mark_custom()

                    pcss_mr_changed, pcss_mr_val = imgui.slider_float(
                        "PCSS Max Radius",
                        float(getattr(et, "pcss_max_radius", 96.0)),
                        2.0,
                        160.0,
                        "%.1f texels",
                    )
                    if pcss_mr_changed:
                        et.pcss_max_radius = pcss_mr_val
                        et.mark_custom()

                    # 0 = classic average-blocker PCSS, 1 = closest blocker.
                    # The closest blocker keeps overlapping shadows sharp
                    # instead of letting a far caster's softness bleed onto a
                    # near caster's edge.
                    pcss_cb_changed, pcss_cb_val = imgui.slider_float(
                        "PCSS Closest Blocker",
                        float(getattr(et, "pcss_closest_blocker_weight", 1.0)),
                        0.0,
                        1.0,
                        "%.2f",
                    )
                    if pcss_cb_changed:
                        et.pcss_closest_blocker_weight = pcss_cb_val
                        et.mark_custom()

                # 5. Shadow Depth Bias
                bias_changed, bias_val = imgui.slider_float("Shadow Depth Bias", et.shadow_bias, 0.0000, 0.0050, "%.5f")
                if bias_changed:
                    et.shadow_bias = bias_val
                    et.mark_custom()

                # 6. Shadow Normal Bias
                nb_changed, nb_val = imgui.slider_float(
                    "Shadow Normal Bias", et.shadow_normal_bias, 0.0000, 0.0050, "%.5f"
                )
                if nb_changed:
                    et.shadow_normal_bias = nb_val
                    et.mark_custom()

            if imgui.collapsing_header("Diagnostics: Render Views"):
                # Mesh Wireframe
                wire_changed, wire_val = imgui.checkbox("Mesh Wireframe", et.show_wireframe)
                if wire_changed:
                    et.show_wireframe = wire_val
                    self.toast.show(f"Mesh Wireframe: {'ON' if wire_val else 'OFF'}", duration=1.5)

            imgui.separator()
            imgui.text_disabled("[F2] Close Panel  |  [F9] Mouse Grab")
        imgui.end()

    def _render_f3_game_tweaks(self) -> None:
        """Renders independent F3 Gameplay Developer Tweaks & Inspector panel."""
        imgui.set_next_window_size(imgui.ImVec2(440.0, 520.0), imgui.Cond_.first_use_ever.value)
        f3_x = 20.0 + (960.0 if self.width >= 1920 else (520.0 if self.width >= 1600 else 320.0))
        imgui.set_next_window_pos(imgui.ImVec2(f3_x, 20.0), imgui.Cond_.first_use_ever.value)
        expanded, p_open = imgui.begin("Gameplay Tweaks & Inspector [F3]", p_open=True)
        if not p_open:
            self.show_game_tweaks = False

        if expanded:
            for cat in self.game_tweaks.get_categories():
                if imgui.collapsing_header(cat, imgui.TreeNodeFlags_.default_open.value):
                    for item in self.game_tweaks.get_items(cat):
                        if item.tweak_type == TweakType.BOOL:
                            changed, val = imgui.checkbox(item.name, bool(item.value))
                            if changed:
                                item.set_value(val)
                                self.toast.show(f"{item.name}: {val}", duration=1.5)
                        elif item.tweak_type in (TweakType.FLOAT, TweakType.INT):
                            min_v = float(item.min_val) if item.min_val is not None else 0.0
                            max_v = float(item.max_val) if item.max_val is not None else 100.0
                            if item.tweak_type == TweakType.INT:
                                changed, val = imgui.slider_int(item.name, int(item.value), int(min_v), int(max_v))
                            else:
                                changed, val = imgui.slider_float(item.name, float(item.value), min_v, max_v, "%.2f")
                            if changed:
                                item.set_value(val)
                        elif item.tweak_type == TweakType.CHOICE:
                            current_idx = item.choices.index(item.value) if item.value in item.choices else 0
                            changed, new_idx = imgui.combo(item.name, current_idx, item.choices)
                            if changed and 0 <= new_idx < len(item.choices):
                                item.set_value(item.choices[new_idx])
                        elif item.tweak_type == TweakType.ACTION:
                            if imgui.button(f"Trigger {item.name}"):
                                item.trigger()
                        elif item.tweak_type == TweakType.WATCH:
                            val = item.read_watch()
                            imgui.text_disabled(f"{item.name}:")
                            imgui.same_line()
                            imgui.text_colored(imgui.ImVec4(0.35, 0.85, 1.0, 1.0), str(val))

            imgui.separator()
            imgui.text_disabled("[F3] Close Panel  |  [F9] Mouse Grab")
        imgui.end()

    def _render_toasts(self) -> None:
        """Renders active toast notifications as an elegant centered HUD banner."""
        if not self.toast:
            return
        active_toasts = self.toast.get_active()
        if not active_toasts:
            return

        imgui.set_next_window_pos(
            imgui.ImVec2(float(self.width) * 0.5, float(self.height) - 40.0),
            imgui.Cond_.always.value,
            imgui.ImVec2(0.5, 1.0),
        )
        imgui.set_next_window_bg_alpha(0.85)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_inputs.value
        )
        imgui.begin("##ToastOverlay", flags=flags)
        for msg, col in active_toasts:
            r = col[0] / 255.0 if col[0] > 1.0 else col[0]
            g = col[1] / 255.0 if col[1] > 1.0 else col[1]
            b = col[2] / 255.0 if col[2] > 1.0 else col[2]
            imgui.text_colored(imgui.ImVec4(r, g, b, 1.0), f"• {msg}")
        imgui.end()

    # --------------------------------------------------------------------------
    # Main Render Call
    # --------------------------------------------------------------------------

    def render(self, target_fbo: moderngl.Framebuffer | None = None) -> None:
        """Draws active ImGui debug panels and notifications."""
        # Attempt lazy initialization if renderer was not ready during __init__
        if (
            self.renderer is None
            and self.ctx is not None
            and pygame.display.get_surface() is not None
            and PygameRenderer is not None
        ):
            try:
                self.renderer = PygameRenderer()
            except Exception as e:
                if not self._warned_no_renderer:
                    log_warn("DebugMenu", f"PygameRenderer lazy initialization failed: {e}")
                    self._warned_no_renderer = True
                self.renderer = None

        if self.renderer is None:
            has_toasts = self.toast is not None and len(self.toast.get_active()) > 0
            if not self._warned_no_renderer and (self.visible or has_toasts):
                if PygameRenderer is None:
                    log_warn(
                        "DebugMenu",
                        "Cannot render debug menu because PyOpenGL is not installed. "
                        "Run 'pip install PyOpenGL' to enable Dear ImGui debug overlays.",
                    )
                else:
                    log_warn("DebugMenu", "PygameRenderer is unavailable or display surface is missing.")
                self._warned_no_renderer = True
            return

        has_toasts = self.toast is not None and len(self.toast.get_active()) > 0
        if not self.visible and not has_toasts:
            return

        # Explicitly bind target FBO or screen backbuffer so ImGui draws on the visible display
        if target_fbo is not None:
            target_fbo.use()
        elif self.ctx is not None and hasattr(self.ctx, "screen") and self.ctx.screen is not None:
            self.ctx.screen.use()

        # Dynamically synchronize with true display window resolution
        if pygame.display.get_init():
            try:
                win_w, win_h = pygame.display.get_window_size()
                if win_w > 0 and win_h > 0 and (win_w != self.width or win_h != self.height):
                    self.resize(win_w, win_h)
            except Exception:
                pass

        if self.imgui_ctx is not None and imgui is not None:
            imgui.set_current_context(self.imgui_ctx)
        self.io.display_size = imgui.ImVec2(float(self.width), float(self.height))
        self.renderer.process_inputs()
        imgui.new_frame()

        if self.f1_style == 1:
            self._render_f1_basic_hud()
        elif self.f1_style == 2:
            self._render_f1_expanded_profiler()

        if self.show_graphics:
            self._render_f2_graphics()

        if self.show_game_tweaks:
            self._render_f3_game_tweaks()

        self._render_toasts()

        # Render custom gameplay UI panels (Station menus, HUDs)
        for cb in self._custom_ui_callbacks:
            try:
                cb()
            except Exception:
                pass

        imgui.render()
        if gl is not None:
            try:
                while gl.glGetError() != 0:
                    pass
            except Exception:
                pass
        self.renderer.render(imgui.get_draw_data())

        if gl is not None:
            try:
                gl.glDisable(gl.GL_SCISSOR_TEST)
                gl.glBindVertexArray(0)
                gl.glUseProgram(0)
                gl.glViewport(0, 0, int(self.width), int(self.height))
            except Exception:
                pass

    def _on_window_resize(self, event: WindowResizeEvent) -> None:
        """Handles window resize event dispatched from windowing subsystem."""
        self.resize(event.width, event.height)

    def resize(self, width: int, height: int) -> None:
        """Handles window resize and synchronizes ImGui viewport dimensions."""
        if width <= 0 or height <= 0:
            return
        self.width = width
        self.height = height
        self.io.display_size = imgui.ImVec2(float(width), float(height))
        if self.renderer is not None:
            try:
                self.renderer._update_textures()
            except Exception:
                pass

    def destroy(self) -> None:
        """Shuts down ImGui context and backend renderer."""
        unsubscribe_event(WindowResizeEvent, self._on_window_resize)
        if self.renderer is not None:
            try:
                self.renderer.shutdown()
            except Exception:
                pass
            self.renderer = None
        if self.imgui_ctx is not None:
            try:
                imgui.destroy_context(self.imgui_ctx)
            except Exception:
                pass
            self.imgui_ctx = None
