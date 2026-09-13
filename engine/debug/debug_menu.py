"""Native Dear ImGui Debug System for PyMordial Engine.

Provides independent, non-intrusive debug panels:
- F1: Performance & Resource Profiler (Multi-Style: Compact HUD -> Expanded Profiler -> Closed)
- F2: Graphics Pipeline & Renderer Tweaks
- F3: Gameplay Developer Tweaks & Inspector
- F9: Mouse Release/Capture for seamless panel interaction without locking gameplay
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pygame
import moderngl
from imgui_bundle import imgui
try:
    from imgui_bundle.python_backends.pygame_backend import PygameRenderer
except (ImportError, ModuleNotFoundError):
    PygameRenderer = None
from engine.logging import log_warn
from engine.debug.monitor import SystemMonitor
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.debug.game_tweaks import GameTweaks, TweakType
from engine.debug.toast import DebugToast
from engine.input.input_manager import InputManager
from engine.gfx.quality_presets import GraphicsQuality
from engine.events import subscribe_event, unsubscribe_event, WindowResizeEvent

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


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
    ) -> None:
        self.ctx = ctx
        self.monitor = monitor
        self.engine_tweaks = engine_tweaks
        self.game_tweaks = game_tweaks
        self.toast = toast
        self.input_mgr = input_mgr
        self.config_filepath = Path(config_filepath) if config_filepath is not None else None

        self.width = screen_width
        self.height = screen_height

        # Independent panel states
        # f1_style: 0 = Off, 1 = Basic HUD overlay, 2 = Expanded Profiler window
        self.f1_style: int = 0
        self.show_graphics: bool = False
        self.show_game_tweaks: bool = False

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
        """Returns True if any debug panel is currently active."""
        return self.f1_style > 0 or self.show_graphics or self.show_game_tweaks

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

        active_ent = self.game_tweaks.get_value("Entities", "Active Count")
        if active_ent is not None:
            imgui.same_line()
            imgui.text_disabled(f"|  Entities: {active_ent}")

        imgui.separator()
        if imgui.small_button("Graphics Options [F2]"):
            self.show_graphics = not self.show_graphics
            if self.input_mgr and self.show_graphics and self.input_mgr.is_mouse_grabbed:
                self.input_mgr.set_mouse_grab(False)
                self.input_mgr._was_mouse_grabbed = False
        imgui.same_line()
        if imgui.small_button("Game Tweaks [F3]"):
            self.show_game_tweaks = not self.show_game_tweaks
            if self.input_mgr and self.show_game_tweaks and self.input_mgr.is_mouse_grabbed:
                self.input_mgr.set_mouse_grab(False)
                self.input_mgr._was_mouse_grabbed = False

        imgui.text_disabled("[F1] Expand HUD  |  [F2] Graphics  |  [F9] Free Mouse")
        imgui.end()

    def _render_f1_expanded_profiler(self) -> None:
        """Renders full performance profiler window with frame time graph and CPU stage latencies."""
        imgui.set_next_window_size(imgui.ImVec2(480.0, 440.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_pos(imgui.ImVec2(20.0, 20.0), imgui.Cond_.first_use_ever.value)
        expanded, p_open = imgui.begin("Performance & Resource Monitor [F1]", p_open=True)
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
            imgui.text(f"   Avg: {m.avg_fps:5.1f}  |  1% Low: {m.one_percent_low_fps:5.1f}")

            imgui.text(f"Frame Time: {m.avg_frame_time_ms:5.2f} ms  (Min: {m.min_frame_time_ms:0.1f}, Max: {m.max_frame_time_ms:0.1f})")

            imgui.separator()
            imgui.text("Frame Time History (ms)")
            ft_history = np.array(m.frame_times_ms, dtype=np.float32)
            if len(ft_history) > 0:
                overlay_str = f"{m.avg_frame_time_ms:.2f} ms (Target: 16.6ms)"
                imgui.plot_lines(
                    "##FT_Plot",
                    ft_history,
                    values_offset=0,
                    overlay_text=overlay_str,
                    scale_min=0.0,
                    scale_max=33.3,
                    graph_size=imgui.ImVec2(440.0, 75.0),
                )

            imgui.separator()
            imgui.text("CPU Stage Latencies (Microseconds)")
            if imgui.begin_table("Subsystems", 3, imgui.TableFlags_.borders_inner_h.value | imgui.TableFlags_.sizing_stretch_prop.value):
                imgui.table_setup_column("Stage")
                imgui.table_setup_column("Latency (us)")
                imgui.table_setup_column("Latency (ms)")
                imgui.table_headers_row()

                for stage, us in m.stage_averages_us.items():
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(stage)
                    imgui.table_next_column()
                    imgui.text(f"{us:6.1f} us")
                    imgui.table_next_column()
                    ms_val = us / 1000.0
                    col = (
                        imgui.ImVec4(0.29, 0.87, 0.50, 1.0)
                        if ms_val <= 4.0
                        else (imgui.ImVec4(0.98, 0.80, 0.18, 1.0) if ms_val <= 8.0 else imgui.ImVec4(0.94, 0.27, 0.27, 1.0))
                    )
                    imgui.text_colored(col, f"{ms_val:0.2f} ms")
                imgui.end_table()

            imgui.separator()
            active_ent = self.game_tweaks.get_value("Entities", "Active Count")
            if active_ent is not None:
                imgui.text(f"Active Entities: {active_ent}")

            if imgui.button("Open Graphics Options [F2]"):
                self.show_graphics = True
                if self.input_mgr and self.input_mgr.is_mouse_grabbed:
                    self.input_mgr.set_mouse_grab(False)
                    self.input_mgr._was_mouse_grabbed = False
            imgui.same_line()
            if imgui.button("Open Game Tweaks [F3]"):
                self.show_game_tweaks = True
                if self.input_mgr and self.input_mgr.is_mouse_grabbed:
                    self.input_mgr.set_mouse_grab(False)
                    self.input_mgr._was_mouse_grabbed = False

            imgui.text_disabled("[F1] Close  |  [F2] Graphics  |  [F9] Mouse Grab")
        imgui.end()

    def _render_f2_graphics(self) -> None:
        """Renders independent F2 Graphics Pipeline & Renderer tweaks panel."""
        imgui.set_next_window_size(imgui.ImVec2(440.0, 480.0), imgui.Cond_.first_use_ever.value)
        f2_x = 20.0 + (500.0 if self.width >= 1600 else 300.0)
        imgui.set_next_window_pos(imgui.ImVec2(f2_x, 20.0), imgui.Cond_.first_use_ever.value)
        expanded, p_open = imgui.begin("Graphics Pipeline & Renderer [F2]", p_open=True)
        if not p_open:
            self.show_graphics = False

        if expanded:
            et = self.engine_tweaks

            # Configuration Persistence
            if imgui.button("Save Configuration", imgui.ImVec2(160.0, 24.0)):
                saved = False
                if self.config_filepath is not None:
                    saved = self.engine_tweaks.save_to_file(self.config_filepath)
                if saved:
                    self.toast.show("Configuration saved to game directory", duration=2.5, color=(56, 189, 248))
                else:
                    self.toast.show("Configuration saved to memory", duration=2.0, color=(147, 197, 253))

            imgui.same_line()
            if imgui.button("Reset Defaults", imgui.ImVec2(130.0, 24.0)):
                self.engine_tweaks.set_quality_preset(GraphicsQuality.CUSTOM)
                self.engine_tweaks.exposure = 1.0
                self.engine_tweaks.tonemap_mode = "ACES"
                self.engine_tweaks.bloom_enabled = True
                self.engine_tweaks.bloom_intensity = 0.045
                self.engine_tweaks.chromatic_aberration_enabled = True
                self.engine_tweaks.chromatic_aberration_intensity = 0.005
                self.engine_tweaks.vignette_enabled = True
                self.engine_tweaks.vignette_intensity = 0.35
                self.engine_tweaks.film_grain_enabled = True
                self.engine_tweaks.film_grain_intensity = 0.04
                if self.config_filepath is not None:
                    self.engine_tweaks.save_to_file(self.config_filepath)
                self.toast.show("Reset graphics configuration to defaults", duration=2.0, color=(250, 204, 21))

            imgui.separator()

            # 1. Quality Preset
            preset_names = ["custom", "low", "medium", "high", "ultra", "cinematic"]
            current_preset = et.quality_preset.value.lower()
            current_idx = preset_names.index(current_preset) if current_preset in preset_names else 0
            changed, new_idx = imgui.combo("Quality Preset", current_idx, [p.upper() for p in preset_names])
            if changed and new_idx != current_idx:
                et.set_quality_preset(GraphicsQuality(preset_names[new_idx]))
                self.toast.show(f"Quality Preset: {preset_names[new_idx].upper()}", duration=2.0)

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

            imgui.separator()
            imgui.text("Global Illumination & Ambient Occlusion")

            # GI Mode (OFF, SSGI, LPV, HYBRID)
            gi_modes = ["OFF", "SSGI", "LPV", "HYBRID"]
            curr_gi_idx = gi_modes.index(et.gi_mode) if et.gi_mode in gi_modes else 3
            gi_changed, new_gi_idx = imgui.combo("GI Mode", curr_gi_idx, gi_modes)
            if gi_changed and new_gi_idx != curr_gi_idx:
                et.gi_mode = gi_modes[new_gi_idx]
                et.mark_custom()
                self.toast.show(f"GI Mode: {et.gi_mode}", duration=2.0)

            if et.gi_mode in ("SSGI", "HYBRID"):
                ssgi_st_changed, ssgi_st_val = imgui.slider_int("SSGI Steps", et.ssgi_steps, 4, 32)
                if ssgi_st_changed:
                    et.ssgi_steps = ssgi_st_val
                    et.mark_custom()

                ssgi_ray_changed, ssgi_ray_val = imgui.slider_int("SSGI Rays", getattr(et, "ssgi_rays", 8), 1, 16)
                if ssgi_ray_changed:
                    et.ssgi_rays = ssgi_ray_val
                    et.mark_custom()

                ssgi_dist_changed, ssgi_dist_val = imgui.slider_float("SSGI Distance", getattr(et, "ssgi_ray_distance", 3.0), 0.5, 8.0, "%.1fm")
                if ssgi_dist_changed:
                    et.ssgi_ray_distance = ssgi_dist_val
                    et.mark_custom()

                ssgi_thick_changed, ssgi_thick_val = imgui.slider_float("SSGI Thickness", getattr(et, "ssgi_thickness", 0.35), 0.05, 1.0, "%.2fm")
                if ssgi_thick_changed:
                    et.ssgi_thickness = ssgi_thick_val
                    et.mark_custom()

                ssgi_in_changed, ssgi_in_val = imgui.slider_float("SSGI Intensity", et.ssgi_intensity, 0.1, 3.0, "%.2f")
                if ssgi_in_changed:
                    et.ssgi_intensity = ssgi_in_val
                    et.mark_custom()

            if et.gi_mode in ("LPV", "HYBRID"):
                lpv_in_changed, lpv_in_val = imgui.slider_float("LPV Intensity", et.lpv_intensity, 0.1, 3.0, "%.2f")
                if lpv_in_changed:
                    et.lpv_intensity = lpv_in_val
                    et.mark_custom()

            # AO Mode (OFF, SSAO, HBAO, GTAO)
            ao_modes = ["OFF", "SSAO", "HBAO", "GTAO"]
            curr_ao_idx = ao_modes.index(et.ao_mode) if et.ao_mode in ao_modes else 3
            ao_changed, new_ao_idx = imgui.combo("AO Mode", curr_ao_idx, ao_modes)
            if ao_changed and new_ao_idx != curr_ao_idx:
                et.ao_mode = ao_modes[new_ao_idx]
                et.mark_custom()
                self.toast.show(f"AO Mode: {et.ao_mode}", duration=2.0)

            if et.ao_mode != "OFF":
                ao_in_changed, ao_in_val = imgui.slider_float("AO Intensity", et.ao_intensity, 0.1, 3.0, "%.2f")
                if ao_in_changed:
                    et.ao_intensity = ao_in_val
                    et.mark_custom()

                ao_rad_changed, ao_rad_val = imgui.slider_float("AO Radius", et.ao_radius, 0.1, 3.0, "%.2f")
                if ao_rad_changed:
                    et.ao_radius = ao_rad_val
                    et.mark_custom()

            imgui.separator()
            imgui.text("Reflections & Anti-Aliasing")

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
                ssr_st_changed, ssr_st_val = imgui.slider_int("SSR Steps", et.ssr_steps, 8, 64)
                if ssr_st_changed:
                    et.ssr_steps = ssr_st_val
                    et.mark_custom()

                ssr_tk_changed, ssr_tk_val = imgui.slider_float("SSR Thickness", et.ssr_thickness, 0.05, 1.5, "%.2fm")
                if ssr_tk_changed:
                    et.ssr_thickness = ssr_tk_val
                    et.mark_custom()

                ssr_mr_changed, ssr_mr_val = imgui.slider_float("SSR Max Roughness", et.ssr_max_roughness, 0.1, 1.0, "%.2f")
                if ssr_mr_changed:
                    et.ssr_max_roughness = ssr_mr_val
                    et.mark_custom()

            # Anti-Aliasing Mode (Mutually Exclusive: OFF, FXAA, SMAA 1x, SMAA 2x, SMAA 4x)
            aa_modes = ["OFF", "FXAA", "SMAA 1x", "SMAA 2x", "SMAA 4x"]
            aa_internal_modes = ["OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X"]
            curr_mode = getattr(et, "aa_mode", "OFF")
            curr_aa_idx = aa_internal_modes.index(curr_mode) if curr_mode in aa_internal_modes else 0

            aa_changed, aa_idx = imgui.combo("Anti-Aliasing", curr_aa_idx, aa_modes)
            if aa_changed:
                chosen_internal = aa_internal_modes[aa_idx]
                et.aa_mode = chosen_internal
                et.taa_enabled = False
                et.mark_custom()
                self.toast.show(f"Anti-Aliasing: {aa_modes[aa_idx]}", duration=1.5)

            # Dynamic Local Point Lights
            pl_changed, pl_val = imgui.checkbox("Dynamic Local Lights (SSBO 3)", et.point_lights_enabled)
            if pl_changed:
                et.point_lights_enabled = pl_val
                et.mark_custom()
                self.toast.show(f"Point Lights: {'ON' if pl_val else 'OFF'}", duration=1.5)

            imgui.separator()
            imgui.text("Physical Atmosphere & Day-Night Cycle")

            # Atmosphere Presets
            atmo_presets = ["EARTH_DAY", "EARTH_SUNSET", "EARTH_NIGHT", "ALIEN_CYAN_PURPLE", "ALIEN_CRIMSON_MARS", "CUSTOM"]
            preset_labels = ["Earth Day (Noon)", "Earth Sunset / Golden Hour", "Earth Night / Starlight", "Alien: Cyan & Purple", "Alien: Crimson Mars", "Custom"]
            cur_p_name = getattr(et, "atmo_preset", "EARTH_DAY").upper()
            cur_p_idx = atmo_presets.index(cur_p_name) if cur_p_name in atmo_presets else 5
            ap_c, ap_idx = imgui.combo("Atmosphere Preset", cur_p_idx, preset_labels)
            if ap_c and ap_idx != cur_p_idx:
                chosen_p = atmo_presets[ap_idx]
                if chosen_p != "CUSTOM":
                    et.apply_atmo_preset(chosen_p)
                else:
                    et.atmo_preset = "CUSTOM"
                self.toast.show(f"Atmosphere: {preset_labels[ap_idx]}", duration=2.0)

            # Time of Day (24h clock)
            tod_hours = int(et.time_of_day)
            tod_mins = int((et.time_of_day - tod_hours) * 60.0)
            time_str = f"{tod_hours:02d}:{tod_mins:02d}"
            tod_c, tod_v = imgui.slider_float(f"Time of Day ({time_str})", et.time_of_day, 0.0, 24.0, "%.2f h")
            if tod_c:
                et.time_of_day = tod_v

            # Day-Night Orbital Cycle Speed
            speed_labels = "Paused" if et.day_speed == 0.0 else f"{et.day_speed:.1f}x"
            ds_c, ds_v = imgui.slider_float(f"Cycle Speed ({speed_labels})", et.day_speed, 0.0, 10.0, "%.1fx")
            if ds_c:
                et.day_speed = ds_v

            # Quick pause / play button
            if et.day_speed == 0.0:
                if imgui.button("Play Day-Night Cycle (1.0x)"):
                    et.day_speed = 1.0
            else:
                if imgui.button("Pause Cycle"):
                    et.day_speed = 0.0

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

                turb_c, turb_v = imgui.slider_float("Atmospheric Turbidity (Haze)", et.turbidity, 1.0, 10.0, "%.2f")
                if turb_c:
                    et.turbidity = turb_v
                    et.atmo_preset = "CUSTOM"

                star_c, star_v = imgui.slider_float("Starfield Brightness", et.star_intensity, 0.0, 5.0, "%.2f")
                if star_c:
                    et.star_intensity = star_v
                    et.atmo_preset = "CUSTOM"

                imgui.tree_pop()

            imgui.separator()
            imgui.text("Froxel Volumetric Fog")

            fog_c, fog_v = imgui.checkbox("Froxel Volumetric Fog", et.volumetric_fog_enabled)
            if fog_c:
                et.volumetric_fog_enabled = fog_v
                et.mark_custom()
                self.toast.show(f"Volumetric Fog: {'ON' if fog_v else 'OFF'}", duration=1.5)

            if et.volumetric_fog_enabled:
                res_names = ["LOW (80x45x32)", "MEDIUM (120x68x48)", "HIGH (160x90x64)", "ULTRA (200x112x80)", "CINEMATIC (240x135x96)"]
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

            imgui.separator()
            imgui.text("Cinematic Camera Optics & Lens Effects (Phase 3)")

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
                mbi_c, mbi_v = imgui.slider_float("Shutter Blur Intensity", et.motion_blur_intensity, 0.1, 3.0, "%.2fx")
                if mbi_c:
                    et.motion_blur_intensity = mbi_v
                    et.mark_custom()

                mbs_c, mbs_v = imgui.slider_int("Motion Blur Taps", et.motion_blur_samples, 4, 32)
                if mbs_c:
                    et.motion_blur_samples = mbs_v
                    et.mark_custom()

            # Anamorphic Lens Flare
            lf_c, lf_v = imgui.checkbox("Anamorphic Lens Flare", getattr(et, "lens_flare_enabled", True))
            if lf_c:
                et.lens_flare_enabled = lf_v
                et.mark_custom()
                self.toast.show(f"Lens Flare: {'ON' if lf_v else 'OFF'}", duration=1.5)

            if getattr(et, "lens_flare_enabled", True):
                lft_c, lft_v = imgui.slider_float("Flare Threshold", et.lens_flare_threshold, 1.0, 5.0, "%.2f")
                if lft_c:
                    et.lens_flare_threshold = lft_v
                    et.mark_custom()

                lfs_c, lfs_v = imgui.slider_float("Streak Intensity", et.lens_flare_streak_intensity, 0.0, 2.0, "%.2f")
                if lfs_c:
                    et.lens_flare_streak_intensity = lfs_v
                    et.mark_custom()

                lfg_c, lfg_v = imgui.slider_float("Ghost Reflections", et.lens_flare_ghost_intensity, 0.0, 1.5, "%.2f")
                if lfg_c:
                    et.lens_flare_ghost_intensity = lfg_v
                    et.mark_custom()

            # HDR Bloom
            bl_c, bl_v = imgui.checkbox("HDR Bloom", getattr(et, "bloom_enabled", True))
            if bl_c:
                et.bloom_enabled = bl_v
                et.mark_custom()
            if getattr(et, "bloom_enabled", True):
                bli_c, bli_v = imgui.slider_float("Bloom Intensity", getattr(et, "bloom_intensity", 0.045), 0.0, 0.20, "%.3f")
                if bli_c:
                    et.bloom_intensity = bli_v
                    et.mark_custom()

            # Lens Imperfections (Chromatic Aberration, Vignette, Film Grain)
            ca_c, ca_v = imgui.checkbox("Chromatic Aberration", getattr(et, "chromatic_aberration_enabled", True))
            if ca_c:
                et.chromatic_aberration_enabled = ca_v
                et.mark_custom()
            if getattr(et, "chromatic_aberration_enabled", True):
                cai_c, cai_v = imgui.slider_float("Spectral Fringing", et.chromatic_aberration_intensity, 0.0, 0.02, "%.4f")
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

            fg_c, fg_v = imgui.checkbox("Filmic Grain (35mm)", getattr(et, "film_grain_enabled", True))
            if fg_c:
                et.film_grain_enabled = fg_v
                et.mark_custom()
            if getattr(et, "film_grain_enabled", True):
                fgi_c, fgi_v = imgui.slider_float("Grain ISO", et.film_grain_intensity, 0.0, 0.15, "%.3f")
                if fgi_c:
                    et.film_grain_intensity = fgi_v
                    et.mark_custom()

            imgui.separator()
            imgui.text("Micro-Geometry & Displacement")

            # POM
            pom_c, pom_v = imgui.checkbox("Parallax Occlusion Mapping (POM)", et.pom_enabled)
            if pom_c:
                et.pom_enabled = pom_v
                et.mark_custom()
            if et.pom_enabled:
                ps_c, ps_v = imgui.slider_float("POM Height Scale", et.pom_height_scale, 0.01, 0.20, "%.3f")
                if ps_c:
                    et.pom_height_scale = ps_v
                    et.mark_custom()
                pss_c, pss_v = imgui.checkbox("POM Sun Self-Shadowing", et.pom_self_shadow)
                if pss_c:
                    et.pom_self_shadow = pss_v
                    et.mark_custom()

            # Hardware Tessellation
            tess_c, tess_v = imgui.checkbox("Hardware GPU Tessellation", et.tess_enabled)
            if tess_c:
                et.tess_enabled = tess_v
                et.mark_custom()
            if et.tess_enabled:
                fc_c, fc_v = imgui.checkbox("GPU Frustum Culling", et.frustum_cull_enabled)
                if fc_c:
                    et.frustum_cull_enabled = fc_v
                    et.mark_custom()
                dnr_c, dnr_v = imgui.slider_float("Near Quality Radius (m)", et.disp_near_radius, 2.0, 20.0, "%.1f m")
                if dnr_c:
                    et.disp_near_radius = dnr_v
                    et.mark_custom()
                dmr_c, dmr_v = imgui.slider_float("Mid Radius Cutoff (m)", et.disp_mid_radius, 10.0, 50.0, "%.1f m")
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
                tds_c, tds_v = imgui.slider_float("Disp Depth Multiplier", et.tess_displacement_scale, 0.1, 3.0, "%.2fx")
                if tds_c:
                    et.tess_displacement_scale = tds_v
                    et.mark_custom()

            # SSDM
            ssdm_c, ssdm_v = imgui.checkbox("Screen-Space Displacement (SSDM)", et.ssdm_enabled)
            if ssdm_c:
                et.ssdm_enabled = ssdm_v
                et.mark_custom()
            if et.ssdm_enabled:
                sc_c, sc_v = imgui.slider_float("SSDM Scale", et.ssdm_scale, 0.01, 0.15, "%.3f")
                if sc_c:
                    et.ssdm_scale = sc_v
                    et.mark_custom()

            imgui.separator()
            imgui.text("Lighting & Shadows")

            # 1. Shadow Map Resolution (1024, 2048, 4096)
            res_options = ["1024", "2048", "4096"]
            curr_res_str = str(et.shadow_resolution)
            res_idx = res_options.index(curr_res_str) if curr_res_str in res_options else 1
            res_changed, new_res_idx = imgui.combo("Shadow Resolution", res_idx, res_options)
            if res_changed and new_res_idx != res_idx:
                et.shadow_resolution = int(res_options[new_res_idx])
                et.mark_custom()
                self.toast.show(f"Shadow Resolution: {et.shadow_resolution}x{et.shadow_resolution}", duration=2.0)

            # 2. Shadow Mode (HARD, PCF, PCSS)
            shadow_modes = ["HARD", "PCF", "PCSS"]
            s_idx = shadow_modes.index(et.shadow_mode) if et.shadow_mode in shadow_modes else 2
            sm_changed, new_sm_idx = imgui.combo("Shadow Mode", s_idx, shadow_modes)
            if sm_changed and new_sm_idx != s_idx:
                et.shadow_mode = shadow_modes[new_sm_idx]
                et.mark_custom()
                self.toast.show(f"Shadow Mode: {et.shadow_mode}", duration=2.0)

            # 3. Shadow Softness (PCF / PCSS)
            if et.shadow_mode in ("PCF", "PCSS"):
                soft_changed, soft_val = imgui.slider_float("Shadow Softness", et.shadow_softness, 0.2, 3.0, "%.2f")
                if soft_changed:
                    et.shadow_softness = soft_val
                    et.mark_custom()

            # 4. Shadow Depth Bias
            bias_changed, bias_val = imgui.slider_float("Shadow Depth Bias", et.shadow_bias, 0.0000, 0.0050, "%.5f")
            if bias_changed:
                et.shadow_bias = bias_val
                et.mark_custom()

            # 4b. Shadow Normal Bias
            nb_changed, nb_val = imgui.slider_float("Shadow Normal Bias", et.shadow_normal_bias, 0.0000, 0.0050, "%.5f")
            if nb_changed:
                et.shadow_normal_bias = nb_val
                et.mark_custom()

            # 4c. Shadow Alignment Sliders (Manual Offset Adjustment)
            off_x_changed, off_x_val = imgui.slider_float("Shadow Align X", et.shadow_offset_x, -0.050, 0.050, "%.4f")
            if off_x_changed:
                et.shadow_offset_x = off_x_val
                et.mark_custom()

            off_y_changed, off_y_val = imgui.slider_float("Shadow Align Y", et.shadow_offset_y, -0.050, 0.050, "%.4f")
            if off_y_changed:
                et.shadow_offset_y = off_y_val
                et.mark_custom()

            # 5. Shadow Draw Distance Slider
            sd_changed, sd_val = imgui.slider_float("Shadow Draw Distance", et.shadow_distance, 50.0, 2000.0, "%.0f m")
            if sd_changed:
                et.shadow_distance = sd_val
                et.mark_custom()

            # 5b. Shadow Cascades (1 to 4)
            casc_options = ["1 Cascade", "2 Cascades", "3 Cascades", "4 Cascades"]
            curr_casc_idx = max(0, min(3, et.csm_cascades - 1))
            casc_changed, new_casc_idx = imgui.combo("Shadow Cascades", curr_casc_idx, casc_options)
            if casc_changed and new_casc_idx != curr_casc_idx:
                et.csm_cascades = new_casc_idx + 1
                et.mark_custom()
                self.toast.show(f"Shadow Cascades: {et.csm_cascades}", duration=2.0)

            # 6. Sun Angle (Azimuth)
            sun_changed, sun_val = imgui.slider_float("Sun Azimuth", et.sun_angle_deg, 0.0, 360.0, "%.1f deg")
            if sun_changed:
                et.sun_angle_deg = sun_val

            # 7. Sun Elevation
            el_changed, el_val = imgui.slider_float("Sun Elevation", et.sun_elevation_deg, 0.0, 90.0, "%.1f deg")
            if el_changed:
                et.sun_elevation_deg = el_val

            imgui.separator()
            imgui.text("Debug Visualizers & Gizmos")

            # Mesh Wireframe
            wire_changed, wire_val = imgui.checkbox("Mesh Wireframe", et.show_wireframe)
            if wire_changed:
                et.show_wireframe = wire_val
                self.toast.show(f"Mesh Wireframe: {'ON' if wire_val else 'OFF'}", duration=1.5)

            # Physics Colliders
            phys_changed, phys_val = imgui.checkbox("Physics Gizmos / Colliders", et.show_physics_colliders)
            if phys_changed:
                et.show_physics_colliders = phys_val
                self.toast.show(f"Physics Gizmos: {'ON' if phys_val else 'OFF'}", duration=1.5)

            # Sun Ray Gizmo
            sun_ray_changed, sun_ray_val = imgui.checkbox("Sun Ray Gizmo", et.show_sun_ray)
            if sun_ray_changed:
                et.show_sun_ray = sun_ray_val

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
