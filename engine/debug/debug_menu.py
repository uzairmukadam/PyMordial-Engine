"""Native Dear ImGui Debug System for PyMordial Engine.

Provides independent, non-intrusive debug panels:
- F1: Performance & Resource Profiler (Multi-Style: Compact HUD -> Expanded Profiler -> Closed)
- F2: Graphics Pipeline & Renderer Tweaks
- F3: Gameplay Developer Tweaks & Inspector
- F9: Mouse Release/Capture for seamless panel interaction without locking gameplay
"""

from __future__ import annotations
import numpy as np
import pygame
import moderngl
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer
from engine.debug.monitor import SystemMonitor
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.debug.game_tweaks import GameTweaks, TweakType
from engine.debug.toast import DebugToast
from engine.input.input_manager import InputManager
from engine.gfx.quality_presets import GraphicsQuality

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
    ) -> None:
        self.ctx = ctx
        self.monitor = monitor
        self.engine_tweaks = engine_tweaks
        self.game_tweaks = game_tweaks
        self.toast = toast
        self.input_mgr = input_mgr

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
        if self.ctx is not None and pygame.display.get_surface() is not None:
            try:
                self.renderer = PygameRenderer()
            except Exception:
                self.renderer = None

        self._apply_theme()

    def _apply_theme(self) -> None:
        """Applies a modern, sleek AAA dark theme with translucent slate & cyan accents."""
        style = imgui.get_style()
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
        if self.renderer is not None:
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

        imgui.text_disabled("[F1] Expand  |  [F9] Mouse")
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
            imgui.text_disabled("[F1] Close  |  [F9] Mouse Grab")
        imgui.end()

    def _render_f2_graphics(self) -> None:
        """Renders independent F2 Graphics Pipeline & Renderer tweaks panel."""
        imgui.set_next_window_size(imgui.ImVec2(440.0, 480.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_pos(imgui.ImVec2(320.0, 20.0), imgui.Cond_.first_use_ever.value)
        expanded, p_open = imgui.begin("Graphics Pipeline & Renderer [F2]", p_open=True)
        if not p_open:
            self.show_graphics = False

        if expanded:
            et = self.engine_tweaks

            # 1. Quality Preset
            preset_names = ["low", "medium", "high", "ultra", "cinematic"]
            current_preset = et.quality_preset.value.lower()
            current_idx = preset_names.index(current_preset) if current_preset in preset_names else 2
            changed, new_idx = imgui.combo("Quality Preset", current_idx, [p.upper() for p in preset_names])
            if changed and new_idx != current_idx:
                et.set_quality_preset(GraphicsQuality(preset_names[new_idx]))
                self.toast.show(f"Quality Preset: {preset_names[new_idx].upper()}", duration=2.0)

            # 2. Tonemapper
            tonemap_options = ["ACES", "AgX", "Reinhard"]
            t_idx = tonemap_options.index(et.tonemap_mode) if et.tonemap_mode in tonemap_options else 0
            t_changed, new_t_idx = imgui.combo("Tonemapper", t_idx, tonemap_options)
            if t_changed and new_t_idx != t_idx:
                et.tonemap_mode = tonemap_options[new_t_idx]
                self.toast.show(f"Tonemapper: {et.tonemap_mode}", duration=2.0)

            # 3. G-Buffer Debug Mode
            gbuf_names = [m.name for m in GBufferDebugMode]
            gbuf_idx = gbuf_names.index(et.gbuffer_debug.name) if et.gbuffer_debug.name in gbuf_names else 0
            g_changed, new_g_idx = imgui.combo("G-Buffer Debug", gbuf_idx, gbuf_names)
            if g_changed and new_g_idx != gbuf_idx:
                et.gbuffer_debug = GBufferDebugMode[gbuf_names[new_g_idx]]
                self.toast.show(f"G-Buffer Mode: {et.gbuffer_debug.name}", duration=2.0)

            imgui.separator()
            imgui.text("Lighting & Shadows")

            # SSCS Contact Shadows
            sscs_changed, sscs_val = imgui.checkbox("Contact Shadows (SSCS)", et.sscs_enabled)
            if sscs_changed:
                et.sscs_enabled = sscs_val
                self.toast.show(f"Contact Shadows: {'ON' if sscs_val else 'OFF'}", duration=1.5)

            # Sun Angle (Azimuth)
            sun_changed, sun_val = imgui.slider_float("Sun Azimuth", et.sun_angle_deg, 0.0, 360.0, "%.1f deg")
            if sun_changed:
                et.sun_angle_deg = sun_val

            # Sun Elevation
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
        imgui.set_next_window_pos(imgui.ImVec2(540.0, 20.0), imgui.Cond_.first_use_ever.value)
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

    def render(self) -> None:
        """Draws active ImGui debug panels and notifications."""
        if self.renderer is None:
            return

        has_toasts = self.toast is not None and len(self.toast.get_active()) > 0
        if not self.visible and not has_toasts:
            return

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
        self.renderer.render(imgui.get_draw_data())

        if gl is not None:
            try:
                gl.glDisable(gl.GL_SCISSOR_TEST)
                gl.glBindVertexArray(0)
                gl.glUseProgram(0)
            except Exception:
                pass

    def resize(self, width: int, height: int) -> None:
        """Handles window resize."""
        self.width = width
        self.height = height
        self.io.display_size = imgui.ImVec2(float(width), float(height))

    def destroy(self) -> None:
        """Shuts down ImGui context and backend renderer."""
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
