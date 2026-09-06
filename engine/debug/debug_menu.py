"""Glassmorphic 3-Tab Debug Menu UI for PyMordial Engine.

Navigable via Keyboard, Mouse, and Gamepad. Renders:
- Tab 1: System Monitor & Performance Profiler
- Tab 2: Engine Graphics, G-Buffer & Pipeline Tweaks
- Tab 3: Game-Specific Developer Tweaks & Inspector
"""

from __future__ import annotations
import pygame
import moderngl
from engine.debug.monitor import SystemMonitor
from engine.debug.engine_tweaks import EngineTweaks
from engine.debug.game_tweaks import GameTweaks, TweakType
from engine.debug.toast import DebugToast
from engine.input.input_manager import InputManager

DEBUG_MENU_VERT = """#version 450 core
uniform vec2 u_ScreenSize;
uniform vec4 u_Rect; // [x, y, w, h] in screen pixel coordinates
out vec2 v_UV;

void main() {
    vec2 pos = vec2(0.0);
    vec2 uv = vec2(0.0);
    if (gl_VertexID == 0) { pos = u_Rect.xy; uv = vec2(0.0, 0.0); }
    else if (gl_VertexID == 1) { pos = vec2(u_Rect.x + u_Rect.z, u_Rect.y); uv = vec2(1.0, 0.0); }
    else if (gl_VertexID == 2) { pos = vec2(u_Rect.x, u_Rect.y + u_Rect.w); uv = vec2(0.0, 1.0); }
    else if (gl_VertexID == 3) { pos = u_Rect.xy + u_Rect.zw; uv = vec2(1.0, 1.0); }

    vec2 ndc = (pos / u_ScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_UV = uv;
}
"""

DEBUG_MENU_FRAG = """#version 450 core
in vec2 v_UV;
out vec4 out_FragColor;
uniform sampler2D u_Texture;

void main() {
    out_FragColor = texture(u_Texture, v_UV);
}
"""


class DebugMenu:
    """Coordinates the 3-tier debug overlay UI."""

    __slots__ = (
        "ctx",
        "monitor",
        "engine_tweaks",
        "game_tweaks",
        "toast",
        "input_mgr",
        "visible",
        "active_tab",
        "selected_item_idx",
        "width",
        "height",
        "panel_w",
        "panel_h",
        "surface",
        "texture",
        "prog",
        "vao",
        "font",
        "font_bold",
        "font_title",
        "_u_screen_size",
        "_u_rect",
    )

    TABS = ["1. System Monitor", "2. Engine & Graphics", "3. Game Tweaks"]

    def __init__(
        self,
        ctx: moderngl.Context,
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

        self.visible = False
        self.active_tab = 0
        self.selected_item_idx = 0

        self.width = screen_width
        self.height = screen_height
        self.panel_w = 460
        self.panel_h = 580

        if not pygame.font.get_init():
            pygame.font.init()

        self.font = pygame.font.SysFont("Consolas", 12)
        self.font_bold = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 15, bold=True)

        self.surface = pygame.Surface((self.panel_w, self.panel_h), pygame.SRCALPHA)
        self.texture = self.ctx.texture((self.panel_w, self.panel_h), 4)

        self.prog = self.ctx.program(vertex_shader=DEBUG_MENU_VERT, fragment_shader=DEBUG_MENU_FRAG)
        self.vao = self.ctx.vertex_array(self.prog, [])

        self._u_screen_size = self.prog.get("u_ScreenSize", None)
        self._u_rect = self.prog.get("u_Rect", None)

    def toggle(self) -> bool:
        """Toggles debug menu visibility."""
        self.visible = not self.visible
        self.input_mgr.set_debug_mode(self.visible)
        return self.visible

    def handle_input(self) -> None:
        """Handles menu navigation from KBM and Gamepad."""
        if not self.visible:
            return

        # Close on B button or Escape
        if self.input_mgr.is_action_pressed("debug_close_b"):
            self.toggle()
            return

        # Tab switching (Q/E or LB/RB)
        if self.input_mgr.is_action_pressed("debug_tab_prev"):
            self.active_tab = (self.active_tab - 1) % len(self.TABS)
            self.selected_item_idx = 0
        elif self.input_mgr.is_action_pressed("debug_tab_next"):
            self.active_tab = (self.active_tab + 1) % len(self.TABS)
            self.selected_item_idx = 0

        # Nav up / down
        if self.input_mgr.is_action_pressed("debug_nav_up"):
            self.selected_item_idx = max(0, self.selected_item_idx - 1)
        elif self.input_mgr.is_action_pressed("debug_nav_down"):
            self.selected_item_idx += 1

        # Tab 2: Engine Tweaks adjustments
        if self.active_tab == 1:
            total_items = 7
            self.selected_item_idx = max(0, min(total_items - 1, self.selected_item_idx))

            nav_act = self.input_mgr.is_action_pressed("debug_nav_activate")
            nav_l = self.input_mgr.is_action_pressed("debug_nav_left")
            nav_r = self.input_mgr.is_action_pressed("debug_nav_right")

            if nav_act:
                if self.selected_item_idx == 0:
                    p = self.engine_tweaks.cycle_quality_preset()
                    self.toast.show(f"Quality Preset: {p.value.upper()}")
                elif self.selected_item_idx == 1:
                    m = self.engine_tweaks.cycle_tonemapper()
                    self.toast.show(f"Tonemapper: {m}")
                elif self.selected_item_idx == 2:
                    g = self.engine_tweaks.cycle_gbuffer_debug()
                    self.toast.show(f"G-Buffer Mode: {g.name}")
                elif self.selected_item_idx == 3:
                    self.engine_tweaks.toggle_wireframe()
                    state_str = "ON" if self.engine_tweaks.show_wireframe else "OFF"
                    self.toast.show(f"Mesh Wireframe: {state_str}")
                elif self.selected_item_idx == 4:
                    self.engine_tweaks.toggle_physics_colliders()
                    state_str = "ON" if self.engine_tweaks.show_physics_colliders else "OFF"
                    self.toast.show(f"Physics Gizmos: {state_str}")
                elif self.selected_item_idx == 5:
                    self.engine_tweaks.sun_angle_deg = (self.engine_tweaks.sun_angle_deg + 15.0) % 360.0
                    self.toast.show(f"Sun Angle: {self.engine_tweaks.sun_angle_deg:0.1f}°")
                elif self.selected_item_idx == 6:
                    self.engine_tweaks.sscs_enabled = not self.engine_tweaks.sscs_enabled
                    state_str = "ON" if self.engine_tweaks.sscs_enabled else "OFF"
                    self.toast.show(f"Contact Shadows: {state_str}")
            elif nav_l:
                if self.selected_item_idx == 0:
                    p = self.engine_tweaks.cycle_quality_preset()
                    self.toast.show(f"Quality Preset: {p.value.upper()}")
                elif self.selected_item_idx == 1:
                    m = self.engine_tweaks.cycle_tonemapper()
                    self.toast.show(f"Tonemapper: {m}")
                elif self.selected_item_idx == 2:
                    g = self.engine_tweaks.cycle_gbuffer_debug()
                    self.toast.show(f"G-Buffer Mode: {g.name}")
                elif self.selected_item_idx == 3:
                    self.engine_tweaks.show_wireframe = False
                    self.toast.show("Mesh Wireframe: OFF")
                elif self.selected_item_idx == 4:
                    self.engine_tweaks.show_physics_colliders = False
                    self.toast.show("Physics Gizmos: OFF")
                elif self.selected_item_idx == 5:
                    self.engine_tweaks.sun_angle_deg = (self.engine_tweaks.sun_angle_deg - 15.0) % 360.0
                    self.toast.show(f"Sun Angle: {self.engine_tweaks.sun_angle_deg:0.1f}°")
                elif self.selected_item_idx == 6:
                    self.engine_tweaks.sscs_enabled = False
                    self.toast.show("Contact Shadows: OFF")
            elif nav_r:
                if self.selected_item_idx == 0:
                    p = self.engine_tweaks.cycle_quality_preset()
                    self.toast.show(f"Quality Preset: {p.value.upper()}")
                elif self.selected_item_idx == 1:
                    m = self.engine_tweaks.cycle_tonemapper()
                    self.toast.show(f"Tonemapper: {m}")
                elif self.selected_item_idx == 2:
                    g = self.engine_tweaks.cycle_gbuffer_debug()
                    self.toast.show(f"G-Buffer Mode: {g.name}")
                elif self.selected_item_idx == 3:
                    self.engine_tweaks.show_wireframe = True
                    self.toast.show("Mesh Wireframe: ON")
                elif self.selected_item_idx == 4:
                    self.engine_tweaks.show_physics_colliders = True
                    self.toast.show("Physics Gizmos: ON")
                elif self.selected_item_idx == 5:
                    self.engine_tweaks.sun_angle_deg = (self.engine_tweaks.sun_angle_deg + 15.0) % 360.0
                    self.toast.show(f"Sun Angle: {self.engine_tweaks.sun_angle_deg:0.1f}°")
                elif self.selected_item_idx == 6:
                    self.engine_tweaks.sscs_enabled = True
                    self.toast.show("Contact Shadows: ON")

        # Tab 3: Game Tweaks adjustments
        elif self.active_tab == 2:
            all_items = []
            for cat in self.game_tweaks.get_categories():
                for it in self.game_tweaks.get_items(cat):
                    all_items.append(it)

            if all_items:
                self.selected_item_idx = max(0, min(self.selected_item_idx, len(all_items) - 1))
                item = all_items[self.selected_item_idx]

                if self.input_mgr.is_action_pressed("debug_nav_activate"):
                    if item.tweak_type == TweakType.BOOL:
                        item.set_value(not item.value)
                        self.toast.show(f"{item.name}: {item.value}")
                    elif item.tweak_type == TweakType.ACTION:
                        item.trigger()
                        self.toast.show(f"Action '{item.name}' Triggered")
                elif self.input_mgr.is_action_pressed("debug_nav_left"):
                    if item.tweak_type == TweakType.BOOL:
                        item.set_value(False)
                        self.toast.show(f"{item.name}: {item.value}")
                    elif item.tweak_type in (TweakType.FLOAT, TweakType.INT):
                        step = item.step if item.step is not None else 1.0
                        item.set_value(item.value - step)
                elif self.input_mgr.is_action_pressed("debug_nav_right"):
                    if item.tweak_type == TweakType.BOOL:
                        item.set_value(True)
                        self.toast.show(f"{item.name}: {item.value}")
                    elif item.tweak_type in (TweakType.FLOAT, TweakType.INT):
                        step = item.step if item.step is not None else 1.0
                        item.set_value(item.value + step)

    def render(self) -> None:
        """Draws the debug menu overlay to screen if visible."""
        if not self.visible:
            return

        s = self.surface
        s.fill((12, 18, 28, 225))  # Glassmorphic dark slate

        # Outer border
        pygame.draw.rect(s, (56, 189, 248, 240), (0, 0, self.panel_w, self.panel_h), 2, border_radius=8)

        # Header Title
        title_surf = self.font_title.render("PYMORDIAL ENGINE — DEBUG SYSTEM", True, (255, 255, 255))
        s.blit(title_surf, (16, 14))

        # Tabs Header
        tab_x = 16
        for i, tab_name in enumerate(self.TABS):
            is_active = (i == self.active_tab)
            bg_col = (30, 58, 138, 255) if is_active else (20, 30, 45, 180)
            text_col = (255, 255, 255) if is_active else (148, 163, 184)
            tab_surf = self.font_bold.render(tab_name, True, text_col)
            tw, th = tab_surf.get_width() + 14, tab_surf.get_height() + 8
            pygame.draw.rect(s, bg_col, (tab_x, 42, tw, th), border_radius=4)
            if is_active:
                pygame.draw.rect(s, (56, 189, 248), (tab_x, 42, tw, th), 1, border_radius=4)
            s.blit(tab_surf, (tab_x + 7, 46))
            tab_x += tw + 6

        # Content area
        pygame.draw.line(s, (51, 65, 85), (16, 76), (self.panel_w - 16, 76), 1)
        y = 88

        # --- TAB 1: SYSTEM MONITOR ---
        if self.active_tab == 0:
            m = self.monitor
            lines = [
                f"FPS:              {m.fps:5.1f}  (Avg: {m.avg_fps:5.1f})",
                f"1% Low FPS:       {m.one_percent_low_fps:5.1f}",
                f"Frame Time:       {m.avg_frame_time_ms:5.2f} ms  (Min: {m.min_frame_time_ms:0.1f}, Max: {m.max_frame_time_ms:0.1f})",
                "",
                "--- CPU Stage Latencies (Microseconds) ---",
            ]
            for stage, us in m.stage_averages_us.items():
                lines.append(f"  {stage:<16} : {us:6.1f} us  ({us/1000.0:0.2f} ms)")

            lines.extend([
                "",
                "--- Mini Frame-Time History ---",
            ])
            for line in lines:
                txt = self.font.render(line, True, (226, 232, 240))
                s.blit(txt, (20, y))
                y += 18

            # Draw sparkline histogram
            hist_x = 24
            hist_y = y + 10
            hist_w = self.panel_w - 48
            hist_h = 60
            pygame.draw.rect(s, (15, 23, 42), (hist_x, hist_y, hist_w, hist_h), border_radius=4)
            pygame.draw.rect(s, (51, 65, 85), (hist_x, hist_y, hist_w, hist_h), 1, border_radius=4)

            # Draw 16.6ms target line
            target_y = hist_y + hist_h - int((16.67 / 33.33) * hist_h)
            pygame.draw.line(s, (74, 222, 128, 120), (hist_x, target_y), (hist_x + hist_w, target_y), 1)

            step = hist_w / len(m.frame_times_ms)
            for i, ft in enumerate(m.frame_times_ms):
                bar_h = min(hist_h, int((ft / 33.33) * hist_h))
                bx = int(hist_x + i * step)
                by = hist_y + hist_h - bar_h
                col = (74, 222, 128) if ft <= 17.0 else (239, 68, 68)
                pygame.draw.line(s, col, (bx, hist_y + hist_h), (bx, by), 1)

        # --- TAB 2: ENGINE & GRAPHICS TWEAKS ---
        elif self.active_tab == 1:
            et = self.engine_tweaks
            items = [
                f"Quality Preset:     {et.quality_preset.value.upper()}",
                f"Tonemapper Mode:    {et.tonemap_mode}",
                f"G-Buffer Debug RT:  {et.gbuffer_debug.name}",
                f"Mesh Wireframes:    {'[ON]' if et.show_wireframe else '[OFF]'}",
                f"Physics Colliders:  {'[ON]' if et.show_physics_colliders else '[OFF]'}",
                f"Sun Direction:      {et.sun_angle_deg:0.1f} deg",
                f"Contact Shadows:    {'[ON]' if et.sscs_enabled else '[OFF]'}",
            ]
            for idx, item_str in enumerate(items):
                is_sel = (idx == self.selected_item_idx)
                prefix = "> " if is_sel else "  "
                col = (56, 189, 248) if is_sel else (203, 213, 225)
                txt = self.font_bold.render(prefix + item_str, True, col)
                if is_sel:
                    pygame.draw.rect(s, (30, 58, 138, 120), (18, y - 2, self.panel_w - 36, 20), border_radius=4)
                s.blit(txt, (20, y))
                y += 24

        # --- TAB 3: GAME-SPECIFIC DEVELOPER TWEAKS ---
        elif self.active_tab == 2:
            all_items = []
            for cat in self.game_tweaks.get_categories():
                all_items.append((True, cat))  # Category header
                for it in self.game_tweaks.get_items(cat):
                    all_items.append((False, it))

            item_counter = 0
            for is_header, obj in all_items:
                if is_header:
                    cat_txt = self.font_bold.render(f"[{obj}]", True, (250, 204, 21))
                    s.blit(cat_txt, (20, y))
                    y += 20
                else:
                    it = obj
                    is_sel = (item_counter == self.selected_item_idx)
                    val_str = str(it.read_watch() if it.tweak_type == TweakType.WATCH else it.value)
                    row_str = f"{it.name:<20} : {val_str}"
                    prefix = "> " if is_sel else "  "
                    col = (56, 189, 248) if is_sel else (226, 232, 240)
                    if is_sel:
                        pygame.draw.rect(s, (30, 58, 138, 120), (18, y - 2, self.panel_w - 36, 20), border_radius=4)
                    txt = self.font.render(prefix + row_str, True, col)
                    s.blit(txt, (20, y))
                    y += 22
                    item_counter += 1

        footer_y = self.panel_h - 26

        # Draw active toast notification if any
        if self.toast:
            active_toasts = self.toast.get_active()
            if active_toasts:
                t_msg, t_col = active_toasts[-1]
                t_surf = self.font_bold.render(f"Notification: {t_msg}", True, t_col)
                s.blit(t_surf, (20, footer_y - 24))

        # Draw Footer with navigation hints
        pygame.draw.line(s, (51, 65, 85), (16, footer_y - 6), (self.panel_w - 16, footer_y - 6), 1)
        hint = "[Q/E or LB/RB] Tabs | [Arrows/DPad] Navigate | [Enter/A] Toggle | [F1/B] Close"
        hint_surf = self.font.render(hint, True, (148, 163, 184))
        s.blit(hint_surf, (20, footer_y))

        # Upload surface to ModernGL texture
        raw = pygame.image.tobytes(s, "RGBA")
        self.texture.write(raw)

        # Draw to screen with alpha blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        if self._u_screen_size is not None:
            self._u_screen_size.value = (float(self.width), float(self.height))
        if self._u_rect is not None:
            self._u_rect.value = (20.0, 20.0, float(self.panel_w), float(self.panel_h))

        self.texture.use(location=0)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.disable(moderngl.BLEND)

    def resize(self, width: int, height: int) -> None:
        """Handles window resize."""
        self.width = width
        self.height = height

    def destroy(self) -> None:
        """Releases GPU texture and programs."""
        if self.texture is not None:
            self.texture.release()
        if self.vao is not None:
            self.vao.release()
        if self.prog is not None:
            self.prog.release()
