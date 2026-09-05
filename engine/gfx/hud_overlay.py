"""On-screen telemetry and controls HUD overlay for PyMordial Engine.

Renders a sleek, modern glassmorphic HUD panel containing real-time performance
metrics, camera/sun coordinates, quality presets, and keybinding shortcuts
directly over the ModernGL framebuffer using zero-VBO screen-space quad rendering.
"""

from __future__ import annotations
import time
from typing import Optional
import pygame
import moderngl


OVERLAY_VERT_GLSL = """#version 450 core
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

    // Convert pixel coordinates (top-down) to NDC [-1, 1] (bottom-up)
    vec2 ndc = (pos / u_ScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_UV = uv;
}
"""

OVERLAY_FRAG_GLSL = """#version 450 core
in vec2 v_UV;
out vec4 out_FragColor;
uniform sampler2D u_Texture;

void main() {
    out_FragColor = texture(u_Texture, v_UV);
}
"""


class HudOverlay:
    """Renders a sleek HUD telemetry and controls overlay on the screen."""

    def __init__(
        self,
        ctx: moderngl.Context,
        screen_width: int,
        screen_height: int,
        panel_width: int = 350,
        panel_height: int = 420,
    ) -> None:
        self.ctx = ctx
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_width = panel_width
        self.panel_height = panel_height

        if not pygame.font.get_init():
            pygame.font.init()

        # Crisp font selection
        font_names = ["Consolas", "Courier New", "Lucida Console", "Segoe UI"]
        self.font = None
        for name in font_names:
            try:
                self.font = pygame.font.SysFont(name, 12, bold=False)
                self.font_bold = pygame.font.SysFont(name, 12, bold=True)
                self.font_title = pygame.font.SysFont(name, 14, bold=True)
                break
            except Exception:
                continue

        if self.font is None:
            self.font = pygame.font.Font(None, 16)
            self.font_bold = self.font
            self.font_title = pygame.font.Font(None, 20)

        # Pre-allocated Pygame RGBA drawing surface
        self.surface = pygame.Surface((self.panel_width, self.panel_height), pygame.SRCALPHA)

        # GPU Texture & Shader Program
        self.texture = self.ctx.texture((self.panel_width, self.panel_height), 4)
        self.prog = self.ctx.program(
            vertex_shader=OVERLAY_VERT_GLSL,
            fragment_shader=OVERLAY_FRAG_GLSL,
        )
        self.vao = self.ctx.vertex_array(self.prog, [])

        # Cached uniform locations
        self.prog["u_Texture"].value = 0

    def update_screen_size(self, width: int, height: int) -> None:
        self.screen_width = width
        self.screen_height = height

    def render(
        self,
        fps: float,
        frame_time_ms: float,
        preset_name: str,
        entity_count: int,
        cam_dist: float,
        cam_yaw: float,
        cam_pitch: float,
        sun_angle: float,
        tonemap_mode: str,
        status_message: str = "",
        status_time: float = 0.0,
        panel_x: int = 16,
        panel_y: int = 16,
    ) -> None:
        """Draws the HUD panel to the screen."""
        self.surface.fill((0, 0, 0, 0))

        # Panel styling
        bg_rect = pygame.Rect(0, 0, self.panel_width, self.panel_height)
        # Translucent dark glass container
        pygame.draw.rect(self.surface, (12, 18, 28, 220), bg_rect, border_radius=8)
        # Subtle glowing cyan/blue border
        pygame.draw.rect(self.surface, (56, 189, 248, 140), bg_rect, width=1, border_radius=8)

        y_offset = 12
        line_height = 17

        # 1. Header Title
        title_surf = self.font_title.render("PYMORDIAL ENGINE — PHASE 2", True, (56, 189, 248))
        self.surface.blit(title_surf, (14, y_offset))
        y_offset += 24

        # 2. Performance Telemetry
        fps_color = (74, 222, 128) if fps >= 55.0 else ((250, 204, 21) if fps >= 30.0 else (248, 113, 113))
        fps_text = f"FPS: {fps:5.1f}   ({frame_time_ms:5.2f} ms)"
        self.surface.blit(self.font_bold.render(fps_text, True, fps_color), (14, y_offset))
        y_offset += line_height

        preset_text = f"Preset: {preset_name.upper():<9} | Reversed-Z: ACTIVE"
        self.surface.blit(self.font.render(preset_text, True, (241, 245, 249)), (14, y_offset))
        y_offset += line_height

        ent_text = f"Entities: {entity_count}  (Direct C-Memory SSBO)"
        self.surface.blit(self.font.render(ent_text, True, (226, 232, 240)), (14, y_offset))
        y_offset += line_height

        cam_text = f"Camera: Dist {cam_dist:4.1f}m | Yaw {cam_yaw:4.0f}° | Pitch {cam_pitch:4.0f}°"
        self.surface.blit(self.font.render(cam_text, True, (148, 163, 184)), (14, y_offset))
        y_offset += line_height

        sun_text = f"Sun: {sun_angle:4.2f} rad   | Tonemap: {tonemap_mode}"
        self.surface.blit(self.font.render(sun_text, True, (148, 163, 184)), (14, y_offset))
        y_offset += line_height + 4

        # Divider
        pygame.draw.line(
            self.surface,
            (51, 65, 85, 180),
            (14, y_offset),
            (self.panel_width - 14, y_offset),
            1,
        )
        y_offset += 8

        # 3. Controls Cheatsheet
        controls_header = self.font_bold.render("CONTROLS & SHORTCUTS", True, (203, 213, 225))
        self.surface.blit(controls_header, (14, y_offset))
        y_offset += line_height

        shortcuts = [
            ("[L-Drag / WASD]", "Orbit 3D Camera"),
            ("[Wheel / Q, E]", "Zoom Camera In / Out"),
            ("[SPACE]", "Spawn 8 PBR Spheres"),
            ("[1 .. 5]", "Quality Presets (Low -> Cine)"),
            ("[T]", "Toggle Tonemap Mode"),
            ("[L]", "Rotate Sun Direction"),
            ("[P] / [R]", "PIE Snapshot / Restore"),
            ("[C]", "Clear Dynamic Spheres"),
            ("[ESC]", "Exit Engine"),
        ]

        for key, desc in shortcuts:
            k_surf = self.font_bold.render(f"{key:<18}", True, (56, 189, 248))
            d_surf = self.font.render(desc, True, (148, 163, 184))
            self.surface.blit(k_surf, (14, y_offset))
            self.surface.blit(d_surf, (150, y_offset))
            y_offset += line_height

        # 4. Status Notification Bar
        now = time.perf_counter()
        if status_message and (now - status_time < 5.0):
            y_offset += 6
            pygame.draw.line(
                self.surface,
                (51, 65, 85, 180),
                (14, y_offset),
                (self.panel_width - 14, y_offset),
                1,
            )
            y_offset += 6
            msg_surf = self.font_bold.render(status_message, True, (250, 204, 21))
            self.surface.blit(msg_surf, (14, y_offset))

        # Upload surface to ModernGL texture
        raw_bytes = pygame.image.tobytes(self.surface, "RGBA")
        self.texture.write(raw_bytes)

        # Render quad overlay
        self.prog["u_ScreenSize"].value = (float(self.screen_width), float(self.screen_height))
        self.prog["u_Rect"].value = (
            float(panel_x),
            float(panel_y),
            float(self.panel_width),
            float(self.panel_height),
        )

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # Restore defaults
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def destroy(self) -> None:
        if self.texture:
            self.texture.release()
        if self.vao:
            self.vao.release()
        if self.prog:
            self.prog.release()
