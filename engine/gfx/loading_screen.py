"""Hardware-accelerated cinematic loading screen and background asset streaming controller."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import pygame
import moderngl

from engine.events import subscribe_event, unsubscribe_event, WindowResizeEvent

if TYPE_CHECKING:
    from engine.gfx.context import RenderContext
    from engine.gfx.texture_atlas import TextureArrayAtlas
    from engine.core.async_loader import BackgroundAssetLoader

SHADER_DIR = Path(__file__).resolve().parent.parent.parent / "shaders"


class LoadingScreen:
    """Renders a minimalist, fully black cinematic loading screen with a bottom-right corner animated icon."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "prog",
        "vao",
        "u_resolution",
        "u_time",
        "u_progress",
        "u_fade_alpha",
        "u_text_texture",
        "text_texture",
        "_text_surface",
        "_font_status",
        "_font_pct",
        "_last_rendered_text_key",
        "current_progress",
        "target_progress",
        "status_text",
        "substatus_text",
        "elapsed_time",
        "fade_alpha",
        "is_headless",
        "_subscribed_resize",
    )

    def __init__(
        self,
        ctx: moderngl.Context | None,
        width: int = 1280,
        height: int = 720,
        is_headless: bool = False,
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.is_headless = is_headless

        self.current_progress: float = 0.0
        self.target_progress: float = 0.0
        self.status_text: str = "Loading..."
        self.substatus_text: str = ""
        self.elapsed_time: float = 0.0
        self.fade_alpha: float = 1.0

        self.prog: moderngl.Program | None = None
        self.vao: moderngl.VertexArray | None = None
        self.u_resolution = None
        self.u_time = None
        self.u_progress = None
        self.u_fade_alpha = None
        self.u_text_texture = None
        self.text_texture: moderngl.Texture | None = None

        self._text_surface: pygame.Surface | None = None
        self._font_status = None
        self._font_pct = None
        self._last_rendered_text_key: tuple[str, int, int, int] = ("", -1, -1, -1)

        # Initialize typography fonts
        self._init_fonts()

        if self.ctx is not None:
            vert_src = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
            frag_src = (SHADER_DIR / "loading_screen.frag").read_text(encoding="utf-8")
            self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            self.vao = self.ctx.vertex_array(self.prog, [])

            self.u_resolution = self.prog.get("u_Resolution", None)
            self.u_time = self.prog.get("u_Time", None)
            self.u_progress = self.prog.get("u_Progress", None)
            self.u_fade_alpha = self.prog.get("u_FadeAlpha", None)
            self.u_text_texture = self.prog.get("u_TextTexture", None)

            if self.u_resolution is not None:
                self.u_resolution.value = (float(width), float(height))
            if self.u_text_texture is not None:
                self.u_text_texture.value = 0

            # Create 2D texture for dynamic typography overlay
            self.text_texture = self.ctx.texture((width, height), 4, dtype="f1")
            self.text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._text_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self._update_text_texture(force=True)

        self._subscribed_resize: bool = False
        try:
            subscribe_event(WindowResizeEvent, self._on_window_resize, priority=90)
            self._subscribed_resize = True
        except Exception:
            self._subscribed_resize = False

    def _on_window_resize(self, event: WindowResizeEvent) -> None:
        """Automatically adapts loading screen viewport and typography when window resizes."""
        self.resize(event.width, event.height)

    def _init_fonts(self) -> None:
        """Initializes typography fonts for corner status and percentage."""
        if pygame.font and not pygame.font.get_init():
            try:
                pygame.font.init()
            except Exception:
                pass
        try:
            self._font_status = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13, bold=False)
            self._font_pct = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12, bold=True)
        except Exception:
            self._font_status = None
            self._font_pct = None

    def resize(self, width: int, height: int) -> None:
        """Adapts loading screen and dynamic text texture to new window dimensions."""
        if width <= 0 or height <= 0 or (width == self.width and height == self.height):
            return
        self.width = width
        self.height = height

        if self.u_resolution is not None:
            self.u_resolution.value = (float(width), float(height))

        if self.ctx is not None:
            if self.text_texture is not None:
                self.text_texture.release()
            self.text_texture = self.ctx.texture((width, height), 4, dtype="f1")
            self.text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._text_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._update_text_texture(force=True)

    def set_progress(self, target: float, status: str | None = None, substatus: str | None = None) -> None:
        """Sets target progress and updates status labels."""
        self.target_progress = max(0.0, min(1.0, float(target)))
        if status is not None:
            self.status_text = status
        if substatus is not None:
            self.substatus_text = substatus

    def update(self, dt: float) -> None:
        """Updates internal clock and exponentially smooths visual progress."""
        self.elapsed_time += dt
        rate = 12.0
        alpha = 1.0 - math.exp(-rate * max(dt, 0.001))
        self.current_progress += (self.target_progress - self.current_progress) * alpha

    def _update_text_texture(self, force: bool = False) -> None:
        """Renders subtle corner typography beside the bottom-right loading icon."""
        if self._text_surface is None or self.text_texture is None:
            return

        pct_val = int(round(self.current_progress * 100.0))
        key = (self.status_text, pct_val, self.width, self.height)
        if not force and key == self._last_rendered_text_key:
            return
        self._last_rendered_text_key = key

        surf = self._text_surface
        surf.fill((0, 0, 0, 0))

        w, h = self.width, self.height
        scale = max(0.70, min(1.8, h / 1080.0))
        margin_x = 56.0 * scale
        margin_y = 56.0 * scale

        # Icon is at (w - margin_x, h - margin_y) in Pygame window coordinates
        icon_x = w - margin_x
        icon_y = h - margin_y

        # Right-aligned text boundary placed to the left of the loading icon
        text_anchor_x = int(icon_x - 28.0 * scale)

        # 1. Subtle stage status message (e.g. Streaming 4K PBR Materials...)
        if self._font_status is not None and self.status_text:
            status_display = self.status_text
            stat_surf = self._font_status.render(status_display, True, (155, 170, 190))
            stat_x = text_anchor_x - stat_surf.get_width()
            while len(status_display) > 10 and stat_x < 10:
                status_display = status_display[:-4] + "..."
                stat_surf = self._font_status.render(status_display, True, (155, 170, 190))
                stat_x = text_anchor_x - stat_surf.get_width()
            stat_y = int(icon_y - 15.0 * scale)
            if stat_x >= 10:
                surf.blit(stat_surf, (stat_x, stat_y))

        # 2. Percentage indicator (e.g. LOADING 62%)
        if self._font_pct is not None:
            pct_txt = f"LOADING {pct_val}%"
            pct_surf = self._font_pct.render(pct_txt, True, (56, 189, 248))
            pct_x = text_anchor_x - pct_surf.get_width()
            pct_y = int(icon_y + 3.0 * scale)
            if pct_x >= 10:
                surf.blit(pct_surf, (pct_x, pct_y))

        # Flip vertically to match ModernGL texture coordinates
        flipped = pygame.transform.flip(surf, False, True)
        self.text_texture.write(pygame.image.tobytes(flipped, "RGBA"))

    def render(
        self,
        target_fbo: moderngl.Framebuffer | None = None,
        render_ctx: RenderContext | None = None,
    ) -> None:
        """Draws the fully black loading screen with bottom-right animated loading icon."""
        # 1. Adapt automatically to target FBO, render context, or window resize
        if target_fbo is not None:
            fbo_w, fbo_h = target_fbo.size
            if fbo_w > 0 and fbo_h > 0 and (fbo_w != self.width or fbo_h != self.height):
                self.resize(fbo_w, fbo_h)
        elif render_ctx is not None and (render_ctx.width != self.width or render_ctx.height != self.height):
            self.resize(render_ctx.width, render_ctx.height)
        elif pygame.display.get_init() and not self.is_headless:
            pygame.event.pump()
            try:
                win_w, win_h = pygame.display.get_window_size()
                if win_w > 0 and win_h > 0 and (win_w != self.width or win_h != self.height):
                    self.resize(win_w, win_h)
            except Exception:
                pass

        if self.ctx is None or self.prog is None or self.vao is None:
            return

        # 2. Update dynamic text texture
        self._update_text_texture()

        # 3. Bind screen or target FBO and ensure full viewport coverage
        if target_fbo is not None:
            target_fbo.use()
        elif hasattr(self.ctx, "screen") and self.ctx.screen is not None:
            self.ctx.screen.use()

        self.ctx.viewport = (0, 0, self.width, self.height)

        # 4. Bind text texture to texture unit 0
        if self.text_texture is not None:
            self.text_texture.use(location=0)

        # 5. Update shader uniforms
        if self.u_resolution is not None:
            self.u_resolution.value = (float(self.width), float(self.height))
        if self.u_time is not None:
            self.u_time.value = float(self.elapsed_time)
        if self.u_progress is not None:
            self.u_progress.value = float(self.current_progress)
        if self.u_fade_alpha is not None:
            self.u_fade_alpha.value = float(self.fade_alpha)

        # 6. Render procedural black background & corner loading animation
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.vao.render(moderngl.TRIANGLES, vertices=3)

        # 7. Swap display buffers if render_ctx provided
        if render_ctx is not None and not self.is_headless:
            render_ctx.swap_buffers()

    def stream_materials(
        self,
        atlas: TextureArrayAtlas,
        loader: BackgroundAssetLoader,
        render_ctx: RenderContext | None = None,
        on_layer_uploaded: Callable[[int, str], None] | None = None,
    ) -> dict[str, int]:
        """Runs the interactive 60 FPS loading loop while uploading decoded material layers."""
        clock = pygame.time.Clock()
        base_progress = 0.15
        progress_range = 0.70

        while not loader.is_material_loading_complete:
            raw_dt = clock.tick(60)
            dt = min(raw_dt * 0.001, 0.1)

            # 1. Drain decoded material layers from background worker queue
            while True:
                layer = loader.poll_material_layer()
                if layer is None:
                    break
                # Upload decoded pixel buffer directly to GPU texture array slice
                layer_idx = atlas.upload_decoded_layer(layer, rebuild_mipmaps=False)
                if on_layer_uploaded is not None:
                    on_layer_uploaded(layer_idx, layer.name)

            # 2. Update progress and status
            frac = loader.material_progress
            self.set_progress(
                target=base_progress + frac * progress_range,
                status=f"Streaming 4K PBR Materials: {loader.current_material_name} ({loader.completed_materials}/{loader.total_materials})",
            )

            # 3. Update animations and render frame
            self.update(dt)
            self.render(render_ctx=render_ctx)

        # Flush any remaining decoded layers that completed in the final frame
        while True:
            layer = loader.poll_material_layer()
            if layer is None:
                break
            layer_idx = atlas.upload_decoded_layer(layer, rebuild_mipmaps=False)
            if on_layer_uploaded is not None:
                on_layer_uploaded(layer_idx, layer.name)

        # 4. Generate Mipmap Pyramids
        self.set_progress(0.90, status="Generating GPU Texture Mipmap Pyramids...")
        for _ in range(3):
            raw_dt = clock.tick(60)
            self.update(min(raw_dt * 0.001, 0.1))
            self.render(render_ctx=render_ctx)

        atlas.rebuild_all_mipmaps()

        self.set_progress(0.95, status="GPU Material Texture Atlas Ready.")
        for _ in range(3):
            raw_dt = clock.tick(60)
            self.update(min(raw_dt * 0.001, 0.1))
            self.render(render_ctx=render_ctx)

        return atlas.name_to_layer

    def fade_out(self, duration: float = 0.3, render_ctx: RenderContext | None = None) -> None:
        """Smoothly fades out loading screen overlay before launching the main scene."""
        if self.is_headless or duration <= 0.0:
            return
        clock = pygame.time.Clock()
        elapsed = 0.0
        while elapsed < duration:
            raw_dt = clock.tick(60)
            dt = min(raw_dt * 0.001, 0.1)
            elapsed += dt
            self.fade_alpha = max(0.0, 1.0 - (elapsed / duration))
            self.update(dt)
            self.render(render_ctx=render_ctx)

    def destroy(self) -> None:
        """Releases GPU shader program, vertex array, and text overlay texture."""
        if getattr(self, "_subscribed_resize", False):
            try:
                unsubscribe_event(WindowResizeEvent, self._on_window_resize)
            except Exception:
                pass
            self._subscribed_resize = False

        if self.text_texture is not None:
            self.text_texture.release()
            self.text_texture = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None
