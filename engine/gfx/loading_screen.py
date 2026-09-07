"""Hardware-accelerated cinematic loading screen and background asset streaming controller."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import pygame
import moderngl

if TYPE_CHECKING:
    from engine.gfx.context import RenderContext
    from engine.gfx.texture_atlas import TextureArrayAtlas
    from engine.core.async_loader import BackgroundAssetLoader

SHADER_DIR = Path(__file__).resolve().parent.parent.parent / "shaders"


class LoadingScreen:
    """Renders a procedural 60 FPS cinematic loading screen with live background asset streaming."""

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
        "_font_title",
        "_font_subtitle",
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
        self.status_text: str = "Initializing Engine..."
        self.substatus_text: str = "PYMORDIAL ENGINE 3D"
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
        self._font_title = None
        self._font_subtitle = None
        self._font_status = None
        self._font_pct = None
        self._last_rendered_text_key: tuple[str, int] = ("", -1)

        # Initialize typography fonts if Pygame is available
        if pygame.font and not pygame.font.get_init():
            try:
                pygame.font.init()
            except Exception:
                pass

        try:
            self._font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 30, bold=True)
            self._font_subtitle = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=False)
            self._font_status = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14, bold=False)
            self._font_pct = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 20, bold=True)
        except Exception:
            pass

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
        """Renders crisp typography onto text_surface and uploads to text_texture."""
        if self._text_surface is None or self.text_texture is None:
            return

        pct_val = int(round(self.current_progress * 100.0))
        key = (self.status_text, pct_val)
        if not force and key == self._last_rendered_text_key:
            return
        self._last_rendered_text_key = key

        surf = self._text_surface
        surf.fill((0, 0, 0, 0))

        w, h = self.width, self.height

        # 1. Title: P Y M O R D I A L   E N G I N E
        if self._font_title is not None:
            title_txt = "P Y M O R D I A L   E N G I N E"
            title_surf = self._font_title.render(title_txt, True, (245, 250, 255))
            title_x = (w - title_surf.get_width()) // 2
            title_y = int(h * 0.16)
            surf.blit(title_surf, (title_x, title_y))

        # 2. Subtitle: ARCHITECTURE & MATERIAL TESTBED
        if self._font_subtitle is not None:
            sub_txt = "ARCHITECTURE & MATERIAL TESTBED"
            sub_surf = self._font_subtitle.render(sub_txt, True, (56, 189, 248))
            sub_x = (w - sub_surf.get_width()) // 2
            sub_y = int(h * 0.16) + 40
            surf.blit(sub_surf, (sub_x, sub_y))

        # 3. Progress Percentage (e.g. 54%)
        # Progress bar is centered at Y = (1.0 - 0.26) * h = 0.74 * h in Pygame coordinates
        if self._font_pct is not None:
            pct_txt = f"{pct_val}%"
            pct_surf = self._font_pct.render(pct_txt, True, (56, 189, 248))
            pct_x = (w - pct_surf.get_width()) // 2
            pct_y = int(h * 0.74) + 16
            surf.blit(pct_surf, (pct_x, pct_y))

        # 4. Status Text (e.g. Streaming 4K PBR Materials...)
        if self._font_status is not None and self.status_text:
            stat_surf = self._font_status.render(self.status_text, True, (186, 210, 235))
            stat_x = (w - stat_surf.get_width()) // 2
            stat_y = int(h * 0.74) + 46
            surf.blit(stat_surf, (stat_x, stat_y))

        # Flip vertically to match ModernGL texture coordinates
        flipped = pygame.transform.flip(surf, False, True)
        self.text_texture.write(pygame.image.tobytes(flipped, "RGBA"))

    def render(
        self,
        target_fbo: moderngl.Framebuffer | None = None,
        render_ctx: RenderContext | None = None,
    ) -> None:
        """Draws the animated loading screen, energy rings, progress bar, and typography."""
        # 1. Pump OS window events to keep Windows responsive
        if pygame.display.get_init() and not self.is_headless:
            pygame.event.pump()

        if self.ctx is None or self.prog is None or self.vao is None:
            return

        # 2. Update dynamic text texture
        self._update_text_texture()

        # 3. Bind screen or target FBO
        if target_fbo is not None:
            target_fbo.use()
        elif hasattr(self.ctx, "screen") and self.ctx.screen is not None:
            self.ctx.screen.use()

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

        # 6. Render procedural shader background & glowing progress bar
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

    def fade_out(self, duration: float = 0.4, render_ctx: RenderContext | None = None) -> None:
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
        if self.text_texture is not None:
            self.text_texture.release()
            self.text_texture = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None
