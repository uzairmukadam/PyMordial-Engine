"""Post-Processing and Tone-Mapping Pipeline."""

from __future__ import annotations
import moderngl

from engine.gfx.quality_presets import RenderConfig


class PostProcessPipeline:
    """Evaluates tone-mapping (ACES/AgX), bloom composite, and generates final HDR/LDR textures."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "config",
        "hdr_texture",
        "hdr_fbo",
        "final_texture",
        "final_fbo",
        "program",
        "quad_vao",
        "_u_exposure",
        "_u_bloom_enabled",
        "_u_bloom_intensity",
        "_u_tonemap_mode",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        config: RenderConfig,
        post_process_glsl: str,
        quad_vert_glsl: str,
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.config = config

        self.program = self.ctx.program(
            vertex_shader=quad_vert_glsl,
            fragment_shader=post_process_glsl,
        )

        # Procedural full-screen quad (0 vertex attributes)
        self.quad_vao = self.ctx.vertex_array(self.program, [])

        # Cache uniform handles to eliminate per-frame dictionary lookups
        self._u_exposure = self.program.get("u_Exposure", None)
        self._u_bloom_enabled = self.program.get("u_BloomEnabled", None)
        self._u_bloom_intensity = self.program.get("u_BloomIntensity", None)
        self._u_tonemap_mode = self.program.get("u_TonemapMode", None)
        if "u_HDRScene" in self.program:
            self.program["u_HDRScene"].value = 0

        self._create_framebuffers()

    def _create_framebuffers(self) -> None:
        # Offscreen HDR16F Texture (Intermediate scene lighting resolve)
        self.hdr_texture = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.hdr_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr_fbo = self.ctx.framebuffer(color_attachments=[self.hdr_texture])

        # Final Resolved Texture (Passed to screen or ImGui Viewport texture handle)
        self.final_texture = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.final_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.final_fbo = self.ctx.framebuffer(color_attachments=[self.final_texture])

    def _destroy_framebuffers(self) -> None:
        if self.ctx.screen is not None:
            try:
                self.ctx.screen.use()
            except Exception:
                pass
        if self.final_fbo is not None:
            self.final_fbo.release()
            self.final_fbo = None
        if self.final_texture is not None:
            self.final_texture.release()
            self.final_texture = None
        if self.hdr_fbo is not None:
            self.hdr_fbo.release()
            self.hdr_fbo = None
        if self.hdr_texture is not None:
            self.hdr_texture.release()
            self.hdr_texture = None

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width <= 0 or new_height <= 0:
            return
        if new_width == self.width and new_height == self.height:
            return
        self._destroy_framebuffers()
        self.width = new_width
        self.height = new_height
        self._create_framebuffers()

    def render(self, target_fbo: moderngl.Framebuffer | None = None) -> None:
        """Executes tonemapping and blits to target_fbo (or screen)."""
        fbo = target_fbo if target_fbo is not None else self.final_fbo
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Bind resolved HDR texture
        self.hdr_texture.use(location=0)

        # Set uniforms via pre-cached handles (zero string lookups)
        if self._u_exposure is not None:
            self._u_exposure.value = self.config.exposure
        if self._u_bloom_enabled is not None:
            self._u_bloom_enabled.value = 1 if self.config.bloom_enabled else 0
        if self._u_bloom_intensity is not None:
            self._u_bloom_intensity.value = self.config.bloom_intensity

        tonemap_idx = 0  # ACES
        if self.config.tonemap_mode == "AgX":
            tonemap_idx = 1
        elif self.config.tonemap_mode == "Reinhard":
            tonemap_idx = 2
        if self._u_tonemap_mode is not None:
            self._u_tonemap_mode.value = tonemap_idx

        # Render procedural full-screen triangle
        self.quad_vao.render(mode=moderngl.TRIANGLES, vertices=3)

    @property
    def output_texture_id(self) -> int:
        """Returns OpenGL texture handle for ImGui editor blitting."""
        return self.final_texture.glo

    def destroy(self) -> None:
        self.hdr_texture.release()
        self.hdr_fbo.release()
        self.final_texture.release()
        self.final_fbo.release()
        self.quad_vao.release()
        self.program.release()
