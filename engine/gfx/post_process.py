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

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width <= 0 or new_height <= 0:
            return
        if new_width == self.width and new_height == self.height:
            return
        self.destroy()
        self.width = new_width
        self.height = new_height
        self._create_framebuffers()

    def render(self, target_fbo: moderngl.Framebuffer | None = None) -> None:
        """Executes tonemapping and blits to target_fbo (or screen)."""
        fbo = target_fbo if target_fbo is not None else self.final_fbo
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Bind resolved HDR texture
        self.hdr_texture.use(location=0)

        # Set uniforms
        if "u_Exposure" in self.program:
            self.program["u_Exposure"].value = self.config.exposure
        if "u_BloomEnabled" in self.program:
            self.program["u_BloomEnabled"].value = 1 if self.config.bloom_enabled else 0
        if "u_BloomIntensity" in self.program:
            self.program["u_BloomIntensity"].value = self.config.bloom_intensity

        tonemap_idx = 0  # ACES
        if self.config.tonemap_mode == "AgX":
            tonemap_idx = 1
        elif self.config.tonemap_mode == "Reinhard":
            tonemap_idx = 2
        if "u_TonemapMode" in self.program:
            self.program["u_TonemapMode"].value = tonemap_idx

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
