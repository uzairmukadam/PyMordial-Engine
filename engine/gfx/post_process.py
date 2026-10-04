"""Post-Processing, Camera Optics, and Tone-Mapping Pipeline."""

from __future__ import annotations
import moderngl

from engine.gfx.quality_presets import RenderConfig


class PostProcessPipeline:
    """Evaluates tone-mapping (ACES/AgX), lens flare, chromatic aberration, vignette, and film grain."""

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
        "_dummy_black_tex",
        "_u_exposure",
        "_u_bloom_enabled",
        "_u_bloom_intensity",
        "_u_tonemap_mode",
        "_u_lens_flare_enabled",
        "_u_chromatic_aberration_enabled",
        "_u_chromatic_aberration_intensity",
        "_u_vignette_enabled",
        "_u_vignette_intensity",
        "_u_vignette_roundness",
        "_u_vignette_smoothness",
        "_u_film_grain_enabled",
        "_u_film_grain_intensity",
        "_u_time",
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

        self._u_lens_flare_enabled = self.program.get("u_LensFlareEnabled", None)
        self._u_chromatic_aberration_enabled = self.program.get("u_ChromaticAberrationEnabled", None)
        self._u_chromatic_aberration_intensity = self.program.get("u_ChromaticAberrationIntensity", None)
        self._u_vignette_enabled = self.program.get("u_VignetteEnabled", None)
        self._u_vignette_intensity = self.program.get("u_VignetteIntensity", None)
        self._u_vignette_roundness = self.program.get("u_VignetteRoundness", None)
        self._u_vignette_smoothness = self.program.get("u_VignetteSmoothness", None)
        self._u_film_grain_enabled = self.program.get("u_FilmGrainEnabled", None)
        self._u_film_grain_intensity = self.program.get("u_FilmGrainIntensity", None)
        self._u_time = self.program.get("u_Time", None)

        # Assign texture unit bindings
        if "u_SceneHDR" in self.program:
            self.program["u_SceneHDR"].value = 0
        if "u_BloomTexture" in self.program:
            self.program["u_BloomTexture"].value = 1
        if "u_LensFlareTexture" in self.program:
            self.program["u_LensFlareTexture"].value = 2

        # 1x1 black fallback texture for inactive bloom or flares
        self._dummy_black_tex = self.ctx.texture((1, 1), 4, dtype="f2")
        self._dummy_black_tex.write(b"\x00" * 8)

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

    def render(
        self,
        target_fbo: moderngl.Framebuffer | None = None,
        flare_texture: moderngl.Texture | None = None,
        time_elapsed: float = 0.0,
    ) -> None:
        """Executes tonemapping and blits to target_fbo (or screen)."""
        fbo = target_fbo if target_fbo is not None else self.final_fbo
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Bind resolved HDR texture
        self.hdr_texture.use(location=0)

        # Fallback for bloom texture (binding 1)
        self._dummy_black_tex.use(location=1)

        # Bind anamorphic lens flare texture (binding 2)
        flare_on = getattr(self.config, "lens_flare_enabled", True) and (flare_texture is not None)
        if flare_on and flare_texture is not None:
            flare_texture.use(location=2)
        else:
            self._dummy_black_tex.use(location=2)

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

        # Lens Flare
        if self._u_lens_flare_enabled is not None:
            self._u_lens_flare_enabled.value = 1 if flare_on else 0

        # Chromatic Aberration
        ca_on = getattr(self.config, "chromatic_aberration_enabled", True)
        if self._u_chromatic_aberration_enabled is not None:
            self._u_chromatic_aberration_enabled.value = 1 if ca_on else 0
        if self._u_chromatic_aberration_intensity is not None:
            self._u_chromatic_aberration_intensity.value = float(getattr(self.config, "chromatic_aberration_intensity", 0.005))

        # Vignette
        vig_on = getattr(self.config, "vignette_enabled", True)
        if self._u_vignette_enabled is not None:
            self._u_vignette_enabled.value = 1 if vig_on else 0
        if self._u_vignette_intensity is not None:
            self._u_vignette_intensity.value = float(getattr(self.config, "vignette_intensity", 0.35))
        if self._u_vignette_roundness is not None:
            self._u_vignette_roundness.value = float(getattr(self.config, "vignette_roundness", 0.85))
        if self._u_vignette_smoothness is not None:
            self._u_vignette_smoothness.value = float(getattr(self.config, "vignette_smoothness", 0.50))

        # Film Grain
        grain_on = getattr(self.config, "film_grain_enabled", True)
        if self._u_film_grain_enabled is not None:
            self._u_film_grain_enabled.value = 1 if grain_on else 0
        if self._u_film_grain_intensity is not None:
            self._u_film_grain_intensity.value = float(getattr(self.config, "film_grain_intensity", 0.03))

        if self._u_time is not None:
            self._u_time.value = float(time_elapsed)

        # Render procedural full-screen triangle
        self.quad_vao.render(mode=moderngl.TRIANGLES, vertices=3)

    @property
    def output_texture_id(self) -> int:
        """Returns OpenGL texture handle for ImGui editor blitting."""
        return self.final_texture.glo

    def destroy(self) -> None:
        self._destroy_framebuffers()
        if self._dummy_black_tex is not None:
            self._dummy_black_tex.release()
            self._dummy_black_tex = None
        if self.quad_vao is not None:
            self.quad_vao.release()
            self.quad_vao = None
        if self.program is not None:
            self.program.release()
            self.program = None
