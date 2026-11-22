"""Post-Processing, Camera Optics, and Tone-Mapping Pipeline."""

from __future__ import annotations

import moderngl

from engine.gfx.quality_presets import RenderConfig
from engine.gfx.shader_utils import expand_includes


class PostProcessPipeline:
    """Evaluates tone-mapping (ACES/AgX), lens flare, chromatic aberration, vignette, and film grain."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "config",
        "hdr_texture",
        "hdr_fbo",
        "base_hdr_texture",
        "base_hdr_fbo",
        "final_texture",
        "final_fbo",
        "program",
        "quad_vao",
        "_dummy_black_tex",
        "_u_exposure",
        "_u_exposure_compensation",
        "_u_bloom_enabled",
        "_u_bloom_intensity",
        "_u_tonemap_mode",
        "_u_white_balance_enabled",
        "_u_white_balance_k",
        "_u_contrast",
        "_u_saturation",
        "_u_lift",
        "_u_gamma",
        "_u_gain",
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
            fragment_shader=expand_includes(post_process_glsl),
        )

        # Procedural full-screen quad (0 vertex attributes)
        self.quad_vao = self.ctx.vertex_array(self.program, [])

        # Cache uniform handles to eliminate per-frame dictionary lookups
        self._u_exposure = self.program.get("u_Exposure", None)
        self._u_exposure_compensation = self.program.get("u_ExposureCompensation", None)
        self._u_bloom_enabled = self.program.get("u_BloomEnabled", None)
        self._u_bloom_intensity = self.program.get("u_BloomIntensity", None)
        self._u_tonemap_mode = self.program.get("u_TonemapMode", None)

        self._u_white_balance_enabled = self.program.get("u_WhiteBalanceEnabled", None)
        self._u_white_balance_k = self.program.get("u_WhiteBalanceK", None)
        self._u_contrast = self.program.get("u_Contrast", None)
        self._u_saturation = self.program.get("u_Saturation", None)
        self._u_lift = self.program.get("u_Lift", None)
        self._u_gamma = self.program.get("u_Gamma", None)
        self._u_gain = self.program.get("u_Gain", None)

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

        # Current-frame base HDR resolve WITHOUT screen-space reflections.
        # SSR traces against this frame's depth but must sample this frame's
        # radiance, so the pipeline resolves once into this target with SSR
        # disabled, runs SSR against it, then resolves again into hdr_texture
        # with SSR enabled. Same format and dimensions as hdr_texture so the
        # two resolves are interchangeable.
        self.base_hdr_texture = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.base_hdr_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.base_hdr_fbo = self.ctx.framebuffer(color_attachments=[self.base_hdr_texture])

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
        if self.base_hdr_fbo is not None:
            self.base_hdr_fbo.release()
            self.base_hdr_fbo = None
        if self.base_hdr_texture is not None:
            self.base_hdr_texture.release()
            self.base_hdr_texture = None

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
        bloom_texture: moderngl.Texture | None = None,
        time_elapsed: float = 0.0,
        exposure_ev: float | None = None,
    ) -> None:
        """Executes tonemapping and blits to target_fbo (or screen).

        ``exposure_ev`` is the auto-exposure controller's adapted EV. When it is
        ``None`` the static ``RenderConfig.exposure`` is used instead.
        """
        fbo = target_fbo if target_fbo is not None else self.final_fbo
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Bind resolved HDR texture
        self.hdr_texture.use(location=0)

        bloom_on = self.config.bloom_enabled and bloom_texture is not None
        if bloom_on:
            bloom_texture.use(location=1)
        else:
            self._dummy_black_tex.use(location=1)

        # Bind anamorphic lens flare texture (binding 2)
        flare_on = getattr(self.config, "lens_flare_enabled", True) and (flare_texture is not None)
        if flare_on and flare_texture is not None:
            flare_texture.use(location=2)
        else:
            self._dummy_black_tex.use(location=2)

        # Set uniforms via pre-cached handles (zero string lookups)
        if self._u_exposure is not None:
            if exposure_ev is not None:
                # Auto-exposure: the adapted EV is the exposure, and the static
                # `exposure` becomes a user bias on top of it.
                self._u_exposure.value = float(2.0**exposure_ev) * float(self.config.exposure)
            else:
                self._u_exposure.value = self.config.exposure
        if self._u_exposure_compensation is not None:
            self._u_exposure_compensation.value = float(getattr(self.config, "exposure_compensation", 0.0))
        if self._u_bloom_enabled is not None:
            self._u_bloom_enabled.value = 1 if bloom_on else 0
        if self._u_bloom_intensity is not None:
            self._u_bloom_intensity.value = self.config.bloom_intensity

        tonemap_idx = 0  # ACES fitted
        if self.config.tonemap_mode == "AgX":
            tonemap_idx = 1
        elif self.config.tonemap_mode == "Neutral":
            tonemap_idx = 2
        elif self.config.tonemap_mode == "Reinhard":
            tonemap_idx = 3
        if self._u_tonemap_mode is not None:
            self._u_tonemap_mode.value = tonemap_idx

        # White balance
        wb_on = bool(getattr(self.config, "white_balance_enabled", False))
        if self._u_white_balance_enabled is not None:
            self._u_white_balance_enabled.value = 1 if wb_on else 0
        if self._u_white_balance_k is not None:
            self._u_white_balance_k.value = float(getattr(self.config, "white_balance_k", 6500.0))

        # Grading
        if self._u_contrast is not None:
            self._u_contrast.value = float(getattr(self.config, "contrast", 1.0))
        if self._u_saturation is not None:
            self._u_saturation.value = float(getattr(self.config, "saturation", 1.0))
        if self._u_lift is not None:
            self._u_lift.value = tuple(getattr(self.config, "lift", (0.0, 0.0, 0.0)))
        if self._u_gamma is not None:
            self._u_gamma.value = tuple(getattr(self.config, "gamma", (1.0, 1.0, 1.0)))
        if self._u_gain is not None:
            self._u_gain.value = tuple(getattr(self.config, "gain", (1.0, 1.0, 1.0)))

        # Lens Flare
        if self._u_lens_flare_enabled is not None:
            self._u_lens_flare_enabled.value = 1 if flare_on else 0

        # Chromatic Aberration
        ca_on = getattr(self.config, "chromatic_aberration_enabled", True)
        if self._u_chromatic_aberration_enabled is not None:
            self._u_chromatic_aberration_enabled.value = 1 if ca_on else 0
        if self._u_chromatic_aberration_intensity is not None:
            self._u_chromatic_aberration_intensity.value = float(
                getattr(self.config, "chromatic_aberration_intensity", 0.005)
            )

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
