"""Low-frequency procedural cloud generation and sky-only compositing."""

from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from engine.gfx.shader_utils import load_shader


class SkyCloudPass:
    """Builds a seamless cloud-noise tile once, then composites it over sky pixels."""

    __slots__ = (
        "ctx",
        "noise_size",
        "noise_texture",
        "noise_fbo",
        "noise_program",
        "noise_vao",
        "cloud_program",
        "cloud_vao",
        "_uniforms",
    )

    def __init__(self, ctx: moderngl.Context, noise_size: int = 512) -> None:
        self.ctx = ctx
        self.noise_size = max(256, int(noise_size))
        self.noise_texture = ctx.texture((self.noise_size, self.noise_size), 4, dtype="f2")
        self.noise_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.noise_texture.repeat_x = True
        self.noise_texture.repeat_y = True
        self.noise_texture.anisotropy = 8.0
        self.noise_fbo = ctx.framebuffer(color_attachments=[self.noise_texture])

        vertex_shader = load_shader("fullscreen_quad.vert")
        noise_fragment = load_shader("sky_cloud_noise.frag")
        cloud_fragment = load_shader("sky_cloud.frag")
        self.noise_program = ctx.program(vertex_shader=vertex_shader, fragment_shader=noise_fragment)
        self.noise_vao = ctx.vertex_array(self.noise_program, [])
        self.cloud_program = ctx.program(vertex_shader=vertex_shader, fragment_shader=cloud_fragment)
        self.cloud_vao = ctx.vertex_array(self.cloud_program, [])

        self._uniforms = {
            name: self.cloud_program.get(name, None)
            for name in (
                "u_InvProjection",
                "u_InvView",
                "u_CameraPosition",
                "u_SunDirection",
                "u_SunColor",
                "u_MoonColor",
                "u_MoonIntensity",
                "u_Time",
                "u_CloudCoverage",
                "u_CloudOpacity",
                "u_CloudAltitude",
                "u_CloudScale",
                "u_CloudLayerSeparation",
                "u_CloudSecondaryOpacity",
                "u_CloudWind",
            )
        }
        self.cloud_program["u_DepthTexture"].value = 0
        self.cloud_program["u_CloudNoise"].value = 1
        self._bake_noise()

    def _bake_noise(self) -> None:
        previous_fbo = self.ctx.fbo
        previous_viewport = self.ctx.viewport
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        try:
            self.noise_fbo.use()
            self.ctx.viewport = (0, 0, self.noise_size, self.noise_size)
            self.noise_vao.render(moderngl.TRIANGLES, vertices=3)
            self.noise_texture.build_mipmaps()
        finally:
            self._restore_target(previous_fbo, previous_viewport)

    def _restore_target(self, framebuffer: Any, viewport: tuple[int, int, int, int]) -> None:
        # A headless standalone context can expose a default framebuffer that is
        # not bindable, and a framebuffer captured before a context switch can be
        # stale. Neither should abort construction, so fall back to the screen and
        # finally to leaving the current binding untouched.
        if framebuffer is not None:
            try:
                framebuffer.use()
                self.ctx.viewport = viewport
                return
            except Exception:
                pass
        screen = getattr(self.ctx, "screen", None)
        if screen is not None:
            try:
                screen.use()
            except Exception:
                pass
        self.ctx.viewport = viewport

    def render(
        self,
        hdr_fbo: Any,
        depth_texture: Any,
        inverse_projection: np.ndarray,
        inverse_view: np.ndarray,
        camera_position: tuple[float, float, float],
        sun_direction: tuple[float, float, float],
        sun_color: tuple[float, float, float],
        moon_color: tuple[float, float, float],
        moon_intensity: float,
        time_elapsed: float,
        cloud_config: Any,
    ) -> None:
        """Composites the cloud deck only where the depth buffer marks sky."""
        if not cloud_config.cloud_enabled or cloud_config.sky_mode.upper() == "SPACE":
            return

        previous_fbo = self.ctx.fbo
        previous_viewport = self.ctx.viewport
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            hdr_fbo.use()
            self.ctx.viewport = previous_viewport
            depth_texture.use(location=0)
            self.noise_texture.use(location=1)

            uniforms = self._uniforms
            uniforms["u_InvProjection"].write(inverse_projection.tobytes())
            uniforms["u_InvView"].write(inverse_view.tobytes())
            uniforms["u_CameraPosition"].value = tuple(float(v) for v in camera_position)
            uniforms["u_SunDirection"].value = tuple(-float(v) for v in sun_direction)
            uniforms["u_SunColor"].value = tuple(float(v) for v in sun_color)
            uniforms["u_MoonColor"].value = tuple(float(v) for v in moon_color)
            uniforms["u_MoonIntensity"].value = float(cloud_config.moon_intensity)
            uniforms["u_Time"].value = float(time_elapsed)
            uniforms["u_CloudCoverage"].value = float(cloud_config.cloud_coverage)
            uniforms["u_CloudOpacity"].value = float(cloud_config.cloud_opacity)
            uniforms["u_CloudAltitude"].value = float(cloud_config.cloud_altitude)
            uniforms["u_CloudScale"].value = float(cloud_config.cloud_scale)
            uniforms["u_CloudLayerSeparation"].value = float(cloud_config.cloud_layer_separation)
            uniforms["u_CloudSecondaryOpacity"].value = float(cloud_config.cloud_secondary_opacity)
            uniforms["u_CloudWind"].value = tuple(float(v) for v in cloud_config.cloud_wind)
            self.cloud_vao.render(moderngl.TRIANGLES, vertices=3)
        finally:
            self.ctx.disable(moderngl.BLEND)
            self._restore_target(previous_fbo, previous_viewport)

    def destroy(self) -> None:
        self.noise_fbo.release()
        self.noise_texture.release()
        self.noise_vao.release()
        self.noise_program.release()
        self.cloud_vao.release()
        self.cloud_program.release()
