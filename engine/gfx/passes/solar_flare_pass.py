"""Physically-motivated solar flare pass.

Renders a sun-anchored flare (disc bloom, aperture starburst, anamorphic streak,
optical ghosts, halo ring) into a half-resolution HDR target. Unlike the
anamorphic flare, this pass is driven by the sun's screen-space position and is
occluded by scene geometry via a depth-buffer test.

The pass is a drop-in alternative to ``CameraOpticsPass.render_flare`` selected
by ``RenderConfig.lens_flare_mode == "SOLAR"``.
"""

from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from engine.gfx.render_graph import RenderPass
from engine.gfx.shader_utils import load_shader


class SolarFlarePass(RenderPass):
    """Sun-anchored physically-based flare with depth occlusion."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "prog",
        "vao",
        "flare_texture",
        "flare_fbo",
        "_u_sun_screen_pos",
        "_u_screen_size",
        "_u_sun_angular_radius",
        "_u_sun_on_screen",
        "_u_flare_threshold",
        "_u_flare_tint",
        "_u_blade_count",
        "_u_bloom_intensity",
        "_u_starburst_intensity",
        "_u_streak_intensity",
        "_u_streak_width",
        "_u_ghost_intensity",
        "_u_halo_intensity",
        "_u_occlusion_fade",
        "_sun_pos_scratch",
    )

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SolarFlarePass", enabled=True)
        self.ctx = ctx
        self.width = max(1, width)
        self.height = max(1, height)

        quad_vert = load_shader("fullscreen_quad.vert")
        flare_frag = load_shader("solar_flare.frag")
        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=flare_frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        # Cache uniform handles once so the render path does no dict lookups.
        self._u_sun_screen_pos = self.prog.get("u_SunScreenPos", None)
        self._u_screen_size = self.prog.get("u_ScreenSize", None)
        self._u_sun_angular_radius = self.prog.get("u_SunAngularRadius", None)
        self._u_sun_on_screen = self.prog.get("u_SunOnScreen", None)
        self._u_flare_threshold = self.prog.get("u_FlareThreshold", None)
        self._u_flare_tint = self.prog.get("u_FlareTint", None)
        self._u_blade_count = self.prog.get("u_BladeCount", None)
        self._u_bloom_intensity = self.prog.get("u_BloomIntensity", None)
        self._u_starburst_intensity = self.prog.get("u_StarburstIntensity", None)
        self._u_streak_intensity = self.prog.get("u_StreakIntensity", None)
        self._u_streak_width = self.prog.get("u_StreakWidth", None)
        self._u_ghost_intensity = self.prog.get("u_GhostIntensity", None)
        self._u_halo_intensity = self.prog.get("u_HaloIntensity", None)
        self._u_occlusion_fade = self.prog.get("u_OcclusionFade", None)

        if "u_SceneHDR" in self.prog:
            self.prog["u_SceneHDR"].value = 0
        if "u_DepthTexture" in self.prog:
            self.prog["u_DepthTexture"].value = 1

        # Pre-allocated scratch so the per-frame path allocates nothing.
        self._sun_pos_scratch = np.zeros(2, dtype=np.float32)

        self.flare_texture: moderngl.Texture | None = None
        self.flare_fbo: moderngl.Framebuffer | None = None
        self._create_buffers()

    def _create_buffers(self) -> None:
        flare_w = max(1, self.width // 2)
        flare_h = max(1, self.height // 2)
        self.flare_texture = self.ctx.texture((flare_w, flare_h), 4, dtype="f2")
        self.flare_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.flare_texture.repeat_x = False
        self.flare_texture.repeat_y = False
        self.flare_fbo = self.ctx.framebuffer(color_attachments=[self.flare_texture])

    def _destroy_buffers(self) -> None:
        if self.flare_fbo is not None:
            self.flare_fbo.release()
            self.flare_fbo = None
        if self.flare_texture is not None:
            self.flare_texture.release()
            self.flare_texture = None

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if width == self.width and height == self.height:
            return
        self._destroy_buffers()
        self.width = width
        self.height = height
        self._create_buffers()

    def render(
        self,
        scene_hdr_texture: moderngl.Texture,
        depth_texture: moderngl.Texture,
        sun_screen_pos: tuple[float, float],
        sun_on_screen: bool,
        config: Any,
        target_fbo: moderngl.Framebuffer | None = None,
    ) -> moderngl.Texture:
        """Renders the solar flare and returns the half-res HDR flare texture."""
        # Self-heal on window resize: the flare target tracks the scene HDR size
        # at half resolution, so a stale allocation would stretch the flare.
        scene_w, scene_h = scene_hdr_texture.size
        if scene_w != self.width or scene_h != self.height:
            self.resize(scene_w, scene_h)

        fbo = target_fbo if target_fbo is not None else self.flare_fbo
        fbo.use()
        fw, fh = self.flare_texture.size
        self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        scene_hdr_texture.use(location=0)
        depth_texture.use(location=1)

        self._sun_pos_scratch[0] = float(sun_screen_pos[0])
        self._sun_pos_scratch[1] = float(sun_screen_pos[1])

        if self._u_sun_screen_pos is not None:
            self._u_sun_screen_pos.value = (float(sun_screen_pos[0]), float(sun_screen_pos[1]))
        if self._u_screen_size is not None:
            self._u_screen_size.value = (float(fw), float(fh))
        if self._u_sun_angular_radius is not None:
            self._u_sun_angular_radius.value = float(
                getattr(config, "sun_angular_radius", 0.00465)
            )
        if self._u_sun_on_screen is not None:
            self._u_sun_on_screen.value = 1.0 if sun_on_screen else 0.0
        if self._u_flare_threshold is not None:
            self._u_flare_threshold.value = float(
                getattr(config, "lens_flare_threshold", 1.8)
            )
        if self._u_flare_tint is not None:
            tint = getattr(config, "lens_flare_tint", (0.25, 0.65, 1.0))
            self._u_flare_tint.value = (float(tint[0]), float(tint[1]), float(tint[2]))
        if self._u_blade_count is not None:
            self._u_blade_count.value = int(getattr(config, "solar_flare_blade_count", 6))
        if self._u_bloom_intensity is not None:
            self._u_bloom_intensity.value = float(
                getattr(config, "solar_flare_bloom_intensity", 1.0)
            )
        if self._u_starburst_intensity is not None:
            self._u_starburst_intensity.value = float(
                getattr(config, "solar_flare_starburst_intensity", 0.6)
            )
        if self._u_streak_intensity is not None:
            self._u_streak_intensity.value = float(
                getattr(config, "lens_flare_streak_intensity", 0.6)
            )
        if self._u_streak_width is not None:
            self._u_streak_width.value = float(
                getattr(config, "lens_flare_streak_width", 32.0)
            )
        if self._u_ghost_intensity is not None:
            self._u_ghost_intensity.value = float(
                getattr(config, "lens_flare_ghost_intensity", 0.35)
            )
        if self._u_halo_intensity is not None:
            self._u_halo_intensity.value = float(
                getattr(config, "lens_flare_halo_intensity", 0.25)
            )
        if self._u_occlusion_fade is not None:
            self._u_occlusion_fade.value = float(
                getattr(config, "solar_flare_occlusion_fade", 1.0)
            )

        self.vao.render(moderngl.TRIANGLES, vertices=3)
        return self.flare_texture

    def execute(self, context: Any) -> None:
        """Render-graph entry point.

        The solar flare needs the sun's screen position and the depth buffer,
        which the pipeline supplies directly through ``render()``. When driven
        from the graph, fall back to the resources published on the blackboard.
        """
        scene_hdr = context.resources.get("hdr_color")
        g_buffer = context.resources.get("g_buffer")
        if scene_hdr is None or g_buffer is None:
            return
        sun_uv = context.resources.get("sun_screen_pos", (0.5, 0.5))
        sun_on_screen = bool(context.resources.get("sun_on_screen", True))
        self.render(
            scene_hdr_texture=scene_hdr,
            depth_texture=g_buffer.depth_texture,
            sun_screen_pos=sun_uv,
            sun_on_screen=sun_on_screen,
            config=context.config,
        )
        context.resources["lens_flare_texture"] = self.flare_texture

    def destroy(self) -> None:
        self._destroy_buffers()
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None


__all__ = ["SolarFlarePass"]
