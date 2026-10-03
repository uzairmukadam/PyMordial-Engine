"""Cinematic Camera Optics & Lens Effects Pass.

Coordinates:
- Physically-based Bokeh Depth of Field (Circle of Confusion, Anamorphic/Hexagonal bokeh)
- Velocity Buffer Motion Blur (reconstructed from current and previous View-Projection matrices)
- Anamorphic Lens Flare & Ghost Reflections (1D horizontal streak, optical ghosts, halo ring)
"""

from __future__ import annotations
from typing import Any
import moderngl
import numpy as np

from engine.gfx.render_graph import RenderPass

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()
ROOT_DIR = SHADER_DIR.parent
_load_shader = load_shader



class CameraOpticsPass(RenderPass):
    """Manages Bokeh Depth of Field, Camera Motion Blur, and Anamorphic Lens Flares."""

    BOKEH_SHAPE_MAP = {
        "CIRCULAR": 0,
        "HEXAGONAL": 1,
        "ANAMORPHIC": 2,
    }

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="CameraOpticsPass", enabled=True)
        self.ctx = ctx
        self.width = max(1, width)
        self.height = max(1, height)

        # 1. Compile Shaders
        quad_vert = _load_shader("fullscreen_quad.vert")
        optics_frag = _load_shader("camera_optics.frag")
        flare_frag = _load_shader("anamorphic_flare.frag")

        self.optics_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=optics_frag,
        )
        self.flare_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=flare_frag,
        )

        self.optics_vao = self.ctx.vertex_array(self.optics_prog, [])
        self.flare_vao = self.ctx.vertex_array(self.flare_prog, [])

        # 2. Cache Optics Uniform Handles
        self._u_dof_enabled = self.optics_prog.get("u_DoFEnabled", None)
        self._u_focus_dist = self.optics_prog.get("u_FocusDistance", None)
        self._u_focal_len = self.optics_prog.get("u_FocalLength", None)
        self._u_fstop = self.optics_prog.get("u_ApertureFStop", None)
        self._u_max_coc = self.optics_prog.get("u_MaxCoCRadius", None)
        self._u_bokeh_shape = self.optics_prog.get("u_BokehShape", None)
        self._u_anamorphic_ratio = self.optics_prog.get("u_AnamorphicRatio", None)

        self._u_mb_enabled = self.optics_prog.get("u_MotionBlurEnabled", None)
        self._u_mb_samples = self.optics_prog.get("u_MotionBlurSamples", None)
        self._u_mb_intensity = self.optics_prog.get("u_MotionBlurIntensity", None)
        self._u_mb_max_radius = self.optics_prog.get("u_MaxMotionRadius", None)
        self._u_prev_vp = self.optics_prog.get("u_PrevViewProjection", None)

        # 3. Cache Flare Uniform Handles
        self._u_flare_thresh = self.flare_prog.get("u_FlareThreshold", None)
        self._u_flare_streak_i = self.flare_prog.get("u_StreakIntensity", None)
        self._u_flare_streak_w = self.flare_prog.get("u_StreakWidth", None)
        self._u_flare_ghost_i = self.flare_prog.get("u_GhostIntensity", None)
        self._u_flare_halo_i = self.flare_prog.get("u_HaloIntensity", None)
        self._u_flare_tint = self.flare_prog.get("u_FlareTint", None)
        self._u_flare_screen = self.flare_prog.get("u_ScreenSize", None)

        if "u_SceneHDR" in self.optics_prog:
            self.optics_prog["u_SceneHDR"].value = 0
        if "u_DepthTexture" in self.optics_prog:
            self.optics_prog["u_DepthTexture"].value = 1
        if "u_SceneHDR" in self.flare_prog:
            self.flare_prog["u_SceneHDR"].value = 0

        # 4. Allocate Intermediate Textures and Framebuffers
        self.optics_texture: moderngl.Texture | None = None
        self.optics_fbo: moderngl.Framebuffer | None = None
        self.flare_texture: moderngl.Texture | None = None
        self.flare_fbo: moderngl.Framebuffer | None = None
        self._create_buffers()

    def _create_buffers(self) -> None:
        """Allocates HDR intermediate textures for optics and lens flares."""
        # Optics HDR16F Texture
        self.optics_texture = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.optics_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.optics_fbo = self.ctx.framebuffer(color_attachments=[self.optics_texture])

        # Flare HDR16F Texture (downsampled to half-res for performance and wide streak reach)
        flare_w = max(1, self.width // 2)
        flare_h = max(1, self.height // 2)
        self.flare_texture = self.ctx.texture((flare_w, flare_h), 4, dtype="f2")
        self.flare_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.flare_fbo = self.ctx.framebuffer(color_attachments=[self.flare_texture])

    def _destroy_buffers(self) -> None:
        """Releases framebuffer attachments."""
        if self.optics_fbo is not None:
            self.optics_fbo.release()
            self.optics_fbo = None
        if self.optics_texture is not None:
            self.optics_texture.release()
            self.optics_texture = None
        if self.flare_fbo is not None:
            self.flare_fbo.release()
            self.flare_fbo = None
        if self.flare_texture is not None:
            self.flare_texture.release()
            self.flare_texture = None

    def resize(self, new_width: int, new_height: int) -> None:
        """Resizes optics and flare framebuffers."""
        if new_width <= 0 or new_height <= 0:
            return
        if new_width == self.width and new_height == self.height:
            return
        self._destroy_buffers()
        self.width = new_width
        self.height = new_height
        self._create_buffers()

    def render_optics(
        self,
        scene_hdr_texture: moderngl.Texture,
        depth_texture: moderngl.Texture,
        prev_vp_mat: np.ndarray,
        config: Any,
        target_fbo: moderngl.Framebuffer | None = None,
    ) -> moderngl.Texture:
        """Applies Depth of Field and Camera Motion Blur to the scene HDR buffer."""
        fbo = target_fbo if target_fbo is not None else self.optics_fbo
        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Bind inputs
        scene_hdr_texture.use(location=0)
        depth_texture.use(location=1)

        # Upload DoF uniforms
        dof_on = getattr(config, "dof_enabled", True)
        if self._u_dof_enabled is not None:
            self._u_dof_enabled.value = 1 if dof_on else 0
        if self._u_focus_dist is not None:
            self._u_focus_dist.value = float(getattr(config, "dof_focus_distance", 5.0))
        if self._u_focal_len is not None:
            self._u_focal_len.value = float(getattr(config, "dof_focal_length", 50.0))
        if self._u_fstop is not None:
            self._u_fstop.value = float(getattr(config, "dof_aperture", 2.8))
        if self._u_max_coc is not None:
            self._u_max_coc.value = float(getattr(config, "dof_max_coc", 24.0))
        if self._u_bokeh_shape is not None:
            shape_str = str(getattr(config, "dof_bokeh_shape", "CIRCULAR")).upper()
            self._u_bokeh_shape.value = self.BOKEH_SHAPE_MAP.get(shape_str, 0)
        if self._u_anamorphic_ratio is not None:
            self._u_anamorphic_ratio.value = float(getattr(config, "dof_anamorphic_ratio", 1.0))

        # Upload Motion Blur uniforms
        mb_on = getattr(config, "motion_blur_enabled", True)
        if self._u_mb_enabled is not None:
            self._u_mb_enabled.value = 1 if mb_on else 0
        if self._u_mb_samples is not None:
            self._u_mb_samples.value = int(getattr(config, "motion_blur_samples", 12))
        if self._u_mb_intensity is not None:
            self._u_mb_intensity.value = float(getattr(config, "motion_blur_intensity", 1.0))
        if self._u_mb_max_radius is not None:
            self._u_mb_max_radius.value = float(getattr(config, "motion_blur_max_radius", 32.0))
        if self._u_prev_vp is not None:
            self._u_prev_vp.write(prev_vp_mat.tobytes())

        # Render full-screen procedural quad
        self.optics_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        return self.optics_texture

    def render_flare(
        self,
        scene_hdr_texture: moderngl.Texture,
        config: Any,
        target_fbo: moderngl.Framebuffer | None = None,
    ) -> moderngl.Texture:
        """Generates anamorphic streak flares, chromatic ghost reflections, and halo."""
        fbo = target_fbo if target_fbo is not None else self.flare_fbo
        fbo.use()
        fw, fh = self.flare_texture.size
        self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        scene_hdr_texture.use(location=0)

        # Upload flare uniforms
        if self._u_flare_thresh is not None:
            self._u_flare_thresh.value = float(getattr(config, "lens_flare_threshold", 1.8))
        if self._u_flare_streak_i is not None:
            self._u_flare_streak_i.value = float(getattr(config, "lens_flare_streak_intensity", 0.6))
        if self._u_flare_streak_w is not None:
            self._u_flare_streak_w.value = float(getattr(config, "lens_flare_streak_width", 32.0))
        if self._u_flare_ghost_i is not None:
            self._u_flare_ghost_i.value = float(getattr(config, "lens_flare_ghost_intensity", 0.35))
        if self._u_flare_halo_i is not None:
            self._u_flare_halo_i.value = float(getattr(config, "lens_flare_halo_intensity", 0.25))
        if self._u_flare_tint is not None:
            tint = getattr(config, "lens_flare_tint", (0.25, 0.65, 1.0))
            self._u_flare_tint.value = (float(tint[0]), float(tint[1]), float(tint[2]))
        if self._u_flare_screen is not None:
            self._u_flare_screen.value = (float(fw), float(fh))

        self.flare_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        return self.flare_texture

    def execute(self, context: Any) -> None:
        """Executes full camera optics and lens flare passes via RenderGraphContext."""
        if not self.enabled:
            return

        resources = getattr(context, "resources", {})
        hdr_fbo = resources.get("hdr_fbo")
        g_buffer = resources.get("g_buffer")
        prev_vp = resources.get("prev_vp_mat", np.identity(4, dtype=np.float32).flatten())
        config = getattr(context, "config", None)

        if hdr_fbo is not None and g_buffer is not None and config is not None:
            depth_tex = g_buffer.depth_texture
            scene_hdr_tex = hdr_fbo.color_attachments[0]

            # 1. Render Optics (DoF + Motion Blur)
            out_optics = self.render_optics(scene_hdr_tex, depth_tex, prev_vp, config)
            # Copy result back into hdr_fbo so subsequent passes read blurred HDR
            self.ctx.copy_framebuffer(hdr_fbo, self.optics_fbo)

            # 2. Render Anamorphic Flares if enabled
            if getattr(config, "lens_flare_enabled", True):
                out_flare = self.render_flare(out_optics, config)
                resources["lens_flare_texture"] = out_flare

    def destroy(self) -> None:
        """Releases all textures, framebuffers, and programs."""
        self._destroy_buffers()
        if self.optics_vao is not None:
            self.optics_vao.release()
            self.optics_vao = None
        if self.flare_vao is not None:
            self.flare_vao.release()
            self.flare_vao = None
        if self.optics_prog is not None:
            self.optics_prog.release()
            self.optics_prog = None
        if self.flare_prog is not None:
            self.flare_prog.release()
            self.flare_prog = None
