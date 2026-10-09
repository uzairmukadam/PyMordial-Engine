"""Hybrid Radiance Cascades (SSRC + FFPC) Pass for PyMordial Engine.

Implements Alexander Sannikov's Radiance Cascades adapted to 3D Deferred Lighting:
- Cascade 0: Near-field Screen-Space Radiance Cascades (SSRC) for contact occlusion and micro-bounces.
- Cascade 1: Mid-field SSRC for surface-to-surface indirect diffuse bounce.
- Cascade 2: Far-Field Probe Cascades (FFPC) for camera-independent off-screen sky and scene bounce.
- Multi-scale cascade merging and bilateral edge-preserving spatial-temporal reconstruction.
"""

from __future__ import annotations
import math
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext
from engine.gfx.shader_utils import load_shader


class RadianceCascadesPass(RenderPass):
    """Next-Generation Global Illumination via Screen-Space Radiance Cascades and Far-Field Probes."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "raw_rc_tex",
        "tex_a",
        "tex_b",
        "_current_filtered",
        "_current_history",
        "black_fallback",
        "raymarch_prog",
        "merge_prog",
        "_u_rm_target_res",
        "_u_rm_interval_c0",
        "_u_rm_interval_c1",
        "_u_rm_interval_c2",
        "_u_rm_steps_c0",
        "_u_rm_steps_c1",
        "_u_rm_thickness",
        "_u_rm_intensity",
        "_u_merge_target_res",
        "_u_merge_prev_vp",
        "_u_merge_temporal_blend",
        "prev_view_proj_mat",
        "_first_frame",
        "_target_res_tuple",
    )

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="RadianceCascadesPass", enabled=True)
        self.ctx = ctx
        self.width = max(width // 2, 1)
        self.height = max(height // 2, 1)
        self._target_res_tuple = (float(self.width), float(self.height))

        # 1. Half-resolution RGBA16F intermediate textures
        self.raw_rc_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_rc_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_rc_tex.repeat_x = False
        self.raw_rc_tex.repeat_y = False

        self.tex_a = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_a.repeat_x = False
        self.tex_a.repeat_y = False

        self.tex_b = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_b.repeat_x = False
        self.tex_b.repeat_y = False

        self._current_filtered = self.tex_a
        self._current_history = self.tex_b

        # 1x1 black fallback
        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # 2. Compile Compute Shaders
        raymarch_src = load_shader("rc_raymarch.comp")
        merge_src = load_shader("rc_merge.comp")

        self.raymarch_prog = self.ctx.compute_shader(raymarch_src)
        self.merge_prog = self.ctx.compute_shader(merge_src)

        # 3. Cache Uniform Handles
        self._u_rm_target_res = self.raymarch_prog.get("u_TargetResolution", None)
        self._u_rm_interval_c0 = self.raymarch_prog.get("u_RC_IntervalC0", None)
        self._u_rm_interval_c1 = self.raymarch_prog.get("u_RC_IntervalC1", None)
        self._u_rm_interval_c2 = self.raymarch_prog.get("u_RC_IntervalC2", None)
        self._u_rm_steps_c0 = self.raymarch_prog.get("u_RC_StepsC0", None)
        self._u_rm_steps_c1 = self.raymarch_prog.get("u_RC_StepsC1", None)
        self._u_rm_thickness = self.raymarch_prog.get("u_RC_Thickness", None)
        self._u_rm_intensity = self.raymarch_prog.get("u_RC_Intensity", None)

        self._u_merge_target_res = self.merge_prog.get("u_TargetResolution", None)
        self._u_merge_prev_vp = self.merge_prog.get("u_PrevViewProjection", None)
        self._u_merge_temporal_blend = self.merge_prog.get("u_TemporalBlend", None)

        self.prev_view_proj_mat = np.eye(4, dtype=np.float32)
        self._first_frame = True

    def resize(self, width: int, height: int) -> None:
        new_w = max(width // 2, 1)
        new_h = max(height // 2, 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h
        self._target_res_tuple = (float(self.width), float(self.height))

        self.raw_rc_tex.release()
        self.tex_a.release()
        self.tex_b.release()

        self.raw_rc_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_rc_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_rc_tex.repeat_x = False
        self.raw_rc_tex.repeat_y = False

        self.tex_a = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_a.repeat_x = False
        self.tex_a.repeat_y = False

        self.tex_b = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_b.repeat_x = False
        self.tex_b.repeat_y = False

        self._current_filtered = self.tex_a
        self._current_history = self.tex_b
        self._first_frame = True

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "HYBRID")
        if gi_mode != "RADIANCE_CASCADES":
            context.resources["rc_texture"] = self.black_fallback
            return

        g_buffer = context.resources.get("g_buffer", None)
        if g_buffer is None:
            context.resources["rc_texture"] = self.black_fallback
            return

        # 1. Dispatch Stage 1: SSRC + FFPC Interval Raymarching
        self.raw_rc_tex.bind_to_image(0, read=False, write=True)

        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        g_buffer.albedo_roughness_texture.use(location=3)

        scene_color = context.resources.get("scene_color", g_buffer.albedo_roughness_texture)
        scene_color.use(location=4)

        sky_lut = context.resources.get("sky_view_lut", None)
        if sky_lut is not None:
            sky_lut.use(location=5)
        else:
            self.black_fallback.use(location=5)

        if self._u_rm_target_res is not None:
            self._u_rm_target_res.value = self._target_res_tuple
        if self._u_rm_interval_c0 is not None:
            self._u_rm_interval_c0.value = float(getattr(context.config, "rc_interval_c0", 0.40))
        if self._u_rm_interval_c1 is not None:
            self._u_rm_interval_c1.value = float(getattr(context.config, "rc_interval_c1", 2.50))
        if self._u_rm_interval_c2 is not None:
            self._u_rm_interval_c2.value = float(getattr(context.config, "rc_interval_c2", 40.0))
        if self._u_rm_steps_c0 is not None:
            self._u_rm_steps_c0.value = int(getattr(context.config, "rc_steps_c0", 6))
        if self._u_rm_steps_c1 is not None:
            self._u_rm_steps_c1.value = int(getattr(context.config, "rc_steps_c1", 10))
        if self._u_rm_thickness is not None:
            self._u_rm_thickness.value = float(getattr(context.config, "rc_thickness", 0.30))
        if self._u_rm_intensity is not None:
            self._u_rm_intensity.value = float(getattr(context.config, "rc_intensity", 1.0))

        gx = (self.width + 7) // 8
        gy = (self.height + 7) // 8
        self.raymarch_prog.run(group_x=gx, group_y=gy, group_z=1)

        # 2. Dispatch Stage 2: Bilateral Merge and Temporal History Accumulation
        self._current_filtered.bind_to_image(0, read=False, write=True)

        self.raw_rc_tex.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.normal_metallic_texture.use(location=3)
        self._current_history.use(location=4)

        if self._u_merge_target_res is not None:
            self._u_merge_target_res.value = self._target_res_tuple

        # Update previous view projection matrix
        curr_vp = getattr(context, "view_proj_mat", None)
        if curr_vp is None and context.frame_context is not None:
            # Reconstruct from UBO 0 buffer if directly accessible
            curr_vp = context.frame_context.buffer[32:48].reshape((4, 4))

        if self._first_frame and curr_vp is not None:
            np.copyto(self.prev_view_proj_mat, curr_vp)
            self._first_frame = False

        if self._u_merge_prev_vp is not None:
            self._u_merge_prev_vp.write(self.prev_view_proj_mat.tobytes())

        if self._u_merge_temporal_blend is not None:
            blend_val = float(getattr(context.config, "rc_temporal_blend", 0.85))
            self._u_merge_temporal_blend.value = blend_val

        self.merge_prog.run(group_x=gx, group_y=gy, group_z=1)

        # Update history matrix for subsequent frame
        if curr_vp is not None:
            np.copyto(self.prev_view_proj_mat, curr_vp)

        # Publish filtered irradiance texture
        context.resources["rc_texture"] = self._current_filtered

        # Ping-pong history buffers
        self._current_filtered, self._current_history = self._current_history, self._current_filtered

    def destroy(self) -> None:
        self.raw_rc_tex.release()
        self.tex_a.release()
        self.tex_b.release()
        self.black_fallback.release()
        self.raymarch_prog.release()
        self.merge_prog.release()
