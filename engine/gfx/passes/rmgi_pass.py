"""Ray-Marched Global Illumination (RMGI) Pass for PyMordial Engine.

Per-pixel raymarch + bilateral/temporal merge (``rmgi_raymarch.comp`` +
``rmgi_merge.comp``).
"""

from __future__ import annotations

import moderngl
import numpy as np

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader


class RMGIPass(RenderPass):
    """Ray-Marched Global Illumination (per-pixel raymarch)."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "resolution_scale",
        "raw_rmgi_tex",
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
        "_u_rm_ray_count",
        "_u_rm_steps_c0",
        "_u_rm_steps_c1",
        "_u_rm_thickness",
        "_u_rm_intensity",
        "_u_merge_target_res",
        "_u_merge_temporal_blend",
        "_u_merge_history_valid",
        "_u_merge_prev_view",
        "_first_frame",
        "_target_res_tuple",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        resolution_scale: float = 1.0,
    ) -> None:
        super().__init__(name="RMGIPass", enabled=True)
        self.ctx = ctx
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(width * self.resolution_scale)), 1)
        self.height = max(int(round(height * self.resolution_scale)), 1)
        self._target_res_tuple = (float(self.width), float(self.height))

        # 1. RGBA16F intermediate textures at the configured render scale
        self.raw_rmgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_rmgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_rmgi_tex.repeat_x = False
        self.raw_rmgi_tex.repeat_y = False

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
        raymarch_src = load_shader("rmgi_raymarch.comp")
        merge_src = load_shader("rmgi_merge.comp")

        self.raymarch_prog = self.ctx.compute_shader(raymarch_src)
        self.merge_prog = self.ctx.compute_shader(merge_src)

        # 3. Cache Uniform Handles
        self._u_rm_target_res = self.raymarch_prog.get("u_TargetResolution", None)
        self._u_rm_interval_c0 = self.raymarch_prog.get("u_RMGI_IntervalC0", None)
        self._u_rm_interval_c1 = self.raymarch_prog.get("u_RMGI_IntervalC1", None)
        self._u_rm_interval_c2 = self.raymarch_prog.get("u_RMGI_IntervalC2", None)
        self._u_rm_ray_count = self.raymarch_prog.get("u_RMGI_RayCount", None)
        self._u_rm_steps_c0 = self.raymarch_prog.get("u_RMGI_StepsC0", None)
        self._u_rm_steps_c1 = self.raymarch_prog.get("u_RMGI_StepsC1", None)
        self._u_rm_thickness = self.raymarch_prog.get("u_RMGI_Thickness", None)
        self._u_rm_intensity = self.raymarch_prog.get("u_RMGI_Intensity", None)

        self._u_merge_target_res = self.merge_prog.get("u_TargetResolution", None)
        self._u_merge_temporal_blend = self.merge_prog.get("u_TemporalBlend", None)
        self._u_merge_history_valid = self.merge_prog.get("u_HistoryValid", None)
        self._u_merge_prev_view = self.merge_prog.get("u_PrevView", None)

        self._first_frame = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int, resolution_scale: float | None = None) -> None:
        new_scale = self.resolution_scale if resolution_scale is None else min(max(float(resolution_scale), 0.25), 1.0)
        new_w = max(int(round(width * new_scale)), 1)
        new_h = max(int(round(height * new_scale)), 1)
        if new_w == self.width and new_h == self.height and new_scale == self.resolution_scale:
            return

        self.resolution_scale = new_scale
        self.width = new_w
        self.height = new_h
        self._target_res_tuple = (float(self.width), float(self.height))

        self.raw_rmgi_tex.release()
        self.tex_a.release()
        self.tex_b.release()

        self.raw_rmgi_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.raw_rmgi_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_rmgi_tex.repeat_x = False
        self.raw_rmgi_tex.repeat_y = False

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

    def invalidate_history(self) -> None:
        self._first_frame = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "RMGI")
        if gi_mode != "RMGI":
            context.resources["rmgi_texture"] = self.black_fallback
            self.invalidate_history()
            return

        g_buffer = context.resources.get("g_buffer", None)
        if g_buffer is None:
            context.resources["rmgi_texture"] = self.black_fallback
            self.invalidate_history()
            return

        self._execute_legacy(context, g_buffer)

    # ------------------------------------------------------------------
    # Legacy path (unchanged)
    # ------------------------------------------------------------------

    def _execute_legacy(self, context: RenderGraphContext, g_buffer) -> None:
        # 1. Dispatch Stage 1: SSRC + FFPC Interval Raymarching
        self.raw_rmgi_tex.bind_to_image(0, read=False, write=True)

        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        g_buffer.albedo_roughness_texture.use(location=3)
        g_buffer.surface_id_texture.use(location=4)

        if self._u_rm_target_res is not None:
            self._u_rm_target_res.value = self._target_res_tuple
        if self._u_rm_interval_c0 is not None:
            self._u_rm_interval_c0.value = float(getattr(context.config, "rmgi_interval_c0", 0.40))
        if self._u_rm_interval_c1 is not None:
            self._u_rm_interval_c1.value = float(getattr(context.config, "rmgi_interval_c1", 2.50))
        if self._u_rm_interval_c2 is not None:
            self._u_rm_interval_c2.value = float(getattr(context.config, "rmgi_interval_c2", 40.0))
        if self._u_rm_ray_count is not None:
            self._u_rm_ray_count.value = max(4, min(32, int(getattr(context.config, "rmgi_ray_count", 16))))
        if self._u_rm_steps_c0 is not None:
            self._u_rm_steps_c0.value = int(getattr(context.config, "rmgi_steps_c0", 8))
        if self._u_rm_steps_c1 is not None:
            self._u_rm_steps_c1.value = int(getattr(context.config, "rmgi_steps_c1", 12))
        if self._u_rm_thickness is not None:
            self._u_rm_thickness.value = float(getattr(context.config, "rmgi_thickness", 0.15))
        if self._u_rm_intensity is not None:
            self._u_rm_intensity.value = float(getattr(context.config, "rmgi_intensity", 1.0))

        gx = (self.width + 7) // 8
        gy = (self.height + 7) // 8
        self.raymarch_prog.run(group_x=gx, group_y=gy, group_z=1)

        # 2. Dispatch Stage 2: Bilateral Merge and Temporal History Accumulation
        self._current_filtered.bind_to_image(0, read=False, write=True)

        self.raw_rmgi_tex.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.normal_metallic_texture.use(location=3)
        self._current_history.use(location=4)

        # Bind motion vectors for temporal reprojection.
        g_buffer.velocity_texture.use(location=5)

        if self._u_merge_target_res is not None:
            self._u_merge_target_res.value = self._target_res_tuple

        history_valid = not self._first_frame
        self._first_frame = False

        if self._u_merge_temporal_blend is not None:
            blend_val = float(getattr(context.config, "rmgi_temporal_blend", 0.96))
            self._u_merge_temporal_blend.value = blend_val
        if self._u_merge_history_valid is not None:
            self._u_merge_history_valid.value = int(history_valid)
        if self._u_merge_prev_view is not None:
            previous_view = context.resources.get("prev_view")
            if previous_view is None:
                raise RuntimeError("Ray-Marched Global Illumination requires the previous frame view matrix.")
            self._u_merge_prev_view.write(previous_view.tobytes())

        self.merge_prog.run(group_x=gx, group_y=gy, group_z=1)

        # Publish filtered irradiance texture
        context.resources["rmgi_texture"] = self._current_filtered

        # Ping-pong history buffers
        self._current_filtered, self._current_history = self._current_history, self._current_filtered

    def destroy(self) -> None:
        self.raw_rmgi_tex.release()
        self.tex_a.release()
        self.tex_b.release()
        self.black_fallback.release()
        self.raymarch_prog.release()
        self.merge_prog.release()
