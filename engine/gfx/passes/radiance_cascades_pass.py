"""Radiance Cascades Pass for PyMordial Engine.

Two implementations coexist:

* **LEGACY** — the original per-pixel raymarch + bilateral/temporal merge
  (``rc_raymarch.comp`` + ``rc_merge.comp``). Unchanged and still the default.
* **CASCADES** — explicit screen-space radiance cascades:

      1. Trace intervals   (``rc_trace_intervals.comp``)  per probe/direction
      2. Merge cascades    (``rc_merge_cascades.comp``)   C2 -> C1 -> C0
      3. Reconstruct       (``rc_reconstruct.comp``)      per-pixel irradiance
      4. Spatial filter    (``rc_spatial_filter.comp``)   edge-aware
      5. Temporal resolve  (``rc_temporal.comp``)         exactly once
      6. Publish + history update

The cascade resource model lives in :mod:`engine.gfx.passes.rc_cascade`.
"""

from __future__ import annotations

import moderngl

from engine.gfx.passes.rc_cascade import (
    RC_CASCADE_DIRECTIONS,
    RC_CASCADE_SPACING,
    RCCascade,
    build_cascades,
)
from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader


class RadianceCascadesPass(RenderPass):
    """Screen-space radiance cascades with a legacy per-pixel fallback."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "resolution_scale",
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
        # Cascade path
        "cascades",
        "trace_prog",
        "merge_cascades_prog",
        "reconstruct_prog",
        "spatial_prog",
        "temporal_prog",
        "angular_prog",
        "angular_tex",
        "angular_fbo",
        "_u_angular_probe_size",
        "_u_angular_direction_count",
        "irradiance_tex",
        "irradiance_fbo",
        "spatial_tex",
        "spatial_fbo",
        "resolved_tex",
        "resolved_fbo",
        "history_tex",
        "history_fbo",
        "_u_trace_target_res",
        "_u_trace_probe_size",
        "_u_trace_probe_height",
        "_u_trace_direction_count",
        "_u_trace_interval",
        "_u_trace_interval_end",
        "_u_trace_global_max_distance",
        "_u_trace_thickness",
        "_u_trace_max_iters",
        "_u_trace_intensity",
        "_u_merge_near_size",
        "_u_merge_far_size",
        "_u_merge_direction_count",
        "_u_merge_far_direction_count",
        "_u_merge_max_cascade",
        "_u_recon_target_res",
        "_u_recon_probe_size",
        "_u_recon_direction_count",
        "_u_recon_debug",
        "_u_spatial_target_res",
        "_u_temporal_target_res",
        "_u_temporal_prev_view",
        "_u_temporal_blend",
        "_u_temporal_history_valid",
        "_u_temporal_enabled",
        "_u_temporal_debug",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        resolution_scale: float = 1.0,
    ) -> None:
        super().__init__(name="RadianceCascadesPass", enabled=True)
        self.ctx = ctx
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(width * self.resolution_scale)), 1)
        self.height = max(int(round(height * self.resolution_scale)), 1)
        self._target_res_tuple = (float(self.width), float(self.height))

        # 1. RGBA16F intermediate textures at the configured render scale
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
        self._u_rm_ray_count = self.raymarch_prog.get("u_RC_RayCount", None)
        self._u_rm_steps_c0 = self.raymarch_prog.get("u_RC_StepsC0", None)
        self._u_rm_steps_c1 = self.raymarch_prog.get("u_RC_StepsC1", None)
        self._u_rm_thickness = self.raymarch_prog.get("u_RC_Thickness", None)
        self._u_rm_intensity = self.raymarch_prog.get("u_RC_Intensity", None)

        self._u_merge_target_res = self.merge_prog.get("u_TargetResolution", None)
        self._u_merge_temporal_blend = self.merge_prog.get("u_TemporalBlend", None)
        self._u_merge_history_valid = self.merge_prog.get("u_HistoryValid", None)
        self._u_merge_prev_view = self.merge_prog.get("u_PrevView", None)

        self._first_frame = True

        # 4. Cascade path resources and programs
        self.cascades: list[RCCascade] = []
        self.trace_prog = self.ctx.compute_shader(load_shader("rc_trace_intervals.comp"))
        self.merge_cascades_prog = self.ctx.compute_shader(load_shader("rc_merge_cascades.comp"))
        self.reconstruct_prog = self.ctx.compute_shader(load_shader("rc_reconstruct.comp"))
        self.spatial_prog = self.ctx.compute_shader(load_shader("rc_spatial_filter.comp"))
        self.temporal_prog = self.ctx.compute_shader(load_shader("rc_temporal.comp"))
        self.angular_prog = self.ctx.compute_shader(load_shader("rc_angular_filter.comp"))

        self._u_trace_target_res = self.trace_prog.get("u_TargetResolution", None)
        self._u_trace_probe_size = self.trace_prog.get("u_ProbeWidth", None)
        self._u_trace_probe_height = self.trace_prog.get("u_ProbeHeight", None)
        self._u_trace_direction_count = self.trace_prog.get("u_DirectionCount", None)
        self._u_trace_interval = self.trace_prog.get("u_IntervalStart", None)
        self._u_trace_interval_end = self.trace_prog.get("u_IntervalEnd", None)
        self._u_trace_global_max_distance = self.trace_prog.get("u_GlobalMaxDistance", None)
        self._u_trace_thickness = self.trace_prog.get("u_Thickness", None)
        self._u_trace_max_iters = self.trace_prog.get("u_MaxIterations", None)
        self._u_trace_intensity = self.trace_prog.get("u_Intensity", None)

        self._u_merge_near_size = self.merge_cascades_prog.get("u_NearProbeSize", None)
        self._u_merge_far_size = self.merge_cascades_prog.get("u_FarProbeSize", None)
        self._u_merge_direction_count = self.merge_cascades_prog.get("u_DirectionCount", None)
        self._u_merge_far_direction_count = self.merge_cascades_prog.get("u_FarDirectionCount", None)
        self._u_merge_max_cascade = self.merge_cascades_prog.get("u_RCMergeMaxCascade", None)

        self._u_recon_target_res = self.reconstruct_prog.get("u_TargetResolution", None)
        self._u_recon_probe_size = self.reconstruct_prog.get("u_ProbeSize", None)
        self._u_recon_direction_count = self.reconstruct_prog.get("u_DirectionCount", None)
        self._u_recon_debug = self.reconstruct_prog.get("u_DebugView", None)

        self._u_spatial_target_res = self.spatial_prog.get("u_TargetResolution", None)

        self._u_angular_probe_size = self.angular_prog.get("u_ProbeSize", None)
        self._u_angular_direction_count = self.angular_prog.get("u_DirectionCount", None)

        self._u_temporal_target_res = self.temporal_prog.get("u_TargetResolution", None)
        self._u_temporal_prev_view = self.temporal_prog.get("u_PrevView", None)
        self._u_temporal_blend = self.temporal_prog.get("u_TemporalBlend", None)
        self._u_temporal_history_valid = self.temporal_prog.get("u_HistoryValid", None)
        self._u_temporal_enabled = self.temporal_prog.get("u_TemporalEnabled", None)
        self._u_temporal_debug = self.temporal_prog.get("u_DebugView", None)

        self._create_cascade_targets()

    # ------------------------------------------------------------------
    # Cascade targets
    # ------------------------------------------------------------------

    def _cascade_intervals(self) -> tuple[tuple[float, float], ...]:
        """Returns the (start, end) distance interval for each cascade.

        The near cascade must cover the range that actually carries visible
        light, not just a thin slab. With a 0.4 m near interval almost all
        visible light came from the coarser far cascades, whose bilinear merge
        smeared contact detail and object bases. Giving the sharp 4 px probe
        grid the whole 0-6 m near field keeps it responsible for the light that
        defines contact and crevices; the far cascades only add distant detail.
        """
        return (
            (0.0, 6.00),
            (6.00, 20.0),
            (20.0, 40.0),
        )

    def _create_cascade_targets(self) -> None:
        self.cascades = build_cascades(
            self.ctx,
            self.width,
            self.height,
            self._cascade_intervals(),
            spacings=RC_CASCADE_SPACING,
            directions=RC_CASCADE_DIRECTIONS,
        )

        def _tex() -> moderngl.Texture:
            tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            return tex

        self.irradiance_tex = _tex()
        self.spatial_tex = _tex()
        self.resolved_tex = _tex()
        self.history_tex = _tex()

        # Angular filter target: same layout as the C0 merged texture.
        c0 = self.cascades[0]
        self.angular_tex = self.ctx.texture_array(
            (c0.probe_width, c0.probe_height, c0.direction_count), 4, dtype="f2"
        )
        self.angular_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.angular_tex.repeat_x = False
        self.angular_tex.repeat_y = False
        self.angular_fbo = self.ctx.framebuffer(color_attachments=[self.angular_tex])

        self.irradiance_fbo = self.ctx.framebuffer(color_attachments=[self.irradiance_tex])
        self.spatial_fbo = self.ctx.framebuffer(color_attachments=[self.spatial_tex])
        self.resolved_fbo = self.ctx.framebuffer(color_attachments=[self.resolved_tex])
        self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_tex])

    def _release_cascade_targets(self) -> None:
        for cascade in self.cascades:
            cascade.release()
        self.cascades = []
        for fbo in (self.irradiance_fbo, self.spatial_fbo, self.resolved_fbo, self.history_fbo, self.angular_fbo):
            fbo.release()
        for tex in (self.irradiance_tex, self.spatial_tex, self.resolved_tex, self.history_tex, self.angular_tex):
            tex.release()

    def cascade_metadata(self) -> list[dict[str, object]]:
        """Returns per-cascade metadata for debugging and tests."""
        return [cascade.describe() for cascade in self.cascades]

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

        self._release_cascade_targets()
        self._create_cascade_targets()

        self._first_frame = True

    def invalidate_history(self) -> None:
        self._first_frame = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "RADIANCE_CASCADES")
        if gi_mode != "RADIANCE_CASCADES":
            context.resources["rc_texture"] = self.black_fallback
            self.invalidate_history()
            return

        g_buffer = context.resources.get("g_buffer", None)
        if g_buffer is None:
            context.resources["rc_texture"] = self.black_fallback
            self.invalidate_history()
            return

        backend = str(getattr(context.config, "rc_backend", "LEGACY")).upper()
        if backend == "CASCADES":
            self._execute_cascades(context, g_buffer)
        else:
            self._execute_legacy(context, g_buffer)

    # ------------------------------------------------------------------
    # Legacy path (unchanged)
    # ------------------------------------------------------------------

    def _execute_legacy(self, context: RenderGraphContext, g_buffer) -> None:
        # 1. Dispatch Stage 1: SSRC + FFPC Interval Raymarching
        self.raw_rc_tex.bind_to_image(0, read=False, write=True)

        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        g_buffer.albedo_roughness_texture.use(location=3)
        g_buffer.surface_id_texture.use(location=4)

        if self._u_rm_target_res is not None:
            self._u_rm_target_res.value = self._target_res_tuple
        if self._u_rm_interval_c0 is not None:
            self._u_rm_interval_c0.value = float(getattr(context.config, "rc_interval_c0", 0.40))
        if self._u_rm_interval_c1 is not None:
            self._u_rm_interval_c1.value = float(getattr(context.config, "rc_interval_c1", 2.50))
        if self._u_rm_interval_c2 is not None:
            self._u_rm_interval_c2.value = float(getattr(context.config, "rc_interval_c2", 40.0))
        if self._u_rm_ray_count is not None:
            self._u_rm_ray_count.value = max(4, min(32, int(getattr(context.config, "rc_ray_count", 16))))
        if self._u_rm_steps_c0 is not None:
            self._u_rm_steps_c0.value = int(getattr(context.config, "rc_steps_c0", 8))
        if self._u_rm_steps_c1 is not None:
            self._u_rm_steps_c1.value = int(getattr(context.config, "rc_steps_c1", 12))
        if self._u_rm_thickness is not None:
            self._u_rm_thickness.value = float(getattr(context.config, "rc_thickness", 0.15))
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

        # Bind motion vectors for temporal reprojection.
        g_buffer.velocity_texture.use(location=5)

        if self._u_merge_target_res is not None:
            self._u_merge_target_res.value = self._target_res_tuple

        history_valid = not self._first_frame
        self._first_frame = False

        if self._u_merge_temporal_blend is not None:
            blend_val = float(getattr(context.config, "rc_temporal_blend", 0.96))
            self._u_merge_temporal_blend.value = blend_val
        if self._u_merge_history_valid is not None:
            self._u_merge_history_valid.value = int(history_valid)
        if self._u_merge_prev_view is not None:
            previous_view = context.resources.get("prev_view")
            if previous_view is None:
                raise RuntimeError("Radiance Cascades requires the previous frame view matrix.")
            self._u_merge_prev_view.write(previous_view.tobytes())

        self.merge_prog.run(group_x=gx, group_y=gy, group_z=1)

        # Publish filtered irradiance texture
        context.resources["rc_texture"] = self._current_filtered

        # Ping-pong history buffers
        self._current_filtered, self._current_history = self._current_history, self._current_filtered

    # ------------------------------------------------------------------
    # Cascade path
    # ------------------------------------------------------------------

    def _execute_cascades(self, context: RenderGraphContext, g_buffer) -> None:
        config = context.config
        radiance_tex = context.resources.get("direct_radiance_texture", None)
        if radiance_tex is None:
            radiance_tex = g_buffer.albedo_roughness_texture
        hiz_bounds_tex = context.resources.get("hiz_bounds_texture", None)
        if hiz_bounds_tex is None:
            # Without the bounds hierarchy the tracer cannot run; publish black.
            context.resources["rc_texture"] = self.black_fallback
            self.invalidate_history()
            return

        thickness = float(getattr(config, "rc_thickness", 0.15))
        max_iters = int(getattr(config, "rc_hiz_max_iterations", 64))
        intensity = float(getattr(config, "rc_intensity", 1.0))

        # 1. Trace intervals, one invocation per probe/direction/cascade.
        for cascade in self.cascades:
            cascade.interval_texture.bind_to_image(0, read=False, write=True, level=0)
            g_buffer.depth_texture.use(location=1)
            g_buffer.normal_metallic_texture.use(location=2)
            radiance_tex.use(location=3)
            hiz_bounds_tex.use(location=4)
            g_buffer.surface_id_texture.use(location=5)

            if self._u_trace_target_res is not None:
                self._u_trace_target_res.value = self._target_res_tuple
            if self._u_trace_probe_size is not None:
                self._u_trace_probe_size.value = cascade.probe_width
            if self._u_trace_probe_height is not None:
                self._u_trace_probe_height.value = cascade.probe_height
            if self._u_trace_direction_count is not None:
                self._u_trace_direction_count.value = cascade.direction_count
            if self._u_trace_interval is not None:
                self._u_trace_interval.value = cascade.interval_start
            if self._u_trace_interval_end is not None:
                self._u_trace_interval_end.value = cascade.interval_end
            if self._u_trace_global_max_distance is not None:
                # Far end of the whole RC system, not this cascade's interval.
                self._u_trace_global_max_distance.value = self._cascade_intervals()[-1][1]
            if self._u_trace_thickness is not None:
                self._u_trace_thickness.value = thickness
            if self._u_trace_max_iters is not None:
                self._u_trace_max_iters.value = max_iters
            if self._u_trace_intensity is not None:
                self._u_trace_intensity.value = intensity

            gx, gy, gz = cascade.dispatch_groups
            self.trace_prog.run(group_x=gx, group_y=gy, group_z=gz)

        self.ctx.memory_barrier(moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 2. Merge far -> near: C2 into C1, then merged C1 into C0.
        #
        # The deepest cascade has no merged texture of its own (nothing merges
        # into it), so its traced interval texture is the far source for the
        # first merge. Every subsequent merge consumes the merged texture the
        # previous iteration just wrote.
        for near_index in range(len(self.cascades) - 2, -1, -1):
            near = self.cascades[near_index]
            far = self.cascades[near_index + 1]
            far_source = far.merged_texture if near_index + 1 < len(self.cascades) - 1 else far.interval_texture

            near.merged_texture.bind_to_image(0, read=False, write=True, level=0)
            near.interval_texture.use(location=1)
            far_source.use(location=2)

            if self._u_merge_near_size is not None:
                self._u_merge_near_size.value = (near.probe_width, near.probe_height)
            if self._u_merge_far_size is not None:
                self._u_merge_far_size.value = (far.probe_width, far.probe_height)
            if self._u_merge_direction_count is not None:
                self._u_merge_direction_count.value = near.direction_count
            if self._u_merge_far_direction_count is not None:
                self._u_merge_far_direction_count.value = far.direction_count
            if self._u_merge_max_cascade is not None:
                # Cascade isolation: admit this far cascade only when its index
                # is within the configured limit (0 = C0 only, 1 = C1 into C0,
                # 2 = full C2 -> C1 -> C0 chain).
                max_cascade = int(getattr(config, "rc_merge_max_cascade", 2))
                self._u_merge_max_cascade.value = 1 if (near_index + 1) <= max_cascade else 0

            gx, gy, gz = near.dispatch_groups
            self.merge_cascades_prog.run(group_x=gx, group_y=gy, group_z=gz)

            self.ctx.memory_barrier(
                moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT
            )

        # 3. Angular filter on merged C0: smooths the coarse azimuth/z bins so a
        # thin occluder does not become a hard radial wedge in reconstruction.
        c0 = self.cascades[0]
        angular_enabled = bool(getattr(config, "rc_angular_filter_enabled", True))
        if angular_enabled:
            self.angular_tex.bind_to_image(0, read=False, write=True, level=0)
            c0.merged_texture.use(location=1)
            if self._u_angular_probe_size is not None:
                self._u_angular_probe_size.value = (c0.probe_width, c0.probe_height)
            if self._u_angular_direction_count is not None:
                self._u_angular_direction_count.value = c0.direction_count
            agx, agy, agz = c0.dispatch_groups
            self.angular_prog.run(group_x=agx, group_y=agy, group_z=agz)
            self.ctx.memory_barrier(
                moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT
            )
            c0_source = self.angular_tex
        else:
            c0_source = c0.merged_texture

        # 4. Reconstruct per-pixel irradiance from merged C0.
        self.irradiance_tex.bind_to_image(0, read=False, write=True, level=0)
        c0_source.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.normal_metallic_texture.use(location=3)

        if self._u_recon_target_res is not None:
            self._u_recon_target_res.value = self._target_res_tuple
        if self._u_recon_probe_size is not None:
            self._u_recon_probe_size.value = (c0.probe_width, c0.probe_height)
        if self._u_recon_direction_count is not None:
            self._u_recon_direction_count.value = c0.direction_count
        if self._u_recon_debug is not None:
            self._u_recon_debug.value = int(getattr(config, "rc_debug_view", 0))

        gx = (self.width + 7) // 8
        gy = (self.height + 7) // 8
        self.reconstruct_prog.run(group_x=gx, group_y=gy, group_z=1)

        self.ctx.memory_barrier(moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 4. Edge-aware spatial filter (skippable for A/B comparison).
        spatial_enabled = bool(getattr(config, "rc_spatial_filter_enabled", True))
        if spatial_enabled:
            self.spatial_tex.bind_to_image(0, read=False, write=True, level=0)
            self.irradiance_tex.use(location=1)
            g_buffer.depth_texture.use(location=2)
            g_buffer.normal_metallic_texture.use(location=3)
            if self._u_spatial_target_res is not None:
                self._u_spatial_target_res.value = self._target_res_tuple
            self.spatial_prog.run(group_x=gx, group_y=gy, group_z=1)

            self.ctx.memory_barrier(
                moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT
            )
            current_irradiance_tex = self.spatial_tex
        else:
            # Bypass: hand the unfiltered reconstruction to the next stage.
            current_irradiance_tex = self.irradiance_tex

        # 5. Temporal resolve, exactly once.
        prev_view = context.resources.get("prev_view", None)
        history_valid = (not self._first_frame) and prev_view is not None

        self.resolved_tex.bind_to_image(0, read=False, write=True, level=0)
        current_irradiance_tex.use(location=1)
        self.history_tex.use(location=2)
        g_buffer.depth_texture.use(location=3)
        g_buffer.normal_metallic_texture.use(location=4)
        g_buffer.velocity_texture.use(location=5)

        if self._u_temporal_target_res is not None:
            self._u_temporal_target_res.value = self._target_res_tuple
        if self._u_temporal_blend is not None:
            self._u_temporal_blend.value = float(getattr(config, "rc_temporal_blend", 0.96))
        if self._u_temporal_history_valid is not None:
            self._u_temporal_history_valid.value = 1 if history_valid else 0
        if self._u_temporal_enabled is not None:
            self._u_temporal_enabled.value = 1 if getattr(config, "rc_temporal_enabled", True) else 0
        if self._u_temporal_debug is not None:
            self._u_temporal_debug.value = int(getattr(config, "rc_temporal_debug_view", 0))
        if self._u_temporal_prev_view is not None and prev_view is not None:
            self._u_temporal_prev_view.write(prev_view.tobytes())

        self.temporal_prog.run(group_x=gx, group_y=gy, group_z=1)

        self.ctx.memory_barrier(moderngl.TEXTURE_FETCH_BARRIER_BIT | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 6. Publish and update history.
        context.resources["rc_texture"] = self.resolved_tex
        self.ctx.copy_framebuffer(self.history_fbo, self.resolved_fbo)
        self._first_frame = False

    def destroy(self) -> None:
        self._release_cascade_targets()
        self.raw_rc_tex.release()
        self.tex_a.release()
        self.tex_b.release()
        self.black_fallback.release()
        self.raymarch_prog.release()
        self.merge_prog.release()
        self.trace_prog.release()
        self.merge_cascades_prog.release()
        self.reconstruct_prog.release()
        self.spatial_prog.release()
        self.temporal_prog.release()
        self.angular_prog.release()
