"""Adaptive Multi-Range Global Illumination (AMRGI) Pass for PyMordial Engine.

A single screen-space diffuse GI pipeline that adapts per pixel between near,
mid and far ray ranges. It is an independent implementation: it does not call,
copy, or run SSGI or RMGI.

Pipeline (two compute dispatches per frame):

1. ``amrgi_trace.comp`` — half-resolution trace. One frame-rotated cosine ray
   per active sample, continued continuously through near -> mid -> far. Writes
   raw radiance + confidence and an auxiliary (hit distance, range class,
   confidence, miss streak) texture used for next-frame adaptation.
2. ``amrgi_reconstruct.comp`` — full-resolution temporal reconstruction plus
   edge-aware spatial upscale.

Resource roles:

    raw_radiance_tex   half-res  rgb = incoming indirect radiance, a = confidence
    aux_tex            half-res  r = hit distance, g = range class, b = conf, a = miss streak
    resolved_tex       full-res  rgb = resolved radiance, a = linear view depth
    history_tex        full-res  previous resolved result (ping-pong)

Energy convention: the published texture is *incoming* indirect diffuse
radiance. The deferred resolve applies ``albedo * (1 - metallic) / PI`` exactly
once, matching the RMGI convention.
"""

from __future__ import annotations

import moderngl

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader


class AMRGIPass(RenderPass):
    """Adaptive Multi-Range Global Illumination (experimental)."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "full_width",
        "full_height",
        "resolution_scale",
        "raw_radiance_tex",
        "aux_tex",
        "prev_aux_tex",
        "resolved_tex",
        "history_tex",
        "black_fallback",
        "trace_prog",
        "reconstruct_prog",
        "_u_trace",
        "_u_recon",
        "_first_frame",
        "_target_res_tuple",
        "_half_res_tuple",
        "_full_res_tuple",
        "requested_backend",
        "active_backend",
        "fallback_reason",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        resolution_scale: float = 0.5,
    ) -> None:
        super().__init__(name="AMRGIPass", enabled=True)
        self.ctx = ctx
        self.requested_backend = "ADAPTIVE"
        self.active_backend = "ADAPTIVE"
        self.fallback_reason: str | None = None

        self.full_width = max(int(width), 1)
        self.full_height = max(int(height), 1)
        self.resolution_scale = min(max(float(resolution_scale), 0.25), 1.0)
        self.width = max(int(round(self.full_width * self.resolution_scale)), 1)
        self.height = max(int(round(self.full_height * self.resolution_scale)), 1)
        self._target_res_tuple = (float(self.width), float(self.height))
        self._half_res_tuple = (float(self.width), float(self.height))
        self._full_res_tuple = (float(self.full_width), float(self.full_height))

        self._create_targets()

        self.black_fallback = self.ctx.texture((1, 1), 4, data=b"\x00" * 8, dtype="f2")

        self.trace_prog = self.ctx.compute_shader(load_shader("amrgi_trace.comp"))
        self.reconstruct_prog = self.ctx.compute_shader(load_shader("amrgi_reconstruct.comp"))

        self._u_trace = {
            name: self.trace_prog.get(name, None)
            for name in (
                "u_TargetResolution", "u_NearEnd", "u_MidEnd", "u_FarEnd",
                "u_NearSteps", "u_MidSteps", "u_FarSteps", "u_RefineIters",
                "u_Thickness", "u_Intensity", "u_RayCount", "u_RadianceClamp",
                "u_HistoryValid", "u_MidRate", "u_FarRate",
            )
        }
        self._u_recon = {
            name: self.reconstruct_prog.get(name, None)
            for name in (
                "u_FullResolution", "u_HalfResolution", "u_TemporalBlend",
                "u_HistoryValid", "u_PrevView", "u_PrevJitter",
                "u_SpatialStrength", "u_DepthEdge", "u_NormalEdge",
            )
        }

        self._first_frame = True

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _make_tex(self, w: int, h: int) -> moderngl.Texture:
        tex = self.ctx.texture((w, h), 4, dtype="f2")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _create_targets(self) -> None:
        self.raw_radiance_tex = self._make_tex(self.width, self.height)
        self.aux_tex = self._make_tex(self.width, self.height)
        self.prev_aux_tex = self._make_tex(self.width, self.height)
        self.resolved_tex = self._make_tex(self.full_width, self.full_height)
        self.history_tex = self._make_tex(self.full_width, self.full_height)

    def _release_targets(self) -> None:
        for tex in (
            self.raw_radiance_tex, self.aux_tex, self.prev_aux_tex,
            self.resolved_tex, self.history_tex,
        ):
            tex.release()

    def invalidate_history(self) -> None:
        """Forces the next frame to discard temporal history."""
        self._first_frame = True

    def resize(self, width: int, height: int, resolution_scale: float | None = None) -> None:
        new_scale = self.resolution_scale if resolution_scale is None else min(max(float(resolution_scale), 0.25), 1.0)
        new_full_w = max(int(width), 1)
        new_full_h = max(int(height), 1)
        new_w = max(int(round(new_full_w * new_scale)), 1)
        new_h = max(int(round(new_full_h * new_scale)), 1)
        if (new_w == self.width and new_h == self.height
                and new_full_w == self.full_width and new_full_h == self.full_height
                and new_scale == self.resolution_scale):
            return

        self.resolution_scale = new_scale
        self.full_width, self.full_height = new_full_w, new_full_h
        self.width, self.height = new_w, new_h
        self._target_res_tuple = (float(self.width), float(self.height))
        self._half_res_tuple = (float(self.width), float(self.height))
        self._full_res_tuple = (float(self.full_width), float(self.full_height))

        self._release_targets()
        self._create_targets()
        self._first_frame = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = str(getattr(context.config, "gi_mode", "AMRGI")).upper()
        self.requested_backend = "ADAPTIVE"
        self.fallback_reason = None
        if gi_mode != "AMRGI":
            self.active_backend = "OFF"
            context.resources["amrgi_texture"] = self.black_fallback
            self.invalidate_history()
            return

        g_buffer = context.resources.get("g_buffer")
        if g_buffer is None:
            self.active_backend = "OFF"
            self.fallback_reason = "no G-buffer"
            context.resources["amrgi_texture"] = self.black_fallback
            self.invalidate_history()
            return

        self.active_backend = "ADAPTIVE"
        self._trace(context, g_buffer)
        self._reconstruct(context, g_buffer)

        context.resources["amrgi_texture"] = self.resolved_tex

        # Ping-pong: resolved becomes next frame's history; aux becomes prev-aux.
        self.resolved_tex, self.history_tex = self.history_tex, self.resolved_tex
        self.aux_tex, self.prev_aux_tex = self.prev_aux_tex, self.aux_tex
        self._first_frame = False

    def _trace(self, context: RenderGraphContext, g_buffer) -> None:
        cfg = context.config
        self.raw_radiance_tex.bind_to_image(0, read=False, write=True)
        self.aux_tex.bind_to_image(1, read=False, write=True)

        g_buffer.depth_texture.use(location=1)
        g_buffer.normal_metallic_texture.use(location=2)
        g_buffer.albedo_roughness_texture.use(location=3)
        g_buffer.surface_id_texture.use(location=4)

        direct_tex = context.resources.get("direct_radiance_texture", None)
        if direct_tex is None:
            direct_tex = g_buffer.albedo_roughness_texture
        direct_tex.use(location=5)
        self.prev_aux_tex.use(location=6)  # previous-frame aux (separate texture, no hazard)

        u = self._u_trace
        if u["u_TargetResolution"] is not None:
            u["u_TargetResolution"].value = self._target_res_tuple
        # Range boundaries: enforce 0 < near < mid < far with safe ordering.
        near_end = max(float(getattr(cfg, "amrgi_near_end", 3.0)), 0.1)
        mid_end = max(float(getattr(cfg, "amrgi_mid_end", 12.0)), near_end + 0.1)
        far_end = max(float(getattr(cfg, "amrgi_far_end", 40.0)), mid_end + 0.1)
        if u["u_NearEnd"] is not None:
            u["u_NearEnd"].value = near_end
        if u["u_MidEnd"] is not None:
            u["u_MidEnd"].value = mid_end
        if u["u_FarEnd"] is not None:
            u["u_FarEnd"].value = far_end
        if u["u_NearSteps"] is not None:
            u["u_NearSteps"].value = max(1, min(32, int(getattr(cfg, "amrgi_near_steps", 7))))
        if u["u_MidSteps"] is not None:
            u["u_MidSteps"].value = max(1, min(32, int(getattr(cfg, "amrgi_mid_steps", 5))))
        if u["u_FarSteps"] is not None:
            u["u_FarSteps"].value = max(1, min(32, int(getattr(cfg, "amrgi_far_steps", 4))))
        if u["u_RefineIters"] is not None:
            u["u_RefineIters"].value = max(0, min(8, int(getattr(cfg, "amrgi_near_refine_steps", 3))))
        if u["u_Thickness"] is not None:
            u["u_Thickness"].value = float(getattr(cfg, "amrgi_thickness", 0.30))
        if u["u_Intensity"] is not None:
            u["u_Intensity"].value = float(getattr(cfg, "amrgi_intensity", 1.0))
        if u["u_RayCount"] is not None:
            u["u_RayCount"].value = max(1, min(4, int(getattr(cfg, "amrgi_ray_count", 1))))
        if u["u_RadianceClamp"] is not None:
            u["u_RadianceClamp"].value = max(0.1, float(getattr(cfg, "amrgi_radiance_clamp", 4.0)))
        if u["u_HistoryValid"] is not None:
            u["u_HistoryValid"].value = 0 if self._first_frame else 1
        if u["u_MidRate"] is not None:
            u["u_MidRate"].value = max(1, int(getattr(cfg, "amrgi_mid_rate", 2)))
        if u["u_FarRate"] is not None:
            u["u_FarRate"].value = max(1, int(getattr(cfg, "amrgi_far_rate", 4)))

        gx = (self.width + 7) // 8
        gy = (self.height + 7) // 8
        self.trace_prog.run(group_x=gx, group_y=gy, group_z=1)

    def _reconstruct(self, context: RenderGraphContext, g_buffer) -> None:
        cfg = context.config
        self.resolved_tex.bind_to_image(0, read=False, write=True)

        self.raw_radiance_tex.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.normal_metallic_texture.use(location=3)
        g_buffer.surface_id_texture.use(location=4)
        self.history_tex.use(location=5)
        g_buffer.velocity_texture.use(location=6)

        u = self._u_recon
        if u["u_FullResolution"] is not None:
            u["u_FullResolution"].value = self._full_res_tuple
        if u["u_HalfResolution"] is not None:
            u["u_HalfResolution"].value = self._half_res_tuple
        if u["u_TemporalBlend"] is not None:
            u["u_TemporalBlend"].value = float(getattr(cfg, "amrgi_temporal_blend", 0.90))
        if u["u_HistoryValid"] is not None:
            u["u_HistoryValid"].value = 0 if self._first_frame else 1
        if u["u_PrevView"] is not None:
            prev_view = context.resources.get("prev_view")
            if prev_view is not None:
                u["u_PrevView"].write(prev_view.tobytes())
        if u["u_PrevJitter"] is not None:
            prev_jitter = context.resources.get("prev_jitter", (0.0, 0.0))
            u["u_PrevJitter"].value = (float(prev_jitter[0]), float(prev_jitter[1]))
        if u["u_SpatialStrength"] is not None:
            u["u_SpatialStrength"].value = max(0.0, min(1.0, float(getattr(cfg, "amrgi_spatial_strength", 1.0))))
        if u["u_DepthEdge"] is not None:
            u["u_DepthEdge"].value = max(0.1, float(getattr(cfg, "amrgi_depth_edge", 1.0)))
        if u["u_NormalEdge"] is not None:
            u["u_NormalEdge"].value = max(1.0, float(getattr(cfg, "amrgi_normal_edge", 16.0)))

        gx = (self.full_width + 7) // 8
        gy = (self.full_height + 7) // 8
        self.reconstruct_prog.run(group_x=gx, group_y=gy, group_z=1)

    def destroy(self) -> None:
        self._release_targets()
        self.black_fallback.release()
        self.trace_prog.release()
        self.reconstruct_prog.release()
