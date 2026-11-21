"""GPU Auto-Exposure: log-luminance histogram and temporal adaptation.

Replaces the constant ``RenderConfig.exposure`` with a camera-like exposure
controller:

1. **Histogram** — a 64x64 dispatch samples the HDR scene, converts each sample
   to log2 luminance, and atomically accumulates a 256-bin histogram.
2. **Adapt** — a single-thread dispatch finds the bin at the configured
   percentile (the "key" luminance), converts it to EV100, and adapts the stored
   exposure toward it with asymmetric speeds.

The adapted EV is published as ``resources["exposure_ev"]`` and consumed by
``PostProcessPipeline``. When the pass is disabled the pipeline falls back to
``RenderConfig.exposure``, so the feature is inert until a project opts in.
"""

from __future__ import annotations

import struct

import moderngl

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader

#: SSBO binding used for the 256-bin histogram. Chosen to sit above the
#: transform/material/light/decal/AABB/visibility bindings (1-6) and the
#: previous-transform buffer (7) reserved by the motion-vector pass.
HISTOGRAM_BINDING = 10

#: Dispatch grid for the histogram build. 64x64 = 4096 samples, which is a
#: ~4x4 pixel stride at 1080p. Exposure is a low-pass signal, so this is
#: indistinguishable from a full-resolution read at a fraction of the cost.
HISTOGRAM_GRID = 64

HISTOGRAM_BINS = 256


class ExposurePass(RenderPass):
    """Builds a log-luminance histogram and adapts the exposure toward it."""

    __slots__ = (
        "ctx",
        "histogram_buffer",
        "exposure_textures",
        "exposure_fbos",
        "_read_index",
        "_write_index",
        "histogram_prog",
        "adapt_prog",
        "_u_hist_screen",
        "_u_hist_min_log",
        "_u_hist_max_log",
        "_u_hist_bins",
        "_u_adapt_min_log",
        "_u_adapt_max_log",
        "_u_adapt_bins",
        "_u_adapt_low_percentile",
        "_u_adapt_high_percentile",
        "_u_adapt_middle_gray",
        "_u_adapt_min_ev",
        "_u_adapt_max_ev",
        "_u_adapt_speed_up",
        "_u_adapt_speed_down",
        "_u_adapt_dt",
        "_u_adapt_history_valid",
        "_history_valid",
        "_adapted_ev",
        "_target_ev",
        "_key_luminance",
        "_readback_scratch",
    )

    def __init__(self, ctx: moderngl.Context) -> None:
        super().__init__(name="ExposurePass", enabled=True)
        self.ctx = ctx

        # 256 x uint32 = 1 KB. Cleared every frame with a single buffer write.
        self.histogram_buffer = ctx.buffer(reserve=HISTOGRAM_BINS * 4)
        self.histogram_buffer.bind_to_storage_buffer(HISTOGRAM_BINDING)

        # 1x1 ping-pong so the adapt pass can read last frame's EV while
        # writing this frame's.
        self.exposure_textures: list[moderngl.Texture] = []
        self.exposure_fbos: list[moderngl.Framebuffer] = []
        for _ in range(2):
            tex = ctx.texture((1, 1), 4, dtype="f2")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = False
            tex.repeat_y = False
            self.exposure_textures.append(tex)
            self.exposure_fbos.append(ctx.framebuffer(color_attachments=[tex]))

        self._read_index = 0
        self._write_index = 1

        self.histogram_prog = ctx.compute_shader(load_shader("exposure_histogram.comp"))
        self.adapt_prog = ctx.compute_shader(load_shader("exposure_adapt.comp"))

        # Samplers default to texture unit 0, which the histogram pass leaves
        # bound to the scene. Without this the adapt pass would read scene
        # radiance as "previous exposure".
        if "u_PreviousExposure" in self.adapt_prog:
            self.adapt_prog["u_PreviousExposure"].value = 1

        self._u_hist_screen = self.histogram_prog.get("u_ScreenSize", None)
        self._u_hist_min_log = self.histogram_prog.get("u_MinLogLum", None)
        self._u_hist_max_log = self.histogram_prog.get("u_MaxLogLum", None)
        self._u_hist_bins = self.histogram_prog.get("u_HistogramBins", None)

        self._u_adapt_min_log = self.adapt_prog.get("u_MinLogLum", None)
        self._u_adapt_max_log = self.adapt_prog.get("u_MaxLogLum", None)
        self._u_adapt_bins = self.adapt_prog.get("u_HistogramBins", None)
        self._u_adapt_low_percentile = self.adapt_prog.get("u_LowPercentile", None)
        self._u_adapt_high_percentile = self.adapt_prog.get("u_HighPercentile", None)
        self._u_adapt_middle_gray = self.adapt_prog.get("u_MiddleGray", None)
        self._u_adapt_min_ev = self.adapt_prog.get("u_MinEV", None)
        self._u_adapt_max_ev = self.adapt_prog.get("u_MaxEV", None)
        self._u_adapt_speed_up = self.adapt_prog.get("u_AdaptSpeedUp", None)
        self._u_adapt_speed_down = self.adapt_prog.get("u_AdaptSpeedDown", None)
        self._u_adapt_dt = self.adapt_prog.get("u_DeltaTime", None)
        self._u_adapt_history_valid = self.adapt_prog.get("u_HistoryValid", None)

        self._history_valid = False
        self._adapted_ev = 0.0
        self._target_ev = 0.0
        self._key_luminance = 0.0

        # Reused every frame so the readback allocates nothing.
        self._readback_scratch = bytearray(8)

    # ---------------- Public surface ----------------

    @property
    def adapted_ev(self) -> float:
        """Exposure value from the most recent frame (0.0 until the first run)."""
        return self._adapted_ev

    @property
    def target_ev(self) -> float:
        """The EV the controller is currently adapting toward."""
        return self._target_ev

    @property
    def key_luminance(self) -> float:
        """Representative scene luminance (cd/m^2) from the last histogram."""
        return self._key_luminance

    @property
    def exposure_multiplier(self) -> float:
        """Linear multiplier equivalent to the adapted EV."""
        return 2.0**self._adapted_ev

    def invalidate_history(self) -> None:
        """Forces the next frame to snap to the target instead of adapting."""
        self._history_valid = False

    def resize(self, width: int, height: int) -> None:
        """No-op: the histogram grid and the 1x1 targets are resolution independent."""

    # ---------------- Execution ----------------

    def execute(self, context: RenderGraphContext) -> None:
        config = context.config
        if not bool(getattr(config, "auto_exposure_enabled", False)):
            context.resources["exposure_ev"] = None
            return

        scene = context.resources.get("hdr_color")
        if scene is None:
            context.resources["exposure_ev"] = None
            return

        min_log = float(getattr(config, "exposure_min_log_luminance", -12.0))
        max_log = float(getattr(config, "exposure_max_log_luminance", 16.0))

        # ---- 1. Clear the histogram ----
        # A zeroed buffer is cheaper and more predictable than a compute clear.
        self.histogram_buffer.write(bytes(HISTOGRAM_BINS * 4))
        self.histogram_buffer.bind_to_storage_buffer(HISTOGRAM_BINDING)

        # ---- 2. Build the histogram ----
        scene.use(location=0)
        if self._u_hist_screen is not None:
            self._u_hist_screen.value = (float(context.width), float(context.height))
        if self._u_hist_min_log is not None:
            self._u_hist_min_log.value = min_log
        if self._u_hist_max_log is not None:
            self._u_hist_max_log.value = max_log
        if self._u_hist_bins is not None:
            self._u_hist_bins.value = HISTOGRAM_BINS

        groups = (HISTOGRAM_GRID + 7) // 8
        self.histogram_prog.run(group_x=groups, group_y=groups, group_z=1)
        self.ctx.memory_barrier()

        # ---- 3. Adapt the exposure ----
        self.exposure_textures[self._read_index].use(location=1)
        self.exposure_textures[self._write_index].bind_to_image(0, read=False, write=True)

        if self._u_adapt_min_log is not None:
            self._u_adapt_min_log.value = min_log
        if self._u_adapt_max_log is not None:
            self._u_adapt_max_log.value = max_log
        if self._u_adapt_bins is not None:
            self._u_adapt_bins.value = HISTOGRAM_BINS
        if self._u_adapt_low_percentile is not None:
            self._u_adapt_low_percentile.value = float(
                getattr(config, "exposure_low_percentile", 0.10)
            )
        if self._u_adapt_high_percentile is not None:
            self._u_adapt_high_percentile.value = float(
                getattr(config, "exposure_high_percentile", 0.90)
            )
        if self._u_adapt_middle_gray is not None:
            self._u_adapt_middle_gray.value = float(
                getattr(config, "exposure_middle_gray", 0.18)
            )
        if self._u_adapt_min_ev is not None:
            self._u_adapt_min_ev.value = float(getattr(config, "exposure_min_ev", -8.0))
        if self._u_adapt_max_ev is not None:
            self._u_adapt_max_ev.value = float(getattr(config, "exposure_max_ev", 12.0))
        if self._u_adapt_speed_up is not None:
            self._u_adapt_speed_up.value = float(getattr(config, "exposure_adapt_speed_up", 1.5))
        if self._u_adapt_speed_down is not None:
            self._u_adapt_speed_down.value = float(getattr(config, "exposure_adapt_speed_down", 4.0))
        if self._u_adapt_dt is not None:
            self._u_adapt_dt.value = float(context.delta_time)
        if self._u_adapt_history_valid is not None:
            self._u_adapt_history_valid.value = 1 if self._history_valid else 0

        self.adapt_prog.run(group_x=1, group_y=1, group_z=1)
        self.ctx.memory_barrier()

        # ---- 4. Publish and swap ----
        self._readback_scratch[:] = self.exposure_textures[self._write_index].read()
        adapted, target, key_lum, _ = struct.unpack("<4e", bytes(self._readback_scratch))
        self._adapted_ev = float(adapted)
        self._target_ev = float(target)
        self._key_luminance = float(key_lum)

        self._read_index, self._write_index = self._write_index, self._read_index
        self._history_valid = True

        context.resources["exposure_ev"] = self._adapted_ev

    def destroy(self) -> None:
        self.histogram_buffer.release()
        for fbo in self.exposure_fbos:
            fbo.release()
        for tex in self.exposure_textures:
            tex.release()
        self.histogram_prog.release()
        self.adapt_prog.release()
