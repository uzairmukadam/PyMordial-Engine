"""Light Propagation Volumes (LPV) Pass for PyMordial Engine.

Maintains a 3D volumetric radiance grid (32x32x32 RGBA16F) and executes GPU compute
injection and neighbor propagation to provide off-screen, camera-independent indirect bounce.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class LPVPass(RenderPass):
    """3D volumetric indirect diffuse bounce using Light Propagation Volumes."""

    def __init__(self, ctx: moderngl.Context, grid_res: int = 32) -> None:
        super().__init__(name="LPVPass", enabled=True)
        self.ctx = ctx
        self.grid_res = grid_res

        # 1. Ping-pong 3D textures (32x32x32 RGBA16F)
        self.vol_a = self.ctx.texture3d((grid_res, grid_res, grid_res), 4, dtype="f2")
        self.vol_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.vol_a.repeat_x = False
        self.vol_a.repeat_y = False
        self.vol_a.repeat_z = False

        self.vol_b = self.ctx.texture3d((grid_res, grid_res, grid_res), 4, dtype="f2")
        self.vol_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.vol_b.repeat_x = False
        self.vol_b.repeat_y = False
        self.vol_b.repeat_z = False

        self._current_src = self.vol_a
        self._current_dst = self.vol_b

        # 1x1x1 black fallback
        self.black_fallback = self.ctx.texture3d((1, 1, 1), 4, data=b"\x00" * 8, dtype="f2")

        # 2. Grid bounds in world space
        self.volume_min = np.array([-32.0, -2.0, -32.0], dtype=np.float32)
        self.volume_size = np.array([64.0, 32.0, 64.0], dtype=np.float32)

        # 3. Compile LPV compute shader
        comp_src = (SHADER_DIR / "lpv_propagate.comp").read_text(encoding="utf-8")
        self.compute_prog = self.ctx.compute_shader(comp_src)

        self._u_min = self.compute_prog.get("u_LPV_Min", None)
        self._u_size = self.compute_prog.get("u_LPV_Size", None)
        self._u_sun_dir = self.compute_prog.get("u_SunDirection_Intensity", None)
        self._u_sun_color = self.compute_prog.get("u_SunColor_Ambient", None)
        self._u_intensity = self.compute_prog.get("u_LPV_Intensity", None)
        self._u_point_light_count = self.compute_prog.get("u_PointLightCount", None)

    def execute(self, context: RenderGraphContext) -> None:
        gi_mode = getattr(context.config, "gi_mode", "HYBRID")
        if gi_mode not in ("LPV", "HYBRID"):
            context.resources["lpv_volume"] = self.black_fallback
            context.resources["lpv_min"] = self.volume_min
            context.resources["lpv_size"] = self.volume_size
            return

        # Center volume grid on camera position with smooth snapping to grid step
        cell_size = self.volume_size[0] / self.grid_res
        cam_x, cam_y, cam_z = context.camera_pos
        snap_x = np.floor(cam_x / cell_size) * cell_size
        snap_z = np.floor(cam_z / cell_size) * cell_size

        self.volume_min[0] = snap_x - self.volume_size[0] * 0.5
        self.volume_min[1] = -2.0  # Ground aligned
        self.volume_min[2] = snap_z - self.volume_size[2] * 0.5

        # Bind image units: src is readonly, dst is writeonly
        self._current_src.bind_to_image(0, read=True, write=False)
        self._current_dst.bind_to_image(1, read=False, write=True)

        if self._u_min is not None:
            self._u_min.value = tuple(self.volume_min)
        if self._u_size is not None:
            self._u_size.value = tuple(self.volume_size)
        if self._u_sun_dir is not None:
            self._u_sun_dir.value = (context.sun_dir[0], context.sun_dir[1], context.sun_dir[2], context.sun_lux)
        if self._u_intensity is not None:
            self._u_intensity.value = getattr(context.config, "lpv_intensity", 1.0)
        if self._u_point_light_count is not None:
            self._u_point_light_count.value = context.resources.get("point_light_count", 0)

        # Dispatch 32x32x32 compute grid (8x8x8 groups of 4x4x4 threads)
        groups = self.grid_res // 4
        self.compute_prog.run(groups, groups, groups)

        # Ping-pong swap
        self._current_src, self._current_dst = self._current_dst, self._current_src

        # Publish propagated 3D volume
        context.resources["lpv_volume"] = self._current_src
        context.resources["lpv_min"] = self.volume_min
        context.resources["lpv_size"] = self.volume_size

    def destroy(self) -> None:
        self.vol_a.release()
        self.vol_b.release()
        self.black_fallback.release()
        self.compute_prog.release()
