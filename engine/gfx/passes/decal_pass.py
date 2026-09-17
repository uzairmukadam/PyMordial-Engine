"""Screen-Space Deferred Decal System.

Projects oriented bounding box (OBB) decals onto existing G-Buffer depth and albedo/roughness.
Allows conforming tire skidmarks, oil puddles, and road markings to wrap seamlessly across
asphalt, curbs, slopes, and curved terrain without geometry clipping, hovering, or Z-fighting.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import moderngl

from engine.core.math_utils import trs_to_mat4
from engine.gfx.render_graph import RenderPass, RenderGraphContext

if TYPE_CHECKING:
    from engine.gfx.g_buffer import GBuffer


SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


def _load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


class DecalPass(RenderPass):
    """Screen-Space Deferred Decal Pass (Phase 9)."""

    MAX_DECALS: int = 128
    # Each decal: 16 floats (mat4 world_to_decal) + 4 floats (color_opacity) + 4 floats (params) = 24 floats = 96 bytes
    DECAL_SIZE_FLOATS: int = 24
    DECAL_SIZE_BYTES: int = DECAL_SIZE_FLOATS * 4

    __slots__ = (
        "ctx",
        "width",
        "height",
        "prog",
        "vao",
        "scratch_albedo_tex",
        "decal_fbo",
        "copy_dst_fbo",
        "ssbo_decals",
        "_cpu_buffer",
        "active_decal_count",
        "_u_active_count",
        "_scratch_mat",
    )

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="DecalPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.active_decal_count: int = 0

        quad_vert = _load_shader("fullscreen_quad.vert")
        decal_frag = _load_shader("decal.frag")

        self.prog = ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=decal_frag,
        )
        self.vao = ctx.vertex_array(self.prog, [])

        if "u_GBufferAlbedoRoughness" in self.prog:
            self.prog["u_GBufferAlbedoRoughness"].value = 0
        if "u_GBufferDepth" in self.prog:
            self.prog["u_GBufferDepth"].value = 1

        self._u_active_count = self.prog.get("u_ActiveDecalCount", None)

        # Pre-allocated CPU buffer and SSBO 4
        self._cpu_buffer = np.zeros(self.MAX_DECALS * self.DECAL_SIZE_FLOATS, dtype=np.float32)
        total_bytes = self.MAX_DECALS * self.DECAL_SIZE_BYTES
        self.ssbo_decals = ctx.buffer(reserve=total_bytes)

        # Scratch HDR albedo texture (RGBA16F to match upgraded G-Buffer RT0)
        self.scratch_albedo_tex = ctx.texture((self.width, self.height), 4, dtype="f2")
        self.scratch_albedo_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.decal_fbo = ctx.framebuffer(color_attachments=[self.scratch_albedo_tex])
        self.copy_dst_fbo: moderngl.Framebuffer | None = None
        self._scratch_mat = np.zeros(16, dtype=np.float32)

    def resize(self, width: int, height: int) -> None:
        """Resizes intermediate textures and framebuffers."""
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        self.scratch_albedo_tex.release()
        self.decal_fbo.release()
        if self.copy_dst_fbo is not None:
            self.copy_dst_fbo.release()
            self.copy_dst_fbo = None

        self.scratch_albedo_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.scratch_albedo_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.decal_fbo = self.ctx.framebuffer(color_attachments=[self.scratch_albedo_tex])

    def add_decal(
        self,
        position: tuple[float, float, float],
        rotation_quat: tuple[float, float, float, float] | np.ndarray,
        size: tuple[float, float, float],
        color: tuple[float, float, float] = (0.04, 0.04, 0.05),
        opacity: float = 0.85,
        roughness: float = 0.95,
        decal_type: int = 0,
    ) -> int:
        """Adds or updates a decal in the cyclic buffer (zero-allocation)."""
        idx = self.active_decal_count % self.MAX_DECALS
        if self.active_decal_count < self.MAX_DECALS:
            self.active_decal_count += 1

        # Compute Model Matrix M = TRS (Column-major flat array)
        m16 = trs_to_mat4(position, rotation_quat, size)
        # Compute World-To-Decal Matrix M^-1 in Column-major (order='F')
        m44 = m16.reshape((4, 4), order="F")
        try:
            inv_m44 = np.linalg.inv(m44)
        except Exception:
            inv_m44 = np.eye(4, dtype=np.float32)

        offset = idx * self.DECAL_SIZE_FLOATS
        # 16 floats: inv_matrix in column-major order for std430 SSBO
        self._cpu_buffer[offset : offset + 16] = inv_m44.flatten(order="F")
        # 4 floats: color_opacity
        self._cpu_buffer[offset + 16] = color[0]
        self._cpu_buffer[offset + 17] = color[1]
        self._cpu_buffer[offset + 18] = color[2]
        self._cpu_buffer[offset + 19] = opacity
        # 4 floats: params (roughness, decal_type, reserved, reserved)
        self._cpu_buffer[offset + 20] = roughness
        self._cpu_buffer[offset + 21] = float(decal_type)
        self._cpu_buffer[offset + 22] = 0.0
        self._cpu_buffer[offset + 23] = 0.0

        return idx

    def clear(self) -> None:
        """Clears all active decals."""
        self.active_decal_count = 0

    def execute(self, context: RenderGraphContext, g_buffer: GBuffer) -> None:
        """Executes decal projection pass onto G-Buffer albedo & roughness."""
        if not self.enabled or self.active_decal_count <= 0:
            return

        count = min(self.active_decal_count, self.MAX_DECALS)
        bytes_to_write = count * self.DECAL_SIZE_BYTES
        self.ssbo_decals.write(self._cpu_buffer[: count * self.DECAL_SIZE_FLOATS].tobytes(), offset=0)
        self.ssbo_decals.bind_to_storage_buffer(binding=4)

        if self.copy_dst_fbo is None:
            self.copy_dst_fbo = self.ctx.framebuffer(
                color_attachments=[g_buffer.rt_albedo_roughness],
            )

        self.decal_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Bind inputs
        g_buffer.rt_albedo_roughness.use(location=0)
        g_buffer.depth_texture.use(location=1)

        if self._u_active_count is not None:
            self._u_active_count.value = count

        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # Copy modified albedo/roughness back into G-Buffer RT0
        self.ctx.copy_framebuffer(self.copy_dst_fbo, self.decal_fbo)

    def destroy(self) -> None:
        self.scratch_albedo_tex.release()
        self.decal_fbo.release()
        if self.copy_dst_fbo is not None:
            self.copy_dst_fbo.release()
        self.ssbo_decals.release()
        self.prog.release()
        self.vao.release()
