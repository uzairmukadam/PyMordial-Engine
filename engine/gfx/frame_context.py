"""Unified Frame Context UBO 0 Manager for PyMordial Engine.

Maintains a single std140 uniform buffer bound to binding 0 across all shaders.
"""

from __future__ import annotations
import numpy as np
import moderngl


class FrameContext:
    """Manages allocation and single-write per frame synchronization of UBO 0."""

    # std140 buffer size: 688 bytes (172 float32s)
    BUFFER_FLOATS = 172
    BUFFER_BYTES = BUFFER_FLOATS * 4

    __slots__ = ("ctx", "ubo", "buffer")

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        # Pre-allocate contiguous memory buffer
        self.buffer = np.zeros(self.BUFFER_FLOATS, dtype=np.float32)
        # Create OpenGL Uniform Buffer Object
        self.ubo = self.ctx.buffer(reserve=self.BUFFER_BYTES)
        self.ubo.bind_to_uniform_block(binding=0)

    def update(
        self,
        view_mat: np.ndarray,
        proj_mat: np.ndarray,
        view_proj_mat: np.ndarray,
        inv_proj_mat: np.ndarray,
        inv_view_mat: np.ndarray,
        camera_pos: tuple[float, float, float] | np.ndarray,
        time_elapsed: float,
        screen_size: tuple[float, float],
        sun_dir: tuple[float, float, float] = (0.35, -0.85, 0.40),
        sun_lux: float = 4.0,
        sun_color: tuple[float, float, float] = (1.0, 0.96, 0.88),
        ambient_factor: float = 0.04,
        csm_matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        cascade_splits: tuple[float, float, float, float] = (10.0, 30.0, 80.0, 150.0),
        fog_color: tuple[float, float, float] = (0.55, 0.65, 0.78),
        fog_density: float = 0.015,
        fog_height_falloff: float = 0.10,
    ) -> None:
        """Packs all frame parameters and uploads to UBO 0 with a single write."""
        buf = self.buffer

        # 0..79: 5 matrices (16 floats each)
        buf[0:16] = view_mat
        buf[16:32] = proj_mat
        buf[32:48] = view_proj_mat
        buf[48:64] = inv_proj_mat
        buf[64:80] = inv_view_mat

        # 80..83: Camera World Pos & Time
        buf[80:83] = camera_pos[:3]
        buf[83] = time_elapsed

        # 84..87: Screen Size (w, h) & Subpixel Jitter
        buf[84] = float(screen_size[0])
        buf[85] = float(screen_size[1])
        buf[86] = 0.0  # Jitter X
        buf[87] = 0.0  # Jitter Y

        # 88..91: Sun Direction & Lux
        sun_dir_norm = np.asarray(sun_dir, dtype=np.float32)
        n = np.linalg.norm(sun_dir_norm)
        sun_dir_norm = sun_dir_norm / (n if n > 1e-6 else 1.0)
        buf[88:91] = sun_dir_norm
        buf[91] = sun_lux

        # 92..95: Sun Color & Ambient Factor
        buf[92:95] = sun_color[:3]
        buf[95] = ambient_factor

        # 96..159: 4 CSM Light Matrices (64 floats)
        if csm_matrices is not None:
            for i in range(4):
                base = 96 + i * 16
                buf[base : base + 16] = csm_matrices[i]

        # 160..163: Cascade Splits
        buf[160:164] = cascade_splits[:4]

        # 164..167: Volumetric Fog Color & Density
        buf[164:167] = fog_color[:3]
        buf[167] = fog_density

        # 168..171: Fog Params (height falloff, max distance)
        buf[168] = fog_height_falloff
        buf[169] = 200.0  # Max distance
        buf[170] = 0.0
        buf[171] = 0.0

        # Upload to GPU in a single subdata call
        self.ubo.write(buf)

    def destroy(self) -> None:
        self.ubo.release()
