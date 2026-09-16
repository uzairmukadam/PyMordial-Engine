"""Spot Light Shadow Maps Manager for PyMordial Engine.

Manages a 2x2 depth texture atlas (up to 4 shadow-casting spot lights)
with single-frustum perspective projection for high-performance localized contact shadows.
"""

from __future__ import annotations
import math
import numpy as np
import moderngl

from engine.core.math_utils import matrix_look_at, mat4_mul


class SpotLightShadowMap:
    """Manages a 2x2 depth atlas for up to 4 dynamic shadow-casting spot lights."""

    __slots__ = (
        "ctx",
        "atlas_size",
        "depth_texture",
        "fbo",
        "spot_matrices",
        "_proj_scratch",
        "_view_scratch",
    )

    def __init__(self, ctx: moderngl.Context, atlas_size: int = 2048) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size

        self.depth_texture: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self._create_atlas()

        # Pre-allocated scratch matrices for zero-allocation per frame
        self._proj_scratch = np.zeros(16, dtype=np.float32)
        self._view_scratch = np.zeros(16, dtype=np.float32)
        self.spot_matrices = [np.identity(4, dtype=np.float32).flatten() for _ in range(4)]

    def _create_atlas(self) -> None:
        self.depth_texture = self.ctx.depth_texture((self.atlas_size, self.atlas_size))
        self.depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False
        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth_texture)

    def resize_atlas(self, new_size: int) -> None:
        """Dynamically resizes shadow atlas if resolution changed."""
        if new_size <= 0 or new_size == self.atlas_size:
            return
        self.destroy()
        self.atlas_size = new_size
        self._create_atlas()

    def clear(self) -> None:
        """Binds FBO and clears depth buffer to 1.0."""
        self.fbo.use()
        self.fbo.clear(depth=1.0)

    def begin_spot(self, spot_idx: int) -> None:
        """Sets viewport to the specific spot quadrant in the 2x2 atlas."""
        half = self.atlas_size // 2
        x = (spot_idx % 2) * half
        y = (spot_idx // 2) * half
        self.ctx.viewport = (x, y, half, half)

    def compute_spot_matrix(
        self,
        position: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        outer_cone_angle_rad: float,
        radius: float,
        spot_idx: int = 0,
    ) -> np.ndarray:
        """Calculates perspective view-projection matrix for a spot light.

        Uses right-handed perspective projection with GL_ZERO_TO_ONE depth range.
        """
        px, py, pz = float(position[0]), float(position[1]), float(position[2])
        dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

        # Normalize direction
        inv_len = 1.0 / (math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6)
        dx, dy, dz = dx * inv_len, dy * inv_len, dz * inv_len

        # Target point
        tx, ty, tz = px + dx, py + dy, pz + dz

        # Avoid gimbal lock when pointing nearly vertically
        up = (0.0, 1.0, 0.0) if abs(dy) < 0.98 else (0.0, 0.0, 1.0)
        matrix_look_at((px, py, pz), (tx, ty, tz), up=up, out=self._view_scratch)

        # Perspective projection for GL_ZERO_TO_ONE depth range
        # Field of view is 2x the outer cone angle
        fov = max(outer_cone_angle_rad * 2.0, math.radians(5.0))
        fov = min(fov, math.radians(160.0))

        tan_half = math.tan(fov * 0.5)
        f = 1.0 / max(tan_half, 1e-4)

        near = 0.40
        far = max(radius, 1.0)

        proj = self._proj_scratch
        proj.fill(0.0)
        proj[0] = f      # aspect ratio = 1.0
        proj[5] = f
        proj[10] = -far / (far - near)
        proj[11] = -1.0
        proj[14] = -(far * near) / (far - near)
        proj[15] = 0.0

        # Combine: LightViewProjection (column-major)
        out_mat = self.spot_matrices[spot_idx]
        mat4_mul(proj, self._view_scratch, out=out_mat)
        return out_mat

    def destroy(self) -> None:
        if self.depth_texture is not None:
            self.depth_texture.release()
            self.depth_texture = None
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
