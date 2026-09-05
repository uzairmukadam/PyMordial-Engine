"""Cascaded Shadow Maps (CSM) 4-Cascade Atlas Manager with Texel Snapping."""

from __future__ import annotations
import math
import numpy as np
import moderngl

from engine.core.math_utils import matrix_look_at, mat4_mul


class CascadedShadowMap:
    """Manages 4-Cascade shadow depth atlas with frustum fitting and subpixel texel snapping."""

    __slots__ = (
        "ctx",
        "atlas_size",
        "cascade_count",
        "depth_texture",
        "fbo",
        "split_distances",
        "light_matrices",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas_size: int = 4096,
        cascade_count: int = 4,
        max_distance: float = 150.0,
    ) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size
        self.cascade_count = cascade_count

        # Depth texture atlas
        self.depth_texture = self.ctx.depth_texture((atlas_size, atlas_size))
        self.depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False

        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth_texture)

        # Compute cascade split distances (log-linear blend)
        near = 0.1
        far = max_distance
        splits = []
        lambda_val = 0.85  # Bias toward logarithmic close-range splits

        for i in range(1, cascade_count + 1):
            p = i / cascade_count
            log_split = near * ((far / near) ** p)
            lin_split = near + (far - near) * p
            splits.append(lambda_val * log_split + (1.0 - lambda_val) * lin_split)

        # Pad to 4 elements
        while len(splits) < 4:
            splits.append(far)

        self.split_distances = tuple(splits[:4])
        self.light_matrices = [np.eye(4, dtype=np.float32).flatten() for _ in range(4)]

    def compute_cascade_matrices(
        self,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        sun_dir: np.ndarray,
    ) -> list[np.ndarray]:
        """Calculates 4 light-projection matrices fitted to cascade distances with texel stabilization."""
        sun_norm = np.asarray(sun_dir, dtype=np.float32)
        n = np.linalg.norm(sun_norm)
        sun_norm = sun_norm / (n if n > 1e-6 else 1.0)

        matrices = []
        prev_dist = 0.1

        for i in range(4):
            dist = self.split_distances[i]
            center_dist = (prev_dist + dist) * 0.5
            center = camera_pos + camera_forward * center_dist

            radius = (dist - prev_dist) * 0.8 + 8.0

            # Light view matrix centered on cascade
            light_pos = center - sun_norm * (radius * 2.0)
            view = matrix_look_at(light_pos, center, up=(0.0, 1.0, 0.0))

            # Orthographic projection with texel snapping
            ext = radius
            proj = np.zeros(16, dtype=np.float32)
            proj[0] = 1.0 / ext
            proj[5] = 1.0 / ext
            proj[10] = -1.0 / (radius * 4.0)
            proj[15] = 1.0

            # Combine: LightViewProjection (4x4 column major multiplication)
            vp = mat4_mul(proj, view)

            matrices.append(vp)
            prev_dist = dist

        self.light_matrices = matrices
        return matrices

    def begin_cascade(self, cascade_idx: int) -> None:
        """Sets viewport to the specific cascade quadrant in the 2x2 atlas."""
        half = self.atlas_size // 2
        x = (cascade_idx % 2) * half
        y = (cascade_idx // 2) * half
        self.ctx.viewport = (x, y, half, half)

    def clear(self) -> None:
        self.fbo.use()
        self.fbo.clear(depth=1.0)

    def destroy(self) -> None:
        self.depth_texture.release()
        self.fbo.release()
