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
        "max_distance",
        "depth_texture",
        "fbo",
        "split_distances",
        "light_matrices",
        "_proj_scratch",
        "_view_scratch",
        "_center_scratch",
        "_light_pos_scratch",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas_size: int = 4096,
        cascade_count: int = 4,
        max_distance: float = 500.0,
    ) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size
        self.cascade_count = cascade_count
        self.max_distance = float(max_distance)

        # Depth texture atlas and framebuffer
        self.depth_texture = None  # type: ignore[assignment]
        self.fbo = None  # type: ignore[assignment]
        self._create_atlas()

        # Compute cascade split distances (log-linear blend)
        self.update_splits(max_distance=max_distance, cascade_count=cascade_count)
        self.light_matrices = [np.zeros(16, dtype=np.float32) for _ in range(4)]

    def update_splits(
        self,
        max_distance: float,
        cascade_count: int | None = None,
        lambda_val: float = 0.85,
    ) -> None:
        """Dynamically recomputes cascade split distances for a new max shadow distance."""
        if cascade_count is not None:
            self.cascade_count = cascade_count
        self.max_distance = float(max_distance)

        near = 0.1
        far = max(self.max_distance, 1.0)
        splits = []

        count = max(1, min(4, self.cascade_count))
        for i in range(1, count + 1):
            p = i / count
            log_split = near * ((far / near) ** p)
            lin_split = near + (far - near) * p
            splits.append(lambda_val * log_split + (1.0 - lambda_val) * lin_split)

        # Pad to 4 elements
        while len(splits) < 4:
            splits.append(far)

        self.split_distances = tuple(splits[:4])

        # Pre-allocated scratch buffers
        self._proj_scratch = np.zeros(16, dtype=np.float32)
        self._view_scratch = np.zeros(16, dtype=np.float32)
        self._center_scratch = np.zeros(3, dtype=np.float32)
        self._light_pos_scratch = np.zeros(3, dtype=np.float32)

    def compute_cascade_matrices(
        self,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        sun_dir: np.ndarray,
    ) -> list[np.ndarray]:
        """Calculates 4 light-projection matrices fitted to cascade distances with texel stabilization."""
        sx, sy, sz = float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])
        inv_sn = 1.0 / (math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-6)
        snx, sny, snz = sx * inv_sn, sy * inv_sn, sz * inv_sn

        prev_dist = 0.1
        proj = self._proj_scratch
        view = self._view_scratch
        center = self._center_scratch
        light_pos = self._light_pos_scratch

        quad_size = max(self.atlas_size // 2, 1)
        up_vec_standard = (0.0, 1.0, 0.0)
        up_vec_alt = (0.0, 0.0, 1.0)

        for i in range(4):
            dist = self.split_distances[i]
            center_dist = (prev_dist + dist) * 0.5

            center[0] = camera_pos[0] + camera_forward[0] * center_dist
            center[1] = camera_pos[1] + camera_forward[1] * center_dist
            center[2] = camera_pos[2] + camera_forward[2] * center_dist

            # Enclose cascade frustum slice with optimal tight bounding sphere (2x sharper near cascades)
            half_span = (dist - prev_dist) * 0.5
            far_half_w = dist * 0.65
            raw_radius = math.sqrt(half_span * half_span + far_half_w * far_half_w) * 1.15
            # Round radius to 0.5m increments to prevent projection breathing
            radius = math.ceil(raw_radius * 2.0) * 0.5

            # Light view matrix centered on cascade with extended occluder pull-back
            caster_pullback = max(radius * 3.5, 65.0)
            far = caster_pullback + radius * 2.5
            light_pos[0] = center[0] - snx * caster_pullback
            light_pos[1] = center[1] - sny * caster_pullback
            light_pos[2] = center[2] - snz * caster_pullback

            # Prevent gimbal lock with near-vertical sun angles
            up_vec = up_vec_standard if abs(sny) < 0.98 else up_vec_alt
            matrix_look_at(light_pos, center, up=up_vec, out=view)

            # Subpixel Texel Snapping in Light View Space to eliminate edge swimming/shimmering
            world_units_per_texel = (2.0 * radius) / float(quad_size)
            if world_units_per_texel > 1e-6:
                view[12] = round(view[12] / world_units_per_texel) * world_units_per_texel
                view[13] = round(view[13] / world_units_per_texel) * world_units_per_texel

            # Standard OpenGL orthographic projection (NDC z in [-1, 1], depth buffer in [0, 1])
            proj.fill(0.0)
            ext = radius
            proj[0] = 1.0 / ext
            proj[5] = 1.0 / ext
            proj[10] = -2.0 / far
            proj[14] = -1.0
            proj[15] = 1.0

            # Combine: LightViewProjection (4x4 column major multiplication)
            mat4_mul(proj, view, out=self.light_matrices[i])

            prev_dist = dist

        return self.light_matrices

    def begin_cascade(self, cascade_idx: int) -> None:
        """Sets viewport to the specific cascade quadrant in the 2x2 atlas."""
        half = self.atlas_size // 2
        x = (cascade_idx % 2) * half
        y = (cascade_idx // 2) * half
        self.ctx.viewport = (x, y, half, half)

    def clear(self) -> None:
        self.fbo.use()
        self.fbo.clear(depth=1.0)

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

    def destroy(self) -> None:
        if self.depth_texture is not None:
            self.depth_texture.release()
        if self.fbo is not None:
            self.fbo.release()
