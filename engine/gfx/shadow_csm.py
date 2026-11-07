"""Stable, scene-fitted cascaded shadow maps."""

from __future__ import annotations
import math
import numpy as np
import moderngl

class CascadedShadowMap:
    """Manages a cascaded shadow atlas with frustum fitting and texel snapping."""

    __slots__ = (
        "ctx",
        "atlas_size",
        "cascade_count",
        "max_distance",
        "split_lambda",
        "depth_texture",
        "fbo",
        "split_distances",
        "light_matrices",
        "stabilize_cascades",
        "cullers",
        "cascade_visible_masks",
        "cascade_draw_batches",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas_size: int = 4096,
        cascade_count: int = 4,
        max_distance: float = 500.0,
        split_lambda: float = 0.85,
    ) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size
        self.cascade_count = cascade_count
        self.max_distance = float(max_distance)
        self.split_lambda = float(split_lambda)

        # Depth texture atlas and framebuffer
        self.depth_texture = None  # type: ignore[assignment]
        self.fbo = None  # type: ignore[assignment]
        self._create_atlas()

        # Compute cascade split distances (log-linear blend)
        self.update_splits(max_distance=max_distance, cascade_count=cascade_count, lambda_val=split_lambda)
        self.light_matrices = [np.zeros(16, dtype=np.float32) for _ in range(4)]
        self.stabilize_cascades: bool = True

        # Phase 1: Per-Cascade Frustum Culling
        from engine.gfx.culling import FrustumCuller
        self.cullers = [FrustumCuller(max_entities=100_000) for _ in range(4)]
        self.cascade_visible_masks: list[np.ndarray | None] = [None] * 4
        self.cascade_draw_batches: list[list[tuple]] = [[] for _ in range(4)]

    def update_splits(
        self,
        max_distance: float,
        cascade_count: int | None = None,
        lambda_val: float = 0.85,
    ) -> None:
        """Computes practical logarithmic/uniform splits for the active cascades."""
        if cascade_count is not None:
            self.cascade_count = max(1, min(4, int(cascade_count)))
        self.max_distance = max(float(max_distance), 1.1)

        near = 0.1
        far = max(self.max_distance, near + 1.0)
        count = max(1, min(4, self.cascade_count))
        blend = min(1.0, max(0.0, float(lambda_val)))
        self.split_lambda = blend
        if count == 1:
            splits = [far]
        else:
            # Keep nearby receivers in a tight high-resolution cascade, then
            # distribute the remaining range using practical log/uniform splits.
            near_split = min(10.0, far * 0.18)
            splits = [near_split]
            remaining_cascades = count - 1
            for index in range(1, count):
                fraction = index / remaining_cascades
                logarithmic = near_split * (far / near_split) ** fraction
                uniform = near_split + (far - near_split) * fraction
                splits.append(blend * logarithmic + (1.0 - blend) * uniform)
        splits.extend([far] * (4 - len(splits)))
        self.split_distances = tuple(splits[:4])

    def compute_cascade_matrices(
        self,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        sun_dir: np.ndarray,
        stabilize: bool | None = None,
        fovy_deg: float = 60.0,
        aspect: float = 16.0 / 9.0,
        aabbs: np.ndarray | None = None,
        active_count: int = 0,
        blend_fraction: float = 0.20,
    ) -> list[np.ndarray]:
        """Fits stable orthographic maps to each camera slice and overlapping casters."""
        forward = np.asarray(camera_forward, dtype=np.float64).copy()
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        sun = np.asarray(sun_dir, dtype=np.float64).copy()
        sun /= max(float(np.linalg.norm(sun)), 1e-8)
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(sun, world_up))) > 0.98:
            world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        light_right = np.cross(sun, world_up)
        light_right /= max(float(np.linalg.norm(light_right)), 1e-8)
        light_up = np.cross(light_right, sun)
        light_up /= max(float(np.linalg.norm(light_up)), 1e-8)

        camera_right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float64))
        if float(np.linalg.norm(camera_right)) < 1e-8:
            camera_right = np.cross(forward, np.array((0.0, 0.0, 1.0), dtype=np.float64))
        camera_right /= max(float(np.linalg.norm(camera_right)), 1e-8)
        camera_up = np.cross(camera_right, forward)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1e-8)

        camera_pos64 = np.asarray(camera_pos, dtype=np.float64)
        quad_size = max(self.atlas_size // 2, 1)
        use_stabilize = self.stabilize_cascades if stabilize is None else bool(stabilize)
        count = max(1, min(4, self.cascade_count))
        blend_fraction = min(0.5, max(0.0, float(blend_fraction)))
        near_clip = 0.1
        tan_half_fov_y = math.tan(math.radians(max(1.0, min(179.0, fovy_deg))) * 0.5)
        safe_aspect = max(float(aspect), 1e-4)
        split_near = near_clip
        caster_centers: np.ndarray | None = None
        caster_extents: np.ndarray | None = None
        if aabbs is not None and active_count > 0:
            caster_count = min(int(active_count), len(aabbs))
            caster_centers = np.asarray(aabbs[:caster_count, :3], dtype=np.float64)
            caster_extents = np.asarray(aabbs[:caster_count, 3:6], dtype=np.float64)

        for cascade in range(count):
            split_far = self.split_distances[cascade]
            fit_near = max(near_clip, split_near - split_near * blend_fraction) if cascade > 0 else near_clip
            fit_far = min(
                self.max_distance,
                split_far + split_far * blend_fraction,
            ) if cascade < count - 1 else split_far
            receiver_depth_min = math.inf
            receiver_depth_max = -math.inf
            for distance in (fit_near, fit_far):
                half_height = distance * tan_half_fov_y
                half_width = half_height * safe_aspect
                slice_center = camera_pos64 + forward * distance
                for x_sign in (-1.0, 1.0):
                    for y_sign in (-1.0, 1.0):
                        corner = (
                            slice_center
                            + camera_right * (x_sign * half_width)
                            + camera_up * (y_sign * half_height)
                        )
                        light_z = float(np.dot(sun, corner))
                        receiver_depth_min = min(receiver_depth_min, light_z)
                        receiver_depth_max = max(receiver_depth_max, light_z)

            center_distance = (fit_near + fit_far) * 0.5
            radius = math.sqrt(
                ((fit_far - fit_near) * 0.5) ** 2
                + (fit_far * tan_half_fov_y) ** 2 * (1.0 + safe_aspect * safe_aspect)
            )
            radius = max(math.ceil(radius * 1.025 * 16.0) / 16.0, 0.5)
            slice_center = camera_pos64 + forward * center_distance
            center_x = float(np.dot(light_right, slice_center))
            center_y = float(np.dot(light_up, slice_center))
            texel_world = (2.0 * radius) / float(quad_size)
            if use_stabilize:
                center_x = math.floor(center_x / texel_world + 0.5) * texel_world
                center_y = math.floor(center_y / texel_world + 0.5) * texel_world

            depth_pad = max(10.0, (receiver_depth_max - receiver_depth_min) * 0.05)
            depth_min = receiver_depth_min - depth_pad
            depth_max = receiver_depth_max + depth_pad

            if caster_centers is not None and caster_extents is not None:
                cx = caster_centers @ light_right
                cy = caster_centers @ light_up
                cz = caster_centers @ sun
                ex = caster_extents @ np.abs(light_right)
                ey = caster_extents @ np.abs(light_up)
                ez = caster_extents @ np.abs(sun)
                overlaps_xy = (
                    (cx + ex >= center_x - radius)
                    & (cx - ex <= center_x + radius)
                    & (cy + ey >= center_y - radius)
                    & (cy - ey <= center_y + radius)
                    & (cz - ez <= receiver_depth_max + depth_pad)
                    & (cz + ez >= receiver_depth_min - self.max_distance)
                )
                if np.any(overlaps_xy):
                    depth_min = min(depth_min, float(np.min((cz - ez)[overlaps_xy])) - depth_pad)
                    depth_max = max(depth_max, float(np.max((cz + ez)[overlaps_xy])))

            depth_range = max(depth_max - depth_min, 1.0)
            matrix = self.light_matrices[cascade].reshape((4, 4), order="F")
            matrix.fill(0.0)
            matrix[0, :3] = light_right / radius
            matrix[0, 3] = -center_x / radius
            matrix[1, :3] = light_up / radius
            matrix[1, 3] = -center_y / radius
            matrix[2, :3] = sun / depth_range
            matrix[2, 3] = -depth_min / depth_range
            matrix[3, 3] = 1.0

            split_near = split_far

        for cascade in range(count, 4):
            self.light_matrices[cascade][:] = self.light_matrices[count - 1]

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

    def cull_cascades(
        self,
        aabbs: np.ndarray,
        active_count: int,
        draw_batches: list[tuple],
        cascade_count: int | None = None,
    ) -> list[list[tuple]]:
        """Extracts planes and filters shadow draw batches for each active cascade frustum.

        Args:
            aabbs: (N, 6) float32 array of world-space AABBs [cx, cy, cz, ex, ey, ez].
            active_count: Number of active entity AABBs to evaluate.
            draw_batches: Base unfiltered shadow caster batches [(mesh, count, base_inst, is_tess, cast_shadow), ...].
            cascade_count: Number of active cascades (defaults to self.cascade_count).

        Returns:
            List of 4 filtered batch lists, one per cascade.
        """
        count = self.cascade_count if cascade_count is None else min(4, cascade_count)
        for i in range(count):
            culler = self.cullers[i]
            # Standard orthographic projection under GL_ZERO_TO_ONE (reverse_z=False for shadow maps)
            culler.extract_planes(self.light_matrices[i], reverse_z=False)
            vis_mask = culler.cull_aabbs(aabbs, active_count)
            self.cascade_visible_masks[i] = vis_mask
            filtered = culler.filter_draw_batches(draw_batches, vis_mask)
            self.cascade_draw_batches[i] = filtered

        return self.cascade_draw_batches

    def destroy(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.depth_texture is not None:
            self.depth_texture.release()
            self.depth_texture = None
