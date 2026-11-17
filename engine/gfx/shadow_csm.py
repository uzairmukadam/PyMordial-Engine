"""Stable, scene-fitted cascaded shadow maps."""

from __future__ import annotations

import math

import moderngl
import numpy as np

SPLIT_MODES = ("PSSM", "GEOMETRIC", "UNIFORM")

# Hard ceiling on the world-space span ratio between neighbouring cascades.
# Texel density is inversely proportional to slice span for a fixed map size, so
# a larger ratio shows up as a visible quality step at the cascade boundary.
MAX_SPAN_RATIO = 2.5


def _pssm_splits(near: float, far: float, count: int, blend: float) -> list[float]:
    """Standard practical split scheme: log/uniform blend anchored at ``near``."""
    log_ratio = far / near
    splits: list[float] = []
    for index in range(1, count + 1):
        fraction = index / count
        logarithmic = near * log_ratio**fraction
        uniform = near + (far - near) * fraction
        splits.append(blend * logarithmic + (1.0 - blend) * uniform)
    return splits


def _max_span_ratio(splits: list[float], near: float) -> float:
    """Worst adjacent world-space span ratio for a split list."""
    previous = near
    worst = 1.0
    previous_span = 0.0
    for split in splits:
        span = split - previous
        if previous_span > 0.0:
            ratio = span / previous_span
            if ratio > worst:
                worst = ratio
        previous_span = span
        previous = split
    return worst


class CascadedShadowMap:
    """Manages a cascaded shadow atlas with frustum fitting and texel snapping."""

    __slots__ = (
        "ctx",
        "atlas_size",
        "cascade_count",
        "max_distance",
        "split_lambda",
        "split_mode",
        "cascade_distance_scale",
        "depth_texture",
        "fbo",
        "split_distances",
        "light_matrices",
        "stabilize_cascades",
        "cullers",
        "cascade_visible_masks",
        "cascade_draw_batches",
        "nonuniform_atlas",
        "cascade_rects",
        "cascade_rect_uv",
        "cascade_rect_uv_array",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas_size: int = 4096,
        cascade_count: int = 4,
        max_distance: float = 500.0,
        split_lambda: float = 0.85,
        split_mode: str = "PSSM",
        nonuniform_atlas: bool = False,
    ) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size
        self.cascade_count = cascade_count
        self.max_distance = float(max_distance)
        self.split_lambda = float(split_lambda)
        self.split_mode = str(split_mode).upper()
        self.nonuniform_atlas = bool(nonuniform_atlas)

        # Depth texture atlas and framebuffer
        self.depth_texture = None  # type: ignore[assignment]
        self.fbo = None  # type: ignore[assignment]
        self._create_atlas()

        # Per-cascade atlas rects (x, y, w, h) in texels, plus the matching
        # (offset_x, offset_y, scale_x, scale_y) UV rects uploaded to the shader.
        self.cascade_rects: list[tuple[int, int, int, int]] = []
        self.cascade_rect_uv: list[tuple[float, float, float, float]] = []
        # Flat (4, 4) float32 mirror of ``cascade_rect_uv`` so the per-frame
        # uniform upload is a zero-allocation ``tobytes()`` on a stable buffer.
        self.cascade_rect_uv_array = np.zeros((4, 4), dtype=np.float32)
        self._update_cascade_rects()

        # Compute cascade split distances
        self.update_splits(
            max_distance=max_distance,
            cascade_count=cascade_count,
            lambda_val=split_lambda,
            mode=self.split_mode,
        )
        self.light_matrices = [np.zeros(16, dtype=np.float32) for _ in range(4)]
        self.stabilize_cascades: bool = True

        # Phase 1: Per-Cascade Frustum Culling
        from engine.gfx.culling import FrustumCuller

        self.cullers = [FrustumCuller(max_entities=100_000) for _ in range(4)]
        self.cascade_visible_masks: list[np.ndarray | None] = [None] * 4
        self.cascade_draw_batches: list[list[tuple]] = [[] for _ in range(4)]

    def _update_cascade_rects(self) -> None:
        """Rebuilds the per-cascade atlas rects for the current atlas size.

        With ``nonuniform_atlas`` disabled the rects reproduce the legacy 2x2
        quadrant layout exactly, so existing behaviour and goldens are unchanged.
        """
        size = int(self.atlas_size)
        rects: list[tuple[int, int, int, int]] = []
        if self.nonuniform_atlas:
            # Near cascades cover less world space per texel, so they get the
            # larger share of the atlas: 2048/2048/1024/1024 in a 4096 atlas.
            half = size // 2
            quarter = size // 4
            rects = [
                (0, 0, half, half),
                (half, 0, half, half),
                (0, half, quarter, quarter),
                (quarter, half, quarter, quarter),
            ]
        else:
            half = size // 2
            rects = [
                (0, 0, half, half),
                (half, 0, half, half),
                (0, half, half, half),
                (half, half, half, half),
            ]
        self.cascade_rects = rects
        inv = 1.0 / float(max(size, 1))
        self.cascade_rect_uv = [(x * inv, y * inv, w * inv, h * inv) for (x, y, w, h) in rects]
        # Refresh the pre-allocated upload buffer in place (no per-frame allocation).
        self.cascade_rect_uv_array[:] = np.asarray(self.cascade_rect_uv, dtype=np.float32)

    def set_nonuniform_atlas(self, enabled: bool) -> None:
        """Toggles per-cascade atlas allocation and rebuilds the rect table."""
        enabled = bool(enabled)
        if enabled == self.nonuniform_atlas:
            return
        self.nonuniform_atlas = enabled
        self._update_cascade_rects()

    def cascade_texel_size(self, cascade_idx: int) -> int:
        """Square texel size of one cascade's atlas rect."""
        return int(self.cascade_rects[max(0, min(3, int(cascade_idx)))][2])

    def update_splits(
        self,
        max_distance: float,
        cascade_count: int | None = None,
        lambda_val: float = 0.85,
        mode: str = "PSSM",
        cascade_distance_scale: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Computes cascade splits with a bounded adjacent span ratio.

        ``mode`` selects the distribution scheme:

        * ``"PSSM"`` - practical split scheme (log/uniform blend). The blend is
          automatically reduced towards uniform spacing when the requested
          ``lambda_val`` would push the adjacent span ratio above
          :data:`MAX_SPAN_RATIO`.
        * ``"GEOMETRIC"`` - pure geometric spacing (constant span ratio).
        * ``"UNIFORM"`` - equal world-space spans.

        ``cascade_distance_scale`` optionally scales each cascade's far distance
        independently, so a project can push the near cascades out (or pull the
        far ones in) without changing the overall shadow distance.
        """
        if cascade_count is not None:
            self.cascade_count = max(1, min(4, int(cascade_count)))
        self.max_distance = max(float(max_distance), 1.1)

        near = 0.1
        far = max(self.max_distance, near + 1.0)
        count = max(1, min(4, self.cascade_count))
        blend = min(1.0, max(0.0, float(lambda_val)))
        self.split_lambda = blend
        split_mode = str(mode).upper()
        if split_mode not in SPLIT_MODES:
            split_mode = "PSSM"
        self.split_mode = split_mode
        self.cascade_distance_scale = (
            tuple(float(v) for v in cascade_distance_scale) if cascade_distance_scale is not None else None
        )

        if count == 1:
            splits = [far]
        elif split_mode == "UNIFORM":
            span = (far - near) / count
            splits = [near + span * (index + 1) for index in range(count)]
        elif split_mode == "GEOMETRIC":
            ratio = far / near
            splits = [near * ratio ** ((index + 1) / count) for index in range(count)]
        else:
            splits = _pssm_splits(near, far, count, blend)
            if _max_span_ratio(splits, near) > MAX_SPAN_RATIO:
                # Bisect for the largest blend that still respects the bound so
                # texel density never drops in a visible step.
                low, high = 0.0, blend
                for _ in range(24):
                    mid = 0.5 * (low + high)
                    if _max_span_ratio(_pssm_splits(near, far, count, mid), near) > MAX_SPAN_RATIO:
                        high = mid
                    else:
                        low = mid
                splits = _pssm_splits(near, far, count, low)

        # Optional per-cascade far-distance scaling. Applied after the base
        # distribution so the shape of the split curve is preserved while each
        # cascade's reach can be tuned independently.
        if cascade_distance_scale is not None:
            scaled: list[float] = []
            for index, split in enumerate(splits):
                scale = float(cascade_distance_scale[min(index, 3)])
                scaled.append(max(near + 1e-3, split * max(scale, 1e-3)))
            # Keep the sequence strictly increasing so cascade selection stays valid.
            for index in range(1, len(scaled)):
                if scaled[index] <= scaled[index - 1]:
                    scaled[index] = scaled[index - 1] + 1e-3
            splits = scaled

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
            fit_far = (
                min(
                    self.max_distance,
                    split_far + split_far * blend_fraction,
                )
                if cascade < count - 1
                else split_far
            )
            receiver_depth_min = math.inf
            receiver_depth_max = -math.inf
            for distance in (fit_near, fit_far):
                half_height = distance * tan_half_fov_y
                half_width = half_height * safe_aspect
                slice_center = camera_pos64 + forward * distance
                for x_sign in (-1.0, 1.0):
                    for y_sign in (-1.0, 1.0):
                        corner = (
                            slice_center + camera_right * (x_sign * half_width) + camera_up * (y_sign * half_height)
                        )
                        light_z = float(np.dot(sun, corner))
                        receiver_depth_min = min(receiver_depth_min, light_z)
                        receiver_depth_max = max(receiver_depth_max, light_z)

            center_distance = (fit_near + fit_far) * 0.5
            radius = math.sqrt(
                ((fit_far - fit_near) * 0.5) ** 2 + (fit_far * tan_half_fov_y) ** 2 * (1.0 + safe_aspect * safe_aspect)
            )
            radius = max(math.ceil(radius * 1.025 * 16.0) / 16.0, 0.5)
            slice_center = camera_pos64 + forward * center_distance
            center_x = float(np.dot(light_right, slice_center))
            center_y = float(np.dot(light_up, slice_center))
            # Snap against this cascade's own rect size so non-uniform atlases
            # keep exact texel alignment per cascade.
            quad_size = max(self.cascade_rects[cascade][2], 1)
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
        """Sets the viewport to this cascade's atlas rect."""
        x, y, w, h = self.cascade_rects[max(0, min(3, int(cascade_idx)))]
        self.ctx.viewport = (x, y, w, h)

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
        self._update_cascade_rects()

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
