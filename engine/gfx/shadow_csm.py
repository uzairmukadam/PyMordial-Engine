"""Cascaded shadow maps for hard directional shadows.

The system is deliberately small: one orthographic light matrix per cascade,
fitted to a camera frustum slice, rendered into a 2x2 atlas, and sampled with a
single hardware depth comparison. Everything that made the previous
implementation hard to reason about (filter modes, penumbra estimation,
non-uniform atlas allocation, split-mode selection) is gone.

What remains is the part that actually determines whether a hard shadow looks
correct:

* **PSSM splits.** A log/uniform blend keeps near cascades tight, which is what
  gives contact shadows their texel density. ``shadow_distance`` is the single
  knob that moves the whole distribution.
* **Rotation-invariant fitting.** Each cascade is fitted to the bounding sphere
  of its frustum slice rather than to the eight projected corners. The sphere is
  invariant under camera yaw, so the depth range does not re-scale when the
  camera turns.
* **Texel snapping.** The light-space centre is snapped to whole texels, so the
  shadow map does not shimmer as the camera moves.
* **Bounded depth slab.** A caster may only stretch the slab by a few slice
  radii. Without the bound a 10km ground quad drags the slab to kilometres and
  collapses depth precision.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

# Safety bound on the world-space span ratio between neighbouring cascades.
#
# This is a guard against pathological configurations (a huge shadow distance
# with a near-logarithmic lambda), NOT a quality target. It must sit above the
# natural PSSM ratio, otherwise the bisection below silently overrides the
# user's lambda and flattens the cascades towards uniform spacing. The natural
# ratio is ~4.7 at 120m and ~7.3 at 1200m.
MAX_SPAN_RATIO = 8.0

# How far past the slice's bounding sphere a caster may extend the depth slab,
# as a multiple of the slice radius. Large ground planes overlap every cascade
# in XY, so without a bound they stretch the slab to kilometres and destroy
# depth precision. A caster beyond this reach cannot cast into the slice.
MAX_CASTER_REACH = 2.0

# Camera near plane used when fitting cascades. Matches the projection near
# plane so cascade 0 starts exactly where the camera does.
CASCADE_NEAR = 0.1


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
        "max_caster_reach",
        "split_lambda",
        "cascade_distances",
        "depth_texture",
        "compare_sampler",
        "fbo",
        "split_distances",
        "light_matrices",
        "cullers",
        "cascade_visible_masks",
        "cascade_draw_batches",
        "cascade_rects",
        "cascade_rect_uv",
        "cascade_rect_uv_array",
        "cascade_resolutions",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas_size: int = 4096,
        cascade_count: int = 4,
        max_distance: float = 500.0,
        split_lambda: float = 0.85,
        cascade_resolutions: tuple[int, ...] | None = None,
    ) -> None:
        self.ctx = ctx
        self.atlas_size = atlas_size
        self.cascade_count = max(1, min(4, int(cascade_count)))
        self.max_distance = float(max_distance)
        self.max_caster_reach = MAX_CASTER_REACH
        self.split_lambda = float(split_lambda)
        self.cascade_distances: tuple[float, float, float, float] | None = None
        # Per-cascade rect sizes. ``None`` keeps the legacy equal 2x2 quadrants.
        self.cascade_resolutions: tuple[int, ...] | None = (
            tuple(int(v) for v in cascade_resolutions) if cascade_resolutions else None
        )

        # Depth texture atlas and framebuffer
        self.depth_texture = None  # type: ignore[assignment]
        self.compare_sampler = None  # type: ignore[assignment]
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

        self.update_splits(
            max_distance=max_distance,
            cascade_count=self.cascade_count,
            lambda_val=split_lambda,
        )
        self.light_matrices = [np.zeros(16, dtype=np.float32) for _ in range(4)]

        # Per-cascade frustum culling
        from engine.gfx.culling import FrustumCuller

        self.cullers = [FrustumCuller(max_entities=100_000) for _ in range(4)]
        self.cascade_visible_masks: list[np.ndarray | None] = [None] * 4
        self.cascade_draw_batches: list[list[tuple]] = [[] for _ in range(4)]

    # ------------------------------------------------------------------
    # Atlas layout
    # ------------------------------------------------------------------

    def _update_cascade_rects(self) -> None:
        """Rebuilds the atlas layout.

        With ``cascade_resolutions`` unset this is the legacy fixed 2x2 quadrant
        layout. When set, the per-cascade sizes are shelf-packed into the atlas
        so the near cascade can claim a larger share of the texel budget than the
        far ones. Packing falls back to the equal layout if the request does not
        fit, so a bad config degrades instead of producing overlapping rects.
        """
        size = int(self.atlas_size)
        rects = self._pack_cascade_rects(size)
        self.cascade_rects = rects
        inv = 1.0 / float(max(size, 1))
        self.cascade_rect_uv = [(x * inv, y * inv, w * inv, h * inv) for (x, y, w, h) in rects]
        # Refresh the pre-allocated upload buffer in place (no per-frame allocation).
        self.cascade_rect_uv_array[:] = np.asarray(self.cascade_rect_uv, dtype=np.float32)

    def _equal_rects(self, size: int) -> list[tuple[int, int, int, int]]:
        half = size // 2
        return [
            (0, 0, half, half),
            (half, 0, half, half),
            (0, half, half, half),
            (half, half, half, half),
        ]

    def _pack_cascade_rects(self, size: int) -> list[tuple[int, int, int, int]]:
        """Shelf-packs per-cascade square sizes into the atlas, in cascade order."""
        resolutions = self.cascade_resolutions
        if not resolutions:
            return self._equal_rects(size)

        count = max(1, min(4, self.cascade_count))
        sizes = [max(1, min(size, int(resolutions[i]) if i < len(resolutions) else size // 2)) for i in range(count)]

        # Tallest first so each shelf wastes as little vertical space as possible.
        order = sorted(range(count), key=lambda i: (-sizes[i], i))
        placed: list[tuple[int, int, int, int] | None] = [None] * count
        cursor_x = 0
        cursor_y = 0
        shelf_height = 0
        for index in order:
            side = sizes[index]
            if cursor_x + side > size:
                cursor_y += shelf_height
                cursor_x = 0
                shelf_height = 0
            if cursor_y + side > size:
                # Does not fit: fall back to the equal layout rather than
                # emitting overlapping rects.
                return self._equal_rects(size)
            placed[index] = (cursor_x, cursor_y, side, side)
            cursor_x += side
            shelf_height = max(shelf_height, side)

        return [rect if rect is not None else (0, 0, 1, 1) for rect in placed]

    def cascade_texel_size(self, cascade_idx: int) -> int:
        """Square texel size of one cascade's atlas rect."""
        return int(self.cascade_rects[max(0, min(3, int(cascade_idx)))][2])

    def set_cascade_resolutions(self, resolutions: tuple[int, ...] | None) -> None:
        """Swaps the per-cascade rect sizes and rebuilds the atlas layout."""
        new_value = tuple(int(v) for v in resolutions) if resolutions else None
        if new_value == self.cascade_resolutions:
            return
        self.cascade_resolutions = new_value
        self._update_cascade_rects()

    # ------------------------------------------------------------------
    # Split distribution
    # ------------------------------------------------------------------

    def update_splits(
        self,
        max_distance: float,
        cascade_count: int | None = None,
        lambda_val: float = 0.85,
        cascade_distances: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Computes cascade splits, either explicit or PSSM-derived.

        ``cascade_distances`` sets each cascade's far plane directly, which is
        how the debug sliders author the distribution: cascade 0 is the tightest
        near-field slice and cascade 3 reaches the far plane. Values are clamped
        to be strictly increasing so cascade selection stays valid.

        When ``cascade_distances`` is ``None`` the distribution falls back to
        PSSM: ``max_distance`` is the far plane of the last cascade and
        ``lambda_val`` blends between uniform (0.0) and logarithmic (1.0)
        spacing. The blend is automatically reduced towards uniform when the
        requested value would push the adjacent span ratio above
        :data:`MAX_SPAN_RATIO`.
        """
        if cascade_count is not None:
            self.cascade_count = max(1, min(4, int(cascade_count)))
        count = max(1, min(4, self.cascade_count))
        near = CASCADE_NEAR

        if cascade_distances is not None:
            # Explicit authoring: each cascade's far plane is given directly.
            splits = [max(near + 1e-3, float(v)) for v in cascade_distances[:4]]
            # Pad short tuples by repeating the last value.
            while len(splits) < 4:
                splits.append(splits[-1])
            # Keep the sequence strictly increasing so cascade selection is valid.
            for index in range(1, len(splits)):
                if splits[index] <= splits[index - 1]:
                    splits[index] = splits[index - 1] + 1e-3
            self.cascade_distances = tuple(splits)
            self.max_distance = splits[count - 1]
            self.split_lambda = min(1.0, max(0.0, float(lambda_val)))
            self.split_distances = tuple(splits)
            return

        self.cascade_distances = None
        self.max_distance = max(float(max_distance), near + 1.0)
        far = max(self.max_distance, near + 1.0)
        blend = min(1.0, max(0.0, float(lambda_val)))
        self.split_lambda = blend

        if count == 1:
            splits = [far]
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

        splits.extend([far] * (4 - len(splits)))
        self.split_distances = tuple(splits[:4])

    # ------------------------------------------------------------------
    # Cascade fitting
    # ------------------------------------------------------------------

    def compute_cascade_matrices(
        self,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        sun_dir: np.ndarray,
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

        # Light basis. A sun directly overhead makes the default up vector
        # degenerate, so fall back to a different axis.
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(sun, world_up))) > 0.98:
            world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        light_right = np.cross(sun, world_up)
        light_right /= max(float(np.linalg.norm(light_right)), 1e-8)
        light_up = np.cross(light_right, sun)
        light_up /= max(float(np.linalg.norm(light_up)), 1e-8)

        camera_pos64 = np.asarray(camera_pos, dtype=np.float64)
        count = max(1, min(4, self.cascade_count))
        blend_fraction = min(0.5, max(0.0, float(blend_fraction)))
        tan_half_fov_y = math.tan(math.radians(max(1.0, min(179.0, fovy_deg))) * 0.5)
        safe_aspect = max(float(aspect), 1e-4)
        split_near = CASCADE_NEAR

        caster_centers: np.ndarray | None = None
        caster_extents: np.ndarray | None = None
        if aabbs is not None and active_count > 0:
            caster_count = min(int(active_count), len(aabbs))
            caster_centers = np.asarray(aabbs[:caster_count, :3], dtype=np.float64)
            caster_extents = np.asarray(aabbs[:caster_count, 3:6], dtype=np.float64)

        for cascade in range(count):
            split_far = self.split_distances[cascade]
            # Overlap neighbouring slices slightly so a receiver near a split
            # boundary is covered by both cascades and the hand-off is seamless.
            fit_near = max(CASCADE_NEAR, split_near - split_near * blend_fraction) if cascade > 0 else CASCADE_NEAR
            fit_far = (
                min(self.max_distance, split_far + split_far * blend_fraction)
                if cascade < count - 1
                else split_far
            )

            center_distance = (fit_near + fit_far) * 0.5
            # Bounding sphere of the slice. Rotation-invariant, so the depth
            # range does not re-scale when the camera yaws.
            radius = math.sqrt(
                ((fit_far - fit_near) * 0.5) ** 2 + (fit_far * tan_half_fov_y) ** 2 * (1.0 + safe_aspect * safe_aspect)
            )
            radius = max(math.ceil(radius * 1.025 * 16.0) / 16.0, 0.5)
            slice_center = camera_pos64 + forward * center_distance

            center_x = float(np.dot(light_right, slice_center))
            center_y = float(np.dot(light_up, slice_center))

            # Snap the light-space centre to whole texels. Without this the map
            # slides sub-texel every frame and the shadow edge shimmers.
            quad_size = max(self.cascade_rects[cascade][2], 1)
            texel_world = (2.0 * radius) / float(quad_size)
            center_x = math.floor(center_x / texel_world + 0.5) * texel_world
            center_y = math.floor(center_y / texel_world + 0.5) * texel_world

            # Depth range from the slice's bounding sphere, not from the eight
            # frustum corners projected onto the light axis. Projecting the
            # corners makes the range depend on camera yaw, which re-scales the
            # depth precision every time the camera turns.
            sphere_center_z = float(np.dot(sun, slice_center))
            # Quantise the slab origin. ``slice_center`` slides continuously with
            # the camera, so an unsnapped origin re-scales the depth encoding
            # every frame and the shadow edge creeps as you move.
            depth_quantum = max(radius * 2.0 / 512.0, 1e-3)
            sphere_center_z = math.floor(sphere_center_z / depth_quantum) * depth_quantum
            depth_min = sphere_center_z - radius
            depth_max = sphere_center_z + radius

            if caster_centers is not None and caster_extents is not None:
                cx = caster_centers @ light_right
                cy = caster_centers @ light_up
                cz = caster_centers @ sun
                ex = caster_extents @ np.abs(light_right)
                ey = caster_extents @ np.abs(light_up)
                ez = caster_extents @ np.abs(sun)
                # Casters are selected against a slightly expanded XY footprint so
                # a caster that grazes the border does not pop in and out as the
                # camera turns. The expansion is a fixed fraction of the cascade
                # radius, so it is itself rotation-invariant.
                margin = radius * 0.05
                overlaps_xy = (
                    (cx + ex >= center_x - radius - margin)
                    & (cx - ex <= center_x + radius + margin)
                    & (cy + ey >= center_y - radius - margin)
                    & (cy - ey <= center_y + radius + margin)
                    & (cz - ez <= depth_max + radius)
                    & (cz + ez >= depth_min - self.max_distance)
                )
                if np.any(overlaps_xy):
                    depth_min = min(depth_min, float(np.min((cz - ez)[overlaps_xy])))
                    depth_max = max(depth_max, float(np.max((cz + ez)[overlaps_xy])))

            # Bound how far a caster may stretch the depth slab. A ground plane
            # authored as a 10km quad overlaps every cascade in XY, and without
            # this clamp it drags the slab to ~17km for a 60m shadow distance.
            # A caster further than a few slice radii along the light axis
            # cannot reach into this slice anyway.
            reach = radius * self.max_caster_reach
            depth_min = max(depth_min, sphere_center_z - radius - reach)
            depth_max = min(depth_max, sphere_center_z + radius + reach)

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

        # Unused cascades mirror the last real one so the shader can index
        # u_LightViewProjection[0..3] without branching.
        for cascade in range(count, 4):
            self.light_matrices[cascade][:] = self.light_matrices[count - 1]

        return self.light_matrices

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
        # Hardware depth-comparison view of the same texture. A tap through this
        # sampler returns a bilinear-filtered 2x2 PCF result, so one fetch does
        # the work of four manual fetches.
        self.compare_sampler = self.ctx.sampler(
            texture=self.depth_texture,
            compare_func="<=",
            filter=(moderngl.LINEAR, moderngl.LINEAR),
            repeat_x=False,
            repeat_y=False,
        )

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
        if self.compare_sampler is not None:
            self.compare_sampler.release()
            self.compare_sampler = None
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.depth_texture is not None:
            self.depth_texture.release()
            self.depth_texture = None
