"""Frustum Culling and Vectorized AABB Occlusion System.

Provides zero-allocation camera frustum plane extraction (supporting GL_ZERO_TO_ONE
and Reversed-Z clip spaces) and SIMD-vectorized NumPy AABB culling for Multi-Draw
Indirect (MDI) batch filtering.
"""

from __future__ import annotations
import math
import numpy as np


class FrustumCuller:
    """Extracts 6 frustum planes and evaluates vectorized AABB visibility with zero per-frame allocations."""

    __slots__ = (
        "max_entities",
        "planes",
        "_planes_xyz",
        "_planes_d",
        "_abs_planes_xyz",
        "visible_mask",
        "_d_scratch",
        "_r_scratch",
        "_filtered_batches",
        "stats_visible_entities",
        "stats_culled_entities",
        "stats_visible_commands",
        "stats_culled_commands",
    )

    def __init__(self, max_entities: int = 100_000) -> None:
        self.max_entities = max_entities
        # 6 planes: [Left, Right, Bottom, Top, Near, Far], layout (6, 4) float32: [A, B, C, D]
        self.planes = np.zeros((6, 4), dtype=np.float32)
        self._planes_xyz = self.planes[:, 0:3]
        self._planes_d = self.planes[:, 3]
        self._abs_planes_xyz = np.zeros((6, 3), dtype=np.float32)

        # Pre-allocated mask and scratch buffers for zero-allocation testing
        self.visible_mask = np.ones(max_entities, dtype=bool)
        self._d_scratch = np.zeros((max_entities, 6), dtype=np.float32)
        self._r_scratch = np.zeros((max_entities, 6), dtype=np.float32)
        self._filtered_batches: list[tuple] = []

        # Telemetry statistics
        self.stats_visible_entities: int = 0
        self.stats_culled_entities: int = 0
        self.stats_visible_commands: int = 0
        self.stats_culled_commands: int = 0

    def extract_planes(self, vp_matrix: np.ndarray, reverse_z: bool = True) -> None:
        """Extracts and normalizes the 6 camera frustum planes from a column-major 4x4 VP matrix.

        Plane equation: A*x + B*y + C*z + D = 0.
        Point (x, y, z) is inside the half-space if A*x + B*y + C*z + D >= 0.

        Args:
            vp_matrix: 16-element 1D array or (4, 4) array in column-major order.
            reverse_z: If True, near plane corresponds to Z=1.0 and far to Z=0.0 (Reversed-Z).
        """
        # Reshape to (4, 4) in column-major order so m[row, col] is standard row-major access
        m = vp_matrix.reshape((4, 4), order="F")

        # Row 0..3 vectors
        r0 = m[0]
        r1 = m[1]
        r2 = m[2]
        r3 = m[3]

        # 0: Left Plane (w + x >= 0)
        self.planes[0] = r3 + r0
        # 1: Right Plane (w - x >= 0)
        self.planes[1] = r3 - r0
        # 2: Bottom Plane (w + y >= 0)
        self.planes[2] = r3 + r1
        # 3: Top Plane (w - y >= 0)
        self.planes[3] = r3 - r1

        if reverse_z:
            # Reversed-Z under GL_ZERO_TO_ONE:
            # Near plane is at z_ndc = 1.0 (w - z >= 0)
            self.planes[4] = r3 - r2
            # Far plane is at z_ndc = 0.0 (z >= 0)
            self.planes[5] = r2
        else:
            # Standard Z under GL_ZERO_TO_ONE:
            # Near plane is at z_ndc = 0.0 (z >= 0)
            self.planes[4] = r2
            # Far plane is at z_ndc = 1.0 (w - z >= 0)
            self.planes[5] = r3 - r2

        # Normalize plane equations
        lengths = np.sqrt(
            self.planes[:, 0] ** 2 + self.planes[:, 1] ** 2 + self.planes[:, 2] ** 2
        )
        # Avoid division by zero
        lengths[lengths < 1e-7] = 1.0
        self.planes /= lengths[:, np.newaxis]

        # Cache absolute values of normal components for AABB projection intervals
        np.abs(self._planes_xyz, out=self._abs_planes_xyz)

    def cull_aabbs(self, aabbs: np.ndarray, count: int) -> np.ndarray:
        """Performs vectorized AABB vs Frustum Planes intersection tests.

        Args:
            aabbs: (N, 6) float32 array: [cx, cy, cz, ex, ey, ez] (centers and half-extents).
            count: Number of active entity AABBs to evaluate.

        Returns:
            Boolean array of shape (count,) where True indicates inside or intersecting frustum.
        """
        if count <= 0:
            self.stats_visible_entities = 0
            self.stats_culled_entities = 0
            return self.visible_mask[:0]

        count = min(count, self.max_entities)
        centers = aabbs[:count, 0:3]
        extents = aabbs[:count, 3:6]

        # Signed distance from center to plane: d = C @ N.T + D
        d = np.matmul(centers, self._planes_xyz.T, out=self._d_scratch[:count])
        d += self._planes_d

        # Maximum projection interval radius: r = E @ |N|.T
        r = np.matmul(extents, self._abs_planes_xyz.T, out=self._r_scratch[:count])

        # Outside if d < -r for ANY plane -> Inside if d >= -r for ALL planes
        # np.all(d >= -r, axis=1)
        sub_mask = self.visible_mask[:count]
        np.all(d >= -r, axis=1, out=sub_mask)

        vis_count = int(np.count_nonzero(sub_mask))
        self.stats_visible_entities = vis_count
        self.stats_culled_entities = count - vis_count

        return sub_mask

    def filter_draw_batches(
        self,
        draw_batches: list[tuple],
        visible_mask: np.ndarray,
    ) -> list[tuple]:
        """Filters MDI draw batches according to the entity visibility mask.

        Batches outside the frustum are completely omitted. Partially visible batches
        are split into contiguous visible sub-runs to minimize draw commands.

        Args:
            draw_batches: List of tuples (mesh_item, count, base_instance[, is_tess]).
            visible_mask: Boolean array where index corresponds to entity dense index.

        Returns:
            Filtered list of draw batch tuples ready for MDI command generation.
        """
        self._filtered_batches.clear()
        mask_len = len(visible_mask)
        in_cmd_count = len(draw_batches)
        out_cmd_count = 0

        for item in draw_batches:
            if len(item) == 4:
                mesh_item, count, base_inst, is_tess = item
            else:
                mesh_item, count, base_inst = item
                is_tess = False

            if count <= 0 or base_inst >= mask_len:
                continue

            end_inst = min(base_inst + count, mask_len)
            sub = visible_mask[base_inst:end_inst]

            # 1. Single-instance fast path
            if count == 1:
                if sub[0]:
                    self._filtered_batches.append((mesh_item, 1, base_inst, is_tess))
                    out_cmd_count += 1
                continue

            # 2. Check if all visible (common case when looking towards scene center)
            any_vis = np.any(sub)
            if not any_vis:
                # Fully culled: 0 draw commands
                continue

            all_vis = np.all(sub)
            if all_vis:
                # Fully visible: single instanced draw command
                self._filtered_batches.append((mesh_item, count, base_inst, is_tess))
                out_cmd_count += 1
                continue

            # 3. Partially visible: find contiguous runs of True
            # Pad with False on both ends to detect transitions via diff
            padded = np.empty(len(sub) + 2, dtype=bool)
            padded[0] = False
            padded[-1] = False
            padded[1:-1] = sub
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                run_len = int(e - s)
                if run_len > 0:
                    self._filtered_batches.append((mesh_item, run_len, base_inst + int(s), is_tess))
                    out_cmd_count += 1

        self.stats_visible_commands = out_cmd_count
        self.stats_culled_commands = max(0, in_cmd_count - out_cmd_count)
        return self._filtered_batches
