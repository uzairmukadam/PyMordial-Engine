"""Unit and Integration Tests for Phase 3: Frustum Culling & MDI Batch Filtering."""

import math
import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.core.math_utils import matrix_look_at, matrix_perspective, mat4_mul
from engine.gfx.culling import FrustumCuller


def create_test_vp(
    cam_pos=(0.0, 2.0, 10.0),
    cam_target=(0.0, 0.0, 0.0),
    fovy_deg=60.0,
    aspect=16.0 / 9.0,
    near=0.1,
    far=100.0,
    reverse_z=True,
) -> np.ndarray:
    """Creates a 16-element column-major VP matrix for deterministic culling testing."""
    view = np.zeros(16, dtype=np.float32)
    proj = np.zeros(16, dtype=np.float32)
    vp = np.zeros(16, dtype=np.float32)

    matrix_look_at(cam_pos, cam_target, up=(0.0, 1.0, 0.0), out=view)
    matrix_perspective(math.radians(fovy_deg), aspect, near=near, far=far, reverse_z=reverse_z, out=proj)
    mat4_mul(proj, view, out=vp)
    return vp


class TestFrustumPlanes:
    """Validates plane extraction geometry and normalization."""

    def test_plane_extraction_and_normalization(self):
        culler = FrustumCuller(max_entities=100)
        vp = create_test_vp(reverse_z=True)
        culler.extract_planes(vp, reverse_z=True)

        assert culler.planes.shape == (6, 4)
        # All 6 plane normal vectors must be unit length (length ~ 1.0)
        lengths = np.sqrt(np.sum(culler.planes[:, 0:3] ** 2, axis=1))
        for l in lengths:
            assert pytest.approx(l, abs=1e-4) == 1.0

    def test_points_inside_and_outside_frustum(self):
        culler = FrustumCuller(max_entities=10)
        # Camera at (0, 0, 10), looking towards origin (0, 0, 0) along -Z
        vp = create_test_vp(cam_pos=(0.0, 0.0, 10.0), cam_target=(0.0, 0.0, 0.0), reverse_z=True)
        culler.extract_planes(vp, reverse_z=True)

        aabbs = np.zeros((4, 6), dtype=np.float32)
        # 0: In front of camera, well inside frustum
        aabbs[0, 0:3] = [0.0, 0.0, 0.0]     # Center
        aabbs[0, 3:6] = [1.0, 1.0, 1.0]     # Half extents

        # 1: Directly behind camera
        aabbs[1, 0:3] = [0.0, 0.0, 25.0]
        aabbs[1, 3:6] = [1.0, 1.0, 1.0]

        # 2: Far off to the left outside FOV
        aabbs[2, 0:3] = [-100.0, 0.0, 0.0]
        aabbs[2, 3:6] = [1.0, 1.0, 1.0]

        # 3: Beyond far plane (Z = -200)
        aabbs[3, 0:3] = [0.0, 0.0, -200.0]
        aabbs[3, 3:6] = [1.0, 1.0, 1.0]

        mask = culler.cull_aabbs(aabbs, count=4)
        assert mask[0] is True or mask[0] == 1  # Inside
        assert mask[1] is False or mask[1] == 0 # Behind
        assert mask[2] is False or mask[2] == 0 # Left
        assert mask[3] is False or mask[3] == 0 # Beyond Far


class TestBatchFiltering:
    """Validates MDI batch list filtering and contiguous run extraction."""

    def test_filter_draw_batches_all_visible(self):
        culler = FrustumCuller(max_entities=100)
        mask = np.ones(10, dtype=bool)

        batches = [
            ("cube", 5, 0, False),
            ("sphere", 1, 5, False),
        ]
        res = culler.filter_draw_batches(batches, mask)
        assert len(res) == 2
        assert res[0] == ("cube", 5, 0, False)
        assert res[1] == ("sphere", 1, 5, False)
        assert culler.stats_visible_commands == 2
        assert culler.stats_culled_commands == 0

    def test_filter_draw_batches_fully_culled(self):
        culler = FrustumCuller(max_entities=100)
        mask = np.zeros(10, dtype=bool)

        batches = [
            ("cube", 5, 0, False),
            ("sphere", 1, 5, False),
        ]
        res = culler.filter_draw_batches(batches, mask)
        assert len(res) == 0
        assert culler.stats_visible_commands == 0
        assert culler.stats_culled_commands == 2

    def test_filter_draw_batches_partially_visible_split(self):
        culler = FrustumCuller(max_entities=100)
        mask = np.array([True, True, False, False, True, False, True, True, True, False], dtype=bool)

        batches = [
            ("cube", 10, 0, False),
        ]
        res = culler.filter_draw_batches(batches, mask)
        # Expected runs:
        # indices 0..1 (len 2, base 0)
        # index 4 (len 1, base 4)
        # indices 6..8 (len 3, base 6)
        assert len(res) == 3
        assert res[0] == ("cube", 2, 0, False)
        assert res[1] == ("cube", 1, 4, False)
        assert res[2] == ("cube", 3, 6, False)


class TestECSAABBIntegration:
    """Verifies that EntityManager computes accurate AABBs upon TRS recomputations."""

    def test_ecs_aabb_computation_and_recompute(self):
        ecs = EntityManager(max_entities=100)
        e_id = ecs.create_entity(
            position=(10.0, 5.0, -20.0),
            scale=(4.0, 2.0, 8.0),
        )
        d_idx = ecs.pool.get_dense_index(e_id)

        aabb = ecs.aabbs[d_idx]
        # Center should match position
        assert pytest.approx(aabb[0], abs=1e-4) == 10.0
        assert pytest.approx(aabb[1], abs=1e-4) == 5.0
        assert pytest.approx(aabb[2], abs=1e-4) == -20.0

        # Half-extents for scale (4, 2, 8) with unit mesh (extents 0.5) -> (2, 1, 4)
        assert pytest.approx(aabb[3], abs=1e-4) == 2.0
        assert pytest.approx(aabb[4], abs=1e-4) == 1.0
        assert pytest.approx(aabb[5], abs=1e-4) == 4.0

        # Now move entity using proxy
        proxy = ecs.get_transform_proxy(e_id)
        proxy.position = (0.0, 1.0, 0.0)
        aabb2 = ecs.aabbs[d_idx]
        assert pytest.approx(aabb2[0], abs=1e-4) == 0.0
        assert pytest.approx(aabb2[1], abs=1e-4) == 1.0
        assert pytest.approx(aabb2[2], abs=1e-4) == 0.0

    def test_sync_dynamic_aabbs(self):
        ecs = EntityManager(max_entities=100)
        # Static entity
        s_id = ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(10.0, 1.0, 10.0), is_static=True)
        ecs.mark_static_split()

        # Dynamic entity (e.g. car chassis or attached wheel)
        d_id = ecs.create_entity(position=(0.0, 1.0, 0.0), scale=(2.0, 1.0, 4.0), is_static=False)
        d_idx = ecs.pool.get_dense_index(d_id)

        # Initially center is (0, 1, 0)
        assert pytest.approx(ecs.aabbs[d_idx, 0], abs=1e-4) == 0.0
        assert pytest.approx(ecs.aabbs[d_idx, 1], abs=1e-4) == 1.0
        assert pytest.approx(ecs.aabbs[d_idx, 2], abs=1e-4) == 0.0

        # Simulate vehicle script directly writing new world_transforms (e.g. driving 50m forward to (0, 1, -50))
        ecs.world_transforms[d_idx, 12:15] = [0.0, 1.0, -50.0]
        # Before sync, AABB is still at (0, 1, 0)
        assert pytest.approx(ecs.aabbs[d_idx, 2], abs=1e-4) == 0.0

        # Run vectorized sync
        ecs.sync_dynamic_aabbs()

        # After sync, AABB center has moved to (0, 1, -50)
        assert pytest.approx(ecs.aabbs[d_idx, 0], abs=1e-4) == 0.0
        assert pytest.approx(ecs.aabbs[d_idx, 1], abs=1e-4) == 1.0
        assert pytest.approx(ecs.aabbs[d_idx, 2], abs=1e-4) == -50.0
        # Static entity AABB untouched
        s_idx = ecs.pool.get_dense_index(s_id)
        assert pytest.approx(ecs.aabbs[s_idx, 0], abs=1e-4) == 0.0
        assert pytest.approx(ecs.aabbs[s_idx, 3], abs=1e-4) == 5.0

