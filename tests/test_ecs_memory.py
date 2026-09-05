"""Unit tests for flat contiguous memory tables and EntityPool."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager, TransformProxy
from engine.core.entity_pool import EntityPool


class TestEntityPool:
    def test_allocation_and_dense_mapping(self, entity_pool: EntityPool):
        id0 = entity_pool.allocate()
        id1 = entity_pool.allocate()
        id2 = entity_pool.allocate()

        assert id0 == 0
        assert id1 == 1
        assert id2 == 2
        assert entity_pool.active_count == 3

        assert entity_pool.get_dense_index(id0) == 0
        assert entity_pool.get_dense_index(id1) == 1
        assert entity_pool.get_dense_index(id2) == 2

    def test_swap_and_pop_removal(self, entity_pool: EntityPool):
        ids = [entity_pool.allocate() for _ in range(5)]  # [0, 1, 2, 3, 4]
        assert entity_pool.active_count == 5

        # Remove middle entity (ID 2, dense index 2)
        res = entity_pool.free(2)
        assert res is not None
        removed_dense, swapped_from = res
        assert removed_dense == 2
        assert swapped_from == 4  # Last entity (ID 4) swapped into slot 2

        assert entity_pool.active_count == 4
        assert not entity_pool.is_valid(2)
        assert entity_pool.get_dense_index(2) == -1

        # Check that ID 4 is now at dense index 2
        assert entity_pool.get_dense_index(4) == 2
        assert entity_pool.get_entity_id(2) == 4

        # Allocate again -> should reuse recycled ID 2
        new_id = entity_pool.allocate()
        assert new_id == 2
        assert entity_pool.get_dense_index(2) == 4  # Appended at end of dense array (index 4)
        assert entity_pool.active_count == 5

    def test_free_invalid_entity(self, entity_pool: EntityPool):
        assert entity_pool.free(999) is None
        assert not entity_pool.is_valid(999)


class TestEntityManagerMemory:
    def test_c_contiguity_and_dtypes(self, ecs: EntityManager):
        assert ecs.world_transforms.flags["C_CONTIGUOUS"]
        assert ecs.rigid_body_state.flags["C_CONTIGUOUS"]
        assert ecs.material_data.flags["C_CONTIGUOUS"]
        assert ecs.scales.flags["C_CONTIGUOUS"]

        assert ecs.world_transforms.dtype == np.float32
        assert ecs.rigid_body_state.dtype == np.float32
        assert ecs.material_data.dtype == np.float32
        assert ecs.scales.dtype == np.float32

        assert ecs.world_transforms.shape == (1000, 16)
        assert ecs.rigid_body_state.shape == (2, 1000, 7)
        assert ecs.material_data.shape == (1000, 8)
        assert ecs.scales.shape == (1000, 3)

    def test_entity_creation_and_destruction(self, ecs: EntityManager):
        e0 = ecs.create_entity(position=(1.0, 2.0, 3.0), color=(1.0, 0.0, 0.0))
        e1 = ecs.create_entity(position=(4.0, 5.0, 6.0), color=(0.0, 1.0, 0.0))
        e2 = ecs.create_entity(position=(7.0, 8.0, 9.0), color=(0.0, 0.0, 1.0))

        assert ecs.active_count == 3
        # Check initial positions in current physics buffer
        np.testing.assert_allclose(ecs.rigid_body_state[1, 0, 0:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ecs.rigid_body_state[1, 1, 0:3], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(ecs.rigid_body_state[1, 2, 0:3], [7.0, 8.0, 9.0])

        # Destroy middle entity e1
        assert ecs.destroy_entity(e1)
        assert ecs.active_count == 2

        # Dense slot 1 should now contain e2's data
        np.testing.assert_allclose(ecs.rigid_body_state[1, 1, 0:3], [7.0, 8.0, 9.0])
        np.testing.assert_allclose(ecs.material_data[1, 0:3], [0.0, 0.0, 1.0])

        # Slot 2 (previously e2) should now be cleared
        np.testing.assert_allclose(ecs.rigid_body_state[1, 2, 0:3], [0.0, 0.0, 0.0])

    def test_transform_proxy(self, ecs: EntityManager):
        ent = ecs.create_entity(position=(0.0, 0.0, 0.0))
        proxy = ecs.get_transform_proxy(ent)

        # Modify via proxy
        proxy.position = (5.0, 15.0, -25.0)
        proxy.scale = (2.0, 3.0, 4.0)

        dense_idx = ecs.pool.get_dense_index(ent)
        np.testing.assert_allclose(ecs.rigid_body_state[1, dense_idx, 0:3], [5.0, 15.0, -25.0])
        np.testing.assert_allclose(ecs.scales[dense_idx], [2.0, 3.0, 4.0])

        # Verify matrix column 3 translation
        mat = proxy.matrix
        np.testing.assert_allclose(mat[12:15], [5.0, 15.0, -25.0])

    def test_instant_pie_snapshot_and_restore(self, ecs: EntityManager):
        e0 = ecs.create_entity(position=(1.0, 2.0, 3.0))
        e1 = ecs.create_entity(position=(4.0, 5.0, 6.0))

        # Take pre-simulation snapshot
        snapshot = ecs.snapshot_memory()

        # Simulate gameplay modifications
        proxy0 = ecs.get_transform_proxy(e0)
        proxy0.position = (99.0, 99.0, 99.0)
        e2 = ecs.create_entity(position=(7.0, 8.0, 9.0))
        assert ecs.active_count == 3

        # Restore snapshot (instant PIE reset)
        ecs.restore_snapshot(snapshot)

        assert ecs.active_count == 2
        np.testing.assert_allclose(ecs.rigid_body_state[1, 0, 0:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ecs.rigid_body_state[1, 1, 0:3], [4.0, 5.0, 6.0])
        assert not ecs.pool.is_valid(e2)
