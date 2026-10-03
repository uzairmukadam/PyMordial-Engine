"""Unit and Integration Tests for Phase 4: Static / Dynamic Entity Separation."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager


class TestStaticDynamicSeparation:
    """Validates static entity tagging, memory slicing, and dirty flag tracking."""

    def test_entity_creation_is_static(self):
        ecs = EntityManager(max_entities=100)
        e_static = ecs.create_entity(position=(0.0, 0.0, 0.0), is_static=True)
        e_dyn = ecs.create_entity(position=(5.0, 0.0, 0.0), is_static=False)

        d_s = ecs.pool.get_dense_index(e_static)
        d_d = ecs.pool.get_dense_index(e_dyn)

        assert ecs.is_static[d_s] is True or ecs.is_static[d_s] == 1
        assert ecs.is_static[d_d] is False or ecs.is_static[d_d] == 0
        assert ecs.is_static_dirty is True

    def test_mark_static_split_and_slices(self):
        ecs = EntityManager(max_entities=100)
        # Create 3 static entities
        for i in range(3):
            ecs.create_entity(position=(float(i), 0.0, 0.0), is_static=True)

        ecs.mark_static_split()
        assert ecs.static_count == 3
        assert np.all(ecs.is_static[:3])
        assert ecs.is_static_dirty is True

        ecs.clear_static_dirty()
        assert ecs.is_static_dirty is False

        # Static slices
        s_trans = ecs.get_static_transforms_view()
        assert s_trans.shape == (3, 16)
        s_mat = ecs.get_static_materials_view()
        assert s_mat.shape == (3, 8)

        # Now spawn 2 dynamic entities
        _d1 = ecs.create_entity(position=(10.0, 1.0, 0.0), is_static=False)
        _d2 = ecs.create_entity(position=(20.0, 1.0, 0.0), is_static=False)
        assert ecs.active_count == 5

        # Dynamic entities must NOT mark static as dirty!
        assert ecs.is_static_dirty is False

        d_trans = ecs.get_dynamic_transforms_view()
        assert d_trans.shape == (2, 16)
        d_mat = ecs.get_dynamic_materials_view()
        assert d_mat.shape == (2, 8)

        # Positions in dynamic slice
        assert pytest.approx(d_trans[0, 12], abs=1e-4) == 10.0
        assert pytest.approx(d_trans[1, 12], abs=1e-4) == 20.0

    def test_dynamic_entity_destruction_preserves_static_clean(self):
        ecs = EntityManager(max_entities=100)
        for i in range(4):
            ecs.create_entity(is_static=True)
        ecs.mark_static_split()
        ecs.clear_static_dirty()

        d1 = ecs.create_entity(position=(1.0, 0.0, 0.0), is_static=False)
        _d2 = ecs.create_entity(position=(2.0, 0.0, 0.0), is_static=False)
        assert ecs.is_static_dirty is False

        # Destroy dynamic entity d1 (swap and pop within dynamic range)
        ecs.destroy_entity(d1)
        # Static dirty flag should remain False because static prefix was untouched!
        assert ecs.is_static_dirty is False
        assert ecs.active_count == 5
        assert ecs.static_count == 4

    def test_static_entity_modification_marks_dirty(self):
        ecs = EntityManager(max_entities=100)
        s1 = ecs.create_entity(position=(0.0, 0.0, 0.0), is_static=True)
        ecs.mark_static_split()
        ecs.clear_static_dirty()

        # Modify static entity transform
        proxy = ecs.get_transform_proxy(s1)
        proxy.position = (100.0, 0.0, 0.0)

        # Must trigger is_static_dirty for GPU re-upload!
        assert ecs.is_static_dirty is True
