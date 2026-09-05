"""Dense-to-Sparse Entity ID Allocator and Free-List Pool.

Maintains stable external 32-bit Entity IDs while keeping active entities
densely packed in contiguous slots for GPU buffer streaming.
"""

from __future__ import annotations
import numpy as np


class EntityPool:
    """Manages dense-to-sparse mapping and O(1) recycling for up to max_entities."""

    __slots__ = (
        "max_entities",
        "sparse",
        "dense",
        "free_list",
        "free_count",
        "active_count",
    )

    def __init__(self, max_entities: int = 100_000) -> None:
        self.max_entities = max_entities
        # Sparse maps EntityID -> DenseIndex (-1 indicates unallocated)
        self.sparse = np.full(max_entities, -1, dtype=np.int32)
        # Dense maps DenseIndex -> EntityID
        self.dense = np.zeros(max_entities, dtype=np.uint32)
        # Pre-populate free-list with all IDs [0 .. max_entities - 1] in reverse order
        # so allocation starts from 0 upwards
        self.free_list = np.arange(max_entities - 1, -1, -1, dtype=np.uint32)
        self.free_count = max_entities
        self.active_count = 0

    def allocate(self) -> int:
        """Allocates a stable EntityID in O(1) time.

        Returns:
            The allocated 32-bit unsigned entity ID.

        Raises:
            RuntimeError: If max_entities capacity is exceeded.
        """
        if self.free_count == 0:
            raise RuntimeError(f"EntityPool capacity exceeded ({self.max_entities})")

        self.free_count -= 1
        entity_id = int(self.free_list[self.free_count])

        dense_idx = self.active_count
        self.dense[dense_idx] = entity_id
        self.sparse[entity_id] = dense_idx
        self.active_count += 1

        return entity_id

    def free(self, entity_id: int) -> tuple[int, int] | None:
        """Frees an entity ID in O(1) using swap-and-pop.

        Args:
            entity_id: The entity ID to free.

        Returns:
            A tuple of (removed_dense_idx, swapped_from_dense_idx) if an entity
            was swapped to maintain density, or None if the entity ID was invalid.
            If the deleted entity was already at the last dense position,
            removed_dense_idx == swapped_from_dense_idx.
        """
        if not self.is_valid(entity_id):
            return None

        dense_idx = int(self.sparse[entity_id])
        last_dense_idx = self.active_count - 1

        if dense_idx != last_dense_idx:
            # Swap last active entity into this slot
            last_entity_id = int(self.dense[last_dense_idx])
            self.dense[dense_idx] = last_entity_id
            self.sparse[last_entity_id] = dense_idx

        # Invalidate removed entity
        self.sparse[entity_id] = -1
        self.dense[last_dense_idx] = 0

        # Return entity_id to free list
        self.free_list[self.free_count] = entity_id
        self.free_count += 1
        self.active_count -= 1

        return dense_idx, last_dense_idx

    def is_valid(self, entity_id: int) -> bool:
        """Returns True if the entity_id is currently allocated and valid."""
        if 0 <= entity_id < self.max_entities:
            dense_idx = self.sparse[entity_id]
            return 0 <= dense_idx < self.active_count
        return False

    def get_dense_index(self, entity_id: int) -> int:
        """Returns the current dense index for an entity ID, or -1 if invalid."""
        if 0 <= entity_id < self.max_entities:
            dense_idx = int(self.sparse[entity_id])
            if 0 <= dense_idx < self.active_count:
                return dense_idx
        return -1

    def get_entity_id(self, dense_index: int) -> int:
        """Returns the entity ID at dense_index, or raises IndexError."""
        if 0 <= dense_index < self.active_count:
            return int(self.dense[dense_index])
        raise IndexError(f"Dense index {dense_index} out of active range [0, {self.active_count})")

    def get_active_dense_indices(self) -> np.ndarray:
        """Returns a slice view of all active dense entity IDs."""
        return self.dense[: self.active_count]

    def clear(self) -> None:
        """Resets the pool to initial empty state."""
        self.sparse.fill(-1)
        self.dense.fill(0)
        self.free_list[:] = np.arange(self.max_entities - 1, -1, -1, dtype=np.uint32)
        self.free_count = self.max_entities
        self.active_count = 0
