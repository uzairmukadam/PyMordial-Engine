"""Contiguous Flat Memory Tables and Entity Manager.

Provides zero-overhead, C-contiguous NumPy float32 storage for transforms,
physics state, and material parameters, mirroring directly into OpenGL SSBOs.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from engine.core.entity_pool import EntityPool
from engine.core.math_utils import (
    batch_nlerp_and_compose_mat4,
    trs_to_mat4,
    quat_identity,
)


class TransformProxy:
    """Lightweight zero-allocation property proxy for entity transformation.

    Reads and writes directly into the underlying contiguous memory tables.
    """

    __slots__ = ("_mgr", "_entity_id")

    def __init__(self, mgr: EntityManager, entity_id: int) -> None:
        self._mgr = mgr
        self._entity_id = entity_id

    @property
    def entity_id(self) -> int:
        return self._entity_id

    @property
    def dense_index(self) -> int:
        idx = self._mgr.pool.get_dense_index(self._entity_id)
        if idx < 0:
            raise KeyError(f"Entity {self._entity_id} is no longer active.")
        return idx

    @property
    def position(self) -> np.ndarray:
        """Position [x, y, z] in world space."""
        return self._mgr.rigid_body_state[1, self.dense_index, 0:3]

    @position.setter
    def position(self, val: np.ndarray | tuple[float, float, float] | list[float]) -> None:
        idx = self.dense_index
        self._mgr.rigid_body_state[1, idx, 0:3] = val
        self._mgr.recompute_matrix(idx)

    @property
    def rotation(self) -> np.ndarray:
        """Rotation quaternion [x, y, z, w]."""
        return self._mgr.rigid_body_state[1, self.dense_index, 3:7]

    @rotation.setter
    def rotation(self, val: np.ndarray | tuple[float, float, float, float] | list[float]) -> None:
        idx = self.dense_index
        self._mgr.rigid_body_state[1, idx, 3:7] = val
        self._mgr.recompute_matrix(idx)

    @property
    def scale(self) -> np.ndarray:
        """Local scale [sx, sy, sz]."""
        return self._mgr.scales[self.dense_index]

    @scale.setter
    def scale(self, val: np.ndarray | tuple[float, float, float] | list[float]) -> None:
        idx = self.dense_index
        self._mgr.scales[idx] = val
        self._mgr.recompute_matrix(idx)

    @property
    def matrix(self) -> np.ndarray:
        """Direct 16 float32 slice into WorldTransforms."""
        return self._mgr.world_transforms[self.dense_index]


class EntityManager:
    """Manages flat memory tables and entities for the engine."""

    __slots__ = (
        "max_entities",
        "pool",
        "world_transforms",
        "rigid_body_state",
        "material_data",
        "scales",
    )

    def __init__(self, max_entities: int = 100_000) -> None:
        self.max_entities = max_entities
        self.pool = EntityPool(max_entities)

        # 1. WorldTransforms: (N, 16) float32, C-contiguous
        # Mirrors layout(std430, binding = 1) readonly buffer TransformBuffer { mat4 u_WorldTransforms[]; };
        self.world_transforms = np.zeros((max_entities, 16), dtype=np.float32)
        assert self.world_transforms.flags["C_CONTIGUOUS"], "WorldTransforms must be C-contiguous"

        # 2. RigidBodyState Double Buffer: (2, N, 7) float32, C-contiguous
        # [0 = Previous Tick, 1 = Current Tick], Layout: [Px, Py, Pz, Qx, Qy, Qz, Qw]
        self.rigid_body_state = np.zeros((2, max_entities, 7), dtype=np.float32)
        # Default quaternion to identity [0, 0, 0, 1]
        self.rigid_body_state[:, :, 6] = 1.0
        assert self.rigid_body_state.flags["C_CONTIGUOUS"], "RigidBodyState must be C-contiguous"

        # 3. MaterialData: (N, 8) float32, C-contiguous
        # Layout: [R, G, B, Roughness, Metallic, AO, AlbedoTexID, NormalTexID]
        # Mirrors layout(std430, binding = 2) readonly buffer MaterialBuffer
        self.material_data = np.zeros((max_entities, 8), dtype=np.float32)
        # Defaults: Color = (1, 1, 1), Roughness = 0.5, Metallic = 0.0, AO = 1.0
        self.material_data[:, 0:3] = 1.0
        self.material_data[:, 3] = 0.5
        self.material_data[:, 4] = 0.0
        self.material_data[:, 5] = 1.0
        assert self.material_data.flags["C_CONTIGUOUS"], "MaterialData must be C-contiguous"

        # 4. Auxiliary Scales: (N, 3) float32, default 1.0
        self.scales = np.ones((max_entities, 3), dtype=np.float32)
        assert self.scales.flags["C_CONTIGUOUS"], "Scales must be C-contiguous"

    @property
    def active_count(self) -> int:
        return self.pool.active_count

    def create_entity(
        self,
        position: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | np.ndarray = (0.0, 0.0, 0.0, 1.0),
        scale: tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> int:
        """Allocates an entity and initializes its slots in contiguous memory tables."""
        entity_id = self.pool.allocate()
        dense_idx = self.pool.get_dense_index(entity_id)

        # Initialize current and previous physics state identically
        self.rigid_body_state[0, dense_idx, 0:3] = position
        self.rigid_body_state[0, dense_idx, 3:7] = rotation
        self.rigid_body_state[1, dense_idx, 0:3] = position
        self.rigid_body_state[1, dense_idx, 3:7] = rotation

        self.scales[dense_idx] = scale
        self.material_data[dense_idx, 0:3] = color
        self.material_data[dense_idx, 3] = roughness
        self.material_data[dense_idx, 4] = metallic
        self.material_data[dense_idx, 5] = 1.0  # AO
        self.material_data[dense_idx, 6:8] = 0.0  # Default tex IDs

        self.recompute_matrix(dense_idx)
        return entity_id

    def destroy_entity(self, entity_id: int) -> bool:
        """Destroys an entity and performs O(1) swap-and-pop row consolidation."""
        res = self.pool.free(entity_id)
        if res is None:
            return False

        dense_idx, last_dense_idx = res
        if dense_idx != last_dense_idx:
            # Move the last entity's data into the freed slot to keep buffers dense
            self.world_transforms[dense_idx] = self.world_transforms[last_dense_idx]
            self.rigid_body_state[:, dense_idx, :] = self.rigid_body_state[:, last_dense_idx, :]
            self.material_data[dense_idx] = self.material_data[last_dense_idx]
            self.scales[dense_idx] = self.scales[last_dense_idx]

        # Reset the now inactive last slot
        self.world_transforms[last_dense_idx].fill(0.0)
        self.rigid_body_state[:, last_dense_idx, :].fill(0.0)
        self.rigid_body_state[:, last_dense_idx, 6] = 1.0
        self.material_data[last_dense_idx].fill(0.0)
        self.scales[last_dense_idx].fill(1.0)

        return True

    def get_transform_proxy(self, entity_id: int) -> TransformProxy:
        """Returns a lightweight TransformProxy for an active entity."""
        if not self.pool.is_valid(entity_id):
            raise KeyError(f"Invalid or inactive entity ID: {entity_id}")
        return TransformProxy(self, entity_id)

    def recompute_matrix(self, dense_idx: int) -> None:
        """Recomputes the 4x4 transform matrix for a single dense entity slot."""
        pos = self.rigid_body_state[1, dense_idx, 0:3]
        rot = self.rigid_body_state[1, dense_idx, 3:7]
        scl = self.scales[dense_idx]
        self.world_transforms[dense_idx] = trs_to_mat4(pos, rot, scl)

    def cache_previous_physics_state(self) -> None:
        """Copies Current State (index 1) to Previous State (index 0) before physics tick."""
        count = self.pool.active_count
        if count > 0:
            np.copyto(self.rigid_body_state[0, :count], self.rigid_body_state[1, :count])

    def interpolate_render_transforms(self, alpha: float) -> None:
        """Sub-frame NLERP interpolation from prev_state to curr_state into WorldTransforms."""
        count = self.pool.active_count
        batch_nlerp_and_compose_mat4(
            self.rigid_body_state[0],
            self.rigid_body_state[1],
            alpha,
            self.world_transforms,
            count,
        )

    def get_active_transforms_view(self) -> np.ndarray:
        """Returns contiguous (active_count, 16) float32 slice for GPU SSBO upload."""
        return self.world_transforms[: self.pool.active_count]

    def get_active_materials_view(self) -> np.ndarray:
        """Returns contiguous (active_count, 8) float32 slice for GPU SSBO upload."""
        return self.material_data[: self.pool.active_count]

    def snapshot_memory(self) -> dict[str, Any]:
        """Creates a zero-allocation memcpy clone of state for Instant Play-In-Editor (PIE)."""
        return {
            "active_count": self.pool.active_count,
            "free_count": self.pool.free_count,
            "sparse": self.pool.sparse.copy(),
            "dense": self.pool.dense.copy(),
            "free_list": self.pool.free_list.copy(),
            "world_transforms": self.world_transforms.copy(),
            "rigid_body_state": self.rigid_body_state.copy(),
            "material_data": self.material_data.copy(),
            "scales": self.scales.copy(),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restores memory tables instantaneously from a PIE snapshot."""
        self.pool.active_count = snapshot["active_count"]
        self.pool.free_count = snapshot["free_count"]
        self.pool.sparse[:] = snapshot["sparse"]
        self.pool.dense[:] = snapshot["dense"]
        self.pool.free_list[:] = snapshot["free_list"]
        self.world_transforms[:] = snapshot["world_transforms"]
        self.rigid_body_state[:] = snapshot["rigid_body_state"]
        self.material_data[:] = snapshot["material_data"]
        self.scales[:] = snapshot["scales"]
