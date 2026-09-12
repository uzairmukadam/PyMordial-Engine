"""Abstract Virtual Camera Base for PyMordial Engine."""

from __future__ import annotations
import math
import numpy as np

from engine.core.math_utils import matrix_look_at, matrix_perspective


class VirtualCamera:
    """Base virtual camera providing view/projection matrices and orientation vectors."""

    __slots__ = (
        "position",
        "target",
        "up",
        "fov",
        "near",
        "far",
        "aspect_ratio",
        "_view_matrix",
        "_proj_matrix",
        "_dirty_view",
        "_dirty_proj",
    )

    def __init__(
        self,
        position: tuple[float, float, float] | np.ndarray = (0.0, 2.0, 5.0),
        target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        up: tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
        fov: float = 75.0,
        near: float = 0.1,
        far: float = 500.0,
        aspect_ratio: float = 16.0 / 9.0,
    ) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect_ratio = aspect_ratio

        self._view_matrix = np.eye(4, dtype=np.float32)
        self._proj_matrix = np.eye(4, dtype=np.float32)
        self._dirty_view = True
        self._dirty_proj = True

    def mark_dirty(self) -> None:
        """Flags cached view and projection matrices as needing recomputation."""
        self._dirty_view = True
        self._dirty_proj = True

    def get_view_matrix(self) -> np.ndarray:
        """Computes and caches the standard 4x4 View matrix."""
        if self._dirty_view:
            self._view_matrix = matrix_look_at(self.position, self.target, self.up)
            self._dirty_view = False
        return self._view_matrix

    def get_projection_matrix(self, reverse_z: bool = True) -> np.ndarray:
        """Computes and caches the standard 4x4 Projection matrix (Reversed-Z support)."""
        if self._dirty_proj:
            self._proj_matrix = matrix_perspective(
                math.radians(self.fov),
                self.aspect_ratio,
                self.near,
                self.far,
                reverse_z=reverse_z,
            )
            self._dirty_proj = False
        return self._proj_matrix

    def get_forward_vector(self) -> np.ndarray:
        """Returns the normalized forward look direction vector."""
        fwd = self.target - self.position
        length = float(np.linalg.norm(fwd))
        if length > 1e-6:
            return fwd / length
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def get_right_vector(self) -> np.ndarray:
        """Returns the normalized camera right vector."""
        fwd = self.get_forward_vector()
        right = np.cross(fwd, self.up)
        length = float(np.linalg.norm(right))
        if length > 1e-6:
            return right / length
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def update(self, dt: float) -> None:
        """Per-frame update hook for camera controllers."""
        pass
