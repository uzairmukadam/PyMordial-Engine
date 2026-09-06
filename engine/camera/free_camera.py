"""6-DOF Free-Fly Debug and Viewport Camera for PyMordial Engine."""

from __future__ import annotations
import math
import numpy as np

from engine.camera.camera import VirtualCamera


class FreeFlyCamera(VirtualCamera):
    """6-DOF free-flying camera with WASD movement, mouse look, and speed boost."""

    __slots__ = (
        "yaw_deg",
        "pitch_deg",
        "move_speed",
        "boost_multiplier",
        "look_sensitivity",
    )

    def __init__(
        self,
        position: tuple[float, float, float] | np.ndarray = (0.0, 5.0, 10.0),
        yaw_deg: float = -90.0,
        pitch_deg: float = -15.0,
        move_speed: float = 10.0,
        look_sensitivity: float = 0.20,
        fov: float = 75.0,
    ) -> None:
        super().__init__(position=position, fov=fov)
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.move_speed = move_speed
        self.boost_multiplier = 3.0
        self.look_sensitivity = look_sensitivity
        self._update_vectors()

    def handle_mouse_look(self, rel_x: float, rel_y: float) -> None:
        """Rotates camera direction based on relative mouse motion."""
        self.yaw_deg += rel_x * self.look_sensitivity
        self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg - rel_y * self.look_sensitivity))
        self._update_vectors()

    def _update_vectors(self) -> None:
        """Recalculates target and forward vectors from spherical angles."""
        yaw_rad = math.radians(self.yaw_deg)
        pitch_rad = math.radians(self.pitch_deg)

        cos_p = math.cos(pitch_rad)
        fx = math.cos(yaw_rad) * cos_p
        fy = math.sin(pitch_rad)
        fz = math.sin(yaw_rad) * cos_p

        fwd = np.array([fx, fy, fz], dtype=np.float32)
        fwd /= np.linalg.norm(fwd)

        self.target = self.position + fwd
        self.mark_dirty()

    def move(
        self,
        dt: float,
        forward_axis: float = 0.0,
        right_axis: float = 0.0,
        up_axis: float = 0.0,
        boost: bool = False,
    ) -> None:
        """Translates the camera in view space."""
        speed = self.move_speed * (self.boost_multiplier if boost else 1.0)
        fwd = self.get_forward_vector()
        right = self.get_right_vector()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        velocity = (fwd * forward_axis + right * right_axis + up * up_axis) * (speed * dt)
        self.position += velocity
        self.target += velocity
        self.mark_dirty()
