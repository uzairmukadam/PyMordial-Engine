"""Third-person follow and orbit camera system for character control.

Features:
- Smooth orbit controls with customizable sensitivity and pitch/yaw clamping
- Character eye-level pivot tracking
- Physical obstacle raycast clipping to prevent the camera from clipping into walls or ground
- Fast responsive pull-in with smooth zoom recovery
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from engine.physics.rapier_world import PhysicsManager


@dataclass(slots=True)
class CharacterCameraConfig:
    """Tuning parameters for the third-person follow camera."""

    distance: float = 6.5           # Default camera follow distance in meters
    min_distance: float = 1.0       # Minimum pull-in distance in meters
    max_distance: float = 20.0      # Maximum zoomed out distance in meters
    target_offset_y: float = 1.35   # Height offset above character origin in meters
    camera_radius: float = 0.20     # Margin from obstacles to avoid clipping in meters
    yaw_sensitivity: float = 0.25   # Mouse drag yaw sensitivity
    pitch_sensitivity: float = 0.25 # Mouse drag pitch sensitivity
    min_pitch_deg: float = -35.0    # Lowest downward pitch angle
    max_pitch_deg: float = 75.0     # Highest upward pitch angle
    zoom_speed: float = 12.0        # Smoothing rate when zooming out


class CharacterCamera:
    """Calculates collision-aware 3rd-person camera positions following a character."""

    __slots__ = (
        "config",
        "yaw_deg",
        "pitch_deg",
        "current_distance",
        "target_pos",
        "camera_pos",
    )

    def __init__(
        self,
        config: CharacterCameraConfig | None = None,
        initial_yaw_deg: float = 45.0,
        initial_pitch_deg: float = 20.0,
    ) -> None:
        self.config = config if config is not None else CharacterCameraConfig()
        self.yaw_deg = initial_yaw_deg
        self.pitch_deg = initial_pitch_deg
        self.current_distance = self.config.distance
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.camera_pos = np.zeros(3, dtype=np.float32)

    def handle_mouse_orbit(self, rel_x: float, rel_y: float) -> None:
        """Applies hardware relative mouse delta to camera orbit angles."""
        self.yaw_deg -= rel_x * self.config.yaw_sensitivity
        self.pitch_deg = max(
            self.config.min_pitch_deg,
            min(self.config.max_pitch_deg, self.pitch_deg - rel_y * self.config.pitch_sensitivity),
        )

    def handle_zoom(self, wheel_delta: float) -> None:
        """Adjusts the ideal follow distance from mouse wheel input."""
        self.config.distance = max(
            self.config.min_distance,
            min(self.config.max_distance, self.config.distance - wheel_delta * 1.5),
        )

    @property
    def is_clipped(self) -> bool:
        """Returns True if the camera is currently pulled in due to obstacle occlusion."""
        return self.current_distance < (self.config.distance - 0.05)

    def update(
        self,
        dt: float,
        character_pos: tuple[float, float, float] | np.ndarray,
        physics: PhysicsManager | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Updates camera position following the character with obstacle clipping.

        Returns:
            (camera_position, camera_target)
        """
        cfg = self.config
        cx = float(character_pos[0])
        cy = float(character_pos[1]) + cfg.target_offset_y
        cz = float(character_pos[2])

        self.target_pos[0] = cx
        self.target_pos[1] = cy
        self.target_pos[2] = cz

        # Convert spherical yaw/pitch coordinates to direction vector from target to camera
        rad_yaw = math.radians(self.yaw_deg)
        rad_pitch = math.radians(self.pitch_deg)

        cos_pitch = math.cos(rad_pitch)
        sin_pitch = math.sin(rad_pitch)
        cos_yaw = math.cos(rad_yaw)
        sin_yaw = math.sin(rad_yaw)

        dir_x = cos_pitch * sin_yaw
        dir_y = sin_pitch
        dir_z = cos_pitch * cos_yaw

        # Obstacle collision query
        target_dist = cfg.distance
        if physics is not None:
            hit = physics.raycast(
                origin=(cx, cy, cz),
                direction=(dir_x, dir_y, dir_z),
                max_distance=cfg.distance,
                solid=True,
            )
            if hit is not None:
                _, hit_dist, _ = hit
                target_dist = max(cfg.min_distance, hit_dist - cfg.camera_radius)

        # Responsive pull-in (instant), smooth pull-out (lerped)
        if target_dist < self.current_distance:
            self.current_distance = target_dist
        else:
            self.current_distance += (target_dist - self.current_distance) * min(1.0, cfg.zoom_speed * dt)

        eye_x = cx + dir_x * self.current_distance
        eye_y = cy + dir_y * self.current_distance
        eye_z = cz + dir_z * self.current_distance

        self.camera_pos[0] = eye_x
        self.camera_pos[1] = eye_y
        self.camera_pos[2] = eye_z

        return (eye_x, eye_y, eye_z), (cx, cy, cz)
