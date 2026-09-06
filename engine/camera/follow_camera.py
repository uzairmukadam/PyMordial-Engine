"""Third-Person Collision-Aware Follow Camera for PyMordial Engine."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from engine.camera.camera import VirtualCamera
from engine.physics.rapier_world import PhysicsManager


@dataclass(slots=True)
class FollowCameraConfig:
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


class FollowCamera(VirtualCamera):
    """Calculates collision-aware 3rd-person camera positions tracking an actor."""

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
        config: FollowCameraConfig | None = None,
        initial_yaw_deg: float = 45.0,
        initial_pitch_deg: float = 20.0,
        fov: float = 75.0,
    ) -> None:
        super().__init__(fov=fov)
        self.config = config if config is not None else FollowCameraConfig()
        self.yaw_deg = initial_yaw_deg
        self.pitch_deg = initial_pitch_deg
        self.current_distance = self.config.distance
        self.target_pos = self.target
        self.camera_pos = self.position

    def handle_mouse_orbit(self, rel_x: float, rel_y: float) -> None:
        """Applies hardware relative mouse delta to camera orbit angles."""
        self.yaw_deg -= rel_x * self.config.yaw_sensitivity
        self.pitch_deg = max(
            self.config.min_pitch_deg,
            min(self.config.max_pitch_deg, self.pitch_deg - rel_y * self.config.pitch_sensitivity),
        )
        self.mark_dirty()

    def handle_zoom(self, wheel_delta: float) -> None:
        """Adjusts the ideal follow distance from mouse wheel input."""
        self.config.distance = max(
            self.config.min_distance,
            min(self.config.max_distance, self.config.distance - wheel_delta * 1.5),
        )
        self.mark_dirty()

    @property
    def is_clipped(self) -> bool:
        """Returns True if the camera is currently pulled in due to obstacle occlusion."""
        return self.current_distance < (self.config.distance - 0.05)

    def update_follow(
        self,
        dt: float,
        character_pos: tuple[float, float, float] | np.ndarray,
        physics: PhysicsManager | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Updates camera position following the target with obstacle clipping."""
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
        dir_x = math.sin(rad_yaw) * cos_pitch
        dir_y = math.sin(rad_pitch)
        dir_z = math.cos(rad_yaw) * cos_pitch

        desired_dist = cfg.distance
        hit_dist = desired_dist

        # Cast ray from target pivot toward the camera position
        if physics is not None:
            hit = physics.raycast(
                origin=(cx, cy, cz),
                direction=(dir_x, dir_y, dir_z),
                max_distance=cfg.distance,
                solid=True,
            )
            if hit is not None:
                _, hit_dist_val, _ = hit
                hit_dist = max(cfg.min_distance, hit_dist_val - cfg.camera_radius)

        if hit_dist < self.current_distance:
            # Snap in immediately to prevent clipping through geometry
            self.current_distance = hit_dist
        else:
            # Smoothly zoom back out when occlusion clears
            t = min(1.0, cfg.zoom_speed * dt)
            self.current_distance += (hit_dist - self.current_distance) * t

        self.camera_pos[0] = cx + dir_x * self.current_distance
        self.camera_pos[1] = cy + dir_y * self.current_distance
        self.camera_pos[2] = cz + dir_z * self.current_distance

        self.mark_dirty()

        cam_tuple = (float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))
        tgt_tuple = (float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2]))
        return (cam_tuple, tgt_tuple)

    def update(
        self,
        dt: float,
        character_pos: tuple[float, float, float] | np.ndarray | None = None,
        physics: PhysicsManager | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Compatible with original CharacterCamera signature."""
        if character_pos is not None:
            return self.update_follow(dt, character_pos, physics)
        return None
