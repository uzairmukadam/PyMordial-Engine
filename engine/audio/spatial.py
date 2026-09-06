"""3D Spatial Audio Calculations and Listener Tracking for PyMordial Engine."""

from __future__ import annotations
from enum import IntEnum
import math
import numpy as np


class AttenuationModel(IntEnum):
    """Distance attenuation curve models."""
    INVERSE_DISTANCE = 0
    LINEAR = 1
    EXPONENTIAL = 2


class AudioListener:
    """Represents the 3D microphone/ears tracking the camera in world space."""

    __slots__ = ("position", "forward", "up", "velocity")

    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def sync_with_camera(self, camera_position: tuple[float, float, float] | np.ndarray, forward_vec: tuple[float, float, float] | np.ndarray) -> None:
        """Aligns listener position and orientation with the active 3D camera."""
        self.position[0] = float(camera_position[0])
        self.position[1] = float(camera_position[1])
        self.position[2] = float(camera_position[2])

        fwd = np.array(forward_vec, dtype=np.float32)
        norm = np.linalg.norm(fwd)
        if norm > 1e-6:
            self.forward = fwd / norm


def calculate_spatial_pan_and_attenuation(
    listener: AudioListener,
    source_position: tuple[float, float, float] | np.ndarray,
    min_distance: float = 1.0,
    max_distance: float = 50.0,
    attenuation_model: AttenuationModel = AttenuationModel.INVERSE_DISTANCE,
    rolloff_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Computes distance attenuation and stereo left/right panning factors.
    
    Returns:
        (left_volume, right_volume, attenuation_factor)
    """
    sx = float(source_position[0])
    sy = float(source_position[1])
    sz = float(source_position[2])

    dx = sx - listener.position[0]
    dy = sy - listener.position[1]
    dz = sz - listener.position[2]

    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    # 1. Attenuation factor calculation
    min_d = max(0.01, min_distance)
    max_d = max(min_d + 0.1, max_distance)

    if distance <= min_d:
        attenuation = 1.0
    elif distance >= max_d:
        attenuation = 0.0
    else:
        if attenuation_model == AttenuationModel.LINEAR:
            attenuation = 1.0 - (distance - min_d) / (max_d - min_d)
        elif attenuation_model == AttenuationModel.EXPONENTIAL:
            attenuation = math.exp(-rolloff_factor * (distance - min_d) / (max_d - min_d))
        else:  # INVERSE_DISTANCE
            attenuation = min_d / (min_d + rolloff_factor * (distance - min_d))

    attenuation = max(0.0, min(1.0, attenuation))
    if attenuation <= 0.0 or distance < 1e-6:
        return (attenuation, attenuation, attenuation)

    # 2. Stereo Panning calculation (-1.0 = hard left, +1.0 = hard right)
    # Right vector = Forward x Up
    right_vec = np.cross(listener.forward, listener.up)
    right_norm = np.linalg.norm(right_vec)
    if right_norm > 1e-6:
        right_vec /= right_norm
    else:
        right_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Normalized direction to sound source
    inv_dist = 1.0 / distance
    dir_to_src = np.array([dx * inv_dist, dy * inv_dist, dz * inv_dist], dtype=np.float32)

    # Dot product with listener right vector gives pan factor in [-1, +1]
    pan = float(np.dot(dir_to_src, right_vec))
    pan = max(-1.0, min(1.0, pan))

    # Convert pan to left/right volume balance
    # Equal-power or linear stereo panning
    left_vol = attenuation * max(0.0, min(1.0, 0.5 * (1.0 - pan)))
    right_vol = attenuation * max(0.0, min(1.0, 0.5 * (1.0 + pan)))

    return (left_vol, right_vol, attenuation)
