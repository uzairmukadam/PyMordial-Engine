"""Third-person follow and orbit camera system for character control.

Re-exports FollowCamera and FollowCameraConfig from engine.camera.
"""

from __future__ import annotations
from engine.camera.follow_camera import FollowCamera, FollowCameraConfig

# Aliases for backwards compatibility
CharacterCamera = FollowCamera
CharacterCameraConfig = FollowCameraConfig

__all__ = [
    "CharacterCamera",
    "CharacterCameraConfig",
    "FollowCamera",
    "FollowCameraConfig",
]
