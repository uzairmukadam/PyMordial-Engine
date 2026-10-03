"""Camera Stack and Blending Manager for PyMordial Engine."""

from __future__ import annotations

from engine.camera.camera import VirtualCamera
from engine.logging import log_info, LogChannel


def _ease_in_out(t: float) -> float:
    """Hermite smoothstep S-curve (3t^2 - 2t^3)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class CameraManager:
    """Coordinates active cameras and provides smooth transition interpolation."""

    __slots__ = (
        "_cameras",
        "active_camera_name",
        "_blend_source",
        "_blend_target",
        "_blend_duration",
        "_blend_time",
        "_is_blending",
        "_blended_cam",
    )

    def __init__(self) -> None:
        self._cameras: dict[str, VirtualCamera] = {}
        self.active_camera_name: str = ""
        self._blend_source: VirtualCamera | None = None
        self._blend_target: VirtualCamera | None = None
        self._blend_duration: float = 0.0
        self._blend_time: float = 0.0
        self._is_blending: bool = False
        self._blended_cam = VirtualCamera()

    def register_camera(self, name: str, camera: VirtualCamera, make_active: bool = False) -> None:
        """Registers a named virtual camera."""
        key = name.lower()
        self._cameras[key] = camera
        if make_active or not self.active_camera_name:
            self.active_camera_name = key
        log_info(LogChannel.CAMERA, f"Registered camera '{name}'")

    def get_camera(self, name: str) -> VirtualCamera | None:
        return self._cameras.get(name.lower(), None)

    @property
    def active_camera(self) -> VirtualCamera:
        """Returns the currently active or blended camera."""
        if self._is_blending:
            return self._blended_cam
        return self._cameras.get(self.active_camera_name, self._blended_cam)

    def switch_to(self, name: str) -> bool:
        """Instantly cuts to a registered camera."""
        key = name.lower()
        if key in self._cameras:
            self._is_blending = False
            self.active_camera_name = key
            log_info(LogChannel.CAMERA, f"Cut to camera '{name}'")
            return True
        return False

    set_active_camera = switch_to

    def blend_to(self, name: str, duration_seconds: float = 1.0) -> bool:
        """Initiates a smooth interpolated transition to the target camera."""
        key = name.lower()
        target = self._cameras.get(key)
        source = self.active_camera

        if target is None or target is source or duration_seconds <= 0.0:
            return self.switch_to(name)

        self._blend_source = source
        self._blend_target = target
        self._blend_duration = duration_seconds
        self._blend_time = 0.0
        self._is_blending = True
        self.active_camera_name = key
        log_info(LogChannel.CAMERA, f"Blending to camera '{name}' over {duration_seconds:.2f}s")
        return True

    def update(self, dt: float) -> None:
        """Advances camera transitions and updates active camera."""
        # Update underlying camera controller
        curr = self._cameras.get(self.active_camera_name)
        if curr is not None:
            curr.update(dt)

        if not self._is_blending or self._blend_source is None or self._blend_target is None:
            return

        self._blend_time += dt
        t = min(1.0, self._blend_time / self._blend_duration)
        alpha = _ease_in_out(t)

        # Interpolate position, target, up, and FOV
        src = self._blend_source
        dst = self._blend_target

        self._blended_cam.position = src.position + (dst.position - src.position) * alpha
        self._blended_cam.target = src.target + (dst.target - src.target) * alpha
        self._blended_cam.up = src.up + (dst.up - src.up) * alpha
        self._blended_cam.fov = src.fov + (dst.fov - src.fov) * alpha
        self._blended_cam.aspect_ratio = dst.aspect_ratio
        self._blended_cam.near = dst.near
        self._blended_cam.far = dst.far
        self._blended_cam.mark_dirty()

        if t >= 1.0:
            self._is_blending = False
            self._blend_source = None
            self._blend_target = None
