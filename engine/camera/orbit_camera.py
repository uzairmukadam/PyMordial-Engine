"""Turntable Orbit Inspection Camera for PyMordial Engine."""

from __future__ import annotations
import math
import numpy as np

from engine.camera.camera import VirtualCamera


class OrbitCamera(VirtualCamera):
    """Turntable camera orbiting a stationary or moving focus pivot."""

    __slots__ = (
        "radius",
        "azimuth_deg",
        "elevation_deg",
        "min_radius",
        "max_radius",
    )

    def __init__(
        self,
        pivot: tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
        radius: float = 8.0,
        azimuth_deg: float = 45.0,
        elevation_deg: float = 30.0,
        fov: float = 75.0,
    ) -> None:
        super().__init__(target=pivot, fov=fov)
        self.radius = radius
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self.min_radius = 0.5
        self.max_radius = 100.0
        self._update_position()

    def handle_orbit(self, rel_x: float, rel_y: float, sensitivity: float = 0.25) -> None:
        """Rotates around the pivot point."""
        self.azimuth_deg -= rel_x * sensitivity
        self.elevation_deg = max(-85.0, min(85.0, self.elevation_deg - rel_y * sensitivity))
        self._update_position()

    def handle_zoom(self, delta: float) -> None:
        """Adjusts distance to pivot."""
        self.radius = max(self.min_radius, min(self.max_radius, self.radius - delta * 1.5))
        self._update_position()

    def _update_position(self) -> None:
        """Computes camera Cartesian position from spherical orbit coordinates."""
        az_rad = math.radians(self.azimuth_deg)
        el_rad = math.radians(self.elevation_deg)

        cos_el = math.cos(el_rad)
        px = self.target[0] + self.radius * math.sin(az_rad) * cos_el
        py = self.target[1] + self.radius * math.sin(el_rad)
        pz = self.target[2] + self.radius * math.cos(az_rad) * cos_el

        self.position[0] = px
        self.position[1] = py
        self.position[2] = pz
        self.mark_dirty()
