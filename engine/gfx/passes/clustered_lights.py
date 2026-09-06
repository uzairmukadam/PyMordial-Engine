"""Clustered Dynamic Local Lighting Pass for PyMordial Engine.

Manages up to 256 dynamic point lights with physical inverse-square attenuation,
packed into GPU SSBO binding 3 for branch-free evaluation during deferred resolve.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext


@dataclass
class PointLight:
    """Dynamic local point light definition."""
    x: float
    y: float
    z: float
    radius: float
    r: float
    g: float
    b: float
    intensity: float


class ClusteredLightingPass(RenderPass):
    """Manages active point lights and uploads them to GPU SSBO 3."""

    def __init__(self, ctx: moderngl.Context, max_lights: int = 256) -> None:
        super().__init__(name="ClusteredLightingPass", enabled=True)
        self.ctx = ctx
        self.max_lights = max_lights
        self.lights: list[PointLight] = []

        # Each light is 2 x vec4 = 8 floats = 32 bytes
        # Layout:
        # float x, y, z, radius
        # float r, g, b, intensity
        self._cpu_buffer = np.zeros((max_lights, 8), dtype=np.float32)
        self.ssbo_lights = self.ctx.buffer(reserve=max_lights * 8 * 4)
        self.ssbo_lights.bind_to_storage_buffer(binding=3)

    def add_light(
        self,
        position: tuple[float, float, float],
        radius: float = 10.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 2.0,
    ) -> PointLight:
        light = PointLight(
            x=position[0],
            y=position[1],
            z=position[2],
            radius=radius,
            r=color[0],
            g=color[1],
            b=color[2],
            intensity=intensity,
        )
        if len(self.lights) < self.max_lights:
            self.lights.append(light)
        return light

    def clear(self) -> None:
        self.lights.clear()

    def execute(self, context: RenderGraphContext) -> None:
        count = min(len(self.lights), self.max_lights)
        context.resources["point_light_count"] = count

        if count > 0:
            for i in range(count):
                lt = self.lights[i]
                self._cpu_buffer[i, 0] = lt.x
                self._cpu_buffer[i, 1] = lt.y
                self._cpu_buffer[i, 2] = lt.z
                self._cpu_buffer[i, 3] = lt.radius
                self._cpu_buffer[i, 4] = lt.r
                self._cpu_buffer[i, 5] = lt.g
                self._cpu_buffer[i, 6] = lt.b
                self._cpu_buffer[i, 7] = lt.intensity

            # Upload sub-buffer
            self.ssbo_lights.write(self._cpu_buffer[:count].tobytes())

        self.ssbo_lights.bind_to_storage_buffer(binding=3)

    def destroy(self) -> None:
        self.ssbo_lights.release()
