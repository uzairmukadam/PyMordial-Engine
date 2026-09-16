"""Clustered Dynamic Local Lighting Pass for PyMordial Engine.

Manages up to 256 dynamic point lights and up to 64 dynamic spot lights with physical inverse-square
attenuation and smooth cone falloff, packed into GPU SSBO binding 3 for branch-free evaluation during deferred resolve.
Also manages shadow-casting slot assignments (up to 4 active perspective shadow casters).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
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


@dataclass
class SpotLight:
    """Dynamic local spot light definition."""
    x: float
    y: float
    z: float
    radius: float
    dir_x: float
    dir_y: float
    dir_z: float
    inner_cone_angle: float   # In radians
    r: float
    g: float
    b: float
    intensity: float
    outer_cone_angle: float   # In radians
    cast_shadow: bool = False
    shadow_bias: float = 0.0015
    shadow_index: int = -1    # Assigned dynamically during pass execution (0..3 or -1)


class ClusteredLightingPass(RenderPass):
    """Manages active point lights and spot lights and uploads them to GPU SSBO 3."""

    POINT_LIGHT_MAX = 256
    POINT_STRIDE_FLOATS = 8     # 2 x vec4 = 32 bytes
    POINT_STRIDE_BYTES = 32
    SPOT_OFFSET_BYTES = 256 * 32  # 8192 bytes
    SPOT_STRIDE_FLOATS = 16    # 4 x vec4 = 64 bytes
    SPOT_STRIDE_BYTES = 64

    def __init__(self, ctx: moderngl.Context, max_lights: int = 256, max_spot_lights: int = 64) -> None:
        super().__init__(name="ClusteredLightingPass", enabled=True)
        self.ctx = ctx
        self.max_lights = max_lights
        self.max_spot_lights = max_spot_lights
        self.lights: list[PointLight] = []
        self.spot_lights: list[SpotLight] = []

        # Point Light CPU buffer: 256 x 8 floats = 8192 bytes
        # Layout:
        # float x, y, z, radius (vec4 pos_radius)
        # float r, g, b, intensity (vec4 color_intensity)
        self._cpu_point_buffer = np.zeros((self.POINT_LIGHT_MAX, self.POINT_STRIDE_FLOATS), dtype=np.float32)

        # Spot Light CPU buffer: max_spot_lights x 16 floats
        # Layout:
        # float x, y, z, radius (vec4 pos_radius)
        # float dir_x, dir_y, dir_z, inner_cos (vec4 dir_inner_cos)
        # float r, g, b, intensity (vec4 color_intensity)
        # float outer_cos, shadow_index, shadow_bias, 0.0 (vec4 params)
        self._cpu_spot_buffer = np.zeros((self.max_spot_lights, self.SPOT_STRIDE_FLOATS), dtype=np.float32)

        # Total SSBO size = Point lights (8192 B) + Spot lights (max_spot_lights * 64 B)
        total_bytes = self.SPOT_OFFSET_BYTES + self.max_spot_lights * self.SPOT_STRIDE_BYTES
        self.ssbo_lights = self.ctx.buffer(reserve=total_bytes)
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

    def add_spot_light(
        self,
        position: tuple[float, float, float],
        direction: tuple[float, float, float],
        radius: float = 25.0,
        inner_cone_angle: float = math.radians(18.0),
        outer_cone_angle: float = math.radians(32.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 4.5,
        cast_shadow: bool = False,
        shadow_bias: float = 0.0015,
    ) -> SpotLight:
        # Normalize direction
        dx, dy, dz = direction[0], direction[1], direction[2]
        inv_len = 1.0 / (math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6)
        dx, dy, dz = dx * inv_len, dy * inv_len, dz * inv_len

        spot = SpotLight(
            x=position[0],
            y=position[1],
            z=position[2],
            radius=radius,
            dir_x=dx,
            dir_y=dy,
            dir_z=dz,
            inner_cone_angle=inner_cone_angle,
            outer_cone_angle=outer_cone_angle,
            r=color[0],
            g=color[1],
            b=color[2],
            intensity=intensity,
            cast_shadow=cast_shadow,
            shadow_bias=shadow_bias,
            shadow_index=-1,
        )
        if len(self.spot_lights) < self.max_spot_lights:
            self.spot_lights.append(spot)
        return spot

    def clear(self) -> None:
        self.lights.clear()
        self.spot_lights.clear()

    def clear_spot_lights(self) -> None:
        self.spot_lights.clear()

    def clear_point_lights(self) -> None:
        self.lights.clear()

    def execute(self, context: RenderGraphContext) -> None:
        point_count = min(len(self.lights), self.max_lights)
        spot_count = min(len(self.spot_lights), self.max_spot_lights)

        context.resources["point_light_count"] = point_count
        context.resources["spot_light_count"] = spot_count
        context.resources["point_light_buffer"] = self.ssbo_lights

        # 1. Upload Point Lights
        if point_count > 0:
            for i in range(point_count):
                lt = self.lights[i]
                self._cpu_point_buffer[i, 0] = lt.x
                self._cpu_point_buffer[i, 1] = lt.y
                self._cpu_point_buffer[i, 2] = lt.z
                self._cpu_point_buffer[i, 3] = lt.radius
                self._cpu_point_buffer[i, 4] = lt.r
                self._cpu_point_buffer[i, 5] = lt.g
                self._cpu_point_buffer[i, 6] = lt.b
                self._cpu_point_buffer[i, 7] = lt.intensity

            self.ssbo_lights.write(self._cpu_point_buffer[:point_count].tobytes(), offset=0)

        # 2. Process and Upload Spot Lights
        shadow_spots: list[SpotLight] = []
        if spot_count > 0:
            shadow_slot = 0
            for i in range(spot_count):
                sp = self.spot_lights[i]
                # Assign shadow index to first 4 shadow casters
                if sp.cast_shadow and shadow_slot < 4:
                    sp.shadow_index = shadow_slot
                    shadow_spots.append(sp)
                    shadow_slot += 1
                else:
                    sp.shadow_index = -1

                # vec4 pos_radius
                self._cpu_spot_buffer[i, 0] = sp.x
                self._cpu_spot_buffer[i, 1] = sp.y
                self._cpu_spot_buffer[i, 2] = sp.z
                self._cpu_spot_buffer[i, 3] = sp.radius

                # vec4 dir_inner_cos
                self._cpu_spot_buffer[i, 4] = sp.dir_x
                self._cpu_spot_buffer[i, 5] = sp.dir_y
                self._cpu_spot_buffer[i, 6] = sp.dir_z
                self._cpu_spot_buffer[i, 7] = math.cos(sp.inner_cone_angle)

                # vec4 color_intensity
                self._cpu_spot_buffer[i, 8] = sp.r
                self._cpu_spot_buffer[i, 9] = sp.g
                self._cpu_spot_buffer[i, 10] = sp.b
                self._cpu_spot_buffer[i, 11] = sp.intensity

                # vec4 params: outer_cos, shadow_index, shadow_bias, reserved
                self._cpu_spot_buffer[i, 12] = math.cos(sp.outer_cone_angle)
                self._cpu_spot_buffer[i, 13] = float(sp.shadow_index)
                self._cpu_spot_buffer[i, 14] = float(sp.shadow_bias)
                self._cpu_spot_buffer[i, 15] = 0.0

            self.ssbo_lights.write(
                self._cpu_spot_buffer[:spot_count].tobytes(),
                offset=self.SPOT_OFFSET_BYTES,
            )

        context.resources["shadow_spot_lights"] = shadow_spots
        self.ssbo_lights.bind_to_storage_buffer(binding=3)

    def destroy(self) -> None:
        self.ssbo_lights.release()
