"""Oscillating and Waypoint Moving Platform Script.

Controls a kinematic_position rigid body smoothly carrying players and props
along an oscillation axis or between waypoints.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from engine.world.mapping.script_runner import EntityScript, register_script_type

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


@register_script_type("MovingPlatformScript")
@register_script_type("OscillatingPlatformScript")
class MovingPlatformScript(EntityScript):
    """Smoothly moves an entity along an axis using Rapier kinematic transforms."""

    __slots__ = (
        "axis",
        "distance",
        "speed",
        "pause_time",
        "_origin_pos",
        "_rot",
        "_phase",
        "_current_pos",
    )

    def on_spawn(self, node: MapEntityNode, app: ProjectApp, params: dict[str, Any]) -> None:
        super().on_spawn(node, app, params)
        raw_axis = params.get("axis", (0.0, 1.0, 0.0))
        self.axis = (float(raw_axis[0]), float(raw_axis[1]), float(raw_axis[2]))
        # Normalize axis
        length = math.sqrt(self.axis[0] ** 2 + self.axis[1] ** 2 + self.axis[2] ** 2)
        if length > 1e-6:
            self.axis = (self.axis[0] / length, self.axis[1] / length, self.axis[2] / length)
        else:
            self.axis = (0.0, 1.0, 0.0)

        self.distance = float(params.get("distance", 6.0))
        self.speed = float(params.get("speed", 1.8))
        self.pause_time = float(params.get("pause_time", 0.5))

        self._origin_pos = tuple(node.world_transform.position)
        self._rot = tuple(node.world_transform.rotation)
        self._phase = 0.0
        self._current_pos = [self._origin_pos[0], self._origin_pos[1], self._origin_pos[2]]

    def on_fixed_update(self, dt: float) -> None:
        if self.node is None or self.app is None or self.node.ecs_id is None:
            return

        # Advance harmonic oscillation: 0.5 * (1.0 - cos(phase))
        self._phase += self.speed * dt
        # Factor smoothly oscillates between 0.0 and 1.0
        factor = 0.5 * (1.0 - math.cos(self._phase))
        travel = factor * self.distance

        self._current_pos[0] = self._origin_pos[0] + self.axis[0] * travel
        self._current_pos[1] = self._origin_pos[1] + self.axis[1] * travel
        self._current_pos[2] = self._origin_pos[2] + self.axis[2] * travel

        pos_tuple = (self._current_pos[0], self._current_pos[1], self._current_pos[2])

        # 1. Update Rapier kinematic position (imparts velocity to carry character)
        if self.node.has_rigid_body:
            self.app.physics.set_transform(self.node.ecs_id, pos_tuple, self._rot)

        # 2. Update ECS Contiguous memory & matrix
        dense_idx = self.app.ecs.pool.get_dense_index(self.node.ecs_id)
        if dense_idx >= 0:
            self.app.ecs.rigid_body_state[1, dense_idx, 0:3] = pos_tuple
            self.app.ecs.recompute_matrix(dense_idx)
