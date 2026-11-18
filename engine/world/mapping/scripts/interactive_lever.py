"""Interactive Lever Script.

Smoothly rotates a kinematic lever handle in real time when interacted with,
signaling connected targets (e.g. sliding gates, lights, alarms).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from engine.world.mapping.script_runner import EntityScript, register_script_type
from engine.world.mapping.transform_math import quat_multiply

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


@register_script_type("InteractiveLeverScript")
@register_script_type("LeverScript")
class InteractiveLeverScript(EntityScript):
    """Rotates a lever handle in real time and signals linked target entities."""

    __slots__ = (
        "target_id",
        "max_angle_rad",
        "rotation_speed",
        "is_pulled",
        "current_angle",
        "target_angle",
        "_origin_pos",
        "_origin_rot",
        "_axis",
    )

    def on_spawn(self, node: MapEntityNode, app: ProjectApp, params: dict[str, Any]) -> None:
        super().on_spawn(node, app, params)
        self.target_id = str(params.get("target_id") or params.get("target_gate_id", ""))
        angle_deg = float(params.get("pull_angle_deg") or params.get("angle_deg", 65.0))
        self.max_angle_rad = math.radians(angle_deg)
        self.rotation_speed = float(params.get("rotation_speed") or params.get("speed", 4.0))

        self.is_pulled = bool(params.get("initial_state", False))
        self.current_angle = self.max_angle_rad if self.is_pulled else 0.0
        self.target_angle = self.current_angle

        self._origin_pos = tuple(node.world_transform.position)
        self._origin_rot = tuple(node.world_transform.rotation)
        # Default rotation around X axis (pitch forward)
        self._axis = params.get("axis", "X").upper()

    def on_interact(self, interactor_id: int | None = None) -> None:
        """Toggles the lever state and signals the target entity."""
        self.is_pulled = not self.is_pulled
        self.target_angle = self.max_angle_rad if self.is_pulled else 0.0

        if self.app is not None and hasattr(self.app, "map_manager"):
            signal = "OPEN" if self.is_pulled else "CLOSE"
            self.app.map_manager.signal_entity(self.target_id, signal)

    def on_fixed_update(self, dt: float) -> None:
        if self.node is None or self.app is None or self.node.ecs_id is None:
            return

        # Slerp angle toward target
        if abs(self.current_angle - self.target_angle) > 1e-4:
            step = self.rotation_speed * dt
            if self.current_angle < self.target_angle:
                self.current_angle = min(self.target_angle, self.current_angle + step)
            else:
                self.current_angle = max(self.target_angle, self.current_angle - step)

            # Construct rotation quaternion around axis
            half = self.current_angle * 0.5
            s = math.sin(half)
            c = math.cos(half)
            if self._axis == "Z":
                rot_delta = (0.0, 0.0, s, c)
            elif self._axis == "Y":
                rot_delta = (0.0, s, 0.0, c)
            else:
                rot_delta = (s, 0.0, 0.0, c)

            new_rot = quat_multiply(self._origin_rot, rot_delta)

            # Update Rapier physics
            if self.node.has_rigid_body:
                self.app.physics.set_transform(self.node.ecs_id, self._origin_pos, new_rot)

            # Update ECS transform
            dense_idx = self.app.ecs.pool.get_dense_index(self.node.ecs_id)
            if dense_idx >= 0:
                self.app.ecs.rigid_body_state[1, dense_idx, 3:7] = new_rot
                self.app.ecs.recompute_matrix(dense_idx)
