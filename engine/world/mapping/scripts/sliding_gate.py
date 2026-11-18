"""Sliding Mechanical Gate Script.

Responds to signals from levers, buttons, or trigger zones by translating
its kinematic_position collider and visual mesh upward or outward in real time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.world.mapping.script_runner import EntityScript, register_script_type

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


@register_script_type("SlidingGateScript")
@register_script_type("GateScript")
class SlidingGateScript(EntityScript):
    """Smoothly translates a mechanical gate when opened or closed."""

    __slots__ = (
        "open_offset",
        "speed",
        "is_open",
        "progress",
        "target_progress",
        "_origin_pos",
        "_rot",
    )

    def on_spawn(self, node: MapEntityNode, app: ProjectApp, params: dict[str, Any]) -> None:
        super().on_spawn(node, app, params)
        raw_offset = params.get("open_offset", (0.0, 6.0, 0.0))
        self.open_offset = (float(raw_offset[0]), float(raw_offset[1]), float(raw_offset[2]))
        self.speed = float(params.get("speed", 2.5))

        self.is_open = bool(params.get("initial_open", False))
        self.progress = 1.0 if self.is_open else 0.0
        self.target_progress = self.progress

        self._origin_pos = tuple(node.world_transform.position)
        self._rot = tuple(node.world_transform.rotation)

    def on_signal(self, signal: str, payload: Any = None) -> None:
        sig_upper = signal.upper()
        if sig_upper == "OPEN":
            self.is_open = True
            self.target_progress = 1.0
        elif sig_upper == "CLOSE":
            self.is_open = False
            self.target_progress = 0.0
        elif sig_upper == "TOGGLE":
            self.is_open = not self.is_open
            self.target_progress = 1.0 if self.is_open else 0.0

    def on_fixed_update(self, dt: float) -> None:
        if self.node is None or self.app is None or self.node.ecs_id is None:
            return

        # Advance progress toward target
        if abs(self.progress - self.target_progress) > 1e-4:
            step = self.speed * dt
            if self.progress < self.target_progress:
                self.progress = min(self.target_progress, self.progress + step)
            else:
                self.progress = max(self.target_progress, self.progress - step)

            # Smooth ease-in-out curve
            t = self.progress
            eased_t = t * t * (3.0 - 2.0 * t)

            new_pos = (
                self._origin_pos[0] + self.open_offset[0] * eased_t,
                self._origin_pos[1] + self.open_offset[1] * eased_t,
                self._origin_pos[2] + self.open_offset[2] * eased_t,
            )

            # Update Rapier physics body
            if self.node.has_rigid_body:
                self.app.physics.set_transform(self.node.ecs_id, new_pos, self._rot)

            # Update ECS transform
            dense_idx = self.app.ecs.pool.get_dense_index(self.node.ecs_id)
            if dense_idx >= 0:
                self.app.ecs.rigid_body_state[1, dense_idx, 0:3] = new_pos
                self.app.ecs.recompute_matrix(dense_idx)
