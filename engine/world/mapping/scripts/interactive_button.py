"""Interactive push-button script for map entities.

Animates a kinematic button, sends a configurable signal to a target entity,
and optionally latches after the first interaction.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from engine.world.mapping.script_runner import EntityScript, register_script_type

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


@register_script_type("InteractiveButtonScript")
@register_script_type("ButtonScript")
class InteractiveButtonScript(EntityScript):
    __slots__ = (
        "target_id", "signal", "one_shot", "speed", "press_axis",
        "press_distance", "pressed", "progress", "target_progress",
        "_origin_pos", "_rot",
    )

    def on_spawn(self, node: MapEntityNode, app: ProjectApp, params: dict[str, Any]) -> None:
        super().on_spawn(node, app, params)
        self.target_id = str(params.get("target_id", ""))
        self.signal = str(params.get("signal", "OPEN")).upper()
        self.one_shot = bool(params.get("one_shot", True))
        self.speed = max(0.01, float(params.get("speed", 6.0)))
        raw_axis = params.get("press_axis", (0.0, -1.0, 0.0))
        axis = (float(raw_axis[0]), float(raw_axis[1]), float(raw_axis[2]))
        length = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
        self.press_axis = (0.0, -1.0, 0.0) if length <= 1e-6 else tuple(v / length for v in axis)
        self.press_distance = max(0.0, float(params.get("press_distance", 0.16)))
        self.pressed = bool(params.get("initial_state", False))
        self.progress = 1.0 if self.pressed else 0.0
        self.target_progress = self.progress
        self._origin_pos = tuple(node.world_transform.position)
        self._rot = tuple(node.world_transform.rotation)

    def on_interact(self, interactor_id: int | None = None) -> None:
        if self.one_shot and self.pressed:
            return
        self.pressed = not self.pressed if not self.one_shot else True
        self.target_progress = 1.0 if self.pressed else 0.0
        if self.app is not None and hasattr(self.app, "map_manager") and self.target_id:
            outgoing = self.signal if self.pressed else "CLOSE"
            self.app.map_manager.signal_entity(self.target_id, outgoing)

    def on_fixed_update(self, dt: float) -> None:
        if self.node is None or self.app is None or self.node.ecs_id is None:
            return
        if abs(self.progress - self.target_progress) <= 1e-4:
            return
        step = self.speed * dt
        if self.progress < self.target_progress:
            self.progress = min(self.target_progress, self.progress + step)
        else:
            self.progress = max(self.target_progress, self.progress - step)
        t = self.progress
        eased = t * t * (3.0 - 2.0 * t)
        new_pos = tuple(self._origin_pos[i] + self.press_axis[i] * self.press_distance * eased for i in range(3))
        if self.node.has_rigid_body:
            self.app.physics.set_transform(self.node.ecs_id, new_pos, self._rot)
        dense_idx = self.app.ecs.pool.get_dense_index(self.node.ecs_id)
        if dense_idx >= 0:
            self.app.ecs.rigid_body_state[1, dense_idx, 0:3] = new_pos
            self.app.ecs.recompute_matrix(dense_idx)
