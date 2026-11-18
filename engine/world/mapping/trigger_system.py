"""Trigger Zone Subsystem and Spatial Overlap Evaluator.

Provides fast, zero-allocation spatial broadphase and event dispatch for invisible
trigger volumes, sensor zones, and stream portals.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from engine.world.mapping.schema import TriggerActionDef, TriggerDef

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


class TriggerZone:
    """Active spatial trigger volume instance."""

    __slots__ = (
        "node",
        "trigger_def",
        "center",
        "half_extents",
        "is_sphere",
        "one_shot",
        "has_fired",
        "was_inside",
    )

    def __init__(self, node: MapEntityNode, trigger_def: TriggerDef) -> None:
        self.node = node
        self.trigger_def = trigger_def
        self.center = tuple(node.world_transform.position)

        dims = trigger_def.dimensions
        if trigger_def.shape == "sphere":
            self.is_sphere = True
            r = dims[0] if len(dims) > 0 else 1.0
            self.half_extents = (r, r, r)
        else:
            self.is_sphere = False
            hx = dims[0] if len(dims) > 0 else 1.0
            hy = dims[1] if len(dims) > 1 else hx
            hz = dims[2] if len(dims) > 2 else hx
            self.half_extents = (hx, hy, hz)

        self.one_shot = trigger_def.one_shot
        self.has_fired = False
        self.was_inside = False

    def update_center(self) -> None:
        """Syncs trigger center with parent node position if dynamic."""
        self.center = tuple(self.node.world_transform.position)

    def test_overlap(self, point: tuple[float, float, float]) -> bool:
        """Vectorized spatial overlap test with zero allocations."""
        px, py, pz = point
        cx, cy, cz = self.center

        if self.is_sphere:
            r = self.half_extents[0]
            dx = px - cx
            dy = py - cy
            dz = pz - cz
            return (dx * dx + dy * dy + dz * dz) <= (r * r)

        hx, hy, hz = self.half_extents
        return (
            (cx - hx) <= px <= (cx + hx)
            and (cy - hy) <= py <= (cy + hy)
            and (cz - hz) <= pz <= (cz + hz)
        )


class TriggerZoneSystem:
    """Monitors active trigger zones and dispatches enter/exit actions."""

    __slots__ = ("_triggers", "_app")

    def __init__(self, app: ProjectApp | None = None) -> None:
        self._triggers: list[TriggerZone] = []
        self._app = app

    def attach_app(self, app: ProjectApp) -> None:
        self._app = app

    def register_trigger(self, node: MapEntityNode, trigger_def: TriggerDef) -> TriggerZone:
        zone = TriggerZone(node, trigger_def)
        self._triggers.append(zone)
        return zone

    def unregister_node_triggers(self, node: MapEntityNode) -> None:
        self._triggers = [t for t in self._triggers if t.node != node]

    def clear(self) -> None:
        self._triggers.clear()

    def on_fixed_update(self, observer_pos: tuple[float, float, float]) -> None:
        """Evaluates all registered trigger zones against observer position with zero allocations."""
        if self._app is None:
            return

        for zone in self._triggers:
            if zone.one_shot and zone.has_fired:
                continue

            is_inside = zone.test_overlap(observer_pos)

            # Trigger entered
            if is_inside and not zone.was_inside:
                zone.was_inside = True
                zone.has_fired = True
                self._dispatch_actions(zone.trigger_def.on_enter, zone)

            # Trigger exited
            elif not is_inside and zone.was_inside:
                zone.was_inside = False
                self._dispatch_actions(zone.trigger_def.on_exit, zone)

    def _dispatch_actions(self, actions: list[TriggerActionDef], zone: TriggerZone) -> None:
        if not self._app:
            return

        for act in actions:
            action_type = act.action_type.upper()

            if action_type in ("STREAM_SECTOR", "STREAM"):
                target = act.target or act.params.get("target")
                if target and hasattr(self._app, "map_manager"):
                    mode = act.params.get("mode", "ACTIVATE")
                    self._app.map_manager.streaming_director.request_stream(str(target), mode=mode)

            elif action_type in ("SIGNAL_ENTITY", "SIGNAL"):
                target = act.target or act.params.get("target")
                signal = act.params.get("signal", "TOGGLE")
                if target and hasattr(self._app, "map_manager"):
                    self._app.map_manager.signal_entity(str(target), signal)

            elif action_type in ("PLAY_SOUND", "AUDIO"):
                asset = act.target or act.params.get("asset")
                if asset and hasattr(self._app, "audio") and self._app.audio is not None:
                    try:
                        self._app.audio.play_sound(str(asset))
                    except Exception:
                        pass

            elif action_type in ("CUSTOM_EVENT", "EVENT"):
                event_name = act.target or act.params.get("event", "TRIGGER_EVENT")
                if hasattr(self._app, "debug_toast") and self._app.debug_toast is not None:
                    self._app.debug_toast.show(f"Event: {event_name}")
