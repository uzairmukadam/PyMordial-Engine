"""Entity Scripting Engine and Lifecycle Dispatcher.

Provides zero-allocation lifecycle hooks for dynamic entities:
on_spawn, on_fixed_update, on_interact, on_signal, on_despawn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.entity_node import MapEntityNode


class EntityScript:
    """Base class for behavioral scripts attached to map entities."""

    __slots__ = ("node", "app", "params", "is_active")

    def __init__(self) -> None:
        self.node: MapEntityNode | None = None
        self.app: ProjectApp | None = None
        self.params: dict[str, Any] = {}
        self.is_active: bool = False

    def on_spawn(self, node: MapEntityNode, app: ProjectApp, params: dict[str, Any]) -> None:
        """Called once when the entity is instantiated into the scene."""
        self.node = node
        self.app = app
        self.params = params
        self.is_active = True

    def on_fixed_update(self, dt: float) -> None:
        """Called every fixed physics step (constant dt). Zero heap allocations permitted."""
        pass

    def on_interact(self, interactor_id: int | None = None) -> None:
        """Called when an interactor (e.g. player pressing E) interacts with this entity."""
        pass

    def on_signal(self, signal: str, payload: Any = None) -> None:
        """Called when a signal or event is delivered to this entity."""
        pass

    def on_despawn(self) -> None:
        """Called when the entity is torn down."""
        self.is_active = False
        self.node = None
        self.app = None


# Global registry for mapping script type names to classes
_SCRIPT_REGISTRY: dict[str, type[EntityScript]] = {}


def register_script_type(type_name: str) -> Callable[[type[EntityScript]], type[EntityScript]]:
    """Decorator to register an EntityScript class by name."""

    def decorator(cls: type[EntityScript]) -> type[EntityScript]:
        _SCRIPT_REGISTRY[type_name] = cls
        return cls

    return decorator


def get_script_class(type_name: str) -> type[EntityScript] | None:
    """Resolves an EntityScript class by identifier name."""
    return _SCRIPT_REGISTRY.get(type_name)


class ScriptRunner:
    """Manages active entity script instances and dispatches fixed physics updates."""

    __slots__ = ("_scripts", "_script_by_entity")

    def __init__(self) -> None:
        self._scripts: list[EntityScript] = []
        self._script_by_entity: dict[str, list[EntityScript]] = {}

    def attach_script(self, script: EntityScript) -> None:
        """Registers an active script instance with the runner."""
        if script not in self._scripts:
            self._scripts.append(script)
            if script.node is not None:
                ent_id = script.node.id
                if ent_id not in self._script_by_entity:
                    self._script_by_entity[ent_id] = []
                self._script_by_entity[ent_id].append(script)

    def detach_script(self, script: EntityScript) -> None:
        """Removes a script from active ticking and triggers on_despawn."""
        if script in self._scripts:
            self._scripts.remove(script)
            if script.node is not None and script.node.id in self._script_by_entity:
                self._script_by_entity[script.node.id].remove(script)
            script.on_despawn()

    def on_fixed_update(self, dt: float) -> None:
        """Dispatches fixed physics tick to all active scripts with zero allocations."""
        for script in self._scripts:
            if script.is_active:
                script.on_fixed_update(dt)

    def signal_entity(self, target_id: str, signal: str, payload: Any = None) -> int:
        """Delivers a signal to all scripts attached to a target entity."""
        scripts = self._script_by_entity.get(target_id)
        if not scripts:
            return 0
        delivered = 0
        for script in scripts:
            if script.is_active:
                script.on_signal(signal, payload)
                delivered += 1
        return delivered

    def clear(self) -> None:
        """Detaches and cleans up all active scripts."""
        for script in list(self._scripts):
            script.on_despawn()
        self._scripts.clear()
        self._script_by_entity.clear()
