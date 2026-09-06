"""Layered Context-Based Input Management for PyMordial Engine."""

from __future__ import annotations
from typing import Sequence
from engine.input.actions import ActionBinding, AxisBinding, Vector2Binding
from engine.logging import log_info, LogChannel


class InputContext:
    """Represents a discrete prioritized layer of input mappings (e.g., UI, Gameplay, Vehicle)."""

    __slots__ = (
        "name",
        "priority",
        "is_active",
        "blocks_lower_contexts",
        "actions",
        "axes",
        "vectors",
        "_consumed_actions",
    )

    def __init__(
        self,
        name: str,
        priority: int = 0,
        blocks_lower_contexts: bool = False,
    ) -> None:
        self.name = name
        self.priority = priority
        self.is_active = True
        self.blocks_lower_contexts = blocks_lower_contexts

        self.actions: dict[str, ActionBinding] = {}
        self.axes: dict[str, AxisBinding] = {}
        self.vectors: dict[str, Vector2Binding] = {}
        self._consumed_actions: set[str] = set()

    def register_action(self, binding: ActionBinding, consume_on_trigger: bool = True) -> ActionBinding:
        """Adds an action to this context."""
        self.actions[binding.name] = binding
        if consume_on_trigger:
            self._consumed_actions.add(binding.name)
        return binding

    def register_axis(self, binding: AxisBinding) -> AxisBinding:
        self.axes[binding.name] = binding
        return binding

    def register_vector2(self, binding: Vector2Binding) -> Vector2Binding:
        self.vectors[binding.name] = binding
        return binding

    def should_consume(self, action_name: str) -> bool:
        """Returns True if this action stops propagation to lower-priority layers."""
        return self.blocks_lower_contexts or (action_name in self._consumed_actions)

    def __repr__(self) -> str:
        return f"<InputContext '{self.name}' priority={self.priority} active={self.is_active}>"


class InputContextStack:
    """Manages active input contexts sorted by execution priority."""

    __slots__ = ("_contexts",)

    def __init__(self) -> None:
        self._contexts: list[InputContext] = []

    def push_context(self, context: InputContext) -> None:
        """Adds a context to the stack and maintains descending priority ordering."""
        if context not in self._contexts:
            self._contexts.append(context)
            self._contexts.sort(key=lambda c: c.priority, reverse=True)
            log_info(LogChannel.INPUT, f"Pushed input context: '{context.name}' (priority {context.priority})")

    def pop_context(self, context_or_name: InputContext | str) -> InputContext | None:
        """Removes a context from the stack."""
        name = context_or_name if isinstance(context_or_name, str) else context_or_name.name
        for i, ctx in enumerate(self._contexts):
            if ctx.name.lower() == name.lower():
                removed = self._contexts.pop(i)
                log_info(LogChannel.INPUT, f"Popped input context: '{removed.name}'")
                return removed
        return None

    def get_context(self, name: str) -> InputContext | None:
        """Finds a context by name."""
        for ctx in self._contexts:
            if ctx.name.lower() == name.lower():
                return ctx
        return None

    def active_contexts(self) -> list[InputContext]:
        """Returns all currently active contexts ordered by priority."""
        return [c for c in self._contexts if c.is_active]

    def clear(self) -> None:
        self._contexts.clear()
