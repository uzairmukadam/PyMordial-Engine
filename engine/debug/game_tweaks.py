"""Tier 3 Debug Subsystem: Game-Specific Developer Tweaks & Inspector.

Provides a declarative, data-driven registry where game developers can expose
custom runtime tunable parameters (booleans, numeric sliders, enums, triggers,
and live watch expressions) organized into clean collapsible categories.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class TweakType(IntEnum):
    """Supported types of game-specific developer tweaks."""

    BOOL = 1
    FLOAT = 2
    INT = 3
    CHOICE = 4
    ACTION = 5
    WATCH = 6


@dataclass(slots=True)
class TweakItem:
    """Represents a single registered developer tweak or watcher."""

    category: str
    name: str
    tweak_type: TweakType
    value: Any = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: list[str] = field(default_factory=list)
    callback: Callable[..., Any] | None = None
    getter: Callable[[], Any] | None = None
    on_changed: Callable[[Any], None] | None = None

    def set_value(self, new_val: Any) -> None:
        """Updates value, clamping if bounded, and fires on_changed callback."""
        if self.tweak_type == TweakType.FLOAT:
            v = float(new_val)
            if self.min_val is not None:
                v = max(self.min_val, v)
            if self.max_val is not None:
                v = min(self.max_val, v)
            self.value = v
        elif self.tweak_type == TweakType.INT:
            v = int(new_val)
            if self.min_val is not None:
                v = max(int(self.min_val), v)
            if self.max_val is not None:
                v = min(int(self.max_val), v)
            self.value = v
        elif self.tweak_type == TweakType.BOOL:
            self.value = bool(new_val)
        elif self.tweak_type == TweakType.CHOICE:
            if new_val in self.choices:
                self.value = new_val
        else:
            self.value = new_val

        if self.on_changed is not None:
            self.on_changed(self.value)

    def trigger(self) -> Any:
        """Executes action callback if tweak is an action."""
        if self.tweak_type == TweakType.ACTION and self.callback is not None:
            return self.callback()
        return None

    def read_watch(self) -> Any:
        """Evaluates live watch expression."""
        if self.tweak_type == TweakType.WATCH and self.getter is not None:
            try:
                return self.getter()
            except Exception as e:
                return f"<Error: {e}>"
        return self.value


class GameTweaks:
    """Declarative registry for game-specific runtime parameters."""

    __slots__ = ("_categories", "_lookup")

    def __init__(self) -> None:
        # category -> list of TweakItems
        self._categories: dict[str, list[TweakItem]] = {}
        # (category, name) -> TweakItem
        self._lookup: dict[tuple[str, str], TweakItem] = {}

    def _register(self, item: TweakItem) -> TweakItem:
        cat_list = self._categories.setdefault(item.category, [])
        cat_list.append(item)
        self._lookup[(item.category, item.name)] = item
        return item

    def add_bool(
        self,
        category: str,
        name: str,
        default: bool = False,
        on_changed: Callable[[bool], None] | None = None,
    ) -> TweakItem:
        """Adds a toggleable boolean checkbox tweak."""
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.BOOL,
                value=default,
                on_changed=on_changed,
            )
        )

    def add_float(
        self,
        category: str,
        name: str,
        default: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 100.0,
        step: float = 0.1,
        on_changed: Callable[[float], None] | None = None,
    ) -> TweakItem:
        """Adds a floating-point numeric slider tweak."""
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.FLOAT,
                value=float(default),
                min_val=float(min_val),
                max_val=float(max_val),
                step=float(step),
                on_changed=on_changed,
            )
        )

    def add_int(
        self,
        category: str,
        name: str,
        default: int = 1,
        min_val: int = 0,
        max_val: int = 100,
        step: int = 1,
        on_changed: Callable[[int], None] | None = None,
    ) -> TweakItem:
        """Adds an integer numeric slider tweak."""
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.INT,
                value=int(default),
                min_val=float(min_val),
                max_val=float(max_val),
                step=float(step),
                on_changed=on_changed,
            )
        )

    def add_choice(
        self,
        category: str,
        name: str,
        choices: list[str],
        default: str | None = None,
        on_changed: Callable[[str], None] | None = None,
    ) -> TweakItem:
        """Adds a multi-choice dropdown/cycler tweak."""
        initial = default if default is not None and default in choices else (choices[0] if choices else "")
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.CHOICE,
                value=initial,
                choices=list(choices),
                on_changed=on_changed,
            )
        )

    def add_action(
        self,
        category: str,
        name: str,
        callback: Callable[[], Any],
    ) -> TweakItem:
        """Adds an executable trigger button tweak."""
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.ACTION,
                callback=callback,
            )
        )

    def add_watch(
        self,
        category: str,
        name: str,
        getter: Callable[[], Any],
    ) -> TweakItem:
        """Adds a live dynamic expression watcher."""
        return self._register(
            TweakItem(
                category=category,
                name=name,
                tweak_type=TweakType.WATCH,
                getter=getter,
            )
        )

    def get_value(self, category: str, name: str, default: Any = None) -> Any:
        """Retrieves current value of a registered tweak."""
        item = self._lookup.get((category, name))
        if item is None:
            return default
        if item.tweak_type == TweakType.WATCH:
            return item.read_watch()
        return item.value

    def set_value(self, category: str, name: str, new_value: Any) -> None:
        """Updates value of a registered tweak."""
        item = self._lookup.get((category, name))
        if item is not None:
            item.set_value(new_value)

    def trigger_action(self, category: str, name: str) -> Any:
        """Triggers a registered action tweak."""
        item = self._lookup.get((category, name))
        if item is not None:
            return item.trigger()
        return None

    def get_categories(self) -> list[str]:
        """Returns sorted list of registered categories."""
        return sorted(self._categories.keys())

    def get_items(self, category: str) -> list[TweakItem]:
        """Returns list of tweaks registered under a specific category."""
        return self._categories.get(category, [])
