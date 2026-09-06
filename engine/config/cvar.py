"""Console Variable (CVar) Implementation for PyMordial Engine."""

from __future__ import annotations
from enum import Flag, auto
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T", int, float, bool, str)


class CVarFlags(Flag):
    """Behavioral and persistence flags for Console Variables."""
    NONE = 0
    ARCHIVE = auto()     # Serialized to disk in settings.toml
    READ_ONLY = auto()   # Cannot be modified at runtime
    CHEAT = auto()       # Requires cheat mode enabled
    RUNTIME_ONLY = auto()# In-memory only, never saved


class CVar(Generic[T]):
    """Strongly typed console variable with bounds validation and change notifications."""

    __slots__ = (
        "name",
        "val_type",
        "default_value",
        "_value",
        "description",
        "flags",
        "min_val",
        "max_val",
        "_callbacks",
    )

    def __init__(
        self,
        name: str,
        default_value: T,
        description: str = "",
        flags: CVarFlags = CVarFlags.ARCHIVE,
        min_val: T | None = None,
        max_val: T | None = None,
    ) -> None:
        self.name = name.lower()
        self.val_type = type(default_value)
        self.default_value = default_value
        self._value = default_value
        self.description = description
        self.flags = flags
        self.min_val = min_val
        self.max_val = max_val
        self._callbacks: list[Callable[[T, T], None]] = []

    @property
    def value(self) -> T:
        """Returns the current strongly-typed value."""
        return self._value

    def get(self) -> T:
        """Convenience getter matching value property."""
        return self._value

    def set(self, new_value: Any) -> bool:
        """Assigns a new value with clamping and notification dispatch."""
        if CVarFlags.READ_ONLY in self.flags:
            return False

        # Type conversion
        try:
            if self.val_type is bool and isinstance(new_value, str):
                parsed = new_value.strip().lower() in ("true", "1", "yes", "on")
            else:
                parsed = self.val_type(new_value)
        except (ValueError, TypeError):
            return False

        # Clamping for numeric values
        if self.val_type in (int, float):
            if self.min_val is not None:
                parsed = max(self.min_val, parsed)
            if self.max_val is not None:
                parsed = min(self.max_val, parsed)

        if parsed == self._value:
            return True

        old_val = self._value
        self._value = parsed

        for cb in self._callbacks:
            try:
                cb(old_val, parsed)
            except Exception:
                pass

        return True

    def set_from_string(self, text: str) -> bool:
        """Parses and applies a value string from a console or config file."""
        return self.set(text)

    def reset(self) -> None:
        """Restores the CVar to its default factory value."""
        self.set(self.default_value)

    def add_callback(self, callback: Callable[[T, T], None]) -> None:
        """Registers a change notification listener (old_val, new_val)."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[T, T], None]) -> None:
        """Detaches a previously registered change listener."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def __repr__(self) -> str:
        return f"<CVar '{self.name}' = {self._value!r} ({self.val_type.__name__})>"
