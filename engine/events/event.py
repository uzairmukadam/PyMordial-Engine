"""Event Base Classes and Standard Engine Events for PyMordial Engine."""

from __future__ import annotations
import time
from typing import Any


class Event:
    """Base class for all engine and game events."""

    __slots__ = ("timestamp", "handled")

    def __init__(self, timestamp: float | None = None, handled: bool = False) -> None:
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.handled = handled

    def stop_propagation(self) -> None:
        """Marks the event as handled, halting dispatch to lower-priority listeners."""
        self.handled = True


# --- Window Events ---

class WindowResizeEvent(Event):
    """Fired when the window resolution or viewport changes."""

    __slots__ = ("width", "height")

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"<WindowResizeEvent {self.width}x{self.height}>"


class WindowModeChangedEvent(Event):
    """Fired when display mode shifts (Windowed, Borderless, Exclusive)."""

    __slots__ = ("mode", "width", "height")

    def __init__(self, mode: int = 0, width: int = 1280, height: int = 720) -> None:
        super().__init__()
        self.mode = mode
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"<WindowModeChangedEvent mode={self.mode} {self.width}x{self.height}>"


class WindowFocusEvent(Event):
    """Fired when OS focus is gained or lost."""

    __slots__ = ("has_focus",)

    def __init__(self, has_focus: bool = True) -> None:
        super().__init__()
        self.has_focus = has_focus

    def __repr__(self) -> str:
        return f"<WindowFocusEvent focused={self.has_focus}>"


class WindowCloseEvent(Event):
    """Fired when user or OS requests engine shutdown."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "<WindowCloseEvent>"


# --- Input Events ---

class InputActionEvent(Event):
    """Fired when a semantic gameplay action transitions state."""

    __slots__ = ("action", "is_pressed", "is_down", "device")

    def __init__(
        self,
        action: str = "",
        is_pressed: bool = False,
        is_down: bool = False,
        device: int = 0,
    ) -> None:
        super().__init__()
        self.action = action
        self.is_pressed = is_pressed
        self.is_down = is_down
        self.device = device

    def __repr__(self) -> str:
        return f"<InputActionEvent '{self.action}' pressed={self.is_pressed} down={self.is_down}>"


# --- Configuration & Time Events ---

class CVarChangedEvent(Event):
    """Fired when a console variable's value is altered."""

    __slots__ = ("name", "old_value", "new_value")

    def __init__(self, name: str = "", old_value: Any = None, new_value: Any = None) -> None:
        super().__init__()
        self.name = name
        self.old_value = old_value
        self.new_value = new_value

    def __repr__(self) -> str:
        return f"<CVarChangedEvent '{self.name}': {self.old_value!r} -> {self.new_value!r}>"


class TimeScaleChangedEvent(Event):
    """Fired when global time dilation changes."""

    __slots__ = ("old_scale", "new_scale")

    def __init__(self, old_scale: float = 1.0, new_scale: float = 1.0) -> None:
        super().__init__()
        self.old_scale = old_scale
        self.new_scale = new_scale

    def __repr__(self) -> str:
        return f"<TimeScaleChangedEvent {self.old_scale:.2f} -> {self.new_scale:.2f}>"


# --- Audio Events ---

class PlaySoundCueEvent(Event):
    """Fired to trigger an audio playback request."""

    __slots__ = ("cue_name", "position", "volume", "bus")

    def __init__(
        self,
        cue_name: str = "",
        position: tuple[float, float, float] | None = None,
        volume: float = 1.0,
        bus: str = "SFX",
    ) -> None:
        super().__init__()
        self.cue_name = cue_name
        self.position = position
        self.volume = volume
        self.bus = bus

    def __repr__(self) -> str:
        return f"<PlaySoundCueEvent '{self.cue_name}' vol={self.volume:.2f} bus='{self.bus}'>"
