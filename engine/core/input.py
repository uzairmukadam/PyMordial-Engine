"""Legacy input module for backwards compatibility.

All engine input is now centralized exclusively in `engine.input`.
"""

from engine.input import (
    InputManager,
    Key,
    MouseButton,
    GamepadButton,
    GamepadAxis,
    DeviceType,
    ActionBinding,
    AxisBinding,
    Vector2Binding,
)

__all__ = [
    "InputManager",
    "Key",
    "MouseButton",
    "GamepadButton",
    "GamepadAxis",
    "DeviceType",
    "ActionBinding",
    "AxisBinding",
    "Vector2Binding",
]
