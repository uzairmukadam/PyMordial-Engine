"""Dedicated AAA-Style Input Subsystem for PyMordial Engine.

Exposes:
- InputManager: Central coordinator for Keyboard, Mouse, and Gamepad inputs
- Key, MouseButton, GamepadButton, GamepadAxis, DeviceType: Hardware codes
- ActionBinding, AxisBinding, Vector2Binding: Declarative binding models
"""

from engine.input.codes import (
    Key,
    MouseButton,
    GamepadButton,
    GamepadAxis,
    DeviceType,
)
from engine.input.actions import (
    ActionBinding,
    AxisBinding,
    Vector2Binding,
)
from engine.input.input_manager import InputManager

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
