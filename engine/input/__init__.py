"""Dedicated AAA-Style Input Subsystem for PyMordial Engine.

Exposes:
- InputManager: Central coordinator for Keyboard, Mouse, and Gamepad inputs
- Key, MouseButton, GamepadButton, GamepadAxis, DeviceType: Hardware codes
- ActionBinding, AxisBinding, Vector2Binding: Declarative binding models
- InputContext, InputContextStack: Layered priority input mapping
- HapticsManager: Gamepad dual-motor vibration controller
- save_bindings_to_file, load_bindings_from_file: Serialization
"""

from engine.input.actions import (
    ActionBinding,
    AxisBinding,
    Vector2Binding,
)
from engine.input.codes import (
    DeviceType,
    GamepadAxis,
    GamepadButton,
    Key,
    MouseButton,
)
from engine.input.context import InputContext, InputContextStack
from engine.input.haptics import HapticsManager
from engine.input.input_manager import InputManager
from engine.input.rebinding import load_bindings_from_file, save_bindings_to_file

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
    "InputContext",
    "InputContextStack",
    "HapticsManager",
    "save_bindings_to_file",
    "load_bindings_from_file",
]
