"""Declarative Action and Axis Binding Configurations for PyMordial Engine.

Encapsulates mappings between semantic gameplay/debug actions and physical
hardware inputs (Keyboard keys, Mouse buttons, and Gamepad sticks/buttons).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from engine.input.codes import Key, MouseButton, GamepadButton


@dataclass(slots=True)
class ActionBinding:
    """Configures a discrete boolean action (e.g. 'jump', 'interact', 'debug_menu')."""

    name: str
    keys: list[int] = field(default_factory=list)
    mouse_buttons: list[int] = field(default_factory=list)
    gamepad_buttons: list[int] = field(default_factory=list)

    def add_keys(self, *keys: int | Key) -> ActionBinding:
        for k in keys:
            if int(k) not in self.keys:
                self.keys.append(int(k))
        return self

    def add_mouse_buttons(self, *buttons: int | MouseButton) -> ActionBinding:
        for b in buttons:
            if int(b) not in self.mouse_buttons:
                self.mouse_buttons.append(int(b))
        return self

    def add_gamepad_buttons(self, *buttons: int | GamepadButton) -> ActionBinding:
        for b in buttons:
            if int(b) not in self.gamepad_buttons:
                self.gamepad_buttons.append(int(b))
        return self


@dataclass(slots=True)
class AxisBinding:
    """Configures a 1D continuous analog axis (e.g. 'move_forward', 'look_pitch')."""

    name: str
    pos_key: int | None = None
    neg_key: int | None = None
    alt_pos_key: int | None = None
    alt_neg_key: int | None = None
    gamepad_axis: int | None = None
    invert_gamepad: bool = False
    deadzone: float = 0.15
    sensitivity: float = 1.0


@dataclass(slots=True)
class Vector2Binding:
    """Combines two 1D axes into a 2D vector with optional circular clamping."""

    name: str
    axis_x: str
    axis_y: str
    normalize: bool = True
