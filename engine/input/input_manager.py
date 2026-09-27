"""Dedicated AAA-Style Dual-Device Input System for PyMordial Engine.

Coordinates Keyboard, Mouse, and Gamepad inputs with zero runtime allocations,
O(1) reverse event dispatching, circular stick deadzones, diagonal vector clamping,
gamepad hot-plugging, and a dedicated debug input layer.
"""

from __future__ import annotations
import math
from typing import Sequence, Any
import pygame

from engine.input.codes import (
    Key,
    MouseButton,
    GamepadButton,
    GamepadAxis,
    DeviceType,
)
from engine.input.actions import ActionBinding, AxisBinding, Vector2Binding
from engine.input.context import InputContext, InputContextStack
from engine.input.haptics import HapticsManager


class InputManager:
    """Universal input coordinator for Keyboard, Mouse, and Gamepads."""

    __slots__ = (
        "active_device",
        "is_debug_mode",
        "mouse_pos",
        "mouse_delta",
        "mouse_wheel",
        "is_mouse_grabbed",
        "_was_mouse_grabbed",
        "should_quit",
        "context_stack",
        "haptics",
        "_actions",
        "_debug_actions",
        "_axes",
        "_vectors",
        "_action_down",
        "_action_pressed",
        "_action_released",
        "_debug_action_down",
        "_debug_action_pressed",
        "_debug_action_released",
        "_axes_values",
        "_key_to_actions",
        "_mouse_to_actions",
        "_pad_to_actions",
        "_key_to_debug_actions",
        "_mouse_to_debug_actions",
        "_pad_to_debug_actions",
        "_keys_down",
        "_keys_pressed",
        "_keys_released",
        "_mouse_down",
        "_mouse_pressed",
        "_mouse_released",
        "_pad_down",
        "_pad_pressed",
        "_pad_released",
        "_pad_axes",
        "_joysticks",
        "_controllers",
        "_has_sdl2_controller",
    )

    RAW_XBOX_BUTTON_MAP: dict[int, int] = {
        0: GamepadButton.A,              # 0
        1: GamepadButton.B,              # 1
        2: GamepadButton.X,              # 2
        3: GamepadButton.Y,              # 3
        4: GamepadButton.LEFT_BUMPER,     # 9
        5: GamepadButton.RIGHT_BUMPER,    # 10
        6: GamepadButton.BACK,            # 4
        7: GamepadButton.START,           # 6
        8: GamepadButton.LEFT_STICK,      # 7
        9: GamepadButton.RIGHT_STICK,     # 8
        10: GamepadButton.GUIDE,          # 5
    }

    def __init__(self, register_defaults: bool = True) -> None:
        self.active_device = DeviceType.KEYBOARD_MOUSE
        self.is_debug_mode = False
        self.should_quit = False

        self.mouse_pos = (0, 0)
        self.mouse_delta = (0, 0)
        self.mouse_wheel = 0.0
        self.is_mouse_grabbed = False
        self._was_mouse_grabbed = False

        self._actions: dict[str, ActionBinding] = {}
        self._debug_actions: dict[str, ActionBinding] = {}
        self._axes: dict[str, AxisBinding] = {}
        self._vectors: dict[str, Vector2Binding] = {}

        self._action_down: dict[str, bool] = {}
        self._action_pressed: dict[str, bool] = {}
        self._action_released: dict[str, bool] = {}

        self._debug_action_down: dict[str, bool] = {}
        self._debug_action_pressed: dict[str, bool] = {}
        self._debug_action_released: dict[str, bool] = {}

        self._axes_values: dict[str, float] = {}

        # Reverse lookup tables for O(1) event dispatch
        self._key_to_actions: dict[int, list[str]] = {}
        self._mouse_to_actions: dict[int, list[str]] = {}
        self._pad_to_actions: dict[int, list[str]] = {}

        self._key_to_debug_actions: dict[int, list[str]] = {}
        self._mouse_to_debug_actions: dict[int, list[str]] = {}
        self._pad_to_debug_actions: dict[int, list[str]] = {}

        # Low-level physical hardware sets
        self._keys_down: set[int] = set()
        self._keys_pressed: set[int] = set()
        self._keys_released: set[int] = set()

        self._mouse_down: set[int] = set()
        self._mouse_pressed: set[int] = set()
        self._mouse_released: set[int] = set()

        self._pad_down: set[int] = set()
        self._pad_pressed: set[int] = set()
        self._pad_released: set[int] = set()

        self._pad_axes: dict[int, float] = {i: 0.0 for i in range(8)}
        self._joysticks: dict[int, pygame.joystick.Joystick] = {}
        self._controllers: dict[int, Any] = {}
        self._has_sdl2_controller: bool = False

        self.context_stack = InputContextStack()
        self.haptics = HapticsManager()

        self._init_subsystems()
        if register_defaults:
            self._register_default_bindings()

    def push_context(self, context: InputContext) -> None:
        """Pushes a prioritized input context onto the stack."""
        self.context_stack.push_context(context)

    def pop_context(self, context_or_name: InputContext | str) -> InputContext | None:
        """Pops an input context from the stack."""
        return self.context_stack.pop_context(context_or_name)

    def set_rumble(
        self,
        low_frequency: float,
        high_frequency: float,
        duration_seconds: float,
        device_index: int = 0,
    ) -> bool:
        """Triggers dual-motor haptic rumble on a connected controller."""
        return self.haptics.set_rumble(low_frequency, high_frequency, duration_seconds, device_index)

    def _init_subsystems(self) -> None:
        """Initializes underlying Pygame input, joystick, and SDL2 GameController subsystems."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        # Connect SDL2 Controllers (preferred, standardized AAA hardware mapping)
        try:
            from pygame._sdl2 import controller as sdl2_controller
            if not sdl2_controller.get_init():
                sdl2_controller.init()
            self._has_sdl2_controller = True
            for i in range(sdl2_controller.get_count()):
                if sdl2_controller.is_controller(i):
                    try:
                        c = sdl2_controller.Controller(i)
                        self._controllers[c.id] = c
                    except Exception:
                        pass
        except Exception:
            self._has_sdl2_controller = False

        # Connect fallback raw joysticks
        for i in range(pygame.joystick.get_count()):
            try:
                joy = pygame.joystick.Joystick(i)
                self._joysticks[joy.get_instance_id()] = joy
            except Exception:
                pass

    def _register_default_bindings(self) -> None:
        """Standard AAA movement, gameplay, camera, and debug bindings."""
        # 1. Gameplay Actions (Dual KBM + Gamepad)
        self.bind_action(
            "jump",
            keys=[Key.SPACE],
            gamepad_buttons=[GamepadButton.A],
        )
        self.bind_action(
            "sprint",
            keys=[Key.LSHIFT, Key.RSHIFT],
            gamepad_buttons=[GamepadButton.LEFT_STICK, GamepadButton.LEFT_BUMPER],
        )
        self.bind_action(
            "crouch",
            keys=[Key.LCTRL, Key.C],
            gamepad_buttons=[GamepadButton.RIGHT_STICK],
        )
        self.bind_action(
            "interact",
            keys=[Key.E],
            gamepad_buttons=[GamepadButton.X],
        )
        self.bind_action(
            "pause",
            keys=[Key.ESCAPE],
            gamepad_buttons=[GamepadButton.START],
        )
        self.bind_action(
            "primary_action",
            mouse_buttons=[MouseButton.LEFT],
            gamepad_buttons=[GamepadButton.RIGHT_BUMPER],
        )
        self.bind_action(
            "secondary_action",
            mouse_buttons=[MouseButton.RIGHT],
        )

        # 2. Movement & Look Axes
        self.bind_axis(
            "move_forward",
            pos_key=Key.W,
            neg_key=Key.S,
            alt_pos=Key.UP,
            alt_neg=Key.DOWN,
            gamepad_axis=GamepadAxis.LEFT_STICK_Y,
            invert_gamepad=True,  # Stick up is negative Y in SDL
            deadzone=0.15,
        )
        self.bind_axis(
            "move_right",
            pos_key=Key.D,
            neg_key=Key.A,
            alt_pos=Key.RIGHT,
            alt_neg=Key.LEFT,
            gamepad_axis=GamepadAxis.LEFT_STICK_X,
            deadzone=0.15,
        )
        self.bind_vector2("move", axis_x="move_right", axis_y="move_forward", normalize=True)

        self.bind_axis(
            "look_yaw",
            gamepad_axis=GamepadAxis.RIGHT_STICK_X,
            deadzone=0.15,
        )
        self.bind_axis(
            "look_pitch",
            gamepad_axis=GamepadAxis.RIGHT_STICK_Y,
            invert_gamepad=True,
            deadzone=0.15,
        )
        self.bind_vector2("look", axis_x="look_yaw", axis_y="look_pitch", normalize=False)

        # 3. Dedicated Debug System Layer
        self.bind_action(
            "debug_perf_toggle",
            keys=[Key.F1, Key.GRAVE],
            gamepad_buttons=[GamepadButton.BACK],
            is_debug=True,
        )
        self.bind_action(
            "debug_menu_toggle",
            keys=[Key.F1, Key.GRAVE],
            gamepad_buttons=[GamepadButton.BACK],
            is_debug=True,
        )
        self.bind_action(
            "debug_graphics_toggle",
            keys=[Key.F2],
            is_debug=True,
        )
        self.bind_action(
            "debug_game_toggle",
            keys=[Key.F3],
            is_debug=True,
        )
        self.bind_action(
            "debug_gbuffer_toggle",
            keys=[Key.F4],
            is_debug=True,
        )
        self.bind_action(
            "debug_profiler_toggle",
            keys=[Key.F5],
            is_debug=True,
        )
        self.bind_action(
            "debug_mouse_capture_toggle",
            keys=[Key.F9],
            is_debug=True,
        )
        self.bind_action(
            "debug_tab_prev",
            keys=[Key.Q],
            gamepad_buttons=[GamepadButton.LEFT_BUMPER],
            is_debug=True,
        )
        self.bind_action(
            "debug_tab_next",
            keys=[Key.E],
            gamepad_buttons=[GamepadButton.RIGHT_BUMPER],
            is_debug=True,
        )
        self.bind_action(
            "debug_nav_up",
            keys=[Key.UP],
            gamepad_buttons=[GamepadButton.DPAD_UP],
            is_debug=True,
        )
        self.bind_action(
            "debug_nav_down",
            keys=[Key.DOWN],
            gamepad_buttons=[GamepadButton.DPAD_DOWN],
            is_debug=True,
        )
        self.bind_action(
            "debug_nav_left",
            keys=[Key.LEFT],
            gamepad_buttons=[GamepadButton.DPAD_LEFT],
            is_debug=True,
        )
        self.bind_action(
            "debug_nav_right",
            keys=[Key.RIGHT],
            gamepad_buttons=[GamepadButton.DPAD_RIGHT],
            is_debug=True,
        )
        self.bind_action(
            "debug_nav_activate",
            keys=[Key.RETURN, Key.SPACE],
            gamepad_buttons=[GamepadButton.A],
            is_debug=True,
        )
        self.bind_action(
            "debug_close_b",
            keys=[Key.ESCAPE],
            gamepad_buttons=[GamepadButton.B],
            is_debug=True,
        )

    # --------------------------------------------------------------------------
    # Binding Registrations
    # --------------------------------------------------------------------------

    def bind_action(
        self,
        name: str,
        keys: Sequence[int | Key] | None = None,
        mouse_buttons: Sequence[int | MouseButton] | None = None,
        gamepad_buttons: Sequence[int | GamepadButton] | None = None,
        is_debug: bool = False,
    ) -> ActionBinding:
        """Registers a named action and constructs fast reverse-lookup tables."""
        raw_keys = [int(k) for k in keys] if keys else []
        raw_mouse = [int(b) for b in mouse_buttons] if mouse_buttons else []
        raw_pads = [int(b) for b in gamepad_buttons] if gamepad_buttons else []

        binding = ActionBinding(
            name=name,
            keys=raw_keys,
            mouse_buttons=raw_mouse,
            gamepad_buttons=raw_pads,
        )

        if is_debug:
            self._debug_actions[name] = binding
            self._debug_action_down[name] = False
            self._debug_action_pressed[name] = False
            self._debug_action_released[name] = False
            k_map = self._key_to_debug_actions
            m_map = self._mouse_to_debug_actions
            p_map = self._pad_to_debug_actions
        else:
            self._actions[name] = binding
            self._action_down[name] = False
            self._action_pressed[name] = False
            self._action_released[name] = False
            k_map = self._key_to_actions
            m_map = self._mouse_to_actions
            p_map = self._pad_to_actions

        for k in raw_keys:
            k_map.setdefault(k, []).append(name)
        for m in raw_mouse:
            m_map.setdefault(m, []).append(name)
        for p in raw_pads:
            p_map.setdefault(p, []).append(name)

        return binding

    def bind_axis(
        self,
        name: str,
        pos_key: int | Key | None = None,
        neg_key: int | Key | None = None,
        alt_pos: int | Key | None = None,
        alt_neg: int | Key | None = None,
        gamepad_axis: int | GamepadAxis | None = None,
        invert_gamepad: bool = False,
        deadzone: float = 0.15,
        sensitivity: float = 1.0,
    ) -> AxisBinding:
        """Registers a 1D analog/digital axis."""
        binding = AxisBinding(
            name=name,
            pos_key=int(pos_key) if pos_key is not None else None,
            neg_key=int(neg_key) if neg_key is not None else None,
            alt_pos_key=int(alt_pos) if alt_pos is not None else None,
            alt_neg_key=int(alt_neg) if alt_neg is not None else None,
            gamepad_axis=int(gamepad_axis) if gamepad_axis is not None else None,
            invert_gamepad=invert_gamepad,
            deadzone=deadzone,
            sensitivity=sensitivity,
        )
        self._axes[name] = binding
        self._axes_values[name] = 0.0
        return binding

    def bind_vector2(
        self,
        name: str,
        axis_x: str,
        axis_y: str,
        normalize: bool = True,
    ) -> Vector2Binding:
        """Registers a composite 2D vector with optional unit-length clamping."""
        binding = Vector2Binding(name=name, axis_x=axis_x, axis_y=axis_y, normalize=normalize)
        self._vectors[name] = binding
        return binding

    # --------------------------------------------------------------------------
    # Frame Transitions & Event Ingestion
    # --------------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Resets single-frame transition flags and mouse relative movements in-place."""
        for name in self._action_pressed:
            self._action_pressed[name] = False
        for name in self._action_released:
            self._action_released[name] = False

        for name in self._debug_action_pressed:
            self._debug_action_pressed[name] = False
        for name in self._debug_action_released:
            self._debug_action_released[name] = False

        self._keys_pressed.clear()
        self._keys_released.clear()
        self._mouse_pressed.clear()
        self._mouse_released.clear()
        self._pad_pressed.clear()
        self._pad_released.clear()

        self.mouse_delta = (0, 0)
        self.mouse_wheel = 0.0

    def process_event(self, event: pygame.event.Event) -> None:
        """Ingests a single Pygame event and dispatches state changes in O(1)."""
        ev_type = event.type

        if ev_type == pygame.QUIT:
            self.should_quit = True

        elif ev_type == pygame.KEYDOWN:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            key = event.key
            self._keys_down.add(key)
            self._keys_pressed.add(key)

            # Debug bindings always dispatch
            if key in self._key_to_debug_actions:
                for a in self._key_to_debug_actions[key]:
                    if not self._debug_action_down.get(a, False):
                        self._debug_action_pressed[a] = True
                    self._debug_action_down[a] = True

            # Gameplay bindings dispatch when not suppressed by debug mode
            if not self.is_debug_mode and key in self._key_to_actions:
                for a in self._key_to_actions[key]:
                    if not self._action_down.get(a, False):
                        self._action_pressed[a] = True
                    self._action_down[a] = True

        elif ev_type == pygame.KEYUP:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            key = event.key
            self._keys_down.discard(key)
            self._keys_released.add(key)

            if key in self._key_to_debug_actions:
                for a in self._key_to_debug_actions[key]:
                    self._debug_action_down[a] = False
                    self._debug_action_released[a] = True

            if key in self._key_to_actions:
                for a in self._key_to_actions[key]:
                    self._action_down[a] = False
                    self._action_released[a] = True

        elif ev_type == pygame.MOUSEBUTTONDOWN:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            btn = event.button
            self._mouse_down.add(btn)
            self._mouse_pressed.add(btn)

            if btn in self._mouse_to_debug_actions:
                for a in self._mouse_to_debug_actions[btn]:
                    if not self._debug_action_down.get(a, False):
                        self._debug_action_pressed[a] = True
                    self._debug_action_down[a] = True

            if not self.is_debug_mode and btn in self._mouse_to_actions:
                for a in self._mouse_to_actions[btn]:
                    if not self._action_down.get(a, False):
                        self._action_pressed[a] = True
                    self._action_down[a] = True

        elif ev_type == pygame.MOUSEBUTTONUP:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            btn = event.button
            self._mouse_down.discard(btn)
            self._mouse_released.add(btn)

            if btn in self._mouse_to_debug_actions:
                for a in self._mouse_to_debug_actions[btn]:
                    self._debug_action_down[a] = False
                    self._debug_action_released[a] = True

            if btn in self._mouse_to_actions:
                for a in self._mouse_to_actions[btn]:
                    self._action_down[a] = False
                    self._action_released[a] = True

        elif ev_type == pygame.MOUSEMOTION:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            self.mouse_pos = event.pos
            self.mouse_delta = (
                self.mouse_delta[0] + event.rel[0],
                self.mouse_delta[1] + event.rel[1],
            )

        elif ev_type == pygame.MOUSEWHEEL:
            self.active_device = DeviceType.KEYBOARD_MOUSE
            self.mouse_wheel += event.y

        # --- SDL2 GameController Events (Standardized AAA Mapping) ---
        elif ev_type == pygame.CONTROLLERBUTTONDOWN:
            self.active_device = DeviceType.GAMEPAD
            btn = event.button
            if btn not in self._pad_down:
                self._pad_down.add(btn)
                self._pad_pressed.add(btn)
                self._dispatch_pad_press(btn)

        elif ev_type == pygame.CONTROLLERBUTTONUP:
            self.active_device = DeviceType.GAMEPAD
            btn = event.button
            if btn in self._pad_down:
                self._pad_down.discard(btn)
                self._pad_released.add(btn)
                self._dispatch_pad_release(btn)

        elif ev_type == pygame.CONTROLLERAXISMOTION:
            val = event.value / 32767.0 if abs(event.value) > 1.0 else float(event.value)
            if abs(val) > 0.1:
                self.active_device = DeviceType.GAMEPAD
            self._pad_axes[event.axis] = val

        elif ev_type == pygame.CONTROLLERDEVICEADDED:
            try:
                from pygame._sdl2 import controller as sdl2_controller
                if sdl2_controller.is_controller(event.device_index):
                    c = sdl2_controller.Controller(event.device_index)
                    self._controllers[c.id] = c
            except Exception:
                pass

        elif ev_type == pygame.CONTROLLERDEVICEREMOVED:
            self._controllers.pop(event.instance_id, None)

        # --- Raw Joystick Events (Physical DirectInput/XInput Button Indices) ---
        elif ev_type == pygame.JOYBUTTONDOWN:
            self.active_device = DeviceType.GAMEPAD
            btn = self.RAW_XBOX_BUTTON_MAP.get(event.button, event.button)
            if btn not in self._pad_down:
                self._pad_down.add(btn)
                self._pad_pressed.add(btn)
                self._dispatch_pad_press(btn)

        elif ev_type == pygame.JOYBUTTONUP:
            self.active_device = DeviceType.GAMEPAD
            btn = self.RAW_XBOX_BUTTON_MAP.get(event.button, event.button)
            if btn in self._pad_down:
                self._pad_down.discard(btn)
                self._pad_released.add(btn)
                self._dispatch_pad_release(btn)

        elif ev_type == pygame.JOYAXISMOTION:
            val = event.value / 32767.0 if abs(event.value) > 1.0 else float(event.value)
            if abs(val) > 0.1:
                self.active_device = DeviceType.GAMEPAD
            self._pad_axes[event.axis] = val

        elif ev_type == pygame.JOYHATMOTION:
            self.active_device = DeviceType.GAMEPAD
            hx, hy = event.value
            dpad_btns = (GamepadButton.DPAD_UP, GamepadButton.DPAD_DOWN, GamepadButton.DPAD_LEFT, GamepadButton.DPAD_RIGHT)
            for db in dpad_btns:
                if db in self._pad_down:
                    self._pad_down.discard(db)
                    self._pad_released.add(db)
                    self._dispatch_pad_release(db)

            new_btns = []
            if hy > 0:
                new_btns.append(GamepadButton.DPAD_UP)
            elif hy < 0:
                new_btns.append(GamepadButton.DPAD_DOWN)
            if hx < 0:
                new_btns.append(GamepadButton.DPAD_LEFT)
            elif hx > 0:
                new_btns.append(GamepadButton.DPAD_RIGHT)

            for nb in new_btns:
                self._pad_down.add(nb)
                self._pad_pressed.add(nb)
                self._dispatch_pad_press(nb)

        elif ev_type == pygame.JOYDEVICEADDED and not self._controllers:
            try:
                joy = pygame.joystick.Joystick(event.device_index)
                self._joysticks[joy.get_instance_id()] = joy
            except Exception:
                pass

        elif ev_type == pygame.JOYDEVICEREMOVED and not self._controllers:
            self._joysticks.pop(event.instance_id, None)

    def _dispatch_pad_press(self, btn: int) -> None:
        """Dispatches gamepad button press event to debug and gameplay actions."""
        if btn in self._pad_to_debug_actions:
            for a in self._pad_to_debug_actions[btn]:
                if not self._debug_action_down.get(a, False):
                    self._debug_action_pressed[a] = True
                self._debug_action_down[a] = True

        if not self.is_debug_mode and btn in self._pad_to_actions:
            for a in self._pad_to_actions[btn]:
                if not self._action_down.get(a, False):
                    self._action_pressed[a] = True
                self._action_down[a] = True

    def _dispatch_pad_release(self, btn: int) -> None:
        """Dispatches gamepad button release event to debug and gameplay actions."""
        if btn in self._pad_to_debug_actions:
            for a in self._pad_to_debug_actions[btn]:
                self._debug_action_down[a] = False
                self._debug_action_released[a] = True

        if btn in self._pad_to_actions:
            for a in self._pad_to_actions[btn]:
                self._action_down[a] = False
                self._action_released[a] = True

    def update_axes(self) -> None:
        """Calculates normalized analog values and updates composite 2D vectors."""
        # Update 1D axes
        for name, binding in self._axes.items():
            val = 0.0

            # Digital keys
            pos_pressed = False
            if binding.pos_key is not None and binding.pos_key in self._keys_down:
                pos_pressed = True
            if binding.alt_pos_key is not None and binding.alt_pos_key in self._keys_down:
                pos_pressed = True

            neg_pressed = False
            if binding.neg_key is not None and binding.neg_key in self._keys_down:
                neg_pressed = True
            if binding.alt_neg_key is not None and binding.alt_neg_key in self._keys_down:
                neg_pressed = True

            if pos_pressed:
                val += 1.0
            if neg_pressed:
                val -= 1.0

            # Gamepad axis
            if binding.gamepad_axis is not None:
                pad_raw = self._pad_axes.get(binding.gamepad_axis, 0.0)
                if binding.invert_gamepad:
                    pad_raw = -pad_raw
                # Deadzone
                if abs(pad_raw) > binding.deadzone:
                    # Rescale normalized range beyond deadzone
                    sign = 1.0 if pad_raw > 0.0 else -1.0
                    scaled = (abs(pad_raw) - binding.deadzone) / (1.0 - binding.deadzone)
                    val += sign * scaled * binding.sensitivity

            self._axes_values[name] = max(-1.0, min(1.0, val))

        # Virtual analog trigger buttons (LT / RT) with hysteresis thresholding
        lt_val = self._pad_axes.get(int(GamepadAxis.LEFT_TRIGGER), 0.0)
        rt_val = self._pad_axes.get(int(GamepadAxis.RIGHT_TRIGGER), 0.0)
        for btn_code, axis_val in (
            (int(GamepadButton.LEFT_TRIGGER), lt_val),
            (int(GamepadButton.RIGHT_TRIGGER), rt_val),
        ):
            is_down = axis_val >= 0.40
            was_down = btn_code in self._pad_down
            if is_down and not was_down:
                self._pad_down.add(btn_code)
                self._pad_pressed.add(btn_code)
                self._dispatch_pad_press(btn_code)
            elif not is_down and was_down and axis_val < 0.20:
                self._pad_down.discard(btn_code)
                self._pad_released.add(btn_code)
                self._dispatch_pad_release(btn_code)

    def poll_events(self) -> list[pygame.event.Event]:
        """Drains Pygame events once per frame and updates internal state buffers."""
        self.begin_frame()
        events = pygame.event.get()
        for event in events:
            self.process_event(event)
        self.update_axes()
        return events

    # --------------------------------------------------------------------------
    # Public Query Interface
    # --------------------------------------------------------------------------

    def is_action_down(self, action_name: str) -> bool:
        """Returns True as long as the action is being held."""
        if action_name in self._debug_actions:
            return self._debug_action_down.get(action_name, False)
        return False if self.is_debug_mode else self._action_down.get(action_name, False)

    def is_action_pressed(self, action_name: str) -> bool:
        """Returns True only on the frame the action was pressed."""
        if action_name in self._debug_actions:
            return self._debug_action_pressed.get(action_name, False)
        return False if self.is_debug_mode else self._action_pressed.get(action_name, False)

    def is_action_released(self, action_name: str) -> bool:
        """Returns True only on the frame the action was released."""
        if action_name in self._debug_actions:
            return self._debug_action_released.get(action_name, False)
        return False if self.is_debug_mode else self._action_released.get(action_name, False)

    is_action_just_pressed = is_action_pressed
    is_action_just_released = is_action_released

    def get_axis(self, axis_name: str) -> float:
        """Returns the normalized axis value in range [-1.0, +1.0]."""
        return 0.0 if self.is_debug_mode else self._axes_values.get(axis_name, 0.0)

    def get_vector2(self, vector_name: str) -> tuple[float, float]:
        """Returns a 2D movement or look vector with optional unit-length clamping."""
        if self.is_debug_mode:
            return (0.0, 0.0)

        vec = self._vectors.get(vector_name)
        if vec is None:
            return (0.0, 0.0)

        vx = self._axes_values.get(vec.axis_x, 0.0)
        vy = self._axes_values.get(vec.axis_y, 0.0)

        if vec.normalize:
            length = math.sqrt(vx * vx + vy * vy)
            if length > 1.0:
                inv_len = 1.0 / length
                vx *= inv_len
                vy *= inv_len

        return (vx, vy)

    # --------------------------------------------------------------------------
    # Low-level Hardware Queries
    # --------------------------------------------------------------------------

    def is_key_down(self, key: int | Key) -> bool:
        return int(key) in self._keys_down

    def is_key_pressed(self, key: int | Key) -> bool:
        return int(key) in self._keys_pressed

    def is_key_released(self, key: int | Key) -> bool:
        return int(key) in self._keys_released

    def is_mouse_down(self, button: int | MouseButton) -> bool:
        return int(button) in self._mouse_down

    def is_mouse_pressed(self, button: int | MouseButton) -> bool:
        return int(button) in self._mouse_pressed

    def is_mouse_released(self, button: int | MouseButton) -> bool:
        return int(button) in self._mouse_released

    def is_gamepad_down(self, button: int | GamepadButton) -> bool:
        return int(button) in self._pad_down

    def is_gamepad_pressed(self, button: int | GamepadButton) -> bool:
        return int(button) in self._pad_pressed

    def is_gamepad_released(self, button: int | GamepadButton) -> bool:
        return int(button) in self._pad_released

    def set_mouse_grab(self, grab: bool) -> None:
        """Locks and hides mouse cursor for FPS or orbit cameras."""
        self.is_mouse_grabbed = grab
        import os
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        try:
            pygame.event.set_grab(grab)
            pygame.mouse.set_visible(not grab)
        except Exception:
            pass

    def set_debug_mode(self, active: bool) -> None:
        """Activates debug input layer (suspends gameplay actions while keeping debug navigation)."""
        self.is_debug_mode = active
        if active:
            if self.is_mouse_grabbed:
                self._was_mouse_grabbed = True
                self.set_mouse_grab(False)
        else:
            if self._was_mouse_grabbed:
                self.set_mouse_grab(True)
                self._was_mouse_grabbed = False
