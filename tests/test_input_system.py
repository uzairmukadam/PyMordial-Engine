"""Unit tests for the dedicated AAA-Style Input System."""

import pygame
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


class TestInputSystem:
    """Comprehensive test suite for PyMordial Engine Input Subsystem."""

    def test_codes_and_bindings(self):
        """Validates hardware enum codes and binding objects."""
        assert Key.W == pygame.K_w
        assert Key.SPACE == pygame.K_SPACE
        assert MouseButton.LEFT == 1
        assert GamepadButton.A == 0
        assert GamepadAxis.LEFT_STICK_X == 0

        action = ActionBinding(name="jump").add_keys(Key.SPACE).add_gamepad_buttons(GamepadButton.A)
        assert Key.SPACE in action.keys
        assert GamepadButton.A in action.gamepad_buttons

        axis = AxisBinding(name="move_x", pos_key=Key.D, neg_key=Key.A, deadzone=0.2)
        assert axis.pos_key == Key.D
        assert axis.deadzone == 0.2

        vec = Vector2Binding(name="move", axis_x="move_x", axis_y="move_y", normalize=True)
        assert vec.normalize is True

    def test_action_edge_detection(self):
        """Validates pressed, held (down), and released edge detection."""
        input_mgr = InputManager(register_defaults=False)
        input_mgr.bind_action("test_jump", keys=[Key.SPACE], gamepad_buttons=[GamepadButton.A])

        # Initially inactive
        assert not input_mgr.is_action_down("test_jump")
        assert not input_mgr.is_action_pressed("test_jump")
        assert not input_mgr.is_action_released("test_jump")

        # 1. Simulate KEYDOWN
        down_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        input_mgr.begin_frame()
        input_mgr.process_event(down_ev)
        assert input_mgr.is_action_pressed("test_jump")
        assert input_mgr.is_action_down("test_jump")
        assert not input_mgr.is_action_released("test_jump")

        # 2. Next frame: still held down, no longer 'pressed' (edge consumed)
        input_mgr.begin_frame()
        assert not input_mgr.is_action_pressed("test_jump")
        assert input_mgr.is_action_down("test_jump")
        assert not input_mgr.is_action_released("test_jump")

        # 3. Simulate KEYUP
        up_ev = pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE)
        input_mgr.process_event(up_ev)
        assert not input_mgr.is_action_down("test_jump")
        assert input_mgr.is_action_released("test_jump")

        # 4. Next frame: fully inactive
        input_mgr.begin_frame()
        assert not input_mgr.is_action_down("test_jump")
        assert not input_mgr.is_action_released("test_jump")

    def test_gamepad_dual_device_actions(self):
        """Validates gamepad button events and active device switching."""
        input_mgr = InputManager(register_defaults=False)
        input_mgr.bind_action("fire", keys=[Key.SPACE], gamepad_buttons=[GamepadButton.A])

        assert input_mgr.active_device == DeviceType.KEYBOARD_MOUSE

        # Simulate Gamepad button press
        joy_ev = pygame.event.Event(pygame.JOYBUTTONDOWN, button=GamepadButton.A, instance_id=0)
        input_mgr.begin_frame()
        input_mgr.process_event(joy_ev)

        assert input_mgr.active_device == DeviceType.GAMEPAD
        assert input_mgr.is_action_pressed("fire")
        assert input_mgr.is_action_down("fire")

        # Release Gamepad button
        joy_up = pygame.event.Event(pygame.JOYBUTTONUP, button=GamepadButton.A, instance_id=0)
        input_mgr.process_event(joy_up)
        assert not input_mgr.is_action_down("fire")
        assert input_mgr.is_action_released("fire")

    def test_vector2_axis_normalization(self):
        """Validates 1D axes calculation and 2D diagonal magnitude clamping."""
        input_mgr = InputManager(register_defaults=False)
        input_mgr.bind_axis("x", pos_key=Key.D, neg_key=Key.A)
        input_mgr.bind_axis("y", pos_key=Key.W, neg_key=Key.S)
        input_mgr.bind_vector2("move", axis_x="x", axis_y="y", normalize=True)

        # Press W and D simultaneously (diagonal movement)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        input_mgr.update_axes()

        assert input_mgr.get_axis("x") == 1.0
        assert input_mgr.get_axis("y") == 1.0

        vec = input_mgr.get_vector2("move")
        # Magnitude without normalization is sqrt(2) = 1.414; with normalization it must be <= 1.0
        mag = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        assert abs(mag - 1.0) < 1e-4
        assert abs(vec[0] - 0.7071) < 1e-3
        assert abs(vec[1] - 0.7071) < 1e-3

    def test_mouse_dynamics_and_wheel(self):
        """Validates mouse motion delta accumulation and scroll wheel."""
        input_mgr = InputManager(register_defaults=False)

        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(150, 200), rel=(10, -5)))
        input_mgr.process_event(pygame.event.Event(pygame.MOUSEWHEEL, y=2.0))

        assert input_mgr.mouse_pos == (150, 200)
        assert input_mgr.mouse_delta == (10, -5)
        assert input_mgr.mouse_wheel == 2.0

        # Next frame clears relative deltas
        input_mgr.begin_frame()
        assert input_mgr.mouse_delta == (0, 0)
        assert input_mgr.mouse_wheel == 0.0

    def test_debug_mode_interception(self):
        """Validates that debug mode suppresses gameplay actions while keeping debug navigation."""
        input_mgr = InputManager(register_defaults=False)
        input_mgr.bind_action("gameplay_jump", keys=[Key.SPACE])
        input_mgr.bind_action("debug_toggle", keys=[Key.F1], is_debug=True)

        # When debug mode is active
        input_mgr.set_debug_mode(True)
        assert input_mgr.is_debug_mode is True

        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))

        # Gameplay action must be suppressed
        assert not input_mgr.is_action_pressed("gameplay_jump")
        assert not input_mgr.is_action_down("gameplay_jump")

        # Debug action must fire
        assert input_mgr.is_action_pressed("debug_toggle")
        assert input_mgr.is_action_down("debug_toggle")

        # Disable debug mode
        input_mgr.set_debug_mode(False)
        assert input_mgr.is_debug_mode is False
