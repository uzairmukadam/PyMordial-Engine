"""Standardized Hardware-Agnostic Input Codes for PyMordial Engine.

Defines symbolic identifiers for Keyboard, Mouse, and Gamepad inputs so gameplay
and engine code never need to directly import or couple to raw Pygame constants.
"""

from __future__ import annotations
from enum import IntEnum
import pygame


class DeviceType(IntEnum):
    """Active user input device type."""

    KEYBOARD_MOUSE = 1
    GAMEPAD = 2


class Key(IntEnum):
    """Standardized keyboard key codes matching Pygame key values."""

    # Letters
    A = pygame.K_a
    B = pygame.K_b
    C = pygame.K_c
    D = pygame.K_d
    E = pygame.K_e
    F = pygame.K_f
    G = pygame.K_g
    H = pygame.K_h
    I = pygame.K_i  # noqa: E741
    J = pygame.K_j
    K = pygame.K_k
    L = pygame.K_l
    M = pygame.K_m
    N = pygame.K_n
    O = pygame.K_o  # noqa: E741
    P = pygame.K_p
    Q = pygame.K_q
    R = pygame.K_r
    S = pygame.K_s
    T = pygame.K_t
    U = pygame.K_u
    V = pygame.K_v
    W = pygame.K_w
    X = pygame.K_x
    Y = pygame.K_y
    Z = pygame.K_z

    # Numbers
    NUM_0 = pygame.K_0
    NUM_1 = pygame.K_1
    NUM_2 = pygame.K_2
    NUM_3 = pygame.K_3
    NUM_4 = pygame.K_4
    NUM_5 = pygame.K_5
    NUM_6 = pygame.K_6
    NUM_7 = pygame.K_7
    NUM_8 = pygame.K_8
    NUM_9 = pygame.K_9

    # Function Keys
    F1 = pygame.K_F1
    F2 = pygame.K_F2
    F3 = pygame.K_F3
    F4 = pygame.K_F4
    F5 = pygame.K_F5
    F6 = pygame.K_F6
    F7 = pygame.K_F7
    F8 = pygame.K_F8
    F9 = pygame.K_F9
    F10 = pygame.K_F10
    F11 = pygame.K_F11
    F12 = pygame.K_F12

    # Modifiers & Navigation
    SPACE = pygame.K_SPACE
    RETURN = pygame.K_RETURN
    ESCAPE = pygame.K_ESCAPE
    TAB = pygame.K_TAB
    BACKSPACE = pygame.K_BACKSPACE
    LSHIFT = pygame.K_LSHIFT
    RSHIFT = pygame.K_RSHIFT
    LCTRL = pygame.K_LCTRL
    RCTRL = pygame.K_RCTRL
    LALT = pygame.K_LALT
    RALT = pygame.K_RALT
    GRAVE = pygame.K_BACKQUOTE  # ` or ~ (Console / Debug)
    MINUS = pygame.K_MINUS
    EQUALS = pygame.K_EQUALS

    # Arrow Keys
    UP = pygame.K_UP
    DOWN = pygame.K_DOWN
    LEFT = pygame.K_LEFT
    RIGHT = pygame.K_RIGHT


class MouseButton(IntEnum):
    """Standardized mouse buttons."""

    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    EXTRA_1 = 4
    EXTRA_2 = 5


class GamepadButton(IntEnum):
    """Standardized SDL/XInput game controller buttons."""

    A = 0               # Bottom Face Button (Cross on PlayStation)
    B = 1               # Right Face Button (Circle on PlayStation)
    X = 2               # Left Face Button (Square on PlayStation)
    Y = 3               # Top Face Button (Triangle on PlayStation)
    BACK = 4            # Select / View / Share
    GUIDE = 5           # Xbox / PS Button
    START = 6           # Menu / Options
    LEFT_STICK = 7      # L3 (Stick press)
    RIGHT_STICK = 8     # R3 (Stick press)
    LEFT_BUMPER = 9     # LB / L1
    RIGHT_BUMPER = 10   # RB / R1
    DPAD_UP = 11        # Directional Pad Up
    DPAD_DOWN = 12      # Directional Pad Down
    DPAD_LEFT = 13      # Directional Pad Left
    DPAD_RIGHT = 14     # Directional Pad Right
    LEFT_TRIGGER = 15   # LT / L2 (Thresholded Trigger Button)
    RIGHT_TRIGGER = 16  # RT / R2 (Thresholded Trigger Button)


class GamepadAxis(IntEnum):
    """Standardized SDL/XInput game controller analog axes."""

    LEFT_STICK_X = 0    # Left Stick Horizontal (-1.0 Left, +1.0 Right)
    LEFT_STICK_Y = 1    # Left Stick Vertical (-1.0 Up, +1.0 Down)
    RIGHT_STICK_X = 2   # Right Stick Horizontal (-1.0 Left, +1.0 Right)
    RIGHT_STICK_Y = 3   # Right Stick Vertical (-1.0 Up, +1.0 Down)
    LEFT_TRIGGER = 4    # Left Trigger LT / L2 (0.0 to 1.0 or -1.0 to 1.0)
    RIGHT_TRIGGER = 5   # Right Trigger RT / R2 (0.0 to 1.0 or -1.0 to 1.0)
