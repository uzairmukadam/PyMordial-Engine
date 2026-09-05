"""Zero-allocation Semantic Input System for PyMordial Engine.

Wraps pygame-ce events and device states into semantic actions and axes
without per-frame allocations.
"""

from __future__ import annotations
import pygame


class InputManager:
    """Manages semantic actions, axes, keyboard and mouse inputs without per-frame allocations."""

    __slots__ = (
        "_actions_down",
        "_actions_pressed",
        "_actions_released",
        "_action_key_bindings",
        "_action_mouse_bindings",
        "_axes_values",
        "_axis_bindings",
        "mouse_pos",
        "mouse_rel",
        "mouse_wheel",
        "_should_quit",
    )

    def __init__(self) -> None:
        self._actions_down: dict[str, bool] = {}
        self._actions_pressed: dict[str, bool] = {}
        self._actions_released: dict[str, bool] = {}
        self._action_key_bindings: dict[str, list[int]] = {}
        self._action_mouse_bindings: dict[str, list[int]] = {}

        self._axes_values: dict[str, float] = {}
        # (pos_key, neg_key)
        self._axis_bindings: dict[str, tuple[int, int]] = {}

        self.mouse_pos = [0, 0]
        self.mouse_rel = [0, 0]
        self.mouse_wheel = 0.0
        self._should_quit = False

        self._register_default_bindings()

    def _register_default_bindings(self) -> None:
        """Sets up default standard movement and gameplay action bindings."""
        self.bind_action("jump", [pygame.K_SPACE])
        self.bind_action("sprint", [pygame.K_LSHIFT, pygame.K_RSHIFT])
        self.bind_action("crouch", [pygame.K_LCTRL, pygame.K_c])
        self.bind_action("interact", [pygame.K_e])
        self.bind_action("pause", [pygame.K_ESCAPE])
        self.bind_action("primary_click", mouse_buttons=[1])
        self.bind_action("secondary_click", mouse_buttons=[3])

        self.bind_axis("move_forward", pos_key=pygame.K_w, neg_key=pygame.K_s)
        self.bind_axis("move_right", pos_key=pygame.K_d, neg_key=pygame.K_a)
        self.bind_axis("move_up", pos_key=pygame.K_e, neg_key=pygame.K_q)

    def bind_action(
        self,
        name: str,
        keys: list[int] | None = None,
        mouse_buttons: list[int] | None = None,
    ) -> None:
        """Registers a named semantic action."""
        self._actions_down[name] = False
        self._actions_pressed[name] = False
        self._actions_released[name] = False
        if keys:
            self._action_key_bindings[name] = list(keys)
        if mouse_buttons:
            self._action_mouse_bindings[name] = list(mouse_buttons)

    def bind_axis(self, name: str, pos_key: int, neg_key: int) -> None:
        """Registers a digital 1D axis mapped to positive and negative keys."""
        self._axes_values[name] = 0.0
        self._axis_bindings[name] = (pos_key, neg_key)

    def poll_events(self) -> None:
        """Polls pygame event queue once per frame and updates internal state buffers."""
        # Reset one-frame transition flags in-place
        for name in self._actions_pressed:
            self._actions_pressed[name] = False
        for name in self._actions_released:
            self._actions_released[name] = False

        self.mouse_rel[0] = 0
        self.mouse_rel[1] = 0
        self.mouse_wheel = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True

            elif event.type == pygame.KEYDOWN:
                for action_name, keys in self._action_key_bindings.items():
                    if event.key in keys:
                        if not self._actions_down[action_name]:
                            self._actions_pressed[action_name] = True
                        self._actions_down[action_name] = True

            elif event.type == pygame.KEYUP:
                for action_name, keys in self._action_key_bindings.items():
                    if event.key in keys:
                        self._actions_down[action_name] = False
                        self._actions_released[action_name] = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                for action_name, buttons in self._action_mouse_bindings.items():
                    if event.button in buttons:
                        if not self._actions_down[action_name]:
                            self._actions_pressed[action_name] = True
                        self._actions_down[action_name] = True

            elif event.type == pygame.MOUSEBUTTONUP:
                for action_name, buttons in self._action_mouse_bindings.items():
                    if event.button in buttons:
                        self._actions_down[action_name] = False
                        self._actions_released[action_name] = True

            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos[0] = event.pos[0]
                self.mouse_pos[1] = event.pos[1]
                self.mouse_rel[0] += event.rel[0]
                self.mouse_rel[1] += event.rel[1]

            elif event.type == pygame.MOUSEWHEEL:
                self.mouse_wheel += event.y

        # Update digital axes from keyboard state
        keys_state = pygame.key.get_pressed()
        for axis_name, (pos_k, neg_k) in self._axis_bindings.items():
            val = 0.0
            if keys_state[pos_k]:
                val += 1.0
            if keys_state[neg_k]:
                val -= 1.0
            self._axes_values[axis_name] = val

    def is_action_down(self, action_name: str) -> bool:
        """Returns True as long as the action is being held down."""
        return self._actions_down.get(action_name, False)

    def is_action_just_pressed(self, action_name: str) -> bool:
        """Returns True only on the initial frame the action was pressed."""
        return self._actions_pressed.get(action_name, False)

    def is_action_just_released(self, action_name: str) -> bool:
        """Returns True only on the frame the action was released."""
        return self._actions_released.get(action_name, False)

    def get_axis(self, axis_name: str) -> float:
        """Returns the normalized axis value (-1.0 to 1.0)."""
        return self._axes_values.get(axis_name, 0.0)

    @property
    def should_quit(self) -> bool:
        return self._should_quit
