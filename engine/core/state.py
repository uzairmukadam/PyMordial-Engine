"""PyMordial Game State Management Subsystem.

Provides stack-based finite state machine (FSM) coordination for managing transitions
between Game, Menu, Loading, and Paused states with deterministic lifecycle hooks.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import pygame

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


class GameState:
    """Base class for all distinct game execution states (e.g. Menu, Playing, Paused)."""

    name: str = "GameState"

    # Execution policy flags
    allow_fixed_update: bool = True    # When False, halts 60Hz physics/kinematic simulation
    allow_variable_update: bool = True # When False, halts per-frame game module updates
    allow_render: bool = True          # When False, skips 3D scene rendering
    show_cursor: bool = False          # Automatically configures pygame.mouse.set_visible
    grab_mouse: bool = False           # Automatically configures app.input_manager.set_mouse_grab

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs: Any) -> None:
        """Called when this state becomes active (transitioned or pushed into)."""
        pass

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        """Called when this state is exited or popped."""
        pass

    def on_pause(self, app: ProjectApp) -> None:
        """Called when a new state is pushed on top of this state in the stack."""
        pass

    def on_resume(self, app: ProjectApp) -> None:
        """Called when a state above this one is popped and this state becomes active again."""
        pass

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        """Fixed-rate logic update for this state."""
        pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        """Variable-rate frame update for this state."""
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        """Event handler for this state. Return True if consumed."""
        return False


class GameStateManager:
    """Stack-based Finite State Machine coordinating application states."""

    __slots__ = ("_app", "_registered_states", "_state_stack")

    def __init__(self, app: ProjectApp | None = None) -> None:
        self._app: ProjectApp | None = app
        self._registered_states: dict[str, GameState] = {}
        self._state_stack: list[GameState] = []

    def attach_app(self, app: ProjectApp) -> None:
        """Binds the state manager to an active ProjectApp instance."""
        self._app = app

    @property
    def current_state(self) -> GameState | None:
        """Returns the currently active state on top of the stack."""
        return self._state_stack[-1] if self._state_stack else None

    @property
    def current_state_name(self) -> str | None:
        """Returns the name of the active state, or None if stack is empty."""
        top = self.current_state
        return top.name if top is not None else None

    @property
    def stack_depth(self) -> int:
        """Returns the number of states on the stack."""
        return len(self._state_stack)

    def register(self, state: GameState, name: str | None = None) -> GameState:
        """Registers a state instance by name."""
        state_name = name or state.name
        self._registered_states[state_name] = state
        return state

    def get_state(self, name: str) -> GameState | None:
        """Retrieves a registered state by name."""
        return self._registered_states.get(name)

    def is_in_state(self, name: str) -> bool:
        """Checks if the currently active state matches the given name."""
        top = self.current_state
        return top is not None and top.name == name

    def set_state(self, name_or_state: str | GameState, **kwargs: Any) -> GameState:
        """Clears the stack and transitions into the target state."""
        state = self._resolve_state(name_or_state)
        prev_name = self.current_state_name

        while self._state_stack:
            popped = self._state_stack.pop()
            popped.on_exit(self._app, next_state=state.name)

        self._state_stack.append(state)
        self._apply_cursor_policy(state)
        state.on_enter(self._app, prev_state=prev_name, **kwargs)
        return state

    def push_state(self, name_or_state: str | GameState, **kwargs: Any) -> GameState:
        """Pushes an overlay/modal state on top of the current state, pausing the previous one."""
        state = self._resolve_state(name_or_state)
        current = self.current_state
        if current is not None:
            current.on_pause(self._app)

        self._state_stack.append(state)
        self._apply_cursor_policy(state)
        state.on_enter(self._app, prev_state=current.name if current else None, **kwargs)
        return state

    def pop_state(self) -> GameState | None:
        """Pops the active state and resumes the previous state underneath."""
        if not self._state_stack:
            return None

        popped = self._state_stack.pop()
        next_top = self.current_state
        popped.on_exit(self._app, next_state=next_top.name if next_top else None)

        if next_top is not None:
            self._apply_cursor_policy(next_top)
            next_top.on_resume(self._app)

        return popped

    def _apply_cursor_policy(self, state: GameState) -> None:
        if self._app and not self._app.config.headless:
            # Always hide OS cursor; in-engine AAA cursor handles UI visual representation
            pygame.mouse.set_visible(False)
            self._app.input_manager.set_mouse_grab(state.grab_mouse)
            if self._app.ui is not None:
                self._app.ui.cursor_visible = state.show_cursor

    def _resolve_state(self, name_or_state: str | GameState) -> GameState:
        if isinstance(name_or_state, str):
            if name_or_state not in self._registered_states:
                raise KeyError(f"GameState '{name_or_state}' is not registered.")
            return self._registered_states[name_or_state]
        return name_or_state

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Dispatches an SDL2/PyGame event to the active state. Returns True if consumed."""
        top = self.current_state
        if top is not None:
            return top.on_event(self._app, event)
        return False

    def fixed_update(self, dt: float) -> None:
        """Dispatches fixed-rate simulation update to active state if allowed."""
        top = self.current_state
        if top is not None and top.allow_fixed_update:
            top.on_fixed_update(self._app, dt)

    def update(self, dt: float) -> None:
        """Dispatches variable-rate frame update to active state if allowed."""
        top = self.current_state
        if top is not None and top.allow_variable_update:
            top.on_update(self._app, dt)
