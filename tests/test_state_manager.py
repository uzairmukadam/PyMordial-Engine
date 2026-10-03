"""Unit Tests for PyMordial Engine Game State Management Subsystem."""

import pygame
from engine.core.state import GameState, GameStateManager
from engine.app.project_app import ProjectApp
from engine.app.config import ProjectConfig


class MockGameState(GameState):
    def __init__(self, name: str, allow_fixed: bool = True, allow_update: bool = True) -> None:
        self.name = name
        self.allow_fixed_update = allow_fixed
        self.allow_variable_update = allow_update
        self.entered = 0
        self.exited = 0
        self.paused = 0
        self.resumed = 0
        self.fixed_updates = 0
        self.updates = 0
        self.last_kwargs = {}
        self.last_prev_state = None
        self.last_next_state = None

    def on_enter(self, app, prev_state=None, **kwargs):
        self.entered += 1
        self.last_prev_state = prev_state
        self.last_kwargs = kwargs

    def on_exit(self, app, next_state=None):
        self.exited += 1
        self.last_next_state = next_state

    def on_pause(self, app):
        self.paused += 1

    def on_resume(self, app):
        self.resumed += 1

    def on_fixed_update(self, app, dt):
        self.fixed_updates += 1

    def on_update(self, app, dt):
        self.updates += 1

    def on_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return True
        return False


def test_state_registration_and_lookup():
    sm = GameStateManager()
    menu = MockGameState("Menu")
    game = MockGameState("Game")

    sm.register(menu)
    sm.register(game)

    assert sm.get_state("Menu") is menu
    assert sm.get_state("Game") is game
    assert sm.get_state("Unknown") is None
    assert sm.current_state is None
    assert sm.current_state_name is None
    assert sm.stack_depth == 0


def test_set_state_transition_lifecycle():
    sm = GameStateManager()
    menu = MockGameState("Menu")
    game = MockGameState("Game")
    sm.register(menu)
    sm.register(game)

    # Initial transition into Menu
    sm.set_state("Menu", difficulty="hard")
    assert sm.current_state_name == "Menu"
    assert sm.is_in_state("Menu") is True
    assert sm.is_in_state("Game") is False
    assert menu.entered == 1
    assert menu.last_prev_state is None
    assert menu.last_kwargs.get("difficulty") == "hard"

    # Transition from Menu to Game
    sm.set_state("Game", track_id="circuit_01")
    assert sm.current_state_name == "Game"
    assert menu.exited == 1
    assert menu.last_next_state == "Game"
    assert game.entered == 1
    assert game.last_prev_state == "Menu"
    assert game.last_kwargs.get("track_id") == "circuit_01"


def test_stack_push_pop_pause_resume():
    sm = GameStateManager()
    game = MockGameState("Playing")
    pause = MockGameState("Paused", allow_fixed=False)
    settings = MockGameState("Settings", allow_fixed=False)

    sm.register(game)
    sm.register(pause)
    sm.register(settings)

    # Enter base game state
    sm.set_state("Playing")
    assert sm.stack_depth == 1
    assert sm.current_state_name == "Playing"

    # Push Pause state (e.g. user presses ESC)
    sm.push_state("Paused")
    assert sm.stack_depth == 2
    assert sm.current_state_name == "Paused"
    assert game.paused == 1
    assert pause.entered == 1
    assert pause.last_prev_state == "Playing"

    # Push Settings modal on top of Pause menu
    sm.push_state("Settings")
    assert sm.stack_depth == 3
    assert sm.current_state_name == "Settings"
    assert pause.paused == 1
    assert settings.entered == 1

    # Pop Settings -> returns to Pause menu
    popped = sm.pop_state()
    assert popped is settings
    assert settings.exited == 1
    assert sm.stack_depth == 2
    assert sm.current_state_name == "Paused"
    assert pause.resumed == 1

    # Pop Pause -> returns to Playing
    popped = sm.pop_state()
    assert popped is pause
    assert pause.exited == 1
    assert sm.stack_depth == 1
    assert sm.current_state_name == "Playing"
    assert game.resumed == 1

    # Pop last state
    popped = sm.pop_state()
    assert popped is game
    assert game.exited == 1
    assert sm.stack_depth == 0
    assert sm.pop_state() is None


def test_event_dispatching():
    sm = GameStateManager()
    game = MockGameState("Game")
    sm.register(game)
    sm.set_state("Game")

    # Space key event should be consumed
    space_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
    assert sm.handle_event(space_ev) is True

    # Return key event should not be consumed
    enter_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
    assert sm.handle_event(enter_ev) is False


def test_fixed_and_variable_update_gating():
    sm = GameStateManager()
    paused = MockGameState("Paused", allow_fixed=False, allow_update=True)
    sm.register(paused)
    sm.set_state("Paused")

    # Fixed update should NOT be executed because allow_fixed_update=False
    sm.fixed_update(1.0 / 60.0)
    assert paused.fixed_updates == 0

    # Variable update SHOULD be executed because allow_variable_update=True
    sm.update(0.016)
    assert paused.updates == 1


def test_project_app_state_manager_integration():
    """Verify state manager functions seamlessly inside a running ProjectApp."""
    cfg = ProjectConfig(
        title="State Test",
        width=640,
        height=480,
        headless=True,
        enable_debug=False,
    )
    app = ProjectApp(config=cfg)

    menu = MockGameState("MenuState", allow_fixed=False)
    play = MockGameState("PlayState", allow_fixed=True)

    app.state_manager.register(menu)
    app.state_manager.register(play)

    # 1. Start in MenuState: physics is paused
    app.state_manager.set_state("MenuState")
    assert app.state_manager.is_in_state("MenuState") is True

    # Step frame: menu variable update called, fixed simulation skipped
    app.step_frame(1.0 / 60.0)
    assert menu.updates == 1
    assert menu.fixed_updates == 0

    # 2. Transition into PlayState: physics and fixed update enabled
    app.state_manager.set_state("PlayState")
    assert app.state_manager.is_in_state("PlayState") is True
    assert menu.exited == 1
    assert play.entered == 1

    app.step_frame(1.0 / 60.0)
    assert play.updates == 1
    assert play.fixed_updates >= 1

    # 3. Push Pause overlay: fixed update frozen
    pause = MockGameState("PauseOverlay", allow_fixed=False)
    app.state_manager.register(pause)
    app.state_manager.push_state("PauseOverlay")
    assert play.paused == 1
    assert pause.entered == 1

    initial_play_fixed = play.fixed_updates
    app.step_frame(1.0 / 60.0)
    assert play.fixed_updates == initial_play_fixed  # Frozen!
    assert pause.updates == 1

    # 4. Clean shutdown
    app.shutdown()
    assert pause.exited == 1
    assert play.exited == 1
    assert app.state_manager.stack_depth == 0
