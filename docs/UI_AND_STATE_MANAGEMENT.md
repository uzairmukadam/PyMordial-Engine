# UI & Game State Management in PyMordial Engine

PyMordial Engine includes a lightweight, GPU-accelerated **Native ModernGL UI Subsystem** along with a deterministic, stack-based **Game State Management Subsystem**. Together, they allow developers to build professional arcade-style main menus, modal pause dialogues, graphics settings screens, and HUDs with zero third-party UI dependencies.

---

## 1. Native ModernGL UI Subsystem

The native UI subsystem (`engine.ui`) renders 2D vector primitives and text directly on top of the ModernGL backbuffer via instanced quad shaders and cached glyph rendering.

### Core Components

```
                   +---------------------------+
                   |         ProjectApp        |
                   +-------------+-------------+
                                 | (has a)
                                 v
                   +---------------------------+
                   |         UIManager         |
                   +-------------+-------------+
                                 | (manages)
                                 v
                   +---------------------------+
                   |       UIScreen Stack      |
                   +-------------+-------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
    +---------+             +---------+             +---------+
    | UIPanel |             | UIButton|             | UISlider|
    +---------+             +---------+             +---------+
```

| Class | Description |
|---|---|
| `UIManager` | Central coordinator attached to `app.ui`. Manages screen stack, event hit-testing, and rendering. |
| `UIScreen` | Base container representing a fullscreen page, menu, or modal dialog. |
| `UIPanel` | Rectangular container supporting vertical gradients, borders, and rounded corners. |
| `UILabel` | Antialiased text widget supporting custom font sizes, colors, and alignments. |
| `UIButton` | Interactive clickable button with hover, pressed states, variants (`primary`, `secondary`, `danger`), and audio feedback hooks. |
| `UISlider` | Draggable value slider with step increments and callback support. |
| `UISegmentGroup` | Multi-option tab or segment selector (e.g. Difficulty: Rookie / Pro / Legend). |
| `UIImage` | Texture blitter for icons, logos, and HUD badges. |
| `UIStyle` | Dataclass configuring colors, corner radii, borders, and paddings. |
| `UITheme` | Global aesthetic palette applied across UI elements. |

### Creating a Custom Screen

```python
from engine.ui import UIScreen, UIPanel, UILabel, UIButton, UIStyle

class MainMenuScreen(UIScreen):
    def __init__(self, on_start, on_quit):
        super().__init__(name="MainMenu")

        # Fullscreen gradient background
        bg = UIPanel(x=0, y=0, w=1280, h=720)
        bg.style = UIStyle(
            bg_color_top=(0.04, 0.06, 0.09, 0.95),
            bg_color_bottom=(0.02, 0.03, 0.05, 0.98),
            corner_radius=0.0,
        )
        self.add_child(bg)

        # Centered Container
        card = UIPanel(x=440, y=200, w=400, h=320)
        card.style = UIStyle(
            bg_color_top=(0.08, 0.10, 0.15, 0.90),
            bg_color_bottom=(0.05, 0.07, 0.10, 0.95),
            border_color=(0.18, 0.60, 0.90, 0.5),
            border_width=1.5,
            corner_radius=12.0,
        )
        bg.add_child(card)

        # Title Label
        title = UILabel("MY GAME", x=0, y=30, w=400, h=40, font_size=28, bold=True, align="center")
        card.add_child(title)

        # Start Button
        btn_start = UIButton(text="START", x=50, y=100, w=300, h=48, variant="primary", on_click=on_start)
        card.add_child(btn_start)

        # Quit Button
        btn_quit = UIButton(text="QUIT", x=50, y=170, w=300, h=48, variant="danger", on_click=on_quit)
        card.add_child(btn_quit)
```

---

## 2. Game State Management Subsystem

The Game State subsystem (`engine.core.state`) provides a stack-based finite state machine (FSM) attached to `app.state_manager`.

### Why Use a State Manager?

Without a state manager, games often end up with scattered boolean flags (`if not is_paused: ...`, `if is_in_menu: ...`) in update loops. PyMordial's `GameStateManager` provides:

1. **Automatic Simulation Gating**: Setting `allow_fixed_update = False` automatically halts the 60 Hz physics simulation and gameplay module ticks without manual checks in your modules.
2. **Modal Pause Stacking**: Pushing a `PausedState` automatically pauses the state underneath (`on_pause`) and resumes it (`on_resume`) when popped.
3. **Cursor and Grab Policies**: States automatically configure `show_cursor` and `grab_mouse` on enter/exit.
4. **Hierarchical Input Interception**: UI screens intercept input first, followed by the active `GameState`, and finally unhandled events fall through to gameplay modules.

### State Lifecycle Hooks

```python
from engine import GameState, ProjectApp
import pygame

class PlayingState(GameState):
    name = "Playing"
    allow_fixed_update = True    # Run 60Hz physics
    allow_variable_update = True # Run per-frame logic
    show_cursor = False          # Hide mouse pointer
    grab_mouse = True            # Lock mouse for FPS/chase look

    def on_enter(self, app: ProjectApp, prev_state: str | None = None, **kwargs) -> None:
        """Called when this state becomes active."""
        pass

    def on_pause(self, app: ProjectApp) -> None:
        """Called when another state is pushed on top (e.g. Pause Menu)."""
        pass

    def on_resume(self, app: ProjectApp) -> None:
        """Called when returning to this state after top state is popped."""
        pass

    def on_exit(self, app: ProjectApp, next_state: str | None = None) -> None:
        """Called when exiting this state."""
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        """Return True if this event was consumed and should not propagate further."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.push_state("Paused")
            return True
        return False
```

---

## 3. Coordinating Menu and Game States

Here is how `GameStateManager` and `UIManager` work together in harmony:

```python
from engine import ProjectApp, ProjectConfig, GameState

class MainMenuState(GameState):
    name = "MainMenu"
    allow_fixed_update = False  # Freeze physics
    show_cursor = True

    def on_enter(self, app: ProjectApp, prev_state=None, **kwargs):
        def start_game():
            app.state_manager.set_state("Playing")

        app.ui.set_screen(MainMenuScreen(on_start=start_game, on_quit=lambda: setattr(app, "is_running", False)))

    def on_exit(self, app: ProjectApp, next_state=None):
        app.ui.set_screen(None)


class PausedState(GameState):
    name = "Paused"
    allow_fixed_update = False  # Freeze physics while paused
    show_cursor = True

    def on_enter(self, app: ProjectApp, prev_state=None, **kwargs):
        def resume():
            app.state_manager.pop_state()

        def main_menu():
            app.state_manager.set_state("MainMenu")

        app.ui.set_screen(PauseMenuScreen(on_resume=resume, on_quit=main_menu))

    def on_exit(self, app: ProjectApp, next_state=None):
        app.ui.set_screen(None)

    def on_event(self, app: ProjectApp, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.state_manager.pop_state()
            return True
        return False
```

### Starting the Engine with States

```python
app = ProjectApp(config=ProjectConfig(title="My Game", width=1280, height=720))

app.state_manager.register(MainMenuState())
app.state_manager.register(PlayingState())
app.state_manager.register(PausedState())

# Boot directly into the main menu
app.state_manager.set_state("MainMenu")
app.run()
```

---

## 4. Examples in the Repository
 
Refer to the runnable standalone examples in the `examples/` directory:
 
- **[examples/01_modern_ui_and_menu.py](file:///d:/Projects/PyMordial-Engine/examples/01_modern_ui_and_menu.py)**: AAA-quality ModernGL glassmorphic UI cards, category badges, sliders (Volume, FOV), segmented buttons (Difficulty), and stack-based `GameStateManager` (Main Menu, Settings, Gameplay, Pause).
- **[examples/02_player_physics_and_cubes.py](file:///d:/Projects/PyMordial-Engine/examples/02_player_physics_and_cubes.py)**: 60 Hz kinematic `CharacterMotor` (WASD, sprint, jump), curb autostepping, dynamic Rapier3D physics cubes, collision-aware `FollowCamera`, in-game telemetry HUD, and pause menu.
- **[examples/03_full_graphics_and_display_menu.py](file:///d:/Projects/PyMordial-Engine/examples/03_full_graphics_and_display_menu.py)**: Complete in-game graphics and window settings menu (Resolution, Mode, VSync, Presets Low-Cinematic, SSAO/GTAO, Bloom, SSR, Fog, AA) with live player and physics scene.
