# PyMordial Engine: Developer Guide & Project Workflow

Welcome to PyMordial Engine! This guide explains the development workflow, how to create and structure game projects, write reusable modules, build worlds, and package standalone distribution releases.

---

## 1. Architecture Philosophy: Engine vs. Projects

PyMordial strictly isolates engine infrastructure from game-specific code:
```
PyMordial-Engine/
├── engine/                       # Core engine framework (Rendering, ECS, Physics, Audio, VFS)
│   ├── app/                      # Project host, configuration, and module contracts
│   ├── world/                    # World builder interfaces and default builders
│   └── tools/                    # Asset cooker and release packagers
├── projects/                     # Standalone game projects directory
│   ├── shotgun_beat_the_speed/  # Example project
│   │   ├── modules/              # Self-contained, portable game feature modules
│   │   ├── world/                # Project-specific level and world builders
│   │   ├── assets/               # Project-local textures, audio, meshes
│   │   ├── config.py             # Project configuration
│   │   └── main.py               # Project entry point
└── tests/                        # Engine unit and integration test suite
```

Key Benefits:
- **Clean Engine Core**: The engine remains lean, fast, and unpolluted by gameplay scripts.
- **Portable Modules**: Modules (`character_motor`, `day_night_cycle`, `heat_system`, etc.) are 100% self-contained folders that can be copy-pasted into any other project.
- **Simple Upgrades**: Updating the engine never breaks project directories or requires modifying game logic.

---

## 2. Creating a New Game Project

To create a new game, add a directory inside `projects/<project_name>/`:

```
projects/my_game/
├── modules/
│   └── __init__.py
├── world/
│   ├── __init__.py
│   └── level.py
├── assets/
├── config.py
└── main.py
```

### Step 1: Define `config.py`
Declare display, physics, and quality presets:
```python
from engine.app.config import ProjectConfig

def get_project_config() -> ProjectConfig:
    return ProjectConfig(
        title="My Epic Game",
        width=1920,
        height=1080,
        fullscreen=False,
        vsync=True,
        quality_preset="ultra",  # "low", "medium", "high", "ultra", "cinematic"
        fixed_hz=60.0,
        gravity=(0.0, -20.0, 0.0),
        enable_debug=True,
    )
```

### Step 2: Implement World Building
You can use the engine's built-in `DefaultWorldBuilder`:
```python
from engine.world.default_world import DefaultWorldBuilder

builder = DefaultWorldBuilder(size=150.0, material_name="mud_cracked_dry_03")
```
Or define your own custom level generator by inheriting from `BaseWorldBuilder`:
```python
from engine.world.base import BaseWorldBuilder
from engine.app.project_app import ProjectApp

class DungeonWorldBuilder(BaseWorldBuilder):
    def build_world(self, app: ProjectApp) -> None:
        # Spawn rooms, obstacles, and colliders
        pass

    def teardown_world(self, app: ProjectApp) -> None:
        # Cleanup entities
        pass
```

### Step 3: Implement `main.py`
Instantiate `ProjectApp`, attach world builder, add modules, and run:
```python
from engine.app.project_app import ProjectApp
from projects.my_game.config import get_project_config
from projects.my_game.world.level import DungeonWorldBuilder
from projects.my_game.modules.character_motor import CharacterMotorModule

def main() -> None:
    config = get_project_config()
    game = ProjectApp(config)
    game.set_world_builder(DungeonWorldBuilder())
    game.add_module(CharacterMotorModule())
    game.run()

if __name__ == "__main__":
    main()
```

---

## 3. Writing Reusable Game Modules

All game features (weapons, inventories, vehicles, weather, AI) should be implemented as `ProjectModule` plugins:

```python
from engine.app.module import ProjectModule
from engine.app.project_app import ProjectApp
import pygame

class JumpPadModule(ProjectModule):
    def __init__(self, jump_force: float = 15.0) -> None:
        super().__init__(name="jump_pad")
        self.jump_force = jump_force

    def on_attach(self, app: ProjectApp) -> None:
        """Called once when module is registered to ProjectApp."""
        print("Jump pad initialized!")

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        """Called at fixed simulation frequency (e.g. 60Hz) alongside physics."""
        # Query trigger collider overlaps and apply velocity impulses
        pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        """Called once per visual frame with variable delta time."""
        # Update animations or camera effects
        pass

    def on_ui(self, app: ProjectApp) -> None:
        """Called during HUD rendering pass."""
        # Draw 2D UI elements
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        """Called for hardware input events. Return True if consumed."""
        return False

    def on_detach(self, app: ProjectApp) -> None:
        """Called when project shuts down to free resources."""
        pass
```

### Module Best Practices
1. **Self-Containment**: A module should reside in its own folder (`projects/<project>/modules/<module_name>/`). Keep assets or sub-scripts inside the module folder if they are specific to that module.
2. **Copy-Paste Portability**: Any module should be droppable into another project with zero code edits.
3. **Zero-Allocation Inner Loop**: In `on_fixed_update` and `on_update`, avoid creating temporary lists, dicts, or tuples. Use pre-allocated NumPy buffers or scalar variables.

---

## 4. Running & Testing Projects

### Development Mode (Interactive)
```powershell
python projects/shotgun_beat_the_speed/main.py
```

### Headless Simulation (CI / Automated Testing)
Run a fixed number of frames without opening a window or GPU swap:
```powershell
python projects/shotgun_beat_the_speed/main.py --headless --frames 60
```

### Graphical Presets & Release Flags
```powershell
python projects/shotgun_beat_the_speed/main.py --preset medium --release
```

---

## 5. Packaging & Distributing Releases

PyMordial includes a dual-target release packager (`engine/tools/packager.py`) that strips debug tools, tree-shakes used modules, and packs game assets into a zero-copy `game.pak` container.

### Target 1: PyInstaller (Beginner-Friendly, Zero C++ Setup)
Best for rapid distribution without installing Visual Studio C++ build tools:
```powershell
python -m engine.tools.packager --project shotgun_beat_the_speed --target pyinstaller --release
```

### Target 2: Nuitka (Native C++ Optimized Binary)
Best for production releases, native C++ performance, and code obfuscation:
```powershell
python -m engine.tools.packager --project shotgun_beat_the_speed --target nuitka --release
```

### Dry Run (Inspect build staging and flags)
```powershell
python -m engine.tools.packager --project shotgun_beat_the_speed --dry-run
```
Output artifact location defaults to `dist/<project_name>/`.

---

## 6. Central Debug Subsystem & Non-Interference Rules

PyMordial provides a built-in 3-tier glassmorphic Dear ImGui developer overlay:
* **F1**: System Monitor (Real-time CPU/GPU frame times, FPS history, 1% low metrics, memory allocator gauges).
* **F2**: Engine Graphics Tweaks (Direct G-Buffer visualization, tonemapping operators, wireframe overlay, shadow cascade inspection).
* **F3**: Game Tweaks (Dynamic developer action and property inspector).

### Developer Rule: Never Override Debug Keys
The engine is intentionally kept light so it provides only foundational primitives without interfering with game design. In return:
- Games must **NEVER** intercept, remap, or consume `F1`, `F2`, or `F3`.
- Gameplay camera switching must use **`V`** or **`F10`**.
- Abilities or menus must use gameplay keys (`Tab`, `Esc`, `M`, `C`, etc.).
- When building in release mode (`--release`), the packager automatically tree-shakes and strips `debug_overlay`, ensuring zero debug code remains in the final distribution.

