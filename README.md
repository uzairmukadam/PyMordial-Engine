# PyMordial Engine (v0.1.0 Beta)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ModernGL](https://img.shields.io/badge/Graphics-ModernGL%204.5%20Core-orange.svg)](https://github.com/moderngl/moderngl)
[![Rapier3D](https://img.shields.io/badge/Physics-Rapier3D%20(Rust)-red.svg)](https://rapier.rs/)
[![Pygame-CE](https://img.shields.io/badge/Window-Pygame--CE-yellow.svg)](https://pyga.me/)

**PyMordial Engine** is a high-performance, modular 3D game engine engineered from the ground up for Python 3.11+, ModernGL (OpenGL 4.5 Core Profile), native Rapier3D physics (Rust), and Pygame-CE.

Designed to bridge the gap between high-level Python productivity and AAA-grade visual fidelity, PyMordial combines an expressive component and module lifecycle with a zero-allocation, contiguous data-oriented execution core.

---

## Key Architectural Highlights

- **Zero-Allocation Hot Paths**: All transforms, physics states, and material indices reside in contiguous C-aligned NumPy Structured Arrays that stream directly into OpenGL Shader Storage Buffer Objects (SSBOs) without creating per-frame Python objects.
- **Hardware-Accelerated Multi-Draw Indirect (MDI)**: Static and dynamic geometries are coalesced into a unified GPU `MegaBuffer`, submitting entire scenes in minimal draw calls.
- **Reversed-Z Floating Point Depth**: 32-bit floating-point depth testing with `GL_GREATER` eliminates Z-fighting across vast outdoor landscapes.
- **Cascaded Shadow Maps (CSM)**: Up to 4 shadow cascades with logarithmic/linear split blending, sub-texel stabilization, and 16-tap Poisson disc PCF filtering.
- **Comprehensive Physical Lighting & Post-Processing**:
  - Clustered Forward+ local point and spot lights (frustum 3D cluster binning).
  - Ground Truth Ambient Occlusion (GTAO), SSAO, and HBAO with bilateral blur.
  - Screen-Space Global Illumination (SSGI) and Screen-Space Reflections (SSR).
  - Volumetric Fog with anisotropic participating media.
  - Physically based Rayleigh and Mie atmospheric scattering (dynamic day-night cycle).
  - Anti-Aliasing suite: Temporal Anti-Aliasing (TAA with YCoCg clamping), SMAA 1X/2X/4X, and FXAA 3.11.
  - ACES, AgX, Reinhard, and filmic tonemapping with camera optics (bokeh depth of field, bloom).
- **Synchronous Rapier3D Physics (Rust)**:
  - In-thread stepping with GIL release to multi-threaded Rayon workers.
  - Kinematic Character Controller (KCC) with slope limits, autostep, and non-penetrating sweeps.
  - Advanced Raycast Vehicle system with suspension springs, anti-roll bars, and Pacejka combined tire slip dynamics.
- **3D Spatialized Audio & Procedural Synthesis**:
  - Distance attenuation (Linear, Logarithmic, Inverse Clamped) and stereo panning relative to camera orientation.
  - Built-in procedural audio synthesizer for dynamic footsteps, impacts, engine tones, jumps, and environmental wind.
- **Native ModernGL 2D UI Subsystem**:
  - High-performance signed-distance-field (SDF) rounded rectangle batch renderer with dropshadows, borders, gradients, and font typography caching.
- **Virtual File System (VFS) & Offline Asset Cooker**:
  - Zero-copy memory-mapped `.pak` container with 64-bit FNV-1a hash table for O(1) asset resolution.
  - CLI cooker compiles `.obj`/`.glb` into 32-byte cache-aligned `.pm_mesh` files and images into `.pm_tex`.
- **Dual-Target Standalone Packaging**:
  - Single-command distribution bundling via PyInstaller (zero-compiler one-dir) or Nuitka (native Ahead-of-Time C++ compilation).
  - Automatic module tree-shaking and release asset packing.

---

## Repository Structure

```
PyMordial-Engine/
├── engine/                       # Core engine systems (Rendering, Physics, ECS, Audio, VFS)
│   ├── app/                      # Application host, configuration, and ProjectModule protocol
│   ├── assets/                   # Asset cooker, formats (.pm_mesh, .pm_tex), and VFS (.pak)
│   ├── audio/                    # 3D spatialized audio engine and procedural sound synth
│   ├── camera/                   # Virtual camera system (FollowCamera, FreeCamera, OrbitCamera)
│   ├── core/                     # Loop, ECS, contiguous memory tables, input, math
│   ├── debug/                    # Central debug suite (F1 Profiler, F2 G-Buffer Visualizer, F3 Tweaks)
│   ├── gfx/                      # ModernGL 4.5 MDI pipeline, passes, and shader management
│   ├── physics/                  # Rapier3D Rust integration, character motor, raycast vehicle
│   ├── ui/                       # Native ModernGL 2D SDF UI subsystem
│   └── tools/                    # Packager and distribution bundlers
├── examples/                     # Standalone reference apps & demos
│   ├── 01_minimal_app.py         # Minimal 3D setup, camera, entity rotation
│   └── 02_game_states_and_ui.py  # Menu, Pause overlay, and Game State Manager
├── projects/                     # Standalone game projects directory
│   └── shotgun_beat_the_speed/   # Example arcade muscle racer
├── shaders/                      # Consolidated GLSL 4.5 core shaders
├── tests/                        # Comprehensive unit and integration test suite
└── docs/                         # Specifications and developer manuals
```

---

## Installation

### Prerequisites
- **Python**: 3.11 or higher
- **GPU**: OpenGL 4.5 Core Profile capable (NVIDIA GeForce 900+, AMD Radeon R7+, or Intel Iris Xe)

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/uzairmukadam/PyMordial-Engine.git
cd PyMordial-Engine

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install PyMordial Engine and developer dependencies in editable mode
pip install -e .[dev]
```

---

## Interactive Demos & Quickstart

### 1. Modern AAA UI, Menus & Game State Transitions
Explore modern glassmorphic ModernGL UI screens, widgets (buttons, sliders, segmented tabs), and stack-based `GameStateManager` transitions:
```bash
python examples/01_modern_ui_and_menu.py
```

### 2. Player Kinematics & Dynamic Physics Cubes
Experience 60 Hz decoupled `CharacterMotor` movement (WASD, sprint, jump), curb autostepping, collision-aware `FollowCamera`, dynamic Rapier3D physics cubes, and an in-game telemetry HUD:
```bash
python examples/02_player_physics_and_cubes.py
```

### 3. Full Graphics & Display In-Game Configuration Menu
Live in-game ModernGL pipeline adjustment without debug overlays: Resolution (720p/900p/1080p), Mode (Windowed/Borderless), VSync, Master Presets (Low to Cinematic), and granular toggles (SSAO/GTAO, Bloom, SSR, Fog, AA):
```bash
python examples/03_full_graphics_and_display_menu.py
```

### 4. Run the Arcade Racer Project
PyMordial includes a complete example game project: **Shotgun: Beat the Speed** (an arcade muscle racer):

```bash
python -m projects.shotgun_beat_the_speed.main
```

### 5. Run Headless Verification (Automated 60-Frame Benchmark)
```bash
python projects/shotgun_beat_the_speed/main.py --headless --frames 60
```

### 6. Run the Test Suite
The engine includes over 380 automated tests verifying ECS memory alignment, determinism, rendering passes, audio, physics, UI, state management, and packaging:

```bash
python -m pytest
```

---

## Creating a Game Project

Game projects in PyMordial are completely decoupled from the core engine:

```
projects/my_game/
├── modules/              # Reusable game features (weapons, AI, abilities)
│   └── __init__.py
├── world/                # World geometry generators & level builders
│   └── level.py
├── assets/               # Textures, models, audio
├── config.py             # Project configuration
└── main.py               # Entry point
```

### 1. Define `config.py`

```python
from engine.app.config import ProjectConfig

def get_project_config() -> ProjectConfig:
    return ProjectConfig(
        title="Neon Drift",
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

### 2. Create a Game Module

```python
from engine.app.module import ProjectModule
from engine.app.project_app import ProjectApp
import pygame

class NitroBoostModule(ProjectModule):
    def __init__(self) -> None:
        super().__init__(name="nitro_boost")
        self.charge = 100.0

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        # Step kinematics / physics impulses during constant-dt ticks
        pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        # Update animations, audio cues, or camera effects
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        # Handle input events (return True if consumed)
        return False
```

### 3. Wire Up `main.py`

```python
from engine import ProjectApp
from projects.my_game.config import get_project_config
from engine.world.default_world import DefaultWorldBuilder
from projects.my_game.modules.nitro import NitroBoostModule

def main() -> None:
    config = get_project_config()
    app = ProjectApp(config)
    app.set_world_builder(DefaultWorldBuilder(size=200.0))
    app.add_module(NitroBoostModule())
    app.run()

if __name__ == "__main__":
    main()
```

---

## Packaging & Distribution

PyMordial features a dual-target release packager that strips debug overlays, tree-shakes unused modules, and archives shaders and assets into a zero-copy `game.pak` file:

### PyInstaller (Zero-Compiler Standalone Distribution)
```bash
python -m engine.tools.packager --project projects/my_game --target pyinstaller --release
```

### Nuitka (Ahead-of-Time C++ Compilation)
```bash
python -m engine.tools.packager --project projects/my_game --target nuitka --release
```

---

## Engine Debug Subsystem

In developer builds (`enable_debug = True`), PyMordial reserves three central hotkeys that must never be overridden by game code:

- **`F1`**: System Performance Monitor, real-time FPS gauge, frame time graph, 1% lows, and ECS memory counters.
- **`F2`**: Graphics Inspector & G-Buffer Visualizer (view Albedo, Normals, Motion Vectors, Depth, and toggle passes in real-time).
- **`F3`**: Game Tweaks Inspector (inspect and modify registered CVars dynamically).

---

## Documentation

- [docs/ENGINE_FEATURES.md](docs/ENGINE_FEATURES.md) — Comprehensive technical specifications for all engine subsystems.
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — Detailed developer tutorials, module design patterns, and world-building guides.
- [docs/UI_AND_STATE_MANAGEMENT.md](docs/UI_AND_STATE_MANAGEMENT.md) — Native ModernGL UI widgets and stack-based Game State Machine guide.
- [docs/LLM_INSTRUCTIONS.md](docs/LLM_INSTRUCTIONS.md) — Strict architectural, memory, and coding guidelines for AI coding agents.

---

## License

PyMordial Engine is released under the [MIT License](LICENSE).