# PyMordial Engine (v0.1.0 Alpha)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ModernGL](https://img.shields.io/badge/Graphics-ModernGL%204.5%20Core-orange.svg)](https://github.com/moderngl/moderngl)
[![Rapier3D](https://img.shields.io/badge/Physics-Rapier3D%20(Rust)-red.svg)](https://rapier.rs/)
[![Pygame-CE](https://img.shields.io/badge/Window-Pygame--CE-yellow.svg)](https://pyga.me/)
[![Development Status](https://img.shields.io/badge/Status-Alpha%200.1.0-brightgreen.svg)]()

**PyMordial Engine** is a high-performance, modular 3D game engine engineered from the ground up for Python 3.11+, ModernGL (OpenGL 4.5 Core Profile), native Rapier3D physics (Rust), and Pygame-CE.

Designed to bridge the gap between high-level Python productivity and AAA-grade visual fidelity, PyMordial combines an expressive component and module lifecycle with a zero-allocation, contiguous data-oriented execution core.

> [!NOTE]
> **Alpha Status**: PyMordial Engine is currently in active **Alpha (v0.1.0 Alpha)**. All core foundational subsystems—including the ModernGL 4.5 deferred rendering pipeline, Rapier3D physics integration, responsive 2D UI canvas scaler, audio engine, asset cooker, and sacred debug suite—are fully functional, thoroughly tested (240+ automated tests), and ready for game development.

---

## Key Architectural Highlights & Features

### 1. Data-Oriented Core & Zero-Allocation Hot Paths
- **Zero-Allocation Inner Loop**: Transforms, physics states, and material indices reside in contiguous, C-aligned NumPy Structured Arrays that stream directly into OpenGL Shader Storage Buffer Objects (SSBOs) without creating per-frame Python objects or garbage-collector overhead.
- **Hardware-Accelerated Multi-Draw Indirect (MDI)**: Static and dynamic geometries are coalesced into a unified GPU `MegaBuffer`, submitting entire scenes in minimal draw calls via `glMultiDrawElementsIndirect`.
- **Reversed-Z Floating Point Depth**: 32-bit floating-point depth testing with `GL_GREATER` eliminates Z-fighting across vast outdoor landscapes and distant horizons.
- **High-Throughput ECS & Free-List Pools**: O(1) entity spawning and recycling with zero heap reallocation.

### 2. State-of-the-Art ModernGL 4.5 Deferred Rendering Pipeline
- **Deferred G-Buffer MRT Layout**:
  - `RT0 (RGBA16F)`: Albedo RGB + Roughness A
  - `RT1 (RGBA16F)`: World Normal RGB + Metallic A
  - `RT2 (RGBA16F)`: Screen-Space Motion Vectors (2D velocity) + Material Layer ID
  - `RT3 (R11G11B10F)`: Emissive RGB + Ambient Occlusion
  - `Depth (DEPTH32F)`: High-precision Reversed-Z depth buffer
- **Cascaded Shadow Maps (CSM)**: Up to 4 shadow cascades with logarithmic/linear split blending, sub-texel stabilization, and 16-tap Poisson disc PCF filtering.
- **Ground Truth Ambient Occlusion (GTAO) & SSAO**: Horizon-based ambient occlusion with bilateral cross-bilateral blur for contact shadows and depth shading.
- **Screen-Space Reflections (SSR)**: Traces glossy and rough reflections across G-Buffer surfaces with edge fading, roughness blur, and multi-object line-of-sight support for complex scene geometry and water planes.
- **Screen-Space Global Illumination (SSGI) & Light Propagation Volumes (LPV)**: Dynamic indirect diffuse bounces and low-frequency volumetric light grids.
- **Clustered Forward+ Local Lighting**: 3D screen frustum cluster binning evaluating hundreds of dynamic point and spot lights simultaneously.
- **Physical Sky & Atmosphere**: Precomputed Rayleigh and Mie atmospheric scattering with ozone absorption and real-time sun/moon solar cycles.
- **Volumetric Fog & Light Shafts**: 3D Froxel grid integration through anisotropic participating media.
- **Comprehensive Anti-Aliasing Suite**:
  - Temporal Anti-Aliasing (TAA) with Halton subpixel jitter and YCoCg neighborhood color clamping.
  - Subpixel Morphological Anti-Aliasing (SMAA 1X/2X/4X) and Fast Approximate Anti-Aliasing (FXAA 3.11).
- **Cinematic Post-Processing & Optics**:
  - Physically based bokeh Depth of Field (DoF), dual-threshold bloom, chromatic aberration, and anamorphic flare.
  - Tone-mapping operators: ACES filmic curve, AgX, Reinhard, and Uncharted 2.

### 3. Synchronous Rapier3D Physics (Rust)
- **Rust Integration**: In-thread simulation stepping with GIL release to multi-threaded Rayon workers.
- **Authoritative Kinematic Character Motor (`CharacterMotor`)**:
  - Ground clamping, curb autostepping, and slope limit sliding.
  - Variable jump cut, air control, jump buffering, and coyote time.
  - Decoupled 60 Hz fixed-step simulation with interpolation alpha for butter-smooth visual rendering.
- **Rigid Body Dynamics**: Dynamic, Kinematic, and Fixed bodies with cuboids, spheres, capsules, convex hulls, and triangle meshes.
- **Raycast Vehicle Physics**: Advanced multi-wheel suspension springs, anti-roll bars, and Pacejka combined tire slip dynamics.
- **Object Interaction & Throwing**: Pick-and-carry physics grabber with distance damping and velocity impulse release.

### 4. Native ModernGL 2D UI Subsystem & Responsive Canvas Scaler
- **Hardware-Accelerated SDF Batch Renderer**: Signed Distance Field (SDF) 2D rendering of antialiased rounded rectangles, gradients, borders, glows, and cached textures in minimal GPU batches.
- **Resolution-Independent Canvas Scaling**:
  - All UI screens, menus, and widgets are authored in a clean reference canvas ($1280 \times 720$).
  - Proportional canvas scaler automatically adapts layouts to any screen resolution ($720\text{p}, 900\text{p}, 1080\text{p}, 1440\text{p}, 4\text{K}$) and aspect ratios (16:9, 16:10, 21:9 ultrawide).
  - Exact mouse coordinate translation preserves 100% pixel-perfect hit-testing across all resolutions.
  - Dynamic font rasterization renders native high-resolution glyphs without bitmap stretching or blur.
- **Widget Library**: `UIPanel`, `UILabel`, `UIButton`, `UISlider`, `UISegmentGroup`, `UIImage`, and `UIScreen`.
- **Captured AAA Cursor**: Hardware-accelerated glowing cyan core cursor with obsidian drop-shadow and mouse-grab pass-through.
- **Stack-Based Game State Manager (`GameStateManager`)**: Clean push/pop lifecycle transitions between MainMenu, Gameplay, Paused, and Settings states.

### 5. 3D Spatialized Audio & Procedural Synthesizer
- **Spatial 3D Sound Cues**: Positional audio with distance attenuation (Linear, Logarithmic, Inverse Clamped) and stereo panning relative to camera orientation.
- **Procedural Sound Synthesizer**: Generates dynamic footsteps, surface impacts, engine tones, jump swooshes, and environmental wind loops at zero disk cost.

### 6. Virtual File System (VFS) & Offline Asset Cooker
- **Memory-Mapped `.pak` Containers**: Zero-copy `.pak` archive with 64-bit FNV-1a hash tables for O(1) asset resolution.
- **Asset Cooker CLI**: Offline utility compiling `.obj`/`.glb` models into cache-aligned 32-byte `.pm_mesh` formats and images into `.pm_tex`.

### 7. Dual-Target Release Packaging
- Single-command packaging for release distribution:
  - `--target pyinstaller`: Beginner-friendly standalone one-dir bundle requiring zero C++ compilers.
  - `--target nuitka`: Ahead-of-Time (AOT) C++ compilation for peak CPU execution speed and binary protection.
- Automatically strips debug overlays and tree-shakes unused modules into a consolidated `game.pak`.

### 8. Sacred Central Debug Subsystem (F1, F2, F3)
- Dedicated developer inspection hotkeys that game code cannot intercept or swallow:
  - **`F1`**: System Performance Monitor, real-time FPS gauge, frame time history, 1% lows, and ECS memory counters.
  - **`F2`**: ModernGL Graphics Inspector & G-Buffer Visualizer (view Albedo, Normals, Motion Vectors, Depth, and toggle passes dynamically).
  - **`F3`**: Game Tweaks Inspector (inspect and modify registered CVars in real-time).

---

## Repository Structure

```
PyMordial-Engine/
├── engine/                       # Core engine systems (Rendering, Physics, ECS, Audio, VFS)
│   ├── app/                      # Application host, configuration, and ProjectModule protocol
│   ├── assets/                   # Asset cooker, formats (.pm_mesh, .pm_tex), and VFS (.pak)
│   ├── audio/                    # 3D spatialized audio engine and procedural sound synth
│   ├── camera/                   # Virtual cameras (FollowCamera, FreeCamera, OrbitCamera, Optics)
│   ├── core/                     # Loop, ECS, contiguous memory tables, input, math
│   ├── debug/                    # Central debug suite (F1 Profiler, F2 G-Buffer Visualizer, F3 Tweaks)
│   ├── events/                   # Synchronous event bus and window/input event types
│   ├── gfx/                      # ModernGL 4.5 MDI pipeline, passes, and shader management
│   ├── input/                    # Action mapping, device contexts, haptics, rebinding
│   ├── logging/                  # Thread-safe structured logging with sinks
│   ├── physics/                  # Rapier3D Rust integration, character motor, raycast vehicle
│   ├── ui/                       # Native ModernGL 2D SDF UI subsystem & canvas scaler
│   └── tools/                    # Packager and distribution bundlers
├── examples/                     # Standalone reference apps & demos
│   ├── 01_modern_ui_and_menu.py  # Modern AAA glassmorphic UI, audio sliders, settings modal
│   ├── 02_player_physics_and_cubes.py # Kinematic character controller, physics cubes, grabber, HUD
│   └── 03_full_graphics_and_display_menu.py # Live in-game graphics & display configuration menu
├── projects/                     # Standalone game projects directory (git-ignored for user games)
│   └── shotgun_beat_the_speed/   # Example arcade muscle racer project
├── shaders/                      # Consolidated GLSL 4.5 core shaders
├── tests/                        # Comprehensive unit and integration test suite (240+ tests)
└── docs/                         # Specifications, guides, and LLM coding instructions
    ├── DEVELOPER_GUIDE.md        # Architecture overview and developer workflows
    ├── ENGINE_FEATURES.md        # Technical subsystem specification
    ├── LLM_INSTRUCTIONS.md       # AI agent pair programming rules and constraints
    └── UI_AND_STATE_MANAGEMENT.md # UI system, widgets, and state machine guide
```

---

## Installation & Setup

### Prerequisites
- **Python**: 3.11 or higher
- **GPU**: OpenGL 4.5 Core Profile capable (NVIDIA GeForce GTX 900+, AMD Radeon R7 300+, or Intel Iris Xe)
- **OS**: Windows 10/11, Linux (X11/Wayland), or macOS (via OpenGL 4.1 Core fallback)

### 1. Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/uzairmukadam/PyMordial-Engine.git
cd PyMordial-Engine

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install PyMordial in Editable Mode

```bash
# Install engine dependencies and developer tools
pip install -e .[dev]
```

---

## Interactive Examples & Demos

PyMordial includes three standalone, fully featured interactive examples demonstrating the major engine capabilities:

### Example 1: Modern AAA UI, Settings Modal & State Machine
Demonstrates the native ModernGL 2D UI system, glassmorphic obsidian styling, category badges, audio volume sliders, graphic quality tabs, pause overlay, and stack-based `GameStateManager` transitions:
```bash
python examples/01_modern_ui_and_menu.py
```
- **Controls**:
  - `Mouse`: Navigate menus, click buttons, drag sliders, toggle segmented tabs.
  - `ESC`: Pause / Resume simulation or return to menu.

---

### Example 2: Player Kinematics, Dynamic Physics Cubes & Object Grabber
Demonstrates the 60 Hz authoritative `CharacterMotor` with collision-aware `FollowCamera`, autostepping curbs, slope sliding, dynamic Rapier3D rigid body cubes, interactive pick-and-carry grabber, and real-time telemetry HUD:
```bash
python examples/02_player_physics_and_cubes.py
```
- **Controls**:
  - `W, A, S, D` or `Left Analog Stick`: Walk / Strafe
  - `Left Shift` or `(L3) / (LB)`: Sprint
  - `Space` or `Gamepad (A)`: Jump
  - `Mouse Motion` or `Right Analog Stick`: Inverted 3D Camera Look (Yaw & Pitch)
  - `Mouse Wheel`: Camera Distance Zoom
  - `E` or `Left Mouse Button` or `Gamepad (X)`: Grab, carry, or throw physics cubes
  - `ESC` or `Gamepad Start`: Pause Menu / Reset Position

---

### Example 3: Full In-Game Graphics & Display Configuration Menu
Demonstrates live runtime ModernGL pipeline reconfiguration without debug overlays: resolution scaling ($1280 \times 720$, $1600 \times 900$, $1920 \times 1080$), window modes (Windowed, Borderless Fullscreen), VSync, Master Quality Presets (Low, Medium, High, Ultra, Cinematic), and granular pipeline toggles (GTAO, SSR, Volumetric Fog, Bloom, Anamorphic Flare, FXAA/SMAA/TAA):
```bash
python examples/03_full_graphics_and_display_menu.py
```
- **Controls**:
  - `W, A, S, D` / `Space` / `Shift`: Player Movement & Jump
  - `Mouse Motion`: Inverted 3D Camera Look
  - `ESC`: Open / Close Graphics & Display Configuration Menu

---

## Sacred Central Debug Subsystem (F1, F2, F3)

In developer builds (`enable_debug = True`), PyMordial reserves three central diagnostic hotkeys that remain always active and can never be overridden by game code:

| Hotkey | Subsystem | Features |
| :--- | :--- | :--- |
| **`F1`** | **System Monitor & Profiler** | Real-time FPS, frame time history graph, 1% lows, GPU/CPU frame times, ECS memory pool counters. |
| **`F2`** | **Engine Graphics Inspector** | G-Buffer visualizer (Albedo, Normals, Depth, Motion Vectors), live tonemapper selection, wireframe mode, pass toggles. |
| **`F3`** | **Game Tweaks Inspector** | Dynamic developer console for inspecting and modifying registered game CVars in real-time. |

---

## Creating a Game Project

Game projects in PyMordial are completely decoupled from the core engine:

```
projects/my_game/
├── config.py             # Project configuration (title, resolution, presets)
├── modules/              # Reusable game features (weapons, abilities, AI)
│   └── nitro.py
├── world/                # Level builders & geometry generators
│   └── level.py
└── main.py               # Game entry point
```

### 1. Define `config.py`
```python
from engine.app.config import ProjectConfig

def get_project_config() -> ProjectConfig:
    return ProjectConfig(
        title="Neon Velocity",
        width=1280,
        height=720,
        fullscreen=False,
        vsync=True,
        quality_preset="high",  # "low", "medium", "high", "ultra", "cinematic"
        fixed_hz=60.0,
        gravity=(0.0, -9.81, 0.0),
        enable_debug=True,
    )
```

### 2. Implement a Reusable `ProjectModule`
```python
import pygame
from engine.app.module import ProjectModule
from engine.app.project_app import ProjectApp

class NitroBoostModule(ProjectModule):
    def __init__(self) -> None:
        super().__init__(name="nitro_boost")
        self.boost_charge: float = 100.0

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        # Authority: step kinematics or apply physics impulses at constant dt
        pass

    def on_update(self, app: ProjectApp, dt: float) -> None:
        # Presentation: update particle effects, UI gauges, or audio
        pass

    def on_event(self, app: ProjectApp, event: pygame.event.Event) -> bool:
        # Return True if the input event is handled
        return False
```

### 3. Bootstrap `main.py`
```python
from engine import ProjectApp
from engine.world.default_world import DefaultWorldBuilder
from projects.my_game.config import get_project_config
from projects.my_game.modules.nitro import NitroBoostModule

def main() -> None:
    config = get_project_config()
    app = ProjectApp(config)
    app.set_world_builder(DefaultWorldBuilder(size=250.0))
    app.add_module(NitroBoostModule())
    app.run()

if __name__ == "__main__":
    main()
```

---

## Standalone Release Packaging

Bundle your project into a self-contained release distribution with stripped debug overlays and zero-copy `.pak` asset archives:

```bash
# PyInstaller (Single-folder standalone executable, no C++ compiler required):
python -m engine.tools.packager --project projects/my_game --target pyinstaller --release

# Nuitka (Ahead-of-Time C++ compilation for peak native execution speed):
python -m engine.tools.packager --project projects/my_game --target nuitka --release
```

---

## Automated Test Suite

PyMordial includes over 240 automated unit, integration, and regression tests:

```bash
# Run all tests
python -m pytest tests/

# Run UI subsystem tests
python -m pytest tests/test_ui_subsystem.py

# Run headless 60-frame verification
python -m pytest tests/test_examples.py
```

---

## Documentation Links

- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — Comprehensive developer workflows, architectural guides, and world building.
- [docs/ENGINE_FEATURES.md](docs/ENGINE_FEATURES.md) — Full technical specifications for all rendering, physics, and ECS subsystems.
- [docs/UI_AND_STATE_MANAGEMENT.md](docs/UI_AND_STATE_MANAGEMENT.md) — ModernGL 2D UI widgets, canvas scaler, and `GameStateManager` guide.
- [docs/LLM_INSTRUCTIONS.md](docs/LLM_INSTRUCTIONS.md) — Strict architectural, zero-allocation, and pair programming rules for AI agents.

---

## License

PyMordial Engine is open-source software released under the [MIT License](LICENSE).