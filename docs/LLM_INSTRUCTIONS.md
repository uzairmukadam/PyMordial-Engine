# PyMordial Engine: AI Agent & LLM Coding Guidelines (v0.1.0 Alpha)

This document defines strict architectural, memory, and coding constraints for AI coding agents (Antigravity, Gemini, Copilot, etc.) working on PyMordial Engine and PyMordial game projects.

---

## Rule 1: Engine Purity & Directory Boundaries

1. **Strict Core vs. Game Separation**:
   - **`engine/`**: Core engine systems only (Rendering, Physics, ECS, Audio, VFS, UI, Tools, App Host). **NEVER** write game-specific gameplay logic, character states, quest scripts, or weapons inside `engine/`.
   - **`projects/<project_name>/`**: Self-contained game projects (e.g. `projects/my_game/`).
   - **`projects/<project_name>/modules/<module_name>/`**: Modular, reusable gameplay packages implementing `ProjectModule`.
   - **`examples/`**: Minimal, self-contained standalone reference demonstrations.
2. **Decoupled Module Contracts**:
   - When writing a feature intended to be reused across projects, ensure it implements the `ProjectModule` protocol (`on_attach`, `on_fixed_update`, `on_update`, `on_render`, `on_event`, `on_detach`) and does not import directly from another sibling module unless guarded with `has_module` checks.

---

## Rule 2: Zero-Allocation Inner Loop Rule

Python's garbage collector pauses cause severe frame drops and stutter. Adhere strictly to these zero-allocation constraints inside `on_fixed_update`, `on_update`, `on_render`, `draw_rect`, `draw_text`, and mathematical helper routines:

1. **NO Heap Allocations Per Frame**:
   - Do **NOT** instantiate `list`, `dict`, or `set` objects in per-frame tick methods.
   - Do **NOT** instantiate new `np.ndarray` objects per tick. Pre-allocate buffer arrays (`scratch_vec`, `scratch_mat`, vertex arrays) during module `__init__` or `on_attach`.
   - Use NumPy's `out=` parameter for matrix multiplication and vector operations:
     ```python
     # BAD (creates temporary heap array)
     vp = proj @ view
     # GOOD (zero allocation)
     mat4_mul(proj, view, out=self._vp_mat)
     ```
2. **Pre-allocate Tuples & Scalar Structs**:
   - Where tuples are required by C-libraries (e.g. Rapier vectors `(x, y, z)`), reuse pre-allocated tuples or unpack directly without intermediate object allocations.

---

## Rule 3: Coordinate Systems & Conventions

- **Coordinate System**: Right-handed, Y-up.
  - `+X` = Right
  - `+Y` = Up
  - `+Z` = Backward (towards the camera)
  - `-Z` = Forward (into the screen)
- **Reversed-Z Depth**:
  - Near plane = `1.0`, Far plane / Infinity = `0.0`.
  - Depth test function is `GL_GREATER`.
- **Rotations & Angles**:
  - Internal trig functions expect radians (`math.radians`, `np.radians`).
  - Configuration files, cameras, and inspector UI display angles in degrees.
- **Camera Look**:
  - `FollowCamera.handle_mouse_orbit(rel_x, rel_y)` applies camera yaw and pitch. Passing `-rel_x, -rel_y` inverts horizontal and vertical mouse look.

---

## Rule 4: Physics & Simulation Decoupling

1. **Fixed Step Only (Authoritative Physics)**:
   - Rigid body forces, kinematic impulses, character motor translations, and collision queries MUST be executed in `on_fixed_update(app, dt)` where `dt` is constant (default `1/60.0` s).
   - **NEVER** step physics or apply character velocities inside `on_update(app, dt)`.
2. **Visual Smoothing (Variable Presentation)**:
   - Render passes receive `alpha` (the accumulator fraction `0.0 <= alpha <= 1.0` between the previous and next fixed physics states) to smoothly interpolate transforms for jitter-free rendering.

---

## Rule 5: Native ModernGL UI Subsystem & Canvas Scaling

1. **Virtual Reference Canvas ($1280 \times 720$)**:
   - All UI screens (`UIScreen`), panels, buttons, cards, and labels are designed in a standard reference coordinate system ($1280 \times 720$).
   - `UIManager` automatically computes uniform canvas scaling and centering offsets:
     $$\text{ui\_scale} = \min\left(\frac{\text{width}}{1280.0}, \frac{\text{height}}{720.0}\right)$$
   - **NEVER** hardcode physical window pixel positions into widget layouts. Always author positions and dimensions in the $1280 \times 720$ reference canvas space.
2. **Input Coordinate Translation**:
   - `UIManager.handle_event` automatically translates physical mouse events from window space into virtual reference coordinates before dispatching to widgets:
     $$\text{vx} = \frac{\text{pos.x} - \text{offset\_x}}{\text{ui\_scale}}, \quad \text{vy} = \frac{\text{pos.y} - \text{offset\_y}}{\text{ui\_scale}}$$
   - Hit-testing (`hit_test`) on all UI widgets uses virtual reference coordinates and remains 100% accurate across any screen resolution.
3. **High-DPI Font Rasterization**:
   - `UIRenderer.draw_text` scales font sizes dynamically to physical screen resolution ($\text{scaled\_size} = \max(8, \text{round}(\text{font\_size} \cdot \text{ui\_scale}))$) for razor-sharp text glyph rendering, returning virtual dimensions for exact layout centering.

---

## Rule 6: Correct Imports & API Surface

Avoid common hallucinated imports:
- **Application & Host**: `from engine.app.project_app import ProjectApp` and `from engine.app.config import ProjectConfig`.
- **Quality Presets**: `from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset` (NOT `engine.config`).
- **Render Context & Window**: `from engine.gfx.context import RenderContext` and `from engine.window.window import WindowMode, VSyncMode`.
- **Material Flags & Displacement**: `from engine.gfx.texture_atlas import DisplacementMode, encode_mat_flags, decode_material_folder`.
- **Cameras**: `from engine.camera.follow_camera import FollowCamera, FollowCameraConfig` and `from engine.camera.free_camera import FreeFlyCamera`.
- **Character Controller**: `from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig`.
- **Audio Cues**: `from engine.audio.sound import SoundCue` and `from engine.audio import get_audio_engine`.
- **State Management**: `from engine.core.state import GameState, GameStateManager`.
- **Native UI**: `from engine.ui import UIManager, UIScreen, UIPanel, UILabel, UIButton, UISlider, UISegmentGroup, UIImage, UIStyle`.

---

## Rule 7: Headless Testing & Verification

1. Every new feature or project must be runnable in headless mode:
   ```bash
   python projects/<project_name>/main.py --headless --frames 60
   ```
2. Automated unit tests must be added to `tests/` verifying functionality without requiring an interactive window or display server.
3. Keep test execution times under 1 second per test by using low resolution (e.g. `320x240`) and headless ModernGL context.

---

## Rule 8: Release Readiness & Standalone Packaging

1. Projects must support release builds where `enable_debug = False`.
2. Release builds strip `debug_overlay` and tree-shake unused modules into a consolidated `game.pak`. Ensure your game scripts do not hard-depend on debug overlay classes.
3. Use `python -m engine.tools.packager --project <name> --target [pyinstaller|nuitka] --release` for distribution bundling.

---

## Rule 9: Engine Lightness & Sacred Debug Subsystem (F1, F2, F3)

1. **Light Engine Philosophy**:
   - The engine is intentionally kept light to provide only core foundational systems without imposing game-specific assumptions.
2. **Never Override Engine Debug Hotkeys**:
   - Game modules must **NEVER** capture, intercept, or return `consumed = True` on reserved engine hotkeys:
     * **`F1`**: System Performance Monitor, FPS, 1% lows, memory gauges.
     * **`F2`**: Engine Graphics Tweaks, G-Buffer debug visualizer, tonemapping, wireframe.
     * **`F3`**: Game Tweaks Developer Inspector.
   - Gameplay camera toggles, abilities, or menus must strictly use standard gameplay keys (`V`, `C`, `Tab`, `F10`) and never capture `F1`-`F3`.
3. **Decoupled Debug Stripping**:
   - The engine debug subsystem is designed to be completely stripped in release packaging without leaving dangling references or breaking game logic. Game code must never mutate or bypass this decoupling.


