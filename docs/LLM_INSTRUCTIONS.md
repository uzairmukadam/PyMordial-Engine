# PyMordial Engine: AI Agent & LLM Coding Guidelines

This document sets strict architectural, memory, and coding constraints for AI coding agents (Antigravity, Gemini, Copilot, etc.) working on PyMordial Engine and PyMordial game projects.

---

## Rule 1: Engine Purity & Directory Boundaries

1. **NEVER** write game-specific gameplay logic, character states, quest scripts, or weapons inside `engine/`.
2. All game-specific logic must reside in `projects/<project_name>/`.
3. Game features must be organized as isolated, modular packages inside `projects/<project_name>/modules/<module_name>/`.
4. When writing a feature intended to be reused across projects, ensure it implements the `ProjectModule` protocol and does not import directly from another sibling module unless guarded with `has_module` checks.

---

## Rule 2: Zero-Allocation Inner Loop Rule

Python's garbage collector pauses cause frame spikes and stutter. Follow these zero-allocation constraints inside `on_fixed_update`, `on_update`, `on_render`, and math helpers:

1. **NO Heap Allocations Per Frame**:
   - Do NOT create `list`, `dict`, or `set` instances in per-frame tick methods.
   - Do NOT instantiate new `np.ndarray` objects per tick. Pre-allocate buffer arrays (`scratch_vec`, `scratch_mat`) during module `__init__` or `on_attach`.
   - Use NumPy's `out=` parameter for matrix multiplication and vector operations:
     ```python
     # BAD
     vp = proj @ view
     # GOOD
     mat4_mul(proj, view, out=self._vp_mat)
     ```
2. **Pre-allocate Tuples & Scalar Structs**:
   - Where tuples are required by C-libraries (e.g. Rapier vectors `(x, y, z)`), reuse scalar variables or unpack directly without intermediate objects.

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

---

## Rule 4: Physics & Simulation Decoupling

1. **Fixed Step Only**:
   - Rigid body forces, kinematic impulses, character motor translations, and collision queries MUST be executed in `on_fixed_update(app, dt)` where `dt` is constant (e.g. `1/60.0`).
   - NEVER step physics inside `on_update(app, dt)`.
2. **Visual Smoothing**:
   - Render passes receive `alpha` (the accumulator fraction between the previous and next fixed physics states) to smoothly interpolate transforms.

---

## Rule 5: Correct Imports & API Surface

Avoid common hallucinated imports:
- **Quality Presets**: `from engine.gfx.quality_presets import GraphicsQuality, RenderConfig` (NOT `engine.config`).
- **Render Context**: `from engine.gfx.context import RenderContext` (NOT `engine.window`).
- **Material Flags & Displacement**: `from engine.gfx.texture_atlas import DisplacementMode, encode_mat_flags, decode_material_folder`.
- **Cameras**: `from engine.camera.follow_camera import FollowCamera, FollowCameraConfig` and `from engine.camera.free_camera import FreeCamera`.
- **Character Controller**: `from engine.physics.character_controller import KinematicCharacterMotor, CharacterMotorConfig`.
- **Audio Cues**: `from engine.audio.sound_cue import SoundCue, AttenuationModel`.

---

## Rule 6: Headless Testing & Verification

1. Every new feature or project must be runnable in headless mode:
   ```powershell
   python projects/<project_name>/main.py --headless --frames 60
   ```
2. Automated unit tests must be added to `tests/` verifying functionality without requiring an interactive window or display server.
3. Keep test execution times under 1 second per test by using low resolution (e.g. `320x240`) and headless context.

---

## Rule 7: Release Readiness & Packaging

1. Projects must support release builds where `enable_debug = False`.
2. Release builds will strip `debug_overlay`. Ensure your game scripts do not hard-depend on debug overlay classes.
3. Use `python -m engine.tools.packager --project <name> --dry-run` to test release staging and tree-shaking before finalizing changes.
