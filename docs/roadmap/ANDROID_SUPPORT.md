# Android Support Documentation for PyMordial Engine

> **Status: PROPOSAL — not implemented.** No Android code path exists in the engine today: there is no `enable_android` flag on `ProjectConfig`, no `build.py`, and no GLES feature-detection branch. This document is a design target for a future horizon, not a description of working behavior. See [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) for scheduling.

> This file describes how to enable and use optional Android support in PyMordial Engine.  All steps are **opt-in** – existing desktop projects continue to work unchanged.

---

## 1. Overview

PyMordial Engine can target **desktop OpenGL 4.5+** and **OpenGL ES 3.0+** (Android).  The Android path is **optional** and is activated only when a developer explicitly opts-in via the engine configuration.  When enabled:

* A unified `build.py` script can produce an **APK** using Chaquopy/Gradle.
* The rendering backend automatically detects the runtime GL version and disables desktop-only features (multi-draw-indirect, compute-shader GI, high-resolution cascaded shadow maps, SSR) in favour of mobile-friendly fallbacks.
* Asset cooking adds mobile-specific compressed texture formats (ASTC/ETC2) and optional mesh down-sampling.
* Input handling gains **gamepad** and **on-screen touch overlay** support.

---

## 2. Enabling Android Support in a Project

1. **Project configuration** – add the flag in `ProjectConfig` (usually located in `projects/<my_game>/config.py`):

```python
from engine.app.config import ProjectConfig, RenderConfig


def get_project_config() -> ProjectConfig:
    return ProjectConfig(
        title="My Game",
        width=1280,
        height=720,
        # ----- Android opt-in -----
        enable_android=True,  # <-- enable Android packaging
        render_config=RenderConfig(
            # mobile-friendly defaults (override as needed)
            max_shadow_cascades=2,
            shadow_map_resolution=1024,
            enable_multi_draw=False,
            gi_mode="OFF",
            enable_ssr=False,
        ),
        # other fields …
    )
```

2. **Gradle/Chaquopy setup** – create an `android/` sub-directory (see the **Build pipeline** section below).  The `build.py` script will automatically invoke this module when `--platform android` is supplied.

---

## 3. Build Pipeline – `build.py`

The repository already contains a unified build script (`build.py`).  To produce an Android APK:

```bash
python build.py --platform android
```

What the script does:
1. Builds wheel files for all Python dependencies (ModernGL-ES, Rapier-Rust, NumPy, etc.).
2. Copies the wheels into `android/app/src/main/python/wheels/`.
3. Calls the Gradle wrapper (`./gradlew assembleRelease`) which runs Chaquopy to embed the Python runtime, the wheels, and the engine code into an APK.
4. Places the resulting `app-release.apk` in `android/app/build/outputs/apk/release/`.

> **Note:** The script assumes you have Java 11+, Android SDK, and Chaquopy installed.  See the *Prerequisites* section below.

---

## 4. Rendering Capability Detection & Feature Fallbacks

At engine start-up the GL context is examined:
* **Version string** – if it contains `OpenGL ES` the ES backend is selected.
* **Extension probing** – checks for `EXT_multi_draw_indirect` and `GL_COMPUTE_SHADER`.

### Desktop-only features automatically disabled on ES:
| Feature | Desktop implementation | Mobile fallback |
|---|---|---|
| Multi-draw-indirect | `glMultiDrawElementsIndirect` (single call) | Simple per-object `glDrawElements` loop |
| Compute-shader GI (SSGI/Radiance Cascades) | Compute shaders | Baked light-map textures (generated offline) |
| High-resolution cascaded shadow maps | 4 cascades @ 2048px | 1-2 cascades @ 1024px (controlled via `RenderConfig`) |
| Screen-Space Reflections (SSR) | Full-screen ray-march | Disabled (can be re-enabled manually on powerful devices) |

Developers can manually override any of these defaults through `RenderConfig`.

---

## 5. RenderConfig – Mobile-specific Tunables

`RenderConfig` lives in `engine/app/config.py`.  The following fields are relevant for Android:

```python
class RenderConfig:
    max_shadow_cascades: int = 2  # 1-2 cascades are safe on most phones
    shadow_map_resolution: int = 1024  # 1024 px square per cascade
    enable_multi_draw: bool = False  # Use simple draw loop when EXT_multi_draw_indirect missing
    gi_mode: str = "OFF"  # Options: OFF, IBL, SSGI, RADIANCE_CASCADES – disabled by default
    enable_ssr: bool = False  # Turn on only on high-end devices
    # existing fields (e.g., enable_taa, quality_preset) remain unchanged
```

These defaults are applied automatically when `enable_android=True`.  They can be overridden per-project.

---

## 6. Asset Cooking for Mobile

The `engine.assets.cooker` now accepts a `--profile mobile` flag.  Example usage:

```bash
python -m engine.assets.cooker --project projects/my_game --profile mobile
```

What the **mobile profile** does:
* **Texture compression** – generates ASTC (Android A-profile) or ETC2 (fallback) versions and stores them alongside the original PNG/TGA files in the `.pak`.
* **Mesh down-sampling** – reduces vertex count for very high-poly meshes (optional, controlled by `mesh_reduction_factor` in the cooker config).
* **Mipmap generation** – creates mip-maps appropriate for mobile GPUs.

During runtime the engine loads the most appropriate mip-level based on screen resolution and device capabilities.

---

## 7. Controls – Gamepad & On-Screen Keys

### 7.1 Gamepad integration
* Enable with `ProjectConfig.enable_gamepad = True`.
* Pygame-CE exposes SDL2 gamepads; the engine maps physical buttons to logical `InputAction` names (`jump`, `shoot`, `menu`, …).  Analog sticks are exposed as `InputAxis` values in `[-1.0, 1.0]`.

### 7.2 On-Screen Touch Overlay
* Create a JSON file `onscreen_keys.json` placed in the project root (or any path referenced in `ProjectConfig`).  Example format:

```json
{
  "buttons": [
    {"name": "move_up",    "rect": [  50, 550, 150, 650 ], "type": "dpad"},
    {"name": "move_down",  "rect": [  50, 750, 150, 850 ], "type": "dpad"},
    {"name": "move_left",  "rect": [   0, 650, 100, 750 ], "type": "dpad"},
    {"name": "move_right", "rect": [ 100, 650, 200, 750 ], "type": "dpad"},
    {"name": "action",     "rect": [1000, 600, 1180, 780 ], "type": "button"}
  ]
}
```
* The UI system draws semi-transparent buttons at the positions defined here (scaled from the $1280 × 720 reference canvas).  Touch events are translated into synthetic `InputEvent` objects that flow through the existing input pipeline.

---

## 8. Known Limitations & Performance Tips

| Area | Limitation | Recommended mitigation |
|------|------------|--------------------------|
| Shader language | ES 3.0 lacks some desktop GLSL features (e.g., explicit layout qualifiers) | Use `#ifdef GL_ES` guards and provide simplified shader paths. |
| Multi-draw | `EXT_multi_draw_indirect` not universally present | Keep `enable_multi_draw=False` for Android; the engine will fall back to per-object draw loops.
| Compute shaders | Not available on many low-end devices | Disable GI modes (`gi_mode="OFF"`) or pre-bake lightmaps.
| Texture size | Mobile GPUs have lower memory bandwidth | Use the mobile asset profile to generate compressed textures and limit shadow map resolution.
| Frame-rate | Phones vary from 30 fps to 60 fps | Set `fixed_hz` in `ProjectConfig` to a target that matches the device (e.g., 30 Hz for low-end, 60 Hz for flagship). |

---

## 9. FAQ

**Q: Do I need to re-write any rendering code?**
A: No.  The engine abstracts the GL backend.  You only need to supply a fallback ES shader (placed under `engine/gfx/shaders/es/`).  Desktop shaders remain unchanged.

**Q: Can I ship both desktop and Android binaries from a single repository?**
A: Yes.  The `build.py` script supports `--platform desktop-pyinstaller`, `--platform desktop-nuitka`, and `--platform android`.  Desktop builds ignore `enable_android`.

**Q: What Android API level is required?**
A: API 21 (Lollipop) or higher – this provides OpenGL ES 3.0.  Chaquopy itself requires at least API 24 for bundled native libraries, but you can adjust the `minSdkVersion` in the Gradle file.

**Q: How do I debug rendering issues on Android?**
A: Enable the built-in **F2** graphics inspector (works on ES when the `debug_overlay` flag is true).  Connect the device via `adb logcat` to capture Python tracebacks.

---

## 10. Prerequisites

* **Java 11** (or newer) and Android SDK installed.
* **Gradle wrapper** (`./gradlew`) present in the `android/` folder.
* **Chaquopy** plugin added to `android/build.gradle` (see Chaquopy docs).
* **Python 3.11+** on the host machine for building wheels.
* **Device** with OpenGL ES 3.0+ (most modern Android phones).

---

## 11. References

* Chaquopy Android-Python integration – <https://chaquo.com/chaquopy/>
* OpenGL ES 3.0 specification – <https://www.khronos.org/registry/OpenGL/specs/es/3.0/> 
* ModernGL-ES wrapper (to be added) – planned in `engine/gfx/modern_gl_es.py`
* Input handling design – `engine/input/README.md`

---

*End of Android Support Documentation.*
