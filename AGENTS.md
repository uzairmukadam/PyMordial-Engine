# AGENTS.md: Antigravity & AI Pair Programming Rules for PyMordial Engine

This repository contains **PyMordial Engine**, a high-performance 3D engine in Python, ModernGL, Rapier, and PyGame-CE, along with independent game projects in `projects/`.

## Key Architectural Principles

1. **Engine vs. Game Project Separation**:
   - `engine/`: Core engine systems only (Graphics, Physics, ECS, Audio, VFS, Tools, App host). Do NOT write game-specific features or scripts here.
   - `engine/modules/<module_name>/`: Official reusable engine capabilities (e.g. `engine/modules/atmosphere/`). Each declares a `MODULE_MANIFEST` and is toggled from `ProjectConfig`, never by editing `pipeline.py`.
   - `projects/<project_name>/`: Self-contained game projects (e.g. `projects/reference_game/`).
   - `projects/<project_name>/modules/<module_name>/`: Portable, reusable gameplay modules implementing `ProjectModule`.
2. **Zero-Allocation Inner Loop**:
   - Never allocate lists, dicts, sets, or NumPy arrays inside `on_fixed_update`, `on_update`, or `on_render`.
   - Pre-allocate scratch vectors and matrices during initialization; use NumPy `out=` parameters.
3. **Physics / Tick Decoupling**:
   - Physics and kinematic movements occur strictly in `on_fixed_update(app, dt)` at constant dt.
   - Interpolation alpha is used for smooth rendering in `on_render(alpha)`.
4. **Dual-Target Packaging**:
   - `python -m engine.tools.packager --project <name> --target [pyinstaller|nuitka] --release`
   - Release builds strip `debug_overlay` and tree-shake unused modules into zero-copy `game.pak`.
5. **Engine Lightness & Sacred Debug Subsystem**:
   - The engine is intentionally kept light to provide only core foundational primitives without interfering with game design.
   - Game projects must **NEVER** override, intercept, or swallow reserved engine debug hotkeys (`F1`: System Monitor/Profiler, `F2`: Engine Graphics & G-Buffer Tweaks, `F3`: Game Tweaks).
   - Game-specific camera toggles, gameplay abilities, or HUD actions must use standard gameplay keys (`V`, `C`, `Tab`, `F10`) and never capture `F1-F3`.
   - The central debug subsystem (`DebugMenu`, `DebugDraw`, `SystemMonitor`, `EngineTweaks`) must always remain functional in developer builds, being stripped only in final release packaging (`--release`).
6. **Responsive UI Canvas Scaling ($1280 \times 720$)**:
   - All UI screens (`UIScreen`), cards, widgets, and layouts are authored strictly in the reference coordinate system ($1280 \times 720$).
   - Never hardcode physical window pixel positions into widget layouts. The engine `UIManager` and `UIRenderer` scale and letterbox proportionally with 100% pixel-accurate hit-testing across any resolution ($720\text{p}, 900\text{p}, 1080\text{p}, 1440\text{p}, 4\text{K}$).

For the complete technical specification and rules, see:
- [docs/README.md](docs/README.md) — Documentation hub and navigation
- [docs/reference/LLM_INSTRUCTIONS.md](docs/reference/LLM_INSTRUCTIONS.md) — Hard rules for AI agents
- [docs/reference/ARCHITECTURE.md](docs/reference/ARCHITECTURE.md) — Layer boundaries, frame flow, pass order, invariants
- [docs/reference/ENGINE_FEATURES.md](docs/reference/ENGINE_FEATURES.md) — Subsystem capability spec
- [docs/reference/TESTING.md](docs/reference/TESTING.md) — Test layout and conventions
- [docs/guides/DEVELOPER_GUIDE.md](docs/guides/DEVELOPER_GUIDE.md) — Building projects, modules, and worlds
- [docs/guides/MODULES.md](docs/guides/MODULES.md) — Module metadata, host, render features, engine modules
- [docs/guides/UI_AND_STATE_MANAGEMENT.md](docs/guides/UI_AND_STATE_MANAGEMENT.md) — Native UI and state machine
- [docs/roadmap/](docs/roadmap/) — Future plans (not descriptions of current behaviour)

