# AGENTS.md: Antigravity & AI Pair Programming Rules for PyMordial Engine

This repository contains **PyMordial Engine**, a high-performance 3D engine in Python, ModernGL, Rapier, and PyGame-CE, along with independent game projects in `projects/`.

## Key Architectural Principles

1. **Engine vs. Game Project Separation**:
   - `engine/`: Core engine systems only (Graphics, Physics, ECS, Audio, VFS, Tools, App host). Do NOT write game-specific features or scripts here.
   - `projects/<project_name>/`: Self-contained game projects (e.g. `projects/shotgun_escape_the_heat/`).
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

For the complete technical specification and rules, see:
- [docs/LLM_INSTRUCTIONS.md](docs/LLM_INSTRUCTIONS.md)
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
- [docs/ENGINE_FEATURES.md](docs/ENGINE_FEATURES.md)
