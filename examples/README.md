# PyMordial examples

Run the interactive samples from the repository root with Python 3.11+ and the
engine dependencies installed:

```powershell
python examples/01_shapes_flycam.py
```

For CI or a quick smoke test, add `--headless --frames 30`. Headless runs
default to the low graphics preset. Use `--quality low|medium|high|ultra|cinematic`
to select a preset explicitly.

## The six samples

| Example | What to explore |
| --- | --- |
| `01_shapes_flycam.py` | Cube, sphere, plane, capsule, cylinder, cone, water, materials, and free flight |
| `02_physics_player.py` | Fixed geometry, dynamic rigid bodies, and a controllable character capsule |
| `03_water_lighting_lab.py` | Five compact connected chambers, pools, a skylight, colored bounce, practical lights, and Radiance Cascades GI |
| `04_data_driven_platformer.py` | The multi-section Sunken Observatory platformer, loaded from JSON and traversed with a capsule player; fall or press R to reset |
| `05_menu_system.py` | Main menu, controls screen, level start, pause/resume, and return-to-menu flow |
| `06_graphics_benchmark.py` | Reproducible scene with configurable entity count, camera flight, FPS/frame-time sampling, and live GI/SSR controls |

`platformer_level.json` defines the platformer spawn, platforms, decorations,
and lights. Edit the data to prototype layout changes without editing the
loader.

## Controls

### Flycam examples

- **W / A / S / D** or **arrow keys**: Fly forward, left, back, and right
- **Q / E**: Fly down / up
- **Mouse**: Look around (captured automatically when gameplay starts)
- **Escape**: Release the mouse; click the window to capture again
- **Shift**: Move faster
- **Mouse wheel**: Adjust fly speed
- **Home**: Restore the starting camera pose
- **F1 / F2 / F3**: Engine performance monitor, graphics inspector, and game tweaks

### Character examples

- **W / A / S / D** or **arrow keys**: Walk
- **Space**: Jump
- **Shift**: Sprint
- **Mouse**: Orbit the third-person camera (captured automatically)
- **Escape**: Release the mouse; click the window to capture again
- **Mouse wheel**: Change follow distance
- **R**: Reset to the start; falling below the level also respawns you automatically

### Menu and benchmark

- In the menu sample, use the pointer to choose **Start exploring**, **Controls**,
  **Resume**, or **Main menu**. **Escape** pauses or returns to the title menu.
- In the menu sample, the cursor is free in the menu and captured automatically
  when the level starts.
- In the benchmark, **1 / 2 / 3** selects GI off / hybrid / Radiance Cascades;
  **4** toggles SSR. Flycam controls remain active.

No example consumes the engine-reserved F1, F2, or F3 keys.
