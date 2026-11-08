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
| `02_physics_player.py` | Fixed geometry, dynamic rigid bodies, a controllable capsule, and an entity-attached ember/smoke fire |
| `03_water_lighting_lab.py` | Five compact connected chambers, pools, a skylight, colored bounce, practical lights, and Radiance Cascades GI |
| `04_data_driven_platformer.py` | The multi-section Sunken Observatory platformer, loaded from JSON and traversed with a capsule player; fall or press R to reset |
| `05_menu_system.py` | Main menu, controls screen, level start, pause/resume, and return-to-menu flow |
| `06_graphics_benchmark.py` | Reproducible scene with configurable entity count, camera flight, FPS/frame-time sampling, and live GI/SSR controls |

`platformer_level.json` defines the platformer spawn, platforms, decorations,
and lights. Edit the data to prototype layout changes without editing the
loader.

## Entity-attached particle effects

Particle presets no longer fill the whole scene with camera-centered dust.
Attach an effect to an entity through `app.particles`; emitters follow that
entity's position and rotation and are removed automatically when it is
destroyed:

```python
firepit = app.ecs.create_entity(shape="cylinder", position=(2.0, 0.5, 0.0))
app.particles.add_fire(firepit, ember_count=96, smoke_count=160)
```

`add_fire` combines a short-lived ember flare with a rising, fading smoke
plume. For a single effect, use `add_emitter(entity_id, mode, count=...)`
with a `ParticleMode` from `engine.gfx.passes.particle_pass` (`EMBERS`, `SMOKE`,
`FIREFLIES`, or `DUST_MOTES`). `offset`, `half_extents`, `size_multiplier`,
`turbulence_strength`, `speed_scale`, and `color` tune each emitter. Emitters
share the particle pass's fixed GPU capacity; removing one releases its slots.

```python
from engine.gfx.passes.particle_pass import ParticleMode

app.particles.add_emitter(
    firepit,
    ParticleMode.SMOKE,
    count=128,
    offset=(0.0, 0.8, 0.0),
    color=(0.45, 0.48, 0.52),
)
```

## Water entity properties

Water simulation and optical settings belong to each water entity, not the
global graphics quality preset:

```python
from engine.core import WaterProperties

pool = app.ecs.create_entity(
    shape="water",
    position=(0.0, 0.25, 0.0),
    dimensions=(12.0, 0.0, 8.0),
    water=WaterProperties(
        wave_amplitude=0.12,
        color_shallow=(0.08, 0.42, 0.48),
        foam_enabled=True,
    ),
)
```

Existing water entities can be tuned through
`app.ecs.get_transform(pool).water`.

## Controls

### Flycam examples

- **W / A / S / D** or **arrow keys**: Fly forward, left, back, and right
- **Q / E**: Fly down / up
- **Left stick**: Fly forward, back, and strafe
- **Right stick**: Look around
- **A / B**: Fly up / down
- **X**: Restore the starting camera pose
- **Left stick click / LB**: Move faster
- **Mouse**: Look around (captured automatically when gameplay starts)
- **Escape**: Release the mouse; click the window to capture again
- **Shift**: Move faster
- **Mouse wheel**: Adjust fly speed
- **Home**: Restore the starting camera pose
- **F1 / F2 / F3**: Engine performance monitor, graphics inspector, and game tweaks

### Character examples

- **W / A / S / D** or **arrow keys**: Walk
- **Left stick**: Walk
- **Right stick**: Orbit the third-person camera
- **A**: Jump
- **Left stick click / LB**: Sprint
- **X**: Reset to the start
- **Space**: Jump
- **Shift**: Sprint
- **Mouse**: Orbit the third-person camera (captured automatically)
- **Escape**: Release the mouse; click the window to capture again
- **Mouse wheel**: Change follow distance
- **R**: Reset to the start; falling below the level also respawns you automatically

### Menu and benchmark

- In the menu sample, use the pointer to choose **Start exploring**, **Controls**,
  **Resume**, or **Main menu**. **Escape** pauses or returns to the title menu.
- In the menu sample, use the **left stick** to move the virtual cursor, **A** to
  activate a button, and **Start** to start, pause, or resume.
- In the menu sample, the cursor is free in the menu and captured automatically
  when the level starts.
- In the benchmark, **1 / 2 / 3** selects GI off / hybrid / Radiance Cascades;
  **4** toggles SSR. **RB** cycles GI modes and **Y** toggles SSR; flycam
  controls remain active.

No example consumes the engine-reserved F1, F2, or F3 keys.
