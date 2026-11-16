# PyMordial Engine: Testing Guide

> **Audience**: developers and AI agents adding or fixing engine behaviour.
> **Companion docs**: [ARCHITECTURE.md](./ARCHITECTURE.md) · [LLM_INSTRUCTIONS.md](./LLM_INSTRUCTIONS.md)

The suite is the engine's safety net. It runs entirely headless — no display server, no audio hardware, no window — so it works in CI and on a headless box.

---

## 1. Running the Suite

```powershell
# Everything
.\.venv\Scripts\python.exe -m pytest tests/ -q

# One file
.\.venv\Scripts\python.exe -m pytest tests/test_render_pipeline.py -q

# One test
.\.venv\Scripts\python.exe -m pytest tests/test_horizon1_foundation.py::TestMDICommandPacking -q

# Stop at first failure, show locals
.\.venv\Scripts\python.exe -m pytest tests/ -x -l
```

**Baseline**: the suite must be fully green. A red suite is never "pre-existing noise" until you have proven it on a clean checkout — state that explicitly if you believe it is.

---

## 2. How Headless Rendering Works

Real GPU tests run against a hidden window with a genuine OpenGL 4.5 context. This catches shader compile errors, GL state bugs, and buffer-layout mistakes that mocks cannot.

```python
from engine.gfx.context import RenderContext

ctx = RenderContext.create_headless(width=320, height=240)
try:
    pipeline = RenderPipeline(ctx)
    pipeline.render_frame(ecs=ecs, camera_pos=(0, 4, 8), camera_target=(0, 0, 0))
    pixels = pipeline.post_process.final_texture.read()
    assert any(b > 0 for b in pixels)
finally:
    ctx.destroy()
```

For pure-GL probes that do not need the engine pipeline, use a standalone context:

```python
import moderngl
ctx = moderngl.create_standalone_context(require=450)
```

**Keep test resolutions small** (`160x120` or `320x240`). A 1080p headless render is slow and proves nothing extra.

---

## 3. Fixtures and Teardown

Two `conftest.py` files cooperate:

| File | Responsibility |
|---|---|
| `conftest.py` (root) | Forces `SDL_AUDIODRIVER=dummy`; runs `gc.collect()` after every test |
| `tests/conftest.py` | Provides `entity_pool`, `ecs`, `engine_loop` fixtures; also forces dummy audio |

The `gc.collect()` teardown is not cosmetic. WGL/GL contexts and buffers are released by finalizers, and without an explicit collection the suite exhausts GPU handles partway through a full run.

**Always release GL resources you create**, even in a failing test:

```python
ctx = RenderContext.create_headless()
try:
    ...
finally:
    ctx.destroy()
```

---

## 4. Test File Map

| File | Covers |
|---|---|
| `test_render_pipeline.py` | Presets, context creation, Reversed-Z, full frame render, resize |
| `test_horizon1_foundation.py` | MDI command packing, MDI vs direct pixel equivalence, spot-shadow MDI, `RenderConfig` shape, project validation |
| `test_frustum_culling.py` | Vectorized AABB culling, batch filtering |
| `test_hiz_pass.py` | Hi-Z pyramid construction and GPU occlusion |
| `test_csm_stabilization.py` | Cascade splits, texel snapping, bias |
| `test_spot_lights.py` | Spot light packing, shadow matrix, quadrant viewport |
| `test_radiance_cascades.py` | RC pass wiring and parameters |
| `test_taa.py` | Temporal accumulation, jitter, history invalidation |
| `test_ecs_memory.py` | SoA layout, free-list recycling, capacity |
| `test_loop_determinism.py` | Fixed-step accumulator, sub-tick counts |
| `test_physics_sync.py` | ECS ↔ Rapier transform sync |
| `test_character_controller.py` | KCC stepping, slopes, jump buffering |
| `test_ui_subsystem.py` | Widget layout, canvas scaling, hit-testing |
| `test_state_manager.py` | State stack, pause/resume, gating |
| `test_debug_system.py` | `EngineTweaks` slots, serialization, G-Buffer cycling |
| `test_example_controls.py` | Flycam/character input conventions |
| `test_project_framework_and_packaging.py` | `ProjectApp`, packager staging |
| `test_asset_cooker.py` | `.pm_mesh` / `.pm_tex` round-trip, MegaBuffer ingestion |

---

## 5. Writing a Good Test

**Test behaviour, not implementation.** Assert what the user observes (pixels, positions, validation errors), not that a private method was called.

**Prove the test can fail.** Before trusting a regression test, revert the fix and confirm it goes red. A test that passes both before and after the fix is worthless.

```powershell
# Temporarily break the fix, run the test, confirm failure, restore
```

**Prefer pixel-level assertions for rendering.** Counting lit pixels catches geometry, culling, and buffer-offset bugs that state assertions miss:

```python
def lit_pixels(fbo, size=64):
    raw = np.frombuffer(fbo.read(components=4, dtype="f1"), dtype=np.uint8)
    return int((raw.reshape(size, size, 4)[:, :, 0] > 0).sum())
```

**Use `pytest.mark.parametrize` for shape variants** rather than copy-pasting:

```python
@pytest.mark.parametrize("mesh_name", ["cube", "sphere", "capsule"])
def test_indirect_matches_direct_draw(gl, mesh_name):
    ...
```

**Keep each test under ~1 second.** If it needs longer, the resolution is probably too high.

---

## 6. Testing Rules for AI Agents

1. **Add tests only for behaviour your change introduces or fixes.** Do not chase coverage.
2. **Put new tests in the existing file** that covers that subsystem. Create a new file only for a genuinely new subsystem.
3. **Never weaken an assertion to make a test pass.** If a test fails, either the code is wrong or the test encoded a wrong expectation — decide which, and say so.
4. **Never delete a failing test** without explaining why its expectation was invalid.
5. **Run the smallest command that covers your change**, then the full suite before declaring done.
6. **Report the exact numbers**: "306 passed" not "tests pass".

---

## 7. Headless Smoke Tests

Every project and example must survive a headless run. This is the cheapest end-to-end check that a change did not break startup, shader compilation, or teardown.

```powershell
# Examples
.\.venv\Scripts\python.exe examples\01_shapes_flycam.py --headless --frames 30

# Scaffolded project
.\.venv\Scripts\python.exe -m engine.tools.project create --project smoke
.\.venv\Scripts\python.exe -m engine.tools.project validate --project smoke
.\.venv\Scripts\python.exe -m engine.tools.project run --project smoke --headless --frames 3
```

The validation harness is the heaviest smoke test and exercises nearly every pass:

```powershell
.\.venv\Scripts\python.exe scripts\run_validation.py --headless --frames 5 --benchmark
```

---

## 7.1 Visual Regression Gate

`tests/test_horizon2_golden.py` renders a fixed scene and compares it against
`tests/baseline/*.npy`. It guards the default sky, lighting, and deferred-resolve output.

```powershell
.\.venv\Scripts\python.exe -m tests.golden_capture --check    # verify
.\.venv\Scripts\python.exe -m tests.golden_capture --write    # re-record intentionally
```

The capture runs in a **subprocess** on purpose. Creating and destroying several GL contexts
inside one process measurably perturbs later frames on some drivers (observed MAE up to 3.2
after unrelated pipeline tests), so an in-process capture would report false regressions
depending on test ordering. Do not "optimize" this into an in-process fixture.

Tolerance is a mean absolute error of 2.0 on the 0–255 scale. Re-record only when a visual
change is intentional, and say so in the commit message.

---

## 8. Linting

```powershell
.\.venv\Scripts\python.exe -m ruff check engine/ tests/ scripts/
```

Config lives in `pyproject.toml`: `select = ["E", "F", "W"]`, `line-length = 120`, `E501` ignored, target `py311`.

Lint only the files you touched when iterating; run the whole tree before declaring done. Pre-existing violations in files you did not modify are not yours to fix — but say so explicitly rather than silently ignoring them.
