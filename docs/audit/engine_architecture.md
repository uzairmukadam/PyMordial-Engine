# Engine Architecture Audit — Phase 1

> **Status**: read-only audit. No runtime code, shader, config, or asset was modified.
> **Baseline**: `pymordial-engine 0.1.0a1`, Python 3.12.10, `.venv`.
> **Method**: static trace of every claim from configuration → Python control code → resource
> creation → pass scheduling → shader binding → framebuffer → consumer. Claims that could not be
> traced end-to-end are marked **UNCONFIRMED** and listed in [open_questions.md](./open_questions.md).

---

## 1. Confirmed Layer Structure

The four-layer stack described in [ARCHITECTURE.md](../reference/ARCHITECTURE.md) is real and
enforced by import direction, not by tooling.

| Layer | Package | Confirmed role |
|---|---|---|
| 4. Projects | `projects/<name>/` | `config.py` → `ProjectConfig`, `main.py` → `ProjectApp`, `modules/`, `world/` |
| 3. App host | `engine/app/` | `ProjectApp`, `ProjectConfig`, `ProjectModule`, `ModuleRegistry`/`ModuleHost`, `project_validator` |
| 2. Subsystems | `engine/gfx`, `physics`, `audio`, `ui`, `input`, `camera`, `assets`, `world`, `debug`, `time`, `config` | Independent services |
| 1. Core | `engine/core`, `engine/events`, `engine/logging` | ECS, loop, state machine, math, event bus, logger |

**Verified import-direction facts**

- `engine/core/` does not import `engine/gfx/` (grep over `engine/core/*.py`).
- `engine/` does not import `projects/` (only `projects/*/main.py` imports `engine`).
- `examples/` is a leaf: nothing in `engine/`, `tests/`, or `projects/` imports `examples/`.

**Confirmed exception to "subsystems do not import each other"**

`engine/gfx/pipeline.py` imports concrete pass classes from `engine/gfx/passes/*` plus
`shadow_csm.py`, `spot_shadows.py`, `texture_atlas.py`, `mega_buffer.py`, `mdi.py`, `culling.py`,
`gpu_profiler.py`, `post_process.py`, `environment.py`. This is intentional composition (the
pipeline *is* the graphics root), but it means the graphics subsystem is one tightly-coupled unit,
not a set of independently swappable services. Any claim that built-in passes are independently
replaceable is **false**.

---

## 2. Startup Sequence (confirmed)

`ProjectApp.__init__` — [project_app.py:71](../../engine/app/project_app.py:71)

```text
1. validate_project_config(config).raise_for_errors()      # no GPU touched
2. get_quality_preset(GraphicsQuality(config.quality_preset.lower()))
   quality_cfg.max_entities = config.max_entities
   if not config.particles_enabled: quality_cfg.particles_enabled=False, particle_mode="OFF"
3. RenderContext(width, height, title, hidden=headless, config=quality_cfg)
   window.set_vsync(...)
4. RenderPipeline(render_ctx, quality_cfg); pipeline.apply_config(pipeline.config)
   render_ctx.pipeline = pipeline
5. EntityManager(max_entities); ecs.set_material_registry(pipeline.material_registry)
   particles = pipeline.particle_pass; particles.entity_manager = ecs
   PhysicsManager(gravity)
6. InputManager(); mouse-grab policy; EngineLoop(ecs, input_manager, fixed_dt=1/fixed_hz)
7. AudioEngine (singleton), CameraManager, UIManager(ctx, w, h), GameStateManager(app)
   MapManager(app)
8. Debug subsystem — only if config.enable_debug and not config.headless
   SystemMonitor, EngineTweaks, GameTweaks, DebugToast, DebugMenu
   engine_tweaks.sync_from_pipeline(pipeline)
9. world_builder=None, _modules=[], _module_host=None, draw-batch lists, pygame.time.Clock
   _bind_loop_callbacks()
```

**Confirmed ordering invariant**: validation precedes GPU allocation. A malformed config fails
before a GL context exists.

**Confirmed coupling**: `RenderPipeline` is built *before* `EntityManager`, but the pipeline needs
`max_entities` to size SSBOs. This works because `quality_cfg.max_entities` is set from
`config.max_entities` before the pipeline is built, and `EntityManager` is then built from the same
`config.max_entities`. The two agree by construction, not by assertion.

**Confirmed risk**: `ProjectApp.__init__` never asserts `pipeline.config.max_entities ==
ecs.max_entities`. `config.max_entities` is read twice; a mutation between the reads lets the culler
write past the SSBO. See [open_questions.md](./open_questions.md) Q1.

---

## 3. Shutdown Sequence (confirmed)

`ProjectApp.shutdown` — [project_app.py:588](../../engine/app/project_app.py:588), called from the
`finally` block of `ProjectApp.run`.

```text
1. pop every GameState until stack_depth == 0
2. for m in reversed(_modules): m.on_detach(app)   (exceptions swallowed)
3. _modules.clear(); _module_host.forget(); _module_host = None
4. world_builder.teardown_world(app); world_builder = None
5. if config.screenshot: render_ctx.save_screenshot(...)
6. ui.destroy(); ui = None
7. audio.stop_all()
8. system_monitor.destroy()
9. pipeline.destroy(); pipeline = None
10. render_ctx.destroy()
```

**Confirmed defect (resource leak, not a crash)**: `RenderPipeline.destroy`
([pipeline.py:1836](../../engine/gfx/pipeline.py:1836)) releases most passes but **never calls**:

- `self.water_pass.destroy()` — `WaterPass.destroy` exists and releases VAO/IBO/VBO/FBOs.
- `self.spot_shadow_map.destroy()` — `SpotLightShadowMap.destroy` exists and releases sampler,
  depth texture, and FBO.
- `self.spot_depth_prog.release()` / `self.spot_depth_vao.release()`.
- `self.cascade_shadow_mdis[i].destroy()` for the four per-cascade MDI objects.
- `self.render_features.destroy_all()` — registered render features are never destroyed.

Verified by grep: `water_pass.destroy` and `spot_shadow_map.destroy` appear **zero** times in
`pipeline.py`. This leaks on every shutdown and matters for the test suite, which forces
`gc.collect()` between tests to avoid WGL context exhaustion.

---

## 4. Frame Lifecycle (confirmed)

`ProjectApp.run` ([project_app.py:484](../../engine/app/project_app.py:484)) is the real loop. It
does **not** call `EngineLoop.run`; it calls `EngineLoop.step_frame` directly.

```text
while is_running:
    raw_ms = clock.tick(0); dt = min(raw_ms*0.001, 0.1)
    [mouse-grab policy from current GameState]
    events = input_manager.poll_events()
    for ev in events:
        window.poll_window_events(ev)          # resize/focus → engine events
        QUIT → stop
        F1/F2/F3 → debug_menu (continue, never reaches gameplay)
        debug_menu.process_event(ev) if blocks_game_input
        ui.handle_event(ev)
        state_manager.handle_event(ev)
        modules.on_event(ev)                   # first consumer wins
    step_frame(dt)                             # → EngineLoop.step_frame
    max_frames check
finally: shutdown()
```

`EngineLoop.step_frame` ([loop.py:75](../../engine/core/loop.py:75)) is the four-phase frame:

| Phase | Work | Confirmed |
|---|---|---|
| 0 | `TimeSystem.tick(frame_time)` → `real_dt`, `game_dt`; `accumulator += game_dt` | yes |
| 1 | `while accumulator >= fixed_dt:` `ecs.cache_previous_physics_state()`, `on_fixed_update(fixed_dt)`, `time_system.step_fixed_tick()`, `accumulator -= fixed_dt` | yes |
| 2 | `alpha = accumulator/fixed_dt`; `ecs.interpolate_render_transforms(alpha)`; `on_update(game_dt, alpha)`; `flush_events()` | yes |
| 3 | `on_render(alpha)` | yes |
| 4 | `on_ui()` | yes |

**Confirmed**: physics steps only in phase 1. `ProjectApp.on_fixed_step` calls
`map_manager.on_fixed_update`, module `on_fixed_update`, `physics.step_simulation(dt)`,
`physics.sync_to_ecs(ecs)` — in that order.

**Confirmed**: `EngineLoop.run` ([loop.py:127](../../engine/core/loop.py:127)) exists but has **no
callers** in `engine/`, `tests/`, `examples/`, `projects/`, or `scripts/`. It is a second, divergent
loop (polls input itself, checks `is_action_pressed("pause")`, uses `FPSLimiter`). `ProjectApp.run`
duplicates that logic with different semantics (no pause action, no limiter, `max_frames` from
config). **Duplicate implementation** — see [system_inventory.md](./system_inventory.md) §Legacy.

---

## 5. Render Frame Order (confirmed against code)

`RenderPipeline.render_frame` ([pipeline.py:1049](../../engine/gfx/pipeline.py:1049)). Order read
from actual call sites, not from documentation.

| # | Stage | Call site | Gate |
|---|---|---|---|
| — | early-out if `ecs.active_count <= 0` | 1064 | always |
| — | `gpu_profiler.begin_frame()` | 1067 | always |
| 1 | SSBO upload (static/dynamic split) | 1075 | always |
| 2 | camera matrices + TAA jitter | 1097 | always |
| 3 | environment `update(dt)` + lighting sync | 1154 | always |
| 4 | UBO 0 upload (`frame_context.update`) | 1190 | always |
| 5 | MDI build + per-cascade shadow culling + camera culling | 1211–1296 | always |
| 0 | `lights_pass.execute` (clustered lights SSBO) | 1308 | always |
| 1 | CSM shadow pass | 1310 | always |
| 1.5 | spot shadow pass | 1350 | `spot_shadows_active` |
| 2 | G-Buffer | 1372 | always |
| 2.5 | SSDM | 1480 | `config.ssdm_enabled` |
| 2.6 | Decals | 1494 | always |
| 2.7 | Hi-Z | 1498 | always |
| 2.8 | Direct radiance | 1505 | always |
| 3 | AO (GTAO/SSAO) | 1509 | always (mode may be OFF) |
| 5 | GI — `rc_pass` **or** `ssgi_pass` | 1522 / 1527 | `gi_mode == "RADIANCE_CASCADES"` else SSGI |
| 6 | IBL | 1533 | always |
| 6 | SSR | 1535 | always (internally gated) |
| 6.5 | `environment.render_sky` / `render_clouds` | 1537 | provider-owned |
| 7 | Deferred resolve | 1674 | always |
| 7.1 | Auto-exposure | 1693 | always (internally gated) |
| 7.5 | Debug wireframes | 1697 | `debug.vertex_count > 0` |
| 7.6 | Volumetric fog | 1706 | always (internally gated) |
| 7.65 | Water | 1710 | `water_pass is not None` |
| 7.7 | Particles | 1723 | `particle_pass is not None` |
| 7.8 | Camera optics / lens flare | 1734–1755 | `camera_optics_pass is not None` |
| 7.9 | TAA | 1767 | `taa_active` |
| 7.95 | Motion blur | 1774 | `motion_blur_enabled` |
| 7.97 | Bloom | 1779 | `bloom_enabled` |
| 8 | Post-process + exactly one AA | 1789–1802 | always |
| — | present (copy final FBO → screen) | 1808 | not headless |
| — | store prev VP, snapshot transforms, clear lights | 1811–1826 | always |

**Confirmed**: the documented order in ARCHITECTURE.md §4 matches the code, with one documentation
omission — **bloom (7.97) is not listed** in the ARCHITECTURE.md table even though it runs between
motion blur and post-process.

**Confirmed**: `gi_mode` selects exactly one GI path; there is no "both" branch.
**Confirmed**: `aa_mode` selects exactly one of FXAA/SMAA in post-process; TAA is a separate earlier
pass gated by `taa_active`.

---

## 6. Render Graph vs. Actual Execution (confirmed divergence)

`RenderPipeline._build_render_graph` ([pipeline.py:700](../../engine/gfx/pipeline.py:700)) builds a
`RenderGraph` of `_BuiltinPass` placeholders. `_BuiltinPass.execute` is a **no-op**
([pipeline.py:81](../../engine/gfx/pipeline.py:81)).

**Confirmed facts**

1. `render_graph.execute(...)` is **never called** anywhere in the repository. The graph is built,
   validated, and used only as an ordering/resource contract for module render features.
2. `render_features.apply_to(...)` runs on registration, adding feature passes to the graph and
   validating them — but those passes are **never executed**, because the graph is never executed
   and `render_frame` only calls the hardcoded built-in passes.
3. Therefore: **a registered `RenderFeature` is validated but never rendered.** This is a
   partially-integrated system, not a working extension point.

This is the most important architectural finding of the audit. It is documented as working in
ARCHITECTURE.md §4 ("Module render features anchor to those stage names via `after`/`before`") and
ENGINE_FEATURES.md §6.3, but the execution path does not exist.

---

## 7. Resource Ownership Summary (confirmed)

| Resource | Owner | Lifetime | Released by |
|---|---|---|---|
| GL context, window | `RenderContext` | app | `render_ctx.destroy()` |
| MegaBuffer VBO/IBO | `RenderPipeline.mega_buffer` | pipeline | `pipeline.destroy()` |
| G-Buffer MRT + depth | `GBuffer` | pipeline | `g_buffer.destroy()` |
| CSM atlas | `CascadedShadowMap` | pipeline | `csm.destroy()` |
| Spot shadow atlas | `SpotLightShadowMap` | pipeline | **NOT RELEASED** (leak) |
| HDR / LDR targets | `PostProcessPipeline` | pipeline | `post_process.destroy()` |
| Per-pass targets | each pass | pipeline | each `pass.destroy()` |
| Water targets | `WaterPass` | pipeline | **NOT RELEASED** (leak) |
| Sky/cloud LUTs | `EnvironmentProvider` | pipeline | `environment.destroy()` |
| ECS SoA arrays | `EntityManager` | app | GC |
| Rapier world | `PhysicsManager` | app | GC (no explicit destroy) |
| UI textures | `UIManager` | app | `ui.destroy()` |
| Debug ImGui state | `DebugMenu` | app | `system_monitor.destroy()` only |

**Confirmed**: `PhysicsManager` has no `destroy()`/`close()`; the Rapier world is released only by
GC. Acceptable for a single-app process, a leak risk in the test suite which builds many apps.

---

## 8. Configuration Paths (confirmed)

There are **three** independent configuration surfaces, and they are not unified:

| Surface | Type | Owner | Persisted | Consumed by |
|---|---|---|---|---|
| `ProjectConfig` | dataclass | project `config.py` | no (code) | `ProjectApp.__init__` |
| `RenderConfig` | dataclass | `quality_presets.py` | via `EngineTweaks.save_to_file` | `RenderPipeline` |
| `CVarRegistry` | registry | `engine/config/registry.py` | `save_to_file`/`load_from_file` | **nothing in runtime** |

**Confirmed**: `CVarRegistry` is exported from `engine.config` and tested (`tests/test_cvars.py`),
but **no runtime code reads a CVar**. Grep for `get_cvar` outside `engine/config/` and `tests/`
returns zero hits. Complete-but-unwired subsystem.

**Confirmed**: `EngineTweaks` is the *actual* live settings surface. It has
`to_dict`/`from_dict`/`save_to_file`/`load_from_file` and `apply_to_pipeline`, and
`ProjectApp.on_render_step` calls `engine_tweaks.apply_to_pipeline(pipeline)` every frame.
`DebugMenu` writes `settings.ini` via `engine.config.ini_export` (debug_menu.py:512, 557).

**Confirmed**: `ProjectConfig.settings_export_path` is passed to `DebugMenu` and is the only path by
which a project opts into user-editable settings.

---

## 9. Feature Enable/Disable Paths (confirmed)

| Feature | Config field | Gate location | Notes |
|---|---|---|---|
| Quality preset | `ProjectConfig.quality_preset` | `ProjectApp.__init__` → `get_quality_preset` | string, lowercased |
| Atmosphere | `ProjectConfig.atmosphere_enabled` | `create_default_environment` | swaps `FlatEnvironment` |
| Atmosphere tuning | `ProjectConfig.atmosphere_settings` | `create_atmosphere_environment` | preset + field overrides |
| Particles | `ProjectConfig.particles_enabled` | `ProjectApp.__init__` | forces `particle_mode="OFF"` |
| Debug subsystem | `ProjectConfig.enable_debug` + `headless` | `ProjectApp.__init__` | both required |
| GI mode | `RenderConfig.gi_mode` | `render_frame` 1518 | `OFF`/`IBL`/`SSGI`/`RADIANCE_CASCADES` |
| AO mode | `RenderConfig.ao_mode` | `ao_pass` | `OFF`/`SSAO`/`GTAO` |
| AA mode | `RenderConfig.aa_mode` | `post_process` | `OFF`/`FXAA`/`SMAA` |
| TAA | `RenderConfig.taa_enabled` | `render_frame` 1763 | independent of `aa_mode` |
| Bloom | `RenderConfig.bloom_enabled` | `render_frame` 1778 | |
| Motion blur | `RenderConfig.motion_blur_enabled` | `render_frame` 1771 | |
| Volumetric fog | `RenderConfig.volumetric_fog_enabled` | `volumetric_fog_pass` | |
| SSDM | `RenderConfig.ssdm_enabled` | `render_frame` 1481 | |
| Tessellation | `RenderConfig.tess_enabled` | G-Buffer path | |
| POM | `RenderConfig.pom_enabled` | G-Buffer uniforms | |
| Lens flare | `RenderConfig.lens_flare_mode` | `render_frame` 1747/1755 | `OFF`/`ANAMORPHIC`/`SOLAR` |
| Auto-exposure | `RenderConfig.auto_exposure_enabled` | `exposure_pass` | |
| Hi-Z culling | `RenderConfig.hiz_cull_enabled` | `render_frame` 1260 | default **False** |
| Frustum culling | `RenderConfig.frustum_cull_enabled` | `render_frame` 1260 | default True |
| Spot shadows | `RenderConfig.spot_shadows_enabled` | `render_frame` 1353 | |
| Clustered lights | `RenderConfig.clustered_lights_enabled` | `lights_pass` | |

**Confirmed**: every one of these is mirrored in `EngineTweaks.__slots__` and written in
`apply_to_pipeline`, which is why F2 works. The ARCHITECTURE.md warning that a new `RenderConfig`
field must be added to `EngineTweaks` is accurate.

---

## 10. Confirmed Architecture Risks

1. **Render features are validated but never executed** (§6). Any module declaring
   `ModuleRole.RENDER_FEATURE` silently renders nothing.
2. **`RenderPipeline.destroy` leaks** the spot shadow atlas, water pass, spot depth program/VAO,
   cascade MDI objects, and all registered render features (§3).
3. **Two loop implementations** (`ProjectApp.run` vs `EngineLoop.run`) with divergent semantics
   (§4).
4. **Three config surfaces**, one of which (`CVarRegistry`) is entirely unwired (§8).
5. **`max_entities` is read twice** from `ProjectConfig` with no equality assertion (§2).
6. **`u_SunAngularRadius` is declared in `deferred_resolve.frag` but never set** by any Python code
   (zero hits in `pipeline.py` and `environment.py`). Dead shader state; a stale test asserts it was
   removed.
7. **`engine/gfx/atmosphere.py` is a deprecated shim** still imported by three test files. No
   runtime code imports it.

---

## 11. Files Inspected

`engine/app/{config,module,module_contracts,module_manifest,module_registry,project_app,project_validator}.py`,
`engine/core/{loop,ecs,state,water,input,character_camera,entity_pool,async_loader}.py`,
`engine/gfx/{pipeline,render_graph,render_feature,render_services,frame_context,environment,context,shader_utils,quality_presets,post_process,atmosphere,material_registry,volumetrics}.py`,
`engine/gfx/passes/*.py` (API surface), `engine/modules/atmosphere/*.py`,
`engine/world/{base,default_world}.py`, `engine/world/mapping/map_manager.py`,
`engine/physics/rapier_world.py`, `engine/input/input_manager.py`, `engine/audio/audio_engine.py`,
`engine/ui/manager.py`, `engine/config/{__init__,registry}.py`,
`engine/assets/{__init__,vfs,resource_cache}.py`, `engine/debug/{__init__,monitor,engine_tweaks}.py`,
`engine/events/{event,event_bus}.py`, `engine/window/window.py`,
`shaders/deferred_resolve.frag` (uniform scan), `shaders/**` (include graph),
`projects/{reference_game,playground}/main.py`, `pyproject.toml`, `conftest.py`, `tests/conftest.py`.

## 12. Files Changed

None. Only `docs/audit/*.md` created.

## 13. Tests Performed

- `python -m pytest --collect-only -q` → **795 tests collected**.
- `python -m pytest -q` (full suite) → **793 passed, 2 failed, 8 warnings, 162.19 s**.

## 14. Results

Baseline is green except two pre-existing failures, both caused by **stale test expectations**, not
engine defects:

| Test | Failure | Root cause |
|---|---|---|
| `tests/test_camera_optics.py::test_quality_presets_camera_optics_scaling` | asserts `low.film_grain_enabled is True` | All five presets set `film_grain_enabled=False` (quality_presets.py:357,406,468,530,593). The test encodes an older preset design. |
| `tests/test_shadow_csm.py::TestRemovedShadowMachinery::test_shader_has_no_pcss_uniforms` | asserts `"u_SunAngularRadius" not in source` | `deferred_resolve.frag:123` still declares `uniform float u_SunAngularRadius = 0.0;`. The uniform is dead (never set) but not removed. |

Both are documentation/test drift, not runtime bugs. Recorded here and **not fixed** in this phase.

## 15. Known Limitations of This Audit

- No GPU-side verification (no frame capture, no shader compile check beyond the test suite).
  Shader *bindings* were traced statically, not observed at runtime.
- `engine/debug/debug_menu.py` (102 KB) and `engine/debug/engine_tweaks.py` (64 KB) were inspected
  at their API surface and specific call sites, not line-by-line.
- `engine/gfx/passes/radiance_cascades_pass.py` (29 KB) and `rc_cascade.py` were inspected at the
  pass-scheduling level only; internal cascade math was not audited.
- `native/pymordial_rapier/src/lib.rs` (27 KB) was not audited; only its Python-facing
  `PhysicsManager` wrapper was.
- `scripts/run_validation.py` (81 KB) and `scripts/run_pom_plane.py` (52 KB) are developer
  harnesses, not engine code, and were not audited.

## 16. Recommended Next Action

Phase 2 should be **one** of the following, chosen explicitly (do not proceed automatically):

1. **Fix the render-feature execution gap** (§6) — highest-value architectural defect, but requires
   a design decision (execute the graph, or execute feature passes inside `render_frame`).
2. **Fix the `RenderPipeline.destroy` leaks** (§3) — smallest, safest, highest-confidence patch.
3. **Repair the two stale tests** (§14) — trivial, but must decide whether the *test* or the
   *engine* is authoritative for `film_grain_enabled` and `u_SunAngularRadius`.
