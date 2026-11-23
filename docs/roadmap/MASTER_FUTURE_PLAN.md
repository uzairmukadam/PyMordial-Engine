# PyMordial Engine: Master Strategic Plan & Horizon Status Tracker

> **Document Status**: Living Master Roadmap & Progress Tracker  
> **Last Updated**: 2026-11-15  
> **Repository Baseline**: 407/407 Core Tests Passing (100% Green)

> **This is the authoritative roadmap.** Other documents in [`roadmap/`](./) are supporting designs and analyses. For current engine behaviour, see [`../reference/`](../reference/). For building a game, see [`../guides/`](../guides/).

---

## 1. Horizon Status Dashboard

Use this dashboard to track overall progress across the 5 strategic horizons. Update the status badges, completion percentages, and commit hashes as work progresses.

| Horizon | Focus Area | Status | Progress | Target Milestone | Last Verified Commit / Date |
|---|---|:---:|:---:|---|---|
| **Horizon 1** | Foundation Clean-up, Pruning & True MDI | `DONE` | 100% | M1: Clean Base | 306/306 tests green (audit follow-up) |
| **Horizon 2** | Module Host Architecture & Render Graph | `DONE` | 100% | M2: Modular Host & Sky | 407/407 tests green |
| **Horizon 3** | Player Settings, World Lifecycle & Persistence | `PLANNED` | 0% | M3: Settings & Saves | — |
| **Horizon 4** | Decima-Grade World Systems & Streaming | `PLANNED` | 0% | M4: Decima World & Foliage | — |
| **Horizon 5** | Packaging, CI Qualification & Distribution | `PLANNED` | 0% | M5: Standalone Release | — |

### Status Legend
- `PLANNED`: Scoped and ready for execution.
- `IN_PROGRESS`: Actively under development in current session/branch.
- `BLOCKED`: Blocked by unresolved design decision or upstream dependency.
- `DONE`: Fully implemented, verified with tests, and merged.

---

## 2. Multi-Tier Hardware Scaling Matrix (The Counterpart Principle)

Every high-end visual technique has a mathematically sound, lightweight counterpart for handhelds (Steam Deck at 15W TDP) and low-end/integrated GPUs:

| Subsystem | High / Ultra Tier (Mid+) | Steam Deck / Medium Tier (15W) | Low-End / Integrated GPU |
|---|---|---|---|
| **Global Illumination** | Ray-Marched Global Illumination (RMGI) | SSGI (Half-res, 8 ray steps) | Baked IBL Split-Sum Env Radiance |
| **Ambient Occlusion** | GTAO (Full-res, 2 directions) | GTAO (Half-res bilateral blur) | SSAO (Fast 8-sample Poisson disc) |
| **Shadow Filtering** | PCSS Variable Penumbra | Vogel PCF (16 taps) | Hard / 4-tap PCF (2 cascades) |
| **Anti-Aliasing** | TAA + CAS Sharpening | TAA (Minimal jitter) | FXAA (Single-pass edge) |
| **Atmosphere / Fog** | Froxel Volumetric Fog 3D | Low-res Froxel ($80 \times 45 \times 32$) | Analytic Height/Distance Fog |
| **Occlusion Culling** | Full Hi-Z Occlusion + MDI | Hi-Z Occlusion + MDI | Vectorized CPU Frustum Cull |
| **Displacement** | Hardware Tessellation / POM | POM (8–16 linear steps) | Standard Tangent Normal Maps |
| **Character Stance** | 2-Bone Analytical IK Foot Clamping | 2-Bone Analytical IK Foot Clamping | Ground Ray Snapping |

---

## 3. Horizon Execution Work-Items & Progress Tracking

---

### Horizon 1: Foundation Clean-up, Boundaries & True MDI
* **Goal**: Strip obsolete prototypes, fix CPU draw submission bottlenecks, and establish rigid project directory boundaries.
* **Status**: `DONE`
* **Lead Target**: Horizon 1.1 Graphics Pruning & 1.2 Hardware MDI

#### Task Breakdown

- [x] **H1.1 — Graphics Pipeline Pruning & GI Consolidation**
  - **Status**: `DONE`
  - **Scope**:
    - Remove redundant 3D Light Propagation Volumes (`lpv_pass.py`, `lpv_propagate.comp`) to eliminate VRAM bandwidth leaks.
    - Remove obsolete HBAO (`hbao.frag`) and SSCS uniforms from `RenderPipeline` and `RenderConfig`.
    - Consolidate `GIMode` into: `OFF`, `IBL`, `SSGI`, `RMGI`.
    - Update `quality_presets.py` to match the Multi-Tier Hardware Matrix.
  - **Verification Gate**: `pytest tests/test_render_pipeline.py tests/test_rmgi.py` passes; no lingering LPV/HBAO references in `engine/` or `shaders/`.
  - **Notes / Log**: Deleted `engine/gfx/passes/lpv_pass.py` and `shaders/lpv_propagate.comp` (dead code — `LPVPass` was never instantiated and `u_LPVVolume` existed in no shader). Removed `GIMode.LPV`/`GIMode.HYBRID` aliases, `lpv_intensity` from `RenderConfig` and `EngineTweaks`, the `u_LPVVolume` texture unit, the `DebugView.LPV` enum, and HBAO/SSCS docstring references. Updated `examples/06_graphics_benchmark.py` to use `SSGI` instead of the removed `HYBRID`.
  - **Follow-up fix (audit)**: The original claim of "zero hits" was false. `scripts/run_validation.py` and `scripts/run_pom_plane.py` still assigned `EngineTweaks.sscs_*` / `lpv_intensity` (raising `AttributeError` at runtime, since `EngineTweaks` uses `__slots__`) and advertised removed `--ao-mode hbao`, `--gi-mode lpv|hybrid`, and `--gbuffer-debug 9` (LPV) choices. All stale script references, argparse choices, help strings, and the benchmark summary line were purged. `README.md`, `docs/reference/LLM_INSTRUCTIONS.md`, `docs/reference/ENGINE_FEATURES.md`, `docs/roadmap/ANDROID_SUPPORT.md`, and `docs/roadmap/MODULAR_ENGINE_*.md` were updated to the four-mode `GIMode`. `engine/`, `shaders/`, and `scripts/` are now clean of `lpv|hbao|sscs` (only historical `docs/roadmap/PYMORDIAL_UPGRADE_ROADMAP.md` retains them as rationale).

- [x] **H1.2 — True Hardware Multi-Draw Indirect Submission**
  - **Status**: `DONE`
  - **Scope**:
    - Modernize `MultiDrawIndirect.submit` in `engine/gfx/mega_buffer.py`: replace the Python loop calling `vao.render` with direct hardware `vao.render_indirect` / `glMultiDrawElementsIndirect`.
    - Eliminate all Python iteration overhead during frame draw dispatches.
  - **Verification Gate**: Single GPU draw call dispatched per batch; benchmark shows zero Python loop overhead.
  - **Notes / Log**: Implemented in `engine/gfx/mdi.py` (the class lives there, not `mega_buffer.py`). `submit()` now uploads the packed command buffer and issues a single `vao.render_indirect(...)` dispatch, with the legacy per-command `vao.render` loop retained only as a fallback for contexts lacking MDI support. `program` became optional since the fast path does not need it.
  - **Follow-up fix (audit)**: Two real defects were found and fixed.
    1. **`baseVertex` double-offset**: `MegaBuffer._add_mesh` already offsets mesh-local indices into global IBO space (`global_indices = indices + base_vertex`), so writing `alloc.base_vertex` into the indirect command applied the offset twice and silently dropped the first `base_vertex` vertices of every mesh. `add_command` now always writes `base_vertex = 0`; per-instance entity lookup is carried solely by `baseInstance` (`gl_BaseInstanceARB`). Verified with a headless GL 4.5 pixel probe: indirect draws now match direct draws exactly for `cube`, `sphere` (base_vertex 24), and `capsule`.
    2. **Silent failure masking**: the fast path was wrapped in `try/except Exception: pass`, which could double-draw if the exception occurred mid-dispatch. Replaced with a one-time `ctx.extensions` capability check (`GL_ARB_multi_draw_indirect` + `GL_ARB_shader_draw_parameters`) resolved in `__init__`, a logged warning on downgrade, and an explicit fallback path. Command overflow is now counted in `dropped_commands` instead of being silently discarded.
    3. **Spot shadow MDI never populated**: `pipeline.py` called `shadow_mdi.submit(...)` for spot shadow maps but never called `shadow_mdi.begin_frame()` nor `shadow_mdi.add_command(...)` (removed in `633f85e`), so `command_count` stayed 0 and spot shadow maps rendered nothing. The frame preamble now resets `shadow_mdi` and repopulates it with all shadow-caster batches.

- [x] **H1.3 — Project Directory Contract & Early Validation**
  - **Status**: `DONE`
  - **Reference**: Modular Plan Phase 0 & 1 (`P0.1`, `P0.2`, `P1.1`, `P1.2`)
  - **Scope**:
    - Standardize `projects/<project_name>/` structure: `config.py`, `main.py`, `modules/`, `world/`, `assets/`.
    - Implement early validation in `engine.app`: validate project path, entry points, configuration schema, and asset roots *before* allocating heavy GPU/physics contexts.
    - Normalize CLI run/test project path resolution.
  - **Verification Gate**: Valid projects boot cleanly; malformed configs fail early with actionable error diagnostics.
  - **Notes / Log**: Added `engine/app/project_validator.py` (`validate_project_config`, `validate_project_directory`, `resolve_project_path`, `ValidationReport`, `ProjectValidationError`) and `engine/tools/project.py` CLI (`validate`/`run`/`create`). `ProjectApp.__init__` now calls `validate_project_config(...).raise_for_errors()` before creating the window or physics world. Added 6 focused tests covering defaults, invalid values, missing entry point, syntax errors, and path resolution.
  - **Follow-up fix (audit)**: `validate_project_config` now also rejects an empty `title`, a `headless` + `fullscreen` conflict, and a non-positive `max_frames`, and reports a non-sized `gravity` value cleanly instead of raising `TypeError` from `len()`. `resolve_project_path` now returns the canonical `projects/<name>` candidate for bare names so diagnostics point at the real project root, and `cmd_validate` removes the `sys.path` entries it inserts.

- [x] **H1.4 — Reference Starter Project Scaffolding**
  - **Status**: `DONE`
  - **Reference**: Modular Plan `P1.4`
  - **Scope**:
    - Provide a clean, minimal reference starter game template demonstrating input, fixed update, a tiny world, and shutdown.
  - **Verification Gate**: Headless smoke test runs and exits cleanly without modifying engine source.
  - **Notes / Log**: `python -m engine.tools.project create --project <name>` scaffolds `config.py`, `main.py`, `modules/`, `world/`, `assets/` with `__init__.py` files. Verified end-to-end: scaffolded project passes `validate` and runs `--headless --frames 10` cleanly. Falls back to inline templates when `projects/starter_template/` is absent.
  - **Follow-up fix (audit)**: The repository had no `projects/` directory at all, so `create`/`resolve_project_path` relied on `mkdir(parents=True)` side effects and bare-name resolution fell back to the engine root. `cmd_create` now resolves the canonical `projects/` root explicitly, `resolve_project_path` prefers `projects/<name>` for bare names, and a `pymordial-project` console script was added to `pyproject.toml` alongside `pymordial-cooker`/`pymordial-packager`. Verified: `create` → `validate` → `run --headless --frames 3` all succeed.

- [x] **H1.5 — Model Shape & Inner-Loop Hygiene (Audit Follow-up)**
  - **Status**: `DONE`
  - **Scope**:
    - Close the gap between fields written by `EngineTweaks` and fields declared on `RenderConfig`.
    - Remove hardcoded frame constants and per-frame allocation churn from the project host.
    - Restore the documented camera look convention so the example-control tests pass.
  - **Verification Gate**: `pytest tests/` fully green; `RenderConfig` declares every field `EngineTweaks.apply_to_pipeline` writes.
  - **Notes / Log**:
    - `RenderConfig` was missing `frustum_cull_enabled`, `hiz_debug_mip`, and `max_entities`. Because it is a plain `@dataclass` (no `__slots__`), `EngineTweaks` writes silently created shadow attributes that the pipeline only read back through `getattr(..., default)` — so the F2 debug toggles had no effect. All three are now declared fields.
    - `get_quality_preset` copied presets via `cfg.__dict__`, which only worked because the dataclass lacked `__slots__`. Replaced with `dataclasses.replace(cfg)`.
    - `RenderPipeline` hardcoded `max_entities = 100_000` for SSBO sizing while `FrustumCuller` read `getattr(config, "max_entities", 100_000)`; both now use the single resolved value, and `ProjectApp` propagates `ProjectConfig.max_entities` into the render config.
    - `ProjectApp.on_render_step` passed a hardcoded `dt=1.0 / 60.0` to `render_frame` (wrong for particle simulation at any other tick rate). `EngineLoop` now exposes the current `game_dt` and the host forwards it.
    - `engine/camera/free_camera.py::handle_mouse_look` had an uncommitted inversion (`yaw -= rel_x`, `pitch += rel_y`) that contradicted `FollowCamera.handle_mouse_orbit` and `docs/reference/LLM_INSTRUCTIONS.md` Rule 3, breaking `tests/test_example_controls.py`. Restored to `yaw += rel_x`, `pitch -= rel_y`; callers invert by passing `-rel_x, -rel_y`.
    - Added `tests/test_horizon1_foundation.py` (22 tests) covering MDI command packing, pixel-level MDI/direct draw equivalence for three primitives, spot-shadow MDI population, `RenderConfig` shape, and validator edge cases.

---

### Horizon 2: Module Host Architecture & Render Extension Graph
* **Goal**: Implement the module system and extract the first official engine feature without breaking turnkey defaults.
* **Status**: `DONE`
* **Prerequisite**: Horizon 1 completed.

#### Task Breakdown

- [x] **H2.1 — Module Metadata Schema & Manifest System**
  - **Status**: `DONE`
  - **Reference**: Modular Plan Phase 2 (`P2.1`, `P2.2`)
  - **Scope**:
    - Define manifest schema (`module.toml` or typed dataclass) supporting `id`, `version`, `api_version`, `role`, `dependencies`, and `config_schema`.
    - Distinguish role-specific contracts (`GameplayModule`, `RenderFeatureModule`, `ToolModule`).
  - **Verification Gate**: Schema parser validates valid manifests and catches invalid keys/types.
  - **Notes / Log**: Implemented as typed dataclasses (no TOML file format, per project decision) in `engine/app/module_manifest.py`: `ModuleRole` (`GAMEPLAY`/`RENDER_FEATURE`/`TOOL`), frozen `ModuleManifest`, `ModuleDependency`, `build_manifest()`, `manifest_from_object()`, and `validate_manifest()` reusing the existing `ValidationReport`. Validation is side-effect free — it never imports or executes module code — and covers id format, semver, API major compatibility, role, `module.path:ClassName` entry shape, self/duplicate dependencies, and capability types. Role contracts live in `engine/app/module_contracts.py`. Modules declare metadata via a `MODULE_MANIFEST` attribute or `get_module_manifest()`; legacy modules without metadata still work. 24 tests in `tests/test_horizon2_manifest.py`.

- [x] **H2.2 — Module Registry & Deterministic Dependency Resolver**
  - **Status**: `DONE`
  - **Reference**: Modular Plan Phase 2 (`P2.3`, `P2.4`, `P2.5`)
  - **Scope**:
    - Implement topological dependency sorting and cycle detection.
    - Implement robust rollback: clean up already-attached modules if any module fails initialization.
  - **Verification Gate**: Missing, cyclic, or failing modules abort safely without leaving dangling resources or event listeners.
  - **Notes / Log**: `engine/app/module_registry.py` provides `ModuleRegistry` (registration, cross-manifest validation, deterministic Kahn topological sort with id tie-break, cycle detection reporting the cycle path) and `ModuleHost` (attach in dependency order, detach in reverse, rollback on failure). `ModuleHost.subscribe()` records host-owned event subscriptions so rollback and detach remove them — no dangling listeners. `ProjectApp.add_module()` now attaches *before* retaining the module, so a failed attach never leaves a half-initialized entry; `ProjectApp.load_modules(registry)` attaches a validated graph with rollback. Render-feature modules dispatch to `ProjectApp.register_render_feature` instead of the gameplay hooks. 22 tests in `tests/test_horizon2_registry.py`.

- [x] **H2.3 — Render Graph Extension Points**
  - **Status**: `DONE`
  - **Reference**: Modular Plan Phase 3 (`P3.1`, `P3.2`)
  - **Scope**:
    - Expose frame context, scene inputs, G-buffer attachments, and pass scheduling contracts in `RenderPipeline`.
    - Validate render graph dependencies and reject missing inputs before execution.
  - **Verification Gate**: Custom render pass can be registered and scheduled via validated interfaces.
  - **Notes / Log**: `RenderPass` gained `inputs`/`outputs`/`after`/`before`/`capabilities`; `RenderGraph` gained `validate()`, `compile()`, `declare_external_input()`, and `remove_pass()`. Validation rejects unknown ordering references, duplicate outputs, missing inputs, and cycles before any GPU work. Passes that declare nothing keep insertion order, so existing behaviour is unchanged. New `engine/gfx/render_services.py` defines the stable renderer surface (`RenderServices`: context, config, frame context, G-buffer, CSM, HDR target, capability queries, texture-unit reservations with the core resolve units marked reserved) and `engine/gfx/render_feature.py` defines `RenderFeature`/`RenderFeatureRegistry`. `RenderPipeline` now builds a validated graph describing its own 19 built-in stages (`_build_render_graph`) and exposes `register_feature`/`unregister_feature`; `ProjectApp` forwards those for the module host. 21 tests in `tests/test_horizon2_render_graph.py`.

- [x] **H2.4 — First Engine Module Extraction: Atmosphere & Sky**
  - **Status**: `DONE`
  - **Reference**: Modular Plan Phase 3 (`P3.3`, `P3.4`)
  - **Scope**:
    - Extract `AtmosphereSystem` into official engine module `pymordial.atmosphere`.
    - Enable, configure, or disable atmosphere via project configuration without editing `pipeline.py`.
  - **Verification Gate**: Projects can toggle or replace the atmosphere pass seamlessly; visual regressions within zero-tolerance.
  - **Notes / Log**: New `engine/modules/atmosphere/` package (`config.py`, `system.py`, `environment.py`, `module.py`) with manifest `pymordial.atmosphere` (role `RENDER_FEATURE`, capabilities `render.sky`/`lighting.environment`/`render.clouds`). New core contract `engine/gfx/environment.py` defines `EnvironmentProvider` plus a `FlatEnvironment` fallback; `create_default_environment()` resolves the provider lazily so a disabled module is never imported. `RenderPipeline` now consumes `self.environment` (with `self.atmosphere` kept as an alias) and no longer reaches into atmosphere config: the ~90-line uniform block became `env.apply_resolve_uniforms()` + `env.apply_sun_direction()`, and sky/cloud rendering became `env.render_sky()`/`env.render_clouds()`. `ProjectConfig` gained `atmosphere_enabled` and `atmosphere_settings` (preset + direct overrides). `engine/gfx/atmosphere.py` remains as a deprecated re-export shim. `EngineTweaks` writes through the provider's public config, so F2 keeps working. Visual regression is gated by `tests/test_horizon2_golden.py` (subprocess capture vs `tests/baseline/`), which stayed at MAE 0.0 through the extraction. 19 tests in `tests/test_horizon2_atmosphere_module.py`, 5 in `tests/test_horizon2_reference_project.py`.

- [x] **H2.5 — Cascaded Shadow Map (CSM) Per-Cascade Frustum Culling & Sloped Bias**
  - **Status**: `DONE`
  - **Scope**:
    - Cull shadow-casting entities against individual cascade view volumes.
    - Implement sloped depth bias:
      $$\text{bias} = \text{clamp}\left(\text{bias}_{\text{base}} \cdot \tan(\arccos(N \cdot L)), \text{bias}_{\text{min}}, \text{bias}_{\text{max}}\right)$$
  - **Verification Gate**: 40–60% vertex reduction on distant shadow cascades; eliminates shadow acne on steep surfaces.
  - **Notes / Log**: Per-cascade culling and the slope-scaled bias were already present from earlier work (`CascadedShadowMap.cull_cascades` + `slope = tan(acos(N·L))` in `deferred_resolve.frag`), so this task was hardening rather than reimplementation. Fixed a zero-allocation violation: `render_frame` built a new `caster_batches` list every frame; it now reuses `_caster_batches_scratch`. The hardcoded `0.5`/`6.0`/`2.0` bias clamps became `RenderConfig.shadow_bias_min_scale`/`shadow_bias_max_scale`/`shadow_bias_slope_scale`, uploaded as `u_ShadowBiasMinScale`/`u_ShadowBiasMaxScale`/`u_ShadowBiasSlopeScale`. Added `stats_shadow_commands`/`stats_shadow_commands_unculled` instrumentation. Note: culling is evaluated **per draw batch**, not per instance, and casters behind a cascade are deliberately retained because they can cast into it. 8 tests in `tests/test_horizon2_csm.py`.

---

### Horizon 3: Player Settings, World Lifecycle & Persistence
* **Goal**: Separate player preferences from developer CVars, establish atomic saves, and upgrade temporal optics.
* **Status**: `PLANNED`
* **Prerequisite**: Horizon 2 completed.

#### Task Breakdown

- [ ] **H3.1 — Typed Player Settings & Safe Display Revert Flow**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 4 (`P4.1`, `P4.2`)
  - **Scope**:
    - Separate player settings (display, quality, audio, keybindings) from engine CVars.
    - Query platform-supported resolutions using `Window.get_supported_resolutions()`.
    - Implement safe display mode change with 15-second auto-revert dialog if unconfirmed.
  - **Verification Gate**: Failing or unconfirmed display changes automatically restore previous working resolution.
  - **Notes / Log**: 

- [ ] **H3.2 — Reusable ModernGL UI Templates**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 4 (`P4.3`)
  - **Scope**:
    - Provide customizable UI templates for Main Menu, Pause Dialog, Settings Screen, and Credits in $1280 \times 720$ canvas space.
  - **Verification Gate**: Screens adapt perfectly to any resolution ($720\text{p}, 1080\text{p}, 1440\text{p}, 4\text{K}, 21:9$) with accurate hit-testing.
  - **Notes / Log**: 

- [ ] **H3.3 — Versioned Preferences & Atomic Save Game API**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 5 (`P5.1`, `P5.2`, `P5.3`)
  - **Scope**:
    - Store player preferences (`user_settings.json`) separately from game progress (`save_game.json`).
    - Implement atomic file writes (`write .tmp` + `atomic rename`) and automatic backup/recovery.
    - Define `SaveParticipant` protocol with versioned schemas and migrations.
  - **Verification Gate**: Interrupted writes never corrupt existing save files; schema migrations upgrade old saves cleanly.
  - **Notes / Log**: 

- [ ] **H3.4 — World Lifecycle & Scene Transitions**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 5 (`P5.4`)
  - **Scope**:
    - Standardize world lifecycle: `load`, `activate`, `deactivate`, `unload`.
    - Guarantee full teardown of physics bodies, entities, draw batches, and event handlers on world switch.
  - **Verification Gate**: Repeated world transitions do not leak GPU memory, physics colliders, or event callbacks.
  - **Notes / Log**: 

- [x] **H3.5 — Depth-Dilated TAA Velocity Fetching & Motion Blur**
  - **Status**: `DONE`
  - **Scope**:
    - In `taa.frag`, sample $3 \times 3$ neighborhood and select velocity of the pixel with closest depth to camera.
    - Implement 2-pass velocity tile-max motion blur (McGuire et al.).
  - **Verification Gate**: Eliminates background edge ghosting behind moving foreground silhouettes; smooth camera/object blur.
  - **Notes / Log**: TAA now does 3x3 depth-dilated **and** magnitude-dilated velocity fetching plus octahedral-normal history rejection. Motion blur moved out of `camera_optics.frag` into `engine/gfx/passes/motion_blur_pass.py`, which reads the G-buffer velocity buffer (RT2) and runs a 2-pass tile-max reconstruction (`motion_blur_tile.frag` + `motion_blur_gather.frag`). It now executes **after** TAA (pass 7.95) instead of before, removing the previous double-blur. New config: `motion_blur_tile_size`, `motion_blur_max_samples`. 4 tests in `tests/test_motion_blur.py`.

---

### Horizon 4: Decima-Grade World Systems & Streaming
* **Goal**: Lift the 100k entity ceiling and deliver seamless open worlds with procedural scatter and physics.
* **Status**: `PLANNED`
* **Prerequisite**: Horizon 3 completed.

#### Task Breakdown

- [ ] **H4.1 — Spatial Sector & Cell Streaming**
  - **Status**: `PLANNED`
  - **Reference**: World Building Roadmap
  - **Scope**:
    - Partition large worlds into configurable grid cells (e.g. $128\text{ m} \times 128\text{ m}$).
    - Asynchronous background streaming worker loading adjacent cells with hysteresis margins.
    - Spatial partitioning of static entity SSBOs: replace fixed global 100k limit with dynamic cell-indexed chunks.
  - **Verification Gate**: Player traversals stream cells in/out with zero frame time spikes during GPU buffer uploads.
  - **Notes / Log**: 

- [ ] **H4.2 — GPU-Driven Clutter & Foliage Indirect Scatter**
  - **Status**: `PLANNED`
  - **Scope**:
    - Compute shader (`foliage_scatter.comp`) evaluating Blue Noise / Poisson distributions across terrain splatmaps.
    - Surviving instance transforms written directly into dynamic instance SSBOs—**zero Python ECS entity allocations**.
    - Shared Gerstner wind vector evaluated in vertex shader (`foliage.vert`).
  - **Verification Gate**: 50,000+ grass/rock instances rendered at $< 0.5\text{ ms}$ GPU cost with zero Python heap allocations.
  - **Notes / Log**: 

- [ ] **H4.3 — BC7 / BC5 Texture Compression & Atlas Residency**
  - **Status**: `PLANNED`
  - **Scope**:
    - Integrate native block compression cooker (`ispc_texcomp` or ctypes wrapper).
    - ModernGL direct loading of BC7 (albedo) and BC5 (tangent normal) texture data.
    - Texture array LRU streaming and residency budgeting to lift the 32-layer cap.
  - **Verification Gate**: 75% VRAM footprint reduction (64 MB down to 16 MB per 4K PBR material set); disk streaming bandwidth reduced by $4\times$.
  - **Notes / Log**: 

- [ ] **H4.4 — Rapier HeightField Colliders & 2-Bone IK Foot Clamping**
  - **Status**: `PLANNED`
  - **Scope**:
    - Wrap Rapier `Collider.heightfield` with 2D elevation grid data matching active terrain chunks.
    - Analytical 2-bone cosine IK in `CharacterMotor` conforming character feet to uneven terrain:
      $$\cos(\theta_{\text{knee}}) = \frac{L_1^2 + L_2^2 - d_{\text{target}}^2}{2 L_1 L_2}$$
  - **Verification Gate**: Character feet conform smoothly to rocky slopes and terrain bumps without capsule penetration.
  - **Notes / Log**: 

- [ ] **H4.5 — 3D Spatialized Audio Acoustics & Occlusion**
  - **Status**: `PLANNED`
  - **Scope**:
    - Distance-based 1-pole Low-Pass Filter (LPF) simulating high-frequency outdoor air absorption:
      $$f_{\text{cutoff}}(d) = f_{\max} \cdot \left(\frac{1}{1 + \alpha_{\text{air}} \cdot d}\right)$$
    - Real-time Rapier geometry raycasts between listener and audio emitters for acoustic obstruction.
  - **Verification Gate**: Distant outdoor sounds sound muffled and bass-heavy; obstructed sounds realistically damp volume and high frequencies.
  - **Notes / Log**: 

---

### Horizon 5: Packaging, CI Qualification & Distribution
* **Goal**: Deliver turnkey production packaging, automated CI testing, and verified release readiness.
* **Status**: `PLANNED`
* **Prerequisite**: Horizons 1–4 completed.

#### Task Breakdown

- [ ] **H5.1 — Deterministic Asset Cooking & Staging Pipeline**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 6 (`P6.1`, `P6.2`, `P6.3`)
  - **Scope**:
    - Explicit build graph resolving transitive engine and project module dependencies.
    - Generate cryptographically verifiable `build_manifest.json` containing engine, module, and cooked asset hashes.
    - Package release assets into zero-copy, memory-mapped `game.pak` archives.
    - Strip dev-only debug overlays (`DebugDraw`, `SystemMonitor`, hotkeys `F1–F3`) in `--release` builds.
  - **Verification Gate**: Rebuilding identical source inputs produces bit-for-bit identical manifests and asset packages.
  - **Notes / Log**: 

- [ ] **H5.2 — Dual-Target Windows Packaging & Headless Smoke Tests**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 6 (`P6.4`)
  - **Scope**:
    - Automated packaging via **Nuitka** (C++ compiled native binary) and **PyInstaller** (turnkey bundle).
    - Headless clean-machine smoke test verifying the packaged binary executes without Python or source dependencies.
  - **Verification Gate**: Standalone binary boots, loads assets from `game.pak`, steps frames, and cleanly shuts down.
  - **Notes / Log**: 

- [ ] **H5.3 — Steam Deck & SteamOS Out-of-the-Box Target**
  - **Status**: `PLANNED`
  - **Scope**:
    - Pre-configure $1280 \times 800$ native resolution and Steam Deck controller action mappings.
    - Profile frame time across passes within the 15W TDP power envelope.
  - **Verification Gate**: Locked 60 FPS ($\approx 12.8\text{ ms}$ total frame time) on Medium tier preset within 15W budget.
  - **Notes / Log**: 

- [ ] **H5.4 — Optional Android Packaging & Capability Fallbacks**
  - **Status**: `PLANNED`
  - **Reference**: [ANDROID_SUPPORT.md](./ANDROID_SUPPORT.md)
  - **Scope**:
    - Verify `build.py --platform android` Chaquopy/Gradle APK pipeline.
    - Verify runtime GLES 3.0 detection cleanly falls back to simple draw loops, 2 shadow cascades, and disables compute GI.
    - Verify on-screen touch overlay in $1280 \times 720$ virtual canvas.
  - **Verification Gate**: APK builds and launches on Android device/emulator with touch controls functional.
  - **Notes / Log**: 

- [ ] **H5.5 — Public Release Qualification & Freeze**
  - **Status**: `PLANNED`
  - **Reference**: Modular Plan Phase 7 (`P7.1`, `P7.2`, `P7.3`)
  - **Scope**:
    - Freeze supported public APIs and document deprecation rules.
    - Run automated soak tests (repeated scene transitions, window resizes, saves, and shutdowns).
    - Audit third-party licenses and finalize developer documentation.
  - **Verification Gate**: Zero known release-blocking bugs; CI green across unit, integration, and packaged tests.
  - **Notes / Log**: 

---

## 4. How to Update Horizon Status (Workflow Protocol)

Follow this procedure during each development session to maintain roadmap synchronization:

1. **Before Starting Work**:
   - Change the selected task status in Section 3 from `PLANNED` to `IN_PROGRESS`.
   - Update Section 1 Horizon Status to `IN_PROGRESS` if not already set.
2. **During Development**:
   - Implement only the single scoped task.
   - Adhere strictly to the non-negotiable rules: Zero-Allocation Inner Loop, Decoupled 60 Hz Physics, Core vs Project Isolation, and $1280 \times 720$ UI Canvas Scaling.
3. **Upon Completion**:
   - Run relevant tests and verify acceptance criteria:
     ```bash
     .\.venv\Scripts\python.exe -m pytest tests/ -q
     ```
   - Mark the task checkbox `[x]` and update status to `DONE`.
   - Add verification notes with commit hash and test results in the **Notes / Log** field.
   - Update the Horizon Progress % in the Section 1 Dashboard.
