# PyMordial Engine: Architecture Reference

> **Audience**: developers modifying engine internals, and AI agents that need an accurate mental model before editing `engine/`.
> **Companion docs**: [ENGINE_FEATURES.md](./ENGINE_FEATURES.md) (what each subsystem does) · [LLM_INSTRUCTIONS.md](./LLM_INSTRUCTIONS.md) (hard rules) · [../guides/DEVELOPER_GUIDE.md](../guides/DEVELOPER_GUIDE.md) (using the engine)

This document describes **how the engine is wired together**: the layer boundaries, the per-frame data flow, the render pass order, and the invariants that will silently break rendering if violated.

---

## 1. Layer Boundaries

The engine is a strict four-layer stack. Dependencies only ever point downward.

```text
┌──────────────────────────────────────────────────────────────┐
│ 4. Projects        projects/<name>/                          │
│    config.py, main.py, modules/, world/, assets/             │
│    Gameplay logic. Never imported by the engine.             │
├──────────────────────────────────────────────────────────────┤
│ 3. App Host        engine/app/                               │
│    ProjectApp, ProjectConfig, ProjectModule, validator       │
│    Owns the loop, wires subsystems, gates on validation.     │
├──────────────────────────────────────────────────────────────┤
│ 2. Subsystems      engine/gfx, physics, audio, ui, input,    │
│                    camera, assets, world, debug, time        │
│    Independent services. Communicate via events + ECS.       │
├──────────────────────────────────────────────────────────────┤
│ 1. Core            engine/core, engine/events, engine/logging│
│    ECS, loop, state machine, math, event bus, logger.        │
│    No knowledge of rendering or gameplay.                    │
└──────────────────────────────────────────────────────────────┘
```

**Enforcement rules**

| Rule | Why it matters |
|---|---|
| `engine/` must never import from `projects/` | Projects are user code; the engine must ship without them. |
| `engine/core/` must never import `engine/gfx/` | Keeps the simulation headless-testable with no GL context. |
| Subsystems talk through `engine.events` or the ECS, not by importing each other's internals | Prevents the circular-import web that makes refactors impossible. |
| `examples/` may import anything; nothing may import `examples/` | Examples are leaves. |

---

## 2. Startup Sequence

`ProjectApp.__init__` is ordered so that cheap validation happens before expensive GPU allocation.

```text
ProjectApp(config)
  │
  ├─ 1. validate_project_config(config).raise_for_errors()
  │      Fails fast on bad resolution / tick rate / preset / gravity /
  │      title / headless+fullscreen conflict. No GPU touched yet.
  │
  ├─ 2. get_quality_preset(config.quality_preset)  → RenderConfig copy
  │      Then overrides: max_entities, particles_enabled
  │
  ├─ 3. RenderContext(...)          ← creates window + GL 4.5 context
  │      Applies Reversed-Z depth func, cull face, viewport.
  │
  ├─ 4. RenderPipeline(render_ctx, quality_cfg)
  │      Builds MegaBuffer, G-Buffer, CSM, all passes, SSBOs.
  │
  ├─ 5. EntityManager(max_entities) + PhysicsManager(gravity)
  │
  ├─ 6. InputManager, EngineLoop(fixed_dt = 1/fixed_hz)
  │
  ├─ 7. AudioEngine, CameraManager, UIManager, GameStateManager
  │
  └─ 8. Debug subsystem (only if enable_debug and not headless)
         SystemMonitor, EngineTweaks, GameTweaks, DebugMenu, DebugToast
```

Step 1 is the contract that makes malformed projects fail with a readable message instead of a driver-level crash. See [project_validator.py](../../engine/app/project_validator.py).

### Window Resize Propagation

`RenderContext.resize()` is the single entry point for a resolution change. It updates the GL viewport **and** forwards the new size to `RenderPipeline.resize()`, which reallocates every intermediate target (G-Buffer, AO + its blur pair, Hi-Z, SSGI, SSR, TAA/SMAA, volumetrics, bloom, HDR, final LDR).

Resizing only the viewport is a bug: the passes keep their old dimensions, so the final blit covers just the bottom-left fraction of the window (AO in particular appears scaled down to roughly half height). `Window.poll_window_events()` translates `pygame.VIDEORESIZE` into a `WindowResizeEvent`, which `RenderContext` (priority 100) and `RenderPipeline` (priority 90) both consume.

---

## 3. Per-Frame Data Flow

`EngineLoop.step_frame(frame_time)` drives one frame in four phases.

```text
                    ┌─────────────────────────────┐
                    │  TimeSystem.tick(frame_time)│
                    │  → real_dt, game_dt         │
                    │  accumulator += game_dt     │
                    └──────────────┬──────────────┘
                                   │
   ┌───────────────────────────────┴───────────────────────────────┐
   │ PHASE 1 — Fixed step (0..N times per frame)                   │
   │   while accumulator >= fixed_dt:                              │
   │     ecs.cache_previous_physics_state()                        │
   │     on_fixed_update(fixed_dt)   ← physics, character motor    │
   │     accumulator -= fixed_dt                                   │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
   ┌───────────────────────────────┴───────────────────────────────┐
   │ PHASE 2 — Variable step (once per frame)                      │
   │   alpha = accumulator / fixed_dt                              │
   │   ecs.interpolate_render_transforms(alpha)                    │
   │   on_update(game_dt, alpha)     ← animation, camera, gameplay │
   │   flush_events()                ← dispatch queued events      │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
   ┌───────────────────────────────┴───────────────────────────────┐
   │ PHASE 3 — Render (once per frame)                             │
   │   on_render(alpha) → ProjectApp gathers draw batches,         │
   │     syncs EngineTweaks → pipeline, calls render_frame(...)    │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
   ┌───────────────────────────────┴───────────────────────────────┐
   │ PHASE 4 — UI (once per frame)                                 │
   │   on_ui() → UIManager.render(), module on_ui() hooks          │
   └───────────────────────────────────────────────────────────────┘
```

**Why physics is fixed-step only.** Rapier's solver is deterministic only for a constant `dt`. Stepping it with a variable delta makes behaviour frame-rate dependent and breaks replay/rollback. `alpha` exists purely so rendering can interpolate between the last two authoritative states.

**`EngineLoop.game_dt`** exposes the current frame's scaled delta so the host can forward a real value to `render_frame(dt=...)` instead of a hardcoded constant.

---

## 4. Render Pass Order

`RenderPipeline.render_frame` executes passes in a fixed order. Several are mutually exclusive.

| # | Pass | Output | Notes |
|---|---|---|---|
| 0 | Clustered lights pre-pass | SSBO 3 | Packs point/spot lights into froxel clusters |
| 1 | Cascaded Shadow Maps | CSM atlas | Per-cascade frustum culling → `cascade_shadow_mdis[i]` |
| 1.5 | Spot Light Shadows | Spot atlas | Reuses `shadow_mdi` (all casters, no per-light culling) |
| 2 | G-Buffer | RT0–RT4 + depth | Reversed-Z, MDI dispatch |
| 2.5 | SSDM | RT3 | Screen-space displacement |
| 2.6 | Decals | HDR | Deferred decal projection |
| 2.7 | Hi-Z pyramid | Hi-Z chain | Feeds occlusion culling + SSR |
| 2.8 | Direct radiance | GI source | Current-frame direct diffuse + emissive for SSGI |
| 3 | Ambient Occlusion | AO mask | GTAO or SSAO per `ao_mode` |
| 4 | Local lights | HDR | Clustered forward+ evaluation |
| 5 | **GI (exclusive)** | HDR | `SSGI` **or** `RMGI` — never both || 6 | IBL + SSR | HDR | Split-sum specular, screen-space reflections |
| 6.5 | Sky & Atmosphere | HDR | Hillaire scattering — provided by the `pymordial.atmosphere` engine module via `EnvironmentProvider` |
| 7 | Deferred resolve | HDR | Combines all lighting into final radiance |
| 7.1 | Auto-exposure | 1×1 EV | Log-luminance histogram + temporal adaptation (`auto_exposure_enabled`) |
| 7.5 | Debug wireframes | HDR | Immediate-mode gizmos |
| 7.6 | Volumetric fog | HDR | Froxel raymarch |
| 7.65 | Water | HDR | Per-entity water surfaces |
| 7.7 | Particles | HDR | GPU compute emitters |
| 7.8 | Camera optics | HDR | DoF + lens flare (`lens_flare_mode`: `ANAMORPHIC` or `SOLAR`) |
| 7.9 | TAA | HDR | Temporal accumulation in linear space |
| 7.95 | Motion blur | HDR | Velocity-buffer tile-max reconstruction (after TAA) |
| 8 | **Post-process (exclusive)** | LDR | Tonemap + exactly one AA: FXAA / SMAA / none |

**Exclusivity invariants**

- Pass 5 runs exactly one GI path. `gi_mode` selects it; there is no "both".
- Pass 8 runs exactly one anti-aliasing path. TAA (7.9) and FXAA/SMAA (8) are mutually exclusive.
- Motion blur (7.95) runs **after** TAA so the temporal filter accumulates a sharp image before the blur is applied once. Running it before TAA double-blurs.
- `u_RMGITexture` is bound at texture unit **15** in `deferred_resolve.frag`. Do not reassign it.
- The Hi-Z pyramid is published as `hiz_texture` / `hiz_mip_count` and consumed by SSR (unit 4) and SSGI (unit 4). Both fall back to the depth texture with a mip count of 1 when the pyramid is unavailable, which disables the accelerated path.
- A second hierarchy, `hiz_bounds_texture` / `hiz_bounds_mip_count`, stores RG32F positive linear-view-depth bounds (R = nearest, G = farthest, empty = nearest > farthest). It is built in the same pass and is currently consumed only by the debug views and the experimental SSR backend. The legacy scalar pyramid is untouched, so existing consumers keep working.
- Pass 2.8 publishes `direct_radiance_texture`: current-frame directional diffuse (with CSM shadows), clustered point/spot diffuse (with spot shadows) and emissive. It excludes all indirect and specular terms, so SSGI sampling it cannot create a recursive feedback loop through the HDR history.
- Ray-Marched Global Illumination uses a per-pixel raymarch (`rmgi_raymarch.comp`) followed by a bilateral/temporal merge (`rmgi_merge.comp`). Its published output is incident irradiance, so the resolve applies receiver albedo exactly once.
- Pass 6.5 is owned by the environment provider. `RenderPipeline` calls `env.render_sky()` /
  `env.render_clouds()` and never reads atmosphere config directly. Disabling
  `ProjectConfig.atmosphere_enabled` swaps in `FlatEnvironment`, which still binds valid LUTs.
- The built-in stage order is mirrored in `RenderPipeline._build_render_graph()` and validated
  at construction. Module render features anchor to those stage names via `after`/`before`.
- GPU pass timing (`RenderPipeline.gpu_profiler`) wraps the G-buffer, Hi-Z, AO, GI, SSR, and
  deferred-resolve submissions with `GL_TIMESTAMP` queries. It is measurement only: it never
  changes pass order, state, or output, and it reads results from older frames so it cannot stall.

---

## 5. Draw Submission: MegaBuffer + MDI

This is the most performance-critical and most easily-broken part of the engine.

### 5.1 The MegaBuffer contract

All geometry lives in **one** vertex buffer and **one** index buffer.

```text
MegaBuffer
├── vbo  : concatenated 32-byte vertices from every registered mesh
├── ibo  : concatenated indices, ALREADY OFFSET into global vertex space
└── allocations[name] = MeshAllocation(first_index, index_count,
                                      base_vertex, vertex_count)
```

When a mesh is added, `_add_mesh` does:

```python
global_indices = (indices + self.total_vertices).astype(np.uint32)
```

**The indices are pre-offset.** This is the single most important fact in the renderer.

### 5.2 The MDI command contract

Because indices are already global, the indirect command's `baseVertex` **must be 0**.

```text
DrawElementsIndirectCommand (20 bytes)
┌──────────────┬────────────────┬─────────────┬─────────────┬───────────────┐
│ count        │ instanceCount  │ firstIndex  │ baseVertex  │ baseInstance  │
│ index_count  │ instance_count │ first_index │ 0 (ALWAYS)  │ base_instance │
└──────────────┴────────────────┴─────────────┴─────────────┴───────────────┘
```

Writing `alloc.base_vertex` into `baseVertex` applies the offset a second time and silently drops the first `base_vertex` vertices of every mesh. The symptom is mangled, half-collapsed geometry — not an error.

Per-instance entity lookup is carried by `baseInstance`, read in the vertex shader as `gl_BaseInstanceARB`:

```glsl
#ifdef GL_ARB_shader_draw_parameters
    uint entity_idx = uint(gl_BaseInstanceARB) + uint(gl_InstanceID);
#else
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
#endif
```

The `u_BaseInstance` uniform fallback is only correct on the per-command `vao.render` path, because a single uniform cannot vary across a multi-draw dispatch.

### 5.3 Fast path vs. fallback

`MultiDrawIndirect.__init__` resolves capability once:

```python
_REQUIRED_EXTENSIONS = ("GL_ARB_multi_draw_indirect", "GL_ARB_shader_draw_parameters")
self.hardware_mdi = all(ext in ctx.extensions for ext in _REQUIRED_EXTENSIONS)
```

- **Fast path** (`hardware_mdi`): one `vao.render_indirect(...)` per batch.
- **Fallback**: one `vao.render(...)` per command, setting `u_BaseInstance` each time.

Never wrap the fast path in `try/except: pass`. A mid-dispatch exception followed by a fallback re-render double-draws the scene.

### 5.4 Draw batch tuple shape

Batches are tuples, and the length encodes optional fields:

| Length | Shape | Meaning |
|---|---|---|
| 3 | `(mesh, count, base_instance)` | Defaults: `is_tess=False`, `cast_shadow=True` |
| 4 | `(mesh, count, base_instance, is_tess)` | Explicit tessellation flag |
| 5 | `(mesh, count, base_instance, is_tess, cast_shadow)` | Full form |

`mesh` is either a `MeshAllocation` or a `str` key into `mega_buffer.allocations`. Consumers must handle both.

---

## 6. ECS Memory Model

`EntityManager` is a Structure-of-Arrays store backed by pre-allocated NumPy arrays.

```text
EntityPool          sparse[dense_idx] ↔ dense[entity_id]  (free-list recycling)
EntityManager
├── world_transforms   (max_entities, 16) float32   → SSBO 1
├── material_data      (max_entities, 8)  float32   → SSBO 2
├── rigid_body_state   (2, max_entities, 7) float32 → prev/current for interpolation
├── aabbs              (max_entities, 6)  float32   → frustum + Hi-Z culling
├── shape_ids          (max_entities,)    uint8
├── is_static          (max_entities,)    bool
└── water_properties   list[WaterProperties | None]
```

**Capacity is fixed at construction.** `max_entities` sizes every array and both SSBOs. `ProjectConfig.max_entities` flows into `RenderConfig.max_entities`, which sizes the SSBOs and the `FrustumCuller` scratch buffers. These must agree — a mismatch means the culler writes past the SSBO.

**Static/dynamic split.** Static entities upload their transform once; dynamic entities re-upload each frame. `_rebuild_mesh_draw_batches` coalesces contiguous same-shape entities into single instanced batches, which is why draw-batch count is far lower than entity count.

---

## 7. Invariants Checklist

Violating any of these produces silent visual corruption rather than an exception.

- [ ] MDI `baseVertex` is always `0` (indices are pre-offset).
- [ ] `RenderConfig.max_entities` matches `EntityManager.max_entities`.
- [ ] Reversed-Z: depth clear is `0.0`, depth func is `GL_GREATER`, near = 1.0.
- [ ] Exactly one GI path runs per frame.
- [ ] Exactly one AA path runs per frame.
- [ ] `u_RMGITexture` stays on texture unit 15.
- [ ] Vertex stride stays 32 bytes (`VERTEX_DTYPE.itemsize` is asserted at import).
- [ ] No `list`/`dict`/`set`/`ndarray` allocation inside `on_fixed_update`, `on_update`, `on_render`.
- [ ] Physics only steps in `on_fixed_update`, never in `on_update`.
- [ ] Game code never consumes `F1`/`F2`/`F3`.

---

## 8. Where to Change What

| You want to change | Edit | Then verify |
|---|---|---|
| A render pass | `engine/gfx/passes/<name>.py` + `pipeline.py` pass order | `pytest tests/test_render_pipeline.py` |
| A quality preset | `engine/gfx/quality_presets.py` `QUALITY_PRESETS` | `pytest tests/test_render_pipeline.py::TestQualityPresets` |
| A tunable exposed in F2 | `engine/debug/engine_tweaks.py` (`__slots__` + defaults + `apply_to_pipeline`) | `pytest tests/test_debug_system.py` |
| Draw submission | `engine/gfx/mdi.py`, `engine/gfx/mega_buffer.py` | `pytest tests/test_horizon1_foundation.py` |
| Culling | `engine/gfx/culling.py`, `engine/gfx/shadow_csm.py` | `pytest tests/test_frustum_culling.py` |
| Physics behaviour | `engine/physics/` | `pytest tests/test_physics_sync.py tests/test_character_controller.py` |
| Project validation | `engine/app/project_validator.py` | `pytest tests/test_horizon1_foundation.py::TestProjectValidation` |
| A new shader | `shaders/` + the owning pass | run the relevant example headless |

**Adding a field to `RenderConfig`**: it must also be added to `EngineTweaks.__slots__`, given a default, and written in `apply_to_pipeline` — otherwise the F2 panel silently does nothing. `RenderConfig` is a plain `@dataclass` (no `__slots__`), so a missing field will *not* raise; it will just create a shadow attribute that nothing reads.
