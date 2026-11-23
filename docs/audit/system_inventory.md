# System Inventory — Phase 1 Audit

> Every system discovered in the repository, with implementation status, primary files, entry
> point, dependencies, outputs, consumers, configurable settings, discovered tests, and major risks.
>
> **Status legend**
> - **complete** — traced end-to-end, exercised by tests, no known gap.
> - **partial** — works, but a documented part of its contract is missing or unverified.
> - **stub** — exists and is importable, but does nothing meaningful.
> - **legacy** — superseded; kept for compatibility.
> - **unknown** — could not be verified in this phase.

---

## A. Core (Layer 1)

### A1. ECS — `EntityManager`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/core/ecs.py](../../engine/core/ecs.py), [engine/core/entity_pool.py](../../engine/core/entity_pool.py) |
| Entry point | `EntityManager(max_entities)`; `create_entity`, `destroy_entity`, `get_transform_proxy` |
| Dependencies | NumPy, `engine.core.materials.MaterialRegistry`, `engine.core.water.WaterProperties` |
| Outputs | SoA arrays: `world_transforms`, `material_data`, `rigid_body_state`, `aabbs`, `shape_ids`, `is_static`; `mesh_draw_batches`, `explicit_mesh_draw_batches` |
| Consumers | `RenderPipeline` (SSBO upload), `PhysicsManager.sync_to_ecs`, `FrustumCuller`, `MapManager`, project modules |
| Settings | `max_entities` (fixed at construction) |
| Tests | `test_ecs_memory.py`, `test_material_id.py`, `test_static_dynamic_separation.py`, `test_physics_sync.py` |
| Risks | Capacity fixed at construction; `max_entities` must match `RenderConfig.max_entities` (no assertion). `_rebuild_mesh_draw_batches` is O(n) on shape change. |

### A2. Entity pool — `EntityPool`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/core/entity_pool.py](../../engine/core/entity_pool.py) |
| Entry point | `EntityPool(max_entities)`; sparse↔dense mapping with free-list recycling |
| Dependencies | NumPy |
| Outputs | `sparse`, `dense`, free list |
| Consumers | `EntityManager` |
| Settings | `max_entities` |
| Tests | `test_ecs_memory.py` |
| Risks | None identified. |

### A3. Engine loop — `EngineLoop`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **partial** |
| Primary files | [engine/core/loop.py](../../engine/core/loop.py) |
| Entry point | `EngineLoop.step_frame(frame_time)` — used. `EngineLoop.run()` — **dead code** |
| Dependencies | `TimeSystem`, `FPSLimiter`, `EntityManager`, `InputManager`, `engine.events.flush_events` |
| Outputs | `fixed_tick_count`, `total_frames`, `fps`, `frame_time_ms`, `game_dt` |
| Consumers | `ProjectApp.step_frame` |
| Settings | `fixed_dt`, `max_accumulator`, `target_fps` |
| Tests | `test_loop_determinism.py`, `test_time_system.py` |
| Risks | `run()` duplicates `ProjectApp.run` with divergent semantics (pause action, FPS limiter). Two loops = two behaviours to keep in sync. |

### A4. Time — `TimeSystem`, `Stopwatch`, `Cooldown`, `FPSLimiter`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/time/time_system.py](../../engine/time/time_system.py) |
| Entry point | `TimeSystem.tick(raw_dt) -> (real_dt, game_dt)` |
| Dependencies | none |
| Outputs | `game_time`, `frame_count`, `fixed_tick_count`, `fps` |
| Consumers | `EngineLoop`, `ProjectApp` |
| Settings | `fixed_dt`, `max_delta_time`, `time_scale` |
| Tests | `test_time_system.py`, `test_loop_determinism.py` |
| Risks | `FPSLimiter` is unused by the real loop. |

### A5. State machine — `GameStateManager`, `GameState`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/core/state.py](../../engine/core/state.py) |
| Entry point | `set_state`, `push_state`, `pop_state`, `handle_event` |
| Dependencies | `ProjectApp` (for `on_enter`/`on_exit`), pygame events |
| Outputs | `current_state`, `stack_depth`; gates `allow_fixed_update`, `allow_variable_update`, `allow_render`, `grab_mouse` |
| Consumers | `ProjectApp` loop callbacks |
| Settings | per-state flags |
| Tests | `test_state_manager.py`, `test_f1_f2_f3_and_mouse_capture.py` |
| Risks | `shutdown` pops states in a `try/except: break` loop — a failing `on_exit` silently abandons the rest of the stack. |

### A6. Event bus — `EventBus`, `Event` hierarchy
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/events/event_bus.py](../../engine/events/event_bus.py), [engine/events/event.py](../../engine/events/event.py) |
| Entry point | `subscribe_event`, `publish_event`, `queue_event`, `flush_events` |
| Dependencies | none |
| Outputs | dispatched events |
| Consumers | `RenderContext`, `RenderPipeline`, `UIManager`, `AudioEngine`, `ModuleHost` |
| Settings | handler priority |
| Tests | `test_event_bus.py` |
| Risks | Global singleton bus; no unsubscribe-on-destroy guarantee outside `ModuleHost`. |

### A7. Logging — `engine.logging`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/logging/logger.py](../../engine/logging/logger.py), [engine/logging/sinks.py](../../engine/logging/sinks.py) |
| Entry point | `get_logger(...)` |
| Dependencies | stdlib `logging` |
| Outputs | `logs/pymordial.log` |
| Consumers | scattered (`ProjectApp` debug init warning) |
| Settings | sinks |
| Tests | `test_logging.py` |
| Risks | Low adoption — most engine code uses `print` or nothing. |

### A8. Math — `engine.core.math_utils`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/core/math_utils.py](../../engine/core/math_utils.py) |
| Entry point | matrix/vector helpers |
| Dependencies | NumPy |
| Outputs | matrices, quaternions |
| Consumers | `FrameContext`, `CameraManager`, `culling`, `shadow_csm` |
| Settings | none |
| Tests | covered indirectly by camera/culling tests |
| Risks | None identified. |

### A9. Materials — `MaterialRegistry`, `MaterialDef`
| Field | Value |
|---|---|
| Kind | non-graphics (data) |
| Status | **complete** |
| Primary files | [engine/core/materials.py](../../engine/core/materials.py), re-export [engine/gfx/material_registry.py](../../engine/gfx/material_registry.py) |
| Entry point | `MaterialRegistry.sync_with_atlas(atlas)` |
| Dependencies | `TextureArrayAtlas` |
| Outputs | material layer indices, flags, displacement depths |
| Consumers | `EntityManager.set_material_registry`, G-Buffer shader |
| Settings | per-material definitions |
| Tests | `test_material_id.py` |
| Risks | `engine/gfx/material_registry.py` is a pure re-export shim (legacy path). |

### A10. Water properties — `WaterProperties`
| Field | Value |
|---|---|
| Kind | non-graphics (data) |
| Status | **complete** |
| Primary files | [engine/core/water.py](../../engine/core/water.py) |
| Entry point | dataclass stored per entity |
| Dependencies | none |
| Outputs | wave/refraction/foam parameters |
| Consumers | `WaterPass` |
| Settings | per-entity |
| Tests | `test_water_pass.py` |
| Risks | None identified. |

### A11. Async loader — `BackgroundAssetLoader`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **partial** |
| Primary files | [engine/core/async_loader.py](../../engine/core/async_loader.py) |
| Entry point | `BackgroundAssetLoader` |
| Dependencies | threading |
| Outputs | loaded assets |
| Consumers | `LoadingScreen` only |
| Settings | worker count |
| Tests | `test_async_loader.py` |
| Risks | Only consumer is `LoadingScreen`, which is itself not used by `ProjectApp`. Effectively an unused feature chain. |

### A12. Legacy input shim — `engine.core.input`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **legacy** |
| Primary files | [engine/core/input.py](../../engine/core/input.py) |
| Entry point | re-exports from `engine.input` |
| Dependencies | `engine.input` |
| Outputs | none |
| Consumers | **none** (zero external references) |
| Settings | n/a |
| Tests | none |
| Risks | Dead module. Safe to remove in a future cleanup phase. |

### A13. Legacy character camera shim — `engine.core.character_camera`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **legacy** |
| Primary files | [engine/core/character_camera.py](../../engine/core/character_camera.py) |
| Entry point | re-exports `FollowCamera` as `CharacterCamera` |
| Dependencies | `engine.camera.follow_camera` |
| Outputs | none |
| Consumers | none in engine; referenced by 3 test/example sites |
| Settings | n/a |
| Tests | indirect |
| Risks | Dead alias. |

---

## B. App Host (Layer 3)

### B1. `ProjectApp`
| Field | Value |
|---|---|
| Kind | both |
| Status | **complete** |
| Primary files | [engine/app/project_app.py](../../engine/app/project_app.py) |
| Entry point | `ProjectApp(config)`; `run()`; `step_frame(dt)`; `shutdown()` |
| Dependencies | every subsystem |
| Outputs | the running application |
| Consumers | `projects/*/main.py`, `examples/*`, `scripts/*` |
| Settings | `ProjectConfig` |
| Tests | `test_horizon1_foundation.py`, `test_playground_project.py`, `test_horizon2_reference_project.py` |
| Risks | Owns the only real loop; `max_entities` read twice; `shutdown` swallows module detach errors. |

### B2. `ProjectConfig`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/app/config.py](../../engine/app/config.py) |
| Entry point | dataclass |
| Dependencies | none |
| Outputs | declarative project settings |
| Consumers | `ProjectApp`, `project_validator`, `create_default_environment` |
| Settings | 18 fields (title, size, vsync, preset, fixed_hz, gravity, debug, particles, headless, max_frames, asset_dir, screenshot, max_entities, atmosphere_*, custom_settings, settings_export_path) |
| Tests | `test_horizon1_foundation.py::TestProjectValidation` |
| Risks | `custom_settings` is an untyped dict — no validation. |

### B3. `ProjectModule` / `GameplayModule` / `RenderFeatureModule` / `ToolModule`
| Field | Value |
|---|---|
| Kind | both |
| Status | **partial** |
| Primary files | [engine/app/module.py](../../engine/app/module.py), [engine/app/module_contracts.py](../../engine/app/module_contracts.py) |
| Entry point | lifecycle hooks |
| Dependencies | `ProjectApp` |
| Outputs | gameplay behaviour |
| Consumers | `ProjectApp._modules`, `ModuleHost` |
| Settings | `name`, `enabled` |
| Tests | `test_horizon2_manifest.py` |
| Risks | `RenderFeatureModule.on_register` is **never called** by `ModuleHost` — the host calls `app.register_render_feature(instance)` instead, which expects a `RenderFeature` (with `create_passes`), not a `RenderFeatureModule` (with `on_register`). The two contracts are incompatible. See [open_questions.md](./open_questions.md) Q2. |

### B4. Module manifest — `ModuleManifest`, `validate_manifest`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/app/module_manifest.py](../../engine/app/module_manifest.py) |
| Entry point | `build_manifest`, `manifest_from_object`, `validate_manifest` |
| Dependencies | `project_validator.ValidationReport` |
| Outputs | validated metadata |
| Consumers | `ModuleRegistry`, `ModuleHost` |
| Settings | id, version, api_version, role, entry, dependencies, capabilities, config_schema |
| Tests | `test_horizon2_manifest.py` |
| Risks | Side-effect free by design (good). |

### B5. Module registry / host — `ModuleRegistry`, `ModuleHost`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/app/module_registry.py](../../engine/app/module_registry.py) |
| Entry point | `register_instance`, `resolve`, `attach_all`, `detach_all` |
| Dependencies | `module_manifest`, `events` |
| Outputs | deterministic attach order, rollback on failure |
| Consumers | `ProjectApp.load_modules` |
| Settings | dependency graph |
| Tests | `test_horizon2_registry.py` |
| Risks | Render-feature attach path is incompatible with `RenderFeatureModule` (see B3). |

### B6. Project validator — `validate_project_config`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/app/project_validator.py](../../engine/app/project_validator.py) |
| Entry point | `validate_project_config(config) -> ValidationReport` |
| Dependencies | none |
| Outputs | errors/warnings |
| Consumers | `ProjectApp.__init__` |
| Settings | n/a |
| Tests | `test_horizon1_foundation.py::TestProjectValidation` |
| Risks | None identified. |

---

## C. Graphics (Layer 2)

### C1. Render context — `RenderContext`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/context.py](../../engine/gfx/context.py) |
| Entry point | `RenderContext(...)`, `create_headless(...)`, `resize`, `destroy` |
| Dependencies | `Window`, ModernGL, `RenderPipeline` (resize forwarding) |
| Outputs | GL context, viewport, `pipeline` back-reference |
| Consumers | everything graphics |
| Settings | width, height, title, hidden, `RenderConfig` |
| Tests | `test_render_pipeline.py`, `test_window_system.py` |
| Risks | Holds a back-reference to `pipeline`; circular ownership (context → pipeline → context). |

### C2. Window — `Window`, `WindowConfig`, `VSyncMode`, `WindowMode`
| Field | Value |
|---|---|
| Kind | graphics (platform) |
| Status | **complete** |
| Primary files | [engine/window/window.py](../../engine/window/window.py) |
| Entry point | `Window(config)`, `set_mode`, `set_resolution`, `set_vsync`, `poll_window_events` |
| Dependencies | pygame-ce |
| Outputs | SDL window, `WindowResizeEvent` etc. |
| Consumers | `RenderContext`, `ProjectApp.run` |
| Settings | size, mode, vsync, resizable |
| Tests | `test_window_system.py` |
| Risks | None identified. |

### C3. Render pipeline — `RenderPipeline`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **partial** |
| Primary files | [engine/gfx/pipeline.py](../../engine/gfx/pipeline.py) (90 KB) |
| Entry point | `render_frame(...)`, `apply_config`, `resize`, `destroy` |
| Dependencies | every pass, `MegaBuffer`, `MultiDrawIndirect`, `GBuffer`, `CascadedShadowMap`, `SpotLightShadowMap`, `TextureArrayAtlas`, `EnvironmentProvider`, `GpuPassProfiler`, `PostProcessPipeline`, `DebugDraw` |
| Outputs | final LDR framebuffer |
| Consumers | `ProjectApp.on_render_step` |
| Settings | `RenderConfig` (≈120 fields) |
| Tests | `test_render_pipeline.py`, `test_phase5_graphics.py`, `test_horizon2_render_graph.py` |
| Risks | **`destroy()` leaks** water pass, spot shadow atlas, spot depth prog/VAO, cascade MDIs, render features. Render features validated but never executed. 90 KB single class with ~200 `__slots__`. |

### C4. Render graph — `RenderGraph`, `RenderPass`, `RenderGraphContext`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **partial** |
| Primary files | [engine/gfx/render_graph.py](../../engine/gfx/render_graph.py) |
| Entry point | `add_pass`, `validate`, `compile`, `execute` |
| Dependencies | none |
| Outputs | validated topological order |
| Consumers | `RenderPipeline._build_render_graph`, `RenderFeatureRegistry.apply_to` |
| Settings | pass `after`/`before`/`inputs`/`outputs` |
| Tests | `test_horizon2_render_graph.py`, `test_phase5_graphics.py` |
| Risks | **`execute()` is never called.** The graph is a validation-only contract. Built-in passes are `_BuiltinPass` no-ops. |

### C5. Render features — `RenderFeature`, `RenderFeatureRegistry`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **partial** |
| Primary files | [engine/gfx/render_feature.py](../../engine/gfx/render_feature.py) |
| Entry point | `register`, `apply_to`, `destroy_all` |
| Dependencies | `RenderGraph`, `RenderServices` |
| Outputs | passes added to the graph |
| Consumers | `RenderPipeline.register_feature` |
| Settings | `name`, `after`, `before`, `requires` |
| Tests | `test_horizon2_render_graph.py` |
| Risks | **Registered features never execute** (graph never executed). `destroy_all()` never called by the pipeline. |

### C6. Render services — `RenderServices`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** (as a contract) |
| Primary files | [engine/gfx/render_services.py](../../engine/gfx/render_services.py) |
| Entry point | `has_capability`, `claim_texture_unit`, `claim_next_texture_unit` |
| Dependencies | ModernGL context |
| Outputs | capability answers, texture-unit reservations |
| Consumers | `RenderFeatureRegistry.apply_to` |
| Settings | `RESERVED_TEXTURE_UNITS`, `FEATURE_TEXTURE_UNITS` |
| Tests | `test_horizon2_render_graph.py` |
| Risks | Contract is sound but its only consumer (render features) never executes. |

### C7. Frame context — `FrameContext` (UBO 0)
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/frame_context.py](../../engine/gfx/frame_context.py) |
| Entry point | `update(...)` |
| Dependencies | ModernGL UBO |
| Outputs | view/proj/vp/inv matrices, jitter, camera pos |
| Consumers | every shader via `common/frame_data.glsl` |
| Settings | none |
| Tests | indirect |
| Risks | None identified. |

### C8. MegaBuffer + MDI
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/mega_buffer.py](../../engine/gfx/mega_buffer.py), [engine/gfx/mdi.py](../../engine/gfx/mdi.py) |
| Entry point | `MegaBuffer.add_mesh`, `MultiDrawIndirect.add_command`, `render_indirect` |
| Dependencies | ModernGL, NumPy |
| Outputs | one VBO + one IBO; indirect command buffers |
| Consumers | G-Buffer, CSM, spot shadows, tessellation |
| Settings | none |
| Tests | `test_horizon1_foundation.py` |
| Risks | Indices are pre-offset; `baseVertex` must be 0. Hardware MDI fallback path exists. |

### C9. G-Buffer — `GBuffer`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/g_buffer.py](../../engine/gfx/g_buffer.py) |
| Entry point | `GBuffer(ctx, w, h, reverse_z)` |
| Dependencies | ModernGL |
| Outputs | RT0 albedo+roughness, RT1 normal+metallic, RT2 motion, RT3 material, RT4 emissive, depth |
| Consumers | resolve, AO, GI, SSR, TAA, SSDM, decals |
| Settings | `reverse_z` |
| Tests | `test_render_pipeline.py` |
| Risks | MRT layout is a hard contract with `gbuffer.frag` and `deferred_resolve.frag`. |

### C10. Culling — `FrustumCuller`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/culling.py](../../engine/gfx/culling.py) |
| Entry point | `cull(...)` |
| Dependencies | NumPy, ECS AABBs |
| Outputs | visible/culled command counts |
| Consumers | `render_frame` (camera + per-cascade shadow) |
| Settings | `frustum_cull_enabled`, `hiz_cull_enabled` |
| Tests | `test_frustum_culling.py` |
| Risks | Scratch buffers sized by `max_entities`; mismatch with ECS is a silent overrun. |

### C11. CSM — `CascadedShadowMap`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/shadow_csm.py](../../engine/gfx/shadow_csm.py) |
| Entry point | `resize_atlas`, `update_splits`, `fbo`, `clear` |
| Dependencies | ModernGL, `culling` |
| Outputs | shadow atlas + per-cascade matrices |
| Consumers | `render_frame` pass 1, `deferred_resolve.frag`, `direct_radiance.frag`, volumetrics |
| Settings | `shadow_resolution`, `csm_cascades`, `shadow_distance`, `csm_split_lambda`, `csm_cascade_distances`, bias fields, `shadow_filter_mode` |
| Tests | `test_shadow_csm.py` (34 KB), `test_csm_stabilization.py`, `test_horizon2_csm.py` |
| Risks | One stale test asserts `u_SunAngularRadius` was removed from the resolve shader; it was not. |

### C12. Spot shadows — `SpotLightShadowMap`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** (functionally) / **leak** |
| Primary files | [engine/gfx/spot_shadows.py](../../engine/gfx/spot_shadows.py) |
| Entry point | `resize_atlas`, `fbo`, `compare_sampler` |
| Dependencies | ModernGL |
| Outputs | spot shadow atlas |
| Consumers | `render_frame` pass 1.5, `direct_radiance.frag`, resolve |
| Settings | `spot_shadows_enabled`, `spot_shadow_resolution`, `spot_shadow_bias` |
| Tests | `test_spot_lights.py` |
| Risks | **Never destroyed** by `pipeline.destroy()`. |

### C13. Texture atlas — `TextureArrayAtlas`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/texture_atlas.py](../../engine/gfx/texture_atlas.py) |
| Entry point | `TextureArrayAtlas(ctx, width, height, max_layers)`, `decode_material_folder` |
| Dependencies | Pillow, ModernGL |
| Outputs | 4 texture arrays (albedo, normal, ORM, height) |
| Consumers | G-Buffer, CSM tess, `MaterialRegistry` |
| Settings | atlas size, `max_layers` |
| Tests | `test_asset_cooker.py` (indirect) |
| Risks | Fixed 2048×2048×32 layers in the pipeline. |

### C14. Shader loading — `shader_utils`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/shader_utils.py](../../engine/gfx/shader_utils.py) |
| Entry point | `load_shader`, `get_shader_dir`, `expand_includes` |
| Dependencies | filesystem, `sys._MEIPASS` |
| Outputs | expanded GLSL source |
| Consumers | every pass |
| Settings | `PYMORDIAL_SHADER_DIR` env var |
| Tests | indirect (all pass tests compile shaders) |
| Risks | Global `_SHADER_CACHE` never invalidated except by `clear_shader_cache()`. |

### C15. Quality presets — `RenderConfig`, `QUALITY_PRESETS`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/quality_presets.py](../../engine/gfx/quality_presets.py) |
| Entry point | `get_quality_preset(GraphicsQuality)` |
| Dependencies | none |
| Outputs | `RenderConfig` copy |
| Consumers | `ProjectApp.__init__`, `EngineTweaks` |
| Settings | 6 presets: CUSTOM, LOW, MEDIUM, HIGH, ULTRA, CINEMATIC |
| Tests | `test_render_pipeline.py::TestQualityPresets`, `test_camera_optics.py` |
| Risks | `RenderConfig` is a plain dataclass (no `__slots__`), so a field missing from `EngineTweaks` silently creates a shadow attribute. One stale test expects `film_grain_enabled=True` on LOW. |

### C16. Post-process — `PostProcessPipeline`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/post_process.py](../../engine/gfx/post_process.py) |
| Entry point | `render(...)`, `final_fbo`, `hdr_fbo` |
| Dependencies | ModernGL, `post_process.frag`, `fxaa.frag`, `smaa_*` |
| Outputs | LDR final texture |
| Consumers | `ProjectApp.on_ui_step` (copy to screen) |
| Settings | `tonemap_mode`, `exposure`, `contrast`, `saturation`, lift/gamma/gain, `white_balance_*`, `chromatic_aberration_*`, `vignette_*`, `film_grain_*`, `aa_mode` |
| Tests | `test_tonemap.py`, `test_camera_optics.py` |
| Risks | Owns HDR + LDR targets; resize must reallocate both. |

### C17. GPU profiler — `GpuPassProfiler`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/gpu_profiler.py](../../engine/gfx/gpu_profiler.py) |
| Entry point | `begin_frame`, `begin(name)`, `end`, `get_ms`, `results` |
| Dependencies | `GL_TIMESTAMP` queries |
| Outputs | per-pass GPU ms (from older frames) |
| Consumers | `DebugMenu` (F1 profiler), `SystemMonitor` |
| Settings | `frame_lag=3`, `enabled` |
| Tests | `test_gpu_profiler.py` |
| Risks | Disables itself if timer queries are unavailable — correct behaviour, but means GPU timings can silently be absent. |

### C18. Environment provider — `EnvironmentProvider`, `FlatEnvironment`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/gfx/environment.py](../../engine/gfx/environment.py) |
| Entry point | `create_default_environment(ctx, config)` |
| Dependencies | `engine.modules.atmosphere` (lazy import) |
| Outputs | sky LUTs, sun vector, lighting tuple |
| Consumers | `RenderPipeline` (pass 6.5, resolve uniforms) |
| Settings | `atmosphere_enabled`, `atmosphere_settings` |
| Tests | `test_horizon2_atmosphere_module.py` |
| Risks | Lazy import means a broken atmosphere module silently degrades to `FlatEnvironment` (broad `except Exception`). |

### C19. Debug draw — `DebugDraw`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/debug/debug_draw.py](../../engine/debug/debug_draw.py) |
| Entry point | `line`, `box`, `sphere`, `render` |
| Dependencies | ModernGL |
| Outputs | immediate-mode wireframes into HDR |
| Consumers | `render_frame` pass 7.5 |
| Settings | none |
| Tests | `test_debug_system.py` |
| Risks | None identified. |

### C20. HUD overlay — `HudOverlay`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **legacy** |
| Primary files | [engine/gfx/hud_overlay.py](../../engine/gfx/hud_overlay.py) |
| Entry point | `HudOverlay(ctx, ...)`, `render(...)` |
| Dependencies | ModernGL |
| Outputs | 2D panel |
| Consumers | **none in runtime**; only `tests/test_render_pipeline.py` and `engine/gfx/__init__.py` |
| Settings | panel size |
| Tests | `test_render_pipeline.py::test_hud_overlay_render` |
| Risks | Superseded by `UIManager` + `DebugMenu`. Dead in the app. |

### C21. Loading screen — `LoadingScreen`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **partial** |
| Primary files | [engine/gfx/loading_screen.py](../../engine/gfx/loading_screen.py) |
| Entry point | `LoadingScreen(...)` |
| Dependencies | `BackgroundAssetLoader`, `TextureArrayAtlas` |
| Outputs | loading UI |
| Consumers | `scripts/run_pom_plane.py`, `scripts/run_validation.py`, tests — **not** `ProjectApp` |
| Settings | none |
| Tests | `test_async_loader.py` |
| Risks | Not wired into the app lifecycle; a project must drive it manually. |

### C22. Volumetric settings — `VolumetricSettings`
| Field | Value |
|---|---|
| Kind | graphics (data) |
| Status | **partial** |
| Primary files | [engine/gfx/volumetrics.py](../../engine/gfx/volumetrics.py) |
| Entry point | `VolumetricSettings.from_render_config(config)` |
| Dependencies | none |
| Outputs | CPU-side fog tuning |
| Consumers | `tests/test_volumetrics_common.py` only |
| Settings | density, height_falloff, anisotropy, ambient, distance |
| Tests | `test_volumetrics_common.py` |
| Risks | Mirrors `RenderConfig` fog fields but is not used by any pass — the passes read `RenderConfig` directly. Duplicate source of truth. |

### C23. Deprecated atmosphere shim — `engine.gfx.atmosphere`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **legacy** |
| Primary files | [engine/gfx/atmosphere.py](../../engine/gfx/atmosphere.py) |
| Entry point | re-exports `AtmosphereConfig`, `AtmosphereSystem`, `ATMOSPHERE_PRESETS` |
| Dependencies | `engine.modules.atmosphere` |
| Outputs | none |
| Consumers | 3 test files only |
| Settings | n/a |
| Tests | `test_atmosphere.py`, `test_horizon2_atmosphere_module.py`, `test_sky_atmosphere_pass.py` |
| Risks | Dead in runtime; keeps a second import path alive. |

---

## D. Graphics Passes (`engine/gfx/passes/`)

All passes are constructed by `RenderPipeline.__init__` and executed by `render_frame`. Status is
**complete** unless noted.

| Pass | File | Entry | Output | Gate | Tests |
|---|---|---|---|---|---|
| Clustered lights | `clustered_lights.py` | `execute(ctx)` | SSBO 3 froxel clusters | always | `test_phase5_graphics.py`, `test_spot_lights.py` |
| IBL | `ibl.py` | `execute(ctx)` | BRDF LUT, env map, irradiance | always | `test_energy_brdf.py` |
| AO | `ao_pass.py` | `execute(ctx)` | AO mask | `ao_mode` | `test_phase5_graphics.py` |
| SSGI | `ssgi_pass.py` | `execute(ctx)` | GI texture | `gi_mode=="SSGI"` | `test_ssgi_reconstruction.py` |
| Radiance cascades | `rmgi_pass.py` + `rc_cascade.py` | `execute(ctx)` | irradiance | `gi_mode=="RMGI"` | `test_rmgi.py`, `test_rmgi_cascades.py`, `test_rc_intervals.py` |
| SSR | `ssr_pass.py` | `execute(ctx)` | reflections | `ssr_enabled` | `test_phase5_graphics.py` |
| SSDM | `ssdm_pass.py` | `execute(...)` | RT3 displacement | `ssdm_enabled` | `test_phase5_graphics.py` |
| Decals | `decal_pass.py` | `execute(ctx, g_buffer)` | HDR | always | `test_phase5_graphics.py` |
| Hi-Z | `hiz_pass.py` | `execute(ctx)` | depth pyramid + bounds | always | `test_hiz_pass.py`, `test_hiz_bounds*.py`, `test_hiz_tracer.py` |
| Direct radiance | `direct_radiance_pass.py` | `execute(ctx)` | GI source | always | `test_phase5_graphics.py` |
| Volumetric fog | `volumetric_fog_pass.py` | `execute(ctx)` | fog composite | `volumetric_fog_enabled` | `test_volumetrics_common.py` |
| Water | `water_pass.py` | `render_water(...)` | HDR | `water_pass is not None` | `test_water_pass.py` |
| Particles | `particle_pass.py` | `render(...)` | HDR | `particle_pass is not None` | `test_particle_system.py` |
| Camera optics | `camera_optics_pass.py` | `render_optics`, `render_flare` | HDR | `camera_optics_pass is not None` | `test_camera_optics.py` |
| Solar flare | `solar_flare_pass.py` | `render(...)` | flare texture | `lens_flare_mode=="SOLAR"` | `test_solar_flare.py` |
| Motion blur | `motion_blur_pass.py` | `execute(ctx)` | HDR | `motion_blur_enabled` | `test_motion_blur.py`, `test_motion_vectors.py` |
| TAA | `taa_pass.py` | `execute(ctx)` | HDR history | `taa_active` | `test_taa.py`, `test_temporal_pingpong.py`, `test_aa_stability.py` |
| FXAA | `fxaa_pass.py` | `execute(ctx)` | LDR | `aa_mode=="FXAA"` | `test_phase5_graphics.py` |
| SMAA | `smaa_pass.py` | `execute(ctx)` | LDR | `aa_mode=="SMAA"` | `test_temporal_pingpong.py` |
| Bloom | `bloom_pass.py` | `render(...)` | bloom texture | `bloom_enabled` | `test_camera_optics.py` |
| Exposure | `exposure_pass.py` | `execute(ctx)` | 1×1 EV | `auto_exposure_enabled` | `test_exposure.py` |
| Blur (shared) | `blur_pass.py` | `BlurPass` | blurred texture | used by AO/SSGI/SSR | `test_blur_common.py` |
| Sky atmosphere | `sky_atmosphere_pass.py` | owned by atmosphere module | sky LUTs | provider | `test_sky_atmosphere_pass.py` |
| Sky cloud | `sky_cloud_pass.py` | owned by atmosphere module | cloud composite | provider | `test_sky_atmosphere_pass.py` |

**Note**: `engine/gfx/passes/__init__.py` is an empty docstring — passes are imported by module path,
not re-exported. This is fine but means there is no single place listing all passes.

---

## E. Physics

### E1. `PhysicsManager` (Rapier wrapper)
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/physics/rapier_world.py](../../engine/physics/rapier_world.py), [native/pymordial_rapier/src/lib.rs](../../native/pymordial_rapier/src/lib.rs) |
| Entry point | `create_body`, `attach_*_collider`, `step_simulation`, `sync_to_ecs`, `raycast` |
| Dependencies | `pymordial_rapier` native module (Rust), NumPy |
| Outputs | body transforms → ECS |
| Consumers | `ProjectApp.on_fixed_step` |
| Settings | gravity |
| Tests | `test_physics_sync.py`, `test_interactive_gameplay_physics.py` |
| Risks | No `destroy()`; relies on GC. Native module must be built (`scripts/build_native.py`). |

### E2. Character motor — `CharacterMotor`, `KinematicCharacterMotor`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/physics/character_motor.py](../../engine/physics/character_motor.py), [engine/physics/character_controller.py](../../engine/physics/character_controller.py) |
| Entry point | `CharacterMotor(config)`, `move(...)` |
| Dependencies | `PhysicsManager` |
| Outputs | kinematic movement |
| Consumers | project modules |
| Settings | `CharacterMotorConfig` |
| Tests | `test_character_controller.py` |
| Risks | `character_controller.py` is a thin alias module. |

### E3. Props / interaction — `engine.physics.props`, `interaction`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/physics/props.py](../../engine/physics/props.py), [engine/physics/interaction.py](../../engine/physics/interaction.py) |
| Entry point | prop factories, interaction queries |
| Dependencies | `PhysicsManager`, ECS |
| Outputs | physics props |
| Consumers | projects, `MapManager` |
| Settings | per-prop |
| Tests | `test_interactive_gameplay_physics.py` |
| Risks | None identified. |

### E4. Vehicle — `RaycastVehicle`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/physics/vehicle/raycast_vehicle.py](../../engine/physics/vehicle/raycast_vehicle.py) (46 KB), `vehicle_config.py` |
| Entry point | `RaycastVehicle(config)` |
| Dependencies | `PhysicsManager` |
| Outputs | vehicle dynamics |
| Consumers | projects (none currently) |
| Settings | `VehicleConfig` |
| Tests | `test_raycast_vehicle.py` |
| Risks | Large, untested-in-app subsystem. |

---

## F. Input

### F1. `InputManager`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/input/input_manager.py](../../engine/input/input_manager.py) (33 KB), `actions.py`, `codes.py`, `context.py`, `haptics.py`, `rebinding.py` |
| Entry point | `poll_events`, `is_action_*`, `get_axis`, `get_vector2`, `set_mouse_grab` |
| Dependencies | pygame-ce |
| Outputs | action/axis state, event list |
| Consumers | `ProjectApp.run`, project modules |
| Settings | bindings, contexts |
| Tests | `test_input_system.py`, `test_input_contexts.py`, `test_example_controls.py` |
| Risks | `_was_mouse_grabbed` is a private attribute mutated by `ProjectApp` — encapsulation leak. |

---

## G. Audio

### G1. `AudioEngine`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/audio/audio_engine.py](../../engine/audio/audio_engine.py), `bus.py`, `sound.py`, `spatial.py`, `voice_pool.py` |
| Entry point | `get_audio_engine()`, `play_sound`, `update`, `stop_all` |
| Dependencies | pygame-ce mixer |
| Outputs | mixed audio |
| Consumers | `ProjectApp.on_variable_step` |
| Settings | `max_channels`, buses |
| Tests | `test_audio_engine.py` |
| Risks | Global singleton; `SDL_AUDIODRIVER=dummy` in tests. |

---

## H. UI

### H1. `UIManager`, `UIScreen`, widgets
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/ui/manager.py](../../engine/ui/manager.py), `elements.py`, `renderer.py`, `shaders.py`, `theme.py` |
| Entry point | `UIManager(ctx, w, h)`, `set_screen`, `handle_event`, `update`, `render` |
| Dependencies | ModernGL, `WindowResizeEvent` |
| Outputs | 2D UI into final FBO |
| Consumers | `ProjectApp.on_ui_step` |
| Settings | 1280×720 reference canvas, theme |
| Tests | `test_ui_subsystem.py` |
| Risks | Reference-resolution scaling must stay consistent with hit-testing. |

---

## I. Camera

### I1. `CameraManager`, `VirtualCamera`, `FreeCamera`, `FollowCamera`, `OrbitCamera`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/camera/](../../engine/camera/) |
| Entry point | `CameraManager.update(dt)`, `active_camera` |
| Dependencies | `InputManager` |
| Outputs | position, target, fov, forward |
| Consumers | `ProjectApp.on_render_step`, audio listener, `MapManager` |
| Settings | per-camera |
| Tests | `test_virtual_camera.py`, `test_example_controls.py` |
| Risks | None identified. |

---

## J. World / Mapping

### J1. `BaseWorldBuilder`, `DefaultWorldBuilder`
| Field | Value |
|---|---|
| Kind | both |
| Status | **complete** |
| Primary files | [engine/world/base.py](../../engine/world/base.py), [engine/world/default_world.py](../../engine/world/default_world.py) |
| Entry point | `build_world(app)`, `teardown_world(app)` |
| Dependencies | `ProjectApp`, `MegaBuffer` |
| Outputs | static geometry, colliders, draw batches |
| Consumers | `ProjectApp.set_world_builder` |
| Settings | size, grid |
| Tests | `test_horizon2_reference_project.py` |
| Risks | None identified. |

### J2. Mapping system — `MapManager`, `MapDef`, `SectorInstance`, `MapEntityNode`, `PrefabManager`, `TriggerSystem`, `StreamingDirector`, `ScriptRunner`
| Field | Value |
|---|---|
| Kind | both |
| Status | **complete** |
| Primary files | [engine/world/mapping/](../../engine/world/mapping/) |
| Entry point | `MapManager(app)`, `load_map`, `on_fixed_update`, `signal_entity`, `trigger_interaction` |
| Dependencies | ECS, physics, `ProjectApp` |
| Outputs | entities, triggers, streaming |
| Consumers | `ProjectApp.on_fixed_step` |
| Settings | map JSON schema |
| Tests | `test_mapping_schema.py`, `test_mapping_scripts.py`, `test_map_streaming.py`, `test_sector_lifecycle.py`, `test_prefab_spawning.py`, `test_trigger_system.py` |
| Risks | Large surface; `schema.py` is 22 KB. |

### J3. Waypoint graph — `WaypointGraph`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **stub** (unwired) |
| Primary files | [engine/world/waypoint_graph.py](../../engine/world/waypoint_graph.py) |
| Entry point | `WaypointGraph` |
| Dependencies | none |
| Outputs | navigation graph |
| Consumers | **none** — only `ProjectApp.waypoint_graph` property returns `getattr(world_builder, "waypoint_graph", None)`, and no builder defines it |
| Settings | n/a |
| Tests | none |
| Risks | Dead subsystem. `ProjectApp.waypoint_graph` always returns `None` in practice. |

---

## K. Assets

### K1. VFS — `VFS`, `PakReader`, `PakWriter`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/assets/vfs.py](../../engine/assets/vfs.py) |
| Entry point | `mount_pak`, `mount_dir`, `read`, `list_files` |
| Dependencies | zstandard, mmap |
| Outputs | virtual file reads |
| Consumers | `ResourceCache`, `packager` |
| Settings | pak paths |
| Tests | `test_asset_cooker.py` |
| Risks | None identified. |

### K2. Mesh/texture formats — `PMMesh`, `PMTex`, `build_pm_mesh`, `cook_image_to_pm_tex`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/assets/mesh_format.py](../../engine/assets/mesh_format.py), `texture_format.py` |
| Entry point | builders |
| Dependencies | NumPy, Pillow |
| Outputs | 32-byte vertex format, prefiltered textures |
| Consumers | `MegaBuffer`, `TextureArrayAtlas` |
| Settings | none |
| Tests | `test_asset_cooker.py` |
| Risks | `VERTEX_STRIDE` is asserted at import — good. |

### K3. Resource cache — `ResourceCache`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/assets/resource_cache.py](../../engine/assets/resource_cache.py) |
| Entry point | `load_mesh`, `load_texture`, `load_gpu_texture`, `register_gpu_mesh` |
| Dependencies | `VFS` |
| Outputs | cached assets |
| Consumers | `RenderPipeline.resources` |
| Settings | none |
| Tests | indirect |
| Risks | `pipeline.destroy()` calls `resources.close()`. |

### K4. Cooker — `engine.assets.cooker`
| Field | Value |
|---|---|
| Kind | tool |
| Status | **complete** |
| Primary files | [engine/assets/cooker.py](../../engine/assets/cooker.py) |
| Entry point | `main()` (console script `pymordial-cooker`) |
| Dependencies | Pillow, zstandard |
| Outputs | `.pm` mesh/texture files, `.pak` |
| Consumers | CLI |
| Settings | CLI args |
| Tests | `test_asset_cooker.py` |
| Risks | None identified. |

---

## L. Debug (Sacred Subsystem)

### L1. `SystemMonitor`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/debug/monitor.py](../../engine/debug/monitor.py) |
| Entry point | `record_stage`, `begin_gpu_frame`, `end_gpu_frame`, `update_runtime_stats` |
| Dependencies | NumPy, `GpuPassProfiler` |
| Outputs | CPU/GPU timings, 1% low, memory gauges |
| Consumers | `ProjectApp` loop callbacks, `DebugMenu` |
| Settings | `history_size`, `target_fps` |
| Tests | `test_debug_system.py` |
| Risks | None identified. |

### L2. `EngineTweaks` (F2)
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/debug/engine_tweaks.py](../../engine/debug/engine_tweaks.py) (64 KB) |
| Entry point | `apply_to_pipeline(pipeline)`, `sync_from_pipeline`, `to_dict`/`from_dict`, `save_to_file`/`load_from_file` |
| Dependencies | `RenderPipeline`, `RenderConfig` |
| Outputs | live graphics settings |
| Consumers | `ProjectApp.on_render_step` (every frame), `DebugMenu` |
| Settings | mirrors ~120 `RenderConfig` fields |
| Tests | `test_debug_system.py`, `test_f1_f2_f3_and_mouse_capture.py` |
| Risks | Must be kept in sync with `RenderConfig` manually; a missing field silently no-ops. |

### L3. `GameTweaks` (F3)
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/debug/game_tweaks.py](../../engine/debug/game_tweaks.py) |
| Entry point | `TweakItem`, `TweakType` |
| Dependencies | none |
| Outputs | declarative developer inspector |
| Consumers | `DebugMenu`, project modules |
| Settings | per-tweak |
| Tests | `test_debug_system.py` |
| Risks | None identified. |

### L4. `DebugMenu` (F1/F2/F3 overlay)
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/debug/debug_menu.py](../../engine/debug/debug_menu.py) (102 KB) |
| Entry point | `render()`, `process_event`, `cycle_f1`, `toggle_graphics`, `toggle_game_tweaks` |
| Dependencies | imgui-bundle, `SystemMonitor`, `EngineTweaks`, `GameTweaks`, `DebugToast` |
| Outputs | ImGui overlay on backbuffer |
| Consumers | `ProjectApp.run` (hotkeys), `ProjectApp.on_ui_step` (render) |
| Settings | `settings_export_path` |
| Tests | `test_debug_system.py`, `test_f1_f2_f3_and_mouse_capture.py`, `test_debug_backends.py` |
| Risks | 102 KB single file. Writes `settings.ini` via `ini_export`. |

### L5. `DebugToast`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/debug/toast.py](../../engine/debug/toast.py) |
| Entry point | `show(msg, duration, color)` |
| Dependencies | none |
| Outputs | transient notification |
| Consumers | `DebugMenu`, `ProjectApp` |
| Settings | none |
| Tests | `test_debug_system.py` |
| Risks | None identified. |

---

## M. Config

### M1. `CVarRegistry`, `CVar`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **stub** (unwired) |
| Primary files | [engine/config/registry.py](../../engine/config/registry.py), `cvar.py` |
| Entry point | `get_cvar_registry`, `get_cvar`, `set_cvar_value` |
| Dependencies | none |
| Outputs | typed console variables |
| Consumers | **none in runtime** (only `tests/test_cvars.py`) |
| Settings | registered cvars |
| Tests | `test_cvars.py` |
| Risks | Complete subsystem with zero runtime integration. Two competing settings systems (`EngineTweaks` vs `CVarRegistry`). |

### M2. INI export — `engine.config.ini_export`
| Field | Value |
|---|---|
| Kind | non-graphics |
| Status | **complete** |
| Primary files | [engine/config/ini_export.py](../../engine/config/ini_export.py) |
| Entry point | `write_ini`, `read_ini`, `format_value`, `parse_value`, `section_for_key` |
| Dependencies | none |
| Outputs | `settings.ini` |
| Consumers | `DebugMenu` (save settings) |
| Settings | n/a |
| Tests | `test_config_ini_export.py` |
| Risks | None identified. |

---

## N. Tools

### N1. Packager — `engine.tools.packager`
| Field | Value |
|---|---|
| Kind | tool |
| Status | **complete** |
| Primary files | [engine/tools/packager.py](../../engine/tools/packager.py) |
| Entry point | `main()` (console script `pymordial-packager`) |
| Dependencies | PyInstaller / Nuitka |
| Outputs | release binary + `game.pak` |
| Consumers | CLI |
| Settings | `--project`, `--target`, `--release` |
| Tests | `test_project_framework_and_packaging.py` |
| Risks | Release mode strips `debug_overlay` and tree-shakes modules — must not strip the sacred debug subsystem in dev builds. |

### N2. Project tool — `engine.tools.project`
| Field | Value |
|---|---|
| Kind | tool |
| Status | **complete** |
| Primary files | [engine/tools/project.py](../../engine/tools/project.py) |
| Entry point | `main()` (console script `pymordial-project`) |
| Dependencies | none |
| Outputs | project scaffolding / run |
| Consumers | CLI |
| Settings | CLI args |
| Tests | `test_project_framework_and_packaging.py` |
| Risks | None identified. |

---

## O. Engine Modules

### O1. Atmosphere module — `pymordial.atmosphere`
| Field | Value |
|---|---|
| Kind | graphics |
| Status | **complete** |
| Primary files | [engine/modules/atmosphere/](../../engine/modules/atmosphere/) |
| Entry point | `create_atmosphere_environment(ctx, settings)` |
| Dependencies | `SkyAtmospherePass`, `SkyCloudPass`, `AtmosphereSystem`, `GlobalLightSystem` |
| Outputs | sky LUTs, sun/moon vectors, lighting tuple, resolve uniforms |
| Consumers | `RenderPipeline.environment` |
| Settings | `AtmosphereConfig` (presets, time_of_day, day_speed, azimuth, latitude, declination, cloud_enabled) |
| Tests | `test_atmosphere.py`, `test_horizon2_atmosphere_module.py`, `test_sky_atmosphere_pass.py`, `test_sun_kinematics.py` |
| Risks | `GlobalLightSystem` is used by `AtmosphereSystem` but is a separate class — two lighting models in one module. |

---

## P. Shaders (`shaders/`)

**Confirmed**: 100 shader files. All are referenced by Python **except** four that are
`#include`-only libraries:

| File | Role |
|---|---|
| `atmosphere.glsl` | included by `deferred_resolve.frag` |
| `hillaire_common.glsl` | included by `atmosphere.glsl` and `hillaire_*.frag` |
| `frame_data.glsl` | included by `camera_optics.frag`, `motion_blur_gather.frag`, `particle_render.*`, `solar_flare.frag` |

These are **not** dead — they are include libraries. No orphaned shader was found.

**Test-only shaders** (referenced only by `tests/`): `depth_util_test.comp`, `hiz_bounds_test.comp`,
`hiz_trace_test.comp`. These are legitimate test fixtures.

---

## Q. Tests

**795 tests collected**, **793 pass**, **2 fail** (both stale expectations — see
[engine_architecture.md](./engine_architecture.md) §14).

Test files by area:

| Area | Files |
|---|---|
| Core | `test_ecs_memory`, `test_loop_determinism`, `test_time_system`, `test_state_manager`, `test_event_bus`, `test_logging`, `test_cvars` |
| App | `test_horizon1_foundation`, `test_horizon2_manifest`, `test_horizon2_registry`, `test_project_framework_and_packaging`, `test_playground_project`, `test_horizon2_reference_project` |
| Graphics | `test_render_pipeline`, `test_phase5_graphics`, `test_horizon2_render_graph`, `test_horizon2_golden`, `test_golden` |
| Shadows | `test_shadow_csm`, `test_csm_stabilization`, `test_horizon2_csm`, `test_spot_lights` |
| GI/AO/SSR | `test_ssgi_reconstruction`, `test_rmgi`, `test_hiz_*` (3 files) |
| AA/Post | `test_taa`, `test_temporal_pingpong`, `test_aa_stability`, `test_tonemap`, `test_exposure`, `test_blur_common`, `test_motion_blur`, `test_motion_vectors` |
| Optics | `test_camera_optics`, `test_solar_flare` |
| Atmosphere | `test_atmosphere`, `test_horizon2_atmosphere_module`, `test_sky_atmosphere_pass`, `test_sun_kinematics` |
| Water/Particles | `test_water_pass`, `test_particle_system` |
| Physics | `test_physics_sync`, `test_character_controller`, `test_interactive_gameplay_physics`, `test_raycast_vehicle` |
| Input | `test_input_system`, `test_input_contexts`, `test_example_controls` |
| Audio | `test_audio_engine` |
| UI | `test_ui_subsystem` |
| World | `test_mapping_schema`, `test_mapping_scripts`, `test_map_streaming`, `test_sector_lifecycle`, `test_prefab_spawning`, `test_trigger_system` |
| Assets | `test_asset_cooker`, `test_async_loader` |
| Debug | `test_debug_system`, `test_debug_backends`, `test_f1_f2_f3_and_mouse_capture`, `test_gpu_profiler` |
| Misc | `test_material_id`, `test_static_dynamic_separation`, `test_frustum_culling`, `test_depth_util`, `test_energy_brdf`, `test_surface_texture_stretching_bug`, `test_virtual_camera`, `test_window_system`, `test_examples` |

---

## R. Generated / Vendored Content (not engine-owned)

| Path | Nature | Audit treatment |
|---|---|---|
| `build/` | PNG probe outputs, `validation_assets/` | Not audited |
| `dist/` | empty | Not audited |
| `logs/pymordial.log` | runtime log | Not audited |
| `native/pymordial_rapier/` | Rust crate (source is engine-owned; `Cargo.lock` is generated) | Python wrapper audited; Rust internals not |
| `scripts/run_validation.py`, `scripts/run_pom_plane.py` | developer harnesses (81 KB / 52 KB) | Not audited |
| `_ao_probe.py` | temporary probe script | Not audited |
| `imgui.ini` | ImGui layout state | Not audited |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | tool caches | Not audited |
| `examples/*.json` | example level data | Not audited |
| `projects/playground/assets/maps/*.json` | project data | Not audited |

---

## S. Summary Counts

| Status | Count | Systems |
|---|---|---|
| complete | 44 | ECS, pool, time, state, events, logging, math, materials, water props, ProjectApp, ProjectConfig, manifest, registry, validator, context, window, render services, frame context, MegaBuffer/MDI, G-Buffer, culling, CSM, texture atlas, shader utils, presets, post-process, GPU profiler, environment, debug draw, all 24 passes, physics, character motor, props, vehicle, input, audio, UI, cameras, world builder, mapping, VFS, formats, resource cache, cooker, monitor, EngineTweaks, GameTweaks, DebugMenu, DebugToast, ini export, packager, project tool, atmosphere module |
| partial | 8 | EngineLoop, async loader, RenderPipeline, RenderGraph, RenderFeature, ProjectModule contracts, LoadingScreen, VolumetricSettings |
| stub | 2 | WaypointGraph, CVarRegistry |
| legacy | 4 | `engine.core.input`, `engine.core.character_camera`, `HudOverlay`, `engine.gfx.atmosphere` |
| unknown | 0 | — |
