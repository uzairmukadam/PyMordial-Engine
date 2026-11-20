# Open Questions — Phase 1 Audit

> Questions that could not be resolved by static inspection alone, or that require a design
> decision from the maintainer. Each entry states the evidence, the impact, and the smallest
> diagnostic that would settle it.
>
> **Nothing here was fixed in Phase 1.** These are inputs to Phase 2 planning.

---

## Q1. Is `max_entities` guaranteed to match between `RenderConfig` and `EntityManager`?

**Evidence**

`ProjectApp.__init__` reads `config.max_entities` twice:

```python
quality_cfg.max_entities = int(getattr(self.config, "max_entities", 50_000))   # line ~78
...
max_ent = getattr(self.config, "max_entities", 50_000)                          # line ~100
self.ecs = EntityManager(max_entities=max_ent)
```

`RenderPipeline` sizes SSBO 1/2 and the `FrustumCuller` scratch buffers from
`quality_cfg.max_entities`. `EntityManager` sizes its SoA arrays from `max_ent`.

**Impact**: if the two differ, the culler writes past the SSBO and the GPU reads garbage
transforms. This is a silent visual corruption, not an exception. ARCHITECTURE.md §7 lists this as
an invariant but nothing enforces it.

**Smallest diagnostic**: add a temporary assertion in `ProjectApp.__init__` after both are built:

```python
assert self.pipeline.config.max_entities == self.ecs.max_entities
```

Run the full suite. If it never fires, the invariant holds by construction today and the question is
only whether to make it permanent.

**Decision needed**: should this become a hard assertion (fail fast) or a validator warning?

---

## Q2. Which render-feature contract is authoritative: `RenderFeatureModule` or `RenderFeature`?

**Evidence**

Two incompatible contracts exist:

| Contract | File | Method |
|---|---|---|
| `RenderFeatureModule` | [module_contracts.py:32](../../engine/app/module_contracts.py:32) | `on_register(services) -> list[passes]` |
| `RenderFeature` | [render_feature.py:19](../../engine/gfx/render_feature.py:19) | `create_passes(services) -> Iterable[RenderPass]` |

`ModuleHost._attach_one` ([module_registry.py:238](../../engine/app/module_registry.py:238)) routes
`ModuleRole.RENDER_FEATURE` modules to `app.register_render_feature(instance)`, which calls
`pipeline.register_feature(feature)`, which calls `feature.create_passes(services)`.

A module written against the documented `RenderFeatureModule` base class has **no**
`create_passes`, so registration raises `AttributeError`.

**Impact**: the documented module-authoring path for render features is broken. `MODULES.md` and
`ENGINE_FEATURES.md` §6.3 describe the `RenderFeatureModule` / `on_register` shape.

**Smallest diagnostic**: write a 10-line `RenderFeatureModule` subclass, register it through
`ModuleRegistry`, and observe the exception. (Not done in this phase — it would be a code change.)

**Decision needed**: unify on one contract. The `RenderFeature` shape is the one the pipeline
actually consumes; `RenderFeatureModule` appears to be the intended public API. Either
`ModuleHost` should adapt `on_register` → `create_passes`, or `RenderFeatureModule` should be
rewritten to subclass `RenderFeature`.

---

## Q3. Should registered render features actually execute?

**Evidence**

`RenderGraph.execute()` is never called. `_BuiltinPass.execute()` is a no-op. `render_frame` calls
the hardcoded built-in passes directly.

**Impact**: a feature that registers successfully (after Q2 is resolved) still renders nothing. The
render graph is currently a **validation-only** artifact.

**Smallest diagnostic**: register a feature whose pass writes a constant color to the HDR target,
run one frame headless, read back the pixel. If the pixel is unchanged, the feature did not execute.

**Decision needed**: two viable designs —

1. **Execute the graph.** Replace the hardcoded call sequence in `render_frame` with
   `render_graph.execute(ctx)`, and make each built-in pass a real `RenderPass` that wraps the
   existing call. This is the "correct" design but is a large, risky refactor of a 90 KB method.
2. **Execute feature passes inline.** Keep the hardcoded built-ins, and after each built-in stage,
   execute any feature passes anchored to it via `after`/`before`. Smaller change, preserves current
   behaviour exactly, but keeps two scheduling mechanisms.

Option 2 is the smaller defensible change and does not alter visual behaviour for projects with no
features registered.

---

## Q4. Is the `active_count <= 0` early-out in `render_frame` intended?

**Evidence**

```python
active_count = ecs.active_count
if active_count <= 0:
    return
```

([pipeline.py:1063](../../engine/gfx/pipeline.py:1063))

**Impact**: an empty scene renders nothing — not even a cleared frame or sky. The previous frame's
contents remain on screen. A project that destroys all entities (e.g. a loading transition) would
show a frozen image rather than a sky or a clear color.

**Smallest diagnostic**: run a headless app with zero entities and read back the final texture. If it
is the previous frame (or uninitialized), the early-out is observable.

**Decision needed**: is "empty scene = no draw" a deliberate optimization, or should the pipeline
still clear and render the sky? If deliberate, it should be documented in ARCHITECTURE.md §4.

---

## Q5. Is `film_grain_enabled` supposed to be on for LOW?

**Evidence**

`tests/test_camera_optics.py::test_quality_presets_camera_optics_scaling` asserts
`low.film_grain_enabled is True`. All five presets set `film_grain_enabled=False`
(quality_presets.py:357, 406, 468, 530, 593).

**Impact**: one failing test. No runtime impact.

**Decision needed**: which is authoritative — the test (film grain is a cheap effect that should be
on even at LOW) or the presets (film grain is a stylistic effect that is off by default)? The
`RenderConfig` default is `False`, which suggests the presets are correct and the test is stale.

---

## Q6. Should `u_SunAngularRadius` be removed from `deferred_resolve.frag`?

**Evidence**

`shaders/deferred_resolve.frag:123` declares `uniform float u_SunAngularRadius = 0.0;`. No Python
code sets it (zero hits in `pipeline.py` and `environment.py`). A test
(`test_shadow_csm.py::TestRemovedShadowMachinery::test_shader_has_no_pcss_uniforms`) asserts it was
removed.

**Impact**: one failing test; one dead uniform (the driver will optimize it out, so no runtime cost).

**Decision needed**: remove the declaration (matching the test's intent) or update the test. Since
the uniform is genuinely unused, removal is the smaller change — but it touches a shader, which is
out of scope for a read-only audit.

---

## Q7. Is `CVarRegistry` intended to replace `EngineTweaks`, or to complement it?

**Evidence**

`CVarRegistry` is complete (typed cvars, save/load, change events) and tested, but **no runtime code
reads a cvar**. `EngineTweaks` is the live settings surface, applied every frame.

**Impact**: two settings systems, one dead. A developer adding a new tunable has to choose, and the
documentation does not say which.

**Decision needed**: either wire `CVarRegistry` into `EngineTweaks` (cvars as the storage layer), or
mark `CVarRegistry` as reserved-for-future and document that `EngineTweaks` is the live surface.

---

## Q8. Is `WaypointGraph` intended to be used by `MapManager` or by AI modules?

**Evidence**

`engine/world/waypoint_graph.py` defines `WaypointGraph`, `WaypointNode`, `WaypointEdge`,
`PointOfInterest`. It has **zero** external references. `ProjectApp.waypoint_graph` returns
`getattr(world_builder, "waypoint_graph", None)`, and no builder defines that attribute, so the
property always returns `None`.

**Impact**: dead subsystem; the `waypoint_graph` property is misleading.

**Decision needed**: is this a planned feature (keep, document as unwired) or abandoned (remove)?

---

## Q9. Is `HudOverlay` still supported?

**Evidence**

`HudOverlay` is exported from `engine/gfx/__init__.py` and tested, but no runtime code constructs it.
`UIManager` + `DebugMenu` cover the same need.

**Impact**: dead class kept alive by an export and a test.

**Decision needed**: deprecate (keep export, add a warning) or remove.

---

## Q10. Is `VolumetricSettings` a second source of truth for fog?

**Evidence**

`engine/gfx/volumetrics.py` mirrors `RenderConfig` fog fields (`fog_density`,
`fog_height_falloff`, `fog_anisotropy`, `fog_ambient`, `fog_distance`) in a dataclass with
`from_render_config`. Only `tests/test_volumetrics_common.py` uses it. The actual fog passes read
`RenderConfig` directly.

**Impact**: a developer changing `VolumetricSettings` would see no effect.

**Decision needed**: either make the fog passes consume `VolumetricSettings`, or delete it and keep
`RenderConfig` as the single source.

---

## Q11. Is the `dt` double-clamp intentional?

**Evidence**

`ProjectApp.run` clamps `dt = min(raw_ms * 0.001, 0.1)`. `TimeSystem.tick` clamps again to
`max_delta_time` (0.20). The app's clamp is tighter, so the time system's clamp is never reached in
the real loop.

**Impact**: none functionally; the tighter clamp means a 100 ms stall produces at most 6 fixed ticks
at 60 Hz, not 12.

**Decision needed**: is 0.1 s the intended maximum frame delta? If so, `TimeSystem.max_delta_time`
should be 0.1 to avoid two different numbers meaning the same thing.

---

## Q12. Should `PhysicsManager` have an explicit `destroy()`?

**Evidence**

`PhysicsManager` has no `destroy`/`close`. The Rapier world is released only by GC. The test suite
builds many apps and relies on `gc.collect()` in `conftest.py`.

**Impact**: potential native memory growth in long test runs or in an editor that creates/destroys
apps repeatedly.

**Decision needed**: add `PhysicsManager.destroy()` and call it from `ProjectApp.shutdown`, or
document that GC is sufficient.

---

## Q13. Are the `RenderPipeline.destroy` omissions deliberate?

**Evidence**

`pipeline.destroy()` does not release `water_pass`, `spot_shadow_map`, `spot_depth_prog`,
`spot_depth_vao`, `cascade_shadow_mdis`, or `render_features.destroy_all()`.

**Impact**: GPU resource leak on every shutdown. In the test suite this is masked by
`gc.collect()`, but it is a real leak in a long-running editor or a process that creates multiple
apps.

**Decision needed**: confirm these are omissions (not intentional "let the context destroy them"
choices) before patching. The presence of `WaterPass.destroy` and `SpotLightShadowMap.destroy`
strongly suggests they were meant to be called.

---

## Q14. Is `engine/gfx/atmosphere.py` still needed?

**Evidence**

It is a deprecated shim re-exporting from `engine.modules.atmosphere`. Three test files import it.
No runtime code does.

**Impact**: a second import path that must be kept in sync.

**Decision needed**: migrate the three tests to `engine.modules.atmosphere` and delete the shim, or
keep it as a documented compatibility layer.

---

## Q15. Does the atmosphere module's `GlobalLightSystem` duplicate `AtmosphereSystem`?

**Evidence**

`engine/modules/atmosphere/system.py` defines `AtmosphereSystem` (which owns a `GlobalLightSystem`)
and `global_light.py` defines `GlobalLightSystem` with its own `refresh()` and `lighting()`. Both
expose `compute_lighting()`-style outputs.

**Impact**: two lighting models in one module; unclear which is authoritative for sun color/lux.

**Smallest diagnostic**: read `system.py:54` and `global_light.py:194` side by side and compare the
returned tuples for the same config.

**Decision needed**: confirm `AtmosphereSystem` delegates to `GlobalLightSystem` (expected) rather
than reimplementing it.

---

## Q16. Is the `engine.core → engine.input` import a deliberate exception?

**Evidence**

`engine/core/__init__.py` re-exports `InputManager`; `engine/core/loop.py` imports
`from engine.input import InputManager` directly.

**Impact**: `import engine.core` pulls in pygame. The "core is headless-testable with no GL context"
claim holds for GL but not for pygame.

**Decision needed**: accept and document, or move `EngineLoop`'s input dependency behind a protocol.

---

## Q17. What is the intended behaviour when the atmosphere module fails to import?

**Evidence**

`create_default_environment` wraps the import in `except Exception: return FlatEnvironment(ctx)`.

**Impact**: a broken atmosphere module silently degrades to a flat sky with no log message. A
developer would see a wrong-looking sky and no error.

**Decision needed**: should this log a warning (it currently does not), or fail loudly?

---

## Q18. Are the `build/` PNG probes and `_ao_probe.py` still needed?

**Evidence**

`build/` contains ~20 PNG probe outputs and `validation_assets/`. `_ao_probe.py` is a temporary
probe script at the repo root. `scripts/run_validation.py` (81 KB) and `scripts/run_pom_plane.py`
(52 KB) are large developer harnesses.

**Impact**: repository noise; unclear which are authoritative validation artifacts.

**Decision needed**: decide whether these belong in `benchmark/results/` (per the required artifact
layout) or should be removed.

---

## Summary of Decisions Needed Before Phase 2

| Priority | Question | Blocks |
|---|---|---|
| **P0** | Q2 + Q3 — render feature contract and execution | Any module-based render extension |
| **P1** | Q13 — confirm `destroy` omissions | Safe leak fix |
| **P1** | Q1 — `max_entities` invariant | Silent corruption risk |
| **P2** | Q5, Q6 — stale tests | Green baseline |
| **P2** | Q4 — empty-scene early-out | Documented behaviour |
| **P3** | Q7, Q8, Q9, Q10, Q14 — dead/unwired subsystems | Cleanup scope |
| **P3** | Q11, Q12, Q15, Q16, Q17, Q18 — minor consistency | Documentation accuracy |
