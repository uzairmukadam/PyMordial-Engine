# Writing Modules for PyMordial Engine

> **Status: `DONE`** — Horizon 2 delivered the module host, render extension graph, and the
> first engine module (`pymordial.atmosphere`). This guide describes the supported contracts.

---

## 1. Two kinds of module

| Kind | Location | Purpose |
|---|---|---|
| **Engine module** | `engine/modules/<name>/` | Reusable engine capability shipped with PyMordial (sky, water, GI, …) |
| **Project module** | `projects/<game>/modules/<name>/` | Game-owned behavior (player, weapons, HUD, …) |

Both use the same metadata contract and the same host. A project module must never modify
files under `engine/`.

---

## 2. Declaring metadata

Expose a `MODULE_MANIFEST` class attribute (or a `get_module_manifest()` callable):

```python
from engine.app.module import ProjectModule
from engine.app.module_manifest import ModuleRole, build_manifest


class CombatModule(ProjectModule):
    name = "CombatModule"

    MODULE_MANIFEST = build_manifest(
        "my_game.combat",
        version="1.0.0",
        role=ModuleRole.GAMEPLAY,
        entry="projects.my_game.modules.combat:CombatModule",
        dependencies=["my_game.core"],
        capabilities=("gameplay.combat",),
        description="Damage, hit reactions, and weapon state.",
    )
```

### Manifest fields

| Field | Required | Notes |
|---|:---:|---|
| `id` | yes | Lowercase dotted segments, e.g. `pymordial.atmosphere` |
| `version` | yes | `MAJOR.MINOR.PATCH` |
| `api_version` | yes | Defaults to `MODULE_API_VERSION`; the major must match the engine |
| `role` | yes | `GAMEPLAY`, `RENDER_FEATURE`, or `TOOL` |
| `entry` | yes | `module.path:ClassName` |
| `dependencies` | no | Strings or `{"id": ..., "version": ..., "optional": ...}` |
| `capabilities` | no | Free-form capability tags |
| `config_schema` | no | Declarative description of tunables |

Validation is **side-effect free**: `validate_manifest()` never imports or executes module
code, so a broken manifest fails before any GPU or physics context is allocated.

Modules without metadata still work — they are treated as gameplay modules with no
dependencies and ordered after declared modules.

---

## 3. Role contracts

### Gameplay (`GameplayModule` / `ProjectModule`)

```python
def on_attach(self, app) -> None: ...
def on_fixed_update(self, app, dt: float) -> None: ...  # 60 Hz simulation
def on_update(self, app, dt: float) -> None: ...  # variable-rate presentation
def on_ui(self, app) -> None: ...
def on_event(self, app, event) -> bool: ...  # return True to consume
def on_detach(self, app) -> None: ...
```

Never capture the reserved debug hotkeys `F1`, `F2`, or `F3`.

### Render feature (`RenderFeatureModule`)

```python
def on_register(self, services: RenderServices) -> list[RenderPass]: ...
def on_unregister(self) -> None: ...
```

`RenderServices` is the only renderer surface a feature may touch. It exposes the GL
context, render config, frame context, G-buffer, CSM, HDR target, capability queries, and
texture-unit reservations. Units used by the core deferred resolve are reserved and cannot
be claimed.

### Tool (`ToolModule`)

Developer-only panels and inspectors. Stripped from release builds by the packager.

---

## 4. Loading modules

```python
from engine.app.module_registry import ModuleRegistry

registry = ModuleRegistry()
registry.register_instance(SpinnerModule())
app.load_modules(registry)
```

`ModuleRegistry.resolve()` produces a deterministic order (Kahn topological sort, ties broken
by id). `ModuleHost` attaches in that order and detaches in reverse.

### Failure behavior

If any module raises during attach, the host:

1. detaches every already-attached module in reverse order,
2. removes every event subscription it made on the module's behalf,
3. raises `ModuleAttachError` naming the failing module.

Startup never continues with a partially initialized module set.

### Diagnostics

| Situation | Result |
|---|---|
| Missing required dependency | `ModuleResolutionError` listing the missing id |
| Dependency cycle | `ModuleResolutionError` with the cycle path |
| Missing optional dependency | Warning only |
| Invalid manifest | `ModuleResolutionError` with per-field errors |
| Attach failure | `ModuleAttachError` after full rollback |

---

## 5. Render features

```python
from engine.gfx.render_feature import RenderFeature
from engine.gfx.render_graph import RenderPass


class OutlinePass(RenderPass):
    def __init__(self):
        super().__init__(
            name="OutlinePass",
            inputs=("g_buffer",),
            outputs=("outline_texture",),
            after=("DeferredResolve",),
        )

    def execute(self, context): ...


class OutlineFeature(RenderFeature):
    name = "OutlineFeature"
    after = ("DeferredResolve",)
    requires = ("compute",)

    def create_passes(self, services):
        return [OutlinePass()]
```

Registering a feature validates the whole graph. Unknown ordering references, duplicate
outputs, missing inputs, and cycles are rejected **before** the first frame runs, and the
feature is not retained on failure.

Built-in stage names you can anchor to: `ClusteredLights`, `CascadedShadows`, `SpotShadows`,
`GBuffer`, `SSDM`, `Decals`, `HiZ`, `AmbientOcclusion`, `GlobalIllumination`, `IBL`, `SSR`,
`SkyAtmosphere`, `DeferredResolve`, `VolumetricFog`, `Water`, `Particles`, `CameraOptics`,
`TAA`, `PostProcess`.

---

## 6. Configuring engine modules

Engine modules are enabled by default so the turnkey path keeps working. Configure or
disable them from project configuration — never by editing `pipeline.py`:

```python
ProjectConfig(
    atmosphere_enabled=True,
    atmosphere_settings={
        "preset": "EARTH_SUNSET",  # any key in ATMOSPHERE_PRESETS
        "time_of_day": 17.5,  # direct AtmosphereConfig overrides
        "day_speed": 0.0,
    },
)
```

Setting `atmosphere_enabled=False` swaps in `FlatEnvironment`, a documented fallback that
still binds valid LUTs so the deferred resolve keeps working.

---

## 7. Visual regression gate

`tests/test_horizon2_golden.py` captures a fixed headless scene in a **subprocess** and
compares it against `tests/baseline/`. Any change to the default sky, lighting, or resolve
path must keep the mean absolute error within tolerance.

```powershell
python -m tests.golden_capture --check    # verify
python -m tests.golden_capture --write    # re-record (only for intentional changes)
```

The capture runs in a subprocess because creating and destroying several GL contexts in one
process measurably perturbs later frames on some drivers.

---

## 8. Reference projects

Two projects ship as end-to-end references.

### `projects/playground/` — the canonical workflow reference

Playground is the complete project workflow in one place: declarative `config.py`, a
project-owned `BaseWorldBuilder` (`world/arena.py`), four manifest-declared modules loaded
through `ModuleRegistry`, a `GameStateManager` Playing/Paused flow, and a headless CI path.

```powershell
python -m engine.tools.project validate --project playground
python -m engine.tools.project run --project playground --headless --frames 5
python projects/playground/main.py --headless --frames 5 --no-atmosphere
python -m engine.tools.packager --project playground --target pyinstaller --dry-run
```

Its module layout is the one the packager expects — **one package directory per module**:

```text
projects/playground/
├── config.py                     # ProjectConfig (display, physics, engine modules)
├── main.py                       # ProjectApp + world builder + ModuleRegistry
├── world/arena.py                # PlaygroundArenaBuilder(BaseWorldBuilder)
└── modules/
    ├── player/__init__.py        # playground.player  (GAMEPLAY)
    ├── hud/__init__.py           # playground.hud     (GAMEPLAY, depends on player)
    ├── menu/__init__.py          # playground.menu    (GAMEPLAY, depends on player)
    └── tweaks/__init__.py        # playground.tweaks  (TOOL, depends on player)
```

> **Packaging gotcha**: `ProjectPackager.detect_modules()` tree-shakes by *directory* under
> `modules/`. A flat `modules/foo.py` is silently dropped from a packaged build. Always use
> `modules/foo/__init__.py`.

### `projects/reference_game/` — the minimal engine-module reference

A smaller example focused on engine-module configuration: it declares atmosphere settings in
`config.py`, registers one gameplay module with a manifest, and can be launched with the
atmosphere disabled:

```powershell
python -m engine.tools.project validate --project reference_game
python -m engine.tools.project run --project reference_game --headless --frames 5
python projects/reference_game/main.py --headless --frames 5 --no-atmosphere
```
