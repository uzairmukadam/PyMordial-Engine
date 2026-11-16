# PyMordial Engine: Modular Architecture Proposal

> **Status: PARTIALLY IMPLEMENTED — Horizon 2 delivered.** The `ProjectModule` contract, `ProjectApp` host, and `BaseWorldBuilder` exist. Horizon 2 added module manifests, deterministic dependency resolution with rollback, render-graph extension points, and the first engine module extraction (`pymordial.atmosphere`). Still outstanding: project manifests, asset-aware build selection, and the remaining feature extractions. See [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) for the schedule.

## 1. Purpose

PyMordial should provide a complete, dependable game-engine experience without requiring developers to modify engine source or assemble a rendering stack themselves. At the same time, its major features should be replaceable or extensible through supported module interfaces.

The design has three goals:

1. Keep the engine core small, cohesive, and relatively stable.
2. Ship official engine features that work together out of the box.
3. Let projects use, replace, extend, or privately develop modules without editing the engine.

Modularity is an integration boundary, not a requirement that every feature be a separate download. Official modules may be distributed with the engine and enabled by default in project templates.

This document describes the target architecture. Some elements, especially module dependency resolution, renderer extension points, project manifests, and asset-aware build selection, are proposals and are not yet fully implemented.

## 2. Architectural layers

```text
┌──────────────────────────────────────────────────────────┐
│ Game Project                                             │
│ Scenes/worlds, assets, project settings, gameplay modules│
├──────────────────────────────────────────────────────────┤
│ Project Modules                                         │
│ Game-specific systems, UI, rules, AI, controls, content  │
├──────────────────────────────────────────────────────────┤
│ Engine Modules                                           │
│ Sky, water, GI, post-processing, animation, tools, etc. │
├──────────────────────────────────────────────────────────┤
│ Engine Core                                              │
│ Lifecycle, ECS, renderer contracts, physics, VFS, input │
└──────────────────────────────────────────────────────────┘
```

Dependencies should point downwards through declared interfaces. The core must not import a particular optional feature implementation in order to function. A project may depend on engine modules, and one engine module may depend on another, but dependency cycles should be rejected.

## 3. Engine core

The core contains foundational systems that are broadly required, tightly integrated with runtime ownership, or necessary to provide stable contracts to modules.

### 3.1 Core responsibilities

- **Application lifecycle and frame loop:** startup, event polling, fixed simulation ticks, variable updates, rendering, presentation, and shutdown.
- **ECS and entity identity:** entity creation/destruction, transforms, core component storage, and stable references or handles.
- **Time and events:** fixed timestep, frame time, time domains, event dispatch, and orderly event ownership.
- **Input abstraction:** platform input polling and action/state queries.
- **Physics service contract:** fixed-step scheduling, body/shape ownership, and synchronization boundaries. A particular physics backend should be replaceable where practical.
- **Resource and VFS foundations:** asset paths, resource lifetime, mounted sources, and common load errors.
- **Window and platform abstraction:** window creation, resize/focus handling, graphics-context creation, and frame presentation.
- **Audio service contract:** device lifecycle, common playback controls, and listener updates. Specialized audio features can be modules.
- **UI foundations:** input routing, canvas sizing, base widget/screen lifecycle, and integration into frame presentation.
- **Renderer foundations:** GPU context/resource ownership, frame data, render graph or equivalent scheduling interface, core scene submission, and renderer-facing extension points.
- **Module host:** discovery, dependency resolution, configuration, lifecycle dispatch, diagnostics, and shutdown ordering.
- **Project and scene loading contracts:** initial project settings and world/scene activation boundaries.

The core may include a basic renderer, basic materials, direct lighting, point/spot lights, and shadows so a new project can render a useful scene without optional feature setup. The core should avoid owning every advanced rendering technique merely because it currently participates in one monolithic pipeline.

### 3.2 Core design constraints

- Keep public interfaces smaller and more stable than their implementations.
- Make ownership explicit: the creator or documented owner releases each resource.
- Surface initialization, dependency, and runtime failures with actionable errors.
- Keep optional feature imports out of core startup paths.
- Avoid public APIs that expose internal buffer layouts or private renderer details unless those are intentionally versioned.
- Maintain a turnkey default configuration; developers should not need to register every standard feature by hand.

## 4. Engine-provided modules

Official modules are reusable engine capabilities distributed and supported with PyMordial. They should have documented dependencies, configuration, lifecycle, quality controls, and compatibility guarantees. Templates and presets can enable a suitable default set.

| Module | Responsibility | Notes |
|---|---|---|
| **Sky and Atmosphere** | Atmospheric scattering, celestial bodies, stars/space backgrounds, and cloud rendering | A useful first extraction candidate; should provide sky lighting data through a renderer contract. |
| **Water** | Water surfaces, waves, optical settings, reflection/refraction, and foam | Surface creation/configuration should be explicit; avoid hidden global water planes. |
| **Global Illumination** | Selectable GI implementations such as SSGI and Radiance Cascades | May be one package with selectable strategies or several modules behind a common GI interface. |
| **Post-Processing** | Tone mapping, bloom, exposure, color grading, motion blur, depth of field, and related effects | Effects should declare ordering and resource requirements. |
| **Anti-Aliasing** | TAA, FXAA, and SMAA strategies | Coordinate with motion vectors, temporal history, and render resolution. |
| **Ambient Occlusion** | GTAO/SSAO and reconstruction/filtering | Can be grouped with GI if the implementation shares substantial infrastructure. |
| **Reflections** | SSR and future reflection strategies | Must define fallbacks for off-screen or invalid screen-space samples. |
| **Volumetrics** | Fog, participating media, light shafts, and future volumetric effects | Quality settings should expose resolution and sampling budgets. |
| **Particles and VFX** | GPU/CPU particle simulation, emitters, and rendering | Integrate with project-defined emitters and resource budgets. |
| **Decals and surface detail** | Deferred decals and optional displacement features | Depends on renderer material and G-buffer contracts. |
| **Editor and developer tools** | Scene inspection, profilers, graphics tweaks, and authoring tools | Development-only tools must be removed or disabled by explicit build configuration, without removing essential release diagnostics. |
| **Animation** | Skeletal animation, blending, animation events, and optional root motion | A future module unless a target game requires it sooner. |
| **Navigation and AI support** | Navigation data, path queries, and reusable AI building blocks | Keep game-specific behavior in project modules. |
| **Networking** | Transport-independent replication/session systems | Likely future work; network authority and simulation semantics need deliberate design. |

This list is a direction, not a requirement to split every row into an independent package immediately. Features with deep shared implementation may remain grouped until a stable boundary is clear.

## 5. Project modules

Project modules contain game-owned behavior. Examples include player controllers, weapons, race managers, quest logic, enemy behavior, and game-specific HUDs.

Project modules:

- Depend on core services and may declare dependencies on engine modules.
- Must not modify files under `engine/`.
- Should own their subscriptions, callbacks, entities, and resources and clean them up during detach or scene unload.
- Should separate deterministic simulation work from presentation work.
- Should not intercept reserved engine debug controls.
- May be private to one game or packaged for reuse by a studio.

The existing `ProjectModule` callback model is a starting point for gameplay modules. It is not by itself sufficient for render features, asset importers, physics backends, or editor extensions. Those need role-specific interfaces hosted by the same module system.

## 6. Module package structure

An engine module and a project module should use a common package/dependency format, with role-specific entry points. A possible package layout is:

```text
my_feature/
    module.toml
    __init__.py
    runtime.py
    config.py
    assets/
    shaders/
    tests/
```

The module manifest should eventually describe:

```toml
[module]
id = "pymordial.sky"
version = "1.0.0"
api = "1"
kind = "engine_feature"
entry = "my_feature.runtime:SkyModule"

[dependencies]
"pymordial.renderer" = "^1.0"

[capabilities]
provides = ["render.sky", "lighting.environment"]
```

This is illustrative syntax; the exact file format is an implementation decision.

The public module contract should support:

- Unique identifier and semantic version.
- Supported engine/module API versions.
- Module role/capabilities.
- Required and optional dependencies, with version ranges.
- Configuration schema and defaults.
- Initialization and shutdown.
- Enable/disable and scene activation/deactivation where applicable.
- Resource ownership and cleanup.
- Diagnostics and failure reporting.

Specific module roles can extend the common lifecycle:

- **Gameplay/system module:** event handling, fixed update, variable update, UI, scene lifecycle.
- **Render feature:** pass/resource declarations, ordering constraints, quality settings, and renderer capability checks.
- **Physics backend or feature:** world creation, fixed-step participation, queries, and synchronization.
- **Asset importer/cooker:** source formats, validation, conversion, and build outputs.
- **Editor extension:** panels, inspectors, commands, and authoring-time asset integration.

Modules should register through a host or service registry, not reach into private fields of `ProjectApp` or `RenderPipeline`.

## 7. Project structure and declaration

A game remains a directory under `projects/`, containing its own code, assets, and worlds. A future project manifest should explicitly declare the entry point, enabled engine modules, project modules, settings, and dependency versions.

Example target layout:

```text
projects/my_game/
    project.toml
    main.py
    config.py
    assets/
        source/
        cooked/
    worlds/
        main_world.py
    modules/
        player/
        combat/
        game_ui/
    tests/
```

Project modules and engine modules should have distinct namespaces even if they share a package format:

- Engine-provided modules are installed or bundled as engine dependencies.
- Game-specific modules live under the project.
- Private reusable modules can be referenced from a local path or private package registry.

Project configuration should make modules explicit. Avoid selecting dependencies by scanning directory names or relying only on static Python import detection; dynamic imports and data-driven activation make such detection incomplete.

## 8. World and scene building

Projects should be able to create worlds in at least two ways:

1. **Code-driven world builders:** Python logic generates or loads a world, useful for procedural, specialized, or rapidly prototyped games.
2. **Data-driven scenes:** serialized entities, transforms, components, references, and module configuration, useful for editor-authored levels and repeatable content workflows.

Both routes should use the same runtime contracts for entity creation, resource references, physics setup, and cleanup. A world builder should not need to know internal render pass ordering.

The runtime should distinguish world/scene operations such as:

- Load data and resolve references.
- Create or activate entities and module instances.
- Prepare rendering, physics, audio, and navigation resources.
- Deactivate or unload owned resources safely.

Cell/region streaming can build on these lifecycle boundaries later.

## 9. Runtime startup and shutdown

### Startup

1. Read and validate project configuration and manifest.
2. Resolve module versions and dependency graph.
3. Create core platform and runtime services.
4. Initialize enabled engine modules in dependency order.
5. Attach project modules.
6. Load the initial world and its resource dependencies.
7. Enter the initial game state.
8. Start event polling and frame execution.

Missing dependencies, unsupported API versions, or module initialization failures should stop startup with a clear diagnostic unless the project explicitly marks the feature optional and provides a fallback.

### Shutdown

Shutdown follows reverse ownership order:

1. Stop accepting gameplay input and end the active state.
2. Detach project modules.
3. Deactivate/unload world and scene-owned resources.
4. Shut down engine modules in reverse dependency order.
5. Release core audio, physics, renderer, VFS, and platform resources.

Shutdown paths must be tested for normal exit, startup failure, and interrupted loading.

## 10. Frame execution

The engine owns frame scheduling. Modules participate in defined phases and do not independently run their own competing main loops.

```text
Platform events and input polling
    ↓
Event dispatch: engine systems → UI → active game state → project modules
    ↓
Zero or more fixed simulation ticks
    ├─ Cache prior simulation transforms
    ├─ Run fixed-step project/gameplay systems
    ├─ Apply movement and physics commands
    ├─ Step the physics backend
    └─ Synchronize physics results to ECS
    ↓
Interpolate presentation transforms
    ↓
Variable-rate updates: camera, presentation, audio, UI
    ↓
Build render frame data and submit visible scene work
    ↓
Execute renderer core and registered render features
    ↓
Render UI and developer overlays
    ↓
Present frame
```

### Simulation rules

- Physics and gameplay that must be frame-rate independent run at the fixed timestep.
- Physics state has a defined owner; systems submit commands before the physics step and consume results afterward.
- Presentation interpolation must not mutate authoritative simulation state.
- Variable-rate updates are for presentation and non-authoritative work unless a system explicitly defines different semantics.
- Pausing and state transitions must specify which simulation, audio, UI, and rendering phases remain active.

### Rendering rules

The core renderer provides scene submission and stable rendering contracts. Engine render modules declare their inputs, outputs, requirements, and ordering constraints. The render scheduler validates the graph and executes a valid order.

A typical configured frame may include:

1. Shadow rendering.
2. G-buffer or forward scene rendering.
3. Depth/visibility and decals.
4. Ambient occlusion and GI.
5. IBL, direct-light resolve, and reflections.
6. Sky and atmosphere composition.
7. Water, volumetrics, particles, and other scene effects.
8. Temporal processing, post-processing, tone mapping, and anti-aliasing.
9. UI, developer overlays, and presentation.

This is not a hardcoded universal order. The selected renderer and modules determine the actual graph; invalid or missing dependencies should be reported.

## 11. Build and distribution pipeline

The release command should produce a reproducible build from declared project inputs:

1. Validate project metadata, module APIs, configuration, asset references, and target platform.
2. Resolve explicit module dependencies and versions; record the resolved graph.
3. Cook source assets into validated runtime formats.
4. Determine required cooked assets from scenes and declared runtime data. Dynamic asset references must be declared or conservatively included.
5. Stage core runtime, selected engine modules, project code, native dependencies, shaders, and cooked assets.
6. Apply explicit release settings: disable development overlays and tools, optimize code/assets where supported, and retain appropriate crash diagnostics.
7. Bundle the Python runtime and native dependencies into the selected Windows distribution target.
8. Smoke-test the staged executable, load representative scenes, and verify packaged asset access.
9. Emit the executable/distribution, build manifest, dependency/license notices, and optional separate debug symbols/logging configuration.

Module selection should be explicit and dependency-driven. Python static import scanning alone cannot reliably identify runtime-loaded modules.

### Asset security expectations

Asset cooking, packing, and compression improve load behavior and distribution convenience. Encryption can deter casual extraction, but assets must be decrypted by the local game and therefore cannot be guaranteed secret from a determined recipient. Do not promise complete IP protection. Consider signing artifacts and validating integrity separately from confidentiality.

### Release diagnostics

Release builds should omit developer-facing overlays when requested, but retain useful error reporting and crash diagnostics. Production builds should fail clearly on required missing assets/modules rather than silently substitute success-shaped defaults.

## 12. Quality presets and hardware targeting

Hardware targeting should be handled primarily through tested quality profiles and per-feature scalability controls, not by requiring developers to hand-assemble a renderer.

Presets may select:

- Render resolution and upscaling strategy.
- Shadow resolution, distance, and filter quality.
- GI/reflection mode and sampling budgets.
- Volumetric and post-process quality.
- Texture/geometry streaming budgets.
- Particle and local-light budgets.

Projects should be able to override settings. Modules should declare meaningful quality controls and capability requirements so unsupported features can be disabled or downgraded deliberately.

## 13. Player-facing settings and developer tuning

The engine should distinguish settings intended for end users from controls intended for game developers:

- **Player-facing settings** are a stable, curated contract presented by the game: display mode, resolution, refresh rate where supported, VSync, frame-rate cap, graphics quality presets, and selected accessibility/performance toggles.
- **Developer tuning** includes implementation-specific values such as lens-flare shape, GI sampling parameters, cloud noise controls, shadow filter internals, and experimental debug visualizers. These belong in editor/debug tooling or project configuration, not the default player settings screen.

An engine module may expose both categories, but it should explicitly classify each setting, provide safe bounds/defaults, and define whether changing it takes effect immediately or requires a restart. The game developer decides which supported player options to expose, rename, or hide. A preset should provide a dependable default without preventing advanced users from overriding it.

### 13.1 Window and display settings

The window service should provide a platform-aware API for:

- Windowed, borderless fullscreen, and exclusive fullscreen modes where supported.
- Resolution and display/monitor selection.
- Refresh-rate selection where the platform and display mode API permit it.
- VSync modes and a separate software frame-rate cap.
- Resizability, aspect-ratio behavior, and window placement where appropriate.
- Runtime mode switching with resize events and renderer/UI resource updates.

The settings UI should query valid display modes rather than assume a fixed list. Resolution and refresh rate are a coupled display mode on some platforms; the engine should select a supported mode or report why the requested combination cannot be applied. VSync and frame-rate limiting are distinct controls and should not be presented as the same setting.

When applying a potentially disruptive mode change, preserve the previous known-good settings and provide a confirmation/revert timeout. If the user does not confirm that the display remains usable, restore the previous mode. Persist only settings that were successfully applied.

### 13.2 Abstract graphics options

The standard player settings menu should expose options at the level of outcomes rather than implementation details. For example, it may expose “Lens Flare: On/Off” without exposing flare streak count, texture shape, or shader thresholds. Those deeper controls remain available to game developers and technical artists through project configuration or developer tools.

Possible player-facing controls include:

- Overall quality preset and individual quality tiers when the developer allows them.
- Resolution scale or upscaling mode where implemented.
- Shadow, effects, and post-processing quality.
- Motion blur, depth of field, lens flare, film grain, and other taste-sensitive effects as simple toggles or limited choices.
- VSync and frame-rate cap.

Each setting must have an explicit owner and policy: engine default, project default, player override, and whether presets reset or preserve player preferences. The engine should provide a settings model and application API; each game owns the final presentation, labels, and available choices.

## 14. Persistent user preferences and game saves

The runtime should expose separate persistence systems for separate kinds of data. Do not mix engine/player preferences with game progress:

1. **Engine/player preferences:** display, graphics, audio, input, and accessibility settings. Store these in a per-user writable configuration directory, not beside read-only installed game assets.
2. **Game save data:** progress, inventory, checkpoints, world state, and other project-defined data. Store in a per-user save directory and let the game define slots, UI, and semantics.
3. **Project defaults:** version-controlled project configuration shipped with the game. These are defaults, not the location where runtime player changes are written.
4. **Developer settings:** editor/debug-only tuning, kept out of player preference files unless deliberately promoted to a player setting.

The engine should provide a storage-location API and robust serialization primitives, but the game should own the schema and meaning of its save-game state. Engine modules may contribute namespaced, versioned data to a save through explicit save participants; they must not silently serialize arbitrary runtime objects.

Persistence requirements:

- Versioned schemas and migration hooks for both preference and game-save formats.
- Validation of loaded values, including supported display modes and module settings.
- Atomic writes (write temporary data, then replace) so an interrupted save does not corrupt the previous file.
- Clear error reporting for inaccessible, malformed, incompatible, or failed writes.
- Optional multiple save slots, metadata, and backup/recovery policy for game saves.
- Optional integrity checks; do not treat local save files as trusted input.
- Headless/test APIs for deterministic export/import and migration tests.

Export/import should mean a documented, portable data format, not a memory dump. A project may choose to support player-facing save export/import or cloud synchronization, but the core should not assume a particular platform or cloud provider.

## 15. Static information pages and screen templates

The UI framework should make it straightforward to build common static and menu screens, such as:

- Title/splash screen.
- Main menu and settings pages.
- Credits.
- About/engine information.
- Loading, pause, and confirmation screens.

The engine can provide reusable screen templates and widgets, including responsive layout, keyboard/gamepad navigation, focus handling, localization-ready text, and accessible input behavior. The game project owns the actual page content, branding, credits, and navigation policy.

Engine name/version and third-party credits may be exposed as metadata that projects can use on an About or legal screen. They should not be forcibly inserted into every game’s title screen. Any engine attribution or license notices required by dependencies should follow their licenses and be documented for the project’s release materials.

These pages are UI screens, not engine states that own gameplay. Navigation should integrate with the game-state and UI-stack lifecycle so a settings screen can pause gameplay when appropriate without duplicating window or graphics settings logic.

## 16. Migration strategy

The modular architecture should be introduced incrementally:

1. **Define contracts:** module identity, lifecycle, dependency/version model, configuration, and error reporting.
2. **Extract a low-risk feature:** sky is a candidate, but first provide renderer pass registration and lighting/environment data interfaces.
3. **Prove defaults:** existing projects continue to receive a functioning sky and render configuration without manual setup.
4. **Add project manifest and module resolver:** explicit dependencies, validation, and build reports.
5. **Convert more features:** water, post-processing, GI, and other systems when each has a stable ownership and ordering boundary.
6. **Deliver common player-facing services:** display/graphics settings model, safe display-mode confirmation/revert, and per-user preference persistence.
7. **Add save-game contracts:** storage locations, versioned project-owned schemas, module save participants, and migration/recovery tests.
8. **Provide UI templates:** title, settings, credits, and About screen examples that projects can customize without inheriting engine branding.
9. **Expand authoring and packaging:** scene serialization/editor support, asset reachability, reproducible builds, and platform smoke tests.
10. **Stabilize public APIs:** publish compatibility guarantees before encouraging third-party or marketplace modules.

Do not move a subsystem to a new directory until its dependencies and ownership have been separated. File relocation without a stable interface would make the project appear modular while preserving the same coupling.

## 17. Definition of success

The architecture is working when a developer can:

- Create a project from a template and run it without editing engine source.
- Use official features by default and configure them through project settings.
- Choose which safe, abstract player settings the game exposes while retaining detailed developer-only tuning.
- Apply, confirm, or revert supported window/display changes and persist successful player preferences.
- Keep engine/player preferences separate from project-owned game saves.
- Build customizable title, credits, and About screens without being forced to adopt engine branding.
- Disable or replace a feature through documented interfaces.
- Add a project-local or private module with declared dependencies.
- Build the same project reproducibly and see exactly which modules/assets were included.
- Receive clear diagnostics for incompatible modules, missing assets, or unsupported capabilities.
- Upgrade the engine without copying project code into the engine or losing customizations.

## 18. Current implementation status

The repository already has project directories, `ProjectApp`, fixed/variable frame phases, `ProjectModule`, world-builder hooks, a render pipeline, asset cooking/VFS foundations, and a PyInstaller/Nuitka-oriented packager.

The target architecture still requires substantial work:

- A module manifest and dependency/version resolver.
- Separate interfaces for gameplay modules, render features, importers, and other extension roles.
- Render graph registration and resource contracts that do not require editing `RenderPipeline`.
- Explicit engine-module selection that the runtime and packager both honor.
- Accurate dependency and asset inclusion instead of copying all engine source and treating every project module directory as used.
- Clear development/release feature policy and packaged-build smoke testing.
- Scene serialization and an authoring workflow complementary to code-driven world builders.
- A polished player-facing display/graphics settings model and settings UI. The window layer already has foundations for window modes, resolution changes, VSync, and monitor metadata, but complete player-facing mode enumeration, refresh-rate selection, confirmation/revert behavior, and project-level presentation need to be verified and unified.
- A clear split between player preferences and developer graphics tuning. Configuration serialization exists in parts of the codebase, but it is not yet a unified, versioned per-user preferences service.
- A separate game-save API with project-owned schemas, migration, atomic writes, and module participation.
- Reusable title, credits, and About screen templates, with project-owned content and optional engine metadata.

These are architectural milestones, not claims about existing behavior.
