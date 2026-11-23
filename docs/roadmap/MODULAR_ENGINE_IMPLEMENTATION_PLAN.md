# PyMordial Engine: Modular Architecture Implementation Plan

> **Status: PARTIALLY IMPLEMENTED — Phases 2 and 3 delivered.** This is the staged execution plan for [MODULAR_ENGINE_ARCHITECTURE.md](./MODULAR_ENGINE_ARCHITECTURE.md). Horizon 2 completed the module host (metadata, registry, resolver, rollback), the renderer extension graph, and the atmosphere extraction. Phases 4–7 remain. Horizon 3+ in [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) is the authoritative schedule.

## 1. Goal

This plan turns the target design in [MODULAR_ENGINE_ARCHITECTURE.md](./MODULAR_ENGINE_ARCHITECTURE.md) into a staged implementation that can be delivered and validated without a risky all-at-once rewrite.

For small, sequential coding tasks with copyable prompts and checkpoints, use [MODULAR_ENGINE_MIGRATION_PROMPTS.md](./MODULAR_ENGINE_MIGRATION_PROMPTS.md).

The target developer experience is:

1. A developer creates or opens a game directory, normally under `projects/`.
2. The game contains its logic, worlds, assets, and project-specific modules.
3. Standard engine features work by default; optional features can be enabled, configured, or replaced through supported interfaces.
4. The developer runs, tests, and packages the game through engine tooling.
5. The release build contains the runtime, required modules, game code, and cooked assets, and is verified before distribution.

The engine remains the Python-based host and runtime. A game directory is a game project, not necessarily a Python package or independently published Python distribution. Project metadata should be introduced only where the runtime or build process needs declarative information that cannot be discovered reliably.

## 2. Principles for every phase

- **Deliver vertical slices:** each phase leaves the engine usable; avoid long-running branches that only move files.
- **Preserve the turnkey path:** ordinary game developers should not have to register every standard system manually.
- **Use explicit dependencies:** do not rely on directory scanning or static Python import analysis as the source of truth for module selection.
- **Separate authoring concerns:** core engine implementation, official reusable features, and game-owned behavior have different ownership and extension contracts.
- **Do not promise perfect secrecy:** asset packaging/encryption may deter casual extraction but cannot guarantee protection when a client must load the assets.
- **Keep developer diagnostics:** production builds can omit developer overlays, but should retain useful error reporting and crash diagnostics.
- **Make compatibility intentional:** define public API/version policy before asking third parties to build against it.
- **Measure before optimizing:** use repeatable benchmark scenes and target hardware rather than visual impressions alone.
- **No silent fallbacks:** report missing dependencies, invalid settings, corrupt assets, and failed saves with actionable diagnostics.

## 3. Phase overview

| Phase | Name | Main outcome |
|---|---|---|
| 0 | Baseline and public-version scope | Agreed target workflow, API boundaries, test and performance baselines |
| 1 | Project workflow and configuration | Reliable create/run/test commands and a minimal project contract |
| 2 | Module host and contracts | Versioned module loading, dependencies, lifecycle, configuration, and cleanup |
| 3 | Renderer extension architecture | Render features register through validated interfaces; sky proves the model |
| 4 | Player settings and UI templates | Safe display/graphics settings, preferences, and reusable information pages |
| 5 | Game saves and world lifecycle | Separate, versioned game-save API and consistent world/scene activation |
| 6 | Asset cook and release pipeline | Deterministic module/asset staging, packaged smoke tests, Windows distribution |
| 7 | Public compatibility and release qualification | Supported API policy, CI gates, docs, representative game, and release sign-off |

Phases 1 and 2 establish the workflow and contracts. Phase 3 depends on Phase 2. Settings/UI work in Phase 4 can proceed after Phase 1 and should reuse the module/configuration contracts where possible. Phase 5 builds on project and world lifecycle foundations. Phase 6 depends on explicit module and asset declarations. Phase 7 is the release gate for the accumulated work.

## 4. Phase 0 — Baseline and public-version scope

### Work

- Choose one representative target game and its minimum supported Windows/hardware profile.
- Define the expected project workflow: game directory under `projects/`, source entry point, configuration ownership, assets, worlds, and project modules.
- Decide which interfaces are public for the next version and which remain internal/experimental.
- Inventory current behavior and document known gaps without claiming proposed systems already exist.
- Establish baseline tests, a small representative graphics scene, build duration, startup time, memory use, and CPU/GPU frame-time measurements.
- Record current packaging/runtime constraints, including Python and native dependency requirements.

### Deliverables

- Versioned architecture decision record or approved sections in the architecture document.
- Baseline project that exercises input, fixed update, physics, rendering, UI, one engine feature, and shutdown.
- Baseline benchmark procedure and recorded results on target hardware.
- Public API inventory and compatibility policy draft.

### Exit criteria

- The team can describe the intended create → run → test → package workflow without contradictory ownership assumptions.
- Baseline project runs headless and interactively and has a reproducible test command.
- Existing limitations and release-blocking risks are documented.

## 5. Phase 1 — Project workflow and configuration

### Work

- Provide a project creation/scaffolding command or a maintained starter template.
- Standardize the project layout while keeping project directories usable as ordinary source directories.
- Decide whether a manifest is needed. Prefer a small declarative manifest for build/module metadata, while retaining Python configuration for game-defined behavior if desired.
- Define project identity, entry point, asset roots, module declarations, startup world, and supported target as explicit inputs to tooling.
- Support projects located under `projects/` and, if intended, a user-supplied path outside the engine repository.
- Add clear commands for validate, run, test, and build. Make project selection consistent across commands.
- Validate project configuration before initializing expensive runtime services.

### Deliverables

- Project template with a minimal runnable game and example game module.
- Project validation API/CLI with actionable diagnostics.
- Documented directory and configuration conventions.
- A test proving that a sample project can be discovered and launched without modifying engine code.

### Exit criteria

- A new developer can create a project, run it, and locate its assets using documented commands.
- Invalid/missing entry points and configuration errors fail early with clear diagnostics.
- The project is not required to be an independently installable Python package.

## 6. Phase 2 — Module host and contracts

### Work

- Define common module identity, version, role, configuration, dependency, and lifecycle metadata.
- Support role-specific contracts rather than forcing every extension into the gameplay `ProjectModule` callback shape.
- Resolve dependencies deterministically, detect cycles and incompatible API versions, and initialize/shut down in dependency order.
- Define ownership for event subscriptions, resources, entities, callbacks, and scene-scoped state.
- Make module initialization failures explicit and roll back already-initialized modules safely.
- Establish enable/disable semantics and decide which changes require restart.
- Add compatibility shims or migration guidance for existing project modules.

### Deliverables

- Module manifest schema and validator.
- Module registry/host with dependency ordering.
- Gameplay module contract and initial contracts for engine feature modules.
- Lifecycle and dependency tests, including failure and cleanup cases.
- A migration example from an existing project module.

### Exit criteria

- Required and optional dependencies resolve reproducibly.
- Missing, cyclic, or incompatible dependencies produce actionable errors.
- Repeated attach/detach and startup failure do not leak owned resources or leave callbacks active.
- Existing project code can migrate without importing engine private implementation details.

## 7. Phase 3 — Renderer extension architecture and first engine module

### Work

- Identify renderer services that are stable enough to expose: frame context, scene inputs, resource declarations, capability queries, and pass scheduling.
- Let render features declare inputs, outputs, ordering constraints, quality settings, and resource ownership.
- Validate render dependencies and reject cycles or missing resources before executing the graph.
- Preserve a useful default pipeline assembled from core renderer functionality and official modules.
- Extract one feature as a proof of concept. Sky/atmosphere is a candidate, but extraction must include its lighting/environment contract and configuration rather than just moving shader files.
- Ensure project settings can disable or replace the feature without editing `RenderPipeline`.
- Measure any added scheduling overhead against the Phase 0 baseline.

### Deliverables

- Renderer feature API and graph validation.
- One official render feature shipped through the module host.
- Default project setup that enables it without manual registration.
- Tests for ordering, missing inputs, feature disablement, resize, resource cleanup, and headless rendering.

### Exit criteria

- The proof feature can be enabled, configured, replaced/disabled, and packaged without changing core renderer source.
- Existing frame output remains within agreed regression tolerances.
- Unsupported graphics capabilities fail clearly or select a documented fallback.
- Core renderer no longer directly owns feature-specific configuration that belongs to the extracted module.

## 8. Phase 4 — Player settings, display controls, and UI templates

### Work

- Separate player-facing settings from developer/technical-art controls.
- Define a typed setting registry with categories, valid choices/ranges, defaults, restart requirements, and project policy for whether the option is exposed.
- Provide display mode, resolution, monitor, refresh rate where supported, VSync, and independent frame-cap settings.
- Query platform-supported display modes; do not assume a hardcoded resolution list is complete.
- Implement apply/confirm/revert behavior for disruptive display changes.
- Add graphics options at an abstract level, such as quality tier and effect toggles. Keep implementation-specific parameters in developer tools/project configuration.
- Create reusable UI examples/templates for title, main menu, settings, credits, and About pages. Projects own branding and page content.
- Preserve existing responsive canvas scaling, input focus, gamepad/keyboard navigation, and localization-ready content boundaries.

### Deliverables

- Player settings model and application API.
- Settings screen example wired to the model.
- Safe display-mode confirmation/revert flow.
- Reusable screen templates and project customization example.
- Tests for supported/unsupported display choices, runtime resize propagation, and settings ownership.

### Exit criteria

- A game can present a safe, curated settings UI without exposing engine tuning internals.
- Applying a failed or unconfirmed display mode restores the last working mode.
- UI, render targets, and input coordinates stay synchronized after resolution changes.
- Engine branding is available as metadata but is not forced into the game’s title or About page.

## 9. Phase 5 — Preferences, game saves, and world lifecycle

### Work

- Provide a per-user writable configuration path separate from installed game files.
- Persist engine/player preferences separately from project-owned game progress and developer settings.
- Define a versioned save participant contract for game modules and engine modules that need persistent state.
- Add schema validation, migrations, atomic writes, error reporting, and backup/recovery policy.
- Define portable export/import formats without serializing arbitrary Python object graphs.
- Standardize world/scene load, activate, deactivate, and unload phases.
- Ensure world teardown releases physics bodies, entities, resources, draw batches, event hooks, and module-owned scene state.
- Add serialized scene/world support only after ownership/reference semantics are clear; keep code-driven world builders supported.

### Deliverables

- Preferences service and tested migrations.
- Save-game storage API and a sample project save implementation.
- Module save participant API with namespace/version handling.
- World lifecycle contract and tests for switching worlds during runtime.
- Documentation for backup, portability, and untrusted local save data.

### Exit criteria

- A game save can be exported, loaded, migrated, and rejected safely if invalid or incompatible.
- Engine preferences never overwrite game progress and vice versa.
- Interrupted writes preserve the last valid save.
- Repeated world transitions do not leak engine or project-owned resources.

## 10. Phase 6 — Asset cooking and release pipeline

### Work

- Make module and asset inputs explicit to the build system.
- Resolve transitive engine-module and project-module dependencies from declared metadata.
- Track asset references from scenes/configuration and support explicit declarations for dynamic/runtime-loaded assets.
- Cook source assets into validated runtime formats with deterministic settings and cache keys.
- Produce a build manifest containing engine version, project version, module versions, target, settings, and cooked asset inventory.
- Stage only required runtime code, native dependencies, modules, shaders, and assets where dependency analysis is reliable; conservatively include declared dynamic dependencies.
- Define release policy for developer overlays, profiling, assertions, logs, and crash diagnostics.
- Package a Windows build with the chosen bundler/compiler, then smoke-test it from a clean staging/output directory.
- Document signing, dependency licenses, and the limitations of encryption/IP protection.

### Deliverables

- Build graph and release report.
- Deterministic staging and cooked-asset manifest.
- Debug and release build profiles with explicit behavior.
- Clean-machine or clean-environment Windows smoke-test procedure.
- Sample packaged game that starts, loads a world, reads assets, and shuts down.

### Exit criteria

- The package uses declared modules and assets rather than assuming directory presence means usage.
- Missing dynamic dependencies are diagnosable and can be declared.
- A packaged build runs without source-tree paths or development-only files.
- Rebuilding identical inputs produces equivalent manifests and asset outputs.
- Release builds retain essential diagnostics and make no unsupported security guarantees.

## 11. Phase 7 — Public compatibility and release qualification

### Work

- Freeze the supported public module/project APIs for the release candidate.
- Publish compatibility guarantees, deprecation rules, migration notes, supported platforms, and known limitations.
- Run all unit/integration tests, headless project tests, renderer tests, and packaged smoke tests in CI.
- Run soak tests for repeated scene transitions, resize/mode changes, saves, and shutdown.
- Profile the representative project on minimum and recommended hardware; compare against documented budgets.
- Validate clean installation, first run, settings persistence, save recovery, and upgrade behavior.
- Audit third-party licenses and generated build contents.
- Maintain a release-blocker list and require sign-off for unresolved high-severity issues.

### Deliverables

- Release candidate and compatibility documentation.
- CI report and supported-hardware results.
- Known-issues list and migration guide.
- Reproducible sample project and packaged example.

### Exit criteria

- A developer can follow the documented project workflow from creation to packaged game.
- Project modules and official engine modules obey documented contracts.
- Release tests cover initialization failure, missing modules/assets, display changes, saves, world teardown, and packaged startup.
- No known critical lifecycle, data-loss, or packaging failures remain.
- Performance claims are tied to repeatable scenes and hardware, not unsupported generalizations.

## 12. Cross-phase quality gates

Every phase should satisfy these gates before it is considered complete:

### Correctness

- Targeted tests cover new behavior and critical edge cases.
- Existing behavior is checked for regression.
- Public inputs are validated and errors are reported clearly.

### Lifecycle and resource ownership

- Startup and shutdown are tested, including partial initialization failure.
- GPU, physics, audio, event, file, and module resources have explicit owners.
- Repeated enable/disable, resize, scene transition, and reload paths are exercised.

### Performance and reliability

- Compare relevant CPU/GPU timing and memory against a recorded baseline.
- Include stress cases proportional to the feature: high entity counts, repeated loads, large settings changes, or long-running saves.
- Avoid claiming performance wins from proxy metrics alone.

### Compatibility and migration

- Changes to public interfaces include migration notes or a compatibility shim.
- Project templates and sample games are updated in the same change.
- Old project behavior is either supported, migrated, or explicitly marked incompatible.

### Documentation and packaging

- User-facing and module-author docs match tested behavior.
- Release staging includes all required runtime data and excludes only explicitly excluded development content.
- A packaged build is tested when the phase changes build/runtime boundaries.

## 13. Recommended next public-version scope

Do not attempt to deliver a full marketplace, a general-purpose editor, open-world streaming, every listed engine module, and a new package manager in one public release.

For the next public version, prioritize:

1. A stable project workflow and clear ownership boundaries.
2. A module manifest/resolver with dependable lifecycle and diagnostics.
3. One renderer feature extracted end-to-end to prove the extension architecture.
4. Abstract player settings with safe display changes and separate preferences.
5. A basic project-owned save API and reusable title/settings/credits examples.
6. A deterministic, smoke-tested Windows package.
7. Documentation, compatibility policy, and a sample game proving the entire path.

Defer marketplace infrastructure until external module APIs and dependency/version policy have proven stable with real projects. Defer broad feature extraction until each module boundary has a concrete user and test case.

## 14. Current starting point


- **Optional target** – Android support is opt‑in via a flag in `ProjectConfig` (e.g. `enable_android = True`).  The engine will only attempt Android packaging when this flag is set.
- **Capability detection** – At runtime the engine queries the OpenGL version and available extensions.  Features that require desktop‑class GL 4.5 (multi‑draw‑indirect, compute‑shader GI, high‑resolution cascaded shadow maps, screen‑space reflections) are automatically disabled on ES devices, falling back to simpler implementations (single draw loop, baked GI textures, reduced cascade count, no SSR).
- **Render‑config tunables** – A new `RenderConfig` section lets developers constrain mobile‑specific limits:
  ```python
  RenderConfig(
      max_shadow_cascades=2,  # lower cascade count on phones
      shadow_map_resolution=1024,  # smaller textures
      enable_multi_draw=False,  # fallback when EXT_multi_draw_indirect unavailable
      gi_mode="OFF",  # disable SSGI/Ray-Marched Global Illumination on low‑end devices
      enable_ssr=False,
  )
  ```
- **Build pipeline** – The unified `build.py` script now includes an `--platform android` target that invokes Chaquopy (Gradle) to produce an APK.  The Android module bundles the Python runtime, compiled wheels for ModernGL‑ES, Rapier‑Rust, NumPy, and other dependencies.
- **Asset handling** – The asset cooker adds a mobile profile that generates ASTC/ETC2 compressed texture variants and downsamples meshes where appropriate, keeping APK size reasonable.
- **Documentation** – New sections in `docs/roadmap/ANDROID_SUPPORT.md` (generated) describe how to enable Android, the required Gradle setup, and the supported feature subset.

### Controls: gamepad & on‑screen keys
- **Gamepad integration** – The input subsystem now recognises SDL2 gamepad devices exposed through pygame‑CE.  Developers can enable it via `enable_gamepad = True` in `ProjectConfig`.  Buttons are mapped to virtual `InputAction` names; axes provide continuous analog values.
- **On‑screen touch overlay** – For mobile devices a configurable JSON overlay (`onscreen_keys.json`) defines virtual buttons (positions in the $1280 × 720 reference canvas) that are rendered by the UI system and translated into synthetic `InputEvent` objects.  This allows developers to supply a minimal D‑pad and action buttons without writing custom UI code.
- **Analog handling** – Analog sticks generate `InputAxis` values in `[-1.0, 1.0]` with configurable dead‑zone and sensitivity.  These values can be used directly by character controllers or camera rigs.
- **Unified input manager** – The existing `engine.input` module merges keyboard, mouse, gamepad, and touch sources into a single event stream, preserving the current API while adding the new device classes.

## 14. Current starting point

The repository already includes project folders and examples, `ProjectApp`, fixed/variable frame phases, `ProjectModule`, world-builder hooks, a large render pipeline, UI screens, runtime window controls, partial CVar/graphics serialization, asset cooking/VFS support, and a PyInstaller/Nuitka-oriented packager.


The repository already includes project folders and examples, `ProjectApp`, fixed/variable frame phases, `ProjectModule`, world-builder hooks, a large render pipeline, UI screens, runtime window controls, partial CVar/graphics serialization, asset cooking/VFS support, and a PyInstaller/Nuitka-oriented packager.

The work should build on these foundations rather than replace them wholesale. In particular:

- The existing `ProjectModule` contract is for project gameplay and should not become the only interface for renderer features or importers.
- The existing `Window` API is a foundation, not proof that all player-facing mode enumeration, refresh-rate selection, confirmation/revert, and persistence behavior is complete.
- Existing engine-tweak serialization and CVar configuration are foundations, not yet a unified player-preferences and save-game service.
- Existing UI screens support custom content, but templates and settings integration still need deliberate design.
- Existing cooker, VFS, and packager are foundations, but module/asset reachability, reproducibility, and packaged smoke testing need to be established.

Each phase should verify these details against current code before implementation and update this plan when a tested design decision changes.
