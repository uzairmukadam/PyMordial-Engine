# PyMordial Engine: Incremental Migration Prompts

> **Status: PARTIALLY EXECUTED — Horizon 2 prompts (P2.1–P2.5, P3.1–P3.4) are complete.** These prompts drive the remaining phases in [MODULAR_ENGINE_IMPLEMENTATION_PLAN.md](./MODULAR_ENGINE_IMPLEMENTATION_PLAN.md). Do not run them until the corresponding Horizon 3+ task is scheduled.

## How to use this guide

This is an execution companion to:

- [Modular architecture proposal](./MODULAR_ENGINE_ARCHITECTURE.md)
- [Multi-phase implementation plan](./MODULAR_ENGINE_IMPLEMENTATION_PLAN.md)

Use **one prompt per coding session**. Do not paste the whole guide into an agent request. Start at Phase 0, select the next unchecked prompt, and stop at its checkpoint. This keeps context small, makes regressions easier to locate, and avoids large refactors that are difficult to review.

Recommended loop:

1. Start from a clean, named branch or checkpoint. Preserve unrelated worktree changes.
2. Paste the short **Shared instructions** below, then exactly one task prompt.
3. Let the agent inspect the relevant code before proposing edits. Ask it to state scope and likely tests before changing code.
4. Review the diff. Run the exact validation requested by the prompt.
5. If validation fails, fix only failures caused by that increment; distinguish pre-existing failures explicitly.
6. Commit or otherwise checkpoint the verified increment before starting the next prompt.
7. Begin the next session with the previous prompt’s short handoff: changed files, design decisions, tests run, unresolved risks, and the next prompt ID.

Do not run several prompts in parallel if they touch the same architecture, API, or files. If a prompt reveals a decision with multiple reasonable designs, stop for a decision rather than implementing a broad guess.

## Shared instructions

Prepend these instructions to each task prompt:

> Work only on the single increment below. First inspect the relevant implementation and tests, then summarize current behavior and the smallest safe change. Preserve unrelated worktree changes. Do not perform a broad refactor, move files without establishing interfaces, or add dependencies unless required. Implement only after the scope is clear. Add focused tests for new behavior, run the smallest relevant test/lint/format checks, and report exact results. If a product/API decision materially changes compatibility or scope, stop and ask instead of guessing. Finish with a concise handoff: files changed, behavior added, validation, known limitations, and the next prompt ID. Do not commit unless explicitly requested.

## Phase 0 — Baseline and public-version scope

### P0.1 — Inventory the real project workflow

**Prompt**

> Read the architecture proposal and implementation plan, then trace the current developer workflow from project configuration through `ProjectApp` startup, world creation, frame execution, and packaging. Do not edit code. Produce a concise table of current behavior, relevant files/APIs, mismatches with the target design, and the smallest migration risks. Explicitly distinguish verified facts from assumptions. Recommend one representative sample project for later end-to-end testing.

**Checkpoint:** Agree on the inventory and sample project before changing APIs.

### P0.2 — Define public and internal API boundaries

**Prompt**

> Using the current code and architecture proposal, identify which APIs should be considered public for the next release and which should remain internal/experimental. Do not edit implementation code. Recommend a minimal compatibility policy for project modules and engine feature modules, including deprecation/error behavior. Keep the output to a decision table and open questions; do not design the marketplace or promise compatibility beyond the next release.

**Checkpoint:** Resolve open API decisions. Record approved decisions in the architecture document only if requested.

### P0.3 — Establish a representative baseline

**Prompt**

> Inspect the existing tests, examples, and project sample. Establish the smallest repeatable baseline that exercises startup, input, at least one fixed update, physics, rendering, UI, one optional engine feature, and shutdown. Prefer reusing an existing example/project. Add only missing test or measurement hooks; do not build a benchmark framework. Record commands and metrics available for CPU frame time, GPU frame time, startup, and memory, noting measurements that cannot currently be collected reliably.

**Checkpoint:** Baseline can be run again and yields comparable results before and after changes.

## Phase 1 — Project workflow and configuration

### P1.1 — Specify the project directory contract

**Prompt**

> Define the minimal game-directory contract for PyMordial. A game is source and assets hosted by the Python engine; it does not have to be an independently installable Python package. Inspect existing projects and propose the smallest convention for entry point, game config, assets, worlds, and project modules. Do not add a manifest unless a concrete runtime/build need requires declarative metadata. Do not change project code yet. Report migration impact for existing projects.

**Checkpoint:** Agree on directory layout and whether/what metadata is necessary.

### P1.2 — Add early project validation

**Prompt**

> Implement the agreed minimal project validation at the existing project startup/build boundary. Validate only required inputs such as project path, entry point, configuration shape, and asset roots that are already part of the workflow. Fail before expensive runtime initialization and report actionable errors. Preserve existing project startup behavior. Add focused valid/invalid project tests; do not add module dependency resolution in this increment.

**Checkpoint:** Existing sample project still starts; invalid projects fail early and clearly.

### P1.3 — Normalize run and test invocation

**Prompt**

> Inspect the current commands and documentation for running and testing projects. Make the smallest change needed to support the agreed project-directory workflow consistently, including an explicit path for projects outside `projects/` only if that is an approved requirement. Do not add build or module selection features here. Add a smoke test for command/path resolution and update directly related docs.

**Checkpoint:** A project can be selected unambiguously without relying on the current working directory accidentally.

### P1.4 — Create or refine the starter project

**Prompt**

> Create or refine a minimal starter game using the agreed project layout. It should start, create a tiny world, demonstrate one project gameplay module, and shut down cleanly. Keep content intentionally small and avoid introducing new engine features. Add a headless smoke test and concise run instructions. Do not modify the existing featured game unless it is the selected baseline.

**Checkpoint:** Starter is reproducible and is the reference project for later phases.

## Phase 2 — Module host and contracts

### P2.1 — Define module roles and metadata

**Prompt**

> Inspect `ProjectModule` and current engine extension points. Propose the minimum module metadata needed for identity, role, version/API compatibility, dependencies, and configuration. Separate gameplay modules from renderer features conceptually. Do not implement discovery, dependency resolution, or file relocation yet. Identify which fields must be explicit and which can remain Python API values for the first release.

**Checkpoint:** Approve the metadata/schema before coding the loader.

### P2.2 — Add manifest parsing and validation

**Prompt**

> Implement parsing and validation for the approved module metadata format. Keep it independent of runtime initialization. Validate identifiers, versions/API fields, dependency declarations, and malformed input. Add focused tests for valid, invalid, duplicate, and unsupported-version manifests. Do not import or execute module code in this increment.

**Checkpoint:** Metadata validation is deterministic, side-effect-free, and reports exact errors.

### P2.3 — Resolve dependency graphs

**Prompt**

> Implement deterministic dependency resolution for validated module metadata: required dependencies, supported version constraints, missing dependencies, and cycle detection. Return a stable initialization order and a useful diagnostic on failure. Do not initialize modules or alter `ProjectApp` lifecycle yet. Add graph-focused tests including independent modules and transitive dependencies.

**Checkpoint:** Same inputs always produce the same order; invalid graphs fail before runtime startup.

### P2.4 — Integrate gameplay module lifecycle

**Prompt**

> Integrate the module host with existing gameplay modules using the approved contract. Preserve compatibility with current project modules where practical. Define attach/detach ordering, enabled behavior, event ownership expectations, and cleanup when an attach hook fails. Do not register render passes in this change. Add tests for normal attach/detach, failure rollback, dependency order, and repeated shutdown.

**Checkpoint:** Existing sample gameplay modules still work; failed initialization leaves no partially active modules.

### P2.5 — Add module enable/disable and compatibility behavior

**Prompt**

> Add the smallest supported enable/disable and API compatibility behavior for loaded gameplay modules. Clarify whether runtime toggling is allowed or requires restart; implement only the agreed behavior. Missing or incompatible required modules must fail clearly; optional dependencies may be absent only when declared optional. Add focused tests and migration notes for existing project modules.

**Checkpoint:** Module state and unsupported API behavior are documented and testable.

## Phase 3 — Renderer extension architecture

### P3.1 — Define render-feature contracts

**Prompt**

> Inspect `RenderPipeline`, render passes, `RenderGraphContext`, and the selected sky pass. Propose the minimum stable render-feature contract: required services/resources, pass ordering, outputs, configuration, capability checks, and resource ownership. Do not refactor the pipeline yet. Produce a dependency/order diagram for extracting sky and identify any renderer functionality that must remain core.

**Checkpoint:** Approve the contract and prove it can express the existing frame without arbitrary ordering callbacks.

### P3.2 — Add render graph declarations and validation

**Prompt**

> Implement the approved render-feature registration/declaration API and graph validation without moving any existing rendering feature. Validate missing inputs, duplicate outputs, invalid ordering, and cycles before execution. Preserve the current hardcoded render path as the active path until validation tests pass. Add focused graph tests; avoid runtime performance work in this increment.

**Checkpoint:** Graph validation catches invalid schedules; existing rendering is unchanged.

### P3.3 — Route one existing stage through the graph

**Prompt**

> Route one low-risk existing render stage through the new graph while leaving all other stages in their current order. Choose the stage with the fewest hidden dependencies after inspecting the code. Verify resource transitions, resize, headless behavior, and image/test output. Do not extract sky or change visual algorithms in this prompt.

**Checkpoint:** The selected stage is graph-driven with no rendering regression.

### P3.4 — Extract sky as the proof engine module

**Prompt**

> Move the sky/atmosphere capability behind the approved engine-feature interface. Include its configuration, shader/resource ownership, lighting/environment outputs, resize, and cleanup—not just file locations. Preserve the default project appearance and make disablement explicit. Avoid unrelated sky visual changes. Add tests for default enablement, disablement, missing capability handling, resource cleanup, and pipeline integration.

**Checkpoint:** A project gets the default sky without manual wiring and can disable/replace it without editing core pipeline code.

### P3.5 — Verify graph cost and migration

**Prompt**

> Measure the new render-feature scheduling path against the Phase 0 baseline using the same sample and settings. Add no optimization unless there is a measured regression. Update migration docs and tests to cover old project configuration. Report exact CPU/GPU measurements, variance, and limitations; do not claim a speedup from noisy measurements.

**Checkpoint:** Scheduling overhead is within the agreed budget or has a narrowly scoped remediation plan.

## Phase 4 — Player settings and UI templates

### P4.1 — Separate player settings from developer tuning

**Prompt**

> Inventory current window controls, CVars, engine tweaks, and serialization. Define a typed player-facing settings model separate from technical/debug controls. Include ownership, allowed values, defaults, restart requirements, and project policy for exposure. Do not build UI or persist settings yet. Keep the initial set limited to window/display and abstract graphics choices.

**Checkpoint:** Agree on the public settings model and application semantics.

### P4.2 — Implement display-mode enumeration and application

**Prompt**

> Implement the agreed display API for windowed, borderless, and supported exclusive modes; monitor, resolution, refresh rate where the platform exposes it; VSync; and an independent frame cap. Query valid platform modes rather than assuming a fixed list. Preserve the current mode if an application fails and report unsupported combinations. Add unit/headless tests where possible and a manual test matrix for platform-dependent behavior.

**Checkpoint:** Supported modes are enumerated accurately, and unsupported requests do not leave the window in a broken state.

### P4.3 — Add display confirmation and revert

**Prompt**

> Add a bounded confirmation/revert flow for disruptive display changes. Save the last known-good mode, apply the candidate, start a timeout, and either commit after confirmation or restore the previous mode. Handle focus loss, window close, repeated changes, and restoration errors explicitly. Add deterministic tests using a fake/recording window backend; do not build the settings screen yet.

**Checkpoint:** An unconfirmed change reliably restores the prior usable mode.

### P4.4 — Apply abstract graphics options

**Prompt**

> Implement the approved player-facing graphics setting mapping for a small set of toggles/quality levels (for example lens flare on/off and overall quality). Keep detailed effect parameters in developer tuning. Define precedence among engine defaults, project defaults, and player overrides. Do not add every renderer control; add only settings with a real stable application path. Add tests for mapping, validation, and preset behavior.

**Checkpoint:** Player options alter only the intended effect/quality behavior and never expose internal tuning accidentally.

### P4.5 — Create settings and information screen examples

**Prompt**

> Add customizable UI examples/templates for a title/main menu, settings, credits, and About screen using existing `UIScreen` and layout conventions. Keep engine name/version available as metadata but do not hardcode engine branding into a game screen. Ensure keyboard/gamepad focus, responsive 1280x720 canvas scaling, and project-owned text/content. Wire the settings example to the settings API without embedding window logic in widgets.

**Checkpoint:** The starter project can navigate the screens at multiple resolutions and customize all game-facing content.

## Phase 5 — Preferences, saves, and world lifecycle

### P5.1 — Resolve per-user storage paths

**Prompt**

> Implement or refine a platform-aware writable user-data path service for engine preferences and game saves. Keep user data outside the installed game directory and keep preference and save namespaces separate. Add tests for path selection using injected platform/environment values; do not implement save serialization yet.

**Checkpoint:** Paths are deterministic, writable where expected, and testable without touching real user data.

### P5.2 — Persist player preferences safely

**Prompt**

> Persist only the approved player settings through the user-data service. Add schema versioning, validation, migration from the current configuration format if applicable, and atomic writes. Corrupt or unsupported settings must report a warning/error and recover using documented defaults without overwriting the source file silently. Add focused migration, invalid-input, and interrupted-write tests.

**Checkpoint:** Preferences survive restart and invalid files do not silently destroy user data.

### P5.3 — Add project-owned game save API

**Prompt**

> Add a versioned save/load API for project-owned game state. Do not serialize arbitrary Python objects. Define a small data-oriented contract with schema version, validation, atomic writes, export/import, and explicit error results. Add a sample save participant in the starter project and tests for round-trip, invalid data, migration, and failed writes. Keep engine preferences separate.

**Checkpoint:** The sample game can save, reload, export, and import progress using a documented format.

### P5.4 — Add module save participants

**Prompt**

> Extend the save API so explicitly registered modules can contribute namespaced, versioned data. Define order, missing participant behavior, migration, and failure policy. Keep participants opt-in; do not introspect arbitrary module fields. Add tests for duplicate namespaces, module missing on load, failed participant serialization, and schema migration.

**Checkpoint:** Module data is isolated and failures cannot produce a falsely successful save.

### P5.5 — Formalize world activation and teardown

**Prompt**

> Inspect existing world builder teardown and project module ownership. Define and implement the smallest world lifecycle contract needed for load, activate, deactivate, and unload while preserving code-driven builders. Ensure entities, physics bodies, draw batches, events, and world-scoped module resources are released. Add repeated world transition and partial-load-failure tests. Do not implement open-world streaming yet.

**Checkpoint:** Repeated world changes are clean; no stale entity, callback, physics body, or draw batch remains.

## Phase 6 — Asset cook and release pipeline

### P6.1 — Make build inputs explicit

**Prompt**

> Connect project validation and module dependency resolution to the packager. Produce a build plan/report that lists core runtime, selected engine modules, project modules, and unresolved dynamic dependencies. Do not change final bundler commands or delete staging content yet. Add tests proving directory presence alone does not mark a module as used.

**Checkpoint:** The build plan is deterministic and explains each included dependency.

### P6.2 — Declare and validate asset references

**Prompt**

> Define how projects declare static and dynamically loaded assets. Integrate with the current cooker/VFS in a small step: validate paths and report missing references, but conservatively include declared dynamic asset roots. Do not attempt unreliable Python source scanning. Add tests for missing, duplicate, and runtime-declared assets.

**Checkpoint:** Required runtime assets are discoverable and omissions are diagnosed before bundling.

### P6.3 — Produce deterministic cooked asset outputs

**Prompt**

> Inspect current asset cooking formats and add deterministic build metadata/cache keys for one asset family only. Preserve existing runtime format compatibility. Test that identical inputs/options produce equivalent outputs and changed inputs invalidate the cache. Do not add encryption in this increment.

**Checkpoint:** One cooker path is repeatable and has a measurable cache invalidation rule.

### P6.4 — Stage a minimal runtime build

**Prompt**

> Update packager staging to consume the explicit build plan. Include required engine runtime, selected modules, project scripts, native/runtime dependencies, shaders, and cooked assets. Exclude only files explicitly classified as development-only. Do not claim Python code is fully opaque or strip diagnostics. Add staging tests that inspect contents and ensure the project runs from staging, not from the source checkout.

**Checkpoint:** A clean staged game starts and loads its world without source-tree paths.

### P6.5 — Add release profiles and Windows smoke test

**Prompt**

> Define explicit debug and release build profiles, preserving useful crash/error diagnostics in both. Build the agreed Windows target and add a smoke-test step that launches the packaged game with a bounded test mode, confirms startup/assets/world load, and checks exit status/logs. Keep signing/encryption out unless separately approved. Document licenses and the limits of asset protection.

**Checkpoint:** The packaged artifact is tested from a clean output directory and produces a useful failure report.

## Phase 7 — Public compatibility and release qualification

### P7.1 — Publish API and migration policy

**Prompt**

> Review the implemented public project/module/render/settings/save APIs and write a concise compatibility policy, migration guide, and known limitations. Verify every documented behavior against code/tests. Do not promise marketplace compatibility or cross-platform behavior that has not been tested.

**Checkpoint:** Docs, API declarations, and the sample project agree.

### P7.2 — Add release CI gates

**Prompt**

> Add or refine CI checks for focused unit/integration suites, headless starter project execution, package staging, and packaged smoke tests. Keep jobs bounded and produce actionable artifacts/logs. Do not run deployment or publish steps. Report platform gaps where a check cannot run in CI.

**Checkpoint:** A PR can detect common API, lifecycle, and packaging regressions before release.

### P7.3 — Run release-candidate qualification

**Prompt**

> Execute the documented release qualification against the starter project and representative game: tests, repeated startup/shutdown, world transitions, display changes, preference/save recovery, packaged startup, and performance baseline. Do not make broad fixes in one change. Produce a prioritized issue list with reproduction steps and classify release blockers versus follow-up work.

**Checkpoint:** Release blockers are fixed in separate small prompts and the qualification is rerun.

### P7.4 — Final clean-machine and upgrade validation

**Prompt**

> Validate the release artifact from a clean Windows environment or documented clean VM: install/extract, first launch, settings persistence, save creation/load, missing/corrupt data behavior, and uninstall/update considerations. Verify included dependency licenses and build manifest. Make no speculative product changes; report any issue as a separate narrow follow-up.

**Checkpoint:** The published distribution can be installed and used without the development checkout.

## 5. Per-prompt review checklist

Before accepting each increment, verify:

- [ ] The diff stays within the prompt’s scope.
- [ ] Existing worktree changes were preserved.
- [ ] Tests fail for the intended regression if the implementation is removed or broken.
- [ ] Relevant tests and lint/type checks pass, or unrelated baseline failures are recorded.
- [ ] Error handling is visible and does not silently pretend success.
- [ ] Resource ownership and cleanup are covered for new lifecycle paths.
- [ ] Documentation describes shipped behavior rather than proposed behavior.
- [ ] The next prompt's prerequisites are genuinely satisfied.

## 6. Handoff format between sessions

After a prompt is complete, save a short note for the next session:

```text
Completed: P2.3 — Resolve dependency graphs
Changed: <files>
Decisions: <approved API choices>
Validation: <commands and pass/fail>
Known issues: <only concrete unresolved items>
Next: P2.4 — Integrate gameplay module lifecycle
```

Keep the handoff factual and compact. In the next session, provide the handoff plus only the next prompt, not the full conversation history.

## 7. Recommended execution order and parallelism

Follow the prompt order as written unless a phase checkpoint explicitly approves a change:

```text
P0 → P1 → P2 → P3 → P4/P5 → P6 → P7
```

Phases 4 and 5 may be planned in parallel, but their code changes should remain small and should not run concurrently if they modify shared settings, module, UI, or persistence interfaces. Phase 6 must wait for module and asset declarations to stabilize. Phase 7 must validate the integrated product rather than individual subsystems in isolation.

Do not skip directly from architecture design to extracting all graphics features. The sky extraction is a proof of the module/render contract; use what it teaches before moving water, GI, post-processing, or other systems.

## 8. Suggested stopping points

If time or token budget is limited, complete and checkpoint one prompt at a time. The most valuable early milestones are:

1. P0.1–P0.3: know the current state and measure a baseline.
2. P1.1–P1.4: establish the project workflow and starter.
3. P2.1–P2.4: create a minimal, reliable gameplay module host.
4. P3.1–P3.4: prove one real engine feature can be modular.
5. Continue with the remaining prompts only after those foundations are stable.

Stopping after a completed checkpoint is preferable to starting multiple unfinished refactors.
