# PyMordial Engine: Roadmap

> **Nothing in this directory describes working code unless its status banner says so.**
> For current behaviour, see [`../reference/`](../reference/). For how to build a game, see [`../guides/`](../guides/).

---

## The Authoritative Tracker

**[MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md)** is the single source of truth for what is done, in progress, and planned. Every other document here is supporting material — a design, an analysis, or a strategy — that feeds into it.

If two documents disagree about status, `MASTER_FUTURE_PLAN.md` wins.

---

## Status Vocabulary

Every roadmap document opens with one of these banners:

| Status | Meaning |
|---|---|
| `DONE` | Implemented, tested, and verified. |
| `IN PROGRESS` | Actively being built. |
| `PROPOSAL` | A target design. Little or none of it exists yet. |
| `DESIGN` | A technical design for one feature. May be partially implemented. |
| `ANALYSIS` | A gap assessment or problem statement. Not a plan of record. |
| `STRATEGY` | High-level direction and rationale. Superseded in part by the master plan. |

---

## Reading Order

Read in this order if you are new to the roadmap.

```text
1. MASTER_FUTURE_PLAN.md          ← what is happening and when
        │
        ├── 2. PYMORDIAL_UPGRADE_ROADMAP.md
        │      Why the engine is shaped this way (pruning rationale,
        │      hardware tiers, Decima inspiration)
        │
        ├── 3. MODULAR_ENGINE_ARCHITECTURE.md
        │      Target design for the module system
        │        └── MODULAR_ENGINE_IMPLEMENTATION_PLAN.md
        │              Staged execution
        │                └── MODULAR_ENGINE_MIGRATION_PROMPTS.md
        │                      Per-session agent prompts
        │
        ├── 4. WORLD_BUILDING_ROADMAP.md
        │      What open-world support still needs
        │
        ├── 5. SSR_PLAN.md
        │      Hi-Z SSR technical design
        │
        └── 6. ANDROID_SUPPORT.md
               Android / GLES proposal
```

---

## Document Index

| Document | Status | Horizon | Purpose |
|---|---|:---:|---|
| [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) | `DONE` / `PLANNED` | 1–5 | **Authoritative** horizon dashboard and task tracker |
| [PYMORDIAL_UPGRADE_ROADMAP.md](./PYMORDIAL_UPGRADE_ROADMAP.md) | `STRATEGY` | all | Beta-transition blueprint, hardware tiers, pruning rationale |
| [MODULAR_ENGINE_ARCHITECTURE.md](./MODULAR_ENGINE_ARCHITECTURE.md) | `PROPOSAL` | 2 | Target module-system design |
| [MODULAR_ENGINE_IMPLEMENTATION_PLAN.md](./MODULAR_ENGINE_IMPLEMENTATION_PLAN.md) | `PROPOSAL` | 2 | Phased delivery of the module system |
| [MODULAR_ENGINE_MIGRATION_PROMPTS.md](./MODULAR_ENGINE_MIGRATION_PROMPTS.md) | `PROPOSAL` | 2 | Copy-paste agent prompts per increment |
| [WORLD_BUILDING_ROADMAP.md](./WORLD_BUILDING_ROADMAP.md) | `ANALYSIS` | 4 | Open-world gap assessment |
| [SSR_PLAN.md](./SSR_PLAN.md) | `DESIGN` | — | Hi-Z accelerated SSR design |
| [ANDROID_SUPPORT.md](./ANDROID_SUPPORT.md) | `PROPOSAL` | — | Android / OpenGL ES 3.0 target |

---

## Horizon Summary

| Horizon | Focus | Status |
|---|---|:---:|
| **1** | Foundation clean-up, pruning, true hardware MDI | `DONE` |
| **2** | Module host architecture & render extension graph | `PLANNED` |
| **3** | Player settings, world lifecycle & persistence | `PLANNED` |
| **4** | Decima-grade world systems & streaming | `PLANNED` |
| **5** | Packaging, CI qualification & distribution | `PLANNED` |

Details, task breakdowns, and verification gates live in [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md).

---

## Adding a Roadmap Document

1. Give it a status banner immediately after the title.
2. Link it from the index table above **and** from the reading-order tree if it belongs to a chain.
3. Reference the horizon it serves, or `—` if it is unscheduled.
4. If it changes what is planned, update [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) too — do not let a side document become a second source of truth.
