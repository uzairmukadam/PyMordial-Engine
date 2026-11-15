# PyMordial Engine Documentation

Everything in this directory is about **working with** or **working on** PyMordial Engine. Future plans live in a separate [`roadmap/`](./roadmap/) tree so that current behaviour and future intent never get confused.

---

## Start Here

Pick the row that matches what you are doing.

| I want to… | Read |
|---|---|
| **Build a game** with the engine | [guides/DEVELOPER_GUIDE.md](./guides/DEVELOPER_GUIDE.md) |
| Build **menus, HUDs, and pause screens** | [guides/UI_AND_STATE_MANAGEMENT.md](./guides/UI_AND_STATE_MANAGEMENT.md) |
| Look up **what a subsystem does** | [reference/ENGINE_FEATURES.md](./reference/ENGINE_FEATURES.md) |
| Understand **how the engine is wired** | [reference/ARCHITECTURE.md](./reference/ARCHITECTURE.md) |
| **Run or write tests** | [reference/TESTING.md](./reference/TESTING.md) |
| **Have an AI agent** work on this repo | [reference/LLM_INSTRUCTIONS.md](./reference/LLM_INSTRUCTIONS.md) |
| See **what is planned next** | [roadmap/](./roadmap/) |

---

## Directory Layout

```text
docs/
├── README.md                          ← you are here
│
├── guides/                            Task-oriented: how to build things
│   ├── DEVELOPER_GUIDE.md             Projects, modules, worlds, packaging
│   └── UI_AND_STATE_MANAGEMENT.md     Native UI widgets + GameStateManager
│
├── reference/                         Lookup-oriented: how things work
│   ├── ARCHITECTURE.md                Layer boundaries, frame flow, pass order, invariants
│   ├── ENGINE_FEATURES.md             Subsystem-by-subsystem capability spec
│   ├── TESTING.md                     Test layout, headless GL, conventions
│   └── LLM_INSTRUCTIONS.md            Hard rules for AI agents
│
└── roadmap/                           Future-oriented: what is not built yet
    ├── README.md                      Index + reading order
    ├── MASTER_FUTURE_PLAN.md          ★ Authoritative status tracker
    ├── PYMORDIAL_UPGRADE_ROADMAP.md   Beta-transition strategy & rationale
    ├── MODULAR_ENGINE_ARCHITECTURE.md Modular target design
    ├── MODULAR_ENGINE_IMPLEMENTATION_PLAN.md  Staged execution plan
    ├── MODULAR_ENGINE_MIGRATION_PROMPTS.md    Per-session agent prompts
    ├── WORLD_BUILDING_ROADMAP.md      Open-world gap analysis
    ├── SSR_PLAN.md                    Hi-Z SSR design
    └── ANDROID_SUPPORT.md             Android/GLES proposal
```

---

## The Three Categories

**`guides/` — how to build.** Written for someone shipping a game. Assumes the engine works and explains how to use it. If a guide is wrong, a developer gets stuck.

**`reference/` — how it works.** Written for someone changing the engine. Describes actual current behaviour and the invariants that must hold. If a reference doc is wrong, a developer introduces a silent bug.

**`roadmap/` — what is not built.** Written for planning. Every document here carries a status banner (`PROPOSAL`, `DESIGN`, `ANALYSIS`, `STRATEGY`) so it is never mistaken for a description of working code. The single authoritative tracker is [roadmap/MASTER_FUTURE_PLAN.md](./roadmap/MASTER_FUTURE_PLAN.md).

---

## Documentation Rules

These keep the tree from decaying back into a junk drawer.

1. **One category per document.** If a doc mixes "how to use it" with "what we might build", split it.
2. **Roadmap docs carry a status banner.** The first block after the title states whether the content is implemented, partial, or proposed.
3. **Reference docs describe reality.** If code and doc disagree, the doc is wrong — fix it in the same change as the code.
4. **No orphan files.** Every document is linked from this README or from `roadmap/README.md`.
5. **Relative links only.** Use `./sibling.md` and `../guides/other.md` so the tree stays portable.
6. **Verify links before committing.** A broken link is a broken doc.

---

## Related Documentation Outside `docs/`

| Location | Contents |
|---|---|
| [`../README.md`](../README.md) | Project overview, install, quick start |
| [`../AGENTS.md`](../AGENTS.md) | Condensed rules for AI pair programming |
| [`../examples/README.md`](../examples/README.md) | The seven runnable examples and their controls |
