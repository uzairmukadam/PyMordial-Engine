# Graphics Benchmarking

`scripts/benchmark_graphics.py` is a small, reproducible development tool that
measures the real deferred render pipeline. It does **not** optimize or change
any graphical feature — it only records what the existing renderer does.

## Purpose

Use it to compare:

- total CPU frame time and total GPU frame time
- individual GPU pass times (G-buffer, AO, GI, SSR, deferred resolve, …)
- the cost of enabling/disabling a feature
- quality presets and backend variants
- different camera workloads

## Invocation

```bash
python scripts/benchmark_graphics.py --config BASELINE_LOW --view VIEW_TOP
python scripts/benchmark_graphics.py --config HIGH --view VIEW_CLOSE --frames 240
python scripts/benchmark_graphics.py --config BASELINE_LOW --view VIEW_TOP \
    --set gi_mode=RMGI --set ssr_enabled=true --output build/benchmarks/rmgi.json
```

Run `--help` for all options.

| Option | Default | Meaning |
|---|---|---|
| `--config` | `BASELINE_LOW` | Named configuration (see below) |
| `--view` | `VIEW_TOP` | Deterministic camera workload |
| `--resolution` | `1280x720` | Render resolution `WIDTHxHEIGHT` |
| `--warmup` | `60` | Frames excluded from statistics |
| `--frames` | `240` | Measured frames |
| `--output` | `build/benchmarks/<config>_<view>_<time>.json` | JSON output path |
| `--set KEY=VALUE` | — | Override any `RenderConfig` field (repeatable) |
| `--windowed` | off | Render to a visible window instead of headless |
| `--quiet` | off | Suppress human-readable output |

## Built-in views

| View | Purpose |
|---|---|
| `VIEW_TOP` | Elevated overview of the whole scene |
| `VIEW_CLOSE` | Close geometry, high screen coverage, contact detail |
| `VIEW_REFLECTIVE` | Emphasizes reflective surfaces / SSR coverage |
| `VIEW_DEPTH` | Long depth range for GI / fog |
| `VIEW_MOTION` | Deterministic 240-frame orbit for temporal effects |

Static views use identical camera state every run. `VIEW_MOTION` uses the same
frame-indexed orbit every run.

## Benchmark configs

Each config is derived from an existing quality preset, then adjusted with
explicit overrides:

| Config | Base preset | Notes |
|---|---|---|
| `BASELINE_LOW` | `low` | GI/SSR/fog/bloom/lens/DOF/motion-blur/tess/SSDM/POM/auto-exposure off |
| `LOW` | `low` | Preset as-is |
| `MEDIUM` | `medium` | Preset as-is |
| `HIGH` | `high` | Preset as-is |
| `ULTRA` | `ultra` | Preset as-is |
| `CINEMATIC` | `cinematic` | Preset as-is |

Use `--set` to override individual fields for feature experiments. Mutually
exclusive settings (one AA mode, one GI mode, one AO mode) are respected by the
engine's existing `apply_config` semantics — the tool never forces conflicting
alternatives.

## Output fields

The JSON result contains:

- `schema_version`, `tool`, `timestamp_utc`
- `resolution`, `scene` (id, entity count, shape counts), `view`
- `warmup_frames`, `measured_frames`, `config_name`, `config`
- `active_feature_state` — the **live** active state (from `pipeline.feature_state()`)
- `device` (GL renderer/version/vendor), `cpu`
- `scene_stats` (visible entities, draw commands, triangle estimate)
- `timing.cpu_frame_ms` and `timing.gpu_frame_ms` — each with min/median/mean/p95/max
- `passes` — per-pass GPU timing statistics

`timing.gpu_frame_ms` is `null` when the asynchronous query has not settled;
`gpu_frame_ms_available` reports this explicitly rather than substituting zero.

## Comparing results

Compare pass timings **only under identical view and configuration**. A pass
that is disabled reports no entry; a pass that is enabled but unavailable
reports its fallback in `active_feature_state`.

## Warmup requirements

Temporal effects (TAA, SSGI/RMGI history, auto-exposure) need several frames to
stabilize. The default `--warmup 60` is sufficient for the built-in views; raise
it if you change temporal settings. Warmup frames are never included in stats.

## Known sources of variance

- GPU clock/power state and driver scheduling (run-to-run GPU medians vary by a
  few percent).
- CPU scheduling noise on shared machines.
- Asynchronous GPU queries lag a few frames; the tool harvests only settled
  results.
- Headless runs do not present to a window, so CPU wall time excludes
  presentation cost; use `--windowed` to include it.

## Why pass timings matter

Total GPU time hides which pass dominates. Per-pass timings let you attribute a
regression to a specific stage (e.g. `RC total` for RMGI, `SSR`, `AO`) and
confirm a feature toggle actually changed the work performed.
