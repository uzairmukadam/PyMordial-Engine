# Auto-Exposure — Phase 3.3

## 1. Original observation

> "Auto-exposure doesn't seem to be working."

## 2. Recovery state

A previous session began Phase 3.3 and looped. The working tree was inspected
before any change:

| Item | Finding |
|---|---|
| `git status` | Clean except three untracked `_diag_exposure*.py` scripts |
| `git diff --stat` | Empty — no uncommitted Phase 3.3 code changes |
| Last commit | `f10bf24` (Phase 3B foundation repair + octahedral fix) |
| Tests | `tests/test_exposure.py` existed with 11 passing tests |
| Docs | `docs/graphics/features/` did not exist |
| Diagnostic code | Only the three untracked scripts; no production instrumentation |

**Conclusion:** no Phase 3.3 code changes had been made. The exposure
implementation was intact and its own unit tests passed. The defect had to be
elsewhere.

## 3. Execution path

```
RenderConfig.auto_exposure_enabled
  → RenderPipeline.render_frame
      → PASS 7 (deferred resolve) writes post_process.hdr_fbo
      → PASS 7.1  ctx.resources["hdr_color"] = hdr_fbo.color_attachments[0]
                  ctx.delta_time = calc_dt
                  exposure_pass.execute(ctx)
                      → exposure_histogram.comp   (64×64 dispatch, 256 bins)
                      → memory_barrier
                      → exposure_adapt.comp       (1×1, ping-pong RGBA16F)
                      → readback → resources["exposure_ev"]
      → PASS 8    post_process.render(exposure_ev=ctx.resources["exposure_ev"])
                      → u_Exposure = 2^EV * config.exposure
                      → tone map → sRGB encode → final_fbo
```

Key resources:

| Resource | Detail |
|---|---|
| Input | `hdr_color` (RGBA16F, post-resolve, pre-bloom) |
| Histogram | SSBO binding 10, 256 × uint32, cleared by a zero write each frame |
| State | 1×1 RGBA16F ping-pong; `.r` = adapted EV, `.g` = target EV, `.b` = key luminance |
| Output | `resources["exposure_ev"]` (float or `None`) |
| Consumer | `PostProcessPipeline.render(exposure_ev=...)` |

## 4. Deterministic diagnostic setup

Synthetic HDR textures were fed directly into `ExposurePass` through a minimal
`RenderGraphContext` stand-in, so every stage could be read back numerically
without relying on visual inspection.

Scenes: A nearly dark (0.002), B medium grey (0.18), C very bright (8.0),
D dark→bright, E bright→dark, F bright point (0.1 % area), G bright sky (60 % area).

## 5. Measurements by stage

### TEST 1 — Input

Distinct HDR values reach the pass. **PASS.**

### TEST 2 — Histogram

| Scene | Total | Bins used | Peak bin |
|---|---|---|---|
| A dark | 4096 | 1 | 28 |
| B medium | 4096 | 1 | 87 |
| C bright | 4096 | 1 | 137 |

`A==B`, `B==C`, `A==C` all False. Population is exactly 4096 (64×64), bins in
range, NaN pixels rejected. **PASS.**

### TEST 3 — Target EV

| Scene | Key luminance | Target EV |
|---|---|---|
| A dark | 0.0021 | +6.672 |
| B medium | 0.1855 | +0.219 |
| C bright | 8.2188 | −5.246 |

Direction is correct: darker scene → higher EV. **PASS.**

### TEST 4 — Adaptation

| Transition | Frame 1 | Frame 60 | Target |
|---|---|---|---|
| D dark→bright | +6.602 | +2.496 | −5.246 |
| E bright→dark | −5.219 | −3.635 | +6.672 |

Moves toward target in both directions, converges, respects clamps, no NaN.
**PASS.**

### TEST 5 — Post-process consumption

`post_process.render` receives `exposure_ev` and sets
`u_Exposure = 2^EV * config.exposure`. Verified end-to-end: enabling the flag
changes the final image mean (120.2 → 145.8) and `adapted_ev` moves off 0.
**PASS.**

### TEST 6 — Final response

Final LDR output differs between auto and fixed exposure. **PASS.**

## 6. First failing stage

**None of the six stages failed.** The exposure pipeline is functionally
correct.

The defect is **reachability**: `auto_exposure_enabled` defaults to `False` and
nothing could turn it on.

| Reachability path | Status before fix |
|---|---|
| Quality presets | No preset sets `auto_exposure_enabled = True` |
| Debug menu (F2) | No control existed |
| `EngineTweaks` | No field, no `to_dict`/`from_dict`/`apply_to_pipeline` entry |
| `ProjectConfig.custom_settings` | Never applied to `RenderConfig` |
| `settings.ini` | Written by "Save Settings" but never read back |

So the feature was inert in every shipped configuration — which is exactly what
"auto-exposure doesn't seem to be working" describes.

## 7. Root cause

The exposure controller was implemented and correct, but had no user-facing
activation path. It was dead code behind a default-off flag.

## 8. Files changed

| File | Change |
|---|---|
| `engine/debug/engine_tweaks.py` | Added 6 slots + defaults, `apply_to_pipeline`, `sync_from_config`, `to_dict` entries |
| `engine/debug/debug_menu.py` | Added "Auto Exposure" checkbox, 5 tuning sliders, live telemetry line |
| `tests/test_exposure.py` | Added `TestAutoExposureReachability` (6 tests) |

No shader, no exposure algorithm, no tone-mapping change.

## 9. Correction

Expose the existing controller through the debug UI and the settings
serialization path, without altering its behaviour:

- `EngineTweaks` gains `auto_exposure_enabled`, `exposure_key`,
  `exposure_min_ev`, `exposure_max_ev`, `exposure_adapt_speed_up`,
  `exposure_adapt_speed_down`.
- `apply_to_pipeline` writes them onto `RenderConfig`.
- `sync_from_config` reads them back.
- `to_dict`/`from_dict` persist them, so "Save Settings" now round-trips.
- The F2 menu shows the toggle, the tuning sliders (only when enabled), and a
  live telemetry line: measured key luminance, target EV, adapted EV, multiplier.

Defaults are unchanged (`False`, key 0.5, EV −8…12, speeds 1.5/4.0), so existing
projects keep their fixed exposure.

## 10. Before/after results

| | Before | After |
|---|---|---|
| Can a user enable auto-exposure? | No | Yes (F2 → Auto Exposure) |
| Persisted across Save Settings? | No | Yes |
| Live telemetry available? | No | Yes |
| Exposure math | Correct | Unchanged |
| Default behaviour | Fixed exposure | Fixed exposure (unchanged) |

## 11. Transition tests

Dark→bright and bright→dark both adapt monotonically toward target over 60
frames with no oscillation, no pumping, and no stuck value. Empty histogram
(all-NaN frame) correctly holds the previous EV rather than snapping to zero.

## 12. Frame-rate test

One second of adaptation from the same start:

| Frame rate | Adapted EV after 1 s | Δ |
|---|---|---|
| 30 fps | +2.592 | −4.080 |
| 120 fps | +2.453 | −4.219 |

Δ differs by 3.4 % — within numerical expectation for a per-frame clamped
integrator. Adaptation is effectively frame-rate independent.

## 13. Automated regression results

- `tests/test_exposure.py`: **17 passed** (11 existing + 6 new reachability).
- Full suite: **837 passed, 0 failed**.
- Ruff: clean on all changed files.

## 14. Remaining limitations

- `ProjectConfig.custom_settings` is still never applied to `RenderConfig`, and
  `settings.ini` is still write-only. Auto-exposure now persists through the
  debug menu's own JSON path, but a project cannot yet declare it in
  `ProjectConfig`. That is a pre-existing configuration-system gap, out of scope
  for this phase.
- The histogram samples a fixed 64×64 grid; very small bright features can be
  missed by the stride. This is intentional low-pass behaviour, not a defect.
- Adaptation speeds are the existing defaults and were not tuned.

## 15. Manual verification instructions

1. Run the Playground project.
2. Press **F2** → *Rendering: Quality & Diagnostics*.
3. Tick **Auto Exposure**. The telemetry line appears.
4. Look at a dark area, then at a bright area. `target EV` should change
   immediately and `adapted EV` should follow over ~1–2 s.
5. Confirm the image brightens in dark areas and darkens in bright areas.
6. Untick **Auto Exposure** — the image should return to the fixed `Exposure`
   slider value.
7. Press **Save Settings (.ini)** and confirm the values persist.

## Classification

**READY FOR MANUAL APPROVAL**
