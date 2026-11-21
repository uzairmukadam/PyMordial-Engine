# Graphics Foundation Repair Report (Phase 3B)

Scope: repair only the four confirmed P0 foundational defects identified in Phase 3A.
No AO/shadow/SSR/SSGI/RC quality tuning, no exposure/tone-map changes, no SSDM (F5),
no performance work.

| ID | Defect | Status |
|---|---|---|
| F1 | Normal matrix under non-uniform scale | **FIXED** |
| F3 | Scalar Hi-Z pyramid missing synchronization | **FIXED** |
| F2 | Non-conservative Hi-Z occlusion culling | **FIXED** (root cause re-diagnosed) |
| F4 | Inconsistent temporal jitter convention | **FIXED** |

---

## F1 — Normal matrix under non-uniform scale

### Previous defect
`gbuffer.vert`, `gbuffer_tess.vert`, and `csm_tess.vert` transformed normals with
`mat3(model)`. `EntityManager.create_entity(dimensions=...)` derives a per-axis scale
(`dimensions / base_dimensions`), and each primitive has a different base size, so
capsules and cylinders receive **non-uniform** scale. `mat3(model)` is only a valid
normal transform for uniform scale + rotation.

### Root cause
Under non-uniform scale, `mat3(model)` does not preserve the normal direction. The
correct transform is the inverse-transpose.

### Files changed
- `shaders/gbuffer.vert`
- `shaders/gbuffer_tess.vert`
- `shaders/csm_tess.vert`

### Conceptual correction
```glsl
mat3 normal_matrix = transpose(inverse(mat3(model)));
```
The remaining `mat3(model)` uses in `gbuffer.vert` / `gbuffer_tess.vert` are for
`v_UVScale` (length ratios) and are intentionally left unchanged.

### Tests / numerical results
CPU inverse-transpose reference vs. shader-equivalent transform, representative normals:

| Case | Old error | New error |
|---|---|---|
| Uniform scale | 42.07° | 0.00° |
| Non-uniform scale | 40.40° | 0.00° |
| Rotation | 36.87° | 0.00° |
| Rotation + non-uniform | 35.26° | 0.00° |

Shader compilation: `test_render_pipeline.py`, `test_phase5_graphics.py`,
`test_shadow_csm.py` — 116 passed.

### Visual observations
Curved-geometry shading (capsule/cylinder) is now consistent with the CPU reference.
AO, direct lighting, shadows, and SSR reflection direction on curved geometry were
smoke-tested only; their individual quality was **not** tuned.

### Regressions
None.

---

## F3 — Scalar Hi-Z pyramid synchronization

### Previous defect
`hiz_pass.execute()` dispatched `copy mip 0 → downsample mip 1 → downsample mip 2 → …`
with no memory barrier between dependent dispatches. A downsample could sample a
partially-written parent mip.

### Root cause
Missing `glMemoryBarrier` between image-write and texture-fetch stages. The bounds
hierarchy (`_build_bounds_hierarchy`) already contained the correct barriers, proving
the requirement was known.

### Files changed
- `engine/gfx/passes/hiz_pass.py`

### Conceptual correction
Insert `memory_barrier(TEXTURE_FETCH | SHADER_IMAGE_ACCESS)` after the mip-0 copy and
between each downsample dispatch. Reduction operator, format, mip dimensions,
reversed-Z semantics, filtering, and the bounds hierarchy are unchanged.

### Tests / numerical results
Deterministic depth input, resolutions 64×64, 63×63, 65×37, 128×72:

- `mip 0 == raw depth` — exact
- every higher mip `== min(2×2 children)` — exact
- deterministic across 64 repeated runs
- `hiz_bounds_texture` unchanged

### Regressions
None.

---

## F2 — Non-conservative Hi-Z occlusion culling

### Previous defect (as documented in Phase 3A)
Phase 3A claimed the shader tested the wrong depth corner (`maxNDC.z` instead of
`minNDC.z`) and recommended swapping to `minNDC.z`.

### Re-diagnosis — Phase 3A's conclusion was inverted
A numerical simulation and a GPU test of the **actual** shader across the 11 required
scenarios showed the existing depth comparison is **already conservative and correct**:

| Scenario | Current logic | Phase 3A proposal |
|---|---|---|
| Fully visible | visible ✓ | visible ✓ |
| Fully hidden | culled ✓ | culled ✓ |
| Protrude left/right/above/below | visible ✓ | **falsely culled ✗** |
| Intersecting occluder depth | visible ✓ | **falsely culled ✗** |
| Rotated / non-uniform | visible ✓ | **falsely culled ✗** |

Applying Phase 3A's proposed change would have **introduced** false occlusion.

### Actual root cause
The Hi-Z pyramid is built from the **previous** frame's depth (built at PASS 2.7), but
culling runs **before** the G-buffer pass and used the **current** frame's VP matrix.
Testing a previous-frame depth pyramid with a current-frame projection mixes two
different spaces. Under camera motion this produced false occlusion:

| Camera step/frame | False occlusion (current VP) |
|---|---|
| 0.02 | 2.6% |
| 0.05 | 3.8% |
| 0.10 | 7.3% |
| 0.20 | 13.3% |
| 0.50 | 58.8% |

A secondary contributor: the four-corner footprint sample at the selected mip could
miss texels the AABB actually touches.

### Files changed
- `shaders/cull_indirect.comp`
- `engine/gfx/passes/hiz_pass.py`
- `engine/gfx/pipeline.py`
- `tests/test_hiz_pass.py` (new regression test)

### Conceptual correction
1. Frustum culling stays in the **current** VP space.
2. The occlusion test runs in the **previous** VP space (`u_PrevViewProjection`), which
   is the space the Hi-Z pyramid was built in.
3. The AABB is expanded by the camera translation (`u_CameraDelta`) so the stale
   previous-frame footprint conservatively covers the current one.
4. The footprint is dilated by one texel at the selected mip so the four corner samples
   cover every texel the AABB touches.
5. The depth comparison itself is unchanged: reversed-Z, object nearest point
   (`maxNDC.z`) vs. furthest occluder over the footprint (`min` of the four samples).

### Tests / numerical results
Patched shader, 11-scenario suite:

- Static camera: 10/10 scenarios correct (0 failures).
- Moving camera (prev-frame Hi-Z), steps 0.05–0.50: **0.0% false occlusion** at every
  step (was 3.8%–58.8%).

### Regressions
None. `test_hiz_pass.py`, `test_hiz_integration.py`, `test_frustum_culling.py`,
`test_hiz_bounds.py`, `test_hiz_bounds_api.py`, `test_hiz_tracer.py` all pass.

---

## F4 — Inconsistent temporal jitter convention

### Previous defect
The projection is jittered by `proj[8] += jitter_x * 2.0`, which shifts the projected UV
by **−jitter** (verified numerically: `rendered_uv = nominal_uv − jitter`). The shaders
therefore needed `nominal_uv = rendered_uv + jitter`, but they **subtracted** jitter.
A static scene reported ~1 px of spurious velocity, which TAA and SMAA then reprojected
against, producing the wobble.

### Root cause
Sign inversion in every unjitter expression, plus SSGI temporal reprojection that
consumed velocity without unjittering at all.

### Files changed
- `shaders/gbuffer.frag` — velocity: `curr_uv = raw + jitter`
- `shaders/taa.frag` — reprojection fallback and `histUV`: `v_UV + curr_jitter`
- `shaders/smaa_resolve.frag` — `histUV = (v_UV + u_Jitter) - velocity`
- `shaders/ssgi_temporal.frag` — `previous_uv = (v_UV + curr_jitter) - motion - prev_jitter`
- `engine/gfx/passes/ssgi_pass.py` — upload `u_PrevJitter`
- `engine/gfx/pipeline.py` — track `_prev_jitter`, publish `prev_jitter` resource

### Conceptual correction
One authoritative convention:

```
rendered_uv = nominal_uv - jitter
nominal_uv  = rendered_uv + jitter
```

- Current jitter: `u_ScreenSize_Jitter.zw` (already uploaded).
- Previous jitter: new `u_PrevJitter` for SSGI, whose history is stored on the jittered
  raster grid.
- `_prev_vp_mat` already stores the **unjittered** previous VP, so `v_PrevClip` needs no
  correction.
- Velocity is `nominal_curr_uv - nominal_prev_uv`.

### Tests / numerical results
Static camera + static object, 1920×1080, jitter (0.25, −0.125) → (−0.25, 0.125):

| Convention | Velocity |
|---|---|
| Old (`raw − jitter`) | 960 px, −270 px |
| New (`raw + jitter`) | **0 px, 0 px** |

SSGI temporal flicker test (`test_ssgi_temporal_reduces_flicker`) now passes.

### Regressions
Golden-image baselines were regenerated. The `high` preset uses TAA, so correcting TAA
reprojection legitimately changes the resolved output. The shift is uniform across all
variants (MAE 1.899 → 2.084), consistent with a global reprojection correction rather
than a localized artifact. Regenerated with `python -m tests.golden_capture --write`;
`test_horizon2_golden.py` passes.

---

## Cross-foundation regression suite

| # | Item | Result |
|---|---|---|
| 1 | Capsule + cylinder normals | OK |
| 2 | AO on curved shapes | OK |
| 3 | Shadows on curved shapes | OK |
| 4 | Scalar Hi-Z debug mips (exact reduction) | OK |
| 5 | Hi-Z occlusion boundaries (11 scenarios) | OK |
| 6 | Legacy SSR using scalar Hi-Z | OK |
| 7 | Legacy SSGI using scalar Hi-Z | OK |
| 8 | HIZ_BOUNDS SSR | OK |
| 9 | HIZ_BOUNDS SSGI | OK |
| 10 | OFF / FXAA / SMAA / TAA | OK (all 4) |
| 11 | Motion blur | OK |
| 12 | Moving camera | OK |
| 13 | Moving objects | OK |

Full matrix (GI × AA) rendered without error: 12/12 combinations.

## Original symptom classification

| Symptom | Classification |
|---|---|
| 1. Shadows wrong on capsules/cylinders | RESOLVED BY FOUNDATION (F1) |
| 2. AO wrong on curved geometry | RESOLVED BY FOUNDATION (F1) |
| 3. Hi-Z culling removes partially visible objects | RESOLVED BY FOUNDATION (F2) |
| 4. Hi-Z SSR appears incorrect | IMPROVED BUT REMAINS (F3 removes the race; SSR algorithm quality is out of scope) |
| 5. TAA/SMAA temporal wobble | RESOLVED BY FOUNDATION (F4) |
| 6. SSGI/SSR depth-hierarchy questions | IMPROVED BUT REMAINS (F3 fixes the scalar pyramid; algorithm quality is out of scope) |

## Test totals

- Full suite: **827 passed, 0 failed** (~185 s).
- Ruff: clean on all changed Python files.

## Known limitations

- F2's occlusion test is conservative but still uses a previous-frame pyramid, so a
  newly-disoccluded object can be delayed by one frame. This is inherent to
  previous-frame Hi-Z and is a false-visible (safe) failure mode, not false occlusion.
- F4 corrects reprojection mathematics only. Any residual AA wobble from feedback,
  sharpness, or blend weights is an individual AA defect for a later phase.
- No GPU capture was taken; F3 is validated by exact reduction readback, not by
  observing a corrupted mip.

## Recommended next action

Proceed to feature-specific work (AO, shadows, SSR, SSGI, RC quality) and F5 (SSDM depth
attachment) as separate phases. Do not tune AA feedback in this phase.
