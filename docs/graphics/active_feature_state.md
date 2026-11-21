# Active Feature State — Phase 2.5

> Reference for the live graphics state surface added in Phase 2.5. This is the contract that
> debug tooling and (later) the benchmark manifest read to prove **what actually executed**, as
> opposed to what was requested.

---

## 1. Why This Exists

Before Phase 2.5, a requested backend could silently fall back:

```text
config: ssr_backend = "HIZ_BOUNDS"
actual: LEGACY  (because hiz_bounds_texture was missing)
reported: nothing
```

A benchmark would then record timings labelled `HIZ_BOUNDS` for a run that executed `LEGACY`.
The state surface makes that impossible to hide.

---

## 2. API

### 2.1 Per-pass attributes

Every switchable pass exposes three attributes, refreshed on each `execute()`:

| Attribute | Type | Meaning |
|---|---|---|
| `requested_backend` | `str` | The backend the configuration asked for |
| `active_backend` | `str` | The backend that actually ran |
| `fallback_reason` | `str \| None` | `None` if the request was honoured, else a short cause |

Implemented on:

- `AmbientOcclusionPass` — `"SSAO"`, `"LEGACY"`, `"HIZ_BOUNDS"`, `"OFF"`
- `SSGIPass` — `"LEGACY"`, `"HIZ_BOUNDS"`, `"OFF"`
- `SSRPass` — `"LEGACY"`, `"HIZ_BOUNDS"`, `"OFF"`
- `RadianceCascadesPass` — `"LEGACY"`, `"CASCADES"`, `"OFF"`

### 2.2 Pipeline aggregate

```python
state = pipeline.feature_state()
```

Returns:

```python
{
  "gi": {
    "requested_mode": "SSGI",          # from config
    "active_mode": "SSGI",             # "OFF" if no GI pass ran
    "requested_backend": "HIZ_BOUNDS",
    "active_backend": "LEGACY",
    "fallback_reason": "missing Hi-Z bounds texture",
    "temporal_enabled": True,
    "resolution_scale": 1.0,
  },
  "ao": {
    "mode": "GTAO",
    "requested_backend": "HIZ_BOUNDS",
    "active_backend": "HIZ_BOUNDS",
    "fallback_reason": None,
    "resolution_scale": 1.0,
  },
  "ssr": {
    "enabled": True,
    "requested_backend": "LEGACY",
    "active_backend": "LEGACY",
    "fallback_reason": None,
    "resolution_scale": 1.0,
  },
  "aa": {
    "requested_mode": "TAA",
    "active_stages": ("TAA",),         # at most one
  },
  "shadows": {"filter": "PCF", "cascades": 4, "resolution": 8192},
  "fog": {"enabled": True, "froxel_resolution": "HIGH"},
  "environment": {"provider": "AtmosphereEnvironment", "atmosphere_enabled": True},
}
```

### 2.3 AA stages

```python
pipeline.active_aa_stages()  # -> ("OFF"|"FXAA"|"SMAA_1X"|"SMAA_2X"|"SMAA_4X"|"TAA") or ()
```

Returns at most one entry. Empty tuple means no AA.

---

## 3. Semantics

### 3.1 `requested` vs `active`

| Situation | `requested` | `active` | `fallback_reason` |
|---|---|---|---|
| Backend honoured | `HIZ_BOUNDS` | `HIZ_BOUNDS` | `None` |
| Bounds texture missing | `HIZ_BOUNDS` | `LEGACY` | `"missing Hi-Z bounds texture"` |
| Feature disabled | `LEGACY` | `OFF` | `None` |
| No G-buffer | `LEGACY` | `OFF` | `"no G-buffer"` |
| GI mode not selected | `LEGACY` | `OFF` | `None` |

### 3.2 `gi_mode="IBL"`

`"IBL"` is **not** a screen-space GI technique. It means "no SSGI/RC; rely on the always-on
split-sum IBL". Reported as:

```python
{"requested_mode": "IBL", "active_mode": "IBL"}   # when ibl_enabled
{"requested_mode": "IBL", "active_mode": "OFF"}   # when ibl_enabled is False
```

`requested_backend` / `active_backend` are `None` for `"IBL"` because no GI pass runs.

### 3.3 AA exclusivity

`aa_mode` is authoritative. `taa_enabled` is derived:

```python
taa_enabled == (aa_mode == "TAA")
```

`apply_config` normalises this. A contradictory config (`aa_mode="FXAA"`, `taa_enabled=True`) is
corrected to `taa_enabled=False`, and `active_aa_stages()` returns `("FXAA",)`.

An unknown `aa_mode` falls back to `"OFF"` and sets `pipeline._aa_warning`.

---

## 4. Consumers

| Consumer | Uses |
|---|---|
| F2 "Active Feature State" panel | `feature_state()` — renders requested vs active with a fallback badge |
| `tests/test_feature_activation.py` | asserts `requested == active` or `fallback_reason is not None` |
| Benchmark manifest (future) | will record `feature_state()` per run |

---

## 5. Verification Matrix (activation, not visual)

| Feature | Requested | Expected `active` | Test |
|---|---|---|---|
| AO | `SSAO` | `SSAO` | `TestAOActivation::test_ssao_reports_ssao` |
| AO | `GTAO`/`LEGACY` | `LEGACY` | `test_gtao_legacy` |
| AO | `GTAO`/`HIZ_BOUNDS` | `HIZ_BOUNDS` or fallback | `test_gtao_hiz_bounds_or_explicit_fallback` |
| AO | `OFF` | `OFF` | `test_ao_off` |
| GI | `OFF` | `OFF` | `test_gi_off` |
| GI | `IBL` | `IBL` | `test_gi_ibl_is_not_off` |
| GI | `SSGI`/`LEGACY` | `SSGI`/`LEGACY` | `test_ssgi_legacy` |
| GI | `SSGI`/`HIZ_BOUNDS` | `HIZ_BOUNDS` or fallback | `test_ssgi_hiz_bounds_or_fallback` |
| GI | `RC`/`LEGACY` | `LEGACY` | `test_rc_legacy` |
| GI | `RC`/`CASCADES` | `CASCADES` | `test_rc_cascades` |
| SSR | `OFF` | `OFF` | `test_ssr_off` |
| SSR | `LEGACY` | `LEGACY` | `test_ssr_legacy` |
| SSR | `HIZ_BOUNDS` | `HIZ_BOUNDS` or fallback | `test_ssr_hiz_bounds_or_fallback` |
| AA | all 6 modes | ≤1 stage | `TestAAExclusivity::test_single_stage` |
| AA | `FXAA` + `taa_enabled` | `("FXAA",)` | `test_spatial_mode_clears_taa_flag` |
| AA | `TAA` | `("TAA",)` | `test_taa_mode_sets_taa_flag` |
| AA | `BOGUS` | `()` | `test_unknown_mode_falls_back_to_off` |
| Shadows | `HARD`/`PCF`/`PCSS` | reported | `TestShadowAndFogState` |
| Fog | 5 resolutions | reported | `TestShadowAndFogState` |
| Motion blur | `motion_blur_samples=7` | uniform == 7 | `TestMotionBlurSamples` |

**31 tests, all passing.**

---

## 6. Known Gaps

1. No strict mode that *fails* a benchmark on unexpected fallback. The state is queryable; the
   policy belongs to the benchmark phase.
2. `feature_state()` does not report per-cascade shadow filter layering (`pcf_cascade_limit`).
3. `environment.provider` reports the class name only; it does not report *why* a fallback occurred
   (the warning is logged, not stored). A future revision could store `environment_fallback_reason`.
