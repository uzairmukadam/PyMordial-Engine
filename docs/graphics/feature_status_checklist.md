# Feature Status Checklist — Phase 2

> **Execution sheet.** One row per variant from [feature_inventory.md](./feature_inventory.md).
> Fill in the result columns as you run [manual_verification_matrix.md](./manual_verification_matrix.md).
>
> **Result values**: `PASS` · `PASS WITH ISSUES` · `FAIL` · `NOT TESTED`
>
> **Rules**
> - `PASS` requires the *stated expected behaviour* to be observed — not merely that something drew.
> - `PASS WITH ISSUES` requires a screenshot and a note naming the artifact.
> - `FAIL` requires a screenshot and the observed-vs-expected difference.
> - Leave `NOT TESTED` rather than guessing.
> - Screenshot path convention: `benchmark/results/verify/<scene>/<feature>_<variant>.png`
>
> **Do not edit the "Status (code)" column** — it is the Phase 2 static finding. The result columns
> are yours to fill.

---

## 0. Session Header

| Field | Value |
|---|---|
| Date | |
| Commit | |
| GPU / driver | |
| Resolution | 1280×720 |
| Quality preset | ultra |
| Warm-up frames | 60 |
| Tester | |

---

## 1. Geometry

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| G1 | MDI base | hardware | working | S1 | | | |
| G2 | MDI fallback | software | working | S1 | | | |
| G3 | Tessellation | `tess_enabled` | working | S3 | | | |
| G4 | Wireframe | `wireframe` | working | S1 | | | |

## 2. Material / PBR

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| M1 | PBR base | GGX | working | S2 | | | |
| M2 | Energy compensation | on/off | working | S2 | | | |
| M3 | Specular AA | on/off | working | S3 | | | |
| M4 | POM | `pom_enabled` | working | S3 | | | |
| M5 | SSDM | `ssdm_enabled` | working | S3 | | | |
| M6 | Material atlas | 4 arrays | working | S3 | | | |

## 3. Direct Lighting

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| L1 | Directional sun | CSM-shadowed | working | S6 | | | |
| L2 | Point lights | ≤256 | working | S1 | | | |
| L3 | Spot lights | ≤64 | working | S7 | | | |
| L4 | Spot shadows | `spot_shadows_enabled` | working (leaks) | S7 | | | |

## 4. Image-Based Lighting

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| I1 | Split-sum IBL | `ibl_enabled` | working | S5 | | | |
| I2 | GI mode IBL | `gi_mode="IBL"` | partial (≡ OFF) | S5 | | | |

## 5. Shadows

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| SH1 | CSM | 1–4 cascades | working | S6 | | | |
| SH2 | Hard filter | `"HARD"` | working | S6 | | | |
| SH3 | PCF | `"PCF"` | working | S6 | | | |
| SH4 | PCSS | `"PCSS"` | working | S6 | | | |
| SH5 | Stabilization | texel-snap | working | S6 | | | |
| SH6 | Legacy removal | `u_SunAngularRadius` | legacy (present) | — | | | |

## 6. Ambient Occlusion

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| A1 | SSAO | `ao_mode="SSAO"` | working | S8 | | | |
| A2 | GTAO legacy | `ao_backend="LEGACY"` | working | S8 | | | |
| A3 | GTAO Hi-Z | `ao_backend="HIZ_BOUNDS"` | working | S8 | | | |
| A4 | AO debug views | `ao_debug_view` 1–4 | working (F2) | S1 | | | |
| A5 | AO backend fallback | `ao_backend="HIZ_BOUNDS"` | observable | S1 | | | |

## 7. Global Illumination

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| GI1 | SSGI legacy | `ssgi_backend="LEGACY"` | working | S1 | | | |
| GI2 | SSGI Hi-Z | `ssgi_backend="HIZ_BOUNDS"` | working | S1 | | | |
| GI3 | SSGI debug | `ssgi_debug_view` 1–8 | working (F2) | S1 | | | |
| GI4 | RC legacy | `rc_backend="LEGACY"` | working | S1 | | | |
| GI5 | RC cascades | `rc_backend="CASCADES"` | working | S1 | | | |
| GI6 | RC debug | `rc_debug_view` | working (F2) | S1 | | | |
| GI7 | GI backend fallback | `ssgi_backend="HIZ_BOUNDS"` | observable | S1 | | | |

## 8. Reflections

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| R1 | SSR legacy | `ssr_backend="LEGACY"` | working | S5 | | | |
| R2 | SSR Hi-Z | `ssr_backend="HIZ_BOUNDS"` | working | S5 | | | |
| R3 | SSR march mode | `ssr_mode` | working (F2) | S5 | | | |
| R4 | SSR debug | `ssr_debug_view` | working (F2) | S5 | | | |
| R5 | SSR backend fallback | `ssr_backend="HIZ_BOUNDS"` | observable | S5 | | | |

## 9. Sky / Atmosphere

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| SK1 | Hillaire sky | `sky_mode="ATMOSPHERE"` | working | S6 | | | |
| SK2 | Space mode | `sky_mode="SPACE"` | working | S6 | | | |
| SK3 | Clouds | `cloud_enabled` | working | S6 | | | |
| SK4 | Presets | 6 presets | working | S6 | | | |
| SK5 | Flat fallback | `atmosphere_enabled=False` | working | S2 | | | |

## 10. Volumetrics / Fog

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| V1 | Froxel fog | `volumetric_fog_enabled` | working | S6 | | | |
| V2 | Fog resolution | LOW→CINEMATIC | working | S6 | | | |
| V3 | Fog debug | `fog_debug_mode` 0–2 | working | S6 | | | |
| V4 | VolumetricSettings | dataclass | stub (unused) | — | | | |

## 11. Transparency

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| T1 | Particle blend | additive/alpha | working | S7 | | | |
| T2 | Water refraction | per-entity | working | S9 | | | |

## 12. Water

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| W1 | Water surface | per-entity | working (leaks) | S9 | | | |
| W2 | Water lab example | `03_water_lighting_lab` | working | S9 | | | |

## 13. Terrain

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| TR1 | Grid terrain | `make_grid_terrain_pm_mesh` | working | S6 | | | |
| TR2 | Heightmap terrain | — | absent | — | | | |

## 14. Particles

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| P1 | Compute sim | `particle_simulate.comp` | working | S7 | | | |
| P2 | DUST_MOTES | `"DUST_MOTES"` | working | S7 | | | |
| P3 | EMBERS | `"EMBERS"` | working | S7 | | | |
| P4 | FIREFLIES | `"FIREFLIES"` | working | S7 | | | |

## 15. Anti-Aliasing

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| AA1 | TAA | `aa_mode="TAA"` | working | S3/S4 | | | |
| AA2 | FXAA | `aa_mode="FXAA"` | working | S3 | | | |
| AA3 | SMAA | `aa_mode="SMAA"` | working | S3 | | | |
| AA4 | No AA | `aa_mode="OFF"` | working | S3 | | | |
| AA5 | AA exclusivity | `aa_mode="FXAA"` + `taa_enabled` | enforced | S3 | | | |

## 16. Resolution Scaling

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| RS1 | AO scale | `ao_resolution_scale` | working | S8 | | | |
| RS2 | SSGI scale | `ssgi_resolution_scale` | working | S1 | | | |
| RS3 | RC scale | `rc_resolution_scale` | working | S1 | | | |
| RS4 | SSR scale | `ssr_resolution_scale` | working | S5 | | | |
| RS5 | Flare scale | `lens_flare_resolution_scale` | working | S6 | | | |

## 17. Tone Mapping

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| TM1 | ACES | `"ACES"` | working | S6 | | | |
| TM2 | AgX | `"AgX"` | working | S6 | | | |
| TM3 | Neutral | `"Neutral"` | working | S6 | | | |
| TM4 | Reinhard | `"Reinhard"` | working (legacy) | S6 | | | |

## 18. Exposure

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| E1 | Manual | `auto_exposure_enabled=False` | working | S6 | | | |
| E2 | Auto | `auto_exposure_enabled=True` | working | S6 | | | |

## 19. Bloom

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| B1 | Bloom | `bloom_enabled` | working | S7 | | | |

## 20. Color Grading

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| CG1 | White balance | `white_balance_enabled` | working (F2) | S6 | | | |
| CG2 | Contrast | `contrast` | working | S6 | | | |
| CG3 | Saturation | `saturation` | working | S6 | | | |
| CG4 | CDL | `lift`/`gamma`/`gain` | working | S6 | | | |
| CG5 | Chromatic aberration | `chromatic_aberration_enabled` | working | S6 | | | |
| CG6 | Vignette | `vignette_enabled` | working | S6 | | | |
| CG7 | Film grain | `film_grain_enabled` | working | S6 | | | |

## 21. Depth of Field

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| D1 | DoF circular | `"CIRCULAR"` | working | S5 | | | |
| D2 | DoF hexagonal | `"HEXAGONAL"` | working | S5 | | | |
| D3 | DoF anamorphic | `"ANAMORPHIC"` | working | S5 | | | |

## 22. Motion Blur

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| MB1 | Motion blur | `motion_blur_enabled` | working | S4 | | | |

## 23. Debug Visualization

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| DB1 | Debug mode 1 | ALBEDO | working | S1 | | | |
| DB2 | Debug mode 2 | NORMALS | working | S1 | | | |
| DB3 | Debug mode 3 | MATERIAL | working | S1 | | | |
| DB4 | Debug mode 4 | DEPTH | working | S1 | | | |
| DB5 | Debug mode 5 | SHADOW_ATLAS | working | S6 | | | |
| DB6 | Debug mode 6 | SHADOW_MASK | working | S6 | | | |
| DB7 | Debug mode 7 | AO | working | S8 | | | |
| DB8 | Debug mode 8 | SSGI | working | S1 | | | |
| DB9 | Debug mode 10 | GI_TOTAL | working | S1 | | | |
| DB10 | Debug mode 11 | HIZ | working | S1 | | | |
| DB11 | Debug mode 12 | SPOT_LIGHTS | working | S7 | | | |
| DB12 | Debug mode 13 | RADIANCE_CASCADES | working | S1 | | | |
| DB13 | Debug mode 14 | CASCADE_INDEX | working | S6 | | | |
| DB14 | Debug mode 15 | CASCADE_COVERAGE | working | S6 | | | |
| DB15 | Debug mode 16 | CASCADE_BOUNDS | working | S6 | | | |
| DB16 | Debug mode 17 | HIZ_BOUNDS_NEAREST | working | S1 | | | |
| DB17 | Debug mode 18 | HIZ_BOUNDS_FARTHEST | working | S1 | | | |
| DB18 | Debug mode 19 | HIZ_BOUNDS_SPAN | working | S1 | | | |
| DB19 | Debug mode 20 | HIZ_BOUNDS_MASK | working | S1 | | | |
| DB20 | Debug mode 21 | HIZ_BOUNDS_INVALID | working | S1 | | | |
| DB21 | Debug mode 22 | HIZ_BOUNDS_RAW | working | S1 | | | |
| DB22 | Debug mode 23 | DIRECT_RADIANCE | working | S1 | | | |
| DB23 | Debug wireframes | `DebugDraw` | working | S1 | | | |
| DB24 | F1/F2/F3 panels | ImGui | working | any | | | |

## 24. Temporal Systems

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| TP1 | TAA history | ping-pong | working | S4 | | | |
| TP2 | SSGI history | ping-pong | working | S1 | | | |
| TP3 | RC history (legacy) | history tex | working | S1 | | | |
| TP4 | RC cascades temporal | `rc_temporal_enabled` | working (F2) | S1 | | | |
| TP5 | Motion vectors | prev VP | working | S4 | | | |

## 25. Culling / LOD

| # | Feature | Variant | Status (code) | Scene | Result | Notes | Screenshot |
|---|---|---|---|---|---|---|---|
| C1 | Frustum culling | `frustum_cull_enabled` | working | S6 | | | |
| C2 | Hi-Z culling | `hiz_cull_enabled` | working | S1 | | | |
| C3 | Mesh LOD | — | absent | — | | | |

## 26. Legacy / Unreachable (confirm, do not fix)

| # | Feature | Variant | Status (code) | Result | Notes |
|---|---|---|---|---|---|
| X1 | `engine.gfx.atmosphere` shim | legacy | legacy | | |
| X2 | `HudOverlay` | legacy | legacy | | |
| X3 | `engine.core.input` | legacy | legacy | | |
| X4 | `engine.core.character_camera` | legacy | legacy | | |
| X5 | `VolumetricSettings` | stub | stub | | |
| X6 | `WaypointGraph` | stub | stub | | |
| X7 | `CVarRegistry` | stub | stub | | |
| X8 | `rc_cascades` unused | **removed** | removed | Phase 2.5: dead field deleted |
| X9 | `motion_blur_samples` unused | **fixed** | working | Phase 2.5: now consumed |
| X10 | `ssr_mode` | **exposed** | working | Phase 2.5: F2 "SSR March Mode" |
| X11 | `ssr_debug_view` | **exposed** | working | Phase 2.5: F2 |
| X12 | `ao_debug_view` | **exposed** | working | Phase 2.5: F2 |
| X13 | `ssgi_debug_view` | **exposed** | working | Phase 2.5: F2 |
| X14 | `rc_debug_view` | **exposed** | working | Phase 2.5: F2 |
| X15 | `rc_temporal_debug_view` | **exposed** | working | Phase 2.5: F2 |
| X16 | `rc_temporal_enabled` | **exposed** | working | Phase 2.5: F2 |
| X17 | `rc_spatial_filter_enabled` | **exposed** | working | Phase 2.5: F2 |
| X18 | `ssgi_surface_reject` | **exposed** | working | Phase 2.5: F2 |
| X19 | `energy_compensation_enabled` | **exposed** | working | Phase 2.5: F2 Diagnostics |
| X20 | `specular_aa_enabled` | **exposed** | working | Phase 2.5: F2 Diagnostics |
| X21 | `object_motion_vectors_enabled` | **exposed** | working | Phase 2.5: F2 Diagnostics |
| X22 | `white_balance_enabled` | **exposed** | working | Phase 2.5: F2 Diagnostics |
| X23 | `u_SunAngularRadius` dead uniform | **not dead** | working | Phase 2.5: used by `csm_sampling.glsl` |

---

## 27. Performance Sweeps

Record **median** and **p95** frame time in ms. Never FPS.

| # | Feature | Pass | Parameter | Baseline ms (median/p95) | Variant ms (median/p95) | Δ | Result |
|---|---|---|---|---|---|---|---|
| PF1 | Tessellation | G-Buffer | `tess_max_level` 8→24 | | | | |
| PF2 | POM | G-Buffer | `pom_max_samples` 16→64 | | | | |
| PF3 | CSM | CSM | `shadow_resolution` 2048→8192 | | | | |
| PF4 | PCF | Resolve | `pcf_samples` 8→32 | | | | |
| PF5 | PCSS | Resolve | `pcss_samples` 12→32 | | | | |
| PF6 | AO | AO | `ao_resolution_scale` 1.0→0.5 | | | | |
| PF7 | SSGI | SSGI total | `ssgi_steps` 8→32 | | | | |
| PF8 | RC | RC total | `rc_ray_count` 8→32 | | | | |
| PF9 | SSR | SSR | `ssr_steps` 24→64 | | | | |
| PF10 | Fog | Volumetric fog | `fog_resolution` LOW→CINEMATIC | | | | |
| PF11 | Bloom | Bloom | `bloom_iterations` 3→6 | | | | |
| PF12 | Auto-exposure | Exposure | on/off | | | | |
| PF13 | Hi-Z culling | G-Buffer | `hiz_cull_enabled` on/off | | | | |
| PF14 | TAA | TAA | on/off | | | | |

---

## 28. Summary Roll-Up

Fill after all rows are complete.

| Category | Total | PASS | PASS WITH ISSUES | FAIL | NOT TESTED |
|---|---|---|---|---|---|
| Geometry | 4 | | | | |
| Material / PBR | 6 | | | | |
| Direct lighting | 4 | | | | |
| IBL | 2 | | | | |
| Shadows | 6 | | | | |
| AO | 4 | | | | |
| GI | 6 | | | | |
| Reflections | 4 | | | | |
| Sky / atmosphere | 5 | | | | |
| Volumetrics | 4 | | | | |
| Transparency | 2 | | | | |
| Water | 2 | | | | |
| Terrain | 2 | | | | |
| Particles | 4 | | | | |
| Anti-aliasing | 4 | | | | |
| Resolution scaling | 5 | | | | |
| Tone mapping | 4 | | | | |
| Exposure | 2 | | | | |
| Bloom | 1 | | | | |
| Color grading | 7 | | | | |
| Depth of field | 3 | | | | |
| Motion blur | 1 | | | | |
| Debug visualization | 24 | | | | |
| Temporal | 5 | | | | |
| Culling / LOD | 3 | | | | |
| Legacy / unreachable | 23 | | | | |
| **TOTAL** | **135** | | | | |

---

## 29. Blocking Issues Log

Record anything that prevents a row from being tested.

| # | Blocker | Affects rows | Workaround | Owner |
|---|---|---|---|---|
| | | | | |

---

## 30. Files Inspected

`docs/graphics/feature_inventory.md`, `docs/graphics/manual_verification_matrix.md`,
`docs/graphics/feature_dependency_matrix.md`, `docs/audit/*.md`.

## 31. Files Changed

None. Only `docs/graphics/*.md` created.

## 32. Tests Performed

None — this is the execution sheet for the manual phase.

## 33. Results

Checklist prepared with **135 rows** across 26 categories, each with Result / Notes / Screenshot
fields, plus 14 performance sweeps and a roll-up table.

## 34. Known Limitations

- Row count (135) counts each variant separately; the inventory's "~120 variants" figure excludes
  the 23 legacy/unreachable confirmations and the 24 debug modes counted individually here.
- The roll-up totals must be recomputed if rows are added during execution.

## 35. Recommended Next Action

Run the matrix in the order given in
[manual_verification_matrix.md](./manual_verification_matrix.md) §11 (S2+S6 → S1+S8 → S3+S5 →
S4+S7+S9), filling this sheet as you go. Stop and report after the first category that produces a
`FAIL`, rather than continuing — a failure in a base feature invalidates every dependent row.
