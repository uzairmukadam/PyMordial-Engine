# Graphics Feature Inventory — Phase 2

> **Documentation-only phase.** No implementation was modified.
> **Basis**: static trace of every feature from config → Python → pass → shader → target → consumer,
> cross-checked against `RenderConfig`, `EngineTweaks`, `DebugMenu`, `QUALITY_PRESETS`, and the
> shader include graph. See [../audit/engine_architecture.md](../audit/engine_architecture.md) for
> the frame order this inventory references.
>
> **Status vocabulary**
> - **working** — traced end-to-end, has a test, no known gap.
> - **partially working** — renders, but a documented parameter or variant is unreachable or unused.
> - **broken** — traced to a defect that prevents the intended result.
> - **stub** — code exists, produces no meaningful output.
> - **legacy** — superseded, still reachable.
> - **unreachable** — implemented but no activation path exists from config or UI.
> - **unknown** — could not be verified statically.
>
> **A feature is NOT marked "verified" here.** Status reflects *code reachability*, not visual
> correctness. Visual correctness is the job of
> [manual_verification_matrix.md](./manual_verification_matrix.md).
>
> **Phase 2.5 update**: several Phase 2 classifications were corrected after independent
> re-confirmation. See [configuration_audit.md](./configuration_audit.md) §3. In particular
> `ssr_mode` and `ssr_debug_view` **are** consumed (now exposed in F2), `motion_blur_samples` is now
> consumed, `u_SunAngularRadius` is **not** dead, and `rc_cascades` was removed. Backend fallbacks
> are now observable via `RenderPipeline.feature_state()`.

---

## 0. Coverage Summary

| Category | Features found | Working | Partial | Unreachable | Legacy |
|---|---|---|---|---|---|
| Geometry rendering | 4 | 3 | 1 | 0 | 0 |
| Material / PBR | 6 | 5 | 1 | 0 | 0 |
| Direct lighting | 4 | 3 | 1 | 0 | 0 |
| Image-based lighting | 2 | 1 | 1 | 0 | 0 |
| Shadows | 6 | 4 | 1 | 0 | 1 |
| Ambient occlusion | 4 | 2 | 1 | 1 | 0 |
| Global illumination | 6 | 3 | 2 | 1 | 0 |
| Reflections | 4 | 2 | 1 | 1 | 0 |
| Sky / atmosphere | 5 | 4 | 1 | 0 | 0 |
| Volumetrics / fog | 4 | 3 | 1 | 0 | 0 |
| Transparency | 2 | 2 | 0 | 0 | 0 |
| Water | 2 | 2 | 0 | 0 | 0 |
| Terrain | 2 | 1 | 1 | 0 | 0 |
| Particles | 4 | 3 | 1 | 0 | 0 |
| Anti-aliasing | 4 | 3 | 1 | 0 | 0 |
| Resolution scaling | 5 | 5 | 0 | 0 | 0 |
| Tone mapping | 4 | 4 | 0 | 0 | 0 |
| Exposure | 2 | 2 | 0 | 0 | 0 |
| Bloom | 1 | 1 | 0 | 0 | 0 |
| Color grading | 7 | 6 | 1 | 0 | 0 |
| Depth of field | 3 | 3 | 0 | 0 | 0 |
| Motion blur | 1 | 1 | 0 | 0 | 0 |
| Debug visualization | 24 modes | 24 | 0 | 0 | 0 |
| Temporal systems | 5 | 4 | 1 | 0 | 0 |
| Culling / LOD | 3 | 2 | 1 | 0 | 0 |
| **Not present** | LOD, upscaling (FSR/DLSS/TSR), alpha-blended geometry, heightmap terrain | — | — | — | — |

**Confirmed absent** (searched, not found): mesh LOD selection, temporal upscaling, FSR/DLSS/TSR,
a general alpha-blended transparent geometry pass, a heightmap terrain system, and a
deferred/forward hybrid switch.

---

## 1. Geometry Rendering

### 1.1 MegaBuffer + Multi-Draw Indirect (base geometry)
| Field | Value |
|---|---|
| Variant | hardware MDI |
| Status | **working** |
| Activation | always; `MultiDrawIndirect.hardware_mdi` resolved at construction |
| Python | [mega_buffer.py](../../engine/gfx/mega_buffer.py), [mdi.py](../../engine/gfx/mdi.py), [pipeline.py:1211](../../engine/gfx/pipeline.py:1211) |
| Shaders | `gbuffer.vert`, `gbuffer.frag` |
| Inputs | ECS `world_transforms`, `material_data`; `MeshAllocation` batches |
| Outputs | G-Buffer RT0–RT4 + depth |
| Temporal | none |
| Depends on | `MegaBuffer` indices pre-offset; `baseVertex` must be 0 |
| Excludes | none |
| Quality params | none (geometry is exact) |
| Intended result | all opaque geometry rasterized once |
| Test scene | S1, S3, S4 |
| Camera | any |
| Expected | no mangled/collapsed meshes; draw count ≪ entity count |
| Known artifacts | if `baseVertex` ≠ 0, geometry collapses (invariant violation) |
| Debug view | `GBufferDebugMode.ALBEDO` (1) |
| Pass/fail | every mesh renders at its ECS transform; no missing first-N vertices |
| Perf testable | yes — draw-call count and CPU gather time |

### 1.2 MDI software fallback
| Field | Value |
|---|---|
| Variant | per-command `vao.render` with `u_BaseInstance` |
| Status | **working** (only on GPUs lacking `GL_ARB_multi_draw_indirect`) |
| Activation | automatic when `hardware_mdi` is False |
| Python | [mdi.py](../../engine/gfx/mdi.py) |
| Shaders | same as 1.1 |
| Inputs / Outputs | same as 1.1 |
| Temporal | none |
| Depends on | `u_BaseInstance` uniform present in vertex shader |
| Excludes | none |
| Quality params | none |
| Intended result | identical image to 1.1 |
| Test scene | S1 |
| Camera | any |
| Expected | pixel-identical to hardware path |
| Known artifacts | none expected; must not double-draw |
| Debug view | ALBEDO |
| Pass/fail | image matches hardware MDI |
| Perf testable | yes (slower by design) |

### 1.3 Hardware tessellation (displacement)
| Field | Value |
|---|---|
| Variant | `tess_enabled` |
| Status | **working** |
| Activation | `RenderConfig.tess_enabled` (default True); F2 "Hardware GPU Tessellation" |
| Python | [pipeline.py:1333](../../engine/gfx/pipeline.py:1333), `gbuffer_tess_prog` |
| Shaders | `gbuffer_tess.vert/.tesc/.tese`, `csm_tess.vert/.tesc/.tese` |
| Inputs | displacement texture array, per-material depth, camera distance |
| Outputs | displaced G-Buffer + displaced shadow casters |
| Temporal | `u_PrevViewProjection` for motion vectors |
| Depends on | `GL_ARB_tessellation_shader`; `TextureArrayAtlas.displacement_array` |
| Excludes | none (coexists with POM) |
| Quality params | `tess_max_level` (24), `tess_med_level` (8), `tess_displacement_scale`, `disp_near_radius` (120), `disp_mid_radius` (300) |
| Intended result | real geometric displacement near camera, fading to flat at mid radius |
| Test scene | S3 (brick/concrete materials) |
| Camera | close (< 120 m) then far (> 300 m) |
| Expected | visible relief up close, smooth flat surface far away, no cracks |
| Known artifacts | tessellation cracks at patch borders; popping at radius boundary |
| Debug view | NORMALS (2) to see displaced normals |
| Pass/fail | relief visible near, no cracks, no popping artifacts at radius |
| Perf testable | yes — GPU time scales with `tess_max_level` |

### 1.4 Wireframe
| Field | Value |
|---|---|
| Variant | `wireframe` |
| Status | **working** |
| Activation | `RenderConfig.wireframe`; F2 "Mesh Wireframe" |
| Python | [pipeline.py:1443](../../engine/gfx/pipeline.py:1443) |
| Shaders | `gbuffer.frag` (polygon mode is GL state) |
| Inputs | same as 1.1 |
| Outputs | G-Buffer with line rasterization |
| Temporal | none |
| Depends on | none |
| Excludes | visually incompatible with AA (lines alias) |
| Quality params | none |
| Intended result | debug wireframe of all geometry |
| Test scene | S1 |
| Camera | any |
| Expected | every triangle edge visible |
| Known artifacts | heavy aliasing |
| Debug view | n/a |
| Pass/fail | wireframe toggles on/off cleanly |
| Perf testable | no (debug only) |

---

## 2. Material and PBR

### 2.1 Cook-Torrance PBR (base)
| Field | Value |
|---|---|
| Variant | GGX/Smith/Schlick |
| Status | **working** |
| Activation | always |
| Python | [core/materials.py](../../engine/core/materials.py), [pipeline.py:1674](../../engine/gfx/pipeline.py:1674) |
| Shaders | `common/pbr_brdf.glsl`, `deferred_resolve.frag` |
| Inputs | RT0 albedo+roughness, RT1 normal+metallic, material layer index |
| Outputs | HDR radiance |
| Temporal | none |
| Depends on | G-Buffer |
| Excludes | none |
| Quality params | `energy_compensation_enabled` (default True) |
| Intended result | physically plausible metal/rough response |
| Test scene | S2 (rough vs metallic), S5 (reflective) |
| Camera | any |
| Expected | metals tint specular by albedo; rough surfaces broaden highlight |
| Known artifacts | energy loss at high roughness without compensation |
| Debug view | MATERIAL (3) |
| Pass/fail | metal/rough distinction visually correct; no white-out |
| Perf testable | yes (resolve cost) |

### 2.2 Energy compensation
| Field | Value |
|---|---|
| Variant | `energy_compensation_enabled` |
| Status | **working** |
| Activation | `RenderConfig.energy_compensation_enabled` (default True) |
| Python | [pipeline.py:1654](../../engine/gfx/pipeline.py:1654) |
| Shaders | `deferred_resolve.frag:468` |
| Inputs | roughness, metallic |
| Outputs | boosted specular at high roughness |
| Temporal | none |
| Depends on | 2.1 |
| Excludes | none |
| Quality params | on/off only |
| Intended result | rough metals do not darken |
| Test scene | S2 |
| Camera | any |
| Expected | rough metal brightness preserved |
| Known artifacts | over-brightening if mis-tuned |
| Debug view | MATERIAL |
| Pass/fail | rough metal luminance stable vs smooth metal |
| Perf testable | no (negligible) |
| **Note** | **Not exposed in F2** (0 hits in `engine_tweaks.py`/`debug_menu.py`). Config-only. |

### 2.3 Specular anti-aliasing
| Field | Value |
|---|---|
| Variant | `specular_aa_enabled` |
| Status | **working** |
| Activation | `RenderConfig.specular_aa_enabled` (default True) |
| Python | [pipeline.py:1405](../../engine/gfx/pipeline.py:1405) |
| Shaders | `gbuffer.frag:330` |
| Inputs | normal variance |
| Outputs | widened roughness in G-Buffer |
| Temporal | none |
| Depends on | G-Buffer |
| Excludes | none |
| Quality params | `specular_aa_strength` (1.0) |
| Intended result | kills specular shimmer on high-frequency normals |
| Test scene | S3 (dense geometry) |
| Camera | slow pan |
| Expected | no crawling specular sparkle |
| Known artifacts | over-blurred highlights if strength too high |
| Debug view | MATERIAL |
| Pass/fail | static camera → no shimmer; pan → no crawl |
| Perf testable | no |
| **Note** | **Not exposed in F2.** Config-only. |

### 2.4 Parallax Occlusion Mapping (POM)
| Field | Value |
|---|---|
| Variant | `pom_enabled` |
| Status | **working** |
| Activation | `RenderConfig.pom_enabled`; F2 "Parallax Occlusion Mapping" |
| Python | [pipeline.py:1384](../../engine/gfx/pipeline.py:1384) |
| Shaders | `gbuffer.frag:92` (`ParallaxOcclusionMap`) |
| Inputs | height texture array, view direction, per-material depth |
| Outputs | perturbed UV + depth in G-Buffer |
| Temporal | none |
| Depends on | `TextureArrayAtlas` height array |
| Excludes | none |
| Quality params | `pom_min_samples` (8), `pom_max_samples` (64), `pom_height_scale`, `pom_self_shadow` |
| Intended result | fake depth relief without geometry |
| Test scene | S3 |
| Camera | grazing angle |
| Expected | bricks appear recessed; no swimming |
| Known artifacts | UV swimming at grazing angles; self-shadow acne |
| Debug view | NORMALS |
| Pass/fail | relief stable under camera motion; no depth inversion |
| Perf testable | yes — sample count scales cost |

### 2.5 Screen-Space Displacement Mapping (SSDM)
| Field | Value |
|---|---|
| Variant | `ssdm_enabled` |
| Status | **working** |
| Activation | `RenderConfig.ssdm_enabled`; F2 "Screen-Space Displacement" |
| Python | [pipeline.py:1481](../../engine/gfx/pipeline.py:1481), [ssdm_pass.py](../../engine/gfx/passes/ssdm_pass.py) |
| Shaders | `ssdm.frag` |
| Inputs | G-Buffer depth/normals, displacement array |
| Outputs | rewritten RT3 depth + normals |
| Temporal | none |
| Depends on | G-Buffer, Hi-Z not required |
| Excludes | none |
| Quality params | `ssdm_scale`, `ssdm_max_distance` (30), `disp_mid_radius` |
| Intended result | mid-distance displacement without tessellation cost |
| Test scene | S3 |
| Camera | mid distance (30–300 m) |
| Expected | relief visible beyond tessellation radius |
| Known artifacts | silhouette mismatch (screen-space cannot displace edges) |
| Debug view | NORMALS |
| Pass/fail | relief appears at mid distance; silhouettes unchanged |
| Perf testable | yes |

### 2.6 Material atlas / texture arrays
| Field | Value |
|---|---|
| Variant | 4 arrays (albedo, normal, ORM, height) |
| Status | **working** |
| Activation | always; `TextureArrayAtlas(ctx, 2048, 2048, 32)` |
| Python | [texture_atlas.py](../../engine/gfx/texture_atlas.py), [pipeline.py:457](../../engine/gfx/pipeline.py:457) |
| Shaders | `gbuffer.frag`, `gbuffer_tess.*`, `csm_tess.*` |
| Inputs | cooked `.pm` textures |
| Outputs | layer-indexed texture arrays |
| Temporal | none |
| Depends on | asset cooker |
| Excludes | none |
| Quality params | atlas size, `max_layers` (32) |
| Intended result | many materials in one draw |
| Test scene | S3 |
| Camera | any |
| Expected | correct material per surface |
| Known artifacts | layer overflow beyond 32 |
| Debug view | ALBEDO |
| Pass/fail | each material shows its own albedo/normal |
| Perf testable | no |

---

## 3. Direct Lighting

### 3.1 Directional sun (CSM-shadowed)
| Field | Value |
|---|---|
| Variant | single directional light |
| Status | **working** |
| Activation | always; `sun_dir`/`sun_lux` from environment provider |
| Python | [pipeline.py:1154](../../engine/gfx/pipeline.py:1154), [direct_radiance_pass.py](../../engine/gfx/passes/direct_radiance_pass.py) |
| Shaders | `direct_radiance.frag`, `deferred_resolve.frag` |
| Inputs | sun direction, lux, color, CSM atlas |
| Outputs | direct diffuse + specular |
| Temporal | none |
| Depends on | CSM |
| Excludes | none |
| Quality params | `sun_intensity` (4.0), `ambient_factor` (0.04) |
| Intended result | sunlit scene with shadows |
| Test scene | S2, S6 |
| Camera | any |
| Expected | consistent sun direction across scene |
| Known artifacts | shadow acne if bias wrong |
| Debug view | SHADOW_MASK (6), DIRECT_RADIANCE (23) |
| Pass/fail | shadows align with sun; no acne |
| Perf testable | yes |

### 3.2 Clustered point lights
| Field | Value |
|---|---|
| Variant | up to 256 |
| Status | **working** |
| Activation | `clustered_lights_enabled`; `pipeline.add_point_light(...)` |
| Python | [clustered_lights.py](../../engine/gfx/passes/clustered_lights.py) |
| Shaders | `deferred_resolve.frag` (SSBO 3) |
| Inputs | point light list |
| Outputs | SSBO 3 froxel clusters |
| Temporal | none |
| Depends on | none |
| Excludes | none |
| Quality params | `max_point_lights` (256) |
| Intended result | many local lights, branch-free |
| Test scene | S1 (indoor), S7 |
| Camera | any |
| Expected | inverse-square falloff, no popping |
| Known artifacts | cluster boundary seams if froxel too coarse |
| Debug view | SPOT_LIGHTS (12) shows spot only; point lights have no dedicated view |
| Pass/fail | lights illuminate correct radius |
| Perf testable | yes |

### 3.3 Clustered spot lights
| Field | Value |
|---|---|
| Variant | up to 64 |
| Status | **working** |
| Activation | `spot_lights_enabled`; `pipeline.add_spot_light(...)` |
| Python | [clustered_lights.py](../../engine/gfx/passes/clustered_lights.py) |
| Shaders | `deferred_resolve.frag` |
| Inputs | spot list (position, direction, radius, inner/outer cone) |
| Outputs | SSBO 3 |
| Temporal | none |
| Depends on | none |
| Excludes | none |
| Quality params | `max_spot_lights` (64), `spot_light_radius` (35), `spot_cone_angle` (32°) |
| Intended result | cone falloff |
| Test scene | S7 |
| Camera | any |
| Expected | smooth cone edge |
| Known artifacts | hard cone edge if inner==outer |
| Debug view | SPOT_LIGHTS (12) |
| Pass/fail | cone shape correct |
| Perf testable | yes |
| **Note** | `spot_light_radius`/`spot_cone_angle` are **defaults for `add_spot_light`**, not global overrides — per-light values win. |

### 3.4 Spot light shadows
| Field | Value |
|---|---|
| Variant | `spot_shadows_enabled` |
| Status | **working** (functionally) / **leaks on destroy** |
| Activation | `RenderConfig.spot_shadows_enabled` (default True) + per-light `cast_shadow=True` |
| Python | [pipeline.py:1350](../../engine/gfx/pipeline.py:1350), [spot_shadows.py](../../engine/gfx/spot_shadows.py) |
| Shaders | `spot_depth.vert`, `deferred_resolve.frag` |
| Inputs | spot casters |
| Outputs | spot shadow atlas |
| Temporal | none |
| Depends on | `SpotLightShadowMap` |
| Excludes | none |
| Quality params | `spot_shadow_resolution` (2048), `spot_shadow_bias` |
| Intended result | spot lights cast shadows |
| Test scene | S7 |
| Camera | any |
| Expected | shadow from spot casters |
| Known artifacts | peter-panning if bias high |
| Debug view | none dedicated |
| Pass/fail | spot shadow visible and attached |
| Perf testable | yes |
| **Risk** | `SpotLightShadowMap.destroy()` is never called by `pipeline.destroy()` (Phase 1 §3). |

---

## 4. Image-Based Lighting

### 4.1 Split-sum GGX IBL
| Field | Value |
|---|---|
| Variant | procedural env map + BRDF LUT |
| Status | **working** |
| Activation | `ibl_enabled` (default True); always executes |
| Python | [ibl.py](../../engine/gfx/passes/ibl.py), [pipeline.py:1533](../../engine/gfx/pipeline.py:1533) |
| Shaders | inline sky shader in `ibl.py`, `brdf_lut.frag`, `ibl_prefilter.comp`, `ibl_irradiance.comp` |
| Inputs | sun direction (regenerates env map when sun moves > ~0.014 rad) |
| Outputs | `brdf_lut`, `env_map`, `irradiance_map` |
| Temporal | none (regenerated on sun change) |
| Depends on | none |
| Excludes | none |
| Quality params | on/off |
| Intended result | ambient specular + diffuse from sky |
| Test scene | S5 (reflective), S6 (day/night) |
| Camera | any |
| Expected | reflections match sky; no black ambient |
| Known artifacts | env map lag when sun moves fast (threshold-gated) |
| Debug view | none dedicated |
| Pass/fail | metal reflects sky color; ambient not black |
| Perf testable | yes (regeneration cost on sun change) |

### 4.2 `gi_mode == "IBL"` as a GI selection
| Field | Value |
|---|---|
| Variant | `gi_mode="IBL"` |
| Status | **partially working / misleading** |
| Activation | `RenderConfig.gi_mode = "IBL"`; F2 GI Mode combo |
| Python | [pipeline.py:1518-1530](../../engine/gfx/pipeline.py:1518) |
| Shaders | `deferred_resolve.frag` (`u_GIEnabled`) |
| Inputs | n/a |
| Outputs | n/a |
| Temporal | n/a |
| Depends on | n/a |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | (documented) IBL-only GI |
| Test scene | S5 |
| Camera | any |
| Expected | IBL ambient only, no SSGI/RC |
| Known artifacts | **`"IBL"` is not a distinct branch.** `render_frame` only branches on `"RADIANCE_CASCADES"` and `"SSGI"`; everything else (including `"IBL"` and `"OFF"`) falls to the `else` that invalidates both GI histories. `u_GIEnabled` is set to 0 for both. IBL itself is controlled independently by `ibl_enabled`. So `gi_mode="IBL"` ≡ `gi_mode="OFF"` for GI purposes. |
| Debug view | GI_TOTAL (10) should be black |
| Pass/fail | confirm `IBL` and `OFF` produce identical GI (expected: yes) |
| Perf testable | no |

---

## 5. Shadows

### 5.1 Cascaded Shadow Maps (CSM)
| Field | Value |
|---|---|
| Variant | 1–4 cascades, PSSM splits |
| Status | **working** |
| Activation | always |
| Python | [shadow_csm.py](../../engine/gfx/shadow_csm.py), [pipeline.py:1310](../../engine/gfx/pipeline.py:1310) |
| Shaders | `csm_depth.vert/.frag`, `common/csm_sampling.glsl` |
| Inputs | per-cascade frustum, casters |
| Outputs | shadow atlas + per-cascade matrices |
| Temporal | none |
| Depends on | `FrustumCuller` (per-cascade) |
| Excludes | none |
| Quality params | `shadow_resolution` (8192), `csm_cascades` (4), `shadow_distance` (500), `csm_split_lambda` (0.85), `csm_cascade_distances` |
| Intended result | stable shadows across large depth range |
| Test scene | S6 (large depth range) |
| Camera | walk from near to far |
| Expected | no cascade popping; shadows stable |
| Known artifacts | cascade seam if splits mis-tuned |
| Debug view | CASCADE_INDEX (14), CASCADE_COVERAGE (15), CASCADE_BOUNDS (16) |
| Pass/fail | no visible cascade transition; shadows attached |
| Perf testable | yes |

### 5.2 Hard shadow filter
| Field | Value |
|---|---|
| Variant | `shadow_filter_mode="HARD"` |
| Status | **working** |
| Activation | `RenderConfig.shadow_filter_mode` |
| Python | [pipeline.py:1602](../../engine/gfx/pipeline.py:1602) |
| Shaders | `common/csm_sampling.glsl` (`SampleHardShadow`) |
| Inputs | shadow atlas |
| Outputs | binary shadow |
| Temporal | none |
| Depends on | 5.1 |
| Excludes | mutually exclusive with PCF/PCSS per cascade |
| Quality params | `shadow_bias`, `shadow_normal_bias` |
| Intended result | crisp shadow edges |
| Test scene | S6 |
| Camera | any |
| Expected | hard-edged shadows |
| Known artifacts | aliasing on edges |
| Debug view | SHADOW_MASK (6) |
| Pass/fail | edges are binary, no gradient |
| Perf testable | yes (cheapest) |

### 5.3 PCF shadow filter
| Field | Value |
|---|---|
| Variant | `shadow_filter_mode="PCF"` (default) |
| Status | **working** |
| Activation | `RenderConfig.shadow_filter_mode` |
| Python | [pipeline.py:1605-1611](../../engine/gfx/pipeline.py:1605) |
| Shaders | `common/csm_sampling.glsl` (Vogel disk) |
| Inputs | shadow atlas |
| Outputs | filtered shadow |
| Temporal | none |
| Depends on | 5.1 |
| Excludes | per-cascade exclusivity with HARD/PCSS |
| Quality params | `pcf_cascade_limit` (3), `pcf_samples` (16), `pcf_radius` (1.5), `pcf_falloff` (2.0) |
| Intended result | soft shadow edges |
| Test scene | S6 |
| Camera | any |
| Expected | smooth penumbra |
| Known artifacts | banding at low sample counts |
| Debug view | SHADOW_MASK |
| Pass/fail | penumbra smooth, no banding at 16 samples |
| Perf testable | yes |

### 5.4 PCSS shadow filter
| Field | Value |
|---|---|
| Variant | `shadow_filter_mode="PCSS"` |
| Status | **working** |
| Activation | `RenderConfig.shadow_filter_mode` |
| Python | [pipeline.py:1613-1627](../../engine/gfx/pipeline.py:1613) |
| Shaders | `common/csm_sampling.glsl` (blocker search + penumbra) |
| Inputs | shadow atlas |
| Outputs | contact-hardening shadow |
| Temporal | none |
| Depends on | 5.1 |
| Excludes | per-cascade exclusivity |
| Quality params | `pcss_cascade_limit` (1), `pcss_search_samples` (16), `pcss_search_radius` (48), `pcss_light_size` (0.10), `pcss_samples` (24), `pcss_falloff`, `pcss_max_radius` (96), `pcss_closest_blocker_weight` |
| Intended result | penumbra widens with distance from blocker |
| Test scene | S6 |
| Camera | any |
| Expected | sharp contact, soft distant shadow |
| Known artifacts | light leaking at thin blockers |
| Debug view | SHADOW_MASK |
| Pass/fail | penumbra width varies with occluder distance |
| Perf testable | yes (most expensive) |

### 5.5 Cascade stabilization
| Field | Value |
|---|---|
| Variant | texel-snapped cascade matrices |
| Status | **working** |
| Activation | always |
| Python | [shadow_csm.py](../../engine/gfx/shadow_csm.py) |
| Shaders | n/a |
| Inputs | camera frustum |
| Outputs | stabilized matrices |
| Temporal | none |
| Depends on | 5.1 |
| Excludes | none |
| Quality params | none |
| Intended result | no shadow shimmer when camera moves |
| Test scene | S6 |
| Camera | slow pan |
| Expected | shadows do not crawl |
| Known artifacts | none expected |
| Debug view | CASCADE_INDEX |
| Pass/fail | static scene + moving camera → no shadow crawl |
| Perf testable | no |
| Tests | `test_csm_stabilization.py` |

### 5.6 Legacy shadow machinery (removed)
| Field | Value |
|---|---|
| Variant | `u_ShadowMode`, `u_ShadowSoftness`, `u_ShadowPenumbraScale`, `u_ShadowOffset`, `POISSON_DISK` |
| Status | **legacy (removed)** |
| Activation | none |
| Python | n/a |
| Shaders | removed from `deferred_resolve.frag` |
| Inputs / Outputs | n/a |
| Temporal | n/a |
| Depends on | n/a |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | n/a |
| Test scene | n/a |
| Camera | n/a |
| Expected | absent |
| Known artifacts | **Phase 2.5 correction**: `u_SunAngularRadius` is **not** dead — `csm_sampling.glsl:278` uses it to derive a physical penumbra from the sun's solid angle. The stale test that asserted its removal was corrected. |
| Debug view | n/a |
| Pass/fail | n/a |
| Perf testable | n/a |

---

## 6. Ambient Occlusion

### 6.1 SSAO
| Field | Value |
|---|---|
| Variant | `ao_mode="SSAO"` |
| Status | **working** |
| Activation | `RenderConfig.ao_mode`; F2 AO Mode |
| Python | [ao_pass.py:152](../../engine/gfx/passes/ao_pass.py:152) |
| Shaders | `ssao.frag` |
| Inputs | G-Buffer depth/normals |
| Outputs | AO mask |
| Temporal | none |
| Depends on | G-Buffer |
| Excludes | mutually exclusive with GTAO |
| Quality params | `ao_radius` (0.35), `ao_intensity` (1.0), `ao_resolution_scale` |
| Intended result | hemisphere-sampled occlusion |
| Test scene | S1 (indoor), S8 (contact) |
| Camera | any |
| Expected | darkening in creases |
| Known artifacts | noise at low sample counts; halo at depth discontinuities |
| Debug view | AO (7) |
| Pass/fail | creases darken; no halo on silhouettes |
| Perf testable | yes |

### 6.2 GTAO (legacy backend)
| Field | Value |
|---|---|
| Variant | `ao_mode="GTAO"`, `ao_backend="LEGACY"` (default) |
| Status | **working** |
| Activation | `RenderConfig.ao_mode="GTAO"` |
| Python | [ao_pass.py:160](../../engine/gfx/passes/ao_pass.py:160) |
| Shaders | `gtao.frag` |
| Inputs | G-Buffer depth/normals |
| Outputs | AO mask |
| Temporal | none |
| Depends on | G-Buffer |
| Excludes | mutually exclusive with SSAO |
| Quality params | `ao_radius`, `ao_intensity`, `ao_resolution_scale` |
| Intended result | ground-truth horizon AO |
| Test scene | S1, S8 |
| Camera | any |
| Expected | cleaner than SSAO, contact darkening |
| Known artifacts | horizon search banding |
| Debug view | AO (7) |
| Pass/fail | AO matches expected contact darkening |
| Perf testable | yes |

### 6.3 GTAO Hi-Z bounds backend
| Field | Value |
|---|---|
| Variant | `ao_mode="GTAO"`, `ao_backend="HIZ_BOUNDS"` |
| Status | **working** |
| Activation | `RenderConfig.ao_backend`; F2 AO Backend combo |
| Python | [ao_pass.py:146](../../engine/gfx/passes/ao_pass.py:146) |
| Shaders | `gtao_hiz.frag`, `common/hiz_bounds.glsl` |
| Inputs | G-Buffer + `hiz_bounds_texture` |
| Outputs | AO mask |
| Temporal | none |
| Depends on | Hi-Z bounds hierarchy |
| Excludes | falls back to LEGACY if `hiz_bounds_texture` is None |
| Quality params | `ao_radius`, `ao_intensity`, `ao_debug_view` |
| Intended result | faster horizon search via bounds hierarchy |
| Test scene | S1, S8 |
| Camera | any |
| Expected | visually close to LEGACY GTAO |
| Known artifacts | differences at thin geometry |
| Debug view | AO (7) |
| Pass/fail | A/B against LEGACY shows no gross difference |
| Perf testable | yes (compare GPU ms) |

### 6.4 AO debug views
| Field | Value |
|---|---|
| Variant | `ao_debug_view` 0–4 |
| Status | **unreachable** |
| Activation | `RenderConfig.ao_debug_view` only |
| Python | [ao_pass.py:178](../../engine/gfx/passes/ao_pass.py:178) |
| Shaders | `gtao_hiz.frag` (`u_AODebugView`) |
| Inputs | n/a |
| Outputs | debug AO visualization |
| Temporal | n/a |
| Depends on | 6.3 |
| Excludes | n/a |
| Quality params | 0=final, 1=mip, 2=occupancy, 3=rejected, 4=refined |
| Intended result | AO internals inspection |
| Test scene | S1 |
| Camera | any |
| Expected | debug visualization |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | n/a |
| Perf testable | no |
| **Note** | Field exists in `EngineTweaks` and is applied to the pipeline, but **no widget in `DebugMenu` sets it** (0 hits). Reachable only by editing config in code. |

---

## 7. Global Illumination

### 7.1 SSGI — legacy backend
| Field | Value |
|---|---|
| Variant | `gi_mode="SSGI"`, `ssgi_backend="LEGACY"` (default) |
| Status | **working** |
| Activation | `RenderConfig.gi_mode`; F2 GI Mode + SSGI Backend |
| Python | [ssgi_pass.py:252](../../engine/gfx/passes/ssgi_pass.py:252) |
| Shaders | `ssgi.frag`, `ssgi_blur.frag`, `ssgi_temporal.frag` |
| Inputs | G-Buffer, `direct_radiance_texture`, Hi-Z (optional) |
| Outputs | `ssgi_resolved` |
| Temporal | ping-pong history (`ssgi_temporal_enabled`, blend 0.90) |
| Depends on | direct radiance pass |
| Excludes | mutually exclusive with Radiance Cascades |
| Quality params | `ssgi_steps` (16), `ssgi_rays` (8), `ssgi_ray_distance` (3.0), `ssgi_thickness` (0.35), `ssgi_intensity` (1.5), `ssgi_resolution_scale`, `ssgi_temporal_blend` |
| Intended result | near-field indirect diffuse + color bleed |
| Test scene | S1 (indoor color bleed), S8 |
| Camera | any |
| Expected | colored bounce light in corners |
| Known artifacts | temporal ghosting on disocclusion; noise |
| Debug view | SSGI (8), GI_TOTAL (10) |
| Pass/fail | color bleed visible; no ghosting after camera stop |
| Perf testable | yes |

### 7.2 SSGI — Hi-Z bounds backend
| Field | Value |
|---|---|
| Variant | `ssgi_backend="HIZ_BOUNDS"` |
| Status | **working** |
| Activation | F2 SSGI Backend combo |
| Python | [ssgi_pass.py:233](../../engine/gfx/passes/ssgi_pass.py:233) |
| Shaders | `ssgi_trace.frag`, `common/hiz_trace.glsl` |
| Inputs | G-Buffer, `hiz_bounds_texture` |
| Outputs | `ssgi_resolved` |
| Temporal | same as 7.1 |
| Depends on | Hi-Z bounds |
| Excludes | falls back to LEGACY if bounds texture missing |
| Quality params | `ssgi_hiz_max_iterations` (64), `ssgi_surface_reject` |
| Intended result | faster, more accurate tracing |
| Test scene | S1, S8 |
| Camera | any |
| Expected | close to LEGACY, fewer misses |
| Known artifacts | different noise character |
| Debug view | SSGI (8) |
| Pass/fail | A/B vs LEGACY: no gross difference, fewer artifacts |
| Perf testable | yes |

### 7.3 SSGI debug views
| Field | Value |
|---|---|
| Variant | `ssgi_debug_view` 0–8 |
| Status | **unreachable** |
| Activation | `RenderConfig.ssgi_debug_view` only |
| Python | [ssgi_pass.py:246](../../engine/gfx/passes/ssgi_pass.py:246) |
| Shaders | `ssgi_trace.frag` (`u_SSGIDebugView`) |
| Inputs | n/a |
| Outputs | debug visualization |
| Temporal | n/a |
| Depends on | 7.2 |
| Excludes | n/a |
| Quality params | 0=off, 1=raw, 2=hit UV, 3=hit dist, 4=hit ratio, 5=iters, 6=miss, 7=self, 8=surface |
| Intended result | SSGI internals inspection |
| Test scene | S1 |
| Camera | any |
| Expected | debug visualization |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | n/a |
| Perf testable | no |
| **Note** | In `EngineTweaks`, **not in `DebugMenu`** (0 hits). Config-only. |

### 7.4 Radiance Cascades — legacy backend
| Field | Value |
|---|---|
| Variant | `gi_mode="RADIANCE_CASCADES"`, `rc_backend="LEGACY"` (default) |
| Status | **working** |
| Activation | `RenderConfig.gi_mode`; F2 GI Mode |
| Python | [radiance_cascades_pass.py:239](../../engine/gfx/passes/radiance_cascades_pass.py:239) |
| Shaders | `rc_raymarch.comp`, `rc_merge.comp` |
| Inputs | G-Buffer, direct radiance |
| Outputs | `rc_texture` |
| Temporal | `rc_merge.comp` temporal blend (`rc_temporal_blend` 0.96) |
| Depends on | direct radiance pass |
| Excludes | mutually exclusive with SSGI |
| Quality params | `rc_ray_count` (16), `rc_thickness` (0.15), `rc_intensity` (1.0), `rc_resolution_scale`, `rc_temporal_blend` |
| Intended result | per-pixel raymarched indirect diffuse |
| Test scene | S1, S8 |
| Camera | any |
| Expected | indirect diffuse, no light leaks |
| Known artifacts | noise; temporal smear |
| Debug view | RADIANCE_CASCADES (13), GI_TOTAL (10) |
| Pass/fail | indirect light present; no leaks through walls |
| Perf testable | yes |

### 7.5 Radiance Cascades — explicit cascades backend
| Field | Value |
|---|---|
| Variant | `rc_backend="CASCADES"` |
| Status | **working** |
| Activation | F2 RC Backend combo |
| Python | [radiance_cascades_pass.py](../../engine/gfx/passes/radiance_cascades_pass.py), [rc_cascade.py](../../engine/gfx/passes/rc_cascade.py) |
| Shaders | `rc_trace_intervals.comp`, `rc_merge_cascades.comp`, `rc_reconstruct.comp`, `rc_spatial_filter.comp`, `rc_temporal.comp`, `rc_angular_filter.comp` |
| Inputs | G-Buffer, Hi-Z bounds |
| Outputs | incident irradiance → resolved |
| Temporal | `rc_temporal.comp` (`rc_temporal_enabled`, default **False**) |
| Depends on | Hi-Z bounds, `build_cascades` |
| Excludes | mutually exclusive with SSGI |
| Quality params | `rc_merge_max_cascade` (2), `rc_angular_filter_enabled` (True), `rc_hiz_max_iterations` (64) |
| Intended result | explicit probe/direction cascades, merged far→near |
| Test scene | S1, S8 |
| Camera | any |
| Expected | smooth indirect irradiance, contact detail |
| Known artifacts | probe-grid banding; angular filter cost |
| Debug view | RADIANCE_CASCADES (13) |
| Pass/fail | irradiance smooth; contact darkening correct |
| Perf testable | yes |
| **Note** | `rc_cascades` config field is **not consumed** — cascade intervals are hardcoded `(0–6, 6–20, 20–40)` m with spacings `(4, 8, 16)` px and directions `(64, 64, 256)`. **Phase 2.5: the dead field was removed.** |

### 7.6 RC debug views
| Field | Value |
|---|---|
| Variant | `rc_debug_view` 0–3, `rc_temporal_debug_view` 0–4 |
| Status | **unreachable** |
| Activation | `RenderConfig.rc_debug_view` / `rc_temporal_debug_view` only |
| Python | [radiance_cascades_pass.py](../../engine/gfx/passes/radiance_cascades_pass.py) |
| Shaders | `rc_reconstruct.comp`, `rc_temporal.comp` |
| Inputs | n/a |
| Outputs | debug visualization |
| Temporal | n/a |
| Depends on | 7.5 |
| Excludes | n/a |
| Quality params | see `RenderConfig` |
| Intended result | RC internals inspection |
| Test scene | S1 |
| Camera | any |
| Expected | debug visualization |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | n/a |
| Perf testable | no |
| **Note** | In `EngineTweaks`, **not in `DebugMenu`** (0 hits). Config-only. |

---

## 8. Reflections

### 8.1 SSR — legacy backend
| Field | Value |
|---|---|
| Variant | `ssr_backend="LEGACY"` (default) |
| Status | **working** |
| Activation | `ssr_enabled` (default True) |
| Python | [ssr_pass.py:186](../../engine/gfx/passes/ssr_pass.py:186) |
| Shaders | `ssr.frag`, `ssr_blur.frag` |
| Inputs | G-Buffer, scene color, Hi-Z pyramid |
| Outputs | `ssr_texture` |
| Temporal | none |
| Depends on | Hi-Z pyramid (falls back to depth, mip=1) |
| Excludes | none |
| Quality params | `ssr_steps` (48), `ssr_max_roughness` (0.65), `ssr_thickness` (0.12), `ssr_max_distance` (30), `ssr_resolution_scale`, `ssr_mode` |
| Intended result | screen-space specular reflections |
| Test scene | S5 (reflective floor) |
| Camera | any |
| Expected | reflections of on-screen geometry |
| Known artifacts | reflections vanish at screen edge; thickness artifacts |
| Debug view | none dedicated |
| Pass/fail | reflection tracks geometry; no edge smear |
| Perf testable | yes |

### 8.2 SSR — Hi-Z bounds backend
| Field | Value |
|---|---|
| Variant | `ssr_backend="HIZ_BOUNDS"` |
| Status | **working** |
| Activation | F2 SSR Backend combo |
| Python | [ssr_pass.py:161](../../engine/gfx/passes/ssr_pass.py:161) |
| Shaders | `ssr_hiz.frag`, `common/hiz_trace.glsl` |
| Inputs | G-Buffer, `hiz_bounds_texture` |
| Outputs | `ssr_texture` |
| Temporal | none |
| Depends on | Hi-Z bounds |
| Excludes | falls back to LEGACY if bounds missing |
| Quality params | `ssr_hiz_max_iterations` (96) |
| Intended result | faster, more accurate tracing |
| Test scene | S5 |
| Camera | any |
| Expected | close to LEGACY, fewer misses |
| Known artifacts | different noise character |
| Debug view | none dedicated |
| Pass/fail | A/B vs LEGACY: no gross difference |
| Perf testable | yes |

### 8.3 SSR march mode (`ssr_mode`)
| Field | Value |
|---|---|
| Variant | `ssr_mode="HIZ"` (default) / `"LINEAR"` |
| Status | **working** |
| Activation | `RenderConfig.ssr_mode`; F2 "SSR March Mode" |
| Python | [ssr_pass.py:211](../../engine/gfx/passes/ssr_pass.py:211) |
| Shaders | `ssr.frag` (`u_SSR_Mode`) |
| Inputs | Hi-Z pyramid |
| Outputs | reflection |
| Temporal | none |
| Depends on | 8.1 |
| Excludes | n/a |
| Quality params | 0=linear, 1=Hi-Z |
| Intended result | A/B march strategy |
| Test scene | S5 |
| Camera | any |
| Expected | Hi-Z faster, similar result |
| Known artifacts | linear march slower |
| Debug view | none |
| Pass/fail | A/B produces similar image; Hi-Z measurably faster |
| Perf testable | yes |
| **Phase 2.5** | Phase 2 wrongly classified this as dead. It is consumed by `ssr_pass.py:211` and is now exposed in F2. |

### 8.4 SSR debug views
| Field | Value |
|---|---|
| Variant | `ssr_debug_view` 0–5 |
| Status | **working** |
| Activation | `RenderConfig.ssr_debug_view`; F2 (HIZ_BOUNDS backend) |
| Python | [ssr_pass.py:232](../../engine/gfx/passes/ssr_pass.py:232) |
| Shaders | `ssr_hiz.frag` (`u_SSRDebugView`) |
| Inputs | n/a |
| Outputs | debug visualization |
| Temporal | n/a |
| Depends on | 8.2 |
| Excludes | n/a |
| Quality params | 0=off, 1=hit UV, 2=distance, 3=iterations, 4=miss reason, 5=final mip |
| Intended result | SSR internals inspection |
| Test scene | S5 |
| Camera | any |
| Expected | debug visualization |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | each mode shows a distinct image |
| Perf testable | no |
| **Phase 2.5** | Phase 2 wrongly classified this as dead. It is consumed by `ssr_pass.py:232` and is now exposed in F2. |

---

## 9. Sky and Atmosphere

### 9.1 Hillaire atmospheric scattering
| Field | Value |
|---|---|
| Variant | `sky_mode="ATMOSPHERE"` |
| Status | **working** |
| Activation | `ProjectConfig.atmosphere_enabled` (default True) |
| Python | [modules/atmosphere/](../../engine/modules/atmosphere/), [environment.py](../../engine/gfx/environment.py) |
| Shaders | `atmosphere.glsl`, `hillaire_common.glsl`, `hillaire_sky_view.frag`, `hillaire_transmittance.frag`, `hillaire_multiscattering.frag` |
| Inputs | sun direction, Rayleigh/Mie/ozone coefficients, time of day |
| Outputs | sky-view LUT, transmittance LUT, multiscattering LUT |
| Temporal | LUTs rebaked when sun moves |
| Depends on | `EnvironmentProvider` contract |
| Excludes | replaced by `FlatEnvironment` when disabled |
| Quality params | `rayleigh_beta`, `mie_beta`, `mie_asymmetry`, `ozone_beta`, `turbidity`, `sky_exposure`, `sky_saturation` |
| Intended result | physically based sky with correct horizon |
| Test scene | S6 (day/night) |
| Camera | look at horizon and zenith |
| Expected | blue zenith, warm horizon, correct sunset |
| Known artifacts | banding in LUT at low resolution |
| Debug view | none dedicated |
| Pass/fail | sky gradient physically plausible at 3 times of day |
| Perf testable | yes (LUT rebake cost) |

### 9.2 Space sky mode
| Field | Value |
|---|---|
| Variant | `sky_mode="SPACE"` |
| Status | **working** |
| Activation | `AtmosphereConfig.sky_mode`; preset `SPACE` |
| Python | [modules/atmosphere/config.py](../../engine/modules/atmosphere/config.py) |
| Shaders | `atmosphere.glsl:33` (`EvaluateSpaceSky`) |
| Inputs | star density, nebula color, sun/moon |
| Outputs | starfield + nebula |
| Temporal | star twinkle uses `time` |
| Depends on | 9.1 |
| Excludes | mutually exclusive with ATMOSPHERE mode |
| Quality params | `star_intensity`, `star_density`, `nebula_color`, `nebula_intensity`, `space_zenith` |
| Intended result | space backdrop |
| Test scene | S6 (SPACE preset) |
| Camera | any |
| Expected | stars, no atmosphere gradient |
| Known artifacts | star aliasing |
| Debug view | none |
| Pass/fail | stars visible, no blue sky |
| Perf testable | no |

### 9.3 Cloud deck
| Field | Value |
|---|---|
| Variant | `cloud_enabled` |
| Status | **working** |
| Activation | `AtmosphereConfig.cloud_enabled` (default True) |
| Python | [sky_cloud_pass.py](../../engine/gfx/passes/sky_cloud_pass.py) |
| Shaders | `sky_cloud.frag`, `sky_cloud_noise.frag`, `common/volumetrics.glsl` |
| Inputs | depth, inverse VP, sun |
| Outputs | cloud composite over sky |
| Temporal | `time_elapsed` for wind |
| Depends on | 9.1 |
| Excludes | none |
| Quality params | `cloud_coverage` (0.46), `cloud_opacity` (0.88), `cloud_altitude` (1800), `cloud_scale` (18000), `cloud_layer_separation`, `cloud_secondary_opacity`, `cloud_wind` |
| Intended result | procedural cloud layer |
| Test scene | S6 |
| Camera | look up |
| Expected | clouds drift with wind |
| Known artifacts | hard horizon cutoff |
| Debug view | none |
| Pass/fail | clouds visible, drift over time, occlude sun |
| Perf testable | yes |

### 9.4 Atmosphere presets
| Field | Value |
|---|---|
| Variant | 6 presets |
| Status | **working** |
| Activation | `atmosphere_settings={"preset": ...}`; F2 atmosphere preset |
| Python | [modules/atmosphere/config.py:131](../../engine/modules/atmosphere/config.py:131) |
| Shaders | as 9.1 |
| Inputs | preset name |
| Outputs | full `AtmosphereConfig` |
| Temporal | n/a |
| Depends on | 9.1 |
| Excludes | n/a |
| Quality params | `EARTH_DAY`, `EARTH_SUNSET`, `EARTH_NIGHT`, `ALIEN_CYAN_PURPLE`, `ALIEN_CRIMSON_MARS`, `SPACE` |
| Intended result | distinct lighting moods |
| Test scene | S6 |
| Camera | any |
| Expected | visibly different sky/light per preset |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | each preset visually distinct |
| Perf testable | no |

### 9.5 Flat environment fallback
| Field | Value |
|---|---|
| Variant | `FlatEnvironment` |
| Status | **working** |
| Activation | `atmosphere_enabled=False` |
| Python | [environment.py](../../engine/gfx/environment.py) |
| Shaders | `deferred_resolve.frag` (constant LUTs) |
| Inputs | constant sky color |
| Outputs | 1×1 LUTs |
| Temporal | none |
| Depends on | none |
| Excludes | replaces 9.1 |
| Quality params | `sky_color`, `sun_direction`, `sun_lux`, `ambient` |
| Intended result | neutral sky, no scattering |
| Test scene | S2 |
| Camera | any |
| Expected | flat sky color |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | sky is uniform color |
| Perf testable | yes (cheapest) |

---

## 10. Volumetrics and Fog

### 10.1 Froxel volumetric fog
| Field | Value |
|---|---|
| Variant | `volumetric_fog_enabled` |
| Status | **working** |
| Activation | `RenderConfig.volumetric_fog_enabled`; F2 "Froxel Volumetric Fog" |
| Python | [volumetric_fog_pass.py](../../engine/gfx/passes/volumetric_fog_pass.py) |
| Shaders | `volumetric_fog_inject.comp`, `volumetric_fog_integrate.comp`, `volumetric_fog_composite.frag`, `common/volumetrics.glsl` |
| Inputs | G-Buffer depth, CSM, point lights |
| Outputs | fog composite into HDR |
| Temporal | none |
| Depends on | CSM, clustered lights |
| Excludes | none |
| Quality params | `fog_density` (0.005), `fog_height_falloff` (0.10), `fog_anisotropy` (0.65), `fog_distance` (400), `fog_ambient` (0.35), `fog_point_lights` |
| Intended result | participating medium with light shafts |
| Test scene | S6 (sun through geometry), S7 |
| Camera | any |
| Expected | god rays, distance haze |
| Known artifacts | froxel banding at low resolution |
| Debug view | `fog_debug_mode` 0–2 (F2) |
| Pass/fail | haze increases with distance; shafts visible |
| Perf testable | yes |

### 10.2 Froxel resolution presets
| Field | Value |
|---|---|
| Variant | `fog_resolution` |
| Status | **working** |
| Activation | `RenderConfig.fog_resolution`; F2 |
| Python | [quality_presets.py](../../engine/gfx/quality_presets.py) `FROXEL_RESOLUTIONS` |
| Shaders | as 10.1 |
| Inputs | resolution key |
| Outputs | 3D froxel grid |
| Temporal | none |
| Depends on | 10.1 |
| Excludes | n/a |
| Quality params | `LOW` (80,45,32), `MEDIUM` (120,68,48), `HIGH` (160,90,64), `ULTRA` (200,112,80), `CINEMATIC` (240,135,96) |
| Intended result | quality/cost tradeoff |
| Test scene | S6 |
| Camera | any |
| Expected | higher res = less banding |
| Known artifacts | banding at LOW |
| Debug view | fog debug |
| Pass/fail | banding visibly reduced from LOW→CINEMATIC |
| Perf testable | yes |

### 10.3 Fog debug modes
| Field | Value |
|---|---|
| Variant | `fog_debug_mode` 0–2 |
| Status | **working** |
| Activation | F2 fog debug combo |
| Python | [volumetric_fog_pass.py:179](../../engine/gfx/passes/volumetric_fog_pass.py:179) |
| Shaders | `volumetric_fog_composite.frag` (`u_DebugMode`) |
| Inputs | n/a |
| Outputs | debug visualization |
| Temporal | n/a |
| Depends on | 10.1 |
| Excludes | n/a |
| Quality params | 0=normal, 1=inscattering only, 2=transmittance only |
| Intended result | fog internals inspection |
| Test scene | S6 |
| Camera | any |
| Expected | isolated fog terms |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | each mode isolates its term |
| Perf testable | no |

### 10.4 VolumetricSettings (CPU mirror)
| Field | Value |
|---|---|
| Variant | `VolumetricSettings` dataclass |
| Status | **stub (unused)** |
| Activation | none in runtime |
| Python | [volumetrics.py](../../engine/gfx/volumetrics.py) |
| Shaders | n/a |
| Inputs | `RenderConfig` |
| Outputs | CPU-side tuning object |
| Temporal | n/a |
| Depends on | n/a |
| Excludes | n/a |
| Quality params | mirrors fog fields |
| Intended result | shared tuning surface |
| Test scene | n/a |
| Camera | n/a |
| Expected | n/a |
| Known artifacts | **duplicate source of truth** — passes read `RenderConfig` directly |
| Debug view | n/a |
| Pass/fail | n/a |
| Perf testable | n/a |

---

## 11. Transparency

### 11.1 Particle blending
| Field | Value |
|---|---|
| Variant | additive (EMBERS) / alpha (DUST_MOTES, FIREFLIES) |
| Status | **working** |
| Activation | `particle_mode`; per-emitter |
| Python | [particle_pass.py:541](../../engine/gfx/passes/particle_pass.py:541) |
| Shaders | `particle_render.vert/.frag` |
| Inputs | particle SSBO |
| Outputs | blended HDR |
| Temporal | none |
| Depends on | compute simulation |
| Excludes | none |
| Quality params | `particle_size`, `particle_turbulence` |
| Intended result | soft additive/alpha particles |
| Test scene | S7 |
| Camera | any |
| Expected | correct blend per mode |
| Known artifacts | sorting (additive is order-independent; alpha is not) |
| Debug view | none |
| Pass/fail | additive glows; alpha particles do not hard-clip |
| Perf testable | yes |

### 11.2 Water refraction blend
| Field | Value |
|---|---|
| Variant | per-entity `WaterProperties` |
| Status | **working** |
| Activation | ECS entity with shape `"water"` + `set_entity_water_properties` |
| Python | [water_pass.py](../../engine/gfx/passes/water_pass.py) |
| Shaders | `water.vert`, `water.frag` |
| Inputs | opaque copy, depth, water props |
| Outputs | refracted/reflected water into HDR |
| Temporal | `time_elapsed` for waves |
| Depends on | opaque copy FBO |
| Excludes | none |
| Quality params | `wave_amplitude`, `wave_speed`, `wave_steepness`, `refraction_enabled`, `refraction_strength`, `foam_*`, `clarity`, `roughness`, `color_shallow`, `color_deep`, `extinction` |
| Intended result | refractive water surface |
| Test scene | S9 (water) |
| Camera | above and below waterline |
| Expected | refraction distortion, foam at edges |
| Known artifacts | refraction of off-screen geometry missing |
| Debug view | none |
| Pass/fail | refraction visible; foam at contact |
| Perf testable | yes |
| **Risk** | `WaterPass.destroy()` never called by `pipeline.destroy()` (Phase 1 §3). |

**Confirmed absent**: there is **no general alpha-blended transparent geometry pass**. Only
particles and water blend. Opaque geometry is always opaque.

---

## 12. Water

### 12.1 Water surface rendering
Covered by 11.2. Additional detail:

| Field | Value |
|---|---|
| Variant | per-entity |
| Status | **working** |
| Activation | ECS shape `"water"` |
| Python | [water_pass.py](../../engine/gfx/passes/water_pass.py), [core/water.py](../../engine/core/water.py) |
| Shaders | `water.vert`, `water.frag` |
| Inputs | `WaterProperties` per entity |
| Outputs | HDR water |
| Temporal | wave time |
| Depends on | opaque copy |
| Excludes | none |
| Quality params | see 11.2 |
| Intended result | animated refractive water |
| Test scene | S9 |
| Camera | above/below |
| Expected | waves animate; refraction follows |
| Known artifacts | below-water rendering is not a separate underwater pass |
| Debug view | none |
| Pass/fail | waves animate smoothly; no z-fighting with floor |
| Perf testable | yes |

### 12.2 Water lighting lab (example)
| Field | Value |
|---|---|
| Variant | `examples/03_water_lighting_lab.py` |
| Status | **working** |
| Activation | run the example |
| Python | [examples/03_water_lighting_lab.py](../../examples/03_water_lighting_lab.py) |
| Shaders | as 12.1 |
| Inputs | 5 chambers with skylights |
| Outputs | reference scene |
| Temporal | n/a |
| Depends on | 12.1 |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | water + colored GI reference |
| Test scene | itself |
| Camera | fly |
| Expected | colored bounce in chambers |
| Known artifacts | n/a |
| Debug view | n/a |
| Pass/fail | use as S9 baseline |
| Perf testable | yes |

---

## 13. Terrain

### 13.1 Grid terrain mesh
| Field | Value |
|---|---|
| Variant | `make_grid_terrain_pm_mesh` |
| Status | **working** |
| Activation | world builder |
| Python | [default_world.py:68](../../engine/world/default_world.py:68) |
| Shaders | `gbuffer.*` |
| Inputs | size, grid cells, uv tiles |
| Outputs | flat grid mesh ready for displacement |
| Temporal | none |
| Depends on | `PMMesh` |
| Excludes | none |
| Quality params | `size`, `grid_cells`, `uv_tiles` |
| Intended result | flat terrain base |
| Test scene | S6 |
| Camera | any |
| Expected | flat ground |
| Known artifacts | none |
| Debug view | ALBEDO |
| Pass/fail | ground renders |
| Perf testable | yes |

### 13.2 Heightmap terrain
| Field | Value |
|---|---|
| Variant | (none) |
| Status | **absent** |
| Activation | n/a |
| Python | n/a |
| Shaders | n/a |
| Inputs | n/a |
| Outputs | n/a |
| Temporal | n/a |
| Depends on | n/a |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | n/a |
| Test scene | n/a |
| Camera | n/a |
| Expected | n/a |
| Known artifacts | **No heightmap terrain system exists.** `make_grid_terrain_pm_mesh` produces a flat grid; displacement is per-material POM/tessellation, not terrain-scale. |
| Debug view | n/a |
| Pass/fail | n/a |
| Perf testable | n/a |

---

## 14. Particles

### 14.1 GPU compute particle simulation
| Field | Value |
|---|---|
| Variant | `particle_simulate.comp` |
| Status | **working** |
| Activation | `particles_enabled` + emitters |
| Python | [particle_pass.py](../../engine/gfx/passes/particle_pass.py) |
| Shaders | `particle_simulate.comp` |
| Inputs | emitter params, curl noise |
| Outputs | particle SSBO |
| Temporal | persistent particle state |
| Depends on | compute capability |
| Excludes | none |
| Quality params | `particle_turbulence` (0.85) |
| Intended result | animated particles |
| Test scene | S7 |
| Camera | any |
| Expected | particles drift with turbulence |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | particles animate |
| Perf testable | yes |

### 14.2 DUST_MOTES
| Field | Value |
|---|---|
| Variant | `particle_mode="DUST_MOTES"` |
| Status | **working** |
| Activation | `RenderConfig.particle_mode` |
| Python | [particle_pass.py:73](../../engine/gfx/passes/particle_pass.py:73) |
| Shaders | `particle_render.*` |
| Inputs | emitter |
| Outputs | alpha-blended dust |
| Temporal | persistent |
| Depends on | 14.1 |
| Excludes | n/a |
| Quality params | `particle_size` |
| Intended result | floating dust |
| Test scene | S7 |
| Camera | any |
| Expected | slow drifting motes |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | dust visible and drifting |
| Perf testable | yes |

### 14.3 EMBERS
| Field | Value |
|---|---|
| Variant | `particle_mode="EMBERS"` |
| Status | **working** |
| Activation | `RenderConfig.particle_mode` |
| Python | [particle_pass.py:73](../../engine/gfx/passes/particle_pass.py:73) |
| Shaders | `particle_render.*` |
| Inputs | emitter with thermal updraft |
| Outputs | additive embers |
| Temporal | persistent |
| Depends on | 14.1 |
| Excludes | n/a |
| Quality params | `particle_size` |
| Intended result | rising glowing embers |
| Test scene | S7 |
| Camera | any |
| Expected | embers rise and glow |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | embers rise, additive glow |
| Perf testable | yes |

### 14.4 FIREFLIES
| Field | Value |
|---|---|
| Variant | `particle_mode="FIREFLIES"` |
| Status | **working** |
| Activation | `RenderConfig.particle_mode` |
| Python | [particle_pass.py:73](../../engine/gfx/passes/particle_pass.py:73) |
| Shaders | `particle_render.*` |
| Inputs | emitter with neutral drift |
| Outputs | alpha-blended fireflies |
| Temporal | persistent |
| Depends on | 14.1 |
| Excludes | n/a |
| Quality params | `particle_size` |
| Intended result | drifting glowing points |
| Test scene | S7 |
| Camera | any |
| Expected | fireflies drift |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | fireflies visible |
| Perf testable | yes |

---

## 15. Anti-Aliasing

### 15.1 TAA
| Field | Value |
|---|---|
| Variant | `taa_enabled` / `aa_mode="TAA"` |
| Status | **working** |
| Activation | `RenderConfig.taa_enabled` or `aa_mode="TAA"` |
| Python | [taa_pass.py](../../engine/gfx/passes/taa_pass.py), [pipeline.py:1763](../../engine/gfx/pipeline.py:1763) |
| Shaders | `taa.frag` |
| Inputs | HDR, G-Buffer depth/normals, prev VP |
| Outputs | accumulated HDR |
| Temporal | ping-pong history (RGBA16F) |
| Depends on | jittered projection, motion vectors |
| Excludes | mutually exclusive with FXAA/SMAA |
| Quality params | `taa_feedback` (0.82), `taa_sharpness` (0.35), `taa_gamma` (1.25), `taa_max_feedback` (0.95), `taa_min_feedback` (0.78), `taa_variance_gamma` (1.25) |
| Intended result | stable edges, no shimmer |
| Test scene | S3 (thin geometry), S4 (moving) |
| Camera | static then pan |
| Expected | edges converge; no ghosting after stop |
| Known artifacts | ghosting on disocclusion; over-blur if feedback too high |
| Debug view | none |
| Pass/fail | static → no shimmer; pan → no ghost trail |
| Perf testable | yes |

### 15.2 FXAA
| Field | Value |
|---|---|
| Variant | `aa_mode="FXAA"` |
| Status | **working** |
| Activation | `RenderConfig.aa_mode`; F2 AA Mode |
| Python | [fxaa_pass.py](../../engine/gfx/passes/fxaa_pass.py) |
| Shaders | `fxaa.frag` |
| Inputs | LDR |
| Outputs | LDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | mutually exclusive with SMAA/TAA |
| Quality params | `fxaa_subpixel` (0.75), `fxaa_edge_threshold` (0.125) |
| Intended result | cheap edge smoothing |
| Test scene | S3 |
| Camera | any |
| Expected | softened edges |
| Known artifacts | text/detail blur |
| Debug view | none |
| Pass/fail | edges smoother, no gross blur |
| Perf testable | yes |

### 15.3 SMAA
| Field | Value |
|---|---|
| Variant | `aa_mode="SMAA"` |
| Status | **working** |
| Activation | `RenderConfig.aa_mode`; F2 AA Mode |
| Python | [smaa_pass.py](../../engine/gfx/passes/smaa_pass.py) |
| Shaders | `smaa_edge.frag`, `smaa_blend.frag`, `smaa_neighborhood.frag`, `smaa_resolve.frag` |
| Inputs | LDR |
| Outputs | LDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | mutually exclusive with FXAA/TAA |
| Quality params | `smaa_threshold` (0.08) |
| Intended result | morphological AA, sharper than FXAA |
| Test scene | S3 |
| Camera | any |
| Expected | smooth edges, preserved detail |
| Known artifacts | missed thin features |
| Debug view | none |
| Pass/fail | edges smooth, detail preserved |
| Perf testable | yes |

### 15.4 No AA
| Field | Value |
|---|---|
| Variant | `aa_mode="OFF"` |
| Status | **working** |
| Activation | `RenderConfig.aa_mode` |
| Python | [post_process.py](../../engine/gfx/post_process.py) |
| Shaders | `post_process.frag` |
| Inputs | HDR |
| Outputs | LDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | n/a |
| Quality params | none |
| Intended result | raw image |
| Test scene | S3 |
| Camera | any |
| Expected | aliased edges |
| Known artifacts | aliasing |
| Debug view | none |
| Pass/fail | baseline for AA comparison |
| Perf testable | yes |

---

## 16. Resolution Scaling

**Confirmed**: there is **no temporal upscaling** (no FSR/DLSS/TSR). Only per-effect internal
resolution scaling, which renders an effect at a fraction of full resolution and upsamples it
bilinearly.

| Variant | Field | Range | Status | Reachable |
|---|---|---|---|---|
| AO scale | `ao_resolution_scale` | 0.25–1.0 | working | F2 |
| SSGI scale | `ssgi_resolution_scale` | 0.25–1.0 | working | F2 |
| RC scale | `rc_resolution_scale` | 0.25–1.0 | working | F2 |
| SSR scale | `ssr_resolution_scale` | 0.25–1.0 | working | F2 |
| Lens flare scale | `lens_flare_resolution_scale` | 0.25–1.0 | working | F2 |

All five are wired through `pipeline.apply_config` → `pass.set_resolution_scale(...)` and verified
by grep. No full-frame render-scale (e.g. 0.75× then upscale) exists.

---

## 17. Tone Mapping

| Variant | `tonemap_mode` | Status | Reachable | Shader |
|---|---|---|---|---|
| ACES (Hill fit) | `"ACES"` (default) | working | F2 | `common/tonemap.glsl` `TonemapACESFitted` |
| AgX (Sobotka 2022) | `"AgX"` | working | F2 | `TonemapAgX` |
| Khronos Neutral | `"Neutral"` | working | F2 | `TonemapNeutral` |
| Reinhard (legacy) | `"Reinhard"` | working (legacy) | F2 | `TonemapReinhard` |

| Field | Value |
|---|---|
| Activation | `RenderConfig.tonemap_mode`; F2 cycle |
| Python | [post_process.py](../../engine/gfx/post_process.py) |
| Shaders | `post_process.frag` (`u_TonemapMode` 0–3) |
| Inputs | HDR |
| Outputs | LDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | n/a |
| Quality params | mode only |
| Intended result | distinct highlight roll-off |
| Test scene | S6 (bright sun) |
| Camera | any |
| Expected | ACES filmic, AgX desaturating highlights, Neutral preserving, Reinhard flat |
| Known artifacts | Reinhard clips highlights |
| Debug view | none |
| Pass/fail | four modes visibly distinct on a bright highlight |
| Perf testable | no |

---

## 18. Exposure

### 18.1 Manual exposure
| Field | Value |
|---|---|
| Variant | `auto_exposure_enabled=False` (default) |
| Status | **working** |
| Activation | default |
| Python | [post_process.py](../../engine/gfx/post_process.py) |
| Shaders | `post_process.frag` (`u_Exposure`) |
| Inputs | `exposure` (1.0), `exposure_compensation` (0.0) |
| Outputs | scaled HDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | mutually exclusive with auto |
| Quality params | `exposure`, `exposure_compensation` |
| Intended result | fixed exposure |
| Test scene | S6 |
| Camera | any |
| Expected | brightness constant |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | brightness stable |
| Perf testable | no |

### 18.2 Auto-exposure
| Field | Value |
|---|---|
| Variant | `auto_exposure_enabled=True` |
| Status | **working** |
| Activation | `RenderConfig.auto_exposure_enabled`; F2 |
| Python | [exposure_pass.py](../../engine/gfx/passes/exposure_pass.py) |
| Shaders | `exposure_histogram.comp`, `exposure_adapt.comp` |
| Inputs | HDR scene |
| Outputs | 1×1 EV texture |
| Temporal | adapted EV state |
| Depends on | resolved HDR |
| Excludes | mutually exclusive with manual |
| Quality params | `exposure_key` (0.5), `exposure_min_ev` (-8), `exposure_max_ev` (12), `exposure_adapt_speed_up` (1.5), `exposure_adapt_speed_down` (4.0), `exposure_min_log_luminance`, `exposure_max_log_luminance` |
| Intended result | eye adaptation |
| Test scene | S6 (day→night) |
| Camera | any |
| Expected | smooth adaptation, no pumping |
| Known artifacts | pumping if speeds too high |
| Debug view | none |
| Pass/fail | adapts smoothly; no oscillation |
| Perf testable | yes (histogram cost) |

---

## 19. Bloom

| Field | Value |
|---|---|
| Variant | `bloom_enabled` |
| Status | **working** |
| Activation | `RenderConfig.bloom_enabled` (default True); F2 |
| Python | [bloom_pass.py](../../engine/gfx/passes/bloom_pass.py), [pipeline.py:1779](../../engine/gfx/pipeline.py:1779) |
| Shaders | `bloom_downsample.frag`, `bloom_upsample.frag`, `common/blur.glsl` |
| Inputs | HDR |
| Outputs | bloom texture added to HDR |
| Temporal | none |
| Depends on | post-process |
| Excludes | none |
| Quality params | `bloom_iterations` (5), `bloom_threshold` (1.0), `bloom_intensity` (0.12) |
| Intended result | glow around bright areas |
| Test scene | S6 (sun), S7 (emissive) |
| Camera | any |
| Expected | bright areas bloom |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | bloom visible on highlights only |
| Perf testable | yes |

---

## 20. Color Grading

| Variant | Field | Status | Reachable |
|---|---|---|---|
| White balance | `white_balance_enabled`, `white_balance_k` | working | **config only** (0 hits in F2) |
| Contrast | `contrast` | working | F2 |
| Saturation | `saturation` | working | F2 |
| Lift/Gamma/Gain (CDL) | `lift`, `gamma`, `gain` | working | F2 |
| Chromatic aberration | `chromatic_aberration_enabled`, `_intensity` | working | F2 |
| Vignette | `vignette_enabled`, `_intensity`, `_roundness`, `_smoothness` | working | F2 |
| Film grain | `film_grain_enabled`, `_intensity` | working | F2 |

| Field | Value |
|---|---|
| Activation | `RenderConfig` fields |
| Python | [post_process.py](../../engine/gfx/post_process.py) |
| Shaders | `post_process.frag` |
| Inputs | LDR (after tonemap) |
| Outputs | graded LDR |
| Temporal | film grain uses frame index |
| Depends on | tonemap |
| Excludes | n/a |
| Quality params | see table |
| Intended result | cinematic grade |
| Test scene | S6 |
| Camera | any |
| Expected | each control has visible effect |
| Known artifacts | grain flicker if intensity high |
| Debug view | none |
| Pass/fail | each control visibly changes image |
| Perf testable | no |
| **Note** | `white_balance_enabled` is **not exposed in F2** (0 hits). Config-only. |

---

## 21. Depth of Field

| Variant | `dof_bokeh_shape` | Status | Reachable |
|---|---|---|---|
| Circular | `"CIRCULAR"` | working | F2 |
| Hexagonal | `"HEXAGONAL"` | working | F2 |
| Anamorphic | `"ANAMORPHIC"` | working | F2 |

| Field | Value |
|---|---|
| Activation | `RenderConfig.dof_enabled`; F2 |
| Python | [camera_optics_pass.py](../../engine/gfx/passes/camera_optics_pass.py) |
| Shaders | `camera_optics.frag` |
| Inputs | HDR, depth |
| Outputs | HDR with DoF |
| Temporal | none |
| Depends on | depth |
| Excludes | none |
| Quality params | `dof_focus_distance` (5.0), `dof_focal_length` (50), `dof_aperture` (2.8), `dof_max_coc` (24), `dof_anamorphic_ratio` |
| Intended result | bokeh blur |
| Test scene | S5 (foreground/background separation) |
| Camera | focus near then far |
| Expected | out-of-focus regions blur with correct bokeh shape |
| Known artifacts | CoC discontinuity at depth edges |
| Debug view | none |
| Pass/fail | bokeh shape distinguishable; focus plane correct |
| Perf testable | yes |

---

## 22. Motion Blur

| Field | Value |
|---|---|
| Variant | tile-max velocity |
| Status | **working** |
| Activation | `RenderConfig.motion_blur_enabled`; F2 |
| Python | [motion_blur_pass.py](../../engine/gfx/passes/motion_blur_pass.py), [pipeline.py:1771](../../engine/gfx/pipeline.py:1771) |
| Shaders | `motion_blur_tile.frag`, `motion_blur_gather.frag` |
| Inputs | HDR, velocity buffer (RT2), depth |
| Outputs | HDR with blur |
| Temporal | none |
| Depends on | motion vectors (`object_motion_vectors_enabled`) |
| Excludes | runs after TAA by design |
| Quality params | `motion_blur_intensity` (1.5), `motion_blur_max_radius` (48), `motion_blur_tile_size` (16), `motion_blur_max_samples` (32) |
| Intended result | directional blur along motion |
| Test scene | S4 (moving objects), S5 (camera pan) |
| Camera | fast pan |
| Expected | blur along velocity, no smear on static |
| Known artifacts | ghosting at disocclusion |
| Debug view | none |
| Pass/fail | blur direction matches motion; static regions sharp |
| Perf testable | yes |
| **Note** | **Phase 2.5**: `motion_blur_samples` is now consumed by the pass (it takes precedence over the legacy `motion_blur_max_samples` cap). |
---

## 23. Debug Visualization

### 23.1 G-Buffer / pipeline debug modes (24)
| Field | Value |
|---|---|
| Variant | `GBufferDebugMode` 0–23 |
| Status | **working** |
| Activation | `RenderConfig.debug_gbuffer`; F2 "G-Buffer Debug" combo |
| Python | [engine_tweaks.py:18](../../engine/debug/engine_tweaks.py:18), [pipeline.py:1633](../../engine/gfx/pipeline.py:1633) |
| Shaders | `deferred_resolve.frag` (`u_GBufferDebug`) |
| Inputs | G-Buffer, shadow atlas, AO, GI, Hi-Z, bounds |
| Outputs | debug color into HDR |
| Temporal | none |
| Depends on | the resource being debugged |
| Excludes | replaces normal resolve output |
| Quality params | mode + `hiz_debug_mip` |
| Intended result | inspect each buffer |
| Test scene | S1–S9 |
| Camera | any |
| Expected | each mode shows its buffer |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | each of the 24 modes shows a distinct, plausible image |
| Perf testable | no |

Modes: `DISABLED(0)`, `ALBEDO(1)`, `NORMALS(2)`, `MATERIAL(3)`, `DEPTH(4)`, `SHADOW_ATLAS(5)`,
`SHADOW_MASK(6)`, `AO(7)`, `SSGI(8)`, `GI_TOTAL(10)`, `HIZ(11)`, `SPOT_LIGHTS(12)`,
`RADIANCE_CASCADES(13)`, `CASCADE_INDEX(14)`, `CASCADE_COVERAGE(15)`, `CASCADE_BOUNDS(16)`,
`HIZ_BOUNDS_NEAREST(17)`, `HIZ_BOUNDS_FARTHEST(18)`, `HIZ_BOUNDS_SPAN(19)`, `HIZ_BOUNDS_MASK(20)`,
`HIZ_BOUNDS_INVALID(21)`, `HIZ_BOUNDS_RAW(22)`, `DIRECT_RADIANCE(23)`.

**Note**: mode 9 is unused (gap between SSGI=8 and GI_TOTAL=10).

### 23.2 Immediate-mode 3D wireframes
| Field | Value |
|---|---|
| Variant | `DebugDraw` |
| Status | **working** |
| Activation | `pipeline.debug.line/box/sphere` |
| Python | [debug_draw.py](../../engine/debug/debug_draw.py) |
| Shaders | `debug_line.vert/.frag` |
| Inputs | line/box/sphere primitives |
| Outputs | HDR wireframes |
| Temporal | none |
| Depends on | none |
| Excludes | none |
| Quality params | none |
| Intended result | gizmos |
| Test scene | S1 |
| Camera | any |
| Expected | wireframes visible |
| Known artifacts | none |
| Debug view | n/a |
| Pass/fail | gizmos render |
| Perf testable | no |

### 23.3 F1 System Monitor / F2 Graphics / F3 Game Tweaks
| Field | Value |
|---|---|
| Variant | ImGui overlay |
| Status | **working** |
| Activation | F1/F2/F3 (reserved) |
| Python | [debug_menu.py](../../engine/debug/debug_menu.py) |
| Shaders | n/a (ImGui) |
| Inputs | monitor stats, tweaks |
| Outputs | overlay on backbuffer |
| Temporal | n/a |
| Depends on | imgui-bundle |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | live tuning |
| Test scene | any |
| Camera | n/a |
| Expected | panels open/close |
| Known artifacts | n/a |
| Debug view | itself |
| Pass/fail | F1/F2/F3 toggle their panels |
| Perf testable | no |

---

## 24. Temporal Systems

| System | History resource | Status | Reset on |
|---|---|---|---|
| TAA | ping-pong RGBA16F | working | resize |
| SSGI temporal | ping-pong + `invalidate_history()` | working | GI mode switch |
| RC temporal (legacy merge) | history tex | working | GI mode switch |
| RC temporal (cascades) | history tex | **partially working** — `rc_temporal_enabled` default **False** and not exposed in F2 | GI mode switch |
| Motion vectors | prev VP + prev transforms | working | first frame |

**Confirmed**: `ssgi_pass.invalidate_history()` and `rc_pass.invalidate_history()` are called on
every GI mode switch, so switching GI does not blend stale history.

---

## 25. Culling and LOD

### 25.1 Frustum culling
| Field | Value |
|---|---|
| Variant | `frustum_cull_enabled` (default True) |
| Status | **working** |
| Activation | `RenderConfig.frustum_cull_enabled`; F2 "Vectorized Frustum Culling" |
| Python | [culling.py](../../engine/gfx/culling.py), [pipeline.py:1260](../../engine/gfx/pipeline.py:1260) |
| Shaders | n/a (CPU) |
| Inputs | ECS AABBs, VP matrix |
| Outputs | visible mask |
| Temporal | none |
| Depends on | `max_entities` scratch |
| Excludes | none |
| Quality params | none |
| Intended result | skip off-screen batches |
| Test scene | S6 (large scene) |
| Camera | rotate away from geometry |
| Expected | off-screen geometry not drawn |
| Known artifacts | none |
| Debug view | none |
| Pass/fail | draw count drops when looking away |
| Perf testable | yes |

### 25.2 Hi-Z occlusion culling
| Field | Value |
|---|---|
| Variant | `hiz_cull_enabled` (default **False**) |
| Status | **working** |
| Activation | `RenderConfig.hiz_cull_enabled`; F2 "GPU Hi-Z Occlusion Culling" |
| Python | [hiz_pass.py](../../engine/gfx/passes/hiz_pass.py) `cull_aabbs_gpu`, [pipeline.py:1260](../../engine/gfx/pipeline.py:1260) |
| Shaders | `cull_indirect.comp` |
| Inputs | AABBs, Hi-Z pyramid |
| Outputs | visibility mask |
| Temporal | uses previous frame's Hi-Z |
| Depends on | Hi-Z pass |
| Excludes | none |
| Quality params | none |
| Intended result | skip occluded geometry |
| Test scene | S1 (indoor, walls occlude) |
| Camera | look at a wall |
| Expected | geometry behind wall culled |
| Known artifacts | one-frame latency (previous Hi-Z) |
| Debug view | HIZ (11) |
| Pass/fail | draw count drops behind occluders; no visible popping |
| Perf testable | yes |

### 25.3 Mesh LOD
| Field | Value |
|---|---|
| Variant | (none) |
| Status | **absent** |
| Activation | n/a |
| Python | n/a |
| Shaders | n/a |
| Inputs | n/a |
| Outputs | n/a |
| Temporal | n/a |
| Depends on | n/a |
| Excludes | n/a |
| Quality params | n/a |
| Intended result | n/a |
| Test scene | n/a |
| Camera | n/a |
| Expected | n/a |
| Known artifacts | **No mesh LOD system exists.** Distance-based quality reduction is limited to tessellation level and POM/SSDM radius cutoffs. |
| Debug view | n/a |
| Pass/fail | n/a |
| Perf testable | n/a |

---

## 26. Legacy and Experimental Variants (consolidated)

| Variant | Kind | Status | Reachable | Note |
|---|---|---|---|---|
| `engine.gfx.atmosphere` shim | legacy import path | legacy | tests only | re-exports `engine.modules.atmosphere` |
| `HudOverlay` | legacy UI | legacy | tests only | superseded by `UIManager` |
| `engine.core.input` | legacy shim | legacy | none | re-exports `engine.input` |
| `engine.core.character_camera` | legacy shim | legacy | none | alias for `FollowCamera` |
| `VolumetricSettings` | duplicate data | stub | tests only | passes read `RenderConfig` |
| `WaypointGraph` | unwired | stub | none | `ProjectApp.waypoint_graph` always None |
| `CVarRegistry` | unwired | stub | tests only | no runtime consumer |
| `gi_mode="IBL"` | misleading | partial | F2 | behaves like `OFF` |
| `rc_cascades` | unused field | **removed** | — | Phase 2.5: dead field deleted |
| `motion_blur_samples` | now consumed | working | F2 | Phase 2.5: pass reads it first |
| `ssr_mode` | now exposed | working | F2 | Phase 2.5: was consumed, now reachable |
| `ssr_debug_view` | now exposed | working | F2 | Phase 2.5: was consumed, now reachable |
| `ao_debug_view` | now exposed | working | F2 | Phase 2.5 |
| `ssgi_debug_view` | now exposed | working | F2 | Phase 2.5 |
| `rc_debug_view` | now exposed | working | F2 | Phase 2.5 |
| `rc_temporal_debug_view` | now exposed | working | F2 | Phase 2.5 |
| `rc_temporal_enabled` | now exposed | working | F2 | Phase 2.5 |
| `rc_spatial_filter_enabled` | now exposed | working | F2 | Phase 2.5 |
| `ssgi_surface_reject` | now exposed | working | F2 | Phase 2.5 |
| `energy_compensation_enabled` | now exposed | working | F2 | Phase 2.5 |
| `specular_aa_enabled` | now exposed | working | F2 | Phase 2.5 |
| `object_motion_vectors_enabled` | now exposed | working | F2 | Phase 2.5 |
| `white_balance_enabled` | now exposed | working | F2 | Phase 2.5 |
| `u_SunAngularRadius` | **not dead** | working | n/a | Phase 2.5: used by `csm_sampling.glsl` |

---

## 27. Files Inspected

`engine/gfx/pipeline.py`, `engine/gfx/quality_presets.py`, `engine/gfx/post_process.py`,
`engine/gfx/environment.py`, `engine/gfx/shadow_csm.py`, `engine/gfx/spot_shadows.py`,
`engine/gfx/texture_atlas.py`, `engine/gfx/culling.py`, `engine/gfx/gpu_profiler.py`,
`engine/gfx/passes/*.py` (all 24), `engine/modules/atmosphere/*.py`,
`engine/debug/engine_tweaks.py`, `engine/debug/debug_menu.py` (feature-relevant sections),
`engine/core/materials.py`, `engine/core/water.py`, `engine/world/default_world.py`,
`shaders/*.frag`, `shaders/*.vert`, `shaders/*.comp`, `shaders/common/*.glsl`,
`examples/03_water_lighting_lab.py`, `examples/06_graphics_benchmark.py`, `examples/_common.py`.

## 28. Files Changed

None. Only `docs/graphics/*.md` created.

## 29. Tests Performed

Static analysis only. No runtime test was executed in this phase (documentation-only).

## 30. Results

- **26 feature categories** inventoried, **~120 variants** recorded.
- **Confirmed absent**: mesh LOD, temporal upscaling (FSR/DLSS/TSR), general alpha-blended
  transparent geometry, heightmap terrain.
- **Confirmed unreachable from UI** (config-only): 12 parameters, listed in §26.
- **Confirmed unused fields** (no consumer at all): `rc_cascades`, `motion_blur_samples`.
- **Confirmed misleading**: `gi_mode="IBL"` behaves identically to `"OFF"`.

## 31. Known Limitations

- Status reflects **code reachability**, not visual correctness. A "working" feature may still look
  wrong; that is what the manual matrix is for.
- Shader *bindings* were traced statically; no shader was compiled or executed in this phase.
- `debug_menu.py` (102 KB) was inspected only for feature-relevant widgets, not line-by-line.
- Radiance-cascade internal math was not audited; only its activation and parameters.

## 32. Recommended Next Action

Execute [manual_verification_matrix.md](./manual_verification_matrix.md) scene-by-scene, recording
PASS / PASS WITH ISSUES / FAIL / NOT TESTED per variant in
[feature_status_checklist.md](./feature_status_checklist.md). Prioritise the P0 items in
[open_questions.md](../audit/open_questions.md) (render-feature execution, destroy leaks) before
trusting any "working" status for module-contributed features.
