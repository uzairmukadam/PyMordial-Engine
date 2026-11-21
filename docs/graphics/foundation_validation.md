# Foundation Validation — Phase 3A

> **Read-only audit.** No rendering algorithm, shader math, or visual parameter was changed.
> Small diagnostic instrumentation only (numeric verification scripts, not committed).
>
> **Purpose**: determine whether the six reported defects share one or more incorrect graphics
> foundations. **No individual effect was fixed.**

---

## 1. Executive Summary

Six reported defects were traced to **four foundational defects**, three of which are P0.

| # | Reported symptom | Root cause | Class |
|---|---|---|---|
| 1 | Shadows wrong on capsules/cylinders | `normal_matrix = mat3(model)` under non-uniform scale | **P0** |
| 2 | AO wrong on curved geometry | Same normal-matrix defect | **P0** |
| 3 | Hi-Z culling removes partially visible objects | Culling tests previous-frame Hi-Z with the **current** VP (space mismatch) | **P0** |
| 4 | Hi-Z SSR incorrect | Scalar Hi-Z pyramid built **without memory barriers** | **P0** |
| 5 | TAA/SMAA wobble | Jitter **sign inverted** in velocity + reprojection | **P0** |
| 6 | SSGI/SSR depth-hierarchy questions | Bounds hierarchy is correct; scalar pyramid is not (see #4) | P0 (shared with #4) |

**One defect explains symptoms 1 and 2.** Curved primitives are the only ones that receive
non-uniform scale from `dimensions`, so they are the only ones that expose the broken normal matrix.

**One defect explains symptoms 4 and 6.** The scalar Hi-Z pyramid is racy, so SSR's Hi-Z march and
any consumer of `hiz_texture` read partially-written mips.

**One defect explains symptom 5.** The jitter sign is inverted, so a static scene reports ~1 px of
spurious velocity, which TAA and SMAA then reproject against — producing the wobble.

---

## 2. A — Geometry Validation

### A.1 Primitive generation (confirmed correct)

| Primitive | Vertices | Indices | Winding | Normals | Tangents |
|---|---|---|---|---|---|
| cube | 24 (4/face) | 36 | CCW outward | per-face, unit | per-face |
| plane | 4 | 6 | CCW +Y | +Y | +X |
| sphere | (24+1)×(16+1) | 2·24·16 − caps | CCW outward | unit, analytic | analytic |
| capsule | 2×(12+1)×(32+1) | 2·32·24 | CCW outward | unit, analytic | analytic |
| cylinder | barrel + 2 caps | 2·32 + 2·32 | CCW outward | unit, analytic | analytic |
| cone | apex + rim + cap | 24 + 24 | CCW outward | unit, analytic | analytic |

**Verified**: sphere, capsule, cylinder, and cone all emit outward-facing CCW triangles with
unit-length analytic normals. Winding is consistent with `GL_CCW` front-face and
`CULL_FACE` back-face culling. **No defect in primitive generation.**

### A.2 The normal-matrix defect (P0)

`shaders/gbuffer.vert:88`:

```glsl
mat3 normal_matrix = mat3(model); // Assumes uniform scale or orthogonal
```

The comment states the assumption. The assumption is **violated by the engine's own API**.

`EntityManager.create_entity(dimensions=...)` converts a requested size into a per-axis scale
(`engine/core/ecs.py:375-379`):

```python
base_dimensions = self.local_half_extents[dense_idx] * 2.0
scalable_axes = base_dimensions > 0.0
self.scales[dense_idx, scalable_axes] = dimensions_array[scalable_axes] / base_dimensions[scalable_axes]
```

Because each primitive has a different base size, the **same** `dimensions=(1,1,1)` produces
different scales:

| Primitive | Base size | Resulting scale | Non-uniform? |
|---|---|---|---|
| cube | (1,1,1) | (1, 1, 1) | no |
| sphere | (2,2,2) | (0.5, 0.5, 0.5) | no |
| plane | (1,0,1) | (1, 1, 1) | no |
| **capsule** | (0.8, 1.8, 0.8) | **(1.25, 0.556, 1.25)** | **yes** |
| **cylinder** | (2, 1, 2) | **(0.5, 1.0, 0.5)** | **yes** |
| cone | (1,1,1) | (1, 1, 1) | no |

**This is exactly why only curved geometry is affected.** Capsule and cylinder are the only
primitives whose base aspect ratio differs from the requested dimensions, so they are the only ones
that receive non-uniform scale — and non-uniform scale is precisely what `mat3(model)` mishandles.

### A.3 Quantified error

For a normal `n`, the correct transform is `normalize(inverse(transpose(M)) · n)`; the shader uses
`normalize(M · n)`.

| Primitive | Scale | Normal | Shader result | Correct result | Error |
|---|---|---|---|---|---|
| capsule | (1.25, 0.556, 1.25) | (0.707, 0.707, 0) | (0.914, 0.406, 0) | (0.406, 0.914, 0) | **42.1°** |
| capsule | (1.25, 0.556, 1.25) | (0.577, 0.577, 0.577) | (0.675, 0.300, 0.675) | (0.376, 0.847, 0.376) | **40.4°** |
| cylinder | (0.5, 1.0, 0.5) | (0.707, 0.707, 0) | (0.447, 0.894, 0) | (0.894, 0.447, 0) | **36.9°** |

Axis-aligned normals (0° error) are unaffected, which is why the flat faces of a cylinder look
correct while its curved barrel does not.

### A.4 Propagation

The same `mat3(model)` appears in:

- `shaders/gbuffer.vert:88` — G-Buffer shading normals (AO, GI, SSR, shadows all consume this)
- `shaders/gbuffer_tess.vert` — tessellated G-Buffer
- `shaders/csm_tess.vert:28` — tessellated **shadow casters**

So the defect corrupts shading normals, AO, GI, SSR, and shadow-caster normals simultaneously.
This is the single shared foundation behind symptoms 1 and 2.

### A.5 Non-defects confirmed

- **AABB transform is conservative.** `recompute_matrix` uses the Arvo absolute-matrix method.
  Verified numerically: for a 45°-rotated capsule the Arvo extents equal the true support extents
  exactly. **No defect.**
- **Normal space is consistent.** G-Buffer stores octahedral **world** normals; AO, SSGI, and SSR
  all decode to world then transform by `mat3(u_View)`. **No defect.**
- **Tangent frame** is Gram-Schmidt re-orthogonalised in the vertex shader. **No defect.**

---

## 3. B — Depth Validation

### B.1 Convention (confirmed correct)

| Property | Value | Source |
|---|---|---|
| Projection | Right-handed perspective | `math_utils.matrix_perspective` |
| Clip depth range | `GL_ZERO_TO_ONE` | context creation |
| Reversed-Z | **yes** (default) | `RenderConfig.reverse_z = True` |
| Near → NDC z | 1.0 | `mat[10] = near/(far-near)` |
| Far → NDC z | 0.0 | same |
| Depth attachment | `depth_texture` (32F) | `g_buffer.py:95` |
| Depth clear | 0.0 | `g_buffer.py:124` |
| Depth func | `GL_GREATER` (`">"`) | `pipeline.py:1399` |
| Near plane | 0.1 | `pipeline.py:1144` |
| Far plane | `max(2000, shadow_distance*3)` | `pipeline.py:1145` |

**Verified numerically**: `matrix_perspective(reverse_z=True)` maps near→1.0 and far→0.0, and
`ReconstructViewPosition` in `common/depth.glsl` inverts it correctly. Round-trip
depth→view→depth is exact to float32 precision.

### B.2 Consumer consistency (confirmed)

| Consumer | Reconstruction | Jitter-aware? | Consistent? |
|---|---|---|---|
| G-Buffer | writes depth | n/a | yes |
| Hi-Z scalar | `hiz_copy.comp` (raw copy) | n/a | yes |
| Hi-Z bounds | `RawDepthToLinearViewDepth` | n/a | yes |
| AO (gtao/ssao) | `u_InvProjection * clip` | **no** | yes (see note) |
| SSGI | `ReconstructViewPosition` | no | yes |
| SSR | `ReconstructViewPosition` | no | yes |
| SSDM | `u_InvProjection * clip` | no | yes |
| Fog | `ReconstructViewPosition` | no | yes |
| TAA | `u_InvProjection` | **yes** | yes |
| Motion blur | depth compare only | n/a | yes |

**Note on jitter**: AO/SSGI/SSR/SSDM reconstruct from the *jittered* depth buffer using the
*jittered* inverse projection, which is self-consistent. They do not need to unjitter because they
operate in the same jittered frame. **No defect.**

### B.3 SSDM depth write (P1)

`ssdm_pass.py:133` creates the SSDM framebuffer with **no depth attachment**:

```python
self.ssdm_fbo = self.ctx.framebuffer(color_attachments=[self.perturbed_normal_tex])
```

`ssdm.frag` writes `gl_FragDepth`, but with no depth attachment those writes are **discarded**.
SSDM therefore perturbs normals (copied back to RT1) but **never modifies depth**, contrary to its
documented contract ("modifying hardware depth").

**Class: P1** — an individual feature defect, not foundational. It does not corrupt other passes
because the depth buffer is simply left unchanged.

---

## 4. C — Hi-Z Pyramid Validation

### 4.1 Scalar pyramid (`hiz_texture`, R32F)

| Property | Value | Correct? |
|---|---|---|
| Mip 0 | raw reversed-Z copy | yes |
| Reduction | `min` of 2×2 (reversed-Z ⇒ min = furthest) | yes |
| Odd dimensions | `min(coord, size-1)` clamp | yes |
| Filtering | `NEAREST_MIPMAP_NEAREST` | yes |
| Mip addressing | `u_SrcLevel` uniform | yes |
| **Barriers** | **none between mip dispatches** | **NO** |

`hiz_pass.py:186-200`:

```python
self.hiz_texture.bind_to_image(0, read=False, write=True, level=0)
self.copy_prog.run(...)                       # writes mip 0

self.hiz_texture.use(location=0)              # samples mip 0..k-1
for k in range(1, self.num_mips):
    self.hiz_texture.bind_to_image(0, read=False, write=True, level=k)
    self.downsample_prog.run(...)             # reads mip k-1, writes mip k
```

There is **no `memory_barrier`** between the copy and the first downsample, nor between successive
downsample dispatches. The bounds hierarchy (`_build_bounds_hierarchy`) *does* insert barriers
(lines 231, 243), which proves the author knew the requirement — the scalar path simply omits it.

**Consequence**: mip *k* may be computed from a partially-written mip *k−1*. The result is
non-deterministic and frame-dependent, which matches "Hi-Z SSR appears incorrect" and "SSGI/SSR
tracing raises questions about depth hierarchy correctness".

**Class: P0.**

### 4.2 Bounds pyramid (`hiz_bounds_texture`, RG32F)

| Property | Value | Correct? |
|---|---|---|
| Mip 0 | reversed-Z → positive linear view depth | yes |
| Empty cell | `(1e30, 0.0)` ⇒ nearest > farthest | yes |
| Reduction | `min(nearest)`, `max(farthest)` over valid children | yes |
| Empty-child skip | `if (c.x <= c.y)` | yes |
| Odd dimensions | clamp to edge | yes |
| Filtering | `NEAREST_MIPMAP_NEAREST` | yes |
| Barriers | present (231, 243) | yes |
| Bilinear sampling | documented as **non-conservative** | yes (documented) |

**The bounds hierarchy is correct.** `hiz_trace.glsl` uses `LoadHiZBounds` (exact cell fetch), not
the non-conservative `SampleHiZBounds`, so the tracer is conservative.

### 4.3 What each texture contains

| Texture | Format | Contents | Consumers |
|---|---|---|---|
| `hiz_texture` | R32F | raw reversed-Z, min-reduced (furthest) | SSR legacy march, SSGI legacy skip, culling |
| `hiz_bounds_texture` | RG32F | positive linear view depth, nearest/farthest interval | SSR HIZ_BOUNDS, SSGI HIZ_BOUNDS, RC, AO HIZ_BOUNDS, debug views |

---

## 5. D — Screen-Space Reconstruction

**Confirmed correct.** `common/depth.glsl`:

```glsl
vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
vec4 view = inverseProjection * clip;
return view.xyz / view.w;
```

- UV orientation: `uv*2-1` matches the G-Buffer's `v_UV` (bottom-left origin). Correct.
- Y inversion: none needed; both G-Buffer and reconstruction use the same convention. Correct.
- Perspective divide: present (`/ view.w`). Correct.
- Pixel-centre offset: `(pos + 0.5)/size` in the bounds copy. Correct.
- Jittered vs non-jittered: consumers use the jittered inverse projection with jittered depth —
  self-consistent. Correct.

**Verified numerically**: reconstructing a known world point at a known distance returns the
original position to float32 precision.

---

## 6. E — Motion Vectors

### 6.1 The jitter sign defect (P0)

`shaders/gbuffer.frag:340-342`:

```glsl
vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5 - u_ScreenSize_Jitter.zw;
vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
vec2 velocity = curr_uv - prev_uv;
```

The current frame is unjittered by **subtracting** jitter; the previous frame is **not** unjittered
at all. But the pipeline applies jitter as `proj[8] += jitter_x * 2.0`, which shifts the projected
UV by **−jitter** (verified numerically: a +0.25 px jitter moves the UV by −0.25 px).

So the correct unjitter is `+jitter`, and it must be applied to **both** frames.

**Numerical proof** (static scene, camera static, only jitter changes):

| Convention | Velocity | Magnitude |
|---|---|---|
| Current shader | (5.86e-4, 1.04e-3) | **1.06 px** |
| Correct (unjitter both, `+jitter`) | (0, 0) | **0.00 px** |

A static scene reports ~1 px of spurious velocity every frame. TAA and SMAA reproject against this
velocity, so the history is sampled from the wrong location — **this is the wobble**.

### 6.2 Same defect in TAA and SMAA

`shaders/taa.frag:203,211,217` and `shaders/smaa_resolve.frag:22` all use `(v_UV - jitter)`.
The sign is wrong in every one.

### 6.3 Non-defects confirmed

- `_prev_vp_mat` stores the **unjittered** VP (`pipeline.py:1836` copies `_unjittered_vp_mat`).
  Correct — jitter must not enter the camera matrix history.
- Per-object motion uses `u_PrevWorldTransforms` (SSBO 7). Correct.
- Velocity units are UV-space (screen fraction). Correct.

---

## 7. F — Bounding Volumes

### 7.1 AABB transform (confirmed correct)

`recompute_matrix` uses the Arvo absolute-matrix method:

```python
self.aabbs[dense_idx, 3] = abs(mat[0])*hx + abs(mat[4])*hy + abs(mat[8])*hz
```

Verified numerically for a 45°-rotated capsule: Arvo extents == true support extents exactly.
**Conservative. No defect.**

### 7.2 Hi-Z culling depth test (P0 — re-diagnosed in Phase 3B)

`shaders/cull_indirect.comp`:

```glsl
occluderDepth = min(min(depth0, depth1), min(depth2, depth3));
float objectDepth = maxNDC.z;
if (objectDepth < occluderDepth) { u_VisibleMask[idx] = 0; return; }
```

**Phase 3A's original claim was inverted.** This comparison is correct and conservative:
in reversed-Z the object is fully occluded only if even its **nearest** point (`maxNDC.z`)
is further than the **furthest** occluder over its footprint. A GPU test of the actual
shader across the 11 required scenarios confirmed 10/10 static cases correct; switching
to `minNDC.z` would have introduced false occlusion on every partially-protruding object.

**The actual defect is a temporal space mismatch.** The Hi-Z pyramid is built from the
**previous** frame's depth (PASS 2.7), but culling runs before the G-buffer pass and used
the **current** frame's VP. Under camera motion this produced false occlusion at rates
from 2.6% (0.02 units/frame) to 58.8% (0.5 units/frame).

**Repair (Phase 3B)**: frustum culling stays in the current VP; the occlusion test runs
in the previous VP (`u_PrevViewProjection`) with the AABB expanded by the camera
translation (`u_CameraDelta`) and the footprint dilated by one texel. Post-repair false
occlusion is 0.0% across all tested motion steps.

**Class: P0 (repaired).**

### 7.3 Near-plane handling (confirmed correct)

`if (corners[i].w <= 0.01) { intersectsNearPlane = true; ... }` → object marked visible.
Conservative. **No defect.**

---

## 8. G — Color Pipeline

### 8.1 Stage formats (confirmed)

| Stage | Format | Range | Encoding |
|---|---|---|---|
| Lighting output | HDR RGBA16F | unbounded | linear |
| Bloom | RGBA16F | unbounded | linear |
| Exposure | 1×1 RGBA16F | EV | linear |
| Tone map | RGBA16F → RGBA8 | [0,1] | **linear** (display-referred) |
| Color grading | RGBA8 | [0,1] | linear |
| sRGB encode | `LinearToSRGB` in `post_process.frag:185` | [0,1] | **sRGB** |
| FXAA/SMAA | RGBA8 | [0,1] | sRGB (operates on encoded) |
| Presentation | backbuffer | [0,1] | sRGB |

### 8.2 Findings

- **Tone mapping occurs exactly once**, in `post_process.frag:149` (`TonemapDispatch`).
- **sRGB encoding occurs exactly once**, at `post_process.frag:185` (`LinearToSRGB`).
- **No double gamma.** FXAA/SMAA operate on already-encoded LDR, which is correct for spatial AA.
- **No stage expects linear but receives encoded.** The HDR chain is linear throughout; the LDR
  chain is encoded throughout.
- **Values exceed 1.0** only in the HDR chain (lighting, bloom, flare), which is intended.
- **Windows HDR** is not involved: the engine presents to a standard sRGB backbuffer and does not
  query or configure HDR. **Not the cause of any reported defect.**

**Class: EXPECTED / no defect.** The color pipeline is correct.

---

## 9. Classification Summary

| ID | Defect | Class | Symptoms |
|---|---|---|---|
| F1 | `mat3(model)` normal matrix under non-uniform scale | **P0** | 1, 2 |
| F2 | Hi-Z culling tests previous-frame Hi-Z with the current VP (space mismatch) | **P0** | 3 |
| F3 | Scalar Hi-Z pyramid missing memory barriers | **P0** | 4, 6 |
| F4 | Jitter sign inverted in velocity + TAA/SMAA reprojection | **P0** | 5 |
| F5 | SSDM writes `gl_FragDepth` to an FBO with no depth attachment | P1 | (silent) |
| F6 | `rc_interval_c0/c1/c2` unused by CASCADES backend | P2 | (documented) |
| — | AABB transform | EXPECTED | none |
| — | Depth convention | EXPECTED | none |
| — | Bounds hierarchy | EXPECTED | none |
| — | Screen-space reconstruction | EXPECTED | none |
| — | Color pipeline | EXPECTED | none |

---

## 10. Files Inspected

`engine/gfx/mega_buffer.py`, `engine/core/ecs.py`, `engine/core/math_utils.py`,
`engine/gfx/g_buffer.py`, `engine/gfx/frame_context.py`, `engine/gfx/pipeline.py`,
`engine/gfx/passes/{hiz_pass,ssdm_pass,taa_pass,smaa_pass,ao_pass,ssgi_pass,ssr_pass}.py`,
`shaders/gbuffer.vert`, `shaders/gbuffer.frag`, `shaders/gbuffer_tess.vert`, `shaders/csm_tess.vert`,
`shaders/csm_depth.vert`, `shaders/hiz_copy.comp`, `shaders/hiz_downsample.comp`,
`shaders/hiz_bounds_copy.comp`, `shaders/hiz_bounds_downsample.comp`, `shaders/cull_indirect.comp`,
`shaders/ssdm.frag`, `shaders/taa.frag`, `shaders/smaa_resolve.frag`, `shaders/post_process.frag`,
`shaders/common/{depth,hiz_bounds,hiz_trace,csm_sampling,tonemap}.glsl`.

## 11. Files Changed

None. Only `docs/graphics/*.md` created.

## 12. Tests Performed

Numeric verification scripts (not committed):
- Jitter sign: static-scene velocity = 1.06 px (should be 0).
- Normal matrix: 36.9°–42.1° error on capsule/cylinder curved normals.
- AABB: Arvo extents == true support extents.
- Depth round-trip: exact to float32.

## 13. Results

Four P0 foundational defects confirmed, each explaining one or more reported symptoms. Three
reported symptoms (1, 2) share a single root cause; two (4, 6) share another.

## 14. Known Limitations

- No GPU capture was taken; the Hi-Z race is proven by code inspection (missing barrier), not by
  observing corrupted mips.
- The culling defect is proven by reading the comparison, not by a rendered A/B.
- SSDM's discarded depth write is proven by the absent depth attachment, not by a depth readback.

## 15. Recommended Next Action

Fix in this order (each is independent, smallest-first):

1. **F4 (jitter sign)** — one-line change in `gbuffer.frag`, `taa.frag`, `smaa_resolve.frag`.
2. **F1 (normal matrix)** — compute `transpose(inverse(mat3(model)))` in the three vertex shaders.
3. **F3 (Hi-Z barriers)** — add `memory_barrier` between scalar mip dispatches.
4. **F2 (culling)** — evaluate the occlusion test in the previous frame's VP space with a
   motion-expanded AABB (see `foundation_repair_report.md`).
5. **F5 (SSDM depth)** — attach the G-Buffer depth to the SSDM FBO, or document that SSDM is
   normal-only.

Do not begin visual tuning until F1–F4 are fixed and re-verified.
