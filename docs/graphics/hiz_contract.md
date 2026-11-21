# Hi-Z Contract — Phase 3A

> The engine maintains **two** depth hierarchies with different contents and different consumers.
> Confusing them is the source of the "Hi-Z SSR incorrect" and "depth hierarchy correctness"
> questions.

---

## 1. The Two Hierarchies

| | `hiz_texture` | `hiz_bounds_texture` |
|---|---|---|
| Format | R32F | RG32F |
| Contents | raw reversed-Z depth | positive linear view depth interval |
| Reduction | `min` (furthest in reversed-Z) | `min(nearest)`, `max(farthest)` |
| Empty cell | 0.0 (indistinguishable from far) | `(1e30, 0.0)` ⇒ nearest > farthest |
| Filtering | `NEAREST_MIPMAP_NEAREST` | `NEAREST_MIPMAP_NEAREST` |
| Mip count | `num_mips` | `num_mips` |
| Published as | `hiz_texture` / `hiz_mip_count` | `hiz_bounds_texture` / `hiz_bounds_mip_count` |
| Built by | `hiz_copy.comp` + `hiz_downsample.comp` | `hiz_bounds_copy.comp` + `hiz_bounds_downsample.comp` |
| Barriers | present (repaired in Phase 3B) | present |

---

## 2. `hiz_texture` — Scalar Reversed-Z Pyramid

### 2.1 Contents

Mip 0 is a **raw copy** of the G-Buffer depth (reversed-Z, 1.0 = near, 0.0 = far).
Each coarser mip stores the **minimum** of its 2×2 children.

In reversed-Z, `min` selects the **furthest** surface. So a texel at mip *k* answers:

> "What is the depth of the furthest surface anywhere in this cell?"

That is the correct conservative value for **occlusion**: if an object is behind the furthest
surface in a cell, it is definitely hidden.

### 2.2 Consumers

| Consumer | Use | Correct? |
|---|---|---|
| SSR legacy march (`ssr.frag:MarchHiZ`) | skip empty space | yes (given correct mips) |
| SSGI legacy (`ssgi.frag`) | empty-space skip | yes |
| Culling (`cull_indirect.comp`) | occlusion test | yes (repaired in Phase 3B) |

### 2.3 The barrier defect (P0 — repaired in Phase 3B)

`hiz_pass.py` originally dispatched the mip chain with **no `memory_barrier`** between
dispatches, so mip *k* could read a partially-written mip *k−1*. The bounds hierarchy in
the same file already inserted barriers, proving the requirement was known.

**Repair**: `memory_barrier(TEXTURE_FETCH | SHADER_IMAGE_ACCESS)` is now inserted after
the mip-0 copy and between each downsample dispatch. Reduction operator, format, mip
dimensions, reversed-Z semantics, filtering, and the bounds hierarchy are unchanged.

**Validation**: mip 0 equals raw depth; every higher mip equals `min(2×2 children)`;
deterministic across repeated runs at 64×64, 63×63, 65×37, 128×72.

**Class: P0 (repaired).**

### 2.4 Odd-resolution handling

`hiz_downsample.comp` clamps source coordinates:

```glsl
ivec2 p0 = min(baseSrc + ivec2(0, 0), srcSize - 1);
```

Re-reading an edge texel is harmless for a min reduction. **Correct.**

---

## 3. `hiz_bounds_texture` — Linear-Depth Bounds Hierarchy

### 3.1 Contents

Mip 0 converts reversed-Z hardware depth into **positive linear view depth** and stores
`(nearest, farthest)` = `(d, d)` for geometry, `(1e30, 0.0)` for background.

Each coarser mip stores:

```glsl
parent.nearest  = min(child.nearest)  over valid children
parent.farthest = max(child.farthest) over valid children
```

A child is valid when `nearest <= farthest`; empty children are skipped naturally because
`min` ignores `1e30` and `max` ignores `0.0`.

### 3.2 Invariants

| Invariant | Holds? |
|---|---|
| Geometry: `nearest <= farthest`, both > 0 | yes |
| Mip 0: `nearest == farthest` | yes |
| Empty: `nearest > farthest` | yes |
| Parent conservatively contains all children | yes |
| NaN rejected | yes (`c.x <= c.y` is false for NaN) |

### 3.3 Consumers

| Consumer | Use | Correct? |
|---|---|---|
| SSR HIZ_BOUNDS (`ssr_hiz.frag` → `TraceHiZ`) | hierarchical ray march | yes |
| SSGI HIZ_BOUNDS (`ssgi_trace.frag`) | hierarchical ray march | yes |
| RC CASCADES (`rc_trace_intervals.comp`) | interval tracing | yes |
| AO HIZ_BOUNDS (`gtao_hiz.frag`) | horizon search | yes |
| Debug views 17–22 | visualization | yes |

### 3.4 Sampling discipline

`common/hiz_bounds.glsl` provides two samplers:

- `LoadHiZBounds(tex, cell, mip)` — **exact cell fetch**, conservative.
- `SampleHiZBounds(tex, uv, mip)` — **bilinear**, explicitly documented as **non-conservative**.

`hiz_trace.glsl` uses `LoadHiZBounds`. **Correct.**

A consumer that used `SampleHiZBounds` for an occlusion decision would be non-conservative and
could miss occluders. None currently does.

---

## 4. Mip Addressing

Both hierarchies use `u_SrcLevel` / `u_SrcLevel`-style uniforms to select the source mip, and
`bind_to_image(..., level=k)` to select the destination. Mip *k* always reads mip *k−1*.

**Verified**: `dispatch_groups[k]` is sized for mip *k*'s dimensions, so no dispatch overruns.

---

## 5. Culling Contract (repaired in Phase 3B)

`cull_indirect.comp` samples `hiz_texture` (scalar) at a mip chosen from the box's screen footprint:

```glsl
float mipLevel = clamp(ceil(log2(max(maxDim, 1.0))), 0.0, float(u_HiZMipCount - 1));
```

Then:

```glsl
occluderDepth = min(min(depth0, depth1), min(depth2, depth3));  // furthest occluder
float objectDepth = maxNDC.z;                                    // NEAREST box point
if (objectDepth < occluderDepth) { cull; }
```

**This comparison is correct and conservative.** In reversed-Z the object is fully
occluded only if even its **nearest** point (`maxNDC.z`) is further than the **furthest**
occluder over its footprint (`min` of the four samples). Phase 3A's earlier claim that
this should use `minNDC.z` was inverted: switching to the farthest corner would cull
objects that are still partially visible.

**The actual defect was a temporal space mismatch.** The Hi-Z pyramid is built from the
**previous** frame's depth, but culling ran before the G-buffer pass and used the
**current** frame's VP matrix. The repaired contract:

1. Frustum culling uses the **current** VP (`u_ViewProjection`).
2. The occlusion test uses the **previous** VP (`u_PrevViewProjection`), matching the
   space the pyramid was built in.
3. The AABB is expanded by the camera translation (`u_CameraDelta`) so the stale
   previous-frame footprint conservatively covers the current one.
4. The footprint is dilated by one texel at the selected mip so the four corner samples
   cover every texel the AABB touches.

**Class: P0 (repaired).**

---

## 6. Debug Views

| Mode | Shows |
|---|---|
| 11 (`HIZ`) | scalar pyramid at `hiz_debug_mip` |
| 17 (`HIZ_BOUNDS_NEAREST`) | bounds nearest |
| 18 (`HIZ_BOUNDS_FARTHEST`) | bounds farthest |
| 19 (`HIZ_BOUNDS_SPAN`) | farthest − nearest |
| 20 (`HIZ_BOUNDS_MASK`) | empty/occupied |
| 21 (`HIZ_BOUNDS_INVALID`) | corrupt cells |
| 22 (`HIZ_BOUNDS_RAW`) | nearest (R) / farthest (G) / empty (B) |

**Recommended diagnostic**: render mode 11 at mips 0–4 and compare against mode 4 (raw depth).
A racy pyramid shows mip-dependent noise that does not match the raw depth.

---

## 7. Invariants Checklist

- [x] `hiz_texture` mip 0 is a raw copy of G-Buffer depth.
- [x] `hiz_texture` reduction is `min` in reversed-Z.
- [x] **A `memory_barrier` separates every mip dispatch.**
- [x] `hiz_bounds_texture` mip 0 stores positive linear view depth.
- [x] Empty cells are `(1e30, 0.0)`.
- [x] Bounds reduction is `min(nearest)`, `max(farthest)` over valid children.
- [x] Occlusion tests use the **nearest** box corner (`maxNDC.z`) against the **furthest**
      occluder, evaluated in the **previous** frame's VP space.
- [x] Conservative consumers use `LoadHiZBounds`, never `SampleHiZBounds`.

---

## 8. Files Inspected

`engine/gfx/passes/hiz_pass.py`, `shaders/hiz_copy.comp`, `shaders/hiz_downsample.comp`,
`shaders/hiz_bounds_copy.comp`, `shaders/hiz_bounds_downsample.comp`,
`shaders/common/{hiz_bounds,hiz_trace,depth}.glsl`, `shaders/cull_indirect.comp`,
`shaders/{ssr,ssr_hiz,ssgi,ssgi_trace,gtao_hiz}.frag`, `shaders/rc_trace_intervals.comp`.

## 9. Files Changed

None.

## 10. Tests Performed

Code inspection of barrier placement; comparison of scalar vs bounds reduction semantics.

## 11. Results

The bounds hierarchy is correct. The scalar pyramid is correct in reduction but **racy** (missing
barriers). The culling consumer tests the wrong corner.

## 12. Known Limitations

The race was not observed on GPU; it is proven by the absent barrier, which is a spec violation
regardless of whether a particular driver happens to serialise.

## 13. Recommended Next Action

1. Add `memory_barrier(TEXTURE_FETCH | SHADER_IMAGE_ACCESS)` after the copy and after each
   downsample dispatch in the scalar path.
2. Change the culling test to `minNDC.z`.
3. Re-run the Hi-Z debug views at mips 0–4 and confirm they match raw depth.
