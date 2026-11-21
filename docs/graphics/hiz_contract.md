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
| Barriers | **none (defect)** | present |

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
| Culling (`cull_indirect.comp`) | occlusion test | **no — wrong corner** |

### 2.3 The barrier defect (P0)

`hiz_pass.py:186-200`:

```python
self.hiz_texture.bind_to_image(0, read=False, write=True, level=0)
self.copy_prog.run(...)                       # writes mip 0

self.hiz_texture.use(location=0)              # samples mip 0..k-1
for k in range(1, self.num_mips):
    self.hiz_texture.bind_to_image(0, read=False, write=True, level=k)
    self.downsample_prog.run(...)             # reads mip k-1, writes mip k
```

**No `memory_barrier` between dispatches.** Mip *k* may read a partially-written mip *k−1*.

The bounds hierarchy in the same file **does** insert barriers (lines 231, 243), proving the
requirement was known. The scalar path omits it.

**Consequence**: `hiz_texture` mips are non-deterministic. SSR's Hi-Z march and SSGI's empty-space
skip read corrupted data, producing the reported "Hi-Z SSR appears incorrect".

**Class: P0.**

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

## 5. Culling Contract (P0 defect)

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

**Defect**: `maxNDC.z` is the box's **nearest** point. The conservative test must use the
**farthest** point (`minNDC.z`). Using the nearest point culls an object as soon as its front face
is behind the occluder, even when its back half is still visible.

**Correct form**:

```glsl
float objectFarthest = minNDC.z;   // reversed-Z: min = farthest
if (objectFarthest < occluderDepth) { cull; }
```

**Class: P0.**

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

- [ ] `hiz_texture` mip 0 is a raw copy of G-Buffer depth.
- [ ] `hiz_texture` reduction is `min` in reversed-Z.
- [ ] **A `memory_barrier` separates every mip dispatch.**
- [ ] `hiz_bounds_texture` mip 0 stores positive linear view depth.
- [ ] Empty cells are `(1e30, 0.0)`.
- [ ] Bounds reduction is `min(nearest)`, `max(farthest)` over valid children.
- [ ] Occlusion tests use the **farthest** box corner.
- [ ] Conservative consumers use `LoadHiZBounds`, never `SampleHiZBounds`.

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
