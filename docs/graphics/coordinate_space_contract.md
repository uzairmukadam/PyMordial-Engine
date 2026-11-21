# Coordinate Space Contract — Phase 3A

> Every space the renderer uses, the transform between them, and the exact matrix each stage must
> apply. The normal-matrix defect (P0) lives here.

---

## 1. Spaces

| Space | Origin | Axes | Produced by | Consumed by |
|---|---|---|---|---|
| **Model** | primitive centre | primitive-local | `mega_buffer.py` generators | vertex shaders |
| **World** | scene origin | Y-up, right-handed | `model * position` | G-Buffer, lighting, shadows |
| **View** | camera | −Z forward, Y-up | `u_View` | AO, SSGI, SSR, fog |
| **Clip** | — | w-divide pending | `u_ViewProjection` | rasteriser |
| **NDC** | — | x,y ∈ [−1,1], z ∈ [0,1] | clip / w | culling, reconstruction |
| **UV** | bottom-left | [0,1]² | NDC·0.5+0.5 | all screen-space passes |
| **Tangent** | surface | T,B,N | vertex shader TBN | POM, normal mapping |

---

## 2. Transform Chain

```text
model_pos
  → model matrix (TRS)          → world_pos
  → u_View                      → view_pos
  → u_Projection                → clip_pos
  → / w                         → ndc
  → *0.5 + 0.5                  → uv
```

### 2.1 Model matrix

Built by `trs_to_mat4(pos, rot, scl)` in `engine/core/math_utils.py`. Column-major, 16 floats.

**Scale is per-axis and may be non-uniform.** `EntityManager.create_entity(dimensions=...)` derives
scale as `dimensions / base_dimensions`, and each primitive has a different base size.

---

## 3. The Normal Matrix (P0 defect)

### 3.1 The rule

A normal is a **co-vector**, not a vector. Under a model matrix `M`, positions transform by `M` but
normals transform by the **inverse transpose**:

```glsl
mat3 normal_matrix = transpose(inverse(mat3(model)));
vec3 world_normal = normalize(normal_matrix * in_normal);
```

### 3.2 What the engine does

`shaders/gbuffer.vert:88`:

```glsl
mat3 normal_matrix = mat3(model); // Assumes uniform scale or orthogonal
```

`shaders/gbuffer_tess.vert` and `shaders/csm_tess.vert:28` do the same.

### 3.3 Why the assumption fails

The comment's assumption holds only for uniform scale or a pure rotation. The engine violates it:

| Primitive | Base size | `dimensions=(1,1,1)` → scale | Non-uniform |
|---|---|---|---|
| cube | (1,1,1) | (1, 1, 1) | no |
| sphere | (2,2,2) | (0.5, 0.5, 0.5) | no |
| plane | (1,0,1) | (1, 1, 1) | no |
| **capsule** | (0.8, 1.8, 0.8) | **(1.25, 0.556, 1.25)** | **yes** |
| **cylinder** | (2, 1, 2) | **(0.5, 1.0, 0.5)** | **yes** |
| cone | (1,1,1) | (1, 1, 1) | no |

Only capsule and cylinder receive non-uniform scale, which is exactly why only curved geometry
shows the defect.

### 3.4 Quantified error

| Primitive | Normal | Shader | Correct | Error |
|---|---|---|---|---|
| capsule | (0.707, 0.707, 0) | (0.914, 0.406, 0) | (0.406, 0.914, 0) | **42.1°** |
| capsule | (0.577, 0.577, 0.577) | (0.675, 0.300, 0.675) | (0.376, 0.847, 0.376) | **40.4°** |
| cylinder | (0.707, 0.707, 0) | (0.447, 0.894, 0) | (0.894, 0.447, 0) | **36.9°** |

Axis-aligned normals are unaffected (0° error), which is why a cylinder's flat caps look correct
while its barrel does not.

### 3.5 Downstream impact

The corrupted world normal is written to G-Buffer RT1 and consumed by:

- AO (horizon search uses the normal)
- SSGI (hemisphere sampling around the normal)
- SSR (reflection direction)
- Shadow normal-offset bias (`csm_sampling.glsl:89`)
- Specular AA (normal variance)

So one vertex-shader line corrupts five downstream systems on curved geometry.

---

## 4. Normal Space Consistency (confirmed correct)

| Stage | Space | Encoding |
|---|---|---|
| G-Buffer RT1 | **world** | octahedral (RG) |
| AO | world → view via `mat3(u_View)` | decoded |
| SSGI | world → view via `mat3(u_View)` | decoded |
| SSR | world → view via `mat3(u_View)` | decoded |
| TAA normal reject | world | decoded |

All consumers decode to world then transform by the **rotation-only** `mat3(u_View)`. This is
correct because the view matrix has no scale. **No defect.**

---

## 5. Tangent Frame (confirmed correct)

`gbuffer.vert`:

```glsl
vec3 T = normalize(normal_matrix * in_tangent.xyz);
vec3 N = v_Normal;
T = normalize(T - dot(T, N) * N);   // Gram-Schmidt
vec3 B = cross(N, T) * in_tangent.w;
v_TBN = mat3(T, B, N);
```

Gram-Schmidt re-orthogonalisation is correct. **However**, `T` is transformed by the same broken
`normal_matrix`, so the tangent frame is also skewed on non-uniformly-scaled curved geometry. Fixing
the normal matrix fixes the TBN too.

---

## 6. UV Orientation (confirmed correct)

- G-Buffer `v_UV` has bottom-left origin.
- Reconstruction uses `uv * 2.0 - 1.0`, matching.
- No Y flip anywhere in the chain.
- `gl_FragCoord` is only used for noise, never for reconstruction.

**No defect.**

---

## 7. Handedness (confirmed correct)

Right-handed view space, −Z forward. `matrix_look_at` produces a right-handed basis.
`matrix_perspective` sets `mat[11] = -1.0`, which is the right-handed perspective convention.

**No defect.**

---

## 8. Invariants Checklist

- [ ] Normals transform by `transpose(inverse(mat3(model)))`, never `mat3(model)`.
- [ ] Tangents transform by the same normal matrix, then Gram-Schmidt against N.
- [ ] G-Buffer stores **world** normals, octahedral-encoded.
- [ ] Consumers transform world → view with `mat3(u_View)` (rotation only).
- [ ] UV origin is bottom-left everywhere.
- [ ] View space is right-handed, −Z forward.

---

## 9. Files Inspected

`engine/gfx/mega_buffer.py`, `engine/core/ecs.py`, `engine/core/math_utils.py`,
`shaders/gbuffer.vert`, `shaders/gbuffer.frag`, `shaders/gbuffer_tess.vert`, `shaders/csm_tess.vert`,
`shaders/common/csm_sampling.glsl`, `shaders/{gtao,ssgi_trace,ssr_hiz}.frag`.

## 10. Files Changed

None.

## 11. Tests Performed

Numeric comparison of `M·n` vs `inverse(M)ᵀ·n` for capsule and cylinder scales; 36.9°–42.1° error.

## 12. Results

One P0 defect: the normal matrix. Everything else in the coordinate chain is correct.

## 13. Known Limitations

The error was computed for the default `dimensions=(1,1,1)`; other dimension requests produce
different (possibly larger) errors.

## 14. Recommended Next Action

Replace `mat3(model)` with `transpose(inverse(mat3(model)))` in `gbuffer.vert`,
`gbuffer_tess.vert`, and `csm_tess.vert`. Precompute it on the CPU and upload as a per-entity mat3
if the per-vertex inverse proves expensive.
