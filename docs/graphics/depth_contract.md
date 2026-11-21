# Depth Contract — Phase 3A

> The single authoritative description of the engine's depth convention and every consumer's
> obligation. Any pass that violates this contract produces silent visual corruption.

---

## 1. The Convention

| Property | Value | Source |
|---|---|---|
| Projection | Right-handed perspective | `engine/core/math_utils.py:matrix_perspective` |
| Clip control | `GL_ZERO_TO_ONE` | context creation |
| Reversed-Z | **enabled by default** (`RenderConfig.reverse_z = True`) | `quality_presets.py:79` |
| Near plane | 0.1 m | `pipeline.py:1144` |
| Far plane | `max(2000.0, shadow_distance * 3.0)` | `pipeline.py:1145` |
| Depth attachment | 32-bit float (`depth_texture`) | `g_buffer.py:95` |
| Depth clear | **0.0** | `g_buffer.py:124` |
| Depth comparison | **`GL_GREATER`** (`">"`) | `pipeline.py:1399` |
| Front face | `GL_CCW` | context creation |
| Cull face | back | context creation |

### 1.1 The mapping

```text
near plane  →  ndc.z = 1.0
far plane   →  ndc.z = 0.0
background  →  raw depth = 0.0   (the clear value)
```

Because the clip depth range is already `[0, 1]`, **the raw depth sample IS the NDC z**. No
remapping is required.

### 1.2 The projection matrix

```python
mat[10] = near / (far - near)
mat[14] = (far * near) / (far - near)
mat[11] = -1.0
```

Verified numerically: `z_ndc(near) = 1.0`, `z_ndc(far) = 0.0`, monotonic decreasing in between.

---

## 2. Linear View Depth

"Linear view depth" is the **positive distance along the camera forward axis**:

```glsl
linearDepth = -viewPosition.z
```

**Smaller is nearer.** This is the **opposite ordering** from reversed-Z hardware depth (where
larger is nearer). Every consumer must be explicit about which it holds.

`common/depth.glsl` provides:

| Function | Input | Output | Ordering |
|---|---|---|---|
| `IsBackgroundDepth(raw)` | raw depth | bool | — |
| `ReconstructViewPosition(uv, raw, invProj)` | raw | view-space xyz | — |
| `RawDepthToLinearViewDepth(uv, raw, invProj)` | raw | positive linear | smaller = nearer |

---

## 3. Consumer Obligations

Every pass that reads depth must satisfy **all** of the following.

### 3.1 Reconstruction

```glsl
vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
vec4 view = inverseProjection * clip;
vec3 viewPos = view.xyz / view.w;
```

- `uv` must be the **same** UV that produced the depth sample.
- `inverseProjection` must be the **same** projection that produced the depth (jittered or not).
- The perspective divide (`/ view.w`) is mandatory.

### 3.2 Background rejection

```glsl
if (!(rawDepth > 1e-6)) { /* background */ }
```

The clear value is 0.0, so any sample at or below `1e-6` is sky. A consumer that treats it as
geometry reconstructs a point at infinity and poisons its result.

### 3.3 Jitter consistency

If the depth buffer was written with a jittered projection, the consumer must reconstruct with the
**jittered** inverse projection. AO, SSGI, SSR, SSDM, and fog all do this — they operate in the
jittered frame and never need to unjitter.

Only **temporal** consumers (TAA, SMAA, motion vectors) must unjitter, because they compare against
a history buffer stored in the nominal (unjittered) frame.

### 3.4 Reversed-Z comparisons

| Test | Correct form |
|---|---|
| "Is A nearer than B?" | `A > B` |
| "Is A behind B?" | `A < B` |
| Conservative occlusion (is the box fully behind the occluder?) | `boxFarthest < occluderNearest` |
| Conservative visibility (could the box be visible?) | `boxNearest >= occluderFarthest` |

In reversed-Z, `boxFarthest` is the **minimum** NDC z of the box corners, and `boxNearest` is the
**maximum**.

---

## 4. Per-Consumer Audit

| Consumer | Reads | Reconstructs | Jitter-aware | Background check | Verdict |
|---|---|---|---|---|---|
| G-Buffer | writes | n/a | n/a | n/a | correct |
| Hi-Z scalar (`hiz_copy.comp`) | raw | no (raw copy) | n/a | no | correct |
| Hi-Z bounds (`hiz_bounds_copy.comp`) | raw | `RawDepthToLinearViewDepth` | n/a | **yes** | correct |
| AO (`gtao.frag`, `ssao.frag`) | raw | `u_InvProjection * clip` | no (same frame) | yes | correct |
| AO Hi-Z (`gtao_hiz.frag`) | raw + bounds | both | no | yes | correct |
| SSGI (`ssgi.frag`, `ssgi_trace.frag`) | raw | `ReconstructViewPosition` | no | yes | correct |
| SSR (`ssr.frag`, `ssr_hiz.frag`) | raw | `ReconstructViewPosition` | no | yes | correct |
| SSDM (`ssdm.frag`) | raw | `u_InvProjection * clip` | no | yes | correct (read) |
| Fog (`volumetric_fog_inject.comp`) | raw | `ReconstructViewPosition` | no | yes | correct |
| TAA (`taa.frag`) | raw | `u_InvProjection` | **yes** | yes | **jitter sign wrong** |
| SMAA (`smaa_resolve.frag`) | raw | via velocity | **yes** | yes | **jitter sign wrong** |
| Motion blur (`motion_blur_gather.frag`) | raw | depth compare only | n/a | yes | correct |
| Culling (`cull_indirect.comp`) | Hi-Z | NDC corners | n/a | n/a | **wrong corner** |

---

## 5. Known Violations

### V1 — Culling uses the nearest corner (P0)

`cull_indirect.comp:139-152` compares `maxNDC.z` (nearest) against the occluder. The conservative
test requires `minNDC.z` (farthest). See [foundation_validation.md](./foundation_validation.md) §7.2.

### V2 — SSDM depth write is discarded (P1)

`ssdm_pass.py:133` creates the SSDM FBO with no depth attachment, so `gl_FragDepth` writes are
dropped. SSDM perturbs normals only. See [foundation_validation.md](./foundation_validation.md) §3.3.

### V3 — Jitter sign inverted (P0)

`gbuffer.frag:340`, `taa.frag:203/211/217`, `smaa_resolve.frag:22` subtract jitter where they must
add it. See [temporal_contract.md](./temporal_contract.md).

---

## 6. Invariants Checklist

- [ ] Depth clear is `0.0` when `reverse_z` is true.
- [ ] Depth func is `GL_GREATER` when `reverse_z` is true.
- [ ] Near plane is 1.0 in NDC, far plane is 0.0.
- [ ] Every reconstruction divides by `view.w`.
- [ ] Every consumer rejects `rawDepth <= 1e-6` as background.
- [ ] Temporal consumers unjitter with the **correct sign** on **both** frames.
- [ ] Occlusion tests use the **farthest** box corner in reversed-Z.
- [ ] A pass that writes `gl_FragDepth` has a depth attachment bound.

---

## 7. Files Inspected

`engine/core/math_utils.py`, `engine/gfx/g_buffer.py`, `engine/gfx/frame_context.py`,
`engine/gfx/pipeline.py`, `engine/gfx/passes/{hiz_pass,ssdm_pass,taa_pass,smaa_pass}.py`,
`shaders/common/{depth,hiz_bounds,hiz_trace}.glsl`, `shaders/{gtao,gtao_hiz,ssao,ssgi,ssgi_trace,ssr,ssr_hiz,taa,smaa_resolve,ssdm,volumetric_fog_inject,motion_blur_gather}.frag`,
`shaders/cull_indirect.comp`, `shaders/hiz_{copy,downsample,bounds_copy,bounds_downsample}.comp`.

## 8. Files Changed

None.

## 9. Tests Performed

Numeric round-trip of `matrix_perspective` and `ReconstructViewPosition`; verified exact to float32.

## 10. Results

The depth convention itself is correct and every consumer except culling, SSDM, and the temporal
passes honours it.

## 11. Known Limitations

No GPU readback of the depth buffer was performed; violations are proven by code inspection.

## 12. Recommended Next Action

Fix V1, V2, V3 in that order after the normal-matrix fix, then re-run the depth round-trip test.
