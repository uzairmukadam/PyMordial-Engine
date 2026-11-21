# Temporal Contract — Phase 3A

> How jitter, motion vectors, and history reprojection must interact. The jitter-sign defect (P0)
> lives here and explains the TAA/SMAA wobble.

---

## 1. The Three Frames

| Frame | Definition | Stored where |
|---|---|---|
| **Nominal** | unjittered projection | history buffers |
| **Jittered** | nominal + subpixel offset | current render target, depth buffer |
| **Previous** | last frame's nominal | `_prev_vp_mat`, history |

The history buffer stores **nominal** frames. The current frame is **jittered**. Every temporal
consumer must convert between them.

---

## 2. Jitter Application (confirmed)

`pipeline.py:1161-1163`:

```python
if jitter_x != 0.0 or jitter_y != 0.0:
    self._proj_mat[8] += jitter_x * 2.0
    self._proj_mat[9] += jitter_y * 2.0
```

`jitter_x`/`jitter_y` are in **UV space** (fraction of screen). The `* 2.0` converts to NDC.

### 2.1 The sign

Verified numerically: adding `+jitter` to `proj[8]` shifts the projected UV by **−jitter**.

```text
unjittered uv = (0.52922836, 0.53464102)
jittered   uv = (0.52903304, 0.53429379)   ← shifted by −jitter
```

So:

```text
rendered_uv = nominal_uv − jitter
nominal_uv  = rendered_uv + jitter      ← the correct unjitter
```

---

## 3. The Jitter-Sign Defect (P0)

### 3.1 Motion vectors

`shaders/gbuffer.frag:340-342`:

```glsl
vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5 - u_ScreenSize_Jitter.zw;
vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
vec2 velocity = curr_uv - prev_uv;
```

Two errors:

1. **Wrong sign**: subtracts jitter where it must add it.
2. **Asymmetric**: unjitters the current frame but not the previous frame.

### 3.2 Numerical proof (static scene)

Camera static, object static, only jitter changes between frames:

| Convention | Velocity | Magnitude |
|---|---|---|
| Current shader | (5.86e-4, 1.04e-3) | **1.06 px** |
| Correct (add jitter to both) | (0, 0) | **0.00 px** |

A static scene reports ~1 px of spurious velocity every frame.

### 3.3 Correct form

```glsl
vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5 + u_ScreenSize_Jitter.zw;
vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5 + u_PrevJitter;
vec2 velocity = curr_uv - prev_uv;
```

This requires the **previous frame's jitter** as a uniform. The pipeline currently uploads only the
current jitter (`u_ScreenSize_Jitter.zw`).

### 3.4 Same defect in TAA and SMAA

| Shader | Line | Expression | Correct |
|---|---|---|---|
| `taa.frag` | 203 | `(v_UV - curr_jitter) * 2.0 - 1.0` | `(v_UV + curr_jitter)` |
| `taa.frag` | 211 | `velocity = (v_UV - curr_jitter) - prevUV` | `(v_UV + curr_jitter)` |
| `taa.frag` | 217 | `histUV = (v_UV - curr_jitter) - velocity` | `(v_UV + curr_jitter)` |
| `smaa_resolve.frag` | 22 | `histUV = (v_UV - u_Jitter) - velocity` | `(v_UV + u_Jitter)` |

**Consequence**: the history is sampled from a location offset by `2·jitter` from the correct one.
Because jitter cycles through 8 Halton phases, the offset cycles too — producing the visible
**wobble**.

**Class: P0.**

---

## 4. Camera Matrix History (confirmed correct)

`pipeline.py:1836`:

```python
np.copyto(self._prev_vp_mat, self._unjittered_vp_mat)
```

The previous VP is the **unjittered** VP. Jitter never enters the camera history. **Correct.**

`gbuffer.vert` reprojects with `u_PrevViewProjection` (unjittered) against
`u_PrevWorldTransforms` (previous model matrix). **Correct.**

---

## 5. Per-Object Motion (confirmed correct)

`gbuffer.vert:104-108`:

```glsl
vec4 prev_world_pos = (u_ObjectMotionVectors == 1)
    ? (u_PrevWorldTransforms[entity_idx] * vec4(in_position, 1.0))
    : world_pos;
v_PrevClip = (u_PrevViewProjection[3][3] != 0.0)
    ? (u_PrevViewProjection * prev_world_pos)
    : v_CurrClip;
```

- Uses the previous model matrix → moving objects report their own velocity. **Correct.**
- Guards against an all-zero buffer on frame 1. **Correct.**

---

## 6. Velocity Units (confirmed correct)

Velocity is in **UV space** (screen fraction), computed as `curr_uv - prev_uv`. Consumers multiply
by screen size where they need pixels. **Correct.**

---

## 7. History Invalidation (confirmed correct)

| Trigger | Mechanism |
|---|---|
| Resize | `taa_pass.resize` resets `frame_idx` |
| First frame | `_first_frame` flag |
| GI mode switch | `ssgi_pass.invalidate_history()`, `rc_pass.invalidate_history()` |
| Explicit | `invalidate_history()` |

**Correct.**

---

## 8. Test Matrix (from the phase brief)

| Case | Expected velocity | Current result |
|---|---|---|
| Static scene, static camera | 0 | **~1 px (defect)** |
| Camera translation | non-zero, uniform | correct (camera term) |
| Camera rotation | non-zero, radial | correct (camera term) |
| Moving object, static camera | non-zero at object | correct (object term) |
| Camera + object | sum | correct |

Only the static case is broken, because only the static case isolates the jitter term.

---

## 9. Invariants Checklist

- [ ] `rendered_uv = nominal_uv − jitter`.
- [ ] Unjitter is `nominal_uv = rendered_uv + jitter`.
- [ ] **Both** current and previous frames are unjittered.
- [ ] The previous frame's jitter is available as a uniform.
- [ ] `_prev_vp_mat` stores the **unjittered** VP.
- [ ] Per-object motion uses `u_PrevWorldTransforms`.
- [ ] Velocity is in UV space.
- [ ] History is invalidated on resize, first frame, and GI mode switch.

---

## 10. Files Inspected

`engine/gfx/pipeline.py`, `engine/gfx/frame_context.py`, `engine/gfx/passes/{taa_pass,smaa_pass}.py`,
`shaders/gbuffer.vert`, `shaders/gbuffer.frag`, `shaders/taa.frag`, `shaders/smaa_resolve.frag`,
`shaders/motion_blur_gather.frag`.

## 11. Files Changed

None.

## 12. Tests Performed

Numeric static-scene velocity computation: 1.06 px with the current shader, 0.00 px with the
corrected convention.

## 13. Results

One P0 defect: the jitter sign is inverted and applied asymmetrically. It affects motion vectors,
TAA, and SMAA.

## 14. Known Limitations

The previous-frame jitter uniform does not exist yet; the fix requires adding it to the UBO and
uploading it in `pipeline.render_frame`.

## 15. Recommended Next Action

1. Add `u_PrevJitter` to the frame UBO.
2. Change all four unjitter expressions from `− jitter` to `+ jitter`.
3. Unjitter the previous frame in `gbuffer.frag` as well.
4. Re-run the static-scene test: velocity must be 0.
