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

## 3. The Jitter-Sign Defect (P0 — repaired in Phase 3B)

### 3.1 Motion vectors

`shaders/gbuffer.frag` originally subtracted jitter where it must add it:

```glsl
// BEFORE (defect)
vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5 - u_ScreenSize_Jitter.zw;
vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
vec2 velocity = curr_uv - prev_uv;
```

The projection is jittered by `proj[8] += jitter_x * 2.0`, which shifts the projected UV
by **−jitter**, so `rendered_uv = nominal_uv − jitter` and the unjitter must **add**.

### 3.2 Numerical proof (static scene)

Camera static, object static, only jitter changes between frames (1920×1080):

| Convention | Velocity | Magnitude |
|---|---|---|
| Old shader (`raw − jitter`) | (960, −270) px | **spurious** |
| Repaired (`raw + jitter`) | (0, 0) px | **0.00 px** |

### 3.3 Repaired form

```glsl
vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5 + u_ScreenSize_Jitter.zw;
vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
vec2 velocity = curr_uv - prev_uv;
```

`v_PrevClip` is produced from `_prev_vp_mat`, which already stores the **unjittered**
previous VP, so the previous UV needs no correction. No `u_PrevJitter` is required for
motion vectors.

### 3.4 Repaired TAA and SMAA

| Shader | Expression (repaired) |
|---|---|
| `taa.frag` | `(v_UV + curr_jitter) * 2.0 - 1.0` |
| `taa.frag` | `velocity = (v_UV + curr_jitter) - prevUV` |
| `taa.frag` | `histUV = (v_UV + curr_jitter) - velocity` |
| `smaa_resolve.frag` | `histUV = (v_UV + u_Jitter) - velocity` |

### 3.5 SSGI temporal (additional consumer)

`ssgi_temporal.frag` consumed velocity without unjittering at all. Its history is stored
on the jittered raster grid, so exact reprojection needs **both** current and previous
jitter:

```glsl
vec2 previous_uv = (v_UV + u_ScreenSize_Jitter.zw) - motion - u_PrevJitter;
```

`u_PrevJitter` is uploaded from `pipeline._prev_jitter` via the `prev_jitter` resource.

**Class: P0 (repaired).**

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
| GI mode switch | `ssgi_pass.invalidate_history()`, `rmgi_pass.invalidate_history()` |
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

- [x] `rendered_uv = nominal_uv − jitter`.
- [x] Unjitter is `nominal_uv = rendered_uv + jitter`.
- [x] The current frame is unjittered in every consumer.
- [x] `_prev_vp_mat` stores the **unjittered** VP, so `v_PrevClip` needs no correction.
- [x] The previous frame's jitter is available as `u_PrevJitter` for SSGI, whose history
      is stored on the jittered raster grid.
- [x] Per-object motion uses `u_PrevWorldTransforms`.
- [x] Velocity is in UV space.
- [x] History is invalidated on resize, first frame, and GI mode switch.

---

## 10. Files Inspected

`engine/gfx/pipeline.py`, `engine/gfx/frame_context.py`, `engine/gfx/passes/{taa_pass,smaa_pass,ssgi_pass}.py`,
`shaders/gbuffer.vert`, `shaders/gbuffer.frag`, `shaders/taa.frag`, `shaders/smaa_resolve.frag`,
`shaders/ssgi_temporal.frag`, `shaders/motion_blur_gather.frag`.

## 11. Files Changed (Phase 3B)

`shaders/gbuffer.frag`, `shaders/taa.frag`, `shaders/smaa_resolve.frag`,
`shaders/ssgi_temporal.frag`, `engine/gfx/passes/ssgi_pass.py`, `engine/gfx/pipeline.py`.

## 12. Tests Performed

Numeric static-scene velocity computation: 960/−270 px with the old shader, 0.00 px with the
repaired convention. SSGI temporal flicker test passes.

## 13. Results

The jitter-sign defect is repaired across motion vectors, TAA, SMAA, and SSGI temporal.

## 14. Known Limitations

Any residual AA wobble from feedback, sharpness, or blend weights is an individual AA defect
for a later phase, not a reprojection defect.

## 15. Recommended Next Action

Proceed to feature-specific AA tuning as a separate phase.
