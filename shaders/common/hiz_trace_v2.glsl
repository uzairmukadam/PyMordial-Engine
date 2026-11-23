// ============================================================================
//  TraceHiZ V2.2 — incremental perspective-correct mip-0 traversal
//
//  This is an ISOLATED experimental module. It does NOT replace TraceHiZ V1
//  and is not used by any production pass yet. It implements only:
//
//    * projected-ray setup (V2.1 equations, float32)
//    * mip-0 screen-space DDA cell traversal
//    * mip-0 depth intersection using V1-compatible hit semantics
//
//  No hierarchy, no ascent/descent, no parent look-ahead.
//
//  Requires: shaders/common/depth.glsl
// ============================================================================

#ifndef HIZ_TRACE_V2_GLSL
#define HIZ_TRACE_V2_GLSL

#include "shaders/common/depth.glsl"
#include "shaders/common/hiz_bounds.glsl"

#define HIZ_V2_MISS_NONE            0
#define HIZ_V2_MISS_LEFT_SCREEN     1
#define HIZ_V2_MISS_EXCEEDED_DIST   2
#define HIZ_V2_MISS_EXCEEDED_ITERS  3
#define HIZ_V2_MISS_INVALID_PROJ    4
#define HIZ_V2_MISS_INVALID_SETUP   5

// Smallest screen-distance step; guarantees forward progress.
const float HIZ_V2_MIN_DS = 1e-6;
// Smallest projected screen length that is still representable.
const float HIZ_V2_MIN_SCREEN_LENGTH = 1e-6;
// Smallest clip w that is still in front of the camera.
const float HIZ_V2_MIN_CLIP_W = 1e-6;

struct HiZRayV2 {
    bool valid;
    vec2 uvStart;
    vec2 uvEnd;
    float screenLength;
    vec2 dirUV;
    float invWStart;
    float invWEnd;
    float tOverWStart;
    float tOverWEnd;
    float dInvWds;
    float dTOverWds;
    float tStart;
    float tEnd;
    // View-space Z components, retained for exact crossing recovery.
    float originZ;
    float directionZ;
};

struct HiZTraceV2Result {
    bool hit;
    vec2 uv;
    float rawDepth;
    float linearDepth;
    float rayDistance;
    int iterations;
    int missReason;
};

// Projects a view-space point to UV. Returns false when behind the camera.
bool HiZProjectV2(vec3 viewPosition, mat4 projection, out vec2 uv, out float clipW) {
    vec4 clip = projection * vec4(viewPosition, 1.0);
    clipW = clip.w;
    if (clip.w <= HIZ_V2_MIN_CLIP_W) {
        return false;
    }
    uv = (clip.xy / clip.w) * 0.5 + 0.5;
    return true;
}

// Builds the incremental projected-ray representation for P(t) = O + tD.
HiZRayV2 SetupHiZRayV2(vec3 originView, vec3 directionView, float tStart, float tEnd, mat4 projection) {
    HiZRayV2 ray;
    ray.valid = false;
    ray.uvStart = vec2(0.0);
    ray.uvEnd = vec2(0.0);
    ray.screenLength = 0.0;
    ray.dirUV = vec2(0.0);
    ray.invWStart = 0.0;
    ray.invWEnd = 0.0;
    ray.tOverWStart = 0.0;
    ray.tOverWEnd = 0.0;
    ray.dInvWds = 0.0;
    ray.dTOverWds = 0.0;
    ray.tStart = tStart;
    ray.tEnd = tEnd;
    ray.originZ = originView.z;
    ray.directionZ = directionView.z;

    if (!(tEnd > tStart)) {
        return ray;
    }

    vec3 pStart = originView + directionView * tStart;
    vec3 pEnd = originView + directionView * tEnd;

    vec2 uvStart, uvEnd;
    float wStart, wEnd;
    if (!HiZProjectV2(pStart, projection, uvStart, wStart)) return ray;
    if (!HiZProjectV2(pEnd, projection, uvEnd, wEnd)) return ray;
    if (wStart <= HIZ_V2_MIN_CLIP_W || wEnd <= HIZ_V2_MIN_CLIP_W) return ray;

    vec2 deltaUV = uvEnd - uvStart;
    float screenLength = length(deltaUV);
    if (!(screenLength > HIZ_V2_MIN_SCREEN_LENGTH)) return ray;

    ray.uvStart = uvStart;
    ray.uvEnd = uvEnd;
    ray.screenLength = screenLength;
    ray.dirUV = deltaUV / screenLength;
    ray.invWStart = 1.0 / wStart;
    ray.invWEnd = 1.0 / wEnd;
    ray.tOverWStart = tStart / wStart;
    ray.tOverWEnd = tEnd / wEnd;
    ray.dInvWds = (ray.invWEnd - ray.invWStart) / screenLength;
    ray.dTOverWds = (ray.tOverWEnd - ray.tOverWStart) / screenLength;
    ray.valid = true;
    return ray;
}

// Evaluates the representation at a screen distance from the start.
void EvalHiZRayV2(HiZRayV2 ray, float s, out vec2 uv, out float t, out float w) {
    uv = ray.uvStart + ray.dirUV * s;
    float invW = ray.invWStart + ray.dInvWds * s;
    float tOverW = ray.tOverWStart + ray.dTOverWds * s;
    w = 1.0 / invW;
    t = tOverW / invW;
}

// Inverts the perspective-correct t(s) relation to recover the screen distance
// at which the ray reaches parameter `t`. Returns false when the denominator is
// degenerate (t does not vary with s in a recoverable way).
bool ScreenDistanceForT(HiZRayV2 ray, float t, out float s) {
    float denom = ray.dTOverWds - t * ray.dInvWds;
    if (abs(denom) < 1e-12) {
        s = 0.0;
        return false;
    }
    s = (t * ray.invWStart - ray.tOverWStart) / denom;
    return true;
}

// ----------------------------------------------------------------------------
//  Shared DDA helpers (used by mip-0 and hierarchical traversal)
// ----------------------------------------------------------------------------

// Computes the current cell coordinate and the screen distance to the next cell
// boundary at the given mip cell size. Applies the screen-boundary clamp and the
// forward nudge so the next iteration lands in the adjacent cell.
void HiZCellExitV2(
    HiZRayV2 ray,
    float s,
    vec2 cellSizeUV,
    out vec2 cellCoord,
    out float sExit
) {
    vec2 uv = ray.uvStart + ray.dirUV * s;
    cellCoord = floor(uv / cellSizeUV);

    float dsX = 1e30;
    float dsY = 1e30;
    if (abs(ray.dirUV.x) > 1e-12) {
        float boundaryX = (ray.dirUV.x > 0.0 ? cellCoord.x + 1.0 : cellCoord.x) * cellSizeUV.x;
        dsX = (boundaryX - uv.x) / ray.dirUV.x;
    }
    if (abs(ray.dirUV.y) > 1e-12) {
        float boundaryY = (ray.dirUV.y > 0.0 ? cellCoord.y + 1.0 : cellCoord.y) * cellSizeUV.y;
        dsY = (boundaryY - uv.y) / ray.dirUV.y;
    }

    float ds = min(dsX, dsY);
    if (!(ds > 0.0)) {
        ds = HIZ_V2_MIN_DS;
    }
    ds = max(ds, HIZ_V2_MIN_DS);

    // Clamp the segment to the screen boundary so the depth interval never
    // spans off-screen depth (which would create false hits at the edge).
    float sScreenExit = ray.screenLength;
    if (abs(ray.dirUV.x) > 1e-12) {
        float bx = ray.dirUV.x > 0.0 ? 1.0 : 0.0;
        sScreenExit = min(sScreenExit, (bx - ray.uvStart.x) / ray.dirUV.x);
    }
    if (abs(ray.dirUV.y) > 1e-12) {
        float by = ray.dirUV.y > 0.0 ? 1.0 : 0.0;
        sScreenExit = min(sScreenExit, (by - ray.uvStart.y) / ray.dirUV.y);
    }
    sExit = min(s + ds, min(ray.screenLength, sScreenExit));

    // Nudge just past the boundary so the next floor() lands in the adjacent
    // cell. A tiny fraction of a cell; cannot skip a cell.
    if (sExit < ray.screenLength) {
        sExit = min(sExit + HIZ_V2_MIN_DS, ray.screenLength);
    }
}

// Directional viewport test: a ray that has reached the boundary in its
// direction of travel has left the screen.
bool HiZLeftScreenV2(HiZRayV2 ray, vec2 uv) {
    return !IsUVInBounds(uv)
        || (ray.dirUV.x > 0.0 && uv.x >= 1.0)
        || (ray.dirUV.x < 0.0 && uv.x <= 0.0)
        || (ray.dirUV.y > 0.0 && uv.y >= 1.0)
        || (ray.dirUV.y < 0.0 && uv.y <= 0.0);
}

// Mip-0 confirmation at the current cell segment. Returns true and fills
// `result` on a hit. Uses the exact crossing recovery from V2.2.1.
bool HiZConfirmMip0V2(
    sampler2D fullResolutionDepth,
    HiZRayV2 ray,
    float s,
    float sExit,
    vec2 cellCoord,
    vec2 cellSizeUV,
    float tEntry,
    float tExit,
    float wEntry,
    float wExit,
    float tolerance,
    mat4 inverseProjection,
    inout HiZTraceV2Result result
) {
    vec2 sampleUV = ray.uvStart + ray.dirUV * s;
    float rawDepth = texture(fullResolutionDepth, sampleUV).r;
    if (IsBackgroundDepth(rawDepth)) {
        return false;
    }

    float linearDepth = RawDepthToLinearViewDepth(sampleUV, rawDepth, inverseProjection);
    float rayNearest = min(wEntry, wExit);
    float rayFarthest = max(wEntry, wExit);
    if (linearDepth < rayNearest - tolerance || linearDepth > rayFarthest + tolerance) {
        return false;
    }

    // Recover the ray parameter at the surface crossing from the view-space
    // depth equation: surfaceDepth = -(O.z + D.z * t).
    float tHit = tEntry;
    if (abs(ray.directionZ) > 1e-6) {
        tHit = (linearDepth + ray.originZ) / (-ray.directionZ);
    } else {
        tHit = 0.5 * (tEntry + tExit);
    }
    tHit = clamp(tHit, tEntry, tExit);

    // hitUV: perspective-correct screen position of the crossing.
    float sHit;
    vec2 hitUV;
    if (ScreenDistanceForT(ray, tHit, sHit)) {
        sHit = clamp(sHit, s, sExit);
        hitUV = ray.uvStart + ray.dirUV * sHit;
    } else {
        hitUV = sampleUV;
    }
    vec2 cellMin = cellCoord * cellSizeUV;
    vec2 cellMax = cellMin + cellSizeUV;
    hitUV = clamp(hitUV, cellMin, cellMax);

    result.hit = true;
    result.uv = hitUV;
    result.rawDepth = rawDepth;
    result.linearDepth = linearDepth;
    result.rayDistance = tHit;
    result.missReason = HIZ_V2_MISS_NONE;
    return true;
}

// ----------------------------------------------------------------------------
//  Mip-0 traversal (V2.2.1 reference path)
// ----------------------------------------------------------------------------

HiZTraceV2Result TraceHiZ2Mip0(
    sampler2D fullResolutionDepth,
    HiZRayV2 ray,
    float thickness,
    int maxIterations,
    mat4 inverseProjection
) {
    HiZTraceV2Result result;
    result.hit = false;
    result.uv = vec2(0.0);
    result.rawDepth = 0.0;
    result.linearDepth = 0.0;
    result.rayDistance = 0.0;
    result.iterations = 0;
    result.missReason = HIZ_V2_MISS_INVALID_SETUP;

    if (!ray.valid) {
        return result;
    }

    vec2 cellSizeUV = 1.0 / vec2(textureSize(fullResolutionDepth, 0));
    float tolerance = max(thickness, 1e-4);
    int budget = max(maxIterations, 1);
    float s = 0.0;

    for (int i = 0; i < budget; ++i) {
        result.iterations = i + 1;
        if (s >= ray.screenLength) {
            result.missReason = HIZ_V2_MISS_EXCEEDED_DIST;
            return result;
        }

        vec2 uv;
        float tEntry, wEntry;
        EvalHiZRayV2(ray, s, uv, tEntry, wEntry);
        if (HiZLeftScreenV2(ray, uv)) {
            result.missReason = HIZ_V2_MISS_LEFT_SCREEN;
            return result;
        }

        vec2 cellCoord;
        float sExit;
        HiZCellExitV2(ray, s, cellSizeUV, cellCoord, sExit);

        vec2 uvExit;
        float tExit, wExit;
        EvalHiZRayV2(ray, sExit, uvExit, tExit, wExit);

        if (HiZConfirmMip0V2(fullResolutionDepth, ray, s, sExit, cellCoord, cellSizeUV,
                             tEntry, tExit, wEntry, wExit, tolerance, inverseProjection, result)) {
            return result;
        }

        s = sExit;
    }

    result.missReason = HIZ_V2_MISS_EXCEEDED_ITERS;
    return result;
}

// ----------------------------------------------------------------------------
//  Descent-only hierarchical traversal (V2.3)
//
//  Policy:
//    * non-overlapping / empty cell -> advance past it, keep mip
//    * overlap and mip > 0          -> descend one mip, do NOT advance
//    * overlap and mip == 0         -> confirm against full-res depth
//  No ascent. No parent look-ahead.
// ----------------------------------------------------------------------------

HiZTraceV2Result TraceHiZ2Hierarchical(
    sampler2D hizBounds,
    sampler2D fullResolutionDepth,
    HiZRayV2 ray,
    int startMip,
    float thickness,
    int maxIterations,
    mat4 inverseProjection
) {
    HiZTraceV2Result result;
    result.hit = false;
    result.uv = vec2(0.0);
    result.rawDepth = 0.0;
    result.linearDepth = 0.0;
    result.rayDistance = 0.0;
    result.iterations = 0;
    result.missReason = HIZ_V2_MISS_INVALID_SETUP;

    if (!ray.valid) {
        return result;
    }

    int maxMip = max(textureQueryLevels(hizBounds) - 1, 0);
    int mip = clamp(startMip, 0, maxMip);
    float tolerance = max(thickness, 1e-4);
    int budget = max(maxIterations, 1);
    float s = 0.0;

    for (int i = 0; i < budget; ++i) {
        result.iterations = i + 1;
        if (s >= ray.screenLength) {
            result.missReason = HIZ_V2_MISS_EXCEEDED_DIST;
            return result;
        }

        vec2 uv;
        float tEntry, wEntry;
        EvalHiZRayV2(ray, s, uv, tEntry, wEntry);
        if (HiZLeftScreenV2(ray, uv)) {
            result.missReason = HIZ_V2_MISS_LEFT_SCREEN;
            return result;
        }

        ivec2 mipSize = textureSize(hizBounds, mip);
        vec2 cellSizeUV = 1.0 / vec2(mipSize);

        vec2 cellCoord;
        float sExit;
        HiZCellExitV2(ray, s, cellSizeUV, cellCoord, sExit);

        vec2 uvExit;
        float tExit, wExit;
        EvalHiZRayV2(ray, sExit, uvExit, tExit, wExit);

        float rayNearest = min(wEntry, wExit);
        float rayFarthest = max(wEntry, wExit);

        HiZDepthBounds bounds = LoadHiZBoundsSize(hizBounds, ivec2(cellCoord), mip, mipSize);

        if (!DepthIntervalsOverlap(rayNearest, rayFarthest, bounds, thickness)) {
            // Proven empty / non-overlapping: advance past this cell, keep mip.
            s = sExit;
            continue;
        }

        if (mip > 0) {
            // Potential overlap: descend one mip WITHOUT advancing so the same
            // ray location is re-evaluated at the finer level.
            mip -= 1;
            continue;
        }

        // mip == 0: confirm against the full-resolution depth buffer.
        if (HiZConfirmMip0V2(fullResolutionDepth, ray, s, sExit, cellCoord, cellSizeUV,
                             tEntry, tExit, wEntry, wExit, tolerance, inverseProjection, result)) {
            return result;
        }

        // No hit at this mip-0 cell: advance using the established behavior.
        s = sExit;
    }

    result.missReason = HIZ_V2_MISS_EXCEEDED_ITERS;
    return result;
}

// ----------------------------------------------------------------------------
//  Descent + single-step ascent traversal (V2.4)
//
//  Identical to the descent-only path except that, after advancing past a
//  proven non-overlapping cell, it performs ONE parent look-ahead: if the parent
//  cell over the forward parent segment is also proven non-overlapping, it
//  ascends exactly one mip. No multi-ascent, no recursion.
// ----------------------------------------------------------------------------

struct HiZTraceV2Diag {
    int startMip;
    int finalMip;
    int highestMip;
    int ascents;
    int descents;
    int parentLookaheads;
    int parentLookaheadSuccess;
};

HiZTraceV2Result TraceHiZ2HierarchicalAscent(
    sampler2D hizBounds,
    sampler2D fullResolutionDepth,
    HiZRayV2 ray,
    int startMip,
    float thickness,
    int maxIterations,
    mat4 inverseProjection,
    out HiZTraceV2Diag diag
) {
    HiZTraceV2Result result;
    result.hit = false;
    result.uv = vec2(0.0);
    result.rawDepth = 0.0;
    result.linearDepth = 0.0;
    result.rayDistance = 0.0;
    result.iterations = 0;
    result.missReason = HIZ_V2_MISS_INVALID_SETUP;

    diag.startMip = startMip;
    diag.finalMip = startMip;
    diag.highestMip = startMip;
    diag.ascents = 0;
    diag.descents = 0;
    diag.parentLookaheads = 0;
    diag.parentLookaheadSuccess = 0;

    if (!ray.valid) {
        return result;
    }

    int maxMip = max(textureQueryLevels(hizBounds) - 1, 0);
    int mip = clamp(startMip, 0, maxMip);
    diag.finalMip = mip;
    diag.highestMip = mip;
    float tolerance = max(thickness, 1e-4);
    int budget = max(maxIterations, 1);
    float s = 0.0;

    for (int i = 0; i < budget; ++i) {
        result.iterations = i + 1;
        if (s >= ray.screenLength) {
            result.missReason = HIZ_V2_MISS_EXCEEDED_DIST;
            diag.finalMip = mip;
            return result;
        }

        vec2 uv;
        float tEntry, wEntry;
        EvalHiZRayV2(ray, s, uv, tEntry, wEntry);
        if (HiZLeftScreenV2(ray, uv)) {
            result.missReason = HIZ_V2_MISS_LEFT_SCREEN;
            diag.finalMip = mip;
            return result;
        }

        ivec2 mipSize = textureSize(hizBounds, mip);
        vec2 cellSizeUV = 1.0 / vec2(mipSize);

        vec2 cellCoord;
        float sExit;
        HiZCellExitV2(ray, s, cellSizeUV, cellCoord, sExit);

        vec2 uvExit;
        float tExit, wExit;
        EvalHiZRayV2(ray, sExit, uvExit, tExit, wExit);

        float rayNearest = min(wEntry, wExit);
        float rayFarthest = max(wEntry, wExit);

        HiZDepthBounds bounds = LoadHiZBoundsSize(hizBounds, ivec2(cellCoord), mip, mipSize);

        if (!DepthIntervalsOverlap(rayNearest, rayFarthest, bounds, thickness)) {
            // Proven empty / non-overlapping: advance past this cell.
            s = sExit;

            // Single-step parent look-ahead from the new location.
            if (mip < maxMip && s < ray.screenLength) {
                vec2 nextUV;
                float nextT, nextW;
                EvalHiZRayV2(ray, s, nextUV, nextT, nextW);
                if (!HiZLeftScreenV2(ray, nextUV)) {
                    int parentMip = mip + 1;
                    ivec2 parentSize = textureSize(hizBounds, parentMip);
                    vec2 parentCellSizeUV = 1.0 / vec2(parentSize);
                    vec2 parentCoord;
                    float parentExit;
                    HiZCellExitV2(ray, s, parentCellSizeUV, parentCoord, parentExit);

                    vec2 parentUVExit;
                    float parentTExit, parentWExit;
                    EvalHiZRayV2(ray, parentExit, parentUVExit, parentTExit, parentWExit);

                    float parentNearest = min(nextW, parentWExit);
                    float parentFarthest = max(nextW, parentWExit);
                    HiZDepthBounds parentBounds =
                        LoadHiZBoundsSize(hizBounds, ivec2(parentCoord), parentMip, parentSize);

                    diag.parentLookaheads += 1;
                    if (!DepthIntervalsOverlap(parentNearest, parentFarthest, parentBounds, thickness)) {
                        mip = parentMip;
                        diag.ascents += 1;
                        diag.parentLookaheadSuccess += 1;
                        diag.highestMip = max(diag.highestMip, mip);
                    }
                }
            }
            continue;
        }

        if (mip > 0) {
            // Potential overlap: descend one mip WITHOUT advancing.
            mip -= 1;
            diag.descents += 1;
            continue;
        }

        // mip == 0: confirm against the full-resolution depth buffer.
        if (HiZConfirmMip0V2(fullResolutionDepth, ray, s, sExit, cellCoord, cellSizeUV,
                             tEntry, tExit, wEntry, wExit, tolerance, inverseProjection, result)) {
            diag.finalMip = 0;
            return result;
        }

        // No hit at this mip-0 cell: advance using the established behavior.
        s = sExit;
    }

    result.missReason = HIZ_V2_MISS_EXCEEDED_ITERS;
    diag.finalMip = mip;
    return result;
}

#endif // HIZ_TRACE_V2_GLSL
