// ============================================================================
//  Reusable Hierarchical Screen-Space Ray Tracer
//
//  Walks a view-space ray through the RG32F Hi-Z depth-bounds hierarchy and
//  reports the first confirmed surface intersection.
//
//  This file contains traversal only. It does not evaluate radiance, apply
//  BRDF weights, or generate rays; those stay in the calling effect.
//
//  Requires: shaders/common/depth.glsl, shaders/common/hiz_bounds.glsl
//
//  Traversal strategy (per iteration):
//    1. Project the current ray point to UV.
//    2. Compute the parametric distance at which the ray leaves the cell that
//       contains that UV at the active mip.
//    3. Compare the ray's depth interval over [t, tExit] with the cell bounds.
//       - No overlap  -> jump past the cell, then look ahead at the parent cell
//                        and ascend one mip only if that parent is PROVEN to
//                        contain no reachable depth interval.
//       - Overlap     -> descend one mip without advancing, so the region is
//                        never skipped.
//    4. At mip zero, confirm against the full-resolution depth buffer.
//
//  The ascent is deliberately a look-ahead rather than a blind coarsen-on-miss.
//  Coarsening on a miss makes the next cell cover a larger screen area, so its
//  exit lands further away and the ray can skip the geometry it was about to
//  reach. Ascending only after the parent has been proven non-overlapping over
//  the segment the ray actually spends inside it cannot hide geometry.
// ============================================================================

#ifndef HIZ_TRACE_GLSL
#define HIZ_TRACE_GLSL

// Miss reasons reported in HiZTraceHit.missReason.
#define HIZ_MISS_NONE            0  // hit == true
#define HIZ_MISS_LEFT_SCREEN     1  // UV left the [0, 1] rectangle
#define HIZ_MISS_EXCEEDED_DIST   2  // ray passed maxDistance
#define HIZ_MISS_EXCEEDED_ITERS  3  // iteration budget exhausted
#define HIZ_MISS_INVALID_PROJ    4  // projection moved behind the camera
#define HIZ_MISS_NO_INTERSECTION 5  // traversal finished without a hit

// UV nudge applied when advancing past a cell so the next iteration cannot
// land back inside the cell it just rejected.
const float HIZ_TRACE_BOUNDARY_EPSILON = 1e-4;

struct HiZTraceRequest {
    vec3 originView;
    vec3 directionView;
    float minDistance;
    float maxDistance;
    float thickness;
    int startMip;
    int maxIterations;
    // When > 0, allow up to this many consecutive proven-empty parent ascents
    // within a single iteration (0 = reference single-step ascent).
    int multiAscent;
};

struct HiZTraceHit {
    bool hit;
    vec2 uv;
    float rawDepth;
    float linearDepth;
    float rayDistance;
    int iterations;
    int finalMip;
    int missReason;
    // Debug-only traversal diagnostics (populated only when the caller asks).
    int highestMip;
    int ascents;
    int descents;
};

// Projects a view-space point to UV. Returns false when the point is behind
// the camera, in which case `uv` is left untouched.
bool HiZProjectToUV(vec3 viewPosition, mat4 projection, out vec2 uv) {
    vec4 clip = projection * vec4(viewPosition, 1.0);
    if (clip.w <= 1e-6) {
        return false;
    }
    uv = (clip.xy / clip.w) * 0.5 + 0.5;
    return true;
}

// Returns the parametric distance at which the ray leaves the cell containing
// `uv` at `mip`. `uvPerUnitT` is d(uv)/dt.
//
// When the ray is parallel to an axis (or moving away from the cell edge) that
// axis contributes no exit, so the other axis decides. If neither axis can
// exit, the caller's distance budget bounds the step.
float HiZCellExitDistance(vec2 uv, vec2 uvPerUnitT, ivec2 cellSize, float t, float maxDistance) {
    vec2 cellSizeUV = 1.0 / vec2(max(cellSize, ivec2(1)));
    vec2 cellMin = floor(uv / cellSizeUV) * cellSizeUV;
    vec2 cellMax = cellMin + cellSizeUV;

    float tExit = maxDistance;

    if (abs(uvPerUnitT.x) > 1e-12) {
        float boundary = uvPerUnitT.x > 0.0 ? cellMax.x : cellMin.x;
        float candidate = t + (boundary - uv.x) / uvPerUnitT.x;
        if (candidate > t) {
            tExit = min(tExit, candidate);
        }
    }
    if (abs(uvPerUnitT.y) > 1e-12) {
        float boundary = uvPerUnitT.y > 0.0 ? cellMax.y : cellMin.y;
        float candidate = t + (boundary - uv.y) / uvPerUnitT.y;
        if (candidate > t) {
            tExit = min(tExit, candidate);
        }
    }

    return tExit;
}

// Core traversal with explicit projection matrices.
HiZTraceHit TraceHiZWithMatrices(
    sampler2D hizTexture,
    sampler2D fullResolutionDepth,
    HiZTraceRequest request,
    mat4 projection,
    mat4 inverseProjection
) {
    HiZTraceHit result;
    result.hit = false;
    result.uv = vec2(0.0);
    result.rawDepth = 0.0;
    result.linearDepth = 0.0;
    result.rayDistance = 0.0;
    result.iterations = 0;
    result.finalMip = 0;
    result.missReason = HIZ_MISS_NO_INTERSECTION;
    result.highestMip = 0;
    result.ascents = 0;
    result.descents = 0;

    float rayLength = length(request.directionView);
    if (rayLength <= 1e-8) {
        result.missReason = HIZ_MISS_INVALID_PROJ;
        return result;
    }
    vec3 rayDir = request.directionView / rayLength;

    float tStart = max(request.minDistance, 0.0);
    float tEnd = max(request.maxDistance, tStart);
    if (tEnd <= tStart) {
        result.missReason = HIZ_MISS_EXCEEDED_DIST;
        return result;
    }

    int maxMip = max(textureQueryLevels(hizTexture) - 1, 0);
    int mip = clamp(request.startMip, 0, maxMip);
    result.highestMip = mip;
    int iterationBudget = max(request.maxIterations, 1);

    float t = tStart;

    for (int i = 0; i < iterationBudget; ++i) {
        result.iterations = i + 1;

        if (t >= tEnd) {
            result.missReason = HIZ_MISS_EXCEEDED_DIST;
            return result;
        }

        vec3 rayPosition = request.originView + rayDir * t;
        vec2 uv;
        if (!HiZProjectToUV(rayPosition, projection, uv)) {
            result.missReason = HIZ_MISS_INVALID_PROJ;
            return result;
        }
        if (!IsUVInBounds(uv)) {
            result.missReason = HIZ_MISS_LEFT_SCREEN;
            return result;
        }

        ivec2 cellSize = textureSize(hizTexture, mip);
        ivec2 cell = HiZCellFromUVSize(uv, cellSize);
        HiZDepthBounds bounds = LoadHiZBoundsSize(hizTexture, cell, mip, cellSize);

        // Parametric exit from this cell. A tiny forward probe gives d(uv)/dt.
        // The same d(uv)/dt is reused by the ascent look-ahead below, since the
        // direction of UV motion is independent of the mip level.
        float probeT = t + max((tEnd - t) * 1e-3, 1e-4);
        vec2 probeUV;
        vec2 uvPerUnitT = vec2(0.0);
        bool haveUVPerUnitT = false;
        float tExit = tEnd;
        if (HiZProjectToUV(request.originView + rayDir * probeT, projection, probeUV)) {
            uvPerUnitT = (probeUV - uv) / (probeT - t);
            haveUVPerUnitT = true;
            tExit = HiZCellExitDistance(uv, uvPerUnitT, cellSize, t, tEnd);
        }
        tExit = clamp(tExit, t, tEnd);

        // Ray depth interval across the segment that stays inside this cell.
        float rayNearest = max(-(request.originView.z + rayDir.z * t), 0.0);
        float rayFarthest = max(-(request.originView.z + rayDir.z * tExit), 0.0);

        if (!DepthIntervalsOverlap(rayNearest, rayFarthest, bounds, request.thickness)) {
            // No possible intersection in this cell. Advance by the smaller of:
            //  * the depth jump needed to reach the cell's nearest bound, which
            //    is what lets rays parallel to the view axis (whose UV never
            //    changes) make progress at all, and
            //  * the cell exit, so a ray that also moves in UV never overshoots
            //    the cell it is currently testing.
            float tAdvance = tExit;
            if (rayFarthest < bounds.nearest) {
                float targetDepth = max(bounds.nearest - request.thickness, rayFarthest);
                float dz = -rayDir.z;  // d(linear depth)/dt
                if (dz > 1e-6) {
                    tAdvance = min(tExit, t + (targetDepth - rayNearest) / dz);
                }
            }
            t = max(tAdvance, t + HIZ_TRACE_BOUNDARY_EPSILON);

            // Look-ahead ascent. The ray has just left a cell that provably
            // holds nothing it can reach, so try the parent cell that covers
            // the post-advance position. The parent is only adopted when its
            // own bounds are ALSO proven non-overlapping over the segment the
            // ray spends inside it. Because a parent's interval is the union of
            // its children's intervals, a proven-empty parent cannot hide
            // geometry, so this coarsens without ever skipping a surface.
            //
            // Ascending only after the parent has been tested is what prevents
            // the ascend/descend ping-pong that a blind coarsen-on-miss causes.
            if (mip < maxMip && t < tEnd) {
                vec3 nextPosition = request.originView + rayDir * t;
                vec2 nextUV;
                if (HiZProjectToUV(nextPosition, projection, nextUV) && IsUVInBounds(nextUV)) {
                    int parentMip = mip + 1;
                    ivec2 parentCellSize = textureSize(hizTexture, parentMip);
                    ivec2 parentCell = HiZCellFromUVSize(nextUV, parentCellSize);
                    HiZDepthBounds parentBounds = LoadHiZBoundsSize(hizTexture, parentCell, parentMip, parentCellSize);

                    // Reuse the already-computed d(uv)/dt: the parent probe would
                    // project the same two ray points, so its result is identical.
                    float parentExit = tEnd;
                    if (haveUVPerUnitT) {
                        parentExit = HiZCellExitDistance(nextUV, uvPerUnitT, parentCellSize, t, tEnd);
                    }
                    parentExit = clamp(parentExit, t, tEnd);

                    // parentRayNearest is identical to rayNearest (same t).
                    float parentRayFarthest = max(-(request.originView.z + rayDir.z * parentExit), 0.0);

                    if (!DepthIntervalsOverlap(rayNearest, parentRayFarthest, parentBounds, request.thickness)) {
                        mip = parentMip;
                        result.ascents += 1;
                        result.highestMip = max(result.highestMip, mip);
                    }
                }
            }
            continue;
        }

        if (mip > 0) {
            // Possible intersection: descend without advancing so the region is
            // refined rather than skipped.
            mip -= 1;
            result.descents += 1;
            continue;
        }

        // Mip zero: confirm against the full-resolution depth buffer.
        float rawDepth = texture(fullResolutionDepth, uv).r;
        if (IsBackgroundDepth(rawDepth)) {
            t = tExit + HIZ_TRACE_BOUNDARY_EPSILON;
            continue;
        }

        float linearDepth = RawDepthToLinearViewDepth(uv, rawDepth, inverseProjection);
        float rayDepth = max(-rayPosition.z, 0.0);
        float delta = rayDepth - linearDepth;
        float tolerance = max(request.thickness, 1e-4);

        if (delta >= -tolerance && delta <= tolerance) {
            result.hit = true;
            result.uv = uv;
            result.rawDepth = rawDepth;
            result.linearDepth = linearDepth;
            result.rayDistance = t;
            result.finalMip = 0;
            result.missReason = HIZ_MISS_NONE;
            return result;
        }

        if (delta < -tolerance) {
            // Ray is still in front of this surface: advance in depth until it
            // reaches the surface's near side, but never past the cell exit.
            float dz = -rayDir.z;
            if (dz > 1e-6) {
                float tDepth = (linearDepth - tolerance - rayDepth) / dz;
                t = max(min(tExit, t + tDepth), t + HIZ_TRACE_BOUNDARY_EPSILON);
            } else {
                t = tExit + HIZ_TRACE_BOUNDARY_EPSILON;
            }
        } else {
            // Ray passed behind the surface without landing in the slab.
            t = tExit + HIZ_TRACE_BOUNDARY_EPSILON;
        }
    }

    result.missReason = HIZ_MISS_EXCEEDED_ITERS;
    return result;
}

// Convenience entry point used by effects that already have the FrameData UBO
// in scope (u_Projection / u_InvProjection).
HiZTraceHit TraceHiZ(
    sampler2D hizTexture,
    sampler2D fullResolutionDepth,
    HiZTraceRequest request
) {
    return TraceHiZWithMatrices(hizTexture, fullResolutionDepth, request, u_Projection, u_InvProjection);
}

#endif // HIZ_TRACE_GLSL
