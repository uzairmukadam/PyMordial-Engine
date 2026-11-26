// ============================================================================
//  Shared Screen-Space Ray Tracing (SSRT) — Linear Traversal
//
//  The generic view-space linear ray march shared by every screen-space ray
//  tracer (diffuse GI, reflections, and later AO / contact shadows / refraction).
//  The traversal mathematics, sign conventions and early-exit behaviour are the
//  same as the original AMRGI ``MarchRange``; effect-specific policies (thickness
//  constants, self-hit / backface / orientation rejection, refinement bounds)
//  are selected through an explicit ``SSRT_TraceConfig`` so each caller can
//  reproduce its own behaviour exactly.
//
//  The tracer returns ONLY generic traversal information. It never samples a
//  radiance payload, never accumulates Monte Carlo samples, and never classifies
//  a hit into an effect-specific range — those stay in the effect shader.
//
//  REQUIRES (declared by the including shader BEFORE this include):
//    - FrameData UBO: u_View, u_Projection, u_InvProjection
//    - sampler2D u_GBufferDepth, u_GBufferNormalMetallic, u_GBufferSurfaceID
//    - common/depth.glsl  (IsUVInBounds, IsBackgroundDepth, ReconstructViewPosition)
//    - common/pbr_brdf.glsl (OctahedralDecode)
//
//  CANONICAL BINDING CONTRACT
//  --------------------------
//    u_GBufferDepth          : Reversed-Z Depth32F, raw sample == NDC z,
//                              background == 0.0, NEAREST, no repeat.
//    u_GBufferNormalMetallic : RG = octahedral WORLD-space normal, NEAREST.
//    u_GBufferSurfaceID      : R32F stable per-entity id (+1; 0 == background).
//    u_View / u_Projection / u_InvProjection : column-major, right-handed view
//                              space, -Z forward, GL_ZERO_TO_ONE clip depth.
//  All ray math is performed in VIEW space.
// ============================================================================

#ifndef SSRT_TRACE_GLSL
#define SSRT_TRACE_GLSL

// ---- Hit outcome flags (set on SSRT_Hit.flags) -----------------------------
#define SSRT_HIT_VALID            (1u << 0)  // a valid surface hit was accepted
#define SSRT_MISS_BACKGROUND      (1u << 1)  // ray only ever saw background
#define SSRT_MISS_SCREEN_EDGE     (1u << 2)  // ray left the viewport / invalid proj
#define SSRT_MISS_MAX_DISTANCE    (1u << 3)  // step budget exhausted without a hit
#define SSRT_REJECT_SELF          (1u << 4)  // last candidate rejected as self-hit
#define SSRT_REJECT_BACKFACE      (1u << 5)  // last candidate rejected as backface
#define SSRT_REJECT_PLANE         (1u << 6)  // last candidate rejected by plane test
#define SSRT_REJECT_CONVEX        (1u << 7)  // last candidate rejected by convex test
#define SSRT_REJECT_ORIENTATION   (1u << 8)  // last candidate rejected by orientation

// ---- Trace policy flags (set on SSRT_TraceConfig.flags) --------------------
#define SSRT_TRACE_REJECT_SELF_ID     (1u << 0)  // surface-ID + normal + tangent self-hit
#define SSRT_TRACE_REJECT_BACKFACE    (1u << 1)  // dot(hitN, -dir) <= 0
#define SSRT_TRACE_REJECT_PLANE       (1u << 2)  // hit behind origin tangent plane
#define SSRT_TRACE_REJECT_CONVEX      (1u << 3)  // convex-surface normal match
#define SSRT_TRACE_REJECT_ORIENTATION (1u << 4)  // dot(dir, hitN) > threshold
#define SSRT_TRACE_REFINE_BOUNDED     (1u << 5)  // refinement also checks thickness bound
#define SSRT_TRACE_SELF_UV_POST_REFINE (1u << 6) // apply the self-UV / min-distance
                                                 // rejection to the refined hit instead
                                                 // of the raw march sample

struct SSRT_Hit {
    bool  hit;
    vec2  uv;
    float distance;
    float confidence;   // 1.0 on a valid hit, 0.0 on a miss (generic validity only;
                        // effects apply their own edge/distance weighting)
    int   stepsTaken;   // number of march steps consumed
    uint  flags;        // SSRT_HIT_VALID or an SSRT_MISS_* / SSRT_REJECT_* reason
};

// Effect-specific traversal policy. Every field is required by at least one
// caller; unused fields are simply ignored when their flag is off.
struct SSRT_TraceConfig {
    // Thickness: clamp(max(base, min(step_len*stepScale, stepCap))
    //                  * (1 + |z|*depthScale), min, max)
    float baseThickness;
    float thicknessStepScale;
    float thicknessStepCap;
    float thicknessDepthScale;
    float thicknessMin;
    float thicknessMax;
    int   refineSteps;
    uint  flags;
    // Rejection thresholds (only read when the matching flag is set).
    float planeThreshold;        // SSRT_TRACE_REJECT_PLANE
    float convexNormalDot;       // SSRT_TRACE_REJECT_CONVEX
    float convexNearDist;        // SSRT_TRACE_REJECT_CONVEX
    float convexPlaneDist;       // SSRT_TRACE_REJECT_CONVEX
    float orientationThreshold;  // SSRT_TRACE_REJECT_ORIENTATION
    // March aborts once the sample's view-space z reaches this value (i.e. the
    // ray passes behind the near plane). Set to a large POSITIVE value to
    // disable the check entirely (diffuse GI never aborts).
    float nearClipZ;
    // UV radius around the origin pixel treated as a self-hit and skipped.
    float selfUVEpsilon;
};

// Marches the SAME ray from tStart to tEnd in view space. Returns a hit or miss;
// never restarts the ray. `prevT` carries the last sampled t for refinement.
SSRT_Hit SSRT_TraceLinear(vec3 origin, vec3 dir, vec2 centerUV, float noise,
                          float tStart, float tEnd, int steps,
                          float originSurfaceID, vec3 normalView,
                          SSRT_TraceConfig cfg) {
    SSRT_Hit h;
    h.hit = false;
    h.uv = vec2(0.0);
    h.distance = 0.0;
    h.confidence = 0.0;
    h.stepsTaken = 0;
    h.flags = SSRT_MISS_MAX_DISTANCE;

    uint lastReject = 0u;
    float span = max(tEnd - tStart, 1e-4);
    float step_len = span / float(max(steps, 1));
    float prevT = tStart;
    for (int s = 0; s < steps; ++s) {
        h.stepsTaken = s + 1;
        float f = (float(s) + 0.5 + noise) / float(steps);
        float t = tStart + span * f;
        vec3 p = origin + dir * t;

        if (p.z >= cfg.nearClipZ) { lastReject = SSRT_MISS_SCREEN_EDGE; break; } // behind near plane

        vec4 clip = u_Projection * vec4(p, 1.0);
        if (clip.w <= 1e-5) { lastReject = SSRT_MISS_SCREEN_EDGE; break; }  // invalid projection
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (!IsUVInBounds(uv)) { lastReject = SSRT_MISS_SCREEN_EDGE; break; } // viewport exit

        if ((cfg.flags & SSRT_TRACE_SELF_UV_POST_REFINE) == 0u &&
            (distance(uv, centerUV) < cfg.selfUVEpsilon || t < max(tStart, 0.05))) {
            prevT = t;
            continue;                                    // self-UV / min distance
        }

        float sceneDepth = texture(u_GBufferDepth, uv).r;
        if (IsBackgroundDepth(sceneDepth)) { prevT = t; continue; }

        float sceneZ = ReconstructViewPosition(uv, sceneDepth, u_InvProjection).z;
        float depthDelta = sceneZ - p.z;
        // Thickness must cover the gap between consecutive samples, otherwise a
        // coarse step can pass straight over a surface and miss it. It is bounded
        // so very coarse steps cannot create huge leaky slabs.
        float inner = max(cfg.baseThickness, min(step_len * cfg.thicknessStepScale, cfg.thicknessStepCap));
        float thickness = clamp(inner * (1.0 + abs(p.z) * cfg.thicknessDepthScale),
                                cfg.thicknessMin, cfg.thicknessMax);
        if (depthDelta >= 0.0 && depthDelta < thickness) {
            vec2 candUV = uv;
            float candT = t;
            float candDepth = sceneDepth;

            // Reflections refine first and then validate the refined hit; diffuse
            // GI validates the raw sample and refines afterwards. The ordering is
            // selected by SSRT_TRACE_SELF_UV_POST_REFINE so both callers keep
            // their original behaviour exactly.
            bool refineFirst = (cfg.flags & SSRT_TRACE_SELF_UV_POST_REFINE) != 0u;
            if (refineFirst && cfg.refineSteps > 0) {
                float a = prevT, b = t;
                for (int r = 0; r < cfg.refineSteps; ++r) {
                    float m = 0.5 * (a + b);
                    vec3 mp = origin + dir * m;
                    vec4 mc = u_Projection * vec4(mp, 1.0);
                    if (mc.w <= 1e-5) break;
                    vec2 muv = mc.xy / mc.w * 0.5 + 0.5;
                    float md = texture(u_GBufferDepth, muv).r;
                    if (IsBackgroundDepth(md)) { a = m; continue; }
                    float mz = ReconstructViewPosition(muv, md, u_InvProjection).z;
                    float mdiff = mz - mp.z;
                    bool accept = (mdiff >= 0.0);
                    if ((cfg.flags & SSRT_TRACE_REFINE_BOUNDED) != 0u) {
                        accept = accept && (mdiff < thickness);
                    }
                    if (accept) { b = m; candUV = muv; candT = m; }
                    else { a = m; }
                }
                candDepth = texture(u_GBufferDepth, candUV).r;
                if (IsBackgroundDepth(candDepth)) { prevT = t; continue; }
            }

            float hitID = texture(u_GBufferSurfaceID, candUV).r;
            vec3 hitN = normalize(mat3(u_View) * OctahedralDecode(texture(u_GBufferNormalMetallic, candUV).rg));
            vec3 hitPos = ReconstructViewPosition(candUV, candDepth, u_InvProjection);
            float planeDist = dot(hitPos - origin, normalView);

            // Surface-ID self-hit rejection (diffuse GI).
            if ((cfg.flags & SSRT_TRACE_REJECT_SELF_ID) != 0u) {
                float tangentDist = abs(dot(normalView, hitPos - origin));
                bool selfHit = originSurfaceID > 0.0 &&
                               abs(hitID - originSurfaceID) < 0.25 &&
                               dot(normalView, hitN) > 0.96 &&
                               tangentDist < max(0.03, candT * 0.02);
                if (selfHit) { lastReject = SSRT_REJECT_SELF; prevT = t; continue; }
            }

            // Backface rejection (diffuse GI).
            if ((cfg.flags & SSRT_TRACE_REJECT_BACKFACE) != 0u) {
                if (dot(hitN, -dir) <= 0.0) { lastReject = SSRT_REJECT_BACKFACE; prevT = t; continue; }
            }

            // Tangent-plane rejection (reflections): hit must be in front of the
            // origin surface's tangent plane.
            if ((cfg.flags & SSRT_TRACE_REJECT_PLANE) != 0u) {
                if (planeDist <= cfg.planeThreshold) { lastReject = SSRT_REJECT_PLANE; prevT = t; continue; }
            }

            // Convex-surface self-hit rejection (reflections): a ray leaving a
            // convex object cannot hit geometry with a matching normal nearby.
            if ((cfg.flags & SSRT_TRACE_REJECT_CONVEX) != 0u) {
                if (dot(normalView, hitN) > cfg.convexNormalDot &&
                    (candT < cfg.convexNearDist || planeDist < cfg.convexPlaneDist)) {
                    lastReject = SSRT_REJECT_CONVEX; prevT = t; continue;
                }
            }

            // Surface-orientation rejection (reflections): the ray must hit a
            // front-facing surface.
            if ((cfg.flags & SSRT_TRACE_REJECT_ORIENTATION) != 0u) {
                if (dot(dir, hitN) > cfg.orientationThreshold) {
                    lastReject = SSRT_REJECT_ORIENTATION; prevT = t; continue;
                }
            }

            // Deferred self-UV / min-distance rejection (reflections): the
            // refined hit must not land on the origin pixel or too near the eye.
            if (refineFirst) {
                if (distance(candUV, centerUV) < cfg.selfUVEpsilon || candT < 0.05) {
                    lastReject = SSRT_REJECT_SELF; prevT = t; continue;
                }
            }

            // Diffuse GI refines after validation.
            if (!refineFirst && cfg.refineSteps > 0) {
                float a = prevT, b = t;
                for (int r = 0; r < cfg.refineSteps; ++r) {
                    float m = 0.5 * (a + b);
                    vec3 mp = origin + dir * m;
                    vec4 mc = u_Projection * vec4(mp, 1.0);
                    if (mc.w <= 1e-5) break;
                    vec2 muv = mc.xy / mc.w * 0.5 + 0.5;
                    float md = texture(u_GBufferDepth, muv).r;
                    if (IsBackgroundDepth(md)) { a = m; continue; }
                    float mz = ReconstructViewPosition(muv, md, u_InvProjection).z;
                    float mdiff = mz - mp.z;
                    bool accept = (mdiff >= 0.0);
                    if ((cfg.flags & SSRT_TRACE_REFINE_BOUNDED) != 0u) {
                        accept = accept && (mdiff < thickness);
                    }
                    if (accept) { b = m; candUV = muv; candT = m; }
                    else { a = m; }
                }
            }

            h.hit = true;
            h.uv = candUV;
            h.distance = candT;
            h.confidence = 1.0;
            h.flags = SSRT_HIT_VALID;
            return h;
        }
        prevT = t;
    }

    h.flags = (lastReject != 0u) ? lastReject : SSRT_MISS_MAX_DISTANCE;
    return h;
}

#endif // SSRT_TRACE_GLSL
