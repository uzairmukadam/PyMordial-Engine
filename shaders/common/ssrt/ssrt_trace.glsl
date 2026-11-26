// ============================================================================
//  Shared Screen-Space Ray Tracing (SSRT) — Linear Traversal
//
//  The generic view-space linear ray march shared by every screen-space ray
//  tracer. This is the exact traversal previously inlined as AMRGI's
//  ``MarchRange``; the mathematics, constants, sign conventions and early-exit
//  behaviour are unchanged. Stage 1 keeps the signature close to the original
//  for behavioural parity; Stage 2 will generalize the hit contract.
//
//  REQUIRES (declared by the including shader BEFORE this include):
//    - FrameData UBO: u_View, u_Projection, u_InvProjection
//    - sampler2D u_GBufferDepth, u_GBufferNormalMetallic, u_GBufferSurfaceID
//    - common/depth.glsl  (IsUVInBounds, IsBackgroundDepth, ReconstructViewPosition)
//    - common/pbr_brdf.glsl (OctahedralDecode)
//
//  Effect-specific concerns (ray generation, range scheduling, hit payload
//  sampling, Monte Carlo accumulation) stay in the effect shader.
// ============================================================================

#ifndef SSRT_TRACE_GLSL
#define SSRT_TRACE_GLSL

struct SSRT_Hit {
    bool hit;
    vec2 uv;
    float distance;
    int rangeClass;   // caller-defined: 0 miss, 1..N range id
};

// Marches the SAME ray from tStart to tEnd in view space. Returns a hit or miss;
// never restarts the ray. `prevT` carries the last sampled t for refinement.
//
// `baseThickness` and `refineIters` are passed explicitly so this file has no
// dependency on effect-specific uniform names; callers pass their own values.
SSRT_Hit SSRT_TraceLinear(vec3 origin, vec3 dir, vec2 centerUV, float noise,
                          float tStart, float tEnd, int steps, int rangeClass,
                          float originSurfaceID, vec3 normalView,
                          bool allowRefine, int refineIters, float baseThickness) {
    SSRT_Hit h;
    h.hit = false;
    h.uv = vec2(0.0);
    h.distance = 0.0;
    h.rangeClass = 0;

    float span = max(tEnd - tStart, 1e-4);
    float step_len = span / float(max(steps, 1));
    float prevT = tStart;
    for (int s = 0; s < steps; ++s) {
        float f = (float(s) + 0.5 + noise) / float(steps);
        float t = tStart + span * f;
        vec3 p = origin + dir * t;

        vec4 clip = u_Projection * vec4(p, 1.0);
        if (clip.w <= 1e-5) break;                       // invalid projection
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (!IsUVInBounds(uv)) break;                    // viewport exit

        if (distance(uv, centerUV) < 0.006 || t < max(tStart, 0.05)) {
            prevT = t;
            continue;                                    // self-UV / min distance
        }

        float sceneDepth = texture(u_GBufferDepth, uv).r;
        if (IsBackgroundDepth(sceneDepth)) { prevT = t; continue; }

        float sceneZ = ReconstructViewPosition(uv, sceneDepth, u_InvProjection).z;
        float depthDelta = sceneZ - p.z;
        // Thickness must cover the gap between consecutive samples, otherwise a
        // coarse mid/far step can pass straight over a surface and miss it. It is
        // bounded so very coarse far steps cannot create huge leaky slabs.
        float thickness = max(baseThickness, min(step_len * 1.35, 1.5)) * (1.0 + abs(p.z) * 0.03);
        if (depthDelta >= 0.0 && depthDelta < thickness) {
            // Surface-ID self-hit rejection.
            float hitID = texture(u_GBufferSurfaceID, uv).r;
            vec3 hitN = normalize(mat3(u_View) * OctahedralDecode(texture(u_GBufferNormalMetallic, uv).rg));
            vec3 hitPos = ReconstructViewPosition(uv, sceneDepth, u_InvProjection);
            float tangentDist = abs(dot(normalView, hitPos - origin));
            bool selfHit = originSurfaceID > 0.0 &&
                           abs(hitID - originSurfaceID) < 0.25 &&
                           dot(normalView, hitN) > 0.96 &&
                           tangentDist < max(0.03, t * 0.02);
            if (selfHit) { prevT = t; continue; }

            // Backface rejection.
            if (dot(hitN, -dir) <= 0.0) { prevT = t; continue; }

            // Optional bounded refinement (near range only).
            vec2 hitUV = uv;
            float hitT = t;
            if (allowRefine) {
                float a = prevT, b = t;
                for (int r = 0; r < refineIters; ++r) {
                    float m = 0.5 * (a + b);
                    vec3 mp = origin + dir * m;
                    vec4 mc = u_Projection * vec4(mp, 1.0);
                    if (mc.w <= 1e-5) break;
                    vec2 muv = mc.xy / mc.w * 0.5 + 0.5;
                    float md = texture(u_GBufferDepth, muv).r;
                    if (IsBackgroundDepth(md)) { a = m; continue; }
                    float mz = ReconstructViewPosition(muv, md, u_InvProjection).z;
                    if (mz - mp.z >= 0.0) { b = m; hitUV = muv; hitT = m; }
                    else { a = m; }
                }
            }

            h.hit = true;
            h.uv = hitUV;
            h.distance = hitT;
            h.rangeClass = rangeClass;
            return h;
        }
        prevT = t;
    }
    return h;
}

#endif // SSRT_TRACE_GLSL
