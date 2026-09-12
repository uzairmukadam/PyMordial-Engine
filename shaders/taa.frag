#version 450 core
// ============================================================
//  AAA Temporal Anti-Aliasing (TAA) Resolve
//
//  References:
//  - Brian Karis, "High Quality Temporal Supersampling" (SIGGRAPH 2014)
//  - Playdead, "Temporal Reprojection Anti-Aliasing in INSIDE" (GDC 2016)
//  - Marco Salvi, "An Excursion in Temporal Supersampling" (GDC 2016)
// ============================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_ResolvedColor;

layout (binding = 0) uniform sampler2D u_CurrentFrame;
layout (binding = 1) uniform sampler2D u_HistoryFrame;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_VelocityTexture;

layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;

    vec4 u_CameraPos_Time;
    vec4 u_ScreenSize_Jitter;

    vec4 u_SunDirection_Intensity;
    vec4 u_SunColor_Ambient;

    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;

    vec4 u_FogColor_Density;
    vec4 u_FogParams;
};

uniform mat4 u_PrevViewProjection;
uniform float u_Feedback = 0.95;   // Base temporal feedback (higher = more stable)

// ============================================================
//  Color-space utilities
// ============================================================

vec3 RGBtoYCoCg(vec3 c) {
    return vec3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r              - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

vec3 YCoCgtoRGB(vec3 c) {
    return vec3(
        c.x + c.y - c.z,
        c.x        + c.z,
        c.x - c.y - c.z
    );
}

float Luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ============================================================
//  Catmull-Rom 5-tap bicubic filter
//  Uses bilinear-combined taps for efficient sharp history lookup.
//  Eliminates the softening of bilinear-only sampling which causes
//  temporal blur buildup over many frames.
// ============================================================

vec3 SampleHistoryCR(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 pos = uv * texSize;
    vec2 tc  = floor(pos - 0.5) + 0.5;
    vec2 f   = pos - tc;
    vec2 f2  = f * f;
    vec2 f3  = f2 * f;

    // Catmull-Rom spline weights
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    // Combine inner pair for bilinear optimization (5 taps instead of 16)
    vec2 w12  = w1 + w2;
    vec2 tc12 = (tc + w2 / max(w12, vec2(1e-6))) / texSize;
    vec2 tc0  = (tc - 1.0) / texSize;
    vec2 tc3  = (tc + 2.0) / texSize;

    float wC = w12.x * w12.y;
    float wT = w12.x * w0.y;
    float wB = w12.x * w3.y;
    float wL = w0.x  * w12.y;
    float wR = w3.x  * w12.y;

    vec3 result =
        texture(tex, vec2(tc12.x, tc12.y)).rgb * wC +
        texture(tex, vec2(tc12.x, tc0.y )).rgb * wT +
        texture(tex, vec2(tc12.x, tc3.y )).rgb * wB +
        texture(tex, vec2(tc0.x,  tc12.y)).rgb * wL +
        texture(tex, vec2(tc3.x,  tc12.y)).rgb * wR;

    return max(result / max(wC + wT + wB + wL + wR, 1e-6), vec3(0.0));
}

// ============================================================
//  AABB ray clipping (Playdead / Karis)
//  Clips point p toward the AABB center along the ray from
//  center → p, finding the closest point on the AABB surface.
//  More stable than hard clamping because it preserves the
//  direction of the history-to-current color difference.
// ============================================================

vec3 ClipToAABB(vec3 boxMin, vec3 boxMax, vec3 p) {
    vec3 center   = 0.5 * (boxMax + boxMin);
    vec3 halfSize = 0.5 * (boxMax - boxMin) + 1e-7;
    vec3 d        = p - center;
    vec3 dNorm    = d / halfSize;
    float maxComp = max(abs(dNorm.x), max(abs(dNorm.y), abs(dNorm.z)));
    if (maxComp > 1.0)
        return center + d / maxComp;
    return p;
}

// ============================================================
//  Main TAA Resolve
// ============================================================

void main() {
    vec2 res   = u_ScreenSize_Jitter.xy;
    vec2 texel = 1.0 / res;

    // ========== 1. DEPTH-DILATED VELOCITY LOOKUP (3x3) ==========
    // Use the closest-to-camera depth in the 3x3 neighborhood for the
    // velocity fetch. This "dilates" moving foreground edges outward so
    // their velocity properly covers the silhouette region, preventing
    // ghost trails behind moving objects.

    float closestDepth = 0.0;
    vec2  closestUV    = v_UV;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2  sUV = v_UV + vec2(x, y) * texel;
            float d   = texture(u_GBufferDepth, sUV).r;
            if (d > closestDepth) {     // Reversed-Z: larger = closer
                closestDepth = d;
                closestUV    = sUV;
            }
        }
    }

    // ========== 2. VELOCITY & REPROJECTION ==========
    // The gbuffer velocity = (curr_jittered_uv - prev_jittered_uv).
    // This naturally maps between jittered screen positions across frames,
    // which is exactly what we need since the history buffer was written
    // at the previous frame's fullscreen quad positions.

    vec2 velocity = texture(u_VelocityTexture, closestUV).xy;

    // Sky / unwritten-velocity fallback: reproject via camera matrices.
    // v_UV is consistent with u_InvProjection (both are in jittered space),
    // so the reconstruction produces correct world positions.
    if (closestDepth < 1e-6 || dot(velocity, velocity) < 1e-12) {
        vec4 clipPos = vec4(v_UV * 2.0 - 1.0, closestDepth, 1.0);
        vec4 viewPos = u_InvProjection * clipPos;
        viewPos     /= max(viewPos.w, 1e-6);
        vec4 wPos    = u_InvView * vec4(viewPos.xyz, 1.0);

        vec4 prevClip = u_PrevViewProjection * wPos;
        vec2 prevUV   = (prevClip.xy / max(prevClip.w, 1e-6)) * 0.5 + 0.5;
        velocity      = v_UV - prevUV;
    }

    vec2 histUV = v_UV - velocity;

    // Out-of-screen rejection → output raw current frame
    if (any(lessThan(histUV, vec2(0.0))) || any(greaterThan(histUV, vec2(1.0)))) {
        out_ResolvedColor = texture(u_CurrentFrame, v_UV);
        return;
    }

    // ========== 3. SAMPLE CURRENT & HISTORY ==========
    // Current frame is sampled at v_UV (jittered position).
    // The jitter IS the temporal supersampling signal — each frame samples
    // a different sub-pixel offset, and the TAA accumulation converges to
    // a supersampled result. Unjittering would defeat this purpose.

    vec3 current = texture(u_CurrentFrame, v_UV).rgb;
    vec3 history = SampleHistoryCR(u_HistoryFrame, histUV, res);

    // ========== 4. NEIGHBORHOOD AABB IN YCoCg ==========
    // Gather 3x3 neighborhood statistics. Operating in YCoCg separates
    // luminance from chroma, giving tighter, more perceptually correct
    // bounding boxes than RGB.

    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec3 s  = texture(u_CurrentFrame, v_UV + vec2(x, y) * texel).rgb;
            vec3 yc = RGBtoYCoCg(s);
            m1 += yc;
            m2 += yc * yc;
        }
    }

    vec3 mu    = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3(0.0)));

    // --- MINIMUM AABB EXTENT (critical for stability) ---
    // On smooth / uniform surfaces all 9 taps return near-identical colors,
    // driving sigma → 0 and collapsing the AABB to a degenerate point.
    // This forces the history to exactly match the current jittered sample,
    // DESTROYING temporal accumulation and causing visible per-frame shimmer
    // and color instability. A sigma floor keeps the box wide enough for the
    // accumulated history to survive the clip unchanged.
    sigma = max(sigma, vec3(0.005));

    float gamma   = 1.25;          // ±1.25σ ≈ 79% coverage — good balance
    vec3  aabbMin = mu - gamma * sigma;
    vec3  aabbMax = mu + gamma * sigma;

    // ========== 5. CLIP HISTORY TO AABB ==========
    vec3 histYCoCg = RGBtoYCoCg(history);
    vec3 clipYCoCg = ClipToAABB(aabbMin, aabbMax, histYCoCg);
    vec3 clipped   = max(YCoCgtoRGB(clipYCoCg), vec3(0.0));

    // ========== 6. DYNAMIC FEEDBACK ==========
    // High feedback (0.95) while stationary for maximum temporal accumulation;
    // ramp down to ~0.80 during fast motion for responsiveness.
    // The smoothstep threshold at 2px safely separates sub-pixel Halton jitter
    // (always < 1px) from real object motion.

    float speedPx = length(velocity * res);
    float fb      = mix(u_Feedback, max(u_Feedback - 0.15, 0.75),
                        smoothstep(2.0, 8.0, speedPx));

    // ========== 7. LUMINANCE-WEIGHTED BLEND (Karis) ==========
    // Per-sample inverse-luminance weighting suppresses firefly artifacts
    // from bright specular highlights in HDR. The weighting preserves
    // chromaticity exactly: w = 1/(1+L) scales all channels equally,
    // so the r:g:b ratio is maintained through the blend.

    float wC  = 1.0 / (1.0 + Luminance(current));
    float wH  = 1.0 / (1.0 + Luminance(clipped));
    float bC  = (1.0 - fb) * wC;
    float bH  = fb          * wH;
    float inv = 1.0 / max(bC + bH, 1e-6);

    vec3 result = (current * bC + clipped * bH) * inv;

    out_ResolvedColor = vec4(max(result, vec3(0.0)), 1.0);
}
