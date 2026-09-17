#version 450 core
// ============================================================
//  PyMordial Engine: Temporal Anti-Aliasing (TAA) Resolve
//
//  Clean Ground-Up Rewrite:
//  - Subpixel projection unjittering and nominal history tracking
//  - 3x3 depth-dilated velocity fetching (eliminates edge ghosting)
//  - Tonemapped YCoCg variance-clipping (zero color shift, zero fireflies)
//  - Normalized 5-tap Catmull-Rom bicubic history reprojection
//  - Velocity-adaptive temporal feedback
//  - Non-haloing contrast-adaptive sharpening (CAS)
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
    vec4 u_ScreenSize_Jitter; // xy = (w, h), zw = (jitter_x, jitter_y in UV space)

    vec4 u_SunDirection_Intensity;
    vec4 u_SunColor_Ambient;

    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;

    vec4 u_FogColor_Density;
    vec4 u_FogParams;
};

uniform mat4 u_PrevViewProjection;
uniform float u_Feedback = 0.95;   // Base temporal feedback (0.80 .. 0.98)
uniform float u_Sharpness = 0.35;  // Post-resolve sharpening (0.0 .. 1.0)
uniform float u_Gamma = 1.25;      // Variance box scale factor (0.75 .. 2.0)

// ============================================================
//  Color Space Transforms: RGB <-> YCoCg
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

// Reversible luminance-weighted compression in YCoCg space
// Preserves chromaticity (Co/Y and Cg/Y) identically, eliminating HDR color shift
vec3 TonemapYCoCg(vec3 yc) {
    float w = 1.0 / (1.0 + max(yc.x, 0.0));
    return yc * w;
}

vec3 UntonemapYCoCg(vec3 yc) {
    float w = 1.0 / max(1.0 - yc.x, 1e-4);
    return yc * w;
}

// ============================================================
//  Catmull-Rom Bicubic History Sampling (5-tap bilinear)
// ============================================================

vec3 SampleHistoryCR(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 pos = uv * texSize;
    vec2 tc  = floor(pos - 0.5) + 0.5;
    vec2 f   = pos - tc;
    vec2 f2  = f * f;
    vec2 f3  = f2 * f;

    // Catmull-Rom weights
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    vec2 w12  = w1 + w2;
    vec2 tc12 = (tc + w2 / max(w12, vec2(1e-5))) / texSize;
    vec2 tc0  = (tc - 1.0) / texSize;
    vec2 tc3  = (tc + 2.0) / texSize;

    float wC = max(w12.x * w12.y, 0.0);
    float wT = max(w12.x * w0.y, 0.0);
    float wB = max(w12.x * w3.y, 0.0);
    float wL = max(w0.x  * w12.y, 0.0);
    float wR = max(w3.x  * w12.y, 0.0);

    float sumW = wC + wT + wB + wL + wR;
    if (sumW < 1e-5) {
        return texture(tex, uv).rgb;
    }

    vec3 result =
        texture(tex, vec2(tc12.x, tc12.y)).rgb * wC +
        texture(tex, vec2(tc12.x, tc0.y )).rgb * wT +
        texture(tex, vec2(tc12.x, tc3.y )).rgb * wB +
        texture(tex, vec2(tc0.x,  tc12.y)).rgb * wL +
        texture(tex, vec2(tc3.x,  tc12.y)).rgb * wR;

    return max(result / sumW, vec3(0.0));
}

// ============================================================
//  AABB Ray Clipping (Playdead / Salvi)
// ============================================================

vec3 ClipToAABB(vec3 boxMin, vec3 boxMax, vec3 p) {
    vec3 center   = 0.5 * (boxMax + boxMin);
    vec3 halfSize = 0.5 * (boxMax - boxMin) + 1e-6;
    vec3 d        = p - center;
    vec3 dNorm    = d / halfSize;
    float maxComp = max(abs(dNorm.x), max(abs(dNorm.y), abs(dNorm.z)));
    if (maxComp > 1.0) {
        return center + d / maxComp;
    }
    return p;
}

// ============================================================
//  Main TAA Resolve Entry Point
// ============================================================

void main() {
    vec2 res   = u_ScreenSize_Jitter.xy;
    vec2 texel = 1.0 / max(res, vec2(1.0));
    vec2 curr_jitter = u_ScreenSize_Jitter.zw;

    // 1. 3x3 Depth-Dilated Velocity Fetching (Reversed-Z: larger = closer to camera)
    float closestDepth = -1.0;
    vec2  closestUV    = v_UV;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 sUV = v_UV + vec2(x, y) * texel;
            float d  = texture(u_GBufferDepth, sUV).r;
            if (d > closestDepth) {
                closestDepth = d;
                closestUV    = sUV;
            }
        }
    }

    vec2 velocity = texture(u_VelocityTexture, closestUV).xy;

    // 2. Camera-Matrix Reprojection Fallback for background / sky
    if (closestDepth <= 1e-6 || dot(velocity, velocity) <= 1e-12) {
        vec2 unjittered_ndc = (v_UV - curr_jitter) * 2.0 - 1.0;
        vec4 clipPos = vec4(unjittered_ndc, max(closestDepth, 0.0), 1.0);
        vec4 viewPos = u_InvProjection * clipPos;
        viewPos     /= max(viewPos.w, 1e-6);
        vec4 wPos    = u_InvView * vec4(viewPos.xyz, 1.0);

        vec4 prevClip = u_PrevViewProjection * wPos;
        vec2 prevUV   = (prevClip.xy / max(prevClip.w, 1e-6)) * 0.5 + 0.5;
        velocity      = (v_UV - curr_jitter) - prevUV;
    }

    // 3. Compute History Lookup UV
    // The history buffer stores nominal unjittered resolved frames.
    // Fragment at v_UV has subpixel offset curr_jitter, so its nominal previous UV is:
    vec2 histUV = (v_UV - curr_jitter) - velocity;

    // Rejection on screen edges
    if (any(lessThan(histUV, vec2(0.0))) || any(greaterThan(histUV, vec2(1.0)))) {
        out_ResolvedColor = texture(u_CurrentFrame, v_UV);
        return;
    }

    // 4. Sample Current Frame and History Frame
    vec3 current_rgb = texture(u_CurrentFrame, v_UV).rgb;
    vec3 history_rgb = SampleHistoryCR(u_HistoryFrame, histUV, res);

    // 5. Gather 3x3 Neighborhood in Tonemapped YCoCg Space
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    vec3 boxMin = vec3(1e6);
    vec3 boxMax = vec3(-1e6);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec3 s_rgb = texture(u_CurrentFrame, v_UV + vec2(x, y) * texel).rgb;
            vec3 s_yc  = TonemapYCoCg(RGBtoYCoCg(s_rgb));
            m1 += s_yc;
            m2 += s_yc * s_yc;
            boxMin = min(boxMin, s_yc);
            boxMax = max(boxMax, s_yc);
        }
    }

    vec3 mu    = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3(0.0)));

    // Minimum variance floor to avoid AABB collapse on smooth surfaces
    sigma = max(sigma, vec3(0.004));

    vec3 aabbMin = clamp(mu - u_Gamma * sigma, boxMin, boxMax);
    vec3 aabbMax = clamp(mu + u_Gamma * sigma, boxMin, boxMax);

    // 6. Clip History to Neighborhood Variance Box
    vec3 hist_yc      = TonemapYCoCg(RGBtoYCoCg(history_rgb));
    vec3 clipped_yc   = ClipToAABB(aabbMin, aabbMax, hist_yc);
    vec3 curr_yc      = TonemapYCoCg(RGBtoYCoCg(current_rgb));

    // 7. Motion-Adaptive Temporal Blend
    float speedPx = length(velocity * res);
    float fb = mix(u_Feedback, max(u_Feedback - 0.15, 0.78), smoothstep(1.5, 6.0, speedPx));

    vec3 blended_yc = mix(curr_yc, clipped_yc, fb);
    vec3 resolved_rgb = max(YCoCgtoRGB(UntonemapYCoCg(blended_yc)), vec3(0.0));

    // 8. Contrast-Adaptive Sharpening (CAS) with Anti-Ringing Clamping
    if (u_Sharpness > 0.001) {
        vec3 n = texture(u_CurrentFrame, v_UV + vec2(0.0,  texel.y)).rgb;
        vec3 s = texture(u_CurrentFrame, v_UV - vec2(0.0,  texel.y)).rgb;
        vec3 e = texture(u_CurrentFrame, v_UV + vec2(texel.x, 0.0)).rgb;
        vec3 w = texture(u_CurrentFrame, v_UV - vec2(texel.x, 0.0)).rgb;

        vec3 minColor = min(min(min(n, s), min(e, w)), resolved_rgb);
        vec3 maxColor = max(max(max(n, s), max(e, w)), resolved_rgb);

        vec3 crossAvg = (n + s + e + w) * 0.25;
        vec3 diff = resolved_rgb - crossAvg;
        resolved_rgb = clamp(resolved_rgb + diff * (u_Sharpness * 0.35), minColor, maxColor);
    }

    out_ResolvedColor = vec4(resolved_rgb, 1.0);
}
