#version 450 core
// ============================================================================
//  SSR Temporal Stabilization
//
//  Reprojects the previous frame's accumulated SSR radiance/confidence using the
//  G-buffer motion vectors, validates the history against current-frame geometry,
//  clamps it to the current 3x3 neighbourhood, and blends with a confidence-aware
//  exponential moving average.
//
//  Output contract (unchanged): RGB = reflected radiance, A = confidence.
//
//  Reprojection: velocity is defined in gbuffer.frag as (rendered_curr_uv -
//  nominal_prev_uv), where rendered_curr_uv already includes the projection
//  jitter. Therefore the previous-frame nominal UV is simply v_UV - velocity.
// ============================================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_SSR;

layout (binding = 0) uniform sampler2D u_CurrentSSR;      // this frame's raw trace
layout (binding = 1) uniform sampler2D u_HistorySSR;      // previous accumulated result
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 4) uniform sampler2D u_GBufferSurfaceID;
layout (binding = 5) uniform sampler2D u_VelocityTexture;
layout (binding = 6) uniform sampler2D u_GBufferAlbedoRoughness;

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

uniform int   u_HistoryValid = 0;        // 0 on first frame / after reset
uniform float u_HistoryWeight = 0.90;    // base EMA history weight (mirror)
uniform float u_HistoryWeightRough = 0.96; // EMA history weight for rough surfaces
uniform float u_DepthThreshold = 0.05;   // relative linear-depth tolerance
uniform float u_NormalThreshold = 0.90;  // min dot(normal_hist, normal_curr)
uniform vec2  u_SSRResolution = vec2(1.0);

#include "shaders/common/depth.glsl"
#include "shaders/common/pbr_brdf.glsl"

float LinearDepth(vec2 uv) {
    float raw = texture(u_GBufferDepth, uv).r;
    if (raw <= 0.000001) return -1.0;
    return -ReconstructViewPosition(uv, raw, u_InvProjection).z;
}

void main() {
    vec4 current = texture(u_CurrentSSR, v_UV);

    // No history available: pass the current sample straight through.
    if (u_HistoryValid == 0) {
        out_SSR = current;
        return;
    }

    float cur_depth = LinearDepth(v_UV);
    if (cur_depth < 0.0) {
        // Background: nothing to stabilize.
        out_SSR = current;
        return;
    }

    vec2 velocity = texture(u_VelocityTexture, v_UV).xy;
    vec2 hist_uv = v_UV - velocity;

    // ---- History validation ----
    bool valid = true;
    if (hist_uv.x < 0.0 || hist_uv.x > 1.0 || hist_uv.y < 0.0 || hist_uv.y > 1.0) {
        valid = false;
    }

    if (valid) {
        float hist_depth = LinearDepth(hist_uv);
        if (hist_depth < 0.0) {
            valid = false;
        } else {
            float rel = abs(hist_depth - cur_depth) / max(cur_depth, 1e-3);
            if (rel > u_DepthThreshold) valid = false;
        }
    }

    if (valid) {
        vec3 n_cur = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
        vec3 n_hist = OctahedralDecode(texture(u_GBufferNormalMetallic, hist_uv).rg);
        if (dot(n_cur, n_hist) < u_NormalThreshold) valid = false;
    }

    if (valid) {
        float id_cur = texture(u_GBufferSurfaceID, v_UV).r;
        float id_hist = texture(u_GBufferSurfaceID, hist_uv).r;
        if (abs(id_cur - id_hist) > 0.5) valid = false;
    }

    if (!valid) {
        out_SSR = current;
        return;
    }

    vec4 history = texture(u_HistorySSR, hist_uv);

    // ---- 3x3 neighbourhood clamp of the current trace (prevents ghosting) ----
    vec2 texel = 1.0 / max(u_SSRResolution, vec2(1.0));
    vec3 nmin = current.rgb;
    vec3 nmax = current.rgb;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec3 s = texture(u_CurrentSSR, v_UV + vec2(float(x), float(y)) * texel).rgb;
            nmin = min(nmin, s);
            nmax = max(nmax, s);
        }
    }
    vec3 clamped_hist = clamp(history.rgb, nmin, nmax);

    // ---- Confidence-aware EMA ----
    float roughness = texture(u_GBufferAlbedoRoughness, v_UV).a;
    float w_hist = mix(u_HistoryWeight, u_HistoryWeightRough, smoothstep(0.0, 0.5, roughness));
    // If the current sample is low-confidence, lean on history; if history is
    // low-confidence, lean on the current sample.
    w_hist *= clamp(history.a, 0.0, 1.0);
    w_hist = clamp(w_hist, 0.0, 0.98);

    vec3 out_rgb = mix(current.rgb, clamped_hist, w_hist);
    float out_a = mix(current.a, history.a, w_hist);

    if (isnan(out_rgb.r) || isnan(out_rgb.g) || isnan(out_rgb.b) || isnan(out_a) ||
        isinf(out_rgb.r) || isinf(out_rgb.g) || isinf(out_rgb.b) || isinf(out_a)) {
        out_rgb = current.rgb;
        out_a = current.a;
    }
    out_SSR = vec4(clamp(out_rgb, vec3(0.0), vec3(40.0)), clamp(out_a, 0.0, 1.0));
}
