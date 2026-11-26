#version 450 core
// ============================================================================
//  SSRT AO — Visibility Reconstruction
//
//  Upsamples the shared local-diffuse visibility (produced by the AMRGI trace at
//  trace resolution) to the AO target resolution with a depth-aware bilateral
//  gather, so AO does not bleed across silhouettes or from background onto
//  foreground. The result is written to the AO raw target and then filtered by
//  the existing bilateral blur, keeping the standard ao_texture contract.
//
//  This shader performs NO ray traversal. It only reconstructs the visibility
//  already accumulated by the diffuse SSRT workload.
// ============================================================================

in vec2 v_UV;
layout (location = 0) out float out_Visibility;

layout (binding = 0) uniform sampler2D u_Visibility;      // trace-res raw visibility (R)
layout (binding = 1) uniform sampler2D u_GBufferDepth;    // full-res depth
layout (binding = 2) uniform sampler2D u_GBufferNormalMetallic;

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

uniform vec2 u_TraceResolution = vec2(1.0);  // resolution of u_Visibility
uniform float u_DepthSigma = 0.10;           // metric view-depth tolerance (m)

float LinearizeViewDepth(float raw_depth, vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return -view.z / max(abs(view.w), 1e-6);
}

void main() {
    float center_raw = texture(u_GBufferDepth, v_UV).r;
    if (center_raw <= 0.000001) {
        out_Visibility = 1.0;   // background is fully open
        return;
    }

    float center_z = LinearizeViewDepth(center_raw, v_UV);
    float sigma = max(u_DepthSigma, center_z * 0.02);
    float two_sigma_sq = 2.0 * sigma * sigma;

    // Gather a 3x3 neighbourhood in trace-texel space around the matching
    // trace-resolution location, weighting each tap by depth similarity.
    vec2 trace_texel = 1.0 / max(u_TraceResolution, vec2(1.0));
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 uv_s = v_UV + vec2(float(x), float(y)) * trace_texel;
            float raw = texture(u_GBufferDepth, uv_s).r;
            if (raw <= 0.000001) {
                continue;
            }
            float z = LinearizeViewDepth(raw, uv_s);
            float diff = z - center_z;
            float w = exp(-(diff * diff) / two_sigma_sq);
            sum += texture(u_Visibility, uv_s).r * w;
            wsum += w;
        }
    }

    out_Visibility = wsum > 1e-5 ? clamp(sum / wsum, 0.0, 1.0) : 1.0;
}
