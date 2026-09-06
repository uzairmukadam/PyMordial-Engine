#version 450 core

in vec2 v_UV;
layout (location = 0) out float out_FilteredAO;

layout (binding = 0) uniform sampler2D u_InputAO;
layout (binding = 1) uniform sampler2D u_GBufferDepth;

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

uniform vec2 u_BlurDirection;      // e.g. vec2(1.0/w, 0.0) or vec2(0.0, 1.0/h)
uniform float u_DepthSigma = 0.08; // 8cm metric view-depth threshold for pin-sharp edge preservation

float GetLinearViewDepth(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return -view.z / max(view.w, 0.000001);
}

void main() {
    float center_ao = texture(u_InputAO, v_UV).r;
    float center_depth = texture(u_GBufferDepth, v_UV).r;

    if (center_depth <= 0.000001) {
        out_FilteredAO = 1.0;
        return;
    }

    float center_z = GetLinearViewDepth(v_UV, center_depth);

    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    float total_ao = center_ao * weights[0];
    float total_weight = weights[0];

    float two_sigma_sq = 2.0 * u_DepthSigma * u_DepthSigma;

    for (int i = 1; i <= 4; ++i) {
        float w = weights[i];
        vec2 offset = u_BlurDirection * float(i);

        // Positive direction
        vec2 uv_pos = v_UV + offset;
        float depth_pos = texture(u_GBufferDepth, uv_pos).r;
        if (depth_pos > 0.000001) {
            float z_pos = GetLinearViewDepth(uv_pos, depth_pos);
            float diff = z_pos - center_z;
            float edge_w = exp(-(diff * diff) / two_sigma_sq) * w;
            total_ao += texture(u_InputAO, uv_pos).r * edge_w;
            total_weight += edge_w;
        }

        // Negative direction
        vec2 uv_neg = v_UV - offset;
        float depth_neg = texture(u_GBufferDepth, uv_neg).r;
        if (depth_neg > 0.000001) {
            float z_neg = GetLinearViewDepth(uv_neg, depth_neg);
            float diff = z_neg - center_z;
            float edge_w = exp(-(diff * diff) / two_sigma_sq) * w;
            total_ao += texture(u_InputAO, uv_neg).r * edge_w;
            total_weight += edge_w;
        }
    }

    out_FilteredAO = total_ao / max(total_weight, 0.0001);
}
