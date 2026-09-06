#version 450 core

in vec2 v_UV;
layout (location = 0) out float out_FilteredAO;

layout (binding = 0) uniform sampler2D u_InputAO;
layout (binding = 1) uniform sampler2D u_GBufferDepth;

uniform vec2 u_BlurDirection; // e.g. vec2(1.0/w, 0.0) or vec2(0.0, 1.0/h)
uniform float u_DepthThreshold = 0.05;

void main() {
    float center_ao = texture(u_InputAO, v_UV).r;
    float center_depth = texture(u_GBufferDepth, v_UV).r;

    if (center_depth <= 0.000001) {
        out_FilteredAO = 1.0;
        return;
    }

    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    float total_ao = center_ao * weights[0];
    float total_weight = weights[0];

    for (int i = 1; i <= 4; ++i) {
        float w = weights[i];
        vec2 offset = u_BlurDirection * float(i);

        // Positive direction
        vec2 uv_pos = v_UV + offset;
        float depth_pos = texture(u_GBufferDepth, uv_pos).r;
        float depth_diff_pos = abs(depth_pos - center_depth);
        float edge_weight_pos = exp(-depth_diff_pos * 100.0) * w;
        total_ao += texture(u_InputAO, uv_pos).r * edge_weight_pos;
        total_weight += edge_weight_pos;

        // Negative direction
        vec2 uv_neg = v_UV - offset;
        float depth_neg = texture(u_GBufferDepth, uv_neg).r;
        float depth_diff_neg = abs(depth_neg - center_depth);
        float edge_weight_neg = exp(-depth_diff_neg * 100.0) * w;
        total_ao += texture(u_InputAO, uv_neg).r * edge_weight_neg;
        total_weight += edge_weight_neg;
    }

    out_FilteredAO = total_ao / max(total_weight, 0.0001);
}
