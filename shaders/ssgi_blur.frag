#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FilteredSSGI;

layout (binding = 0) uniform sampler2D u_InputSSGI;
layout (binding = 1) uniform sampler2D u_GBufferDepth;
layout (binding = 2) uniform sampler2D u_GBufferNormalMetallic;

uniform vec2 u_BlurDirection;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

void main() {
    vec4 center_ssgi = texture(u_InputSSGI, v_UV);
    float center_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_depth <= 0.000001) {
        out_FilteredSSGI = vec4(0.0);
        return;
    }

    vec3 center_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);

    const float weights[4] = float[](0.35, 0.25, 0.12, 0.05);
    vec4 total_color = center_ssgi * weights[0];
    float total_weight = weights[0];

    for (int i = 1; i <= 3; ++i) {
        float w = weights[i];
        vec2 offset = u_BlurDirection * float(i) * 1.5;

        // Sample positive offset
        vec2 uv_pos = v_UV + offset;
        float depth_pos = texture(u_GBufferDepth, uv_pos).r;
        vec3 normal_pos = OctahedralDecode(texture(u_GBufferNormalMetallic, uv_pos).rg);
        float norm_match_pos = max(dot(center_normal, normal_pos), 0.0);
        float depth_weight_pos = exp(-abs(depth_pos - center_depth) * 80.0) * pow(norm_match_pos, 4.0) * w;

        total_color += texture(u_InputSSGI, uv_pos) * depth_weight_pos;
        total_weight += depth_weight_pos;

        // Sample negative offset
        vec2 uv_neg = v_UV - offset;
        float depth_neg = texture(u_GBufferDepth, uv_neg).r;
        vec3 normal_neg = OctahedralDecode(texture(u_GBufferNormalMetallic, uv_neg).rg);
        float norm_match_neg = max(dot(center_normal, normal_neg), 0.0);
        float depth_weight_neg = exp(-abs(depth_neg - center_depth) * 80.0) * pow(norm_match_neg, 4.0) * w;

        total_color += texture(u_InputSSGI, uv_neg) * depth_weight_neg;
        total_weight += depth_weight_neg;
    }

    out_FilteredSSGI = total_color / max(total_weight, 0.0001);
}
