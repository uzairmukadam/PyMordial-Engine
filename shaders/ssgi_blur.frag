#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FilteredSSGI;

layout (binding = 0) uniform sampler2D u_InputSSGI;
layout (binding = 1) uniform sampler2D u_GBufferDepth;
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

uniform vec2 u_BlurDirection;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 GetViewPos(vec2 uv, float raw_depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / view.w;
}

void main() {
    vec4 center_ssgi = texture(u_InputSSGI, v_UV);
    float center_raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_raw_depth <= 0.000001) {
        out_FilteredSSGI = vec4(0.0);
        return;
    }

    vec3 center_pos_view = GetViewPos(v_UV, center_raw_depth);
    vec3 center_normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 center_normal_view = normalize(mat3(u_View) * center_normal_world);

    // 5-tap Gaussian kernel weights
    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec4 total_color = center_ssgi * weights[0];
    float total_weight = weights[0];

    // Tangent plane tolerance: 8cm along surface normal
    const float PLANE_SIGMA_SQ2 = 2.0 * 0.08 * 0.08;

    for (int i = 1; i <= 4; ++i) {
        float w = weights[i];
        vec2 offset = u_BlurDirection * (float(i) * 1.5);

        // Positive offset
        vec2 uv_pos = v_UV + offset;
        float depth_raw_pos = texture(u_GBufferDepth, uv_pos).r;
        if (depth_raw_pos > 0.000001) {
            vec3 pos_pos = GetViewPos(uv_pos, depth_raw_pos);
            vec3 norm_world_pos = OctahedralDecode(texture(u_GBufferNormalMetallic, uv_pos).rg);
            vec3 norm_view_pos = normalize(mat3(u_View) * norm_world_pos);

            // Plane distance test: distance of neighbor point to center surface tangent plane
            float plane_dist = abs(dot(center_normal_view, pos_pos - center_pos_view));
            float w_z = exp(-(plane_dist * plane_dist) / PLANE_SIGMA_SQ2);
            float w_n = pow(max(dot(center_normal_view, norm_view_pos), 0.0), 4.0);
            float weight_pos = w * w_z * w_n;

            total_color += texture(u_InputSSGI, uv_pos) * weight_pos;
            total_weight += weight_pos;
        }

        // Negative offset
        vec2 uv_neg = v_UV - offset;
        float depth_raw_neg = texture(u_GBufferDepth, uv_neg).r;
        if (depth_raw_neg > 0.000001) {
            vec3 pos_neg = GetViewPos(uv_neg, depth_raw_neg);
            vec3 norm_world_neg = OctahedralDecode(texture(u_GBufferNormalMetallic, uv_neg).rg);
            vec3 norm_view_neg = normalize(mat3(u_View) * norm_world_neg);

            float plane_dist = abs(dot(center_normal_view, pos_neg - center_pos_view));
            float w_z = exp(-(plane_dist * plane_dist) / PLANE_SIGMA_SQ2);
            float w_n = pow(max(dot(center_normal_view, norm_view_neg), 0.0), 4.0);
            float weight_neg = w * w_z * w_n;

            total_color += texture(u_InputSSGI, uv_neg) * weight_neg;
            total_weight += weight_neg;
        }
    }

    out_FilteredSSGI = total_color / max(total_weight, 0.0001);
}
