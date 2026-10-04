#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FilteredSSR;

layout (binding = 0) uniform sampler2D u_InputSSR;
layout (binding = 1) uniform sampler2D u_GBufferDepth;
layout (binding = 2) uniform sampler2D u_GBufferAlbedoRoughness;

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
uniform float u_DepthSigma = 0.10; // 10cm metric view-depth threshold for edge preservation

float GetLinearViewDepth(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return -view.z / max(abs(view.w), 1e-6);
}

void main() {
    vec4 center_ssr = texture(u_InputSSR, v_UV);
    float center_depth = texture(u_GBufferDepth, v_UV).r;

    if (center_depth <= 0.000001 || center_ssr.a <= 0.0001) {
        out_FilteredSSR = vec4(0.0);
        return;
    }

    float roughness = texture(u_GBufferAlbedoRoughness, v_UV).a;

    // For pure mirror surfaces (roughness < 0.04), keep reflections pin-sharp with zero blur
    if (roughness < 0.04) {
        out_FilteredSSR = center_ssr;
        return;
    }

    float center_z = GetLinearViewDepth(v_UV, center_depth);

    // Roughness-scaled kernel radius: glossy surfaces get smooth specular spread
    float kernel_radius = clamp(roughness * 12.0 + 1.5, 2.0, 8.0);

    const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec4 total_ssr = center_ssr * weights[0];
    float total_weight = weights[0];

    float two_sigma_sq = 2.0 * u_DepthSigma * u_DepthSigma;

    for (int i = 1; i <= 4; ++i) {
        float w = weights[i];
        vec2 offset = u_BlurDirection * (float(i) * kernel_radius);

        // Positive direction
        vec2 uv_pos = v_UV + offset;
        if (uv_pos.x >= 0.0 && uv_pos.x <= 1.0 && uv_pos.y >= 0.0 && uv_pos.y <= 1.0) {
            float depth_pos = texture(u_GBufferDepth, uv_pos).r;
            if (depth_pos > 0.000001) {
                vec4 s_pos = texture(u_InputSSR, uv_pos);
                if (s_pos.a > 0.0001) {
                    float z_pos = GetLinearViewDepth(uv_pos, depth_pos);
                    float diff = z_pos - center_z;
                    float edge_w = exp(-(diff * diff) / two_sigma_sq) * w;
                    total_ssr += s_pos * edge_w;
                    total_weight += edge_w;
                }
            }
        }

        // Negative direction
        vec2 uv_neg = v_UV - offset;
        if (uv_neg.x >= 0.0 && uv_neg.x <= 1.0 && uv_neg.y >= 0.0 && uv_neg.y <= 1.0) {
            float depth_neg = texture(u_GBufferDepth, uv_neg).r;
            if (depth_neg > 0.000001) {
                vec4 s_neg = texture(u_InputSSR, uv_neg);
                if (s_neg.a > 0.0001) {
                    float z_neg = GetLinearViewDepth(uv_neg, depth_neg);
                    float diff = z_neg - center_z;
                    float edge_w = exp(-(diff * diff) / two_sigma_sq) * w;
                    total_ssr += s_neg * edge_w;
                    total_weight += edge_w;
                }
            }
        }
    }

    vec4 result = total_ssr / max(total_weight, 0.0001);
    if (isnan(result.r) || isnan(result.g) || isnan(result.b) || isnan(result.a) ||
        isinf(result.r) || isinf(result.g) || isinf(result.b) || isinf(result.a)) {
        result = vec4(0.0);
    }
    out_FilteredSSR = clamp(result, vec4(0.0), vec4(40.0, 40.0, 40.0, 1.0));
}
