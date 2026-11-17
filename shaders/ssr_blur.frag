#version 450 core

#include "shaders/common/blur.glsl"

in vec2 v_UV;
layout (location = 0) out vec4 out_FilteredSSR;

layout (binding = 0) uniform sampler2D u_InputSSR;
layout (binding = 1) uniform sampler2D u_GBufferDepth;
layout (binding = 2) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 3) uniform sampler2D u_GBufferNormalMetallic;

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

void main() {
    vec4 center_ssr = texture(u_InputSSR, v_UV);
    float center_depth = texture(u_GBufferDepth, v_UV).r;

    if (center_depth <= 0.000001) {
        out_FilteredSSR = vec4(0.0);
        return;
    }

    float roughness = texture(u_GBufferAlbedoRoughness, v_UV).a;

    // Keep near-mirror surfaces sharp and avoid filling misses from adjacent reflection rays.
    if (roughness < 0.04) {
        out_FilteredSSR = center_ssr;
        return;
    }

    // Roughness-scaled kernel radius: glossy surfaces get smooth specular spread
    float kernel_radius = clamp(roughness * 6.0 + 1.0, 1.0, 6.0);

    vec4 result = BilateralBlurDepthNormal(
        u_InputSSR, u_GBufferDepth, u_GBufferNormalMetallic,
        v_UV, u_BlurDirection, kernel_radius, u_DepthSigma, 16.0, u_InvProjection,
        BLUR_USE_NORMAL | BLUR_WEIGHT_ALPHA | BLUR_CLAMP_UV | BLUR_ALPHA_BY_KERNEL,
        mat3(1.0)
    );

    if (isnan(result.r) || isnan(result.g) || isnan(result.b) || isnan(result.a) ||
        isinf(result.r) || isinf(result.g) || isinf(result.b) || isinf(result.a)) {
        result = vec4(0.0);
    }
    out_FilteredSSR = clamp(result, vec4(0.0), vec4(40.0, 40.0, 40.0, 1.0));
}
