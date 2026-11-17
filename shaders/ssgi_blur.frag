#version 450 core

#include "shaders/common/blur.glsl"

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

void main() {
    float center_raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_raw_depth <= 0.000001) {
        out_FilteredSSGI = vec4(0.0);
        return;
    }

    // Distance-adaptive tangent plane tolerance prevents neighbor rejection at camera distance
    float center_z = LinearizeViewDepth(center_raw_depth, v_UV, u_InvProjection);
    float plane_sigma = max(0.12, abs(center_z) * 0.025);

    vec4 result = BilateralBlurDepthNormal(
        u_InputSSGI, u_GBufferDepth, u_GBufferNormalMetallic,
        v_UV, u_BlurDirection, 2.0, plane_sigma, 2.0, u_InvProjection,
        BLUR_USE_NORMAL | BLUR_METRIC_PLANE,
        mat3(u_View)
    );

    if (isnan(result.r) || isnan(result.g) || isnan(result.b) || isnan(result.a) ||
        isinf(result.r) || isinf(result.g) || isinf(result.b) || isinf(result.a)) {
        result = vec4(0.0);
    }
    out_FilteredSSGI = clamp(result, vec4(0.0), vec4(40.0, 40.0, 40.0, 1.0));
}
