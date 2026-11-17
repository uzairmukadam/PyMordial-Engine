#version 450 core

#include "shaders/common/blur.glsl"

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

void main() {
    float center_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_depth <= 0.000001) {
        out_FilteredAO = 1.0;
        return;
    }

    // Scale depth threshold with view distance to preserve geometric edges while correctly
    // filtering perspective planar gradients across distant surfaces
    float center_z = LinearizeViewDepth(center_depth, v_UV, u_InvProjection);
    float depth_threshold = max(u_DepthSigma, center_z * 0.035);

    out_FilteredAO = BilateralBlurDepth(
        u_InputAO, u_GBufferDepth, v_UV, u_BlurDirection, 1.0, depth_threshold, u_InvProjection
    ).r;
}
