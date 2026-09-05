#version 450 core

layout (location = 0) in vec3 in_position;

// UBO 0: Frame Data
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

// SSBO 1: World Transforms
layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

uniform uint u_CascadeIndex;
uniform uint u_BaseInstance;

void main() {
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
    mat4 model = u_WorldTransforms[entity_idx];
    gl_Position = u_LightViewProjection[u_CascadeIndex] * (model * vec4(in_position, 1.0));
}
