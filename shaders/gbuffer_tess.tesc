#version 450 core

layout (vertices = 3) out;

in vec3 tc_Position[];
in vec3 tc_Normal[];
in vec2 tc_UV[];
in vec4 tc_Tangent[];
in flat uint tc_EntityID[];

out vec3 te_Position[];
out vec3 te_Normal[];
out vec2 te_UV[];
out vec4 te_Tangent[];
out flat uint te_EntityID[];

// Common Frame Data UBO (binding 0)
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

// Tessellation Uniforms
uniform int u_TessEnabled = 1;
uniform float u_TessMaxLevel = 16.0;
uniform float u_TessDistanceMin = 2.0;
uniform float u_TessDistanceMax = 30.0;

float calc_edge_level(vec3 p0, vec3 p1) {
    vec3 mid = (p0 + p1) * 0.5;
    float dist = length(mid - u_CameraPos_Time.xyz);
    float t = clamp((u_TessDistanceMax - dist) / max(u_TessDistanceMax - u_TessDistanceMin, 0.001), 0.0, 1.0);
    return mix(1.0, u_TessMaxLevel, t);
}

void main() {
    // Pass-through per-vertex attributes
    te_Position[gl_InvocationID] = tc_Position[gl_InvocationID];
    te_Normal[gl_InvocationID] = tc_Normal[gl_InvocationID];
    te_UV[gl_InvocationID] = tc_UV[gl_InvocationID];
    te_Tangent[gl_InvocationID] = tc_Tangent[gl_InvocationID];
    te_EntityID[gl_InvocationID] = tc_EntityID[gl_InvocationID];

    // Compute dynamic tessellation levels on first invocation
    if (gl_InvocationID == 0) {
        if (u_TessEnabled == 1) {
            float l0 = calc_edge_level(tc_Position[1], tc_Position[2]);
            float l1 = calc_edge_level(tc_Position[2], tc_Position[0]);
            float l2 = calc_edge_level(tc_Position[0], tc_Position[1]);

            gl_TessLevelOuter[0] = l0;
            gl_TessLevelOuter[1] = l1;
            gl_TessLevelOuter[2] = l2;
            gl_TessLevelInner[0] = (l0 + l1 + l2) * 0.333333;
        } else {
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = 1.0;
            gl_TessLevelOuter[2] = 1.0;
            gl_TessLevelInner[0] = 1.0;
        }
    }
}
