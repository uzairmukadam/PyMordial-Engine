#version 450 core

layout (vertices = 3) out;

in vec3 tc_Position[];
in vec3 tc_Normal[];
in vec2 tc_UV[];
in flat uint tc_EntityID[];

out vec3 te_Position[];
out vec3 te_Normal[];
out vec2 te_UV[];
out flat uint te_EntityID[];

uniform float u_TessMaxLevel = 8.0;

void main() {
    te_Position[gl_InvocationID] = tc_Position[gl_InvocationID];
    te_Normal[gl_InvocationID] = tc_Normal[gl_InvocationID];
    te_UV[gl_InvocationID] = tc_UV[gl_InvocationID];
    te_EntityID[gl_InvocationID] = tc_EntityID[gl_InvocationID];

    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = u_TessMaxLevel;
        gl_TessLevelOuter[1] = u_TessMaxLevel;
        gl_TessLevelOuter[2] = u_TessMaxLevel;
        gl_TessLevelInner[0] = u_TessMaxLevel;
    }
}
