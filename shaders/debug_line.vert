#version 450 core

layout(std140, binding = 0) uniform FrameContext {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;
    vec3 u_CameraPos;
    float u_Time;
    vec2 u_ScreenSize;
    vec2 u_Padding0;
    vec3 u_SunDirection;
    float u_SunIntensity;
    mat4 u_CSM_Matrices[4];
    vec4 u_CSM_Splits;
    float u_FogDensity;
    float u_FogHeightFalloff;
    vec2 u_Padding1;
};

in vec3 in_position;
in vec4 in_color;

out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_ViewProjection * vec4(in_position, 1.0);
}
