#version 450 core

// Per-vertex geometry from Mega-VBO
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_tangent;

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

// SSBO 1: World Transforms (mat4 per entity)
layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

// SSBO 2: Material Data (two vec4 per entity)
layout (std430, binding = 2) readonly buffer MaterialBuffer {
    vec4 u_MaterialData[];
};

// Outputs to G-Buffer Fragment
out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_UV;
out vec4 v_CurrClip;
out vec4 v_PrevClip;
out flat uint v_EntityID;

uniform uint u_BaseInstance; // Set per draw batch or via indirect command
uniform mat4 u_PrevViewProjection;

void main() {
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
    v_EntityID = entity_idx;

    mat4 model = u_WorldTransforms[entity_idx];
    mat3 normal_matrix = mat3(model); // Assumes uniform scale or orthogonal

    vec4 world_pos = model * vec4(in_position, 1.0);
    v_WorldPos = world_pos.xyz;
    v_Normal = normalize(normal_matrix * in_normal);
    v_UV = in_uv;

    v_CurrClip = u_ViewProjection * world_pos;
    v_PrevClip = (u_PrevViewProjection[3][3] != 0.0) ? (u_PrevViewProjection * world_pos) : v_CurrClip;

    gl_Position = v_CurrClip;
}
