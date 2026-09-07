#version 450 core

// Per-vertex geometry from Mega-VBO for Tessellation Patches
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_tangent;

// SSBO 1: World Transforms
layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

// Outputs to Tessellation Control Shader
out vec3 tc_Position;
out vec3 tc_Normal;
out vec2 tc_UV;
out vec4 tc_Tangent;
out flat uint tc_EntityID;

uniform uint u_BaseInstance;

void main() {
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
    tc_EntityID = entity_idx;

    mat4 model = u_WorldTransforms[entity_idx];
    mat3 normal_matrix = mat3(model);

    vec4 world_pos = model * vec4(in_position, 1.0);
    tc_Position = world_pos.xyz;
    tc_Normal = normalize(normal_matrix * in_normal);
    tc_UV = in_uv;
    tc_Tangent = vec4(normalize(normal_matrix * in_tangent.xyz), in_tangent.w);
}
