#version 450 core
#extension GL_ARB_shader_draw_parameters : enable

// Per-vertex geometry from Mega-VBO for Tessellation Patches
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_tangent;

// SSBO 1: World Transforms
layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

// SSBO 7: Previous-frame world transforms (see gbuffer.vert).
layout (std430, binding = 7) readonly buffer PrevTransformBuffer {
    mat4 u_PrevWorldTransforms[];
};

// Outputs to Tessellation Control Shader
out vec3 tc_Position;
out vec3 tc_Normal;
out vec2 tc_UV;
out vec2 tc_UVScale;
out vec4 tc_Tangent;
out vec3 tc_PrevPosition;
out flat uint tc_EntityID;

uniform uint u_BaseInstance;
uniform int u_ObjectMotionVectors = 1;

void main() {
#ifdef GL_ARB_shader_draw_parameters
    uint entity_idx = uint(gl_BaseInstanceARB) + uint(gl_InstanceID);
#else
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
#endif
    tc_EntityID = entity_idx;

    mat4 model = u_WorldTransforms[entity_idx];
    mat3 normal_matrix = mat3(model);

    vec4 world_pos = model * vec4(in_position, 1.0);
    tc_Position = world_pos.xyz;
    // Previous-frame world position, before displacement. The evaluation shader
    // adds the same displacement to both so the motion vector stays consistent.
    tc_PrevPosition = (u_ObjectMotionVectors == 1)
        ? (u_PrevWorldTransforms[entity_idx] * vec4(in_position, 1.0)).xyz
        : world_pos.xyz;
    tc_Normal = normalize(normal_matrix * in_normal);
    tc_UV = in_uv;
    vec3 local_tangent = normalize(in_tangent.xyz);
    vec3 local_bitangent = cross(in_normal, local_tangent) * in_tangent.w;
    tc_UVScale = vec2(
        length(mat3(model) * local_tangent),
        length(mat3(model) * local_bitangent)
    );
    tc_Tangent = vec4(normalize(normal_matrix * in_tangent.xyz), in_tangent.w);
}
