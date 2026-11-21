#version 450 core
#extension GL_ARB_shader_draw_parameters : enable

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

out vec3 tc_Position;
out vec3 tc_Normal;
out vec2 tc_UV;
out flat uint tc_EntityID;

uniform uint u_BaseInstance;

void main() {
#ifdef GL_ARB_shader_draw_parameters
    uint entity_idx = uint(gl_BaseInstanceARB) + uint(gl_InstanceID);
#else
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
#endif
    tc_EntityID = entity_idx;

    mat4 model = u_WorldTransforms[entity_idx];
    // Inverse-transpose so non-uniform scale does not skew shadow-caster normals.
    mat3 normal_matrix = transpose(inverse(mat3(model)));

    tc_Position = (model * vec4(in_position, 1.0)).xyz;
    tc_Normal = normalize(normal_matrix * in_normal);
    tc_UV = in_uv;
}
