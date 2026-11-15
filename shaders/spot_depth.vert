#version 450 core
#extension GL_ARB_shader_draw_parameters : enable

layout (location = 0) in vec3 in_position;

// SSBO 1: World Transforms
layout (std430, binding = 1) readonly buffer TransformBuffer {
    mat4 u_WorldTransforms[];
};

uniform mat4 u_SpotLightViewProjection;
uniform uint u_BaseInstance;

void main() {
#ifdef GL_ARB_shader_draw_parameters
    uint entity_idx = uint(gl_BaseInstanceARB) + uint(gl_InstanceID);
#else
    uint entity_idx = u_BaseInstance + uint(gl_InstanceID);
#endif
    mat4 model = u_WorldTransforms[entity_idx];
    gl_Position = u_SpotLightViewProjection * (model * vec4(in_position, 1.0));
}
