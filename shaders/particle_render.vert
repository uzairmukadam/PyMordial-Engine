#version 430 core

#include "common/frame_data.glsl"

layout(location = 0) in vec2 in_position; // Unit quad vertex [-1, 1]

struct Particle {
    vec4 pos_life;      // xyz = world position, w = current lifetime remaining (s)
    vec4 vel_maxlife;   // xyz = velocity, w = initial max lifetime (s)
    vec4 color_alpha;   // rgb = base tint, a = target opacity
    vec4 params;        // x = size/radius, y = rotation angle (rad), z = rotation speed, w = mode
};

layout(std430, binding = 4) readonly buffer ParticleBuffer {
    Particle particles[];
};

out vec2  v_UV;
out vec4  v_Color;
out float v_LifeFraction;
out vec3  v_WorldPos;
out float v_ViewDepth;
flat out int v_Mode;

void main() {
    Particle p = particles[gl_InstanceID];

    // Cull inactive or dead particles
    if (p.pos_life.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }

    // Camera billboard basis vectors
    vec3 camRight = normalize(vec3(u_View[0][0], u_View[1][0], u_View[2][0]));
    vec3 camUp    = normalize(vec3(u_View[0][1], u_View[1][1], u_View[2][1]));

    // 2D particle rotation
    float cosR = cos(p.params.y);
    float sinR = sin(p.params.y);
    vec2 rotPos = vec2(
        in_position.x * cosR - in_position.y * sinR,
        in_position.x * sinR + in_position.y * cosR
    );

    float radius = p.params.x;
    vec3 worldPos = p.pos_life.xyz + (camRight * rotPos.x + camUp * rotPos.y) * radius;

    vec4 viewPos = u_View * vec4(worldPos, 1.0);

    v_UV = in_position; // [-1, 1] for circular distance test
    v_Color = p.color_alpha;
    v_LifeFraction = clamp(p.pos_life.w / max(p.vel_maxlife.w, 0.001), 0.0, 1.0);
    v_WorldPos = worldPos;
    v_ViewDepth = -viewPos.z;
    v_Mode = int(p.params.w);

    gl_Position = u_Projection * viewPos;
}
