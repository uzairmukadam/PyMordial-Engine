#version 450 core

in vec2 v_UV;
layout (location = 0) out vec2 out_Edges;

layout (binding = 0) uniform sampler2D u_ColorTexture;
uniform vec2 u_InverseScreenSize;
uniform float u_EdgeThreshold = 0.08; // SMAA edge threshold (0.05 to 0.15)

float SmaaLuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    // Sample center and immediate neighbors (left, top, right, bottom)
    vec3 c_c = texture(u_ColorTexture, v_UV).rgb;
    vec3 c_l = texture(u_ColorTexture, v_UV - vec2(u_InverseScreenSize.x, 0.0)).rgb;
    vec3 c_t = texture(u_ColorTexture, v_UV - vec2(0.0, u_InverseScreenSize.y)).rgb;
    vec3 c_r = texture(u_ColorTexture, v_UV + vec2(u_InverseScreenSize.x, 0.0)).rgb;
    vec3 c_b = texture(u_ColorTexture, v_UV + vec2(0.0, u_InverseScreenSize.y)).rgb;

    float l_c = SmaaLuma(c_c);
    float l_l = SmaaLuma(c_l);
    float l_t = SmaaLuma(c_t);
    float l_r = SmaaLuma(c_r);
    float l_b = SmaaLuma(c_b);

    // Local contrast adaptation
    float delta_l = abs(l_c - l_l);
    float delta_t = abs(l_c - l_t);
    float delta_r = abs(l_c - l_r);
    float delta_b = abs(l_c - l_b);

    float max_delta = max(max(delta_l, delta_t), max(delta_r, delta_b));
    float adapted_threshold = max(u_EdgeThreshold * 0.5, max_delta * 0.25);

    vec2 edges = vec2(0.0);
    if (delta_l > adapted_threshold) {
        edges.x = 1.0; // Horizontal edge (left-right boundary)
    }
    if (delta_t > adapted_threshold) {
        edges.y = 1.0; // Vertical edge (top-bottom boundary)
    }

    out_Edges = edges;
}
