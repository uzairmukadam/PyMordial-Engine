#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_Weights; // (left, right, top, bottom)

layout (binding = 0) uniform sampler2D u_EdgeTexture;
uniform vec2 u_InverseScreenSize;
uniform int u_MaxSearchSteps = 16;

// Searches along horizontal edge for left and right endpoints
vec2 SearchHorizontal(vec2 uv, int max_steps) {
    float d_left = 0.0;
    float d_right = 0.0;

    // Search left
    vec2 p_left = uv - vec2(u_InverseScreenSize.x, 0.0);
    for (int i = 0; i < max_steps; ++i) {
        float e = texture(u_EdgeTexture, p_left).y;
        if (e < 0.5) break;
        d_left += 1.0;
        p_left.x -= u_InverseScreenSize.x;
    }

    // Search right
    vec2 p_right = uv + vec2(u_InverseScreenSize.x, 0.0);
    for (int i = 0; i < max_steps; ++i) {
        float e = texture(u_EdgeTexture, p_right).y;
        if (e < 0.5) break;
        d_right += 1.0;
        p_right.x += u_InverseScreenSize.x;
    }

    return vec2(d_left, d_right);
}

// Searches along vertical edge for up and down endpoints
vec2 SearchVertical(vec2 uv, int max_steps) {
    float d_up = 0.0;
    float d_down = 0.0;

    // Search up
    vec2 p_up = uv - vec2(0.0, u_InverseScreenSize.y);
    for (int i = 0; i < max_steps; ++i) {
        float e = texture(u_EdgeTexture, p_up).x;
        if (e < 0.5) break;
        d_up += 1.0;
        p_up.y -= u_InverseScreenSize.y;
    }

    // Search down
    vec2 p_down = uv + vec2(0.0, u_InverseScreenSize.y);
    for (int i = 0; i < max_steps; ++i) {
        float e = texture(u_EdgeTexture, p_down).x;
        if (e < 0.5) break;
        d_down += 1.0;
        p_down.y += u_InverseScreenSize.y;
    }

    return vec2(d_up, d_down);
}

// Analytical trapezoidal subpixel area coverage
vec2 CalculateArea(vec2 dist) {
    float d1 = dist.x;
    float d2 = dist.y;
    float total = d1 + d2 + 1.0;

    // Smooth trapezoidal falloff
    float a1 = clamp(0.5 * (1.0 - (d1 + 0.5) / total), 0.0, 0.5);
    float a2 = clamp(0.5 * (1.0 - (d2 + 0.5) / total), 0.0, 0.5);
    return vec2(a1, a2);
}

void main() {
    vec2 e = texture(u_EdgeTexture, v_UV).xy;
    vec4 weights = vec4(0.0);

    // Horizontal edge above current pixel (e.y)
    if (e.y > 0.0) {
        vec2 dist_h = SearchHorizontal(v_UV, u_MaxSearchSteps);
        vec2 area_h = CalculateArea(dist_h);
        weights.z = area_h.x; // Top-left weight
        weights.w = area_h.y; // Top-right weight
    }

    // Vertical edge to the left of current pixel (e.x)
    if (e.x > 0.0) {
        vec2 dist_v = SearchVertical(v_UV, u_MaxSearchSteps);
        vec2 area_v = CalculateArea(dist_v);
        weights.x = area_v.x; // Left-up weight
        weights.y = area_v.y; // Left-down weight
    }

    out_Weights = weights;
}
