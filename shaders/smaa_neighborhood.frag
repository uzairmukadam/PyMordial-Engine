#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_Color;

layout (binding = 0) uniform sampler2D u_ColorTexture;
layout (binding = 1) uniform sampler2D u_BlendTexture;
uniform vec2 u_InverseScreenSize;

void main() {
    // Current pixel's weights: x=left, y=right, z=top, w=bottom
    vec4 a = texture(u_BlendTexture, v_UV);

    // Right neighbor's left weight
    float w_right = texture(u_BlendTexture, v_UV + vec2(u_InverseScreenSize.x, 0.0)).x;
    // Bottom neighbor's top weight
    float w_bottom = texture(u_BlendTexture, v_UV + vec2(0.0, u_InverseScreenSize.y)).z;

    vec4 color = texture(u_ColorTexture, v_UV);

    // Compute blended color from neighbors
    vec4 sum = color;
    float total_weight = 1.0;

    // Left blend
    if (a.x > 0.0) {
        vec4 c_left = texture(u_ColorTexture, v_UV - vec2(u_InverseScreenSize.x, 0.0));
        sum += c_left * a.x;
        total_weight += a.x;
    }
    // Right blend
    if (w_right > 0.0) {
        vec4 c_right = texture(u_ColorTexture, v_UV + vec2(u_InverseScreenSize.x, 0.0));
        sum += c_right * w_right;
        total_weight += w_right;
    }
    // Top blend
    if (a.z > 0.0) {
        vec4 c_top = texture(u_ColorTexture, v_UV - vec2(0.0, u_InverseScreenSize.y));
        sum += c_top * a.z;
        total_weight += a.z;
    }
    // Bottom blend
    if (w_bottom > 0.0) {
        vec4 c_bottom = texture(u_ColorTexture, v_UV + vec2(0.0, u_InverseScreenSize.y));
        sum += c_bottom * w_bottom;
        total_weight += w_bottom;
    }

    out_Color = sum / total_weight;
}
