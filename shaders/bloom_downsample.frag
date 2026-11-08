#version 450 core

in vec2 v_UV;
layout(location = 0) out vec4 out_Color;

uniform sampler2D u_Source;
uniform float u_Threshold;

vec3 sample_source(vec2 uv) {
    vec3 color = texture(u_Source, uv).rgb;
    if (u_Threshold < 0.0) {
        return color;
    }
    float luminance = max(max(color.r, color.g), color.b);
    float knee = max(u_Threshold * 0.5, 1e-4);
    float soft = clamp(luminance - u_Threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, luminance - u_Threshold) / max(luminance, 1e-4);
    return color * max(contribution, 0.0);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_Source, 0));
    vec3 color = vec3(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float weight = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
            color += sample_source(v_UV + texel * vec2(x, y)) * weight;
        }
    }
    color /= 16.0;
    out_Color = vec4(color, 1.0);
}
