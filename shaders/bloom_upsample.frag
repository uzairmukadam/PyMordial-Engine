#version 450 core

#include "shaders/common/blur.glsl"

in vec2 v_UV;
layout(location = 0) out vec4 out_Color;

uniform sampler2D u_LowResolution;
uniform sampler2D u_HighResolution;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_LowResolution, 0));
    // The 3x3 tent (1,2,1)^2 / 16, shared with the downsample stage.
    vec3 low = TentBlur3x3(u_LowResolution, v_UV, texel).rgb;
    out_Color = vec4(texture(u_HighResolution, v_UV).rgb + low, 1.0);
}
