#version 450 core

in vec2 v_UV;
layout(location = 0) out vec4 out_Color;

uniform sampler2D u_LowResolution;
uniform sampler2D u_HighResolution;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_LowResolution, 0));
    vec3 low = vec3(0.0);
    low += texture(u_LowResolution, v_UV + texel * vec2(-1.0, -1.0)).rgb;
    low += texture(u_LowResolution, v_UV + texel * vec2( 0.0, -1.0)).rgb * 2.0;
    low += texture(u_LowResolution, v_UV + texel * vec2( 1.0, -1.0)).rgb;
    low += texture(u_LowResolution, v_UV + texel * vec2(-1.0,  0.0)).rgb * 2.0;
    low += texture(u_LowResolution, v_UV).rgb * 4.0;
    low += texture(u_LowResolution, v_UV + texel * vec2( 1.0,  0.0)).rgb * 2.0;
    low += texture(u_LowResolution, v_UV + texel * vec2(-1.0,  1.0)).rgb;
    low += texture(u_LowResolution, v_UV + texel * vec2( 0.0,  1.0)).rgb * 2.0;
    low += texture(u_LowResolution, v_UV + texel * vec2( 1.0,  1.0)).rgb;
    low /= 16.0;
    out_Color = vec4(texture(u_HighResolution, v_UV).rgb + low, 1.0);
}
