#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FinalColor;

layout (binding = 0) uniform sampler2D u_SceneHDR;
layout (binding = 1) uniform sampler2D u_BloomTexture;

uniform float u_Exposure;        // default 1.0
uniform float u_BloomIntensity;  // default 0.04
uniform int u_TonemapMode;       // 0 = ACES, 1 = AgX, 2 = Reinhard
uniform int u_BloomEnabled;

// ACES Filmic Tonemapping Curve (Narkowicz 2015 fit)
vec3 TonemapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// AgX approximation for rich highlight preservation
vec3 TonemapAgX(vec3 val) {
    val = max(val, 0.0);
    // Log encoding & soft compression
    vec3 x = val / (val + 0.1875);
    return pow(x, vec3(1.2));
}

vec3 TonemapReinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

void main() {
    vec3 hdr = texture(u_SceneHDR, v_UV).rgb * u_Exposure;

    if (u_BloomEnabled == 1) {
        vec3 bloom = texture(u_BloomTexture, v_UV).rgb * u_BloomIntensity;
        hdr += bloom;
    }

    // Apply Tonemapping
    vec3 ldr;
    if (u_TonemapMode == 1) {
        ldr = TonemapAgX(hdr);
    } else if (u_TonemapMode == 2) {
        ldr = TonemapReinhard(hdr);
    } else {
        ldr = TonemapACES(hdr);
    }

    // Subtle filmic vignette
    vec2 coord = (v_UV - 0.5) * 2.0;
    float vignette = 1.0 - dot(coord, coord) * 0.18;
    ldr *= clamp(vignette, 0.0, 1.0);

    // Gamma correction (linear -> sRGB)
    vec3 srgb = pow(ldr, vec3(1.0 / 2.2));

    out_FinalColor = vec4(srgb, 1.0);
}
