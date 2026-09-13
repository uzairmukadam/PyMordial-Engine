#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FinalColor;

layout (binding = 0) uniform sampler2D u_SceneHDR;
layout (binding = 1) uniform sampler2D u_BloomTexture;
layout (binding = 2) uniform sampler2D u_LensFlareTexture;

// Lighting & Exposure
uniform float u_Exposure;        // default 1.0
uniform float u_BloomIntensity;  // default 0.04
uniform int   u_TonemapMode;     // 0 = ACES, 1 = AgX, 2 = Reinhard
uniform int   u_BloomEnabled;
uniform int   u_LensFlareEnabled;

// Optical Lens Imperfections
uniform int   u_ChromaticAberrationEnabled;
uniform float u_ChromaticAberrationIntensity; // 0.0 to 0.02
uniform int   u_VignetteEnabled;
uniform float u_VignetteIntensity;            // 0.0 to 1.5
uniform float u_VignetteRoundness;            // 0.5 to 2.0 (1.0 = circular)
uniform float u_VignetteSmoothness;           // 0.1 to 1.0
uniform int   u_FilmGrainEnabled;
uniform float u_FilmGrainIntensity;          // 0.0 to 0.20
uniform float u_Time;

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
    vec3 x = val / (val + 0.1875);
    return pow(x, vec3(1.2));
}

vec3 TonemapReinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

// High-speed pseudo-random hash for animated 35mm film grain
float hashGrain(vec2 p, float t) {
    vec2 p2 = p * 1000.0 + mod(t * 123.456, 100.0);
    return fract(sin(dot(p2, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // 1. Chromatic Aberration (Radial Spectral Separation)
    vec3 hdr;
    if (u_ChromaticAberrationEnabled == 1 && u_ChromaticAberrationIntensity > 0.0001) {
        vec2 toCenter = v_UV - 0.5;
        float distSq = dot(toCenter, toCenter);
        vec2 shift = toCenter * distSq * (u_ChromaticAberrationIntensity * 3.0);

        float r = texture(u_SceneHDR, v_UV - shift).r;
        float g = texture(u_SceneHDR, v_UV).g;
        float b = texture(u_SceneHDR, v_UV + shift).b;
        hdr = vec3(r, g, b) * u_Exposure;
    } else {
        hdr = texture(u_SceneHDR, v_UV).rgb * u_Exposure;
    }

    // 2. Anamorphic Lens Flare & Ghost Reflections
    if (u_LensFlareEnabled == 1) {
        vec3 flare = texture(u_LensFlareTexture, v_UV).rgb;
        hdr += flare;
    }

    // 3. Bloom Composite
    if (u_BloomEnabled == 1) {
        vec3 bloom = texture(u_BloomTexture, v_UV).rgb * u_BloomIntensity;
        hdr += bloom;
    }

    // 4. Tonemapping Operator (Linear HDR -> LDR)
    vec3 ldr;
    if (u_TonemapMode == 1) {
        ldr = TonemapAgX(hdr);
    } else if (u_TonemapMode == 2) {
        ldr = TonemapReinhard(hdr);
    } else {
        ldr = TonemapACES(hdr);
    }

    // 5. Physically-Parameterized Lens Vignette
    if (u_VignetteEnabled == 1 && u_VignetteIntensity > 0.01) {
        vec2 coord = abs(v_UV - 0.5) * 2.0;
        float roundExp = max(u_VignetteRoundness * 2.0, 0.5);
        float vDist = pow(pow(coord.x, roundExp) + pow(coord.y, roundExp), 1.0 / roundExp);
        float smoothVal = max(u_VignetteSmoothness, 0.05);
        float vigFalloff = smoothstep(1.0, 1.0 - smoothVal, vDist * u_VignetteIntensity);
        ldr *= clamp(vigFalloff, 0.0, 1.0);
    }

    // 6. Luminance-Masked Animated Filmic Grain (Silver Halide Emulation)
    if (u_FilmGrainEnabled == 1 && u_FilmGrainIntensity > 0.001) {
        float grain = hashGrain(v_UV, u_Time);
        float lum = dot(ldr, vec3(0.2126, 0.7152, 0.0722));
        // Mask grain: pronounced in midtones/shadows, soft in highlights
        float grainMask = 1.0 - smoothstep(0.75, 1.0, lum);
        ldr += (grain - 0.5) * u_FilmGrainIntensity * grainMask;
        ldr = max(vec3(0.0), ldr);
    }

    // 7. Gamma Correction (Linear -> sRGB)
    vec3 srgb = pow(clamp(ldr, 0.0, 1.0), vec3(1.0 / 2.2));

    out_FinalColor = vec4(srgb, 1.0);
}
