#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_FinalColor;

layout (binding = 0) uniform sampler2D u_SceneHDR;
layout (binding = 1) uniform sampler2D u_BloomTexture;
layout (binding = 2) uniform sampler2D u_LensFlareTexture;

#include "shaders/common/tonemap.glsl"

// Exposure & Tone-Mapping
uniform float u_Exposure;              // Linear exposure multiplier (2^EV)
uniform float u_ExposureCompensation;  // EV stops applied on top of u_Exposure
uniform int   u_TonemapMode;           // 0 = ACES fitted, 1 = AgX, 2 = Neutral, 3 = Reinhard
uniform float u_BloomIntensity;
uniform int   u_BloomEnabled;
uniform int   u_LensFlareEnabled;

// White balance (Kelvin -> RGB gain)
uniform int   u_WhiteBalanceEnabled;
uniform float u_WhiteBalanceK;

// ASC-CDL style grading, applied in display-referred space after the curve
uniform float u_Contrast;              // 1.0 = neutral, pivot 0.18
uniform float u_Saturation;            // 1.0 = neutral
uniform vec3  u_Lift;                  // additive shadow offset
uniform vec3  u_Gamma;                 // midtone power
uniform vec3  u_Gain;                  // multiplicative highlight scale

// Optical Lens Imperfections
uniform int   u_ChromaticAberrationEnabled;
uniform float u_ChromaticAberrationIntensity; // 0.0 to 0.02
uniform int   u_VignetteEnabled;
uniform float u_VignetteIntensity;            // 0.0 to 1.5
uniform float u_VignetteRoundness;            // 0.5 to 2.0 (1.0 = circular)
uniform float u_VignetteSmoothness;           // 0.1 to 1.0
uniform int   u_FilmGrainEnabled;
uniform float u_FilmGrainIntensity;           // 0.0 to 0.20
uniform float u_Time;

// ---------------------------------------------------------------------------
//  White balance
//
//  Tanner Helland's blackbody approximation, normalised so 6500 K is a no-op.
//  Applied in linear space before the tone curve, which is where a real
//  camera's white balance sits.
// ---------------------------------------------------------------------------
vec3 KelvinToRGB(float kelvin) {
    float t = clamp(kelvin, 1000.0, 40000.0) / 100.0;
    vec3 rgb;

    if (t <= 66.0) {
        rgb.r = 255.0;
        rgb.g = 99.4708025861 * log(t) - 161.1195681661;
        rgb.b = (t <= 19.0) ? 0.0 : 138.5177312231 * log(t - 10.0) - 305.0447927307;
    } else {
        rgb.r = 329.698727446 * pow(t - 60.0, -0.1332047592);
        rgb.g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
        rgb.b = 255.0;
    }

    return clamp(rgb / 255.0, 0.0, 1.0);
}

vec3 WhiteBalanceGain(float kelvin) {
    vec3 reference = KelvinToRGB(6500.0);
    vec3 target = KelvinToRGB(kelvin);
    // `kelvin` is the temperature of the illuminant being compensated for, so
    // the correction is its inverse: a higher value warms the image and a lower
    // value cools it. This matches the Unreal and Unity grading sliders.
    // Normalised on green so the overall exposure does not shift.
    return (reference / max(target, vec3(1e-4))) * (target.g / max(reference.g, 1e-4));
}

// ---------------------------------------------------------------------------
//  Film grain
// ---------------------------------------------------------------------------
uint hashGrain(uint value) {
    value = value * 747796405u + 2891336453u;
    uint word = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    return (word >> 22u) ^ word;
}

float grainRandom(uvec2 pixel, uint frame, uint stream) {
    uint seed = pixel.x * 0x9E3779B9u
        ^ pixel.y * 0x85EBCA6Bu
        ^ frame * 0xC2B2AE35u
        ^ stream * 0x27D4EB2Fu;
    return float(hashGrain(seed) & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
//  Grading
// ---------------------------------------------------------------------------
vec3 ApplyContrast(vec3 c, float contrast) {
    if (abs(contrast - 1.0) < 1e-4) {
        return c;
    }
    // Pivot on middle grey so contrast does not shift overall exposure.
    const float pivot = 0.18;
    return max(pow(max(c, 0.0) / pivot, vec3(contrast)) * pivot, 0.0);
}

vec3 ApplySaturation(vec3 c, float saturation) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return max(mix(vec3(luma), c, saturation), 0.0);
}

vec3 ApplyCDL(vec3 c, vec3 lift, vec3 gamma, vec3 gain) {
    c = c * gain + lift * (1.0 - c);
    return pow(max(c, 0.0), 1.0 / max(gamma, vec3(1e-3)));
}

void main() {
    // 1. Chromatic Aberration (radial spectral separation), sampled in HDR
    vec3 hdr;
    if (u_ChromaticAberrationEnabled == 1 && u_ChromaticAberrationIntensity > 0.0001) {
        vec2 toCenter = v_UV - 0.5;
        float distSq = dot(toCenter, toCenter);
        vec2 shift = toCenter * distSq * (u_ChromaticAberrationIntensity * 3.0);

        float r = texture(u_SceneHDR, v_UV - shift).r;
        float g = texture(u_SceneHDR, v_UV).g;
        float b = texture(u_SceneHDR, v_UV + shift).b;
        hdr = vec3(r, g, b);
    } else {
        hdr = texture(u_SceneHDR, v_UV).rgb;
    }

    // 2. Exposure (linear) with EV compensation
    float exposure = u_Exposure * exp2(u_ExposureCompensation);
    hdr *= max(exposure, 0.0);

    // 3. White balance (linear)
    if (u_WhiteBalanceEnabled == 1) {
        hdr *= WhiteBalanceGain(u_WhiteBalanceK);
    }

    // 4. Lens flare & bloom, both additive in linear HDR
    if (u_LensFlareEnabled == 1) {
        hdr += texture(u_LensFlareTexture, v_UV).rgb;
    }
    if (u_BloomEnabled == 1) {
        hdr += texture(u_BloomTexture, v_UV).rgb * u_BloomIntensity;
    }

    // 5. Tone-map: linear HDR -> display-referred linear [0, 1]
    vec3 ldr = TonemapDispatch(hdr, u_TonemapMode);

    // 6. Grading in display-referred space
    ldr = ApplyContrast(ldr, u_Contrast);
    ldr = ApplySaturation(ldr, u_Saturation);
    ldr = ApplyCDL(ldr, u_Lift, u_Gamma, u_Gain);
    ldr = clamp(ldr, 0.0, 1.0);

    // 7. Physically-parameterised lens vignette
    if (u_VignetteEnabled == 1 && u_VignetteIntensity > 0.01) {
        vec2 coord = abs(v_UV - 0.5) * 2.0;
        float roundExp = max(u_VignetteRoundness * 2.0, 0.5);
        float vDist = pow(pow(coord.x, roundExp) + pow(coord.y, roundExp), 1.0 / roundExp);
        float smoothVal = max(u_VignetteSmoothness, 0.05);
        float vigFalloff = smoothstep(1.0, 1.0 - smoothVal, vDist * u_VignetteIntensity);
        ldr *= clamp(vigFalloff, 0.0, 1.0);
    }

    // 8. Luminance-masked animated filmic grain (silver halide emulation)
    if (u_FilmGrainEnabled == 1 && u_FilmGrainIntensity > 0.001) {
        uvec2 pixel = uvec2(gl_FragCoord.xy);
        uint frame = uint(floor(max(u_Time, 0.0) * 24.0));
        float grain = (
            grainRandom(pixel, frame, 0u)
            + grainRandom(pixel, frame, 1u)
            + grainRandom(pixel, frame, 2u)
            + grainRandom(pixel, frame, 3u)
        ) * 0.5 - 1.0;
        float lum = dot(ldr, vec3(0.2126, 0.7152, 0.0722));
        // Pronounced in midtones/shadows, soft in highlights.
        float grainMask = 1.0 - smoothstep(0.75, 1.0, lum);
        ldr += grain * u_FilmGrainIntensity * grainMask;
        ldr = max(ldr, vec3(0.0));
    }

    // 9. Display encode (exact sRGB transfer function)
    out_FinalColor = vec4(LinearToSRGB(ldr), 1.0);
}
