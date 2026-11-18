// ============================================================================
//  Canonical Volumetric Scattering Library
//  Shared phase functions, procedural noise, CSM shadow sampling and froxel
//  slice integration used by volumetric fog, particles and cloud shading.
//
//  Consumers that do not sample cascaded shadow maps (particles, clouds) should
//  `#define VOLUMETRICS_NO_CSM` before including this file so the CSM sampler
//  and its uniforms are not declared.
// ============================================================================

#ifndef VOLUMETRICS_GLSL
#define VOLUMETRICS_GLSL

#ifndef PI
#define PI 3.14159265359
#endif

// ----------------------------------------------------------------------------
//  Phase Functions
// ----------------------------------------------------------------------------

// Single-lobe Henyey-Greenstein scattering phase function (normalised by 1/4PI).
// g > 0 forward scattering (Mie haze, god rays), g < 0 backward scattering.
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 0.001), 1.5));
}

// AAA dual-lobe Henyey-Greenstein phase function:
// blends forward Mie diffraction (god rays) with soft backward retro-reflection
// (silvery rim haze).
float DualLobeHenyeyGreenstein(float cosTheta, float g) {
    float gForward = clamp(g, 0.0, 0.98);
    float pForward = HenyeyGreenstein(cosTheta, gForward);
    float pBack = HenyeyGreenstein(cosTheta, -0.22);
    return mix(pBack, pForward, 0.85);
}

// Cornette-Shanks phase function: energy-conserving alternative to
// Henyey-Greenstein with a more realistic forward peak for water droplets.
float CornetteShanksPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    float k = (3.0 / (8.0 * PI)) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + cosTheta * cosTheta) / pow(max(denom, 1e-4), 1.5);
}

// Peak-normalised Henyey-Greenstein lobe: returns 1.0 at cosTheta = 1 and falls
// off towards 0. Useful where the phase is a pure directional shape multiplier
// rather than a physically normalised scattering term.
float HenyeyGreensteinPeakNormalized(float cosTheta, float g) {
    return HenyeyGreenstein(cosTheta, g) / max(HenyeyGreenstein(1.0, g), 1e-6);
}

// ----------------------------------------------------------------------------
//  Procedural Noise
// ----------------------------------------------------------------------------

// Simple 3D hash for spatial mist turbulence.
float Hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Trilinearly interpolated value noise built on Hash31.
float ValueNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = Hash31(i + vec3(0, 0, 0));
    float n100 = Hash31(i + vec3(1, 0, 0));
    float n010 = Hash31(i + vec3(0, 1, 0));
    float n110 = Hash31(i + vec3(1, 1, 0));
    float n001 = Hash31(i + vec3(0, 0, 1));
    float n101 = Hash31(i + vec3(1, 0, 1));
    float n011 = Hash31(i + vec3(0, 1, 1));
    float n111 = Hash31(i + vec3(1, 1, 1));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);

    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}

// Interleaved gradient noise for sub-voxel temporal & spatial anti-banding.
float InterleavedGradientNoise(vec2 screenPos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

// ----------------------------------------------------------------------------
//  Froxel Slice Mapping & Integration
// ----------------------------------------------------------------------------

// Logarithmic depth distribution: maps a normalised slice fraction in [0, 1]
// to a view-space depth in metres.
float FroxelSliceToViewDepth(float sliceFraction, float near, float far) {
    return near * pow(far / near, sliceFraction);
}

// Analytic Beer-Lambert integration of a homogeneous slice of thickness deltaZ.
// Returns the in-scattered radiance accumulated over the slice.
vec3 BeerLambertIntegrate(vec3 scattering, float extinction, float deltaZ) {
    float stepTransmittance = exp(-extinction * deltaZ);
    return scattering * ((1.0 - stepTransmittance) / max(extinction, 1e-6));
}

// ----------------------------------------------------------------------------
//  Volumetric Medium
// ----------------------------------------------------------------------------

struct VolumetricMedium {
    vec3 scattering;   // in-scattered radiance per unit extinction
    float extinction;  // extinction coefficient (1/m)
    float anisotropy;  // Henyey-Greenstein g
};

// Phase-weighted in-scattering for one medium sample lit by `sunRadiance`.
// `shadow` is the CSM visibility term in [0, 1].
vec3 EvaluateVolumetricMedium(VolumetricMedium medium, vec3 sunRadiance, float cosTheta, float shadow) {
    return medium.scattering * sunRadiance * DualLobeHenyeyGreenstein(cosTheta, medium.anisotropy) * shadow;
}

// ----------------------------------------------------------------------------
//  Cascaded Shadow Map Sampling
//  Requires the consumer to declare u_LightViewProjection[4] and u_CascadeSplits
//  (normally via the FrameData uniform block).
// ----------------------------------------------------------------------------

#ifndef VOLUMETRICS_NO_CSM

#include "shaders/common/shadows.glsl"

uniform int u_CascadeCount = 4;
layout (binding = 1) uniform sampler2D u_ShadowAtlas;
// Per-cascade atlas rect: xy = atlas UV offset, zw = atlas UV scale.
uniform vec4 u_CascadeAtlasRect[4] = vec4[4](
    vec4(0.0, 0.0, 0.5, 0.5),
    vec4(0.5, 0.0, 0.5, 0.5),
    vec4(0.0, 0.5, 0.5, 0.5),
    vec4(0.5, 0.5, 0.5, 0.5)
);

// Sample a single cascade shadow map with 4-tap bilinear PCF and soft border feathering.
float SampleSingleCascadeShadow(int cascade, vec3 worldPos) {
    vec4 lightClip = u_LightViewProjection[cascade] * vec4(worldPos, 1.0);
    vec3 projCoords = lightClip.xyz / max(lightClip.w, 1e-6);

    // Remap NDC XY [-1, 1] to UV [0, 1]; Z is already in [0, 1] under GL_ZERO_TO_ONE
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // Bounds check: if outside light near/far clip range, voxel is unshadowed
    if (projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 1.0;
    }

    // Distance from cascade border in UV space [0, 1]
    vec2 borderDist = min(projCoords.xy, 1.0 - projCoords.xy);
    float edgeFactor = min(borderDist.x, borderDist.y);
    if (edgeFactor <= 0.0) {
        return 1.0; // Outside XY cascade bounds
    }

    vec4 rect = u_CascadeAtlasRect[cascade];
    vec2 uv = ShadowAtlasUv(projCoords.xy, rect);
    vec2 uvMin = ShadowAtlasUvMin(rect, 0.001);
    vec2 uvMax = ShadowAtlasUvMax(rect, 0.001);

    // 4-Tap Bilinear PCF filtering inside the cascade rect
    vec2 fullSize = vec2(textureSize(u_ShadowAtlas, 0));
    vec2 texelSize = 1.0 / fullSize;
    vec2 pos = uv * fullSize - 0.5;
    vec2 f = fract(pos);
    vec2 baseUV = (floor(pos) + 0.5) * texelSize;

    float currentDepth = projCoords.z;
    float bias = 0.0012;

    float d00 = texture(u_ShadowAtlas, clamp(baseUV, uvMin, uvMax)).r;
    float d10 = texture(u_ShadowAtlas, clamp(baseUV + vec2(texelSize.x, 0.0), uvMin, uvMax)).r;
    float d01 = texture(u_ShadowAtlas, clamp(baseUV + vec2(0.0, texelSize.y), uvMin, uvMax)).r;
    float d11 = texture(u_ShadowAtlas, clamp(baseUV + texelSize, uvMin, uvMax)).r;

    float s00 = (currentDepth > d00 + bias) ? 0.0 : 1.0;
    float s10 = (currentDepth > d10 + bias) ? 0.0 : 1.0;
    float s01 = (currentDepth > d01 + bias) ? 0.0 : 1.0;
    float s11 = (currentDepth > d11 + bias) ? 0.0 : 1.0;

    float shadowVal = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);

    // Smooth edge blend: feathers into 1.0 at cascade border to eliminate any sharp seams
    float borderFade = smoothstep(0.0, 0.08, edgeFactor);
    return mix(1.0, shadowVal, borderFade);
}

// CSM shadow test for a 3D point in world space with seamless split transitions.
float SampleCascadeShadow(vec3 worldPos, float viewDepth) {
    float maxShadowDist = u_CascadeSplits[u_CascadeCount - 1];
    if (viewDepth >= maxShadowDist) {
        return 1.0;
    }

    int cascade = 0;
    for (int c = 0; c < u_CascadeCount - 1; ++c) {
        if (viewDepth > u_CascadeSplits[c]) {
            cascade = c + 1;
        }
    }

    float shadow = SampleSingleCascadeShadow(cascade, worldPos);

    // Smooth cascade split transition blending to eliminate seams
    if (cascade < u_CascadeCount - 1) {
        float splitVal = u_CascadeSplits[cascade];
        float blendRange = splitVal * 0.20;
        float blendStart = splitVal - blendRange;
        if (viewDepth > blendStart) {
            float blendT = clamp((viewDepth - blendStart) / blendRange, 0.0, 1.0);
            float nextShadow = SampleSingleCascadeShadow(cascade + 1, worldPos);
            shadow = mix(shadow, nextShadow, blendT);
        }
    } else {
        // Final cascade: smooth distance fade-out towards max shadow distance
        float fadeRange = maxShadowDist * 0.20;
        float fadeStart = maxShadowDist - fadeRange;
        if (viewDepth > fadeStart) {
            float fadeT = clamp((viewDepth - fadeStart) / max(fadeRange, 0.001), 0.0, 1.0);
            shadow = mix(shadow, 1.0, fadeT);
        }
    }

    return shadow;
}

#endif // VOLUMETRICS_NO_CSM

#endif // VOLUMETRICS_GLSL
