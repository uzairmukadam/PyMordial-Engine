#version 450 core

// Physically-motivated solar flare anchored to the sun's screen-space position.
//
// Components (each independently weighted by a uniform):
//   1. Sun disc bloom      - bright core + soft radial glow, sized from the
//                            physical solar angular radius projected to pixels.
//   2. Starburst spikes    - N-blade aperture diffraction pattern with
//                            wavelength-dependent spike sharpness.
//   3. Anamorphic streak   - horizontal streak through the solar core with
//                            chromatic dispersion.
//   4. Optical ghosts      - internal lens reflections through the lens axis,
//                            tinted by the atmosphere transmittance.
//   5. Halo ring           - thin circular lens ring around the optical axis.
//
// The whole flare is faded out by a depth-buffer occlusion test at the sun's
// screen position, so it disappears when the sun is hidden behind geometry.

#include "common/frame_data.glsl"

in vec2 v_UV;
layout(location = 0) out vec4 out_FlareColor;

layout(binding = 0) uniform sampler2D u_SceneHDR;
layout(binding = 1) uniform sampler2D u_DepthTexture;   // Reversed-Z 32F

uniform vec2  u_SunScreenPos;        // Sun centre in UV space (optional override)
uniform vec2  u_ScreenSize;          // Flare target resolution (px)
uniform float u_SunAngularRadius;    // Physical solar angular radius (rad, ~0.266 deg)
uniform float u_SunOnScreen;         // 1 = sun in front of camera, 0 = behind
uniform float u_FlareThreshold;      // Luminance cutoff for scene-driven taps
uniform vec3  u_FlareTint;           // Base flare tint

uniform int   u_BladeCount;          // Aperture blade count (diffraction spikes)
uniform float u_BloomIntensity;      // Sun disc bloom weight
uniform float u_StarburstIntensity;  // Diffraction spike weight
uniform float u_StreakIntensity;     // Anamorphic streak weight
uniform float u_StreakWidth;         // Horizontal streak spread (px)
uniform float u_GhostIntensity;      // Optical ghost weight
uniform float u_HaloIntensity;       // Halo ring weight
uniform float u_OcclusionFade;       // 0 = ignore geometry, 1 = full occlusion

// Derives the sun's screen-space position from the frame UBO so the pass does
// not depend on the host computing it. `u_SunDirection_Intensity.xyz` points
// from the sun into the scene, so the direction *towards* the sun is its
// negation. Returns (uv, onScreen).
vec3 resolveSunScreenPos() {
    vec3 toSun = -normalize(u_SunDirection_Intensity.xyz);
    vec4 clip = u_ViewProjection * vec4(toSun, 0.0);
    if (clip.w <= 1e-6) {
        return vec3(0.5, 0.5, 0.0); // Sun is behind the camera
    }
    vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    float onScreen = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) ? 1.0 : 0.0;
    return vec3(uv, onScreen);
}

float getLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Reversed-Z (near = 1.0, far = 0.0) view-space depth reconstruction.
float linearizeDepth(float rawDepth, vec2 uv) {
    if (rawDepth <= 1e-6) {
        return 2000.0; // Sky / infinitely far background
    }
    vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
    vec4 viewPos = u_InvProjection * clip;
    return max(-viewPos.z / max(viewPos.w, 1e-6), 0.05);
}

vec3 sampleThreshold(vec2 uv) {
    vec3 col = texture(u_SceneHDR, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
    float lum = getLuminance(col);
    float excess = max(0.0, lum - u_FlareThreshold);
    return col * (excess / max(lum, 1e-4));
}

// -----------------------------------------------------------------------------
// Depth-buffer occlusion: the sun sits at infinity, so any finite geometry in
// front of it hides the flare. A 5-tap cross avoids single-texel aliasing on
// thin occluders (fences, foliage, wires).
// -----------------------------------------------------------------------------
float sunVisibility(vec2 sunUV) {
    vec2 texel = 1.0 / max(u_ScreenSize, vec2(1.0));
    float raw = texture(u_DepthTexture, sunUV).r;
    raw += texture(u_DepthTexture, clamp(sunUV + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r;
    raw += texture(u_DepthTexture, clamp(sunUV - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r;
    raw += texture(u_DepthTexture, clamp(sunUV + vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r;
    raw += texture(u_DepthTexture, clamp(sunUV - vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r;
    raw *= 0.2;

    float viewDepth = linearizeDepth(raw, sunUV);
    float occluded = 1.0 - smoothstep(150.0, 600.0, viewDepth);
    return 1.0 - occluded * clamp(u_OcclusionFade, 0.0, 1.0);
}

// -----------------------------------------------------------------------------
// 1. Sun disc bloom: bright core with a soft radial falloff.
// -----------------------------------------------------------------------------
vec3 sunBloom(vec2 dpx, float radiusPx, float energy) {
    float r = length(dpx);
    float core = exp(-pow(r / max(radiusPx * 1.6, 0.5), 2.0));
    float glow = exp(-r / max(radiusPx * 7.0, 1.0));
    return (core + glow * 0.35) * energy * u_FlareTint;
}

// -----------------------------------------------------------------------------
// 2. Starburst diffraction spikes from an N-blade aperture diaphragm.
//    Blue diffracts into a tighter spike than red, giving a spectral fringe.
// -----------------------------------------------------------------------------
vec3 starburst(vec2 dpx, float radiusPx, float energy) {
    float r = length(dpx);
    float theta = atan(dpx.y, dpx.x);
    float blades = float(clamp(u_BladeCount, 2, 16));

    // cos(N/2 * theta) yields N diffraction lobes around the solar core.
    float lobe = abs(cos(theta * blades * 0.5));
    vec3 spikes = pow(vec3(lobe), vec3(24.0, 30.0, 38.0));

    float falloff = exp(-r / max(radiusPx * 45.0, 1.0));
    return spikes * falloff * energy * u_FlareTint;
}

// -----------------------------------------------------------------------------
// 3. Anamorphic horizontal streak through the solar core (chromatic dispersion).
// -----------------------------------------------------------------------------
vec3 anamorphicStreak(vec2 dpx, float radiusPx, float energy) {
    float width = max(u_StreakWidth, 1.0);
    float vertical = exp(-pow(dpx.y / max(radiusPx * 2.0, 1.0), 2.0));

    // Red spreads wider than blue along the streak.
    vec3 spread = vec3(1.06, 1.0, 0.94);
    vec3 profile = exp(-abs(dpx.x * spread) / width);
    return profile * vertical * energy * u_FlareTint;
}

// -----------------------------------------------------------------------------
// 4. Optical ghosts: internal reflections mirrored through the lens axis.
// -----------------------------------------------------------------------------
vec3 opticalGhosts(vec2 dpx, float radiusPx, vec3 atmoTint) {
    vec2 toCenter = (vec2(0.5) - u_SunScreenPos) * u_ScreenSize;

    const int GHOST_COUNT = 4;
    const float ghostScales[4] = float[4](-0.8, 0.4, -0.3, 0.7);
    const vec3 ghostColors[4] = vec3[4](
        vec3(0.20, 0.60, 1.00),  // Cyan
        vec3(0.90, 0.40, 0.20),  // Amber
        vec3(0.40, 0.90, 0.30),  // Emerald
        vec3(0.30, 0.50, 0.95)   // Deep blue
    );

    vec3 ghosts = vec3(0.0);
    for (int i = 0; i < GHOST_COUNT; ++i) {
        vec2 ghostOffset = toCenter * ghostScales[i];
        float d = length(dpx - ghostOffset);
        float disc = exp(-pow(d / max(radiusPx * 3.0, 1.0), 2.0));
        ghosts += disc * ghostColors[i];
    }
    return ghosts * atmoTint;
}

// -----------------------------------------------------------------------------
// 5. Halo ring around the optical axis.
// -----------------------------------------------------------------------------
vec3 haloRing(vec2 dpx, float radiusPx, vec3 atmoTint) {
    vec2 toCenter = (vec2(0.5) - u_SunScreenPos) * u_ScreenSize;
    vec2 ringCenter = toCenter * 0.55;
    float ringRadius = length(toCenter) * 0.55;

    float d = length(dpx - ringCenter);
    float ring = exp(-pow((d - ringRadius) / max(radiusPx * 1.5, 1.0), 2.0));
    return ring * vec3(0.4, 0.75, 1.0) * atmoTint;
}

void main() {
    // Prefer the host-supplied position when it is meaningful; otherwise derive
    // it from the frame UBO so the pass is self-sufficient.
    vec3 derived = resolveSunScreenPos();
    vec2 sunPos = (u_SunOnScreen >= 0.0 && u_SunScreenPos != vec2(0.0))
        ? u_SunScreenPos
        : derived.xy;
    float onScreen = (u_SunOnScreen >= 0.0) ? u_SunOnScreen : derived.z;

    vec2 sunUV = clamp(sunPos, vec2(0.0), vec2(1.0));
    vec2 dpx = (v_UV - sunPos) * u_ScreenSize;

    // Projected solar disc radius in pixels from the physical angular radius.
    // u_Projection[1][1] == 1 / tan(fovy / 2).
    float radiusPx = max(u_SunAngularRadius * u_Projection[1][1] * 0.5 * u_ScreenSize.y, 1.0);

    // Solar radiance as seen through the atmosphere (drives bloom/streak energy).
    vec3 sunSample = texture(u_SceneHDR, sunUV).rgb;
    float sunLum = getLuminance(sunSample);
    float energy = max(sunLum, 1.0);

    // Chromaticity of the sun disc == atmosphere transmittance tint.
    vec3 atmoTint = sunLum > 1e-4
        ? clamp(sunSample / sunLum, vec3(0.0), vec3(2.0))
        : vec3(1.0, 0.85, 0.65);

    float visibility = sunVisibility(sunUV) * clamp(onScreen, 0.0, 1.0);
    // Fade out as the sun leaves the frame.
    vec2 edge = smoothstep(vec2(0.0), vec2(0.08), sunUV)
              * smoothstep(vec2(0.0), vec2(0.08), vec2(1.0) - sunUV);
    visibility *= edge.x * edge.y;

    vec3 total = vec3(0.0);
    if (visibility > 0.001) {
        if (u_BloomIntensity > 0.001) {
            total += sunBloom(dpx, radiusPx, energy) * u_BloomIntensity;
        }
        if (u_StarburstIntensity > 0.001) {
            total += starburst(dpx, radiusPx, energy) * u_StarburstIntensity;
        }
        if (u_StreakIntensity > 0.001) {
            total += anamorphicStreak(dpx, radiusPx, energy) * u_StreakIntensity;
        }
        if (u_GhostIntensity > 0.001) {
            total += opticalGhosts(dpx, radiusPx, atmoTint) * u_GhostIntensity;
        }
        if (u_HaloIntensity > 0.001) {
            total += haloRing(dpx, radiusPx, atmoTint) * u_HaloIntensity;
        }
        total *= visibility;
    }

    out_FlareColor = vec4(total, 1.0);
}
