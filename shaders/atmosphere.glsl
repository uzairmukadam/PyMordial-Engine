// ============================================================================
//  Sébastien Hillaire (Eurographics 2020) Physically Based Sky & Atmosphere
//  Real-time O(1) Sky-View LUT Sampling, Solar Disc & Night Celestial Shading
// ============================================================================

#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL

#include "shaders/hillaire_common.glsl"

// Sébastien Hillaire 2020 Atmospheric LUT Samplers
layout (binding = 10) uniform sampler2D u_SkyViewLUT;
layout (binding = 11) uniform sampler2D u_TransmittanceLUT;

// Parameterized Atmospheric Visual Options
uniform vec3 u_NightZenithColor = vec3(0.012, 0.025, 0.050);  // Night zenith tint
uniform vec3 u_NightHorizonColor = vec3(0.035, 0.055, 0.090); // Night horizon tint
uniform int u_SkyMode = 0;                                     // 0: atmosphere, 1: space
uniform vec3 u_SpaceZenithColor = vec3(0.0015, 0.0020, 0.0060);
uniform vec3 u_NebulaColor = vec3(0.16, 0.055, 0.28);
uniform float u_NebulaIntensity = 0.35;
uniform vec3 u_AtmosphereSunDir = vec3(0.35, -0.85, 0.40);    // True astronomical sun vector
uniform vec3 u_AtmosphereMoonDir = vec3(-0.35, 0.85, -0.40);  // True astronomical moon vector
uniform float u_SunDiscSize = 0.00465;                          // Solar disc angular radius (radians)
uniform float u_MoonDiscSize = 0.00450;                         // Lunar disc angular radius (radians)
uniform vec3 u_MoonColor = vec3(0.70, 0.82, 1.0);
uniform float u_MoonIntensity = 0.40;
uniform float u_StarIntensity = 1.0;
uniform float u_StarDensity = 1.0;
uniform float u_SkyExposure = 1.0;                              // Sky radiance multiplier
uniform float u_SkySaturation = 1.0;                            // Luminance-preserving saturation

vec3 EvaluateSpaceSky(vec3 d, vec3 sunDir, vec3 sunColor, float sunLux, float time) {
    float phi = atan(d.z, d.x);
    float theta = asin(clamp(d.y, -1.0, 1.0));
    float band = exp(-pow((theta - 0.22 * sin(phi * 1.7 + 0.4)) / 0.28, 2.0));
    float nebulaNoise = 0.5
        + 0.25 * sin(phi * 5.0 + theta * 8.0)
        + 0.15 * sin(phi * 11.0 - theta * 13.0)
        + 0.10 * sin(phi * 19.0 + theta * 17.0);
    float nebula = smoothstep(0.28, 0.78, nebulaNoise) * band;
    vec3 space = u_SpaceZenithColor + u_NebulaColor * nebula * u_NebulaIntensity;

    // Sparse natural starfield: coarse cell grid, few occupied cells, steep
    // magnitude curve so a handful of bright stars stand out against faint ones.
    vec2 starCoord = vec2(phi, theta) * (90.0 * u_StarDensity);
    vec2 cell = floor(starCoord);
    vec2 f = fract(starCoord);
    vec2 hash = fract(sin(vec2(
        dot(cell, vec2(127.1, 311.7)),
        dot(cell, vec2(269.5, 183.3))
    )) * 43758.5453);
    if (hash.x > 0.975 && u_StarIntensity > 0.0) {
        vec2 starPos = 0.25 + 0.50 * hash;
        float dist = length(f - starPos);
        float point = exp(-dist * dist * 220.0);
        float twinkle = 0.94 + 0.06 * sin(time * 2.1 + hash.y * 62.83);
        float magnitude = (hash.x - 0.975) / 0.025;
        vec3 tint = mix(vec3(0.78, 0.86, 1.0), vec3(1.0, 0.78, 0.54), hash.y);
        space += tint * point * twinkle * (0.4 + 2.4 * magnitude * magnitude) * u_StarIntensity;
    }

    vec3 L = normalize(-sunDir);
    float sunAngle = acos(clamp(dot(d, L), -1.0, 1.0));
    if (sunAngle < u_SunDiscSize) {
        float edge = 1.0 - smoothstep(0.85, 1.0, sunAngle / max(u_SunDiscSize, 1e-5));
        space += sunColor * sunLux * 35.0 * edge;
    }

    float moonAngle = acos(clamp(dot(d, normalize(-u_AtmosphereMoonDir)), -1.0, 1.0));
    if (moonAngle < u_MoonDiscSize) {
        float edge = 1.0 - smoothstep(0.85, 1.0, moonAngle / max(u_MoonDiscSize, 1e-5));
        space += u_MoonColor * u_MoonIntensity * 12.0 * edge;
    }
    return space;
}

vec3 EvaluateAtmosphere(vec3 rayDir, vec3 sunDir, vec3 sunColor, float sunLux, float time, vec3 cameraPos) {
    vec3 d = normalize(rayDir);
    if (u_SkyMode == 1) {
        return EvaluateSpaceSky(d, sunDir, sunColor, sunLux, time);
    }
    vec3 L = normalize(-sunDir); // direction pointing toward the sun
    float sunElevation = -sunDir.y;

    // Altitude of observer relative to planet center
    float r = clamp(u_RBottom + max(1.0, cameraPos.y), u_RBottom + 1.0, u_RTop - 10.0);

    vec3 N = vec3(0.0, 1.0, 0.0); // Local zenith vector

    // 1. Calculate View Zenith and Sun Relative Azimuth for Sky-View LUT
    float cosZenith = clamp(dot(d, N), -1.0, 1.0);
    float viewZenith = acos(cosZenith);

    vec3 d_horiz = d - cosZenith * N;
    vec3 L_horiz = L - dot(L, N) * N;
    float len_dh = length(d_horiz);
    float len_lh = length(L_horiz);

    float sunAzimuth = 0.0;
    if (len_dh > 1e-5 && len_lh > 1e-5) {
        float cosAz = clamp(dot(d_horiz / len_dh, L_horiz / len_lh), -1.0, 1.0);
        sunAzimuth = acos(cosAz);
    }

    vec2 skyViewUv = SkyViewParamsToUv(viewZenith, sunAzimuth, r);
    vec3 skyRadiance = texture(u_SkyViewLUT, skyViewUv).rgb * 4.0;

    // 2. Solar Disc with Limb Darkening & Atmospheric Transmittance
    float cosTheta = dot(d, L);
    float sunAng = acos(clamp(cosTheta, -1.0, 1.0));

    // Sample solar transmittance through atmosphere
    float mu_s = clamp(dot(N, L), -1.0, 1.0);
    vec2 transUv = LutTransmittanceParamsToUv(r, mu_s);
    vec3 T_sun = texture(u_TransmittanceLUT, transUv).rgb;
    vec3 sunRadiance = sunColor * sunLux;

    // Check if view ray intersects the planet terrain
    float tg1, tg2;
    bool hits_ground = RaySphereIntersect(vec3(0.0, r, 0.0), d, u_RBottom, tg1, tg2) && (tg1 > 0.0);

    if (!hits_ground) {
        if (sunAng < u_SunDiscSize && sunElevation > -0.05) {
            float normDist = sunAng / u_SunDiscSize;
            float limb = 1.0 - 0.35 * pow(normDist, 1.5);
            vec3 disc = (sunRadiance * 35.0) * limb * T_sun;
            float disc_edge = 1.0 - smoothstep(0.85, 1.0, normDist);
            skyRadiance += disc * disc_edge;
        }

        // Forward Mie Corona Glow around sun
        float corona = pow(max(cosTheta, 0.0), 48.0) * 0.35 + pow(max(cosTheta, 0.0), 256.0) * 1.5;
        skyRadiance += sunRadiance * corona * T_sun * 0.25;
    }

    // 3. Night Sky foundation (Lunar disc & Twinkling Starfield)
    //
    // Physically motivated twilight. The solar transmittance along the sun
    // direction measures how much direct sunlight still reaches the observer, so
    // the sky darkens through blue -> orange -> deep blue as it falls instead of
    // snapping at a fixed elevation. The elevation ramp keeps the horizon glow
    // alive through civil twilight, and the transmittance term makes the falloff
    // depend on the planet's air column (thicker or hazier atmospheres produce
    // longer, deeper twilights).
    float mu_twilight = max(mu_s, 0.12);
    vec2 twilightUv = LutTransmittanceParamsToUv(r, mu_twilight);
    vec3 T_twilight = texture(u_TransmittanceLUT, twilightUv).rgb;
    float twilightLum = dot(T_twilight, vec3(0.2126, 0.7152, 0.0722));
    vec2 zenithUv = LutTransmittanceParamsToUv(r, 1.0);
    float zenithLum = max(dot(texture(u_TransmittanceLUT, zenithUv).rgb, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
    float sunTransmittance = clamp(twilightLum / zenithLum, 0.0, 1.0);
    float dayFactor = smoothstep(-0.14, 0.03, sunElevation) * mix(0.55, 1.0, sunTransmittance);

    // Sky grading: exposure scales the physical radiance before the night blend,
    // saturation is a luminance-preserving lerp towards (or away from) grey.
    skyRadiance *= u_SkyExposure;
    float skyLum = dot(skyRadiance, vec3(0.2126, 0.7152, 0.0722));
    skyRadiance = mix(vec3(skyLum), skyRadiance, u_SkySaturation);

    vec3 nightSky = mix(u_NightHorizonColor, u_NightZenithColor, pow(max(d.y, 0.0), 0.7));

    // Lunar disc & halo, placed on the moon's own orbit rather than antipodal to
    // the sun so it rises and sets independently.
    vec3 M = normalize(-u_AtmosphereMoonDir);
    float moonCosTheta = dot(d, M);
    float moonAng = acos(clamp(moonCosTheta, -1.0, 1.0));
    if (moonAng < u_MoonDiscSize && d.y > 0.0) {
        float mNorm = moonAng / u_MoonDiscSize;
        nightSky += u_MoonColor * u_MoonIntensity * 12.0 * (1.0 - smoothstep(0.85, 1.0, mNorm));
    }
    nightSky += u_MoonColor * pow(max(moonCosTheta, 0.0), 64.0) * 0.80;

    // Twinkling Starfield (naturally fades out in daylight)
    float starDayFade = 1.0 - smoothstep(0.0, 0.12, dayFactor);
    if (d.y > 0.02 && u_StarIntensity > 0.0 && starDayFade > 0.001) {
        // High-frequency celestial spherical mapping. The cell grid is coarser
        // than a naive 180/cell mapping so the sky reads as a sparse natural
        // starfield rather than a dense speckle.
        float phi = atan(d.z, d.x);
        float theta = asin(clamp(d.y, -1.0, 1.0));
        vec2 starCoord = vec2(phi, theta) * (90.0 * u_StarDensity);

        vec2 cell = floor(starCoord);
        vec2 f = fract(starCoord);

        // Fast hash for this celestial grid cell
        vec2 hash = fract(sin(vec2(dot(cell, vec2(127.1, 311.7)), dot(cell, vec2(269.5, 183.3)))) * 43758.5453);

        // Only a small fraction of cells host a star, and most of those are dim.
        if (hash.x > 0.975) {
            vec2 starPos = 0.25 + 0.50 * hash; // jittered within cell interior
            float dist = length(f - starPos);
            float starPoint = exp(-dist * dist * 220.0); // tighter pinpoint

            float twinkle = 0.85 + 0.15 * sin(time * 3.5 + hash.y * 62.83);
            float starMag = (hash.x - 0.975) / 0.025;
            // Steep magnitude curve: a few bright stars, many faint ones.
            float starBrightness = (0.35 + 2.2 * starMag * starMag) * starPoint * twinkle * u_StarIntensity;

            // Star color temperature variation (cool blue-white to warm amber)
            vec3 starTint = mix(vec3(0.85, 0.92, 1.0), vec3(1.0, 0.82, 0.60), hash.y);
            nightSky += starTint * starBrightness * smoothstep(0.02, 0.15, d.y) * starDayFade;
        }
    }

    return mix(nightSky, skyRadiance, dayFactor);
}

#endif // ATMOSPHERE_GLSL
