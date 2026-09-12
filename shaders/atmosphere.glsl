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
uniform vec3 u_AtmosphereSunDir = vec3(0.35, -0.85, 0.40);    // True astronomical sun vector
uniform float u_SunDiscSize = 0.045;                           // Solar disc angular radius (radians)
uniform float u_MoonDiscSize = 0.040;                          // Lunar disc angular radius (radians)
uniform vec3 u_MoonColor = vec3(0.70, 0.82, 1.0);
uniform float u_StarIntensity = 1.0;
uniform float u_StarDensity = 1.0;

vec3 EvaluateAtmosphere(vec3 rayDir, vec3 sunDir, vec3 sunColor, float sunLux, float time, vec3 cameraPos) {
    vec3 d = normalize(rayDir);
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
            float disc_edge = smoothstep(1.0, 0.85, normDist);
            skyRadiance += disc * disc_edge;
        }

        // Forward Mie Corona Glow around sun
        float corona = pow(max(cosTheta, 0.0), 48.0) * 0.35 + pow(max(cosTheta, 0.0), 256.0) * 1.5;
        skyRadiance += sunRadiance * corona * T_sun * 0.25;
    }

    // 3. Night Sky foundation (Lunar disc & Twinkling Starfield)
    float dayFactor = smoothstep(-0.06, 0.02, sunElevation);
    vec3 nightSky = mix(u_NightHorizonColor, u_NightZenithColor, pow(max(d.y, 0.0), 0.7));

    // Lunar disc & halo (facing opposite the sun)
    float moonCosTheta = dot(d, sunDir);
    float moonAng = acos(clamp(moonCosTheta, -1.0, 1.0));
    if (moonAng < u_MoonDiscSize && d.y > 0.0) {
        float mNorm = moonAng / u_MoonDiscSize;
        nightSky += u_MoonColor * 12.0 * smoothstep(1.0, 0.85, mNorm);
    }
    nightSky += u_MoonColor * pow(max(moonCosTheta, 0.0), 64.0) * 0.80;

    // Twinkling Starfield (naturally fades out in daylight)
    float starDayFade = 1.0 - smoothstep(0.0, 0.12, dayFactor);
    if (d.y > 0.02 && u_StarIntensity > 0.0 && starDayFade > 0.001) {
        // High-frequency celestial spherical mapping
        float phi = atan(d.z, d.x);
        float theta = asin(clamp(d.y, -1.0, 1.0));
        vec2 starCoord = vec2(phi, theta) * (180.0 * u_StarDensity);

        vec2 cell = floor(starCoord);
        vec2 f = fract(starCoord);

        // Fast hash for this celestial grid cell
        vec2 hash = fract(sin(vec2(dot(cell, vec2(127.1, 311.7)), dot(cell, vec2(269.5, 183.3)))) * 43758.5453);

        if (hash.x > 0.92) {
            vec2 starPos = 0.25 + 0.50 * hash; // jittered within cell interior
            float dist = length(f - starPos);
            float starPoint = exp(-dist * dist * 140.0); // Fine pinpoint star

            float twinkle = 0.70 + 0.30 * sin(time * 3.5 + hash.y * 62.83);
            float starMag = (hash.x - 0.92) / 0.08;
            float starBrightness = (2.0 + 8.0 * starMag) * starPoint * twinkle * u_StarIntensity;

            // Star color temperature variation (cool blue-white to warm amber)
            vec3 starTint = mix(vec3(0.85, 0.92, 1.0), vec3(1.0, 0.82, 0.60), hash.y);
            nightSky += starTint * starBrightness * smoothstep(0.02, 0.15, d.y) * starDayFade;
        }
    }

    return mix(nightSky, skyRadiance, dayFactor);
}

#endif // ATMOSPHERE_GLSL
