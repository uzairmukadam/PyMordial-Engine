#version 450 core

// ============================================================================
//  Sébastien Hillaire (Eurographics 2020) Sky-View LUT (192x108)
//  Real-time dynamic sky luminance LUT parameterized by view zenith & sun azimuth
// ============================================================================

in vec2 v_UV;
out vec4 out_SkyView;

layout (binding = 0) uniform sampler2D u_TransmittanceLUT;
layout (binding = 1) uniform sampler2D u_MultiScatteringLUT;

uniform vec3 u_CameraWorldPos; // Camera position relative to planet center
uniform vec3 u_SunDirection;    // Direction toward the sun (opposite to light vector)
uniform vec3 u_SunColor = vec3(1.0, 1.0, 1.0);
uniform float u_SunIntensity = 4.0;

#include "shaders/hillaire_common.glsl"

vec3 SampleSunTransmittance(vec3 p, vec3 L) {
    float r = length(p);
    float mu_s = dot(p, L) / max(r, 1e-4);

    float tg1, tg2;
    if (RaySphereIntersect(p, L, u_RBottom, tg1, tg2) && tg1 > 0.0) {
        return vec3(0.0);
    }

    vec2 uv = LutTransmittanceParamsToUv(r, mu_s);
    return texture(u_TransmittanceLUT, uv).rgb;
}

vec3 SampleMultiScattering(vec3 p, vec3 L) {
    float r = length(p);
    float mu_s = dot(p, L) / max(r, 1e-4);

    float u = clamp(mu_s * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - u_RBottom) / max(u_RTop - u_RBottom, 1e-4), 0.0, 1.0);
    return texture(u_MultiScatteringLUT, vec2(u, v)).rgb;
}

void main() {
    vec3 p0 = u_CameraWorldPos;
    float r = length(p0);

    // If camera is slightly below ground, clamp to just above surface (10 meters)
    if (r < u_RBottom + 10.0) {
        p0 = normalize(p0) * (u_RBottom + 10.0);
        r = u_RBottom + 10.0;
    }

    float viewZenith, sunAzimuth;
    UvToSkyViewParams(v_UV, r, viewZenith, sunAzimuth);

    vec3 N = normalize(p0);
    vec3 L = normalize(u_SunDirection);

    // Construct orthonormal tangent-space basis aligned with local zenith and sun
    vec3 Ts = L - dot(L, N) * N;
    float lenTs = length(Ts);
    if (lenTs > 1e-5) {
        Ts /= lenTs;
    } else {
        Ts = abs(N.y) < 0.99 ? normalize(cross(N, vec3(0.0, 1.0, 0.0))) : normalize(cross(N, vec3(1.0, 0.0, 0.0)));
    }
    vec3 Bs = cross(N, Ts);

    // Compute view direction from angles
    float sinZenith = sin(viewZenith);
    float cosZenith = cos(viewZenith);
    float cosAzimuth = cos(sunAzimuth);
    float sinAzimuth = sin(sunAzimuth);

    vec3 d = normalize(N * cosZenith + Ts * (sinZenith * cosAzimuth) + Bs * (sinZenith * sinAzimuth));

    // Ray-sphere intersections
    float t_ground1, t_ground2;
    bool hits_ground = RaySphereIntersect(p0, d, u_RBottom, t_ground1, t_ground2) && (t_ground1 > 0.0);

    float t_top1, t_top2;
    RaySphereIntersect(p0, d, u_RTop, t_top1, t_top2);
    float t_max = hits_ground ? t_ground1 : max(0.0, t_top2);

    // 32-step raymarching
    const int STEPS = 32;
    float dt = t_max / float(STEPS);

    vec3 optical_depth = vec3(0.0);
    vec3 L_sky = vec3(0.0);

    float cosTheta = dot(d, L);
    float phaseR = RayleighPhase(cosTheta);
    float phaseM = CornetteShanksMiePhase(cosTheta, u_MieG);

    vec3 sunRadiance = u_SunColor * u_SunIntensity;

    for (int i = 0; i < STEPS; ++i) {
        float t = (float(i) + 0.5) * dt;
        vec3 pi = p0 + d * t;

        float dR, dM, dO;
        GetAtmosphereDensities(pi, dR, dM, dO);
        vec3 sigma_s, sigma_e;
        GetExtinctionScattering(pi, sigma_s, sigma_e);

        optical_depth += sigma_e * dt;
        vec3 T_view = exp(-optical_depth);
        vec3 T_sun = SampleSunTransmittance(pi, L);
        vec3 L_ms = SampleMultiScattering(pi, L);

        vec3 rayleigh_scat = u_RayleighBeta * dR;
        vec3 mie_scat = vec3(u_MieBetaScat) * dM;

        // Single scattering + multi-scattering
        vec3 S = (rayleigh_scat * phaseR + mie_scat * phaseM) * T_sun * sunRadiance + sigma_s * L_ms * sunRadiance;
        L_sky += S * T_view * dt;
    }

    // Ground bounce reflection if ray hits terrain
    if (hits_ground) {
        vec3 p_ground = p0 + d * t_ground1;
        vec3 N_ground = normalize(p_ground);
        float NdotL = max(0.0, dot(N_ground, L));
        vec3 T_sun_ground = SampleSunTransmittance(p_ground, L);
        vec3 T_view_ground = exp(-optical_depth);
        vec3 ground_radiance = (u_GroundAlbedo / PI) * NdotL * T_sun_ground * sunRadiance;
        L_sky += ground_radiance * T_view_ground;
    }

    out_SkyView = vec4(L_sky, 1.0);
}
