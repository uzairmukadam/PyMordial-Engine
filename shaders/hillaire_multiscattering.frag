#version 450 core

// ============================================================================
//  Sébastien Hillaire (Eurographics 2020) Multiple Scattering LUT (32x32)
//  Evaluates 2nd-order and infinite geometric-series atmospheric scattering
// ============================================================================

in vec2 v_UV;
out vec4 out_MultiScattering;

layout (binding = 0) uniform sampler2D u_TransmittanceLUT;

#include "shaders/hillaire_common.glsl"

vec3 GetSunTransmittance(vec3 p, vec3 L) {
    float r = length(p);
    float mu_s = dot(p, L) / max(r, 1e-4);

    float tg1, tg2;
    if (RaySphereIntersect(p, L, u_RBottom, tg1, tg2) && tg1 > 0.0) {
        return vec3(0.0);
    }

    vec2 uv = LutTransmittanceParamsToUv(r, mu_s);
    return texture(u_TransmittanceLUT, uv).rgb;
}

void main() {
    float mu_s = clamp(v_UV.x * 2.0 - 1.0, -1.0, 1.0);
    float r = u_RBottom + clamp(v_UV.y, 0.0, 1.0) * (u_RTop - u_RBottom);

    vec3 p0 = vec3(0.0, r, 0.0);
    float sinSun = sqrt(max(0.0, 1.0 - mu_s * mu_s));
    vec3 L = vec3(sinSun, mu_s, 0.0);

    // Integrate over 64 directions uniformly distributed across the sphere
    const int NUM_DIR_SAMPLES = 64;
    vec3 L2nd_sum = vec3(0.0);
    vec3 Fms_sum = vec3(0.0);

    for (int i = 0; i < NUM_DIR_SAMPLES; ++i) {
        // Fibonacci sphere point distribution
        float fi = float(i) + 0.5;
        float cosTheta = 1.0 - 2.0 * fi / float(NUM_DIR_SAMPLES);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi = fi * 3.883222077; // Golden angle * 2 ~= 2.39996323 * golden ratio
        vec3 d = vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);

        float t_ground1, t_ground2;
        bool hits_ground = RaySphereIntersect(p0, d, u_RBottom, t_ground1, t_ground2) && (t_ground1 > 0.0);

        float t_top1, t_top2;
        RaySphereIntersect(p0, d, u_RTop, t_top1, t_top2);
        float t_max = hits_ground ? t_ground1 : max(0.0, t_top2);

        const int MARCH_STEPS = 20;
        float dt = t_max / float(MARCH_STEPS);

        vec3 optical_depth = vec3(0.0);
        vec3 L_dir = vec3(0.0);
        vec3 fms_dir = vec3(0.0);

        float cosSunView = dot(d, L);
        float phaseR = RayleighPhase(cosSunView);
        float phaseM = CornetteShanksMiePhase(cosSunView, u_MieG);

        for (int j = 0; j < MARCH_STEPS; ++j) {
            float t = (float(j) + 0.5) * dt;
            vec3 pj = p0 + d * t;

            float dR, dM, dO;
            GetAtmosphereDensities(pj, dR, dM, dO);
            vec3 sigma_s, sigma_e;
            GetExtinctionScattering(pj, sigma_s, sigma_e);

            optical_depth += sigma_e * dt;
            vec3 T_view = exp(-optical_depth);
            vec3 T_sun = GetSunTransmittance(pj, L);

            vec3 rayleigh_scat = u_RayleighBeta * dR;
            vec3 mie_scat = vec3(u_MieBetaScat) * dM;

            // 1st order in-scattering
            vec3 S = (rayleigh_scat * phaseR + mie_scat * phaseM) * T_sun;
            L_dir += S * T_view * dt;
            fms_dir += sigma_s * T_view * dt;
        }

        // Ground albedo reflection
        if (hits_ground) {
            vec3 p_ground = p0 + d * t_ground1;
            vec3 N_ground = normalize(p_ground);
            float NdotL = max(0.0, dot(N_ground, L));
            vec3 T_sun_ground = GetSunTransmittance(p_ground, L);
            vec3 T_view_ground = exp(-optical_depth);
            vec3 ground_radiance = (u_GroundAlbedo / PI) * NdotL * T_sun_ground;
            L_dir += ground_radiance * T_view_ground;
        }

        L2nd_sum += L_dir;
        Fms_sum += fms_dir;
    }

    vec3 L1 = L2nd_sum / float(NUM_DIR_SAMPLES);
    vec3 Fms = clamp(Fms_sum / float(NUM_DIR_SAMPLES), vec3(0.0), vec3(0.999));

    // Sébastien Hillaire geometric series for infinite scattering bounces
    vec3 psi_ms = L1 / max(vec3(1.0) - Fms, vec3(1e-4));

    out_MultiScattering = vec4(psi_ms, 1.0);
}
