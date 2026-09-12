#version 450 core

// ============================================================================
//  Sébastien Hillaire (Eurographics 2020) Transmittance LUT (256x64)
// ============================================================================

in vec2 v_UV;
out vec4 out_Transmittance;

#include "shaders/hillaire_common.glsl"

void main() {
    float r;
    float mu;
    UvToLutTransmittanceParams(v_UV, r, mu);

    vec3 p0 = vec3(0.0, r, 0.0);
    float sinTheta = sqrt(max(0.0, 1.0 - mu * mu));
    vec3 d = vec3(sinTheta, mu, 0.0);

    float t_ground1, t_ground2;
    bool hits_ground = RaySphereIntersect(p0, d, u_RBottom, t_ground1, t_ground2) && (t_ground1 > 0.0);

    float t_top1, t_top2;
    if (!RaySphereIntersect(p0, d, u_RTop, t_top1, t_top2) || t_top2 <= 0.0) {
        out_Transmittance = vec4(1.0);
        return;
    }

    if (hits_ground) {
        // Ray intersects the planet surface: zero transmittance to space
        out_Transmittance = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t_max = t_top2;
    const int SAMPLE_COUNT = 40;
    float dt = t_max / float(SAMPLE_COUNT);

    vec3 optical_depth = vec3(0.0);
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        float t = (float(i) + 0.5) * dt;
        vec3 p = p0 + d * t;
        vec3 sigma_s, sigma_e;
        GetExtinctionScattering(p, sigma_s, sigma_e);
        optical_depth += sigma_e * dt;
    }

    vec3 transmittance = exp(-optical_depth);
    out_Transmittance = vec4(transmittance, 1.0);
}
