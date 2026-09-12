// ============================================================================
//  Sébastien Hillaire (Eurographics 2020) Physically Based Sky & Atmosphere
//  Common Mathematical Formulations & Coordinate Transformations
// ============================================================================

#ifndef HILLAIRE_COMMON_GLSL
#define HILLAIRE_COMMON_GLSL

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Default Physical Planetary Constants (Earth-scale, units: meters)
const float DEFAULT_R_BOTTOM = 6360000.0; // 6360 km
const float DEFAULT_R_TOP    = 6460000.0; // 6460 km (100 km atmospheric shell)

const float DEFAULT_H_R = 8000.0;         // Rayleigh scale height: 8.0 km
const float DEFAULT_H_M = 1200.0;         // Mie scale height: 1.2 km
const float DEFAULT_HO_CENTER = 25000.0;  // Ozone layer peak center: 25.0 km
const float DEFAULT_HO_WIDTH  = 15000.0;  // Ozone layer half-width: 15.0 km

// Uniform parameters for real-time customizable planets & atmospheres
uniform float u_RBottom = 6360000.0;
uniform float u_RTop    = 6460000.0;

uniform float u_H_R = 8000.0;
uniform float u_H_M = 1200.0;
uniform float u_HO_Center = 25000.0;
uniform float u_HO_Width  = 15000.0;

uniform vec3  u_RayleighBeta = vec3(5.802e-6, 13.558e-6, 33.100e-6);
uniform float u_MieBetaScat  = 3.996e-6;
uniform float u_MieBetaExt   = 4.440e-6;
uniform float u_MieG         = 0.80;
uniform vec3  u_OzoneBeta    = vec3(0.650e-6, 1.881e-6, 0.085e-6);

uniform vec3  u_GroundAlbedo = vec3(0.15, 0.15, 0.15);
uniform float u_Turbidity    = 1.0;

// Ray-Sphere intersection helper
bool RaySphereIntersect(vec3 p, vec3 d, float rad, out float t1, out float t2) {
    float b = dot(p, d);
    float c = dot(p, p) - rad * rad;
    float delta = b * b - c;
    if (delta < 0.0) return false;
    float s = sqrt(delta);
    t1 = -b - s;
    t2 = -b + s;
    return true;
}

// Atmosphere component densities at world position p (centered at origin)
void GetAtmosphereDensities(vec3 p, out float dR, out float dM, out float dO) {
    float h = max(0.0, length(p) - u_RBottom);
    dR = exp(-h / u_H_R);
    dM = exp(-h / u_H_M);
    dO = max(0.0, 1.0 - abs(h - u_HO_Center) / u_HO_Width);
}

// Extinction and scattering coefficients at point p
void GetExtinctionScattering(vec3 p, out vec3 sigma_s, out vec3 sigma_e) {
    float dR, dM, dO;
    GetAtmosphereDensities(p, dR, dM, dO);

    vec3 rayleigh_scat = u_RayleighBeta * dR;
    vec3 mie_scat = vec3(u_MieBetaScat) * dM;
    vec3 mie_ext = vec3(u_MieBetaExt) * dM;
    vec3 ozone_ext = u_OzoneBeta * dO;

    sigma_s = rayleigh_scat + mie_scat;
    sigma_e = (rayleigh_scat + mie_ext + ozone_ext) * u_Turbidity;
}

// Phase Functions
float RayleighPhase(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

float CornetteShanksMiePhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cosTheta * cosTheta)) / ((2.0 + g2) * pow(max(denom, 1e-4), 1.5));
}

// ============================================================================
//  Transmittance LUT Coordinate Mappings
// ============================================================================

void UvToLutTransmittanceParams(in vec2 uv, out float r, out float mu) {
    float H = sqrt(max(0.0, u_RTop * u_RTop - u_RBottom * u_RBottom));
    float rho = H * uv.y;
    r = sqrt(rho * rho + u_RBottom * u_RBottom);
    float d_min = u_RTop - r;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);
    mu = (d == 0.0) ? 1.0 : (H * H - d * d - rho * rho) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec2 LutTransmittanceParamsToUv(in float r, in float mu) {
    float H = sqrt(max(0.0, u_RTop * u_RTop - u_RBottom * u_RBottom));
    float rho = sqrt(max(0.0, r * r - u_RBottom * u_RBottom));
    float discriminant = r * r * (mu * mu - 1.0) + u_RTop * u_RTop;
    float d = max(0.0, (-r * mu + sqrt(max(0.0, discriminant))));
    float d_min = u_RTop - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / max(d_max - d_min, 1e-4);
    float x_r = rho / max(H, 1e-4);
    return vec2(clamp(x_mu, 0.0, 1.0), clamp(x_r, 0.0, 1.0));
}

// ============================================================================
//  Sky-View LUT Coordinate Mappings
// ============================================================================

void UvToSkyViewParams(in vec2 uv, in float r, out float viewZenith, out float sunAzimuth) {
    // Non-linear azimuth mapping: uv.x in [0, 1] -> [0, PI]
    sunAzimuth = uv.x * uv.x * PI;

    float sinHorizon = u_RBottom / r;
    float horizonZenith = PI - acos(clamp(sqrt(max(0.0, 1.0 - sinHorizon * sinHorizon)), 0.0, 1.0));

    if (uv.y < 0.5) {
        // Ground / below horizon
        float coord = 1.0 - uv.y * 2.0;
        viewZenith = horizonZenith + coord * coord * (PI - horizonZenith);
    } else {
        // Sky / above horizon
        float coord = (uv.y - 0.5) * 2.0;
        viewZenith = horizonZenith - coord * coord * horizonZenith;
    }
}

vec2 SkyViewParamsToUv(in float viewZenith, in float sunAzimuth, in float r) {
    float u = sqrt(clamp(sunAzimuth / PI, 0.0, 1.0));

    float sinHorizon = u_RBottom / r;
    float horizonZenith = PI - acos(clamp(sqrt(max(0.0, 1.0 - sinHorizon * sinHorizon)), 0.0, 1.0));

    float v;
    if (viewZenith > horizonZenith) {
        float coord = sqrt(clamp((viewZenith - horizonZenith) / max(PI - horizonZenith, 1e-4), 0.0, 1.0));
        v = 0.5 * (1.0 - coord);
    } else {
        float coord = sqrt(clamp((horizonZenith - viewZenith) / max(horizonZenith, 1e-4), 0.0, 1.0));
        v = 0.5 + 0.5 * coord;
    }
    return vec2(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
}

#endif // HILLAIRE_COMMON_GLSL
