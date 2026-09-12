// ============================================================
//  Bruneton Spherical Atmospheric Scattering (GLSL)
//  Eric Bruneton & Sebastien Hillaire spherical atmosphere model
//  Rayleigh & Mie barometric integration with Earth shadow & MS
// ============================================================

#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Planetary Spherical Shell Geometry (Units: meters)
const float RG_EARTH = 6360000.0;   // Planet ground radius: 6360 km
const float RT_EARTH = 6420000.0;   // Atmosphere top radius: 6420 km (60 km shell)
const float HR_EARTH = 8000.0;      // Rayleigh scale height: 8.0 km
const float HM_EARTH = 1200.0;      // Mie scale height: 1.2 km
const float HO_CENTER = 25000.0;    // Ozone layer center altitude: 25.0 km
const float HO_WIDTH = 15000.0;     // Ozone layer half-width: 15.0 km

// Parameterized Atmosphere Uniforms
uniform vec3 u_RayleighBeta = vec3(5.8e-6, 13.5e-6, 33.1e-6); // Scattering coefficients (m^-1)
uniform vec3 u_MieBeta = vec3(21.0e-6);
uniform float u_MieG = 0.78;                                   // Mie forward asymmetry
uniform vec3 u_OzoneBeta = vec3(0.65e-6, 1.88e-6, 0.085e-6);   // Ozone Chappuis band
uniform vec3 u_AtmosphereGroundColor = vec3(0.20, 0.18, 0.15); // Planet surface ambient bounce
uniform vec3 u_NightZenithColor = vec3(0.012, 0.025, 0.05);   // Midnight zenith tint
uniform vec3 u_NightHorizonColor = vec3(0.035, 0.055, 0.09);  // Midnight horizon tint
uniform float u_SunDiscSize = 0.045;                           // Angular radius in radians
uniform float u_MoonDiscSize = 0.040;
uniform vec3 u_MoonColor = vec3(0.70, 0.82, 1.0);
uniform float u_StarIntensity = 1.0;
uniform float u_StarDensity = 1.0;
uniform float u_AtmosphereTurbidity = 2.0;                      // Haze / turbidity multiplier

// Ray-Sphere intersection helper
bool BrunetonRaySphere(vec3 r0, vec3 rd, float rad, out float t1, out float t2) {
    float b = dot(r0, rd);
    float c = dot(r0, r0) - rad * rad;
    float d = b * b - c;
    if (d < 0.0) return false;
    float s = sqrt(d);
    t1 = -b - s;
    t2 = -b + s;
    return true;
}

// Barometric exponential density distributions
void BrunetonGetDensities(vec3 p, out float dR, out float dM, out float dO) {
    float h = max(0.0, length(p) - RG_EARTH);
    dR = exp(-h / HR_EARTH);
    dM = exp(-h / HM_EARTH);
    dO = max(0.0, 1.0 - abs(h - HO_CENTER) / HO_WIDTH);
}

// Optical depth integration along Sun ray with planetary self-shadowing
bool BrunetonSunOpticalDepth(vec3 p, vec3 sun_dir, out vec3 tau_sun) {
    float tg1, tg2;
    // Check if planet ground blocks the sun (Earth twilight shadow)
    if (BrunetonRaySphere(p, sun_dir, RG_EARTH, tg1, tg2)) {
        if (tg1 > 0.0) {
            tau_sun = vec3(1e6);
            return false;
        }
    }
    float tt1, tt2;
    BrunetonRaySphere(p, sun_dir, RT_EARTH, tt1, tt2);
    float t_exit = max(0.0, tt2);

    int sun_steps = 4;
    float ds = t_exit / float(sun_steps);
    float od_r = 0.0, od_m = 0.0, od_o = 0.0;
    for (int j = 0; j < sun_steps; ++j) {
        float sj = (float(j) + 0.5) * ds;
        vec3 ps = p + sun_dir * sj;
        float dr, dm, doo;
        BrunetonGetDensities(ps, dr, dm, doo);
        od_r += dr * ds;
        od_m += dm * ds;
        od_o += doo * ds;
    }
    tau_sun = (u_RayleighBeta * od_r + u_MieBeta * 1.1 * od_m + u_OzoneBeta * od_o) * u_AtmosphereTurbidity;
    return true;
}

vec3 EvaluateAtmosphere(vec3 rayDir, vec3 sunDir, vec3 sunColor, float sunLux, float time) {
    vec3 d = normalize(rayDir);
    vec3 L = normalize(-sunDir);
    float sunElevation = -sunDir.y;

    // Observer position on planet surface (100m elevation above sea level)
    vec3 P0 = vec3(0.0, RG_EARTH + 100.0, 0.0);

    // Night Sky foundation (Moon & Stars)
    float dayFactor = clamp(sunElevation * 6.0 + 0.25, 0.0, 1.0);
    vec3 nightSky = mix(u_NightHorizonColor, u_NightZenithColor, pow(max(d.y, 0.0), 0.7));

    // Lunar disc & halo
    float moonCosTheta = dot(d, sunDir);
    float moonAng = acos(clamp(moonCosTheta, -1.0, 1.0));
    if (moonAng < u_MoonDiscSize && d.y > 0.0) {
        float mNorm = moonAng / u_MoonDiscSize;
        nightSky += u_MoonColor * 3.5 * smoothstep(1.0, 0.85, mNorm);
    }
    nightSky += u_MoonColor * pow(max(moonCosTheta, 0.0), 64.0) * 0.20;

    // Twinkling Starfield
    if (d.y > 0.01 && u_StarIntensity > 0.0) {
        vec3 starCoord = d * (260.0 * u_StarDensity);
        vec3 starCell = floor(starCoord);
        float starHash = fract(sin(dot(starCell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
        if (starHash > 0.985) {
            float twinkle = 0.65 + 0.35 * sin(time * 3.0 + starHash * 100.0);
            float starBrightness = pow((starHash - 0.985) / 0.015, 3.0) * twinkle * u_StarIntensity;
            nightSky += vec3(starBrightness) * smoothstep(0.01, 0.15, d.y);
        }
    }

    if (dayFactor <= 0.001) {
        return nightSky;
    }

    // Intersect view ray with top of atmosphere
    float tt1, tt2;
    if (!BrunetonRaySphere(P0, d, RT_EARTH, tt1, tt2)) {
        return nightSky;
    }
    float t_max = tt2;
    bool hits_ground = false;
    float tg1, tg2;
    if (BrunetonRaySphere(P0, d, RG_EARTH, tg1, tg2)) {
        if (tg1 > 0.0) {
            t_max = min(t_max, tg1);
            hits_ground = true;
        }
    }

    // Numerical raymarching (16 primary sample steps)
    const int STEPS = 16;
    float dt = t_max / float(STEPS);
    float od_r_view = 0.0, od_m_view = 0.0, od_o_view = 0.0;
    vec3 I_R = vec3(0.0);
    vec3 I_M = vec3(0.0);
    vec3 I_MS = vec3(0.0);

    for (int i = 0; i < STEPS; ++i) {
        float ti = (float(i) + 0.5) * dt;
        vec3 pi = P0 + d * ti;

        float dr, dm, doo;
        BrunetonGetDensities(pi, dr, dm, doo);
        od_r_view += dr * dt;
        od_m_view += dm * dt;
        od_o_view += doo * dt;

        vec3 tau_view = (u_RayleighBeta * od_r_view + u_MieBeta * 1.1 * od_m_view + u_OzoneBeta * od_o_view) * u_AtmosphereTurbidity;
        vec3 T_view = exp(-tau_view);

        vec3 tau_sun;
        bool sun_vis = BrunetonSunOpticalDepth(pi, L, tau_sun);
        vec3 T_sun = sun_vis ? exp(-tau_sun) : vec3(0.0);

        // Single scattering accumulation
        I_R += dr * T_view * T_sun * dt;
        I_M += dm * T_view * T_sun * dt;

        // Multiple scattering approximation (diffuse atmospheric ambient fill)
        vec3 psi_ms = max(vec3(0.0), exp(-tau_sun * 0.35)) * 0.15;
        I_MS += (dr * u_RayleighBeta + dm * u_MieBeta) * T_view * psi_ms * dt;
    }

    // Phase Functions
    float cosTheta = dot(d, L);
    // Rayleigh Cornette-Shanks phase
    float PR = (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
    // Mie Cornette-Shanks phase with forward asymmetry g
    float g = clamp(u_MieG, 0.0, 0.99);
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    float PM = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cosTheta * cosTheta)) / ((2.0 + g2) * pow(max(denom, 0.0001), 1.5));

    // Combine day sky radiance
    vec3 sunRadiance = sunColor * sunLux;
    vec3 daySky = (I_R * u_RayleighBeta * PR + I_M * u_MieBeta * PM) * (4.0 * PI) * sunRadiance + I_MS * (4.0 * PI) * sunRadiance;

    // Solar Disc with smooth limb darkening
    if (!hits_ground) {
        float sunAng = acos(clamp(cosTheta, -1.0, 1.0));
        if (sunAng < u_SunDiscSize && sunElevation > -0.05) {
            float normDist = sunAng / u_SunDiscSize;
            float limb = 1.0 - 0.35 * pow(normDist, 1.5);
            vec3 tau_view_total = (u_RayleighBeta * od_r_view + u_MieBeta * 1.1 * od_m_view + u_OzoneBeta * od_o_view) * u_AtmosphereTurbidity;
            vec3 T_disc = exp(-tau_view_total);
            vec3 disc = sunRadiance * 35.0 * limb * T_disc;
            daySky += disc * smoothstep(1.0, 0.85, normDist);
        }
        // Solar Corona glow
        float corona = pow(max(cosTheta, 0.0), 48.0) * 0.4 + pow(max(cosTheta, 0.0), 256.0) * 1.8;
        vec3 tau_view_total = (u_RayleighBeta * od_r_view + u_MieBeta * 1.1 * od_m_view + u_OzoneBeta * od_o_view) * u_AtmosphereTurbidity;
        daySky += sunRadiance * corona * exp(-tau_view_total) * 0.20;
    }

    // Ground bounce reflection below horizon
    if (hits_ground) {
        vec3 tau_ground = (u_RayleighBeta * od_r_view + u_MieBeta * 1.1 * od_m_view + u_OzoneBeta * od_o_view) * u_AtmosphereTurbidity;
        vec3 T_ground = exp(-tau_ground);
        daySky += u_AtmosphereGroundColor * (sunRadiance * max(0.0, sunElevation) * 0.3 + 0.05) * T_ground;
    }

    // Final smooth twilight interpolation
    return mix(nightSky, daySky, dayFactor);
}

#endif // ATMOSPHERE_GLSL
