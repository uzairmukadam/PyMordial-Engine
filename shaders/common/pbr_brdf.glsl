// Cook-Torrance GGX Physically Based Rendering (PBR) & Normal Octahedral Encoding

#ifndef PBR_BRDF_GLSL
#define PBR_BRDF_GLSL

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Octahedral unit-vector encoding (projects sphere to [-1, 1] octahedron)
//
// The lower-hemisphere branch must use a sign function that never returns 0.
// GLSL's built-in sign(0.0) == 0.0, so a normal with an exactly-zero component
// (e.g. a cylinder barrel where n.y == 0) would have that component collapsed
// to 0 in the encoded value, decoding back to a wrong axis-aligned normal.
vec2 OctahedralSignNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 OctahedralEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * OctahedralSignNotZero(n.xy);
    return oct * 0.5 + 0.5; // Map [-1, 1] to [0, 1]
}

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0; // Map [0, 1] to [-1, 1]
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// GGX Normal Distribution Function (Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0000001);
}

// Smith Joint Masking-Shadowing Function
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.0000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Schlick Fresnel Approximation
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ----------------------------------------------------------------------------
//  Multi-scatter energy compensation (Karis 2013 / Fdez-Aguera 2019)
//
//  Single-scatter GGX loses energy as roughness rises: microfacets shadow each
//  other, and the light that would have bounced between them is simply dropped.
//  At roughness 1.0 a rough metal can lose 20-40% of its reflected energy, which
//  reads as "rough metals go dark".
//
//  The fix scales the single-scatter lobe by a factor derived from the
//  directional albedo of the BRDF (E) and its average (E_avg):
//
//      F_ms = F0 * E_avg / (1 - E_avg * (1 - E))
//      scale = 1 + F0 * (1 / E - 1)
//
//  `E` comes from the BRDF LUT's blue channel (see brdf_lut.frag); `E_avg` is
//  the analytic fit below, which is accurate to ~1% over the full roughness
//  range and needs no extra texture.
// ----------------------------------------------------------------------------

// Average directional albedo of the single-scatter GGX BRDF, fitted by
// Fdez-Aguera. Returns the fraction of energy the single-scatter term captures.
float GGXAverageAlbedo(float roughness) {
    float r = clamp(roughness, 0.0, 1.0);
    return 1.0 - 0.28 * r * r;
}

// Multi-scatter compensation factor for a specular lobe.
//
// `E` is the directional albedo at this (NdotV, roughness) from the LUT.
// Returns a multiplier >= 1.0 that restores the missing energy.
vec3 GGXEnergyCompensation(vec3 F0, float roughness, float E) {
    float E_avg = GGXAverageAlbedo(roughness);
    float e = clamp(E, 0.0, 1.0);
    // Guard the denominator: at grazing angles with very rough surfaces the
    // single-scatter albedo approaches 1 and the correction would explode.
    float denom = max(1.0 - E_avg * (1.0 - e), 1e-3);
    vec3 F_ms = F0 * (E_avg / denom);
    vec3 scale = 1.0 + F_ms * (1.0 / max(e, 1e-3) - 1.0);
    // A valid GGX lobe never has E below ~0.7, so a larger correction means the
    // input is degenerate (bad LUT fetch, NaN). Cap it so one bad pixel cannot
    // blow out the frame.
    return clamp(scale, vec3(1.0), vec3(4.0));
}

// ----------------------------------------------------------------------------
//  Analytic environment BRDF (Karis 2013 "mobile" approximation)
//
//  Used where the split-sum LUT is unavailable or where a cheap approximation
//  is preferable to a texture fetch (for example the analytic fallback path).
// ----------------------------------------------------------------------------
vec3 EnvBRDFApprox(vec3 F0, float roughness, float NdotV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return F0 * AB.x + AB.y;
}

// ----------------------------------------------------------------------------
//  Specular anti-aliasing (Toksvig / normal variance)
//
//  A high-frequency normal map produces sub-pixel normal variation that the
//  rasteriser cannot resolve, so the specular highlight aliases and shimmers.
//  Folding the normal's variance into roughness widens the lobe just enough to
//  cover the unresolved detail.
//
//  `variance` is the squared length of the screen-space normal derivative,
//  computed in the G-buffer pass from dFdx/dFdy of the shading normal.
// ----------------------------------------------------------------------------
float SpecularAA(float roughness, float variance, float strength) {
    // Toksvig: a normal whose length is shortened by averaging has variance
    // 1 - |n|^2. Here the caller supplies the derivative-based variance
    // directly, which is cheaper and does not need the averaged normal.
    float v = max(variance, 0.0) * max(strength, 0.0);
    // Convert variance to an equivalent roughness increase. The sqrt keeps the
    // response linear in normal deviation rather than quadratic.
    float widened = sqrt(roughness * roughness + v);
    return clamp(widened, 0.0, 1.0);
}

#endif // PBR_BRDF_GLSL
