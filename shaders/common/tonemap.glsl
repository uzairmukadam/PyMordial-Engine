// ============================================================================
//  Canonical Tone-Mapping & Display Encoding Library
//
//  Every operator here takes linear scene radiance (unbounded, in the working
//  space) and returns display-referred linear values in [0, 1]. The caller is
//  responsible for the final sRGB encode via LinearToSRGB().
//
//  Operators:
//    * TonemapAgX          - Sobotka 2022 AgX (log2 encode, inset/outset
//                            matrices, sigmoid polynomial). Hue-preserving
//                            highlight desaturation.
//    * TonemapACESFitted   - Stephen Hill's fit of the ACES RRT+ODT
//                            (ACEScg -> sRGB primaries, RRT+ODT, inverse).
//    * TonemapNeutral      - Khronos PBR Neutral (Khronos 2024). Designed for
//                            product/asset viewers: preserves albedo hue and
//                            saturation in the midtones.
//    * TonemapReinhard     - Legacy extended Reinhard, kept for A/B.
//
//  All matrices are column-major mat3 constructors, matching GLSL's
//  column-major convention.
// ============================================================================

#ifndef TONEMAP_GLSL
#define TONEMAP_GLSL

#ifndef PI
#define PI 3.14159265359
#endif

// ----------------------------------------------------------------------------
//  Display encoding
// ----------------------------------------------------------------------------

// Exact sRGB transfer function (IEC 61966-2-1). The linear segment near black
// matters: a pure pow(1/2.2) crushes shadow detail and shifts the black point.
vec3 LinearToSRGB(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 SRGBToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

// ----------------------------------------------------------------------------
//  AgX (Sobotka 2022)
//
//  The pipeline is: linear -> log2 encode -> inset matrix (compress gamut
//  toward the achromatic axis) -> sigmoid polynomial -> outset matrix (restore
//  saturation) -> linear. The inset/outset pair is what produces the
//  characteristic AgX highlight roll-off: bright saturated colours desaturate
//  toward white instead of clipping to a hue.
// ----------------------------------------------------------------------------

const mat3 AGX_INSET = mat3(
    0.856627153315983, 0.137318972929847, 0.111898212999950,
    0.0951212405381588, 0.761241990602591, 0.0767994186031903,
    0.0482516061458583, 0.101439036467562, 0.811302368396859
);

const mat3 AGX_OUTSET = mat3(
     1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
    -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
    -0.016493938717834573, -0.016493938717834257, 1.2519364065950405
);

// AgX sigmoid, evaluated on the log2-encoded, inset value.
vec3 AgXSigmoid(vec3 x) {
    // Six-degree minimax fit of the AgX contrast curve.
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
         - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 TonemapAgX(vec3 color) {
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;

    // 1. Inset: compress the gamut toward achromatic. This must happen in
    //    LINEAR space, before the log encode — applying a matrix to a
    //    log-encoded value is not the same transform and lifts the blacks.
    vec3 v = AGX_INSET * clamp(color, 0.0, 65504.0);

    // 2. Log2 encode into the AgX working range.
    v = clamp(log2(max(v, 1e-10)), min_ev, max_ev);
    v = (v - min_ev) / (max_ev - min_ev);

    // 3. Sigmoid contrast curve.
    v = AgXSigmoid(v);

    // 4. Outset: restore saturation.
    v = AGX_OUTSET * v;

    return clamp(v, 0.0, 1.0);
}

// ----------------------------------------------------------------------------
//  ACES (Stephen Hill fit of RRT+ODT)
//
//  Converts to ACEScg, applies the RRT+ODT approximation, converts back to
//  sRGB primaries. This is the reference "filmic" look used by RAGE and most
//  Unreal-derived pipelines.
// ----------------------------------------------------------------------------

const mat3 ACES_INPUT = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

const mat3 ACES_OUTPUT = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 TonemapACESFitted(vec3 color) {
    color = ACES_INPUT * max(color, 0.0);
    color = RRTAndODTFit(color);
    color = ACES_OUTPUT * color;
    return clamp(color, 0.0, 1.0);
}

// ----------------------------------------------------------------------------
//  Khronos PBR Neutral (2024)
//
//  Preserves albedo hue and saturation in the midtones and only desaturates
//  where the display would otherwise clip. Preferred for asset viewers and
//  any scene where material colour fidelity matters more than film response.
// ----------------------------------------------------------------------------

vec3 TonemapNeutral(vec3 color) {
    const float start_compression = 0.8 - 0.04;
    const float desaturation = 0.15;

    vec3 c = max(color, 0.0);
    float x = min(c.r, min(c.g, c.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= offset;

    float peak = max(c.r, max(c.g, c.b));
    if (peak < start_compression) {
        return c;
    }

    const float d = 1.0 - start_compression;
    float new_peak = 1.0 - d * d / (peak + d - start_compression);
    c *= new_peak / peak;

    // Desaturate toward the achromatic axis as the peak approaches 1.0.
    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3(new_peak), g);
}

// ----------------------------------------------------------------------------
//  Legacy extended Reinhard
// ----------------------------------------------------------------------------

vec3 TonemapReinhard(vec3 x) {
    const float white = 4.0;
    vec3 num = x * (1.0 + x / (white * white));
    return clamp(num / (1.0 + x), 0.0, 1.0);
}

// ----------------------------------------------------------------------------
//  Dispatch
// ----------------------------------------------------------------------------

// mode: 0 = ACES fitted, 1 = AgX, 2 = Neutral, 3 = Reinhard
vec3 TonemapDispatch(vec3 color, int mode) {
    if (mode == 1) {
        return TonemapAgX(color);
    }
    if (mode == 2) {
        return TonemapNeutral(color);
    }
    if (mode == 3) {
        return TonemapReinhard(color);
    }
    return TonemapACESFitted(color);
}

#endif // TONEMAP_GLSL
