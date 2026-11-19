// ============================================================================
//  Hi-Z Depth Bounds — shared constants and types
//
//  The bounds hierarchy stores a conservative depth INTERVAL per cell:
//
//      R = nearest valid positive linear view depth
//      G = farthest valid positive linear view depth
//
//  Invariants:
//    * Valid geometry: nearest <= farthest, both > 0.
//    * Mip zero: nearest == farthest (a single surface per pixel).
//    * Empty cell: nearest > farthest. The canonical empty pair is
//      (HIZ_EMPTY_NEAREST, HIZ_EMPTY_FARTHEST) = (large finite, 0.0), so a
//      plain min/max reduction over children naturally ignores empty cells:
//      min() skips the large sentinel and max() skips the 0.0.
//
//  Linear view depth is positive and SMALLER IS NEARER, which is the opposite
//  ordering from reversed-Z hardware depth. See shaders/common/depth.glsl.
// ============================================================================

#ifndef HIZ_BOUNDS_GLSL
#define HIZ_BOUNDS_GLSL

// Large but finite: large enough that min() never selects it over real
// geometry, small enough that it stays well inside float range so arithmetic
// on it cannot overflow to Inf.
const float HIZ_EMPTY_NEAREST = 1e30;
const float HIZ_EMPTY_FARTHEST = 0.0;

struct HiZDepthBounds {
    float nearest;
    float farthest;
};

// True when the cell contains no geometry (nearest > farthest).
bool HiZBoundsEmpty(HiZDepthBounds bounds) {
    return bounds.nearest > bounds.farthest;
}

// ----------------------------------------------------------------------------
//  Sampling
// ----------------------------------------------------------------------------

// Clamps a requested mip to the levels the texture actually has.
int HiZClampMip(sampler2D hizTexture, int mip) {
    int maxMip = max(textureQueryLevels(hizTexture) - 1, 0);
    return clamp(mip, 0, maxMip);
}

// Loads the bounds of one exact cell at `mip`.
//
// Out-of-range cells are clamped to the texture edge rather than rejected, so a
// caller that walks slightly past the border still gets a defined interval.
HiZDepthBounds LoadHiZBounds(sampler2D hizTexture, ivec2 cell, int mip) {
    int level = HiZClampMip(hizTexture, mip);
    ivec2 size = textureSize(hizTexture, level);
    ivec2 clamped = clamp(cell, ivec2(0), max(size - 1, ivec2(0)));
    vec2 raw = texelFetch(hizTexture, clamped, level).rg;
    return HiZDepthBounds(raw.x, raw.y);
}

// Samples the bounds at a UV with bilinear filtering at `mip`.
//
// Bilinear interpolation of a min/max pair is NOT conservative: the filtered
// nearest can be larger than the true nearest and the filtered farthest smaller
// than the true farthest. Callers that need a conservative test must use
// LoadHiZBounds() on the covering cells instead.
HiZDepthBounds SampleHiZBounds(sampler2D hizTexture, vec2 uv, int mip) {
    int level = HiZClampMip(hizTexture, mip);
    vec2 raw = textureLod(hizTexture, clamp(uv, vec2(0.0), vec2(1.0)), float(level)).rg;
    return HiZDepthBounds(raw.x, raw.y);
}

// Converts a UV into the cell index that contains it at `mip`.
ivec2 HiZCellFromUV(sampler2D hizTexture, vec2 uv, int mip) {
    int level = HiZClampMip(hizTexture, mip);
    ivec2 size = textureSize(hizTexture, level);
    ivec2 cell = ivec2(floor(clamp(uv, vec2(0.0), vec2(1.0)) * vec2(size)));
    return clamp(cell, ivec2(0), max(size - 1, ivec2(0)));
}

// ----------------------------------------------------------------------------
//  Interval comparison
// ----------------------------------------------------------------------------

// Tests whether a ray's depth interval can intersect a cell's geometry interval.
//
// Interval semantics (all values are positive linear view depth):
//   * Both intervals are treated as CLOSED: touching endpoints count as overlap.
//   * The ray interval is normalised, so a reversed (nearest > farthest) input
//     is accepted and swapped.
//   * `thickness` expands the geometry interval symmetrically, which models a
//     surface of finite depth rather than an infinitely thin shell.
//   * An empty cell never overlaps, regardless of the ray interval.
//
// This is a pure geometric predicate. It deliberately contains no effect
// policy: no bias, no confidence, no distance fade.
bool DepthIntervalsOverlap(
    float rayNearest,
    float rayFarthest,
    HiZDepthBounds geometry,
    float thickness
) {
    if (HiZBoundsEmpty(geometry)) {
        return false;
    }

    float rayMin = min(rayNearest, rayFarthest);
    float rayMax = max(rayNearest, rayFarthest);

    float expand = max(thickness, 0.0);
    float geoMin = geometry.nearest - expand;
    float geoMax = geometry.farthest + expand;

    return rayMin <= geoMax && rayMax >= geoMin;
}

#endif // HIZ_BOUNDS_GLSL
