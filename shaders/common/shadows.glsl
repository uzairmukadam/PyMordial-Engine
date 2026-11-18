// ============================================================================
//  Canonical Shadow Sampling Library
//
//  Sampler-agnostic helpers shared by every shadow consumer (deferred resolve,
//  volumetric fog, particles, clouds). Nothing here declares a uniform or a
//  sampler, so a consumer can bind its own atlas at whatever unit it likes.
//
//  Three ideas carry the quality:
//
//  1. Hardware depth comparison. A `sampler2DShadow` tap returns a bilinear
//     filtered 2x2 PCF result, so one fetch resolves what four manual fetches
//     used to. That is what makes a 32-tap filter affordable at 4K.
//
//  2. Receiver-plane depth bias (Isidoro 2006). A receiver's shadow-space depth
//     is linear in shadow UV, so it can be evaluated exactly at every tap
//     instead of approximated by one constant bias. A constant bias must cover
//     the worst tap in the whole footprint, which is exactly what detaches
//     shadows from their casters (peter-panning).
//
//  3. Decorrelated tap rotation. Per-pixel interleaved gradient noise removes
//     the concentric banding of a fixed disk. The phase is derived from
//     shadow-space position, so the pattern is anchored to the world rather
//     than to the screen or the frame counter.
// ============================================================================

#ifndef SHADOWS_GLSL
#define SHADOWS_GLSL

// Golden-ratio conjugate. Successive frames advance the tap pattern by an
// irrational fraction of a turn, so the union of N frames is maximally spread.
#define SHADOW_GOLDEN_RATIO_CONJ 0.6180339887498949

// Tap rotation phase, locked to shadow-space position.
//
// Deriving the rotation from the shadow-space UV rather than the screen position
// or the frame counter makes the dither pattern world-anchored. That matters for
// two visible defects:
//
//   * A time-varying phase changes the noise pattern every frame, which reads as
//     a "refresh scan" sweeping across the penumbra as TAA smears the successive
//     patterns together.
//   * A screen-space phase crawls whenever the camera moves, so the penumbra
//     shimmers during a pan even though nothing in the scene moved.
//
// Shadow-space UV is a projection of world position, so the pattern is fixed to
// the world: identical frame to frame and identical under camera motion, while
// still decorrelated between neighbouring pixels.
float ShadowTapPhase(vec2 shadow_uv, float texels) {
    vec2 p = shadow_uv * texels;
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))) * 6.28318530718;
}

// Receiver-plane depth at a tap. `gradient` is d(shadow depth) / d(shadow UV),
// so this is exact for planar receivers and a close approximation elsewhere.
float ShadowPlaneDepth(float base_depth, vec2 gradient, vec2 uv, vec2 uv0) {
    return base_depth + dot(gradient, uv - uv0);
}

// World-space penumbra half-width for a receiver/blocker gap, from the physical
// angular radius of the light source.
float ShadowPenumbraWorld(float metric_gap, float angular_radius) {
    return max(metric_gap, 0.0) * angular_radius;
}

// Filter radius in atlas UV, clamped so it never exceeds the region the blocker
// search actually covered. Letting the filter grow past the search footprint is
// what turns a wide penumbra into an unanchored blurry blob.
float ShadowFilterRadius(float penumbra_world, float uv_per_world, float min_radius, float max_radius) {
    return clamp(penumbra_world * uv_per_world, min_radius, max_radius);
}

// Tap count for a disk of `radius_texels`. Scaling with the squared radius keeps
// taps-per-texel roughly constant, so a wide penumbra is not under-sampled into
// visible banding while a tight contact shadow stays cheap.
int ShadowTapCount(float radius_texels, float density, int min_taps, int max_taps) {
    return clamp(int(radius_texels * radius_texels * density), min_taps, max_taps);
}

// ----------------------------------------------------------------------------
//  Per-cascade atlas rect mapping
//
//  `rect` is (offset_x, offset_y, scale_x, scale_y) in atlas UV. Cascades may
//  occupy different fractions of the atlas (non-uniform allocation), so every
//  consumer must derive its UVs from this rect rather than assuming a 2x2 grid.
// ----------------------------------------------------------------------------

vec2 ShadowAtlasUv(vec2 local_uv, vec4 rect) {
    return local_uv * max(rect.zw, vec2(1e-6)) + rect.xy;
}

// Inset bounds keep bilinear taps from bleeding across a neighbouring cascade.
vec2 ShadowAtlasUvMin(vec4 rect, float inset) {
    return rect.xy + max(rect.zw, vec2(1e-6)) * inset;
}

vec2 ShadowAtlasUvMax(vec4 rect, float inset) {
    return rect.xy + max(rect.zw, vec2(1e-6)) * (1.0 - inset);
}

#endif // SHADOWS_GLSL
