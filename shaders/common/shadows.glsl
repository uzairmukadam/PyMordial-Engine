// ============================================================================
//  Canonical Shadow Sampling Library
//
//  Sampler-agnostic helpers shared by every shadow consumer (deferred resolve,
//  volumetric fog, particles, clouds). Nothing here declares a uniform or a
//  sampler, so a consumer can bind its own atlas at whatever unit it likes.
//
//  Two ideas carry the quality:
//
//  1. Hardware depth comparison. A `sampler2DShadow` tap returns a bilinear
//     filtered 2x2 PCF result, so one fetch resolves what four manual fetches
//     used to.
//
//  2. Receiver-plane depth bias (Isidoro 2006). A receiver's shadow-space depth
//     is linear in shadow UV, so it can be evaluated exactly at every tap
//     instead of approximated by one constant bias. A constant bias must cover
//     the worst tap in the whole footprint, which is exactly what detaches
//     shadows from their casters (peter-panning).
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
//     a "refresh scan" sweeping across the edge as TAA smears the successive
//     patterns together.
//   * A screen-space phase crawls whenever the camera moves, so the edge
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

// ----------------------------------------------------------------------------
//  Per-cascade atlas rect mapping
//
//  `rect` is (offset_x, offset_y, scale_x, scale_y) in atlas UV. Every consumer
//  must derive its UVs from this rect rather than assuming a 2x2 grid.
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
