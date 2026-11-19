// ============================================================================
//  Radiance Cascade Interval Composition
//
//  Composes a NEAR interval N with a FAR interval F:
//
//      merged.rgb = N.rgb + N.a * F.rgb
//      merged.a   = N.a * F.a
//
//  Semantics:
//      RGB = radiance accumulated inside the interval
//      A   = transmittance remaining after the interval (1 = fully open)
//
//  The near interval is in front, so its radiance is added first and only the
//  light that survives it (N.a) reaches the far interval. This is the standard
//  front-to-back compositing operator and is associative, which is what makes
//  the three-stage merge (C2 -> C1 -> C0) valid.
//
//  Clamp policy (explicit): radiance is clamped to [0, 1e4] and transmittance
//  to [0, 1] on write. NaN/Inf inputs are replaced with a fully transparent,
//  zero-radiance interval so a single bad probe cannot poison the chain.
// ============================================================================

#ifndef RC_COMPOSITE_GLSL
#define RC_COMPOSITE_GLSL

const float RC_MAX_RADIANCE = 1e4;

// Sanitises one interval sample.
vec4 RCSanitizeInterval(vec4 interval) {
    if (any(isnan(interval)) || any(isinf(interval))) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return vec4(clamp(interval.rgb, vec3(0.0), vec3(RC_MAX_RADIANCE)), clamp(interval.a, 0.0, 1.0));
}

// Composes a near interval in front of a far interval.
vec4 RCCompositeIntervals(vec4 nearInterval, vec4 farInterval) {
    vec4 n = RCSanitizeInterval(nearInterval);
    vec4 f = RCSanitizeInterval(farInterval);
    vec3 radiance = n.rgb + n.a * f.rgb;
    // Clamp the composed result too: two individually valid intervals can sum
    // past the ceiling.
    return vec4(clamp(radiance, vec3(0.0), vec3(RC_MAX_RADIANCE)), clamp(n.a * f.a, 0.0, 1.0));
}

#endif // RC_COMPOSITE_GLSL
