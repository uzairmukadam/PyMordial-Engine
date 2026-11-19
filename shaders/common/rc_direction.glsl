// ============================================================================
//  Radiance Cascade Direction Encoding
//
//  Every cascade samples the same Fibonacci sphere of RC_MAX_DIRECTIONS
//  directions and uses a prefix of it. A cascade with `direction_count`
//  directions therefore owns global direction indices 0 .. direction_count-1.
//
//  This makes near-to-far direction mapping an exact index match: a near
//  cascade always has fewer directions than a far cascade, so direction `d` in
//  the near cascade is the same world direction as direction `d` in the far
//  cascade. No resampling or rotation is required.
//
//  Directions are expressed in VIEW space (camera looks down -Z).
// ============================================================================

#ifndef RC_DIRECTION_GLSL
#define RC_DIRECTION_GLSL

const int RC_MAX_DIRECTIONS = 32;
const float RC_PI = 3.14159265358979323846;
const float RC_GOLDEN_ANGLE = 2.39996322972865332;

// Returns the unit direction for a global direction index.
//
// Uses a spherical Fibonacci lattice: index i of N maps to
//     z   = 1 - 2 * (i + 0.5) / N
//     phi = i * golden_angle
// which distributes points evenly over the whole sphere. The full sphere is
// used (not a hemisphere) because a probe's interval radiance is directionally
// complete: the receiver later applies its own cosine weighting.
vec3 RCDirection(int index, int count) {
    int n = clamp(count, 1, RC_MAX_DIRECTIONS);
    int i = clamp(index, 0, n - 1);
    float z = 1.0 - 2.0 * (float(i) + 0.5) / float(n);
    float r = sqrt(max(1.0 - z * z, 0.0));
    float phi = float(i) * RC_GOLDEN_ANGLE;
    return vec3(r * cos(phi), r * sin(phi), z);
}

// Solid angle represented by one direction of an N-direction sphere.
float RCDirectionSolidAngle(int count) {
    return 4.0 * RC_PI / float(clamp(count, 1, RC_MAX_DIRECTIONS));
}

#endif // RC_DIRECTION_GLSL
