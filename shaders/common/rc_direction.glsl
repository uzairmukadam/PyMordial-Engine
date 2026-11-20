// ============================================================================
//  Radiance Cascade Direction Encoding
//
//  Directions are expressed in VIEW space (camera looks down -Z).
//
//  The three cascade direction counts use a HIERARCHICAL equal-solid-angle
//  layout instead of independent Fibonacci sets. Uniform sampling in azimuth
//  phi and in z = cos(theta) gives equal solid-angle cells because spherical
//  solid angle is dOmega = dphi * dz, so every direction owns exactly 4*pi/N.
//
//      count =  16 :  4 azimuth bins x 4 z bins
//      count =  64 :  8 azimuth bins x 8 z bins
//      count = 256 : 16 azimuth bins x 16 z bins
//
//  Index bit layouts are chosen so the sets NEST with FOUR children per parent:
//  a parent direction d at count N has children 4*d .. 4*d+3 at count 4N, and
//  those four children tile the parent's cell exactly. The two newly introduced
//  bits are always the least significant bits of the child index.
//
//      count =  16 : [az1 az0 z1 z0]                 az = i >> 2,  z = i & 3
//      count =  64 : [az1 az0 z1 z0 az2 z2]          az = 4*((i>>5)&1) + 2*((i>>4)&1) + ((i>>1)&1)
//                                                    z  = 4*((i>>3)&1) + 2*((i>>2)&1) + (i&1)
//      count = 256 : [az1 az0 z1 z0 az2 z2 az3 z3]   az = 8*((i>>7)&1) + 4*((i>>6)&1) + 2*((i>>3)&1) + ((i>>1)&1)
//                                                    z  = 8*((i>>5)&1) + 4*((i>>4)&1) + 2*((i>>2)&1) + (i&1)
//
//  C0 -> C1 and C1 -> C2 each refine both the azimuth and the z bin. Any other
//  direction count falls back to the original Fibonacci lattice so unrelated
//  debug or configuration values keep working.
// ============================================================================

#ifndef RC_DIRECTION_GLSL
#define RC_DIRECTION_GLSL

const int RC_MAX_DIRECTIONS = 256;
const float RC_PI = 3.14159265358979323846;
const float RC_GOLDEN_ANGLE = 2.39996322972865332;

// Returns the unit direction for a direction index.
//
// For the three cascade counts (16, 64, 256) this decodes the hierarchical
// azimuth/z bin layout described above and returns the CENTRE of that bin.
// For any other count it returns the original spherical Fibonacci lattice.
vec3 RCDirection(int index, int count) {
    int n = clamp(count, 1, RC_MAX_DIRECTIONS);
    int i = clamp(index, 0, n - 1);

    int azimuthCount;
    int zCount;
    int azimuthIndex;
    int zIndex;

    if (n == 16) {
        // 4 azimuth x 4 z. Bits: [az1 az0 z1 z0]
        azimuthCount = 4;
        zCount = 4;
        azimuthIndex = i >> 2;
        zIndex = i & 3;
    } else if (n == 64) {
        // 8 azimuth x 8 z. Bits: [az1 az0 z1 z0 az2 z2]
        azimuthCount = 8;
        zCount = 8;
        azimuthIndex = (((i >> 5) & 1) << 2) | (((i >> 4) & 1) << 1) | ((i >> 1) & 1);
        zIndex = (((i >> 3) & 1) << 2) | (((i >> 2) & 1) << 1) | (i & 1);
    } else if (n == 256) {
        // 16 azimuth x 16 z. Bits: [az1 az0 z1 z0 az2 z2 az3 z3]
        azimuthCount = 16;
        zCount = 16;
        azimuthIndex = (((i >> 7) & 1) << 3) | (((i >> 6) & 1) << 2)
                     | (((i >> 3) & 1) << 1) | ((i >> 1) & 1);
        zIndex = (((i >> 5) & 1) << 3) | (((i >> 4) & 1) << 2)
               | (((i >> 2) & 1) << 1) | (i & 1);
    } else {
        // Fallback: original Fibonacci lattice for unexpected counts.
        float zf = 1.0 - 2.0 * (float(i) + 0.5) / float(n);
        float rf = sqrt(max(1.0 - zf * zf, 0.0));
        float phif = float(i) * RC_GOLDEN_ANGLE;
        return vec3(rf * cos(phif), rf * sin(phif), zf);
    }

    float phi = 2.0 * RC_PI * (float(azimuthIndex) + 0.5) / float(azimuthCount);
    float z = -1.0 + 2.0 * (float(zIndex) + 0.5) / float(zCount);
    float r = sqrt(max(1.0 - z * z, 0.0));
    return vec3(r * cos(phi), r * sin(phi), z);
}

// Solid angle represented by one direction of an N-direction sphere.
float RCDirectionSolidAngle(int count) {
    return 4.0 * RC_PI / float(clamp(count, 1, RC_MAX_DIRECTIONS));
}

// Inverse of RCDirection's hierarchical decode: rebuilds the direction index
// from an (azimuthIndex, zIndex) pair. Only the three nested counts are
// supported; any other count returns the clamped linear index so callers that
// use it for neighbour lookups degrade gracefully.
int RCDirectionIndex(int azimuthIndex, int zIndex, int count) {
    int n = clamp(count, 1, RC_MAX_DIRECTIONS);
    if (n == 16) {
        int az = clamp(azimuthIndex, 0, 3);
        int z = clamp(zIndex, 0, 3);
        return (az << 2) | z;
    } else if (n == 64) {
        int az = clamp(azimuthIndex, 0, 7);
        int z = clamp(zIndex, 0, 7);
        // bits: [az2 az1 z2 z1 az0 z0]
        return (((az >> 2) & 1) << 5) | (((az >> 1) & 1) << 4)
             | (((z >> 2) & 1) << 3) | (((z >> 1) & 1) << 2)
             | ((az & 1) << 1) | (z & 1);
    } else if (n == 256) {
        int az = clamp(azimuthIndex, 0, 15);
        int z = clamp(zIndex, 0, 15);
        // bits: [az3 az2 z3 z2 az1 z1 az0 z0]
        return (((az >> 3) & 1) << 7) | (((az >> 2) & 1) << 6)
             | (((z >> 3) & 1) << 5) | (((z >> 2) & 1) << 4)
             | (((az >> 1) & 1) << 3) | (((z >> 1) & 1) << 2)
             | ((az & 1) << 1) | (z & 1);
    }
    return clamp(azimuthIndex, 0, n - 1);
}

// Decodes a direction index into its (azimuthIndex, zIndex) bin coordinates.
// Returns false for counts that do not use the hierarchical layout.
bool RCDirectionBins(int index, int count, out int azimuthIndex, out int zIndex, out int azimuthCount, out int zCount) {
    int n = clamp(count, 1, RC_MAX_DIRECTIONS);
    int i = clamp(index, 0, n - 1);
    if (n == 16) {
        azimuthCount = 4; zCount = 4;
        azimuthIndex = i >> 2; zIndex = i & 3;
        return true;
    } else if (n == 64) {
        azimuthCount = 8; zCount = 8;
        azimuthIndex = (((i >> 5) & 1) << 2) | (((i >> 4) & 1) << 1) | ((i >> 1) & 1);
        zIndex = (((i >> 3) & 1) << 2) | (((i >> 2) & 1) << 1) | (i & 1);
        return true;
    } else if (n == 256) {
        azimuthCount = 16; zCount = 16;
        azimuthIndex = (((i >> 7) & 1) << 3) | (((i >> 6) & 1) << 2)
                     | (((i >> 3) & 1) << 1) | ((i >> 1) & 1);
        zIndex = (((i >> 5) & 1) << 3) | (((i >> 4) & 1) << 2)
               | (((i >> 2) & 1) << 1) | (i & 1);
        return true;
    }
    azimuthIndex = i; zIndex = 0; azimuthCount = n; zCount = 1;
    return false;
}

// Two nearest far-cascade directions for one near-cascade direction.
//
// RCDirection(i, N) is NOT nested across direction counts: the polar term
// depends on N, so index i at count N is a different world direction from
// index i at count 2N. Merging cascades therefore has to resample angularly
// instead of reusing the numeric index.
struct RCAngularMatch {
    int index0;
    int index1;
    float weight0;
    float weight1;
};

// Finds the two far directions closest to `nearDirection` and returns them with
// normalized weights. Ranking uses dot(nearDirection, farDirection); the
// weights use a strongly local kernel so a far direction only contributes when
// it is genuinely close to the near direction.
RCAngularMatch RCFindTwoNearestDirections(vec3 nearDirection, int farDirectionCount) {
    RCAngularMatch match;
    match.index0 = 0;
    match.index1 = 0;
    match.weight0 = 1.0;
    match.weight1 = 0.0;

    int count = clamp(farDirectionCount, 1, RC_MAX_DIRECTIONS);
    vec3 nearDir = normalize(nearDirection);

    float bestDot0 = -2.0;
    float bestDot1 = -2.0;

    for (int i = 0; i < count; ++i) {
        float d = dot(nearDir, RCDirection(i, count));
        if (d > bestDot0) {
            bestDot1 = bestDot0;
            match.index1 = match.index0;
            bestDot0 = d;
            match.index0 = i;
        } else if (d > bestDot1) {
            bestDot1 = d;
            match.index1 = i;
        }
    }

    float raw0 = pow(max(bestDot0, 0.0), 32.0);
    float raw1 = pow(max(bestDot1, 0.0), 32.0);
    float rawSum = raw0 + raw1;

    if (rawSum > 1e-12) {
        match.weight0 = raw0 / rawSum;
        match.weight1 = raw1 / rawSum;
    } else {
        // Both candidates are effectively perpendicular: fall back to the
        // single nearest direction so the caller never divides by zero.
        match.weight0 = 1.0;
        match.weight1 = 0.0;
    }

    return match;
}

#endif // RC_DIRECTION_GLSL
