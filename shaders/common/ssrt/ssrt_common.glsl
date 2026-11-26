// ============================================================================
//  Shared Screen-Space Ray Tracing (SSRT) — Common Utilities
//
//  Generic, effect-agnostic helpers shared by every screen-space ray tracer
//  (GI, reflections, AO, contact shadows, refraction). These were extracted
//  verbatim from the AMRGI trace shader so the current behaviour is preserved
//  exactly; they contain no GI-specific logic.
//
//  Effect-specific ray generation (e.g. cosine hemisphere sampling for diffuse
//  GI) deliberately stays in the effect shader, not here.
//
//  Depth reconstruction, UV bounds checks and octahedral normal decoding are
//  NOT duplicated here — they already live in common/depth.glsl and
//  common/pbr_brdf.glsl and must be reused from there.
// ============================================================================

#ifndef SSRT_COMMON_GLSL
#define SSRT_COMMON_GLSL

// Interleaved Gradient Noise (Jorge Jimenez). A smooth spatial gradient used for
// dithering; NOT suitable as a per-pixel seed (it produces a visible checkerboard
// when used that way — use SSRT_Hash01 for seeds).
float SSRT_InterleavedGradientNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Integer bit-mixing hash (PCG-style). Unlike the smooth gradient noise above,
// this decorrelates neighbouring pixels and independent dimensions. Returns a
// well-distributed value in [0, 1).
uint SSRT_HashUint(uint x) {
    x = x * 747796405u + 2891336453u;
    uint w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}

float SSRT_Hash01(uvec3 v) {
    uint h = SSRT_HashUint(v.x ^ SSRT_HashUint(v.y ^ SSRT_HashUint(v.z)));
    return float(h) * (1.0 / 4294967296.0);
}

// Stable normal-oriented basis with a finite fallback for degenerate normals.
void SSRT_BuildBasis(vec3 n, out vec3 tangent, out vec3 bitangent) {
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    tangent = cross(up, n);
    float len = length(tangent);
    tangent = (len > 1e-5) ? (tangent / len) : vec3(1.0, 0.0, 0.0);
    bitangent = cross(n, tangent);
}

// Screen-edge confidence: fades a hit toward the viewport border so a ray that
// leaves the screen does not pop. `inner`/`outer` are the UV fade band edges.
float SSRT_EdgeConfidence(vec2 uv, float inner, float outer) {
    vec2 edge = smoothstep(vec2(inner), vec2(outer), uv)
              * (1.0 - smoothstep(vec2(1.0 - outer), vec2(1.0 - inner), uv));
    return edge.x * edge.y;
}

#endif // SSRT_COMMON_GLSL
