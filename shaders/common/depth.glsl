// ============================================================================
//  Canonical Depth Utility
//  Shared reconstruction / projection math for Hi-Z, SSR, SSGI, AO and RC.
//
//  CLIP-SPACE Z CONVENTION (must match the engine's projection matrices)
//  ---------------------------------------------------------------------
//  The engine uses a right-handed view space (camera looks down -Z) with
//  GL_ZERO_TO_ONE clip depth and Reversed-Z:
//
//      near plane  ->  ndc.z = 1.0
//      far plane   ->  ndc.z = 0.0
//      background  ->  raw depth 0.0 (the G-Buffer clears depth to 0.0)
//
//  Because the depth range is already [0, 1], the raw depth sample IS the NDC
//  z value: no `* 2.0 - 1.0` remap is applied to z (only xy is remapped from
//  [0, 1] UV to [-1, 1] NDC).
//
//  LINEAR VIEW DEPTH
//  -----------------
//  "Linear view depth" is the positive distance along the camera forward axis:
//
//      linearDepth = -viewPosition.z
//
//  It is always >= 0 for geometry in front of the camera, and SMALLER values
//  are NEARER. This is the opposite ordering from reversed-Z hardware depth,
//  which is why every consumer must convert before comparing.
//
//  All helpers are pure functions with no sampler or uniform dependencies, so
//  this include can be dropped into any shader stage.
// ============================================================================

#ifndef DEPTH_GLSL
#define DEPTH_GLSL

// Raw reversed-Z depth at or below this value is treated as background.
// The G-Buffer clears depth to 0.0, so anything this close to zero is sky.
const float DEPTH_BACKGROUND_EPSILON = 1e-6;

// Returns true when a raw reversed-Z depth sample contains no geometry.
//
// Written as a negated comparison so NaN also reports as background: a NaN
// depth can never be valid geometry, and letting it through would poison every
// downstream reconstruction with NaN.
bool IsBackgroundDepth(float rawDepth) {
    return !(rawDepth > DEPTH_BACKGROUND_EPSILON);
}

// Returns true when a UV lies inside the [0, 1] screen rectangle (inclusive).
bool IsUVInBounds(vec2 uv) {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

// Reconstructs the view-space position of a pixel from its UV and raw depth.
//
// `inverseProjection` must be the inverse of the projection matrix that
// produced the depth buffer (including any TAA subpixel jitter), otherwise the
// reconstruction is offset by the jitter.
//
// Returns vec3(0.0) for background samples so callers never see NaN/Inf.
vec3 ReconstructViewPosition(vec2 uv, float rawDepth, mat4 inverseProjection) {
    if (IsBackgroundDepth(rawDepth)) {
        return vec3(0.0);
    }
    vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
    vec4 view = inverseProjection * clip;
    // Guard against a degenerate w (a projection that maps this depth to the
    // camera plane). Falling back to 1.0 keeps the result finite.
    float w = abs(view.w) > 1e-9 ? view.w : 1.0;
    return view.xyz / w;
}

// Converts a raw reversed-Z depth sample into positive linear view depth.
//
// Smaller values are nearer. Background samples return 0.0, which callers must
// treat as "no geometry" rather than "at the camera".
float RawDepthToLinearViewDepth(vec2 uv, float rawDepth, mat4 inverseProjection) {
    if (IsBackgroundDepth(rawDepth)) {
        return 0.0;
    }
    vec3 viewPosition = ReconstructViewPosition(uv, rawDepth, inverseProjection);
    return max(-viewPosition.z, 0.0);
}

// Projects a view-space position back into [0, 1] UV space.
//
// Returns vec2(-1.0) when the point is behind the camera (clip w <= 0), which
// IsUVInBounds() rejects. Callers that need the raw NDC z should project
// directly; this helper only exposes the screen-space UV.
vec2 ProjectViewPositionToUV(vec3 viewPosition, mat4 projection) {
    vec4 clip = projection * vec4(viewPosition, 1.0);
    if (clip.w <= 1e-9) {
        return vec2(-1.0);
    }
    return (clip.xy / clip.w) * 0.5 + 0.5;
}

#endif // DEPTH_GLSL
