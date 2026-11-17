#version 450 core

// Motion Blur Pass 2: directional gather (McGuire et al. 2012,
// "A Reconstruction Filter for Plausible Motion Blur").
//
// Per-pixel sample count is derived from the tile-max velocity produced by
// motion_blur_tile.frag, so slow pixels stay cheap and fast pixels get more taps.
// Velocity is dilated 3x3 to stop background bleeding at silhouettes, and taps
// are depth-weighted so background never smears over foreground geometry.

#include "common/frame_data.glsl"

in vec2 v_UV;
layout(location = 0) out vec4 out_Color;

layout(binding = 0) uniform sampler2D u_SceneHDR;
layout(binding = 1) uniform sampler2D u_VelocityTexture;
layout(binding = 2) uniform sampler2D u_DepthTexture;   // Reversed-Z 32F
layout(binding = 3) uniform sampler2D u_TileMaxTexture; // rg = max velocity, b = max magnitude

uniform ivec2 u_TileSize;      // tile footprint in pixels
uniform int   u_MaxSamples;    // upper bound on gather taps
uniform float u_Intensity;     // shutter scale applied to the velocity vector
uniform float u_MaxRadius;     // max blur length in pixels

float linearizeDepth(float rawDepth, vec2 uv) {
    if (rawDepth <= 1e-6) {
        return 2000.0; // Sky / far background
    }
    vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
    vec4 viewPos = u_InvProjection * clip;
    return max(-viewPos.z / max(viewPos.w, 1e-6), 0.05);
}

// Interleaved gradient noise: stable per-pixel dither for the gather offsets.
float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec2 screenSize = u_ScreenSize_Jitter.xy;
    vec2 texel = 1.0 / screenSize;
    vec3 centerColor = texture(u_SceneHDR, v_UV).rgb;

    // -------------------------------------------------------------------------
    // 1. Velocity dilation (3x3 max) to stop background bleeding at silhouettes
    // -------------------------------------------------------------------------
    vec2  velocity  = texture(u_VelocityTexture, v_UV).xy;
    float velMag2   = dot(velocity, velocity);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 v = texture(u_VelocityTexture, v_UV + vec2(float(x), float(y)) * texel).xy;
            float m2 = dot(v, v);
            if (m2 > velMag2) {
                velMag2 = m2;
                velocity = v;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Tile-max driven sample count
    // -------------------------------------------------------------------------
    ivec2 tileCoord = ivec2(gl_FragCoord.xy) / max(u_TileSize, ivec2(1));
    vec2 tileMax = texture(u_TileMaxTexture, (vec2(tileCoord) + 0.5) * vec2(u_TileSize) / screenSize).rg;

    float tileMagPx = length(tileMax * screenSize) * u_Intensity;
    int samples = clamp(int(ceil(tileMagPx * 0.5)), 1, max(u_MaxSamples, 1));

    // -------------------------------------------------------------------------
    // 3. Clamp the blur vector to the configured pixel radius
    // -------------------------------------------------------------------------
    vec2 blurVec = velocity * u_Intensity;
    float blurPx = length(blurVec * screenSize);
    if (blurPx > u_MaxRadius) {
        blurVec *= u_MaxRadius / max(blurPx, 1e-4);
        blurPx = u_MaxRadius;
    }

    // Sub-pixel motion: nothing to reconstruct, pass the input through untouched.
    if (blurPx < 0.5 || samples <= 1) {
        out_Color = vec4(centerColor, 1.0);
        return;
    }

    // -------------------------------------------------------------------------
    // 4. Directional gather with depth-aware weighting
    // -------------------------------------------------------------------------
    float centerDepth = linearizeDepth(texture(u_DepthTexture, v_UV).r, v_UV);
    float jitter = interleavedGradientNoise(gl_FragCoord.xy);

    vec3  accum  = centerColor;
    float weight = 1.0;

    float invSamples = 1.0 / float(samples);
    for (int i = 0; i < samples; ++i) {
        // Centered taps in [-0.5, 0.5] with per-pixel dither to break banding.
        float t = (float(i) + jitter) * invSamples - 0.5;
        vec2 tapUV = v_UV + blurVec * t;

        vec3 tapColor = texture(u_SceneHDR, tapUV).rgb;
        float tapDepth = linearizeDepth(texture(u_DepthTexture, tapUV).r, tapUV);

        // Reject taps that sit behind the center fragment (background bleeding).
        float depthWeight = 1.0;
        if (tapDepth > centerDepth) {
            depthWeight = clamp(1.0 - (tapDepth - centerDepth) / max(centerDepth * 0.25, 0.5), 0.0, 1.0);
        }

        accum  += tapColor * depthWeight;
        weight += depthWeight;
    }

    out_Color = vec4(accum / max(weight, 1e-4), 1.0);
}
