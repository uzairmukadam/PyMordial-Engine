#version 450 core

#include "shaders/common/blur.glsl"

in vec2 v_UV;
layout (location = 0) out vec4 out_FilteredSSGI;

layout (binding = 0) uniform sampler2D u_InputSSGI;
layout (binding = 1) uniform sampler2D u_GBufferDepth;
layout (binding = 2) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 3) uniform sampler2D u_HistorySSGI;      // previous frame's resolved result
layout (binding = 4) uniform sampler2D u_MotionVectors;    // UV-space velocity

layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;

    vec4 u_CameraPos_Time;
    vec4 u_ScreenSize_Jitter;

    vec4 u_SunDirection_Intensity;
    vec4 u_SunColor_Ambient;

    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;

    vec4 u_FogColor_Density;
    vec4 u_FogParams;
};

uniform vec2 u_BlurDirection;
uniform vec2 u_TargetResolution;
uniform mat4 u_PrevView;
uniform float u_TemporalBlend;   // 0 = no history, ~0.9 = heavy accumulation
uniform int u_HistoryValid;      // 0 on the first frame or after a resize
uniform int u_TemporalEnabled;   // 0 disables reprojection entirely

// Reconstructs the view-space position of a pixel from its raw depth.
vec3 ViewPosFromDepth(vec2 uv, float raw_depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / max(abs(view.w), 1e-6);
}

void main() {
    float center_raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_raw_depth <= 0.000001) {
        out_FilteredSSGI = vec4(0.0);
        return;
    }

    // Distance-adaptive tangent plane tolerance prevents neighbor rejection at
    // camera distance.
    float center_z = LinearizeViewDepth(center_raw_depth, v_UV, u_InvProjection);
    float plane_sigma = max(0.12, abs(center_z) * 0.025);

    vec4 result = BilateralBlurDepthNormal(
        u_InputSSGI, u_GBufferDepth, u_GBufferNormalMetallic,
        v_UV, u_BlurDirection, 2.0, plane_sigma, 2.0, u_InvProjection,
        BLUR_USE_NORMAL | BLUR_METRIC_PLANE,
        mat3(u_View)
    );

    // ------------------------------------------------------------------
    //  Temporal accumulation
    //
    //  A single-frame SSGI estimate is inherently noisy: a handful of rays per
    //  pixel cannot resolve a smooth irradiance field. Spatial filtering alone
    //  either leaves the noise or blurs across geometry. Accumulating across
    //  frames is what actually converges it, and it is the only thing that
    //  keeps the image stable while the camera moves.
    //
    //  Reprojection uses the G-buffer velocity buffer, so both camera and
    //  object motion are handled. The history is clamped to the current
    //  spatial neighbourhood, which is what prevents ghosting: a sample that
    //  no longer belongs to this surface is rejected rather than smeared.
    // ------------------------------------------------------------------
    if (u_TemporalEnabled != 0 && u_HistoryValid != 0) {
        vec2 motion = texture(u_MotionVectors, v_UV).rg;
        vec2 previous_uv = v_UV - motion;

        if (all(greaterThanEqual(previous_uv, vec2(0.0))) &&
            all(lessThanEqual(previous_uv, vec2(1.0)))) {
            vec4 history = texture(u_HistorySSGI, previous_uv);

            // Depth validation: reproject this pixel's world position into the
            // previous frame and compare against the depth the history stored.
            // A mismatch means the history sample belongs to a different
            // surface, so it must not be blended in.
            vec3 center_view = ViewPosFromDepth(v_UV, center_raw_depth);
            vec4 world_pos = u_InvView * vec4(center_view, 1.0);
            vec4 prev_view_pos = u_PrevView * world_pos;
            float expected_prev_depth = abs(prev_view_pos.z);
            float history_depth_error = abs(history.a - expected_prev_depth);
            float depth_tolerance = max(0.10, expected_prev_depth * 0.02);
            float depth_confidence = 1.0 - smoothstep(depth_tolerance, depth_tolerance * 2.0, history_depth_error);

            // Neighbourhood clamp. The spatial result is the local mean; the
            // history is clamped to a band around it so a stale sample cannot
            // pull the result away from what this frame actually measured.
            vec3 history_rgb = history.rgb;
            vec3 clamp_margin = max(abs(result.rgb) * 0.35, vec3(0.02));
            history_rgb = clamp(history_rgb, result.rgb - clamp_margin, result.rgb + clamp_margin);

            // Motion confidence: fast screen-space motion means the reprojection
            // is less trustworthy, so the history weight drops.
            float motion_pixels = length(motion * u_TargetResolution);
            float motion_confidence = 1.0 - smoothstep(1.0, 8.0, motion_pixels);

            float temporal_weight = clamp(u_TemporalBlend, 0.0, 0.95)
                                  * depth_confidence
                                  * motion_confidence;

            result.rgb = mix(result.rgb, history_rgb, temporal_weight);
        }
    }

    if (isnan(result.r) || isnan(result.g) || isnan(result.b) || isnan(result.a) ||
        isinf(result.r) || isinf(result.g) || isinf(result.b) || isinf(result.a)) {
        result = vec4(0.0);
    }

    // Alpha carries the view depth so the next frame can validate the history.
    out_FilteredSSGI = vec4(clamp(result.rgb, vec3(0.0), vec3(40.0)), abs(center_z));
}
