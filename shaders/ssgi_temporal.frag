#version 450 core
// ============================================================================
//  SSGI Temporal Resolve
//
//  Runs exactly once per frame, after both spatial filter directions. It is the
//  only stage that reads history and motion vectors.
//
//  Inputs:
//      u_CurrentSSGI   — vertically filtered spatial result (rgb + hit ratio)
//      u_HistorySSGI   — previous frame's resolved result
//      u_MotionVectors — UV-space velocity (G-buffer RT2.rg)
//
//  History contract (documented, stable across frames):
//      rgb = resolved indirect radiance
//      a   = positive linear view depth of the pixel that produced rgb
//
//  The alpha channel is what makes reprojection safe: a history sample is only
//  blended when its stored depth matches the depth this pixel reprojects to in
//  the previous frame. A mismatch means the sample belongs to a different
//  surface (disocclusion), so it is rejected instead of smeared.
//
//  Output contract:
//      rgb = temporally resolved indirect radiance
//      a   = this frame's positive linear view depth (becomes next history alpha)
// ============================================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_ResolvedSSGI;

#include "shaders/common/blur.glsl"

layout (binding = 0) uniform sampler2D u_CurrentSSGI;
layout (binding = 1) uniform sampler2D u_HistorySSGI;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_MotionVectors;

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

uniform mat4 u_PrevView;
uniform vec2 u_TargetResolution;
uniform float u_TemporalBlend;   // 0 = no history, ~0.9 = heavy accumulation
uniform int u_HistoryValid;      // 0 on the first frame, after a resize, or after invalidate_history()
uniform int u_TemporalEnabled;   // 0 disables reprojection entirely

vec3 ViewPosFromDepth(vec2 uv, float raw_depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / max(abs(view.w), 1e-6);
}

void main() {
    float center_raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_raw_depth <= 0.000001) {
        out_ResolvedSSGI = vec4(0.0);
        return;
    }

    vec4 current = texture(u_CurrentSSGI, v_UV);
    if (isnan(current.r) || isnan(current.g) || isnan(current.b) ||
        isinf(current.r) || isinf(current.g) || isinf(current.b)) {
        current = vec4(0.0);
    }

    float center_z = LinearizeViewDepth(center_raw_depth, v_UV, u_InvProjection);
    vec3 result_rgb = current.rgb;

    if (u_TemporalEnabled != 0 && u_HistoryValid != 0) {
        vec2 motion = texture(u_MotionVectors, v_UV).rg;
        vec2 previous_uv = v_UV - motion;

        if (all(greaterThanEqual(previous_uv, vec2(0.0))) &&
            all(lessThanEqual(previous_uv, vec2(1.0)))) {
            vec4 history = texture(u_HistorySSGI, previous_uv);

            // Depth validation: reproject this pixel's world position into the
            // previous frame and compare against the depth the history stored.
            vec3 center_view = ViewPosFromDepth(v_UV, center_raw_depth);
            vec4 world_pos = u_InvView * vec4(center_view, 1.0);
            vec4 prev_view_pos = u_PrevView * world_pos;
            float expected_prev_depth = abs(prev_view_pos.z);
            float history_depth_error = abs(history.a - expected_prev_depth);
            float depth_tolerance = max(0.10, expected_prev_depth * 0.02);
            float depth_confidence = 1.0 - smoothstep(depth_tolerance, depth_tolerance * 2.0, history_depth_error);

            // Neighbourhood clamp: the spatial result is the local mean, so the
            // history is clamped to a band around it. A stale sample can then
            // never pull the result away from what this frame measured.
            vec3 history_rgb = history.rgb;
            vec3 clamp_margin = max(abs(current.rgb) * 0.35, vec3(0.02));
            history_rgb = clamp(history_rgb, current.rgb - clamp_margin, current.rgb + clamp_margin);

            // Fast screen-space motion makes the reprojection less trustworthy.
            float motion_pixels = length(motion * u_TargetResolution);
            float motion_confidence = 1.0 - smoothstep(1.0, 8.0, motion_pixels);

            float temporal_weight = clamp(u_TemporalBlend, 0.0, 0.95)
                                  * depth_confidence
                                  * motion_confidence;

            result_rgb = mix(current.rgb, history_rgb, temporal_weight);
        }
    }

    if (isnan(result_rgb.r) || isnan(result_rgb.g) || isnan(result_rgb.b) ||
        isinf(result_rgb.r) || isinf(result_rgb.g) || isinf(result_rgb.b)) {
        result_rgb = vec3(0.0);
    }

    // Alpha carries this frame's view depth so the next frame can validate the
    // reprojection against it.
    out_ResolvedSSGI = vec4(clamp(result_rgb, vec3(0.0), vec3(40.0)), abs(center_z));
}
