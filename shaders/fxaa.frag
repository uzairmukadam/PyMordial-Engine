#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_Color;

layout (binding = 0) uniform sampler2D u_ColorTexture;
uniform vec2 u_InverseScreenSize;

// FXAA 3.11 Quality Presets (Preset 39: 12 search iterations)
#define FXAA_SEARCH_STEPS 12
const float FXAA_QUALITY_STEPS[12] = float[12](
    1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0
);

uniform float u_SubpixelQuality = 0.75;
uniform float u_EdgeThreshold = 0.125;
uniform float u_EdgeThresholdMin = 0.04;

float FxaaLuma(vec3 rgb) {
    return rgb.g * (0.587 / 0.299) + rgb.r * 0.299 + rgb.b * 0.114;
}

void main() {
    vec3 color_center = texture(u_ColorTexture, v_UV).rgb;
    float luma_m = FxaaLuma(color_center);

    // 1. Sample cross neighbors
    float luma_n = FxaaLuma(texture(u_ColorTexture, v_UV + vec2(0.0, -u_InverseScreenSize.y)).rgb);
    float luma_s = FxaaLuma(texture(u_ColorTexture, v_UV + vec2(0.0,  u_InverseScreenSize.y)).rgb);
    float luma_w = FxaaLuma(texture(u_ColorTexture, v_UV + vec2(-u_InverseScreenSize.x, 0.0)).rgb);
    float luma_e = FxaaLuma(texture(u_ColorTexture, v_UV + vec2( u_InverseScreenSize.x, 0.0)).rgb);

    float luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)));
    float luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)));
    float range = luma_max - luma_min;

    // Early-out if contrast is below local threshold (not an edge)
    if (range < max(u_EdgeThresholdMin, luma_max * u_EdgeThreshold)) {
        out_Color = vec4(color_center, 1.0);
        return;
    }

    // 2. Sample corner neighbors
    float luma_nw = FxaaLuma(texture(u_ColorTexture, v_UV + vec2(-u_InverseScreenSize.x, -u_InverseScreenSize.y)).rgb);
    float luma_ne = FxaaLuma(texture(u_ColorTexture, v_UV + vec2( u_InverseScreenSize.x, -u_InverseScreenSize.y)).rgb);
    float luma_sw = FxaaLuma(texture(u_ColorTexture, v_UV + vec2(-u_InverseScreenSize.x,  u_InverseScreenSize.y)).rgb);
    float luma_se = FxaaLuma(texture(u_ColorTexture, v_UV + vec2( u_InverseScreenSize.x,  u_InverseScreenSize.y)).rgb);

    // 3. Subpixel antialiasing blend calculation
    float luma_l = (luma_n + luma_s + luma_w + luma_e) * 2.0 + (luma_nw + luma_ne + luma_sw + luma_se);
    float subpix = abs(luma_l * (1.0 / 12.0) - luma_m) / range;
    float subpix_clamp = clamp(subpix, 0.0, 1.0);
    float subpix_blend = (-2.0 * subpix_clamp + 3.0) * subpix_clamp * subpix_clamp;
    subpix_blend = subpix_blend * subpix_blend * u_SubpixelQuality;

    // 4. Edge direction test: Horizontal vs Vertical
    float edge_horz = abs(luma_nw + luma_ne - 2.0 * luma_n) * 2.0
                    + abs(luma_w  + luma_e  - 2.0 * luma_m) * 2.0
                    + abs(luma_sw + luma_se - 2.0 * luma_s) * 2.0;

    float edge_vert = abs(luma_nw + luma_sw - 2.0 * luma_w) * 2.0
                    + abs(luma_n  + luma_s  - 2.0 * luma_m) * 2.0
                    + abs(luma_ne + luma_se - 2.0 * luma_e) * 2.0;

    bool is_horizontal = (edge_horz >= edge_vert);

    // 5. Determine edge boundary and gradient
    float luma_1 = is_horizontal ? luma_n : luma_w;
    float luma_2 = is_horizontal ? luma_s : luma_e;
    float gradient_1 = abs(luma_1 - luma_m);
    float gradient_2 = abs(luma_2 - luma_m);

    bool is_1_steeper = (gradient_1 >= gradient_2);
    float gradient_scaled = 0.25 * max(gradient_1, gradient_2);

    vec2 step_dir = is_horizontal ? vec2(u_InverseScreenSize.x, 0.0) : vec2(0.0, u_InverseScreenSize.y);
    vec2 normal_dir = is_horizontal ? vec2(0.0, u_InverseScreenSize.y) : vec2(u_InverseScreenSize.x, 0.0);
    if (is_1_steeper) {
        normal_dir = -normal_dir;
    }

    vec2 uv_edge = v_UV + normal_dir * 0.5;

    // 6. Raymarch along the edge in both directions to find endpoints
    vec2 uv_pos = uv_edge + step_dir;
    vec2 uv_neg = uv_edge - step_dir;

    float luma_edge_target = is_1_steeper ? (luma_1 + luma_m) * 0.5 : (luma_2 + luma_m) * 0.5;

    bool done_pos = false;
    bool done_neg = false;

    float luma_end_pos = 0.0;
    float luma_end_neg = 0.0;

    for (int i = 0; i < FXAA_SEARCH_STEPS; ++i) {
        float step_scale = FXAA_QUALITY_STEPS[i];

        if (!done_pos) {
            luma_end_pos = FxaaLuma(texture(u_ColorTexture, uv_pos).rgb) - luma_edge_target;
            done_pos = (abs(luma_end_pos) >= gradient_scaled);
            if (!done_pos) {
                uv_pos += step_dir * step_scale;
            }
        }

        if (!done_neg) {
            luma_end_neg = FxaaLuma(texture(u_ColorTexture, uv_neg).rgb) - luma_edge_target;
            done_neg = (abs(luma_end_neg) >= gradient_scaled);
            if (!done_neg) {
                uv_neg -= step_dir * step_scale;
            }
        }

        if (done_pos && done_neg) {
            break;
        }
    }

    // 7. Calculate subpixel offset from distance to endpoints
    float dist_pos = is_horizontal ? (uv_pos.x - v_UV.x) : (uv_pos.y - v_UV.y);
    float dist_neg = is_horizontal ? (v_UV.x - uv_neg.x) : (v_UV.y - uv_neg.y);

    bool is_closer_to_pos = (dist_pos < dist_neg);
    float dist_min = min(dist_pos, dist_neg);
    float dist_total = dist_pos + dist_neg;

    float edge_blend = 0.5 - (dist_min / max(dist_total, 1e-5));
    bool is_correct_side = ((is_closer_to_pos ? luma_end_pos : luma_end_neg) < 0.0) == (luma_m < luma_edge_target);
    if (!is_correct_side) {
        edge_blend = 0.0;
    }

    // 8. Blend final color
    float final_blend = max(subpix_blend, edge_blend);
    vec2 final_uv = v_UV + normal_dir * final_blend;

    out_Color = vec4(texture(u_ColorTexture, final_uv).rgb, 1.0);
}
