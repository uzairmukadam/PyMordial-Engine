#version 450 core
// ============================================================================
//  GTAO with Hi-Z depth-bounds rejection
//
//  Identical horizon-based AO model to gtao.frag. The only addition is a
//  conservative early-out: each horizon step first queries the RG32F bounds
//  hierarchy at the mip whose texel footprint matches the step's screen-space
//  radius. If the ray's depth interval cannot reach the cell's geometry
//  interval, the step is skipped without a full-resolution depth fetch.
//
//  AO-specific horizon integration, weighting and power curve are unchanged and
//  stay in this file. The shared helpers only supply depth reconstruction,
//  background rules and interval comparison.
//
//  Debug output (u_AODebugView):
//      0 = final AO, 1 = selected mip, 2 = bounds occupancy,
//      3 = rejected sample count, 4 = refined sample count
// ============================================================================

#include "shaders/common/depth.glsl"
#include "shaders/common/hiz_bounds.glsl"

in vec2 v_UV;
layout (location = 0) out float out_AO;

layout (binding = 0) uniform sampler2D u_GBufferDepth;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_HiZBoundsTexture;

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

uniform float u_Radius = 0.35;
uniform float u_Intensity = 1.2;
uniform float u_Power = 1.5;
uniform int u_Directions = 4;
uniform int u_Steps = 6;
uniform int u_AODebugView = 0;

const float PI = 3.14159265358979323846;
const float HALF_PI = 1.5707963267948966;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

float FastACos(float inX) {
    float x = abs(inX);
    float res = -0.156583 * x + HALF_PI;
    res *= sqrt(max(0.0, 1.0 - x));
    return (inX >= 0.0) ? res : PI - res;
}

// Selects the mip whose texel footprint best matches the AO step's screen-space
// radius. A larger footprint means a coarser (cheaper, more conservative) cell.
int SelectAOMip(vec2 screen_radius_uv, vec2 screen_size) {
    float footprint_px = max(screen_radius_uv.x * screen_size.x, screen_radius_uv.y * screen_size.y);
    int max_mip = max(textureQueryLevels(u_HiZBoundsTexture) - 1, 0);
    int mip = int(floor(log2(max(footprint_px, 1.0))));
    return clamp(mip, 0, max_mip);
}

void main() {
    float center_depth = texture(u_GBufferDepth, v_UV).r;
    if (IsBackgroundDepth(center_depth)) {
        out_AO = 1.0;
        return;
    }

    vec3 view_pos = ReconstructViewPosition(v_UV, center_depth, u_InvProjection);
    vec3 world_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 view_normal = normalize((u_View * vec4(world_normal, 0.0)).xyz);
    vec3 view_dir = normalize(-view_pos);

    float noise = InterleavedGradientNoise(gl_FragCoord.xy);

    float view_z = max(-view_pos.z, 0.1);
    float proj_scale = u_Projection[1][1] * 0.5;
    vec2 screen_radius_uv = vec2(
        (u_Radius * proj_scale / view_z) * (u_ScreenSize_Jitter.y / u_ScreenSize_Jitter.x),
        (u_Radius * proj_scale / view_z)
    );
    vec2 min_radius_uv = 4.0 / u_ScreenSize_Jitter.xy;
    vec2 max_radius_uv = vec2(0.20);
    screen_radius_uv = clamp(screen_radius_uv, min_radius_uv, max_radius_uv);

    int num_dirs = clamp(u_Directions, 1, 6);
    int num_steps = clamp(u_Steps, 2, 12);

    int selected_mip = SelectAOMip(screen_radius_uv, u_ScreenSize_Jitter.xy);
    float center_linear = max(-view_pos.z, 0.0);

    float visibility = 0.0;
    float rejected = 0.0;
    float refined = 0.0;
    float occupied_hits = 0.0;
    float total_queries = 0.0;

    for (int d = 0; d < num_dirs; ++d) {
        float phi = (float(d) + noise) * (PI / float(num_dirs));
        vec2 dir_2d = vec2(cos(phi), sin(phi));

        vec3 dir_view = vec3(dir_2d.x, dir_2d.y, 0.0);
        vec3 ortho_dir_view = dir_view - dot(dir_view, view_dir) * view_dir;
        vec3 axis_vec = normalize(cross(ortho_dir_view, view_dir));
        vec3 proj_normal = view_normal - axis_vec * dot(view_normal, axis_vec);

        float proj_norm_len = length(proj_normal);
        if (proj_norm_len < 0.001) continue;

        float sign_norm = sign(dot(ortho_dir_view, proj_normal));
        float cos_norm = clamp(dot(proj_normal, view_dir) / proj_norm_len, 0.0, 1.0);
        float n = sign_norm * FastACos(cos_norm);

        float low_horizon_cos0 = cos(n + HALF_PI);
        float low_horizon_cos1 = cos(n - HALF_PI);

        float horizon_cos0 = low_horizon_cos0;
        float horizon_cos1 = low_horizon_cos1;

        for (int s = 1; s <= num_steps; ++s) {
            float alpha = (float(s) + noise) / float(num_steps);

            for (int side = 0; side < 2; ++side) {
                vec2 uv_s = (side == 0) ? v_UV + dir_2d * screen_radius_uv * alpha
                                        : v_UV - dir_2d * screen_radius_uv * alpha;
                if (!IsUVInBounds(uv_s)) {
                    continue;
                }

                // Conservative bounds rejection at the footprint-matched mip.
                // The AO ray spans at most u_Radius in view depth, so the
                // interval [center - radius, center + radius] is the widest
                // depth range this step could possibly reach.
                total_queries += 1.0;
                HiZDepthBounds bounds = SampleHiZBounds(u_HiZBoundsTexture, uv_s, selected_mip);
                if (HiZBoundsEmpty(bounds)) {
                    rejected += 1.0;
                    continue;
                }
                occupied_hits += 1.0;

                float ray_nearest = max(center_linear - u_Radius, 0.0);
                float ray_farthest = center_linear + u_Radius;
                if (!DepthIntervalsOverlap(ray_nearest, ray_farthest, bounds, u_Radius)) {
                    rejected += 1.0;
                    continue;
                }

                // Possible occlusion: refine against full-resolution depth.
                refined += 1.0;
                float d_s = texture(u_GBufferDepth, uv_s).r;
                if (IsBackgroundDepth(d_s)) {
                    continue;
                }
                vec3 p_s = ReconstructViewPosition(uv_s, d_s, u_InvProjection);
                vec3 delta = p_s - view_pos;
                float dist = length(delta);
                if (dist <= 0.002 || dist >= u_Radius || dot(delta, view_normal) <= 0.002) {
                    continue;
                }

                float dist_ratio = dist / u_Radius;
                float weight = clamp(1.0 - dist_ratio * dist_ratio, 0.0, 1.0);
                float shc = dot(delta / dist, view_dir);
                float low_horizon = (side == 0) ? low_horizon_cos0 : low_horizon_cos1;
                shc = mix(low_horizon, shc, weight);
                if (side == 0) {
                    horizon_cos0 = max(horizon_cos0, shc);
                } else {
                    horizon_cos1 = max(horizon_cos1, shc);
                }
            }
        }

        float h0 = -FastACos(clamp(horizon_cos1, -1.0, 1.0));
        float h1 = FastACos(clamp(horizon_cos0, -1.0, 1.0));

        h0 = n + clamp(h0 - n, -HALF_PI, HALF_PI);
        h1 = n + clamp(h1 - n, -HALF_PI, HALF_PI);

        float iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;

        visibility += proj_norm_len * (iarc0 + iarc1);
    }

    visibility /= float(num_dirs);
    float ao = clamp(pow(clamp(visibility, 0.0, 1.0), u_Power * u_Intensity), 0.0, 1.0);

    float fade_start = 50.0;
    float fade_end = 90.0;
    float dist_fade = smoothstep(fade_start, fade_end, view_z);
    ao = mix(ao, 1.0, dist_fade);

    if (u_AODebugView == 1) {
        out_AO = float(selected_mip) / float(max(textureQueryLevels(u_HiZBoundsTexture) - 1, 1));
        return;
    }
    if (u_AODebugView == 2) {
        out_AO = total_queries > 0.0 ? occupied_hits / total_queries : 0.0;
        return;
    }
    if (u_AODebugView == 3) {
        out_AO = total_queries > 0.0 ? rejected / total_queries : 0.0;
        return;
    }
    if (u_AODebugView == 4) {
        out_AO = total_queries > 0.0 ? refined / total_queries : 0.0;
        return;
    }

    out_AO = ao;
}
