#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_SSR;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_SceneColor;

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

uniform int u_SSR_Steps = 32;
uniform float u_SSR_MaxDistance = 20.0;
uniform float u_SSR_Thickness = 0.40;
uniform float u_SSR_MaxRoughness = 0.65;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 GetViewPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / max(abs(view.w), 1e-6);
}

// Jorge Jimenez's Interleaved Gradient Noise for uniform blue-noise spatial dithering
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (raw_depth <= 0.000001) {
        out_SSR = vec4(0.0);
        return;
    }

    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    float roughness = albedo_rough.a;

    // Skip rough surfaces that don't produce specular reflections
    if (roughness > u_SSR_MaxRoughness) {
        out_SSR = vec4(0.0);
        return;
    }

    vec3 view_pos = GetViewPos(v_UV, raw_depth);
    float v_len = length(view_pos);
    if (v_len <= 1e-5) {
        out_SSR = vec4(0.0);
        return;
    }

    vec3 world_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 view_normal = normalize((u_View * vec4(world_normal, 0.0)).xyz);
    vec3 view_dir = -view_pos / v_len;
    vec3 R = reflect(-view_dir, view_normal);

    // If reflection ray points towards the surface or into camera clipping plane, abort
    if (dot(view_normal, R) <= 0.001) {
        out_SSR = vec4(0.0);
        return;
    }

    // Interleaved Gradient Noise for sub-texel ray jittering
    float jitter = InterleavedGradientNoise(gl_FragCoord.xy);

    // Normal offset to avoid self-reflection acne
    vec3 ray_origin = view_pos + view_normal * 0.02;
    int steps = clamp(u_SSR_Steps, 24, 80);
    float step_len = u_SSR_MaxDistance / float(steps);

    vec2 hit_uv = vec2(0.0);
    float hit_found = 0.0;
    float hit_dist = 0.0;
    vec3 final_hit_normal = vec3(0.0, 1.0, 0.0);

    for (int s = 1; s <= steps; ++s) {
        // Dithered ray marching: eliminates uniform stepping bands
        float t = (float(s) - 0.5 + jitter) * step_len;
        vec3 curr_view = ray_origin + R * t;

        // Abort if ray passes behind the near clipping plane
        if (curr_view.z >= -0.08) break;

        vec4 clip_sample = u_Projection * vec4(curr_view, 1.0);
        if (clip_sample.w <= 1e-5) break;
        vec3 ndc_sample = clip_sample.xyz / clip_sample.w;
        vec2 sample_uv = ndc_sample.xy * 0.5 + 0.5;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            break;
        }

        float scene_depth = texture(u_GBufferDepth, sample_uv).r;
        if (scene_depth <= 0.000001) continue;

        vec3 scene_view = GetViewPos(sample_uv, scene_depth);

        // View-space depth comparison (-Z is metric distance from camera)
        float ray_depth = -curr_view.z;
        float geo_depth = -scene_view.z;
        float depth_diff = ray_depth - geo_depth;

        // Adaptive thickness threshold guarantees step_len cannot overshoot geometry
        float adaptive_thickness = clamp(max(u_SSR_Thickness, step_len * 0.85) * (1.0 + ray_depth * 0.02), 0.15, 0.65);

        if (depth_diff >= 0.0 && depth_diff < adaptive_thickness) {
            // Sub-pixel binary search bisection (6 iterations)
            float t_min = max(0.0, t - step_len);
            float t_max = t;
            vec2 best_uv = sample_uv;
            float best_t = t;

            for (int b = 0; b < 6; ++b) {
                float t_mid = (t_min + t_max) * 0.5;
                vec3 p_mid = ray_origin + R * t_mid;
                vec4 c_mid = u_Projection * vec4(p_mid, 1.0);
                if (c_mid.w <= 1e-5) break;
                vec2 uv_mid = (c_mid.xy / c_mid.w) * 0.5 + 0.5;

                float d_mid = texture(u_GBufferDepth, uv_mid).r;
                if (d_mid <= 0.000001) {
                    t_min = t_mid;
                    continue;
                }
                vec3 s_mid = GetViewPos(uv_mid, d_mid);
                float diff_mid = (-p_mid.z) - (-s_mid.z);

                if (diff_mid >= 0.0 && diff_mid < adaptive_thickness) {
                    t_max = t_mid;
                    best_uv = uv_mid;
                    best_t = t_mid;
                } else if (diff_mid < 0.0) {
                    t_min = t_mid;
                } else {
                    t_min = t_mid;
                }
            }

            // Sub-pixel self-origin rejection
            if (length(best_uv - v_UV) < 0.008 || best_t < 0.05) {
                continue;
            }

            // Reconstruct hit position in view space
            float hit_depth = texture(u_GBufferDepth, best_uv).r;
            if (hit_depth <= 0.000001) continue;
            vec3 hit_view_pos = GetViewPos(best_uv, hit_depth);

            // 1. Tangent plane distance check:
            // A reflection ray traveling in direction R (where dot(view_normal, R) > 0) strictly travels into the
            // positive half-space of the originating surface plane. Any geometry lying behind or on the plane is
            // a self-intersection or background occlusion error.
            float plane_dist = dot(hit_view_pos - view_pos, view_normal);
            if (plane_dist <= 0.02) {
                continue;
            }

            // 2. Surface orientation check:
            // The reflection ray travels along vector R. For a valid front-facing collision, the hit surface's normal
            // must face towards the incoming ray (dot(R, hit_normal_view) < 0).
            // If dot(R, hit_normal_view) >= -0.05, the surface faces away from the ray (ray is hitting it from behind or passing behind it).
            vec3 hit_normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, best_uv).rg);
            vec3 hit_normal_view = normalize((u_View * vec4(hit_normal_world, 0.0)).xyz);
            if (dot(R, hit_normal_view) >= -0.05) {
                continue;
            }

            // 3. Planar self-intersection check: if surface normal is identical to originating surface and very close to the plane
            if (dot(view_normal, hit_normal_view) > 0.98 && plane_dist < 0.10) {
                continue;
            }

            hit_uv = best_uv;
            hit_found = 1.0;
            hit_dist = best_t;
            final_hit_normal = hit_normal_world;
            break;
        }
    }

    if (hit_found > 0.0) {
        // Screen edge vignette fade to eliminate hard boundary popping
        vec2 edge_coords = abs(hit_uv - 0.5) * 2.0;
        float edge_factor = clamp(1.0 - max(edge_coords.x, edge_coords.y), 0.0, 1.0);
        edge_factor = smoothstep(0.0, 0.15, edge_factor);

        // Ray travel distance attenuation
        float dist_fade = clamp(1.0 - (hit_dist / u_SSR_MaxDistance), 0.0, 1.0);

        // Roughness fade
        float rough_fade = 1.0 - smoothstep(0.0, u_SSR_MaxRoughness, roughness);

        // Grazing reflection fade against surface normal
        float grazing_fade = smoothstep(0.001, 0.12, dot(view_normal, R));

        vec3 reflected_color = texture(u_SceneColor, hit_uv).rgb;
        if (isnan(reflected_color.r) || isnan(reflected_color.g) || isnan(reflected_color.b) ||
            isinf(reflected_color.r) || isinf(reflected_color.g) || isinf(reflected_color.b)) {
            reflected_color = vec3(0.0);
        }
        reflected_color = clamp(reflected_color, vec3(0.0), vec3(40.0));

        // In PyMordial's deferred architecture, Pass 6 (SSR) samples G-Buffer albedo before
        // Pass 7 (Deferred Resolve). We calculate realistic daylight radiance for the hit geometry:
        vec3 to_sun = normalize(-u_SunDirection_Intensity.xyz);
        float NdotL = max(dot(final_hit_normal, to_sun), 0.0);
        vec3 sun_radiance = u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * 0.35);
        vec3 hit_lit_radiance = reflected_color * (sun_radiance * NdotL + vec3(0.40));
        reflected_color = max(reflected_color * 0.6, hit_lit_radiance);

        // Energy fade: smoothly fade alpha if surface is unlit/dark so it gracefully falls back to sky reflection
        float refl_lum = dot(reflected_color, vec3(0.2126, 0.7152, 0.0722));
        float energy_fade = smoothstep(0.01, 0.08, refl_lum);

        float alpha = edge_factor * dist_fade * rough_fade * grazing_fade * energy_fade;
        if (isnan(alpha) || isinf(alpha)) alpha = 0.0;
        out_SSR = vec4(reflected_color, clamp(alpha, 0.0, 1.0));
    } else {
        out_SSR = vec4(0.0);
    }
}
