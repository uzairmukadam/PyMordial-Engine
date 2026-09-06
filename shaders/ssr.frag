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

uniform int u_SSR_Steps = 24;
uniform float u_SSR_MaxDistance = 15.0;
uniform float u_SSR_Thickness = 0.35;
uniform float u_SSR_MaxRoughness = 0.65;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 GetWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    view /= view.w;
    return (u_InvView * vec4(view.xyz, 1.0)).xyz;
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (raw_depth <= 0.000001) {
        out_SSR = vec4(0.0);
        return;
    }

    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    float roughness = albedo_rough.a;

    // Skip rough surfaces that don't produce mirror-like specular reflections
    if (roughness > u_SSR_MaxRoughness) {
        out_SSR = vec4(0.0);
        return;
    }

    vec3 world_pos = GetWorldPos(v_UV, raw_depth);
    vec3 N = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 V = normalize(u_CameraPos_Time.xyz - world_pos);
    vec3 R = reflect(-V, N);

    // If reflection ray points into the surface or towards the camera, abort
    if (dot(N, R) <= 0.0) {
        out_SSR = vec4(0.0);
        return;
    }

    vec3 ray_origin = world_pos + N * 0.05;
    int steps = clamp(u_SSR_Steps, 8, 48);
    float step_len = u_SSR_MaxDistance / float(steps);

    vec2 hit_uv = vec2(0.0);
    float hit_found = 0.0;

    for (int s = 1; s <= steps; ++s) {
        vec3 curr_world = ray_origin + R * (float(s) * step_len);

        vec4 clip_sample = u_ViewProjection * vec4(curr_world, 1.0);
        if (clip_sample.w <= 0.0) break;
        vec3 ndc_sample = clip_sample.xyz / clip_sample.w;
        vec2 sample_uv = ndc_sample.xy * 0.5 + 0.5;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            break;
        }

        float scene_depth = texture(u_GBufferDepth, sample_uv).r;
        if (scene_depth <= 0.000001) continue;

        vec3 scene_world = GetWorldPos(sample_uv, scene_depth);
        float depth_diff = length(curr_world - scene_world);

        // Reversed-Z: scene_depth >= ndc_sample.z
        if (ndc_sample.z <= scene_depth && depth_diff < u_SSR_Thickness) {
            // Binary search refinement for sub-step precision
            vec3 p_start = curr_world - R * step_len;
            vec3 p_end = curr_world;
            for (int b = 0; b < 4; ++b) {
                vec3 p_mid = (p_start + p_end) * 0.5;
                vec4 c_mid = u_ViewProjection * vec4(p_mid, 1.0);
                vec3 n_mid = c_mid.xyz / c_mid.w;
                vec2 uv_mid = n_mid.xy * 0.5 + 0.5;
                float d_mid = texture(u_GBufferDepth, uv_mid).r;
                if (n_mid.z <= d_mid) {
                    p_end = p_mid;
                    hit_uv = uv_mid;
                } else {
                    p_start = p_mid;
                }
            }
            hit_found = 1.0;
            break;
        }
    }

    if (hit_found > 0.0) {
        // Screen edge vignette fade to eliminate hard popping
        vec2 edge_coords = abs(hit_uv - 0.5) * 2.0;
        float edge_factor = clamp(1.0 - max(edge_coords.x, edge_coords.y), 0.0, 1.0);
        edge_factor = smoothstep(0.0, 0.2, edge_factor);

        // Roughness fade
        float rough_fade = 1.0 - smoothstep(0.0, u_SSR_MaxRoughness, roughness);

        vec3 reflected_color = texture(u_SceneColor, hit_uv).rgb;
        float alpha = edge_factor * rough_fade;
        out_SSR = vec4(reflected_color, alpha);
    } else {
        out_SSR = vec4(0.0);
    }
}
