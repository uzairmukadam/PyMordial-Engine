#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_SSGI;

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

uniform int u_SSGI_Steps = 16;
uniform int u_SSGI_RayCount = 8;
uniform float u_SSGI_RayDistance = 3.0;
uniform float u_SSGI_Thickness = 0.35;
uniform float u_SSGI_Intensity = 1.5;

const float PI = 3.14159265358979323846;
const float GOLDEN_RATIO = 1.618033988749895;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// Interleaved Gradient Noise (IGN)
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

vec3 GetViewPos(vec2 uv, float raw_depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / max(abs(view.w), 1e-6);
}

// Low-discrepancy Fibonacci spiral hemisphere sample (uniform coverage without clumping)
vec3 FibonacciHemisphereSample(int index, int num_samples, float phi_rot, vec3 N) {
    float phi = 2.0 * PI * fract(float(index) * GOLDEN_RATIO + phi_rot);
    float cosTheta = sqrt((float(index) + 0.5) / float(num_samples));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = cross(up, N);
    float tlen = length(tangent);
    tangent = (tlen > 1e-5) ? (tangent / tlen) : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (raw_depth <= 0.000001) {
        out_SSGI = vec4(0.0);
        return;
    }

    vec3 view_pos = GetViewPos(v_UV, raw_depth);
    vec3 normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 normal_view = normalize(mat3(u_View) * normal_world);

    // Normal offset in view space to avoid self-intersection
    vec3 ray_origin = view_pos + normal_view * 0.035;

    float noise = InterleavedGradientNoise(gl_FragCoord.xy);
    vec3 indirect_radiance = vec3(0.0);
    float total_hits = 0.0;

    int rays = clamp(u_SSGI_RayCount, 1, 16);
    int steps = clamp(u_SSGI_Steps, 4, 32);
    float step_len = u_SSGI_RayDistance / float(steps);

    for (int r = 0; r < rays; ++r) {
        vec3 ray_dir = FibonacciHemisphereSample(r, rays, noise, normal_view);

        vec3 prev_p = ray_origin;
        vec3 curr_p = ray_origin;
        bool hit = false;
        vec2 hit_uv = vec2(0.0);

        for (int s = 1; s <= steps; ++s) {
            float t = (float(s) - 0.5 + noise) * step_len;
            curr_p = ray_origin + ray_dir * t;

            vec4 clip_sample = u_Projection * vec4(curr_p, 1.0);
            if (clip_sample.w <= 1e-5) break;
            vec2 sample_uv = (clip_sample.xy / clip_sample.w) * 0.5 + 0.5;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }

            float scene_depth = texture(u_GBufferDepth, sample_uv).r;
            if (scene_depth <= 0.000001) {
                prev_p = curr_p;
                continue;
            }

            float scene_view_z = GetViewPos(sample_uv, scene_depth).z;
            // In view space (negative Z): ray is behind surface when scene_view_z - curr_p.z >= 0
            float depth_diff = scene_view_z - curr_p.z;
            float thickness = max(u_SSGI_Thickness, step_len * 1.35) * (1.0 + abs(curr_p.z) * 0.03);

            if (depth_diff >= 0.0 && depth_diff < thickness) {
                // 5-Iteration Binary Search Bisection
                vec3 b_start = prev_p;
                vec3 b_end = curr_p;
                hit_uv = sample_uv;

                for (int b = 0; b < 5; ++b) {
                    vec3 b_mid = (b_start + b_end) * 0.5;
                    vec4 c_mid = u_Projection * vec4(b_mid, 1.0);
                    if (c_mid.w <= 1e-5) break;
                    vec2 uv_mid = (c_mid.xy / c_mid.w) * 0.5 + 0.5;
                    float d_mid = texture(u_GBufferDepth, uv_mid).r;
                    float z_mid = GetViewPos(uv_mid, d_mid).z;

                    if (z_mid - b_mid.z >= 0.0) {
                        b_end = b_mid;
                        hit_uv = uv_mid;
                    } else {
                        b_start = b_mid;
                    }
                }

                // Backface rejection
                vec3 hit_normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, hit_uv).rg);
                vec3 hit_normal_view = normalize(mat3(u_View) * hit_normal_world);

                if (dot(hit_normal_view, -ray_dir) > 0.05) {
                    hit = true;
                    curr_p = b_end;
                }
                break;
            }

            prev_p = curr_p;
        }

        if (hit) {
            // Sample lit radiance (HDR lit buffer containing sunlight + dynamic point lights)
            vec3 hit_radiance = texture(u_SceneColor, hit_uv).rgb;
            if (isnan(hit_radiance.r) || isnan(hit_radiance.g) || isnan(hit_radiance.b) ||
                isinf(hit_radiance.r) || isinf(hit_radiance.g) || isinf(hit_radiance.b) ||
                dot(hit_radiance, hit_radiance) < 0.0001) {
                vec3 hit_albedo = texture(u_GBufferAlbedoRoughness, hit_uv).rgb;
                hit_radiance = hit_albedo * u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * 0.25);
            }

            // Suppress extreme specular fireflies in indirect diffuse
            float lum = dot(hit_radiance, vec3(0.2126, 0.7152, 0.0722));
            float max_lum = 3.5;
            if (lum > max_lum) {
                hit_radiance *= (max_lum / lum);
            }

            float d = length(curr_p - ray_origin);
            float dist_atten = clamp(1.0 - (d * d) / (u_SSGI_RayDistance * u_SSGI_RayDistance), 0.0, 1.0);

            // Screen-edge vignette attenuation
            vec2 edge = smoothstep(0.0, 0.08, hit_uv) * (1.0 - smoothstep(0.92, 1.0, hit_uv));
            float edge_fade = edge.x * edge.y;

            float NdotL = max(dot(normal_view, ray_dir), 0.0);
            indirect_radiance += hit_radiance * NdotL * dist_atten * edge_fade;
            total_hits += edge_fade;
        }
    }

    if (total_hits > 0.0) {
        indirect_radiance = (indirect_radiance / float(rays)) * u_SSGI_Intensity;
    }

    // Comprehensive NaN / Inf sanitizer
    if (isnan(indirect_radiance.r) || isnan(indirect_radiance.g) || isnan(indirect_radiance.b) ||
        isinf(indirect_radiance.r) || isinf(indirect_radiance.g) || isinf(indirect_radiance.b)) {
        indirect_radiance = vec3(0.0);
    }
    float out_hits = clamp(total_hits / float(rays), 0.0, 1.0);
    if (isnan(out_hits) || isinf(out_hits)) out_hits = 0.0;

    out_SSGI = vec4(clamp(indirect_radiance, vec3(0.0), vec3(40.0)), out_hits);
}
