#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_SSGI;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;

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

uniform int u_SSGI_Steps = 12;
uniform int u_SSGI_RayCount = 4;
uniform float u_SSGI_RayDistance = 2.5;
uniform float u_SSGI_Thickness = 0.4;
uniform float u_SSGI_Intensity = 1.2;

const float PI = 3.14159265358979323846;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// Spatial dithering noise
float Hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 GetWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    view /= view.w;
    return (u_InvView * vec4(view.xyz, 1.0)).xyz;
}

// Cosine-weighted hemisphere sample oriented around normal N
vec3 CosineSampleHemisphere(vec2 xi, vec3 N) {
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(xi.y);
    float sinTheta = sqrt(max(1.0 - xi.y, 0.0));

    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (raw_depth <= 0.000001) {
        out_SSGI = vec4(0.0);
        return;
    }

    vec3 world_pos = GetWorldPos(v_UV, raw_depth);
    vec3 N = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);

    // Normal offset to avoid self-intersection
    vec3 ray_origin = world_pos + N * 0.04;

    float noise = Hash12(gl_FragCoord.xy);
    vec3 indirect_radiance = vec3(0.0);
    float total_weight = 0.0;

    int rays = clamp(u_SSGI_RayCount, 1, 8);
    int steps = clamp(u_SSGI_Steps, 4, 24);

    for (int r = 0; r < rays; ++r) {
        vec2 xi = vec2(fract(float(r) / float(rays) + noise), fract(noise * 7.13 + float(r) * 0.37));
        vec3 ray_dir = CosineSampleHemisphere(xi, N);

        // Raymarch along ray_dir
        vec3 curr_world_pos = ray_origin;
        float step_length = u_SSGI_RayDistance / float(steps);

        for (int s = 1; s <= steps; ++s) {
            curr_world_pos = ray_origin + ray_dir * (float(s) * step_length);

            // Project to screen space
            vec4 clip_sample = u_ViewProjection * vec4(curr_world_pos, 1.0);
            if (clip_sample.w <= 0.0) break;
            vec3 ndc_sample = clip_sample.xyz / clip_sample.w;
            vec2 sample_uv = ndc_sample.xy * 0.5 + 0.5;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }

            float scene_depth = texture(u_GBufferDepth, sample_uv).r;
            if (scene_depth <= 0.000001) continue;

            vec3 hit_world_pos = GetWorldPos(sample_uv, scene_depth);
            float dist_to_surface = length(curr_world_pos - hit_world_pos);

            // In Reversed-Z: scene_depth >= ndc_sample.z means surface is in front or at sample point
            if (ndc_sample.z <= scene_depth && dist_to_surface < u_SSGI_Thickness) {
                vec3 hit_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, sample_uv).rg);
                // Backface rejection (bounce surface must face back towards our pixel)
                if (dot(hit_normal, -ray_dir) > 0.05) {
                    vec3 hit_albedo = texture(u_GBufferAlbedoRoughness, sample_uv).rgb;
                    float NdotL = max(dot(N, ray_dir), 0.0);
                    // Distance attenuation (inverse square with windowing)
                    float d = float(s) * step_length;
                    float atten = clamp(1.0 - (d / u_SSGI_RayDistance), 0.0, 1.0);

                    indirect_radiance += hit_albedo * NdotL * atten;
                    total_weight += 1.0;
                    break;
                }
            }
        }
    }

    if (total_weight > 0.0) {
        indirect_radiance = (indirect_radiance / float(rays)) * u_SSGI_Intensity;
    }

    out_SSGI = vec4(indirect_radiance, total_weight / float(rays));
}
