#version 450 core

in vec2 v_UV;
layout (location = 0) out float out_AO;

layout (binding = 0) uniform sampler2D u_GBufferDepth;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;

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
uniform float u_AngleBias = 0.15; // In radians (~8.6 degrees)
uniform int u_Directions = 4;
uniform int u_Steps = 6;

const float PI = 3.14159265358979323846;

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
    return view.xyz / view.w;
}

// Jorge Jimenez's Interleaved Gradient Noise
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

void main() {
    float center_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_depth <= 0.000001) {
        out_AO = 1.0;
        return;
    }

    vec3 view_pos = GetViewPos(v_UV, center_depth);
    vec3 world_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 view_normal = normalize((u_View * vec4(world_normal, 0.0)).xyz);

    float noise = InterleavedGradientNoise(gl_FragCoord.xy);
    float sin_bias = sin(u_AngleBias);

    // Physically projected radius in UV coordinates
    float view_z = max(-view_pos.z, 0.1);
    float proj_scale = u_Projection[1][1] * 0.5;
    vec2 screen_radius_uv = vec2(
        (u_Radius * proj_scale / view_z) * (u_ScreenSize_Jitter.y / u_ScreenSize_Jitter.x),
        (u_Radius * proj_scale / view_z)
    );
    // Clamp to ensure both fine contact crevice sampling (min 4 pixels) and reasonable maximum
    vec2 min_radius_uv = 4.0 / u_ScreenSize_Jitter.xy;
    vec2 max_radius_uv = vec2(0.20);
    screen_radius_uv = clamp(screen_radius_uv, min_radius_uv, max_radius_uv);

    int num_dirs = clamp(u_Directions, 2, 8);
    int num_steps = clamp(u_Steps, 2, 12);
    float total_obscurance = 0.0;

    for (int d = 0; d < num_dirs; ++d) {
        // Orthogonal direction rotated by IGN blue noise
        float angle = (float(d) + noise) * (2.0 * PI / float(num_dirs));
        vec2 dir = vec2(cos(angle), sin(angle));

        float max_horizon = 0.0;

        for (int s = 1; s <= num_steps; ++s) {
            // Linear sub-texel stepping to reliably sample 1-3 pixel contact seams
            float alpha = (float(s) - 0.5 + noise * 0.5) / float(num_steps);
            vec2 sample_uv = v_UV + dir * screen_radius_uv * alpha;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            float sample_depth = texture(u_GBufferDepth, sample_uv).r;
            if (sample_depth <= 0.000001) continue;

            vec3 sample_pos = GetViewPos(sample_uv, sample_depth);
            vec3 delta = sample_pos - view_pos;
            float dist = length(delta);

            if (dist < u_Radius && dist > 0.002) {
                vec3 delta_dir = delta / dist;
                // Elevation relative to view-space surface normal
                float elevation = dot(delta_dir, view_normal);

                if (elevation > sin_bias) {
                    // Physical inverse-square falloff W(r) = max(0, 1 - (r/R)^2)
                    float dist_ratio = dist / u_Radius;
                    float falloff = max(0.0, 1.0 - dist_ratio * dist_ratio);
                    float obs = (elevation - sin_bias) * falloff;
                    max_horizon = max(max_horizon, obs);
                }
            }
        }

        total_obscurance += max_horizon;
    }

    float ao = clamp(1.0 - (total_obscurance / float(num_dirs)) * u_Intensity, 0.0, 1.0);
    ao = pow(ao, u_Power);

    // Distance fade-out: Ambient Occlusion is localized to close-range contact crevices;
    // fade smoothly to unoccluded (1.0) beyond 70m to eliminate distant screen-space noise
    float dist_fade = clamp((view_z - 70.0) / 50.0, 0.0, 1.0);
    out_AO = mix(ao, 1.0, dist_fade);
}
