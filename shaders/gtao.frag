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

uniform float u_Radius = 0.75;
uniform float u_Intensity = 1.0;
uniform float u_Power = 1.5;
uniform int u_Directions = 3;
uniform int u_Steps = 4;

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

// Interleaved gradient noise for spatial dithering
float SpatialNoise(vec2 coord) {
    return fract(52.9829189 * fract(dot(coord, vec2(0.06711056, 0.00583715))));
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
    vec3 view_dir = normalize(-view_pos);

    // Screen-space radius inversely proportional to view depth
    float screen_radius = clamp(u_Radius / max(-view_pos.z, 0.1), 0.002, 0.15);
    float noise = SpatialNoise(gl_FragCoord.xy);

    float visibility = 0.0;
    float num_directions = float(clamp(u_Directions, 1, 6));
    float num_steps = float(clamp(u_Steps, 2, 8));

    for (int d = 0; d < u_Directions; ++d) {
        float angle = (float(d) + noise) * (PI / num_directions);
        vec2 dir = vec2(cos(angle), sin(angle));

        float horizon_cos = -1.0;

        for (int s = 1; s <= u_Steps; ++s) {
            float step_ratio = float(s) / num_steps;
            vec2 sample_uv = v_UV + dir * screen_radius * (step_ratio * step_ratio);

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            float sample_depth = texture(u_GBufferDepth, sample_uv).r;
            if (sample_depth <= 0.000001) continue;

            vec3 sample_pos = GetViewPos(sample_uv, sample_depth);
            vec3 delta = sample_pos - view_pos;
            float dist = length(delta);

            if (dist < u_Radius * 2.0 && dist > 0.001) {
                vec3 delta_dir = delta / dist;
                // Cosine of angle between view direction and horizon
                float h_cos = dot(delta_dir, view_dir);
                // Weight by normal orientation
                float norm_weight = max(dot(delta_dir, view_normal), 0.0);
                // Falloff with distance
                float falloff = clamp(1.0 - (dist / (u_Radius * 2.0)), 0.0, 1.0);
                horizon_cos = max(horizon_cos, h_cos * norm_weight * falloff);
            }
        }

        visibility += clamp(1.0 - max(horizon_cos, 0.0), 0.0, 1.0);
    }

    float ao = visibility / num_directions;
    ao = clamp(pow(ao, u_Power * u_Intensity), 0.0, 1.0);
    out_AO = ao;
}
