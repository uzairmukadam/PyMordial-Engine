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
uniform float u_Intensity = 1.2;
uniform float u_Power = 1.5;
uniform float u_Bias = 0.025;
uniform int u_Samples = 20;

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

// Jorge Jimenez's Interleaved Gradient Noise for spatial dithering
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

    // Build orthonormal basis (T, B, N) aligned with view normal
    vec3 up = abs(view_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, view_normal));
    vec3 bitangent = cross(view_normal, tangent);

    // Dither rotation using Interleaved Gradient Noise
    float noise = InterleavedGradientNoise(gl_FragCoord.xy);
    float rot_angle = noise * 2.0 * PI;
    float cos_rot = cos(rot_angle);
    float sin_rot = sin(rot_angle);
    vec3 rot_tangent = tangent * cos_rot + bitangent * sin_rot;
    vec3 rot_bitangent = -tangent * sin_rot + bitangent * cos_rot;
    mat3 tbn = mat3(rot_tangent, rot_bitangent, view_normal);

    int num_samples = clamp(u_Samples, 8, 32);
    float occlusion = 0.0;
    float valid_samples = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        // Fibonacci hemisphere point
        float z = (float(i) + 0.5) / float(num_samples);
        float r_xy = sqrt(max(0.0, 1.0 - z * z));
        float phi = float(i) * 2.39996323; // Golden angle (137.5 deg)
        vec3 hemi_sample = vec3(r_xy * cos(phi), r_xy * sin(phi), z);

        // Cubic distance scaling: concentrate samples close to surface for contact crevices
        float scale = float(i + 1) / float(num_samples);
        scale = mix(0.08, 1.0, scale * scale);

        vec3 sample_offset = tbn * (hemi_sample * (u_Radius * scale));
        vec3 sample_view_pos = view_pos + sample_offset;

        // Project back to screen UV
        vec4 sample_clip = u_Projection * vec4(sample_view_pos, 1.0);
        if (sample_clip.w <= 0.0) continue;
        vec2 sample_uv = (sample_clip.xy / sample_clip.w) * 0.5 + 0.5;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        float sample_depth = texture(u_GBufferDepth, sample_uv).r;
        if (sample_depth <= 0.000001) continue;

        vec3 actual_view_pos = GetViewPos(sample_uv, sample_depth);

        // Depth test in view space (-Z is distance from camera)
        float depth_delta = (-actual_view_pos.z) - (-sample_view_pos.z);
        float dist_to_surface = abs(view_pos.z - actual_view_pos.z);

        // Smooth distance attenuation to avoid occluding across depth discontinuities
        float range_atten = smoothstep(0.0, 1.0, u_Radius / (dist_to_surface + 0.001));

        if (depth_delta < -u_Bias) {
            occlusion += range_atten;
        }
        valid_samples += 1.0;
    }

    float ao = 1.0;
    if (valid_samples > 0.0) {
        ao = clamp(1.0 - (occlusion / valid_samples) * u_Intensity, 0.0, 1.0);
        ao = pow(ao, u_Power);
    }

    // Distance fade-out: Ambient Occlusion is localized to close-range contact crevices;
    // fade smoothly to unoccluded (1.0) beyond 70m to eliminate distant screen-space noise
    float view_z = max(-view_pos.z, 0.1);
    float dist_fade = clamp((view_z - 70.0) / 50.0, 0.0, 1.0);
    out_AO = mix(ao, 1.0, dist_fade);
}
