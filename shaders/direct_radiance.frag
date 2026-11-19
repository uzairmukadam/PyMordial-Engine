#version 450 core
// ============================================================================
//  Direct Radiance Pass
//
//  Produces the current-frame GI source radiance consumed by SSGI (and, later,
//  Radiance Cascades). It is deliberately NOT the final composition: it exists
//  so screen-space GI never samples a stale or previously composited HDR frame.
//
//  CONTRACT — what this texture contains:
//      * directional (sun) diffuse, including cascaded shadow visibility
//      * clustered point-light diffuse
//      * clustered spot-light diffuse, including spot shadow visibility
//      * emissive radiance
//
//  CONTRACT — what this texture does NOT contain:
//      * any indirect diffuse (no SSGI, no RC, no IBL irradiance)
//      * any specular term (no direct specular, no SSR, no environment specular)
//      * any previous-frame data
//
//  Because it excludes indirect and specular terms, sampling it from SSGI can
//  never create a recursive feedback loop through the HDR history.
//
//  Output: rgb = direct diffuse + emissive radiance, a = 1.0
// ============================================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_DirectRadiance;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_ShadowAtlas;
layout (binding = 9) uniform sampler2DShadow u_ShadowAtlasCmp;
layout (binding = 12) uniform sampler2D u_SpotShadowAtlas;
layout (binding = 16) uniform sampler2DShadow u_SpotShadowAtlasCmp;

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

struct PointLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 color_intensity; // rgb = color, w = intensity
};

struct SpotLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 dir_inner_cos;   // xyz = normalized direction, w = cos(inner_cone_angle)
    vec4 color_intensity; // rgb = color, w = intensity
    vec4 params;          // x = cos(outer_cone_angle), y = shadow_index (-1 if none), z = bias, w = reserved
};

layout (std430, binding = 3) buffer LightBuffer {
    PointLightData u_PointLights[256];
    SpotLightData u_SpotLights[];
};

uniform int u_CascadeCount = 4;
uniform float u_ShadowBias = 0.0015;
uniform float u_ShadowNormalBias = 0.0010;
uniform float u_ShadowBiasSlopeScale = 2.0;
uniform float u_ShadowBiasMinScale = 0.5;
uniform float u_ShadowBiasMaxScale = 6.0;
uniform int u_ShadowFilterMode = 0;
uniform int u_PCFCascadeLimit = 3;
uniform int u_PCFSamples = 16;
uniform float u_PCFRadius = 1.5;
uniform float u_PCFFalloff = 2.0;
uniform int u_PCSS_CascadeLimit = 1;
uniform int u_PCSSSearchSamples = 16;
uniform float u_PCSSSearchRadius = 48.0;
uniform float u_PCSSLightSize = 0.10;
uniform int u_PCSSSamples = 24;
uniform float u_PCSSFalloff = 2.0;
uniform float u_PCSSMaxRadius = 96.0;
uniform float u_PCSSClosestBlockerWeight = 1.0;
uniform float u_SunAngularRadius = 0.0;
uniform vec4 u_CascadeAtlasRect[4] = vec4[4](
    vec4(0.0, 0.0, 0.5, 0.5),
    vec4(0.5, 0.0, 0.5, 0.5),
    vec4(0.0, 0.5, 0.5, 0.5),
    vec4(0.5, 0.5, 0.5, 0.5)
);
uniform int u_PointLightCount = 0;
uniform int u_SpotLightCount = 0;
uniform int u_SpotShadowEnabled = 1;
uniform mat4 u_SpotLightViewProjection[4];

const float PI = 3.14159265358979323846;

vec2 VogelDiskSample(int sample_index, int sample_count, float phi) {
    float r = sqrt((float(sample_index) + 0.5) / float(sample_count));
    float theta = float(sample_index) * 2.39996323 + phi;
    return vec2(r * cos(theta), r * sin(theta));
}

float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

#include "shaders/common/pbr_brdf.glsl"
#include "shaders/common/shadows.glsl"
#include "shaders/common/csm_sampling.glsl"

// Evaluates localized perspective spot light shadow occlusion.
float SampleSpotShadow(int shadow_idx, vec3 world_pos, vec3 N, float bias) {
    if (shadow_idx < 0 || shadow_idx >= 4 || u_SpotShadowEnabled == 0) return 1.0;

    vec4 light_space_pos = u_SpotLightViewProjection[shadow_idx] * vec4(world_pos + N * (bias * 0.5), 1.0);
    if (light_space_pos.w <= 0.0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.z < 0.0 || proj_coords.z > 1.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    vec2 atlas_offset = vec2(float(shadow_idx % 2) * 0.5, float(shadow_idx / 2) * 0.5);
    vec2 uv = proj_coords.xy * 0.5 + atlas_offset;
    vec2 uv_min = atlas_offset + vec2(0.0005);
    vec2 uv_max = atlas_offset + vec2(0.4995);

    float current_depth = proj_coords.z;
    vec2 full_size = vec2(textureSize(u_SpotShadowAtlas, 0));
    vec2 texel_size = 1.0 / full_size;

    float radius = texel_size.x * 2.0;
    float phi = InterleavedGradientNoise(gl_FragCoord.xy) * 6.2831853;
    float sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        vec2 offset = VogelDiskSample(i, 9, phi) * radius;
        vec2 tap = clamp(uv + offset, uv_min, uv_max);
        float r2 = (float(i) + 0.5) / 9.0;
        float w = (1.0 - r2) * (1.0 - r2);
        sum += texture(u_SpotShadowAtlasCmp, vec3(tap, current_depth - bias)) * w;
        weight_sum += w;
    }
    return sum / max(weight_sum, 1e-5);
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (raw_depth <= 0.000001) {
        out_DirectRadiance = vec4(0.0);
        return;
    }

    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= max(abs(view_pos.w), 1e-6);
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    vec4 normal_metal = texture(u_GBufferNormalMetallic, v_UV);

    vec3 raw_albedo = albedo_rough.rgb;
    float max_c = max(max(raw_albedo.r, raw_albedo.g), raw_albedo.b);
    vec3 albedo = (max_c > 1.0) ? (raw_albedo / max_c) : raw_albedo;
    vec3 emissive = (max_c > 1.0) ? (raw_albedo - albedo) : vec3(0.0);
    float roughness = clamp(albedo_rough.a, 0.04, 1.0);
    vec3 N = OctahedralDecode(normal_metal.rg);
    float metallic = normal_metal.b;

    vec3 V = normalize(u_CameraPos_Time.xyz - world_pos.xyz);
    vec3 L = normalize(-u_SunDirection_Intensity.xyz);

    // Diffuse-only BRDF weight. Specular is intentionally excluded from this
    // texture; the final resolve still computes it.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlick(max(dot(normalize(V + L), V), 0.0), F0);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);
    vec3 radiance = u_SunColor_Ambient.rgb * u_SunDirection_Intensity.w;

    float shadow = 1.0;
    if (NdotL > 0.0) {
        int cascade_idx;
        float coverage;
        shadow = CalculateCSMShadow(world_pos.xyz, N, L, -view_pos.z, cascade_idx, coverage);
    }

    vec3 direct_sun = kD * albedo / PI * radiance * NdotL * shadow;

    // Clustered point lights (diffuse only).
    vec3 point_lights_accum = vec3(0.0);
    int light_count = min(u_PointLightCount, 128);
    for (int i = 0; i < light_count; ++i) {
        vec3 pl_pos = u_PointLights[i].pos_radius.xyz;
        float pl_radius = u_PointLights[i].pos_radius.w;
        vec3 pl_delta = pl_pos - world_pos.xyz;
        float pl_dist = length(pl_delta);

        if (pl_dist < pl_radius) {
            vec3 pl_L = pl_delta / max(pl_dist, 0.0001);
            float pl_NdotL = max(dot(N, pl_L), 0.0);
            if (pl_NdotL > 0.0) {
                float ratio = pl_dist / pl_radius;
                float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                float atten = (win * win) / (pl_dist * pl_dist + 1.0);

                vec3 pl_H = normalize(V + pl_L);
                vec3 pl_F = FresnelSchlick(max(dot(pl_H, V), 0.0), F0);
                vec3 pl_diff = (vec3(1.0) - pl_F) * (1.0 - metallic) * albedo / PI;

                vec3 pl_rad = u_PointLights[i].color_intensity.rgb * u_PointLights[i].color_intensity.w;
                point_lights_accum += pl_diff * pl_rad * pl_NdotL * atten;
            }
        }
    }

    // Clustered spot lights (diffuse only, with spot shadows).
    vec3 spot_lights_accum = vec3(0.0);
    int spot_count = min(u_SpotLightCount, 64);
    for (int i = 0; i < spot_count; ++i) {
        vec3 sp_pos = u_SpotLights[i].pos_radius.xyz;
        float sp_radius = u_SpotLights[i].pos_radius.w;
        vec3 sp_delta = sp_pos - world_pos.xyz;
        float sp_dist = length(sp_delta);

        if (sp_dist < sp_radius) {
            vec3 sp_L = sp_delta / max(sp_dist, 0.0001);
            vec3 sp_dir = normalize(u_SpotLights[i].dir_inner_cos.xyz);
            float inner_cos = u_SpotLights[i].dir_inner_cos.w;
            float outer_cos = u_SpotLights[i].params.x;

            float cos_theta = dot(-sp_L, sp_dir);
            if (cos_theta > outer_cos) {
                float spot_cone = smoothstep(outer_cos, inner_cos, cos_theta);
                float sp_NdotL = max(dot(N, sp_L), 0.0);
                if (sp_NdotL > 0.0) {
                    float ratio = sp_dist / sp_radius;
                    float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                    float atten = (win * win) / (sp_dist * sp_dist + 1.0);

                    int shadow_idx = int(u_SpotLights[i].params.y);
                    float sp_bias = u_SpotLights[i].params.z;
                    float sp_shadow = (shadow_idx >= 0) ? SampleSpotShadow(shadow_idx, world_pos.xyz, N, sp_bias) : 1.0;

                    vec3 sp_H = normalize(V + sp_L);
                    vec3 sp_F = FresnelSchlick(max(dot(sp_H, V), 0.0), F0);
                    vec3 sp_diff = (vec3(1.0) - sp_F) * (1.0 - metallic) * albedo / PI;

                    vec3 sp_rad = u_SpotLights[i].color_intensity.rgb * u_SpotLights[i].color_intensity.w;
                    spot_lights_accum += sp_diff * sp_rad * sp_NdotL * atten * spot_cone * sp_shadow;
                }
            }
        }
    }

    vec3 result = direct_sun + point_lights_accum + spot_lights_accum + emissive;
    if (any(isnan(result)) || any(isinf(result))) {
        result = vec3(0.0);
    }
    out_DirectRadiance = vec4(clamp(result, vec3(0.0), vec3(1e4)), 1.0);
}
