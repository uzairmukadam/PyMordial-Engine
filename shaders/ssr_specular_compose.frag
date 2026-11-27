#version 450 core
// ============================================================================
//  SSR Specular Compose
//
//  Replaces the second full deferred resolve. The base deferred resolve already
//  produced the complete lit HDR image *without* SSR (direct sun, point/spot
//  lights, diffuse IBL, indirect diffuse/GI, and environment specular computed
//  from the pure prefiltered environment). This pass applies ONLY the specular
//  delta that SSR introduces, so no lighting term is recomputed and environment
//  specular is never double-counted.
//
//  The delta mirrors the deferred-resolve specular block exactly:
//
//      env      = prefiltered environment radiance along R
//      newEnv   = max(mix(env, ssr_rgb, a), env * (1 - a * 0.70))
//      delta    = (newEnv - env) * (F_env * brdf.x + brdf.y) * spec_ao * energy_comp
//      composed = base_hdr + delta
//
//  When SSR misses (a == 0) newEnv == env, so delta == 0 and the base HDR
//  environment reflection is preserved untouched.
// ============================================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_HDRColor;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 4) uniform sampler2D u_AOTexture;
layout (binding = 6) uniform sampler2D u_BRDFLUT;
layout (binding = 7) uniform sampler2D u_EnvironmentMap;
layout (binding = 8) uniform sampler2D u_SSRTexture;
layout (binding = 14) uniform sampler2D u_BaseHDR;

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

uniform int u_AOEnabled = 1;
uniform int u_IBLEnabled = 1;
uniform int u_EnergyCompensationEnabled = 1;

#include "shaders/common/pbr_brdf.glsl"

void main() {
    vec3 base = texture(u_BaseHDR, v_UV).rgb;

    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    // Background / sky pixels carry no surface specular; the base HDR already
    // holds the correct sky radiance.
    if (raw_depth <= 0.000001 || u_IBLEnabled != 1) {
        out_HDRColor = vec4(base, 1.0);
        return;
    }

    // Reconstruct world position from Reversed-Z depth.
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= max(abs(view_pos.w), 1e-6);
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    vec4 normal_metal = texture(u_GBufferNormalMetallic, v_UV);

    vec3 raw_albedo = albedo_rough.rgb;
    float max_c = max(max(raw_albedo.r, raw_albedo.g), raw_albedo.b);
    vec3 albedo = (max_c > 1.0) ? (raw_albedo / max_c) : raw_albedo;
    float roughness = clamp(albedo_rough.a, 0.04, 1.0);
    vec3 N = OctahedralDecode(normal_metal.rg);
    float metallic = normal_metal.b;
    float ao = normal_metal.a;
    if (u_AOEnabled == 1) {
        ao *= texture(u_AOTexture, v_UV).r;
    }

    vec3 V = normalize(u_CameraPos_Time.xyz - world_pos.xyz);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NdotV = max(dot(N, V), 0.0);

    // Environment specular terms (identical to the deferred resolve).
    vec3 R = reflect(-V, N);
    vec3 brdf = texture(u_BRDFLUT, vec2(NdotV, roughness)).rgb;
    vec2 env_uv = vec2(atan(R.z, R.x) / (2.0 * PI) + 0.5,
                       asin(clamp(R.y, -0.999, 0.999)) / PI + 0.5);
    float max_env_lod = float(max(textureQueryLevels(u_EnvironmentMap) - 1, 0));
    vec3 env_radiance = textureLod(u_EnvironmentMap, env_uv, roughness * max_env_lod).rgb;

    // SSR blend (identical to the deferred resolve).
    vec4 ssr_sample = texture(u_SSRTexture, v_UV);
    if (isnan(ssr_sample.r) || isnan(ssr_sample.g) || isnan(ssr_sample.b) ||
        isinf(ssr_sample.r) || isinf(ssr_sample.g) || isinf(ssr_sample.b) ||
        isnan(ssr_sample.a) || isinf(ssr_sample.a)) {
        ssr_sample = vec4(0.0);
    }
    vec3 safe_ssr_rgb = clamp(ssr_sample.rgb, vec3(0.0), vec3(30.0));
    float safe_ssr_a = clamp(ssr_sample.a, 0.0, 1.0);

    vec3 new_env = env_radiance;
    if (safe_ssr_a > 0.001) {
        vec3 blended_refl = mix(env_radiance, safe_ssr_rgb, safe_ssr_a);
        new_env = max(blended_refl, env_radiance * (1.0 - safe_ssr_a * 0.70));
    }

    // Specular occlusion + energy compensation (identical to the deferred resolve).
    float spec_ao = clamp(ao * ao + sqrt(NdotV) * (1.0 - ao), 0.0, 1.0);
    vec3 energy_comp = vec3(1.0);
    if (u_EnergyCompensationEnabled == 1) {
        float E = texture(u_BRDFLUT, vec2(max(NdotV, 0.001), roughness)).b;
        energy_comp = GGXEnergyCompensation(F0, roughness, E);
    }
    vec3 F_env = FresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 spec_weight = (F_env * brdf.x + brdf.y) * spec_ao * energy_comp;

    // Only the specular delta is added; every other lighting term is already in
    // the base HDR and must not be recomputed.
    vec3 delta = (new_env - env_radiance) * spec_weight;
    vec3 composed = base + delta;

    if (isnan(composed.r) || isnan(composed.g) || isnan(composed.b) ||
        isinf(composed.r) || isinf(composed.g) || isinf(composed.b)) {
        composed = base;
    }
    out_HDRColor = vec4(clamp(composed, vec3(0.0), vec3(1e4)), 1.0);
}
