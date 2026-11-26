#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_SSR;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_SceneColor;
// Shared SSRT tracer binding contract: the common tracer reads the surface-ID
// G-buffer for self-hit rejection. SSR keeps its own reflection-specific
// rejection policy, so this is bound for contract consistency only.
layout (binding = 4) uniform sampler2D u_GBufferSurfaceID;

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
// Resolution of the SSR target itself. Pixel-space ray math must use this rather
// than the full viewport size, otherwise a downscaled SSR target mis-measures
// ray lengths in pixels.
uniform vec2 u_TargetResolution = vec2(1.0);

#include "shaders/common/depth.glsl"
#include "shaders/common/pbr_brdf.glsl"
#include "shaders/common/ssrt/ssrt_common.glsl"
#include "shaders/common/ssrt/ssrt_trace.glsl"

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

    vec3 view_pos = ReconstructViewPosition(v_UV, raw_depth, u_InvProjection);
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
    int steps = clamp(u_SSR_Steps, 24, 128);

    // ---- Shared SSRT trace policy (specular reflections) ----
    // Preserves the previous SSR traversal exactly:
    //   thickness = clamp(max(base, step_len*0.20) * (1 + ray_depth*0.004), 0.03, 0.35)
    //   six-iteration bounded binary refinement
    //   tangent-plane / convex-surface / surface-orientation self-hit rejection
    //   abort once the ray passes behind the near plane (z >= -0.08)
    SSRT_TraceConfig traceCfg;
    traceCfg.baseThickness        = u_SSR_Thickness;
    traceCfg.thicknessStepScale   = 0.20;
    traceCfg.thicknessStepCap     = 1e30;
    traceCfg.thicknessDepthScale  = 0.004;
    traceCfg.thicknessMin         = 0.03;
    traceCfg.thicknessMax         = 0.35;
    traceCfg.refineSteps          = 6;
    traceCfg.flags                = SSRT_TRACE_REJECT_PLANE | SSRT_TRACE_REJECT_CONVEX |
                                    SSRT_TRACE_REJECT_ORIENTATION | SSRT_TRACE_REFINE_BOUNDED |
                                    SSRT_TRACE_SELF_UV_POST_REFINE;
    traceCfg.planeThreshold       = 0.02;
    traceCfg.convexNormalDot      = 0.65;
    traceCfg.convexNearDist       = 2.5;
    traceCfg.convexPlaneDist      = 1.5;
    traceCfg.orientationThreshold = -0.05;
    traceCfg.nearClipZ            = -0.08;
    traceCfg.selfUVEpsilon        = 0.008;

    // SSR does not use surface-ID self-hit rejection (it relies on the
    // reflection-specific plane/convex/orientation tests above), so the origin
    // surface ID is passed as 0 to keep the shared policy disabled.
    SSRT_Hit hit = SSRT_TraceLinear(ray_origin, R, v_UV, jitter, 0.0, u_SSR_MaxDistance, steps,
                                    0.0, view_normal, traceCfg);

    vec2 hit_uv = hit.uv;
    float hit_found = hit.hit ? 1.0 : 0.0;
    float hit_dist = hit.distance;
    vec3 final_hit_normal = vec3(0.0, 1.0, 0.0);
    if (hit.hit) {
        final_hit_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, hit_uv).rg);
    }

    if (hit_found > 0.0) {
        // Screen edge vignette fade to eliminate hard boundary popping.
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

        // SSR runs before deferred resolve, so approximate direct and ambient hit lighting from material albedo.
        vec3 to_sun = normalize(-u_SunDirection_Intensity.xyz);
        float NdotL = max(dot(final_hit_normal, to_sun), 0.0);
        vec3 sun_radiance = u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * 0.35);
        vec3 hit_lit_radiance = reflected_color * (sun_radiance * NdotL + vec3(0.40));
        reflected_color = max(reflected_color * 0.7, hit_lit_radiance);

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
