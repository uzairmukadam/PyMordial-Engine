#version 450 core
// ============================================================================
//  Experimental SSR using the shared Hi-Z tracer
//
//  This shader is a thin wrapper around shaders/common/hiz_trace.glsl. All
//  traversal lives in the shared tracer; everything SSR-specific (roughness
//  gating, confidence, edge fade, radiance lookup, BRDF approximation) stays
//  here.
//
//  Debug views are selected with u_SSRDebugView:
//      0 = off (normal reflection colour)
//      1 = hit UV
//      2 = hit distance
//      3 = iteration count
//      4 = miss reason
//      5 = final mip
// ============================================================================

#include "shaders/common/depth.glsl"
#include "shaders/common/hiz_bounds.glsl"

in vec2 v_UV;
layout (location = 0) out vec4 out_SSR;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_SceneColor;
layout (binding = 4) uniform sampler2D u_HiZTexture;   // RG32F linear-depth bounds

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

uniform float u_SSR_MaxDistance = 20.0;
uniform float u_SSR_Thickness = 0.40;
uniform float u_SSR_MaxRoughness = 0.65;
uniform int u_SSR_MaxIterations = 96;
uniform int u_SSRDebugView = 0;
uniform vec2 u_TargetResolution = vec2(1.0);

#include "shaders/common/hiz_trace.glsl"

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// False-colour ramp for the debug views.
vec3 DebugRamp(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(t, 1.0 - abs(t * 2.0 - 1.0), 1.0 - t);
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    if (IsBackgroundDepth(raw_depth)) {
        out_SSR = vec4(0.0);
        return;
    }

    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    float roughness = albedo_rough.a;
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

    if (dot(view_normal, R) <= 0.001) {
        out_SSR = vec4(0.0);
        return;
    }

    // Normal offset to avoid self-reflection acne.
    vec3 ray_origin = view_pos + view_normal * 0.02;

    HiZTraceRequest request;
    request.originView = ray_origin;
    request.directionView = R;
    request.minDistance = 0.05;
    request.maxDistance = u_SSR_MaxDistance;
    request.thickness = u_SSR_Thickness;
    request.startMip = 0;
    request.maxIterations = u_SSR_MaxIterations;

    HiZTraceHit hit = TraceHiZ(u_HiZTexture, u_GBufferDepth, request);

    // ---- Debug views -------------------------------------------------------
    if (u_SSRDebugView == 1) {
        out_SSR = vec4(hit.hit ? vec3(hit.uv, 0.0) : vec3(0.0), 1.0);
        return;
    }
    if (u_SSRDebugView == 2) {
        out_SSR = vec4(hit.hit ? DebugRamp(hit.rayDistance / max(u_SSR_MaxDistance, 1e-3)) : vec3(0.0), 1.0);
        return;
    }
    if (u_SSRDebugView == 3) {
        out_SSR = vec4(DebugRamp(float(hit.iterations) / float(max(u_SSR_MaxIterations, 1))), 1.0);
        return;
    }
    if (u_SSRDebugView == 4) {
        // Distinct colour per miss reason; green when the ray hit.
        vec3 c = vec3(0.0);
        if (hit.hit) {
            c = vec3(0.0, 1.0, 0.0);
        } else if (hit.missReason == HIZ_MISS_LEFT_SCREEN) {
            c = vec3(1.0, 0.0, 0.0);
        } else if (hit.missReason == HIZ_MISS_EXCEEDED_DIST) {
            c = vec3(1.0, 1.0, 0.0);
        } else if (hit.missReason == HIZ_MISS_EXCEEDED_ITERS) {
            c = vec3(1.0, 0.0, 1.0);
        } else if (hit.missReason == HIZ_MISS_INVALID_PROJ) {
            c = vec3(0.0, 1.0, 1.0);
        } else {
            c = vec3(0.0, 0.0, 1.0);
        }
        out_SSR = vec4(c, 1.0);
        return;
    }
    if (u_SSRDebugView == 5) {
        out_SSR = vec4(DebugRamp(float(hit.finalMip) / 8.0), 1.0);
        return;
    }

    // ---- Normal reflection shading ----------------------------------------
    if (!hit.hit) {
        out_SSR = vec4(0.0);
        return;
    }

    // SSR-specific validation: reject self-hits and back faces.
    vec3 hit_view_pos = ReconstructViewPosition(hit.uv, hit.rawDepth, u_InvProjection);
    float plane_dist = dot(hit_view_pos - view_pos, view_normal);
    vec3 hit_normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, hit.uv).rg);
    vec3 hit_normal_view = normalize((u_View * vec4(hit_normal_world, 0.0)).xyz);

    bool self_hit = dot(view_normal, hit_normal_view) > 0.65 && (hit.rayDistance < 2.5 || plane_dist < 1.5);
    bool back_facing = dot(R, hit_normal_view) > -0.05;
    bool too_close = length(hit.uv - v_UV) < 0.008 || hit.rayDistance < 0.05;

    if (plane_dist <= 0.02 || self_hit || back_facing || too_close) {
        out_SSR = vec4(0.0);
        return;
    }

    // Screen edge vignette fade to eliminate hard boundary popping.
    vec2 edge_coords = abs(hit.uv - 0.5) * 2.0;
    float edge_factor = clamp(1.0 - max(edge_coords.x, edge_coords.y), 0.0, 1.0);
    edge_factor = smoothstep(0.0, 0.15, edge_factor);

    float dist_fade = clamp(1.0 - (hit.rayDistance / u_SSR_MaxDistance), 0.0, 1.0);
    float rough_fade = 1.0 - smoothstep(0.0, u_SSR_MaxRoughness, roughness);
    float grazing_fade = smoothstep(0.001, 0.12, dot(view_normal, R));

    vec3 reflected_color = texture(u_SceneColor, hit.uv).rgb;
    if (any(isnan(reflected_color)) || any(isinf(reflected_color))) {
        reflected_color = vec3(0.0);
    }
    reflected_color = clamp(reflected_color, vec3(0.0), vec3(40.0));

    // SSR runs before deferred resolve, so approximate direct and ambient hit
    // lighting from material albedo.
    vec3 to_sun = normalize(-u_SunDirection_Intensity.xyz);
    float NdotL = max(dot(hit_normal_world, to_sun), 0.0);
    vec3 sun_radiance = u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * 0.35);
    vec3 hit_lit_radiance = reflected_color * (sun_radiance * NdotL + vec3(0.40));
    reflected_color = max(reflected_color * 0.7, hit_lit_radiance);

    float refl_lum = dot(reflected_color, vec3(0.2126, 0.7152, 0.0722));
    float energy_fade = smoothstep(0.01, 0.08, refl_lum);

    float alpha = edge_factor * dist_fade * rough_fade * grazing_fade * energy_fade;
    if (isnan(alpha) || isinf(alpha)) alpha = 0.0;
    out_SSR = vec4(reflected_color, clamp(alpha, 0.0, 1.0));
}
