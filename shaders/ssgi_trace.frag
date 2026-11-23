#version 450 core
// ============================================================================
//  SSGI Trace — shared Hi-Z hierarchical tracer backend
//
//  Keeps every SSGI-specific responsibility here and delegates only the
//  screen-space intersection to shaders/common/hiz_trace.glsl:
//
//      * cosine-weighted hemisphere ray generation
//      * stochastic rotation
//      * conservative origin offset
//      * hit-normal backface rejection
//      * surface-ID rejection
//      * distance attenuation and screen-edge confidence
//      * direct-radiance sampling and diffuse integration
//
//  Output contract (matches the legacy tracer so the spatial/temporal stages
//  are backend agnostic):
//      rgb = indirect diffuse radiance
//      a   = ray hit ratio (confidence)
//
//  Debug views (u_SSGIDebugView):
//      0 = off, 1 = raw radiance, 2 = first hit UV, 3 = first hit distance,
//      4 = hit ratio, 5 = average tracer iterations, 6 = miss reason,
//      7 = origin self-rejection, 8 = surface rejection
// ============================================================================

#include "shaders/common/depth.glsl"
#include "shaders/common/hiz_bounds.glsl"

in vec2 v_UV;
layout (location = 0) out vec4 out_SSGI;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_DirectRadiance;   // current-frame direct diffuse + emissive
layout (binding = 4) uniform sampler2D u_HiZTexture;       // RG32F linear-depth bounds
layout (binding = 5) uniform sampler2D u_GBufferSurfaceID;

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

uniform int u_SSGI_RayCount = 8;
uniform float u_SSGI_RayDistance = 3.0;
uniform float u_SSGI_Thickness = 0.35;
uniform float u_SSGI_Intensity = 1.5;
uniform int u_SSGI_MaxIterations = 64;
uniform int u_SSGIDebugView = 0;
// Surface-ID rejection is stricter than the legacy marcher, which only rejects
// backfaces. Off by default so the two backends stay comparable.
uniform int u_SSGI_SurfaceReject = 0;

#include "shaders/common/hiz_trace.glsl"

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

float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
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
    if (IsBackgroundDepth(raw_depth)) {
        out_SSGI = vec4(0.0);
        return;
    }

    vec3 view_pos = ReconstructViewPosition(v_UV, raw_depth, u_InvProjection);
    vec3 normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 normal_view = normalize(mat3(u_View) * normal_world);
    float origin_surface_id = texture(u_GBufferSurfaceID, v_UV).r;

    // Conservative origin offset along the normal to avoid immediate
    // self-intersection with the receiver's own surface.
    vec3 ray_origin = view_pos + normal_view * max(0.035, abs(view_pos.z) * 0.0008);

    float noise = InterleavedGradientNoise(gl_FragCoord.xy);
    vec3 indirect_radiance = vec3(0.0);
    float total_hits = 0.0;
    float total_iterations = 0.0;
    float self_rejections = 0.0;
    float surface_rejections = 0.0;

    int rays = clamp(u_SSGI_RayCount, 1, 16);
    float max_distance = max(u_SSGI_RayDistance, 1e-3);

    // First-hit debug capture.
    bool have_first_hit = false;
    vec2 first_hit_uv = vec2(0.0);
    float first_hit_distance = 0.0;
    int first_miss_reason = HIZ_MISS_NO_INTERSECTION;

    for (int r = 0; r < rays; ++r) {
        vec3 ray_dir = FibonacciHemisphereSample(r, rays, noise, normal_view);

        HiZTraceRequest request;
        request.originView = ray_origin;
        request.directionView = ray_dir;
        // Start beyond the thickness slab. The tracer accepts a hit whenever the
        // ray is within `thickness` of a surface, so starting inside the slab
        // would immediately "hit" the receiver's own surface.
        request.minDistance = max(0.05, u_SSGI_Thickness);
        request.maxDistance = max_distance;
        request.thickness = u_SSGI_Thickness;
        request.startMip = 0;
        request.maxIterations = u_SSGI_MaxIterations;
        request.multiAscent = 0;

        HiZTraceHit hit = TraceHiZ(u_HiZTexture, u_GBufferDepth, request);
        total_iterations += float(hit.iterations);

        if (!have_first_hit) {
            first_miss_reason = hit.missReason;
            if (hit.hit) {
                have_first_hit = true;
                first_hit_uv = hit.uv;
                first_hit_distance = hit.rayDistance;
            }
        }

        if (!hit.hit) {
            continue;
        }

        // SSGI-specific hit validation (kept outside the shared tracer).
        if (length(hit.uv - v_UV) < 0.008) {
            self_rejections += 1.0;
            continue;
        }

        vec3 hit_normal_world = OctahedralDecode(texture(u_GBufferNormalMetallic, hit.uv).rg);
        vec3 hit_normal_view = normalize(mat3(u_View) * hit_normal_world);

        // Backface rejection: a diffuse bounce must land on a surface facing
        // back towards the receiver.
        if (dot(hit_normal_view, -ray_dir) <= 0.05) {
            surface_rejections += 1.0;
            continue;
        }

        // Surface-ID rejection: never integrate the receiver's own surface.
        // Optional because it also rejects legitimate same-object bounces.
        if (u_SSGI_SurfaceReject != 0) {
            float hit_surface_id = texture(u_GBufferSurfaceID, hit.uv).r;
            if (origin_surface_id > 0.0 && abs(hit_surface_id - origin_surface_id) < 0.25) {
                surface_rejections += 1.0;
                continue;
            }
        }

        // Current-frame direct radiance at the accepted hit.
        vec3 hit_radiance = texture(u_DirectRadiance, hit.uv).rgb;
        if (any(isnan(hit_radiance)) || any(isinf(hit_radiance))) {
            hit_radiance = vec3(0.0);
        }
        if (dot(hit_radiance, hit_radiance) < 0.0001) {
            vec3 hit_albedo = texture(u_GBufferAlbedoRoughness, hit.uv).rgb;
            hit_radiance = hit_albedo * u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * 0.25);
        }

        // Suppress extreme specular fireflies in indirect diffuse.
        float lum = dot(hit_radiance, vec3(0.2126, 0.7152, 0.0722));
        hit_radiance *= 1.0 / (1.0 + lum * 0.75);
        hit_radiance = clamp(hit_radiance, vec3(0.0), vec3(1.75));

        float d = hit.rayDistance;
        float dist_atten = clamp(1.0 - (d * d) / (max_distance * max_distance), 0.0, 1.0);

        // Screen-edge vignette attenuation.
        vec2 edge = smoothstep(0.0, 0.08, hit.uv) * (1.0 - smoothstep(0.92, 1.0, hit.uv));
        float edge_fade = edge.x * edge.y;

        float NdotL = max(dot(normal_view, ray_dir), 0.0);
        indirect_radiance += hit_radiance * NdotL * dist_atten * edge_fade;
        total_hits += edge_fade;
    }

    if (total_hits > 0.0) {
        indirect_radiance = (indirect_radiance / float(rays)) * u_SSGI_Intensity;
    }

    if (any(isnan(indirect_radiance)) || any(isinf(indirect_radiance))) {
        indirect_radiance = vec3(0.0);
    }
    float out_hits = clamp(total_hits / float(rays), 0.0, 1.0);
    if (isnan(out_hits) || isinf(out_hits)) out_hits = 0.0;

    // ---- Debug views -------------------------------------------------------
    if (u_SSGIDebugView == 1) {
        out_SSGI = vec4(clamp(indirect_radiance, vec3(0.0), vec3(40.0)), out_hits);
        return;
    }
    if (u_SSGIDebugView == 2) {
        out_SSGI = vec4(have_first_hit ? vec3(first_hit_uv, 0.0) : vec3(0.0), out_hits);
        return;
    }
    if (u_SSGIDebugView == 3) {
        float t = have_first_hit ? clamp(first_hit_distance / max_distance, 0.0, 1.0) : 0.0;
        out_SSGI = vec4(vec3(t), out_hits);
        return;
    }
    if (u_SSGIDebugView == 4) {
        out_SSGI = vec4(vec3(out_hits), out_hits);
        return;
    }
    if (u_SSGIDebugView == 5) {
        float avg_iters = total_iterations / float(rays);
        out_SSGI = vec4(vec3(clamp(avg_iters / float(max(u_SSGI_MaxIterations, 1)), 0.0, 1.0)), out_hits);
        return;
    }
    if (u_SSGIDebugView == 6) {
        vec3 c = vec3(0.0);
        if (first_miss_reason == HIZ_MISS_NONE) c = vec3(0.0, 1.0, 0.0);
        else if (first_miss_reason == HIZ_MISS_LEFT_SCREEN) c = vec3(1.0, 0.0, 0.0);
        else if (first_miss_reason == HIZ_MISS_EXCEEDED_DIST) c = vec3(1.0, 1.0, 0.0);
        else if (first_miss_reason == HIZ_MISS_EXCEEDED_ITERS) c = vec3(1.0, 0.0, 1.0);
        else if (first_miss_reason == HIZ_MISS_INVALID_PROJ) c = vec3(0.0, 1.0, 1.0);
        else c = vec3(0.0, 0.0, 1.0);
        out_SSGI = vec4(c, out_hits);
        return;
    }
    if (u_SSGIDebugView == 7) {
        out_SSGI = vec4(vec3(self_rejections / float(rays)), out_hits);
        return;
    }
    if (u_SSGIDebugView == 8) {
        out_SSGI = vec4(vec3(surface_rejections / float(rays)), out_hits);
        return;
    }

    out_SSGI = vec4(clamp(indirect_radiance, vec3(0.0), vec3(40.0)), out_hits);
}
