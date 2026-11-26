#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_HDRColor;

// G-Buffer Inputs
layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;          // Reversed-Z Depth32F
layout (binding = 3) uniform sampler2D u_ShadowAtlas;
// Hardware depth-compare view of the same atlas. A tap through this sampler
// returns a bilinear-filtered 2x2 PCF result, so one fetch does the work of four
// manual fetches. The raw sampler above stays bound for the blocker search and
// the atlas debug view, which need the stored depth rather than a comparison.
layout (binding = 9) uniform sampler2DShadow u_ShadowAtlasCmp;

// Phase 5 Advanced Inputs
layout (binding = 4) uniform sampler2D u_AOTexture;            // GTAO / SSAO
layout (binding = 5) uniform sampler2D u_SSGITexture;          // Screen-Space Global Illumination
layout (binding = 6) uniform sampler2D u_BRDFLUT;              // Split-sum 2D BRDF LUT
layout (binding = 7) uniform sampler2D u_EnvironmentMap;       // Prefiltered HDR Environment Map
layout (binding = 8) uniform sampler2D u_SSRTexture;           // Screen-Space Reflections
layout (binding = 12) uniform sampler2D u_SpotShadowAtlas;     // 2x2 Spot Light Perspective Shadow Depth Atlas
// Hardware depth-compare view of the spot atlas (see u_ShadowAtlasCmp).
layout (binding = 16) uniform sampler2DShadow u_SpotShadowAtlasCmp;
layout (binding = 13) uniform sampler2D u_IrradianceMap;       // Diffuse environment irradiance
layout (binding = 15) uniform sampler2D u_RMGITexture;           // Ray-Marched Global Illumination (RMGI)
// Adaptive Multi-Range Global Illumination (AMRGI). Bound to unit 19.
layout (binding = 19) uniform sampler2D u_AMRGITexture;
// AMRGI diagnostics: raw trace radiance (half-res) and aux (hit dist/range/conf).
layout (binding = 20) uniform sampler2D u_AMRGIRawTexture;
layout (binding = 21) uniform sampler2D u_AMRGIAuxTexture;
// Shared local-diffuse visibility (Stage 3), trace-res R16F. Bound to unit 22.
layout (binding = 22) uniform sampler2D u_SSRTVIsibilityTexture;
// Current-frame direct diffuse + emissive radiance (GI source). Bound to unit 18.
layout (binding = 18) uniform sampler2D u_DirectRadianceTexture;

// Unified Frame Context UBO 0
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

// SSBO 3: Dynamic Clustered Local Point & Spot Lights
struct PointLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 color_intensity; // rgb = color, w = intensity
};

struct SpotLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 dir_inner_cos;   // xyz = normalized direction, w = cos(inner_cone_angle)
    vec4 color_intensity; // rgb = color, w = intensity
    vec4 params;          // x = cos(outer_cone_angle), y = shadow_index (-1 if no shadow), z = shadow_bias, w = reserved
};

layout (std430, binding = 3) buffer LightBuffer {
    PointLightData u_PointLights[256];
    SpotLightData u_SpotLights[];
};

// Configurable quality & feature uniforms
uniform int u_CascadeCount = 4;  // 1 to 4
uniform int u_GBufferDebug = 0;  // 0=Off, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas, 6=ShadowMask, 8=SSGI, 10=IndirectDiffuse, 12=SpotLights, 13=RMGI, 14=CascadeIndex, 15=CascadeCoverage

// Hard-shadow parametrization
uniform float u_ShadowBias = 0.0015;   // Base world-space shadow depth bias (meters)
uniform float u_ShadowNormalBias = 0.0010; // Normal offset bias scale
uniform float u_ShadowBiasSlopeScale = 2.0; // Slope multiplier for the base bias
uniform float u_ShadowBiasMinScale = 0.5;   // Lower clamp, as a multiple of the base bias
uniform float u_ShadowBiasMaxScale = 6.0;   // Upper clamp, as a multiple of the base bias

// Shadow filtering tiers. Each tier is limited to a prefix of the cascades:
// the far cascades keep the cheaper path because their texels already span
// several screen pixels, so a wider filter there costs taps without adding
// visible detail.
uniform int u_ShadowFilterMode = 0;   // 0 = HARD, 1 = PCF, 2 = PCSS
uniform int u_PCFCascadeLimit = 3;    // cascades with index < this use PCF
uniform int u_PCFSamples = 16;        // Vogel disk taps (4 to 64)
uniform float u_PCFRadius = 1.5;      // filter radius in cascade texels
uniform float u_PCFFalloff = 2.0;     // radial weight exponent (1 = linear cone, 2 = smooth bell)

// PCSS (percentage-closer soft shadows). The penumbra is derived from the
// receiver-to-blocker gap rather than a fixed radius, so contact shadows stay
// sharp and only elevated casters soften.
uniform int u_PCSS_CascadeLimit = 1;      // cascades with index < this use PCSS
uniform int u_PCSSSearchSamples = 16;     // blocker search taps (4 to 64)
uniform float u_PCSSSearchRadius = 48.0;  // blocker search radius in cascade texels
uniform float u_PCSSLightSize = 0.10;     // light-source size factor (penumbra scale)
uniform int u_PCSSSamples = 24;           // penumbra filter taps (4 to 64)
uniform float u_PCSSFalloff = 2.0;        // radial weight exponent
uniform float u_PCSSMaxRadius = 96.0;     // penumbra radius clamp in cascade texels
// 0.0 = average blocker gap (classic PCSS), 1.0 = closest blocker gap.
// The closest blocker keeps overlapping shadows sharp instead of letting a far
// caster's softness bleed onto a near caster's edge.
uniform float u_PCSSClosestBlockerWeight = 1.0;
// Physical penumbra scale: tan of the sun's angular radius (0.265 deg for the
// real sun). When > 0 the penumbra is derived from the light's solid angle
// instead of the authored ``u_PCSSLightSize`` factor, which is what produces
// true contact hardening: a caster touching the receiver has a zero gap and
// therefore a zero penumbra, while a caster 2 m up softens by 2 * tan(0.265deg).
uniform float u_SunAngularRadius = 0.0;
// Per-cascade atlas rect: xy = atlas UV offset, zw = atlas UV scale.
uniform vec4 u_CascadeAtlasRect[4] = vec4[4](
    vec4(0.0, 0.0, 0.5, 0.5),
    vec4(0.5, 0.0, 0.5, 0.5),
    vec4(0.0, 0.5, 0.5, 0.5),
    vec4(0.5, 0.5, 0.5, 0.5)
);

// Phase 5 Toggles & Settings
uniform int u_AOEnabled = 1;
uniform int u_GIEnabled = 4;     // 0=Off/IBL, 1=SSGI, 4=Ray-Marched Global Illumination (RMGI)
uniform int u_IBLEnabled = 1;
uniform int u_SSREnabled = 1;
uniform int u_EnergyCompensationEnabled = 1;
uniform int u_PointLightCount = 0;
uniform int u_SpotLightCount = 0;
uniform int u_SpotShadowEnabled = 1;
uniform mat4 u_SpotLightViewProjection[4];

#include "shaders/atmosphere.glsl"

#include "shaders/common/shadows.glsl"
#include "shaders/common/pbr_brdf.glsl"
#include "shaders/common/depth.glsl"

// Jorge Jimenez's Interleaved Gradient Noise (IGN) for uniform blue-noise spatial distribution
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

// Vogel Disk Sampling with Golden Angle Spiral for low-discrepancy circular filtering
vec2 VogelDiskSample(int sample_index, int sample_count, float phi) {
    float r = sqrt((float(sample_index) + 0.5) / float(sample_count));
    float theta = float(sample_index) * 2.39996323 + phi; // 2.39996323 rad = 137.507764 deg (Golden Angle)
    return vec2(r * cos(theta), r * sin(theta));
}


// Evaluates localized perspective spot light shadow occlusion.
//
// Uses the same hardware comparison sampler as the directional path, so a tap
// resolves a bilinear 2x2 PCF footprint. The reference depth is offset by the
// receiver-plane gradient rather than a single constant, which keeps the bias
// small enough that contact shadows stay welded to their casters.
float SampleSpotShadow(int shadow_idx, vec3 world_pos, vec3 N, float bias) {
    if (shadow_idx < 0 || shadow_idx >= 4 || u_SpotShadowEnabled == 0) return 1.0;

    vec4 light_space_pos = u_SpotLightViewProjection[shadow_idx] * vec4(world_pos + N * (bias * 0.5), 1.0);
    if (light_space_pos.w <= 0.0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // Remap NDC XY [-1, 1] to UV [0, 1]; Z is already in [0, 1] under GL_ZERO_TO_ONE clip control
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    // Out-of-bounds check (outside spot cone/frustum)
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

    // 9-tap Vogel disk through the comparison sampler. A single bilinear tap
    // already resolves 4 texels, so this covers a 6x6 texel footprint for the
    // cost of nine fetches, which is enough to hide the perspective aliasing a
    // spot light shows at grazing angles.
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

// OctahedralDecode, DistributionGGX, GeometrySchlickGGX, GeometrySmith,
// FresnelSchlick and FresnelSchlickRoughness come from common/pbr_brdf.glsl.

// Jimenez Multi-Bounce Approximation for rich indirect saturation inside crevices
vec3 MultiBounceAO(float visibility, vec3 albedo) {
    vec3 a = 2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c = 2.7552 * albedo + 0.6903;
    vec3 x = vec3(visibility);
    return clamp(max(x, ((x * a + b) * x + c) * x), 0.0, 1.0);
}

// Linearize depth from Reversed-Z buffer to positive metric view-space depth in meters
float LinearizeDepth(float depth, vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 vpos = u_InvProjection * clip;
    return -vpos.z / max(vpos.w, 0.000001);
}


#include "shaders/common/csm_sampling.glsl"

void main() {
    if (u_GBufferDebug == 5) {
        float s = texture(u_ShadowAtlas, v_UV).r;
        out_HDRColor = vec4(vec3(s), 1.0);
        return;
    }

    float raw_depth = texture(u_GBufferDepth, v_UV).r;

    if (u_GBufferDebug == 4) {
        out_HDRColor = vec4(vec3(raw_depth), 1.0);
        return;
    } else if (u_GBufferDebug == 23) {
        // Current-frame direct radiance (GI source) debug view.
        out_HDRColor = vec4(texture(u_DirectRadianceTexture, v_UV).rgb, 1.0);
        return;
    }

    // Physical Atmosphere & Celestial Sky Dome for background pixels
    if (raw_depth <= 0.000001) {
        if (u_GBufferDebug > 0) {
            out_HDRColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        vec3 ray_ndc = vec3(v_UV * 2.0 - 1.0, 1.0);
        vec4 ray_view = u_InvProjection * vec4(ray_ndc, 1.0);
        ray_view.z = -1.0;
        ray_view.w = 0.0;
        vec3 ray_dir = normalize((u_InvView * ray_view).xyz);

        vec3 sky = EvaluateAtmosphere(
            ray_dir,
            u_AtmosphereSunDir,
            u_SunColor_Ambient.rgb,
            u_SunDirection_Intensity.w,
            u_CameraPos_Time.w,
            u_CameraPos_Time.xyz
        );

        out_HDRColor = vec4(sky, 1.0);
        return;
    }

    // Reconstruct world position from Reversed-Z depth
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= max(abs(view_pos.w), 1e-6);
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    // Read G-Buffer parameters
    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    vec4 normal_metal = texture(u_GBufferNormalMetallic, v_UV);

    vec3 raw_albedo = albedo_rough.rgb;
    // Phase 8: Extract HDR emissive radiance (> 1.0) and normalize diffuse albedo
    float max_c = max(max(raw_albedo.r, raw_albedo.g), raw_albedo.b);
    vec3 albedo = (max_c > 1.0) ? (raw_albedo / max_c) : raw_albedo;
    vec3 emissive = (max_c > 1.0) ? (raw_albedo - albedo) : vec3(0.0);
    float roughness = clamp(albedo_rough.a, 0.04, 1.0);
    vec3 N = OctahedralDecode(normal_metal.rg);
    float metallic = normal_metal.b;
    float ao = normal_metal.a;

    // Multiply by screen-space GTAO/SSAO if enabled
    if (u_AOEnabled == 1) {
        float gtao = texture(u_AOTexture, v_UV).r;
        ao *= gtao;
    }

    if (u_GBufferDebug == 1) {
        out_HDRColor = vec4(albedo, 1.0);
        return;
    } else if (u_GBufferDebug == 2) {
        out_HDRColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    } else if (u_GBufferDebug == 3) {
        out_HDRColor = vec4(roughness, metallic, ao, 1.0);
        return;
    }

    vec3 V = normalize(u_CameraPos_Time.xyz - world_pos.xyz);
    vec3 L = normalize(-u_SunDirection_Intensity.xyz);
    vec3 H = normalize(V + L);

    // PBR Cook-Torrance Direct Sun Lighting
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    float NdotL = max(dot(N, L), 0.0);
    vec3 radiance = u_SunColor_Ambient.rgb * u_SunDirection_Intensity.w;

    int shadow_cascade = 0;
    float shadow_coverage = 0.0;
    float shadow = CalculateCSMShadow(
        world_pos.xyz, N, L, -view_pos.z, shadow_cascade, shadow_coverage
    );

    if (u_GBufferDebug == 6) {
        out_HDRColor = vec4(vec3(shadow), 1.0);
        return;
    } else if (u_GBufferDebug == 7) {
        out_HDRColor = vec4(vec3(ao), 1.0);
        return;
    } else if (u_GBufferDebug == 14) {
        // Selected cascade index as a false-colour image.
        vec3 cascade_tint = vec3(0.0);
        if (shadow_cascade == 0) {
            cascade_tint = vec3(0.15, 0.45, 1.00);
        } else if (shadow_cascade == 1) {
            cascade_tint = vec3(0.20, 0.95, 0.35);
        } else if (shadow_cascade == 2) {
            cascade_tint = vec3(1.00, 0.85, 0.15);
        } else {
            cascade_tint = vec3(1.00, 0.25, 0.20);
        }
        out_HDRColor = vec4(cascade_tint, 1.0);
        return;
    } else if (u_GBufferDebug == 15) {
        // Per-cascade coverage: brightness falls off toward the rect border,
        // so the fade band and any uncovered receivers are directly visible.
        out_HDRColor = vec4(vec3(shadow_coverage), 1.0);
        return;
    } else if (u_GBufferDebug == 16) {
        // Cascade bounds: each cascade's slice is tinted, and the split
        // distances are drawn as bright boundary lines so the start and end of
        // every cascade is directly readable in world space.
        float view_depth = -view_pos.z;
        vec3 band = vec3(0.0);
        if (shadow_cascade == 0) {
            band = vec3(0.15, 0.45, 1.00);
        } else if (shadow_cascade == 1) {
            band = vec3(0.20, 0.95, 0.35);
        } else if (shadow_cascade == 2) {
            band = vec3(1.00, 0.85, 0.15);
        } else {
            band = vec3(1.00, 0.25, 0.20);
        }

        // Beyond the last cascade there is no shadow coverage at all.
        float max_shadow_dist = u_CascadeSplits[u_CascadeCount - 1];
        if (view_depth >= max_shadow_dist) {
            band = vec3(0.10, 0.10, 0.12);
        }

        // Boundary lines. The width is derived from the per-pixel depth
        // gradient (fwidth) so the line stays a constant number of screen
        // pixels wide at every distance. A world-space width would shrink to
        // sub-pixel near the horizon and vanish exactly where the far cascades
        // meet.
        float depth_per_pixel = max(fwidth(view_depth), 1e-5);
        float line_half_width = depth_per_pixel * 2.0;
        float edge = 0.0;
        for (int i = 0; i < u_CascadeCount; ++i) {
            float d = abs(view_depth - u_CascadeSplits[i]);
            edge = max(edge, 1.0 - smoothstep(line_half_width * 0.5, line_half_width, d));
        }
        // Near-plane start of cascade 0.
        edge = max(edge, 1.0 - smoothstep(line_half_width * 0.5, line_half_width, view_depth));

        // The debug view still passes through tonemapping, which compresses a
        // plain 1.0 to mid grey. Push the boundary well into HDR so it resolves
        // as a crisp white line instead of a faint tint.
        vec3 result = mix(band, vec3(12.0), edge);
        out_HDRColor = vec4(result, 1.0);
        return;
    }

    // Subtle direct contact crevice micro-shadowing
    float direct_crevice = clamp(ao + max(dot(N, L), 0.0), 0.0, 1.0);

    // Multi-scatter energy compensation. Single-scatter GGX drops the light that
    // would bounce between microfacets, so rough metals lose 20-40% of their
    // reflected energy. The BRDF LUT's blue channel carries the directional
    // albedo E; the compensation factor restores the missing energy.
    vec3 energy_comp = vec3(1.0);
    if (u_EnergyCompensationEnabled == 1) {
        float E = texture(u_BRDFLUT, vec2(max(dot(N, V), 0.001), roughness)).b;
        energy_comp = GGXEnergyCompensation(F0, roughness, E);
    }

    vec3 direct_sun = (kD * albedo / PI + specular * energy_comp) * radiance * NdotL * shadow * direct_crevice;

    // Dynamic Clustered Point Lights (SSBO 3)
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
                // Smooth physical inverse-square attenuation with windowing
                float ratio = pl_dist / pl_radius;
                float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                float atten = (win * win) / (pl_dist * pl_dist + 1.0);

                vec3 pl_H = normalize(V + pl_L);
                float pl_NDF = DistributionGGX(N, pl_H, roughness);
                float pl_G = GeometrySmith(N, V, pl_L, roughness);
                vec3 pl_F = FresnelSchlick(max(dot(pl_H, V), 0.0), F0);

                vec3 pl_spec = (pl_NDF * pl_G * pl_F) / (4.0 * max(dot(N, V), 0.0) * pl_NdotL + 0.0001);
                vec3 pl_diff = (vec3(1.0) - pl_F) * (1.0 - metallic) * albedo / PI;

                vec3 pl_rad = u_PointLights[i].color_intensity.rgb * u_PointLights[i].color_intensity.w;
                point_lights_accum += (pl_diff + pl_spec * energy_comp) * pl_rad * pl_NdotL * atten;
            }
        }
    }

    // Dynamic Clustered Spot Lights (SSBO 3 + Perspective Spot Shadows)
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

            // Spot cone angle check: dot product between vector from light to fragment (-sp_L) and spot direction
            float cos_theta = dot(-sp_L, sp_dir);
            if (cos_theta > outer_cos) {
                float spot_cone = smoothstep(outer_cos, inner_cos, cos_theta);
                float sp_NdotL = max(dot(N, sp_L), 0.0);

                if (sp_NdotL > 0.0) {
                    // Smooth physical inverse-square distance attenuation with windowing
                    float ratio = sp_dist / sp_radius;
                    float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                    float atten = (win * win) / (sp_dist * sp_dist + 1.0);

                    // Spot shadow factor
                    int shadow_idx = int(u_SpotLights[i].params.y);
                    float sp_bias = u_SpotLights[i].params.z;
                    float sp_shadow = (shadow_idx >= 0) ? SampleSpotShadow(shadow_idx, world_pos.xyz, N, sp_bias) : 1.0;

                    vec3 sp_H = normalize(V + sp_L);
                    float sp_NDF = DistributionGGX(N, sp_H, roughness);
                    float sp_G = GeometrySmith(N, V, sp_L, roughness);
                    vec3 sp_F = FresnelSchlick(max(dot(sp_H, V), 0.0), F0);

                    vec3 sp_spec = (sp_NDF * sp_G * sp_F) / (4.0 * max(dot(N, V), 0.0) * sp_NdotL + 0.0001);
                    vec3 sp_diff = (vec3(1.0) - sp_F) * (1.0 - metallic) * albedo / PI;

                    vec3 sp_rad = u_SpotLights[i].color_intensity.rgb * u_SpotLights[i].color_intensity.w;
                    spot_lights_accum += (sp_diff + sp_spec * energy_comp) * sp_rad * sp_NdotL * atten * spot_cone * sp_shadow;
                }
            }
        }
    }

    if (u_GBufferDebug == 12) {
        out_HDRColor = vec4(spot_lights_accum, 1.0);
        return;
    }

    // Global Illumination (SSGI vs RMGI)
    vec3 indirect_diffuse = vec3(0.0);
    vec4 ssgi_sample = vec4(0.0);
    vec4 rmgi_sample = vec4(0.0);
    vec4 amrgi_sample = vec4(0.0);

    if (u_GIEnabled == 5) {
        // Adaptive Multi-Range Global Illumination (AMRGI)
        amrgi_sample = texture(u_AMRGITexture, v_UV);
        if (isnan(amrgi_sample.r) || isnan(amrgi_sample.g) || isnan(amrgi_sample.b) ||
            isinf(amrgi_sample.r) || isinf(amrgi_sample.g) || isinf(amrgi_sample.b)) {
            amrgi_sample = vec4(0.0);
        }
        // AMRGI raw is a cosine-weighted mean of incoming radiance Li, i.e.
        // irradiance/PI. Cosine-weighted sampling already cancels the cos/pdf
        // term, so the receiver's Lambertian response is just albedo (NOT
        // albedo/PI — that would apply 1/PI twice). Convention B: raw stores
        // pre-BRDF incoming radiance; the receiver BRDF is applied here once.
        vec3 amrgi_diffuse = clamp(amrgi_sample.rgb, vec3(0.0), vec3(30.0)) * albedo * (1.0 - metallic);
        indirect_diffuse += amrgi_diffuse;
    } else if (u_GIEnabled == 4) {
        // Ray-Marched Global Illumination (RMGI)
        rmgi_sample = texture(u_RMGITexture, v_UV);
        if (isnan(rmgi_sample.r) || isnan(rmgi_sample.g) || isnan(rmgi_sample.b) ||
            isinf(rmgi_sample.r) || isinf(rmgi_sample.g) || isinf(rmgi_sample.b)) {
            rmgi_sample = vec4(0.0);
        }
        // RMGI publishes incident diffuse irradiance, so the
        // receiver's Lambertian BRDF (albedo / PI) is applied exactly once here.
        vec3 rmgi_diffuse = clamp(rmgi_sample.rgb, vec3(0.0), vec3(30.0)) * (albedo / PI) * (1.0 - metallic);
        indirect_diffuse += rmgi_diffuse;
    } else if (u_GIEnabled == 1 || u_GIEnabled == 2 || u_GIEnabled == 3) {
        // SSGI (Screen-Space Near-Field Indirect Diffuse Bounce & Contact Color Bleed)
        ssgi_sample = texture(u_SSGITexture, v_UV);
        if (isnan(ssgi_sample.r) || isnan(ssgi_sample.g) || isnan(ssgi_sample.b) ||
            isinf(ssgi_sample.r) || isinf(ssgi_sample.g) || isinf(ssgi_sample.b) ||
            isnan(ssgi_sample.a) || isinf(ssgi_sample.a)) {
            ssgi_sample = vec4(0.0);
        }
        vec3 ssgi_diffuse = clamp(ssgi_sample.rgb, vec3(0.0), vec3(30.0)) * albedo * (1.0 - metallic);
        indirect_diffuse += ssgi_diffuse;
    }

    if (u_GBufferDebug == 8) {
        out_HDRColor = vec4(ssgi_sample.rgb, 1.0);
        return;
    } else if (u_GBufferDebug == 10) {
        out_HDRColor = vec4(indirect_diffuse, 1.0);
        return;
    } else if (u_GBufferDebug == 13) {
        out_HDRColor = vec4(rmgi_sample.rgb, 1.0);
        return;
    } else if (u_GBufferDebug == 24) {
        out_HDRColor = vec4(amrgi_sample.rgb, 1.0);
        return;
    } else if (u_GBufferDebug == 25) {
        // AMRGI raw trace radiance (before temporal/spatial reconstruction).
        out_HDRColor = vec4(texture(u_AMRGIRawTexture, v_UV).rgb, 1.0);
        return;
    } else if (u_GBufferDebug == 26) {
        // AMRGI confidence (grayscale).
        float c = texture(u_AMRGIRawTexture, v_UV).a;
        out_HDRColor = vec4(vec3(c), 1.0);
        return;
    } else if (u_GBufferDebug == 27) {
        // AMRGI hit range: black=miss, red=near, green=mid, blue=far.
        float rc = texture(u_AMRGIAuxTexture, v_UV).g;
        vec3 col = vec3(0.0);
        if (rc > 0.5 && rc < 1.5) col = vec3(1.0, 0.0, 0.0);
        else if (rc >= 1.5 && rc < 2.5) col = vec3(0.0, 1.0, 0.0);
        else if (rc >= 2.5) col = vec3(0.0, 0.0, 1.0);
        out_HDRColor = vec4(col, 1.0);
        return;
    } else if (u_GBufferDebug == 28) {
        // AMRGI hit distance (normalized by 40 m).
        float d = texture(u_AMRGIAuxTexture, v_UV).r;
        out_HDRColor = vec4(vec3(clamp(d / 40.0, 0.0, 1.0)), 1.0);
        return;
    } else if (u_GBufferDebug == 29) {
        // AMRGI validity: green=valid hit, blue=miss, red=background.
        float rc = texture(u_AMRGIAuxTexture, v_UV).g;
        float c = texture(u_AMRGIRawTexture, v_UV).a;
        vec3 col = vec3(0.0);
        if (rc > 0.5) col = vec3(0.0, 1.0, 0.0);
        else if (c > 0.0) col = vec3(0.0, 0.0, 1.0);
        else col = vec3(0.3, 0.0, 0.0);
        out_HDRColor = vec4(col, 1.0);
        return;
    } else if (u_GBufferDebug == 30) {
        // Shared local-diffuse visibility (raw, trace-res): white=open, black=occluded.
        float v = texture(u_SSRTVIsibilityTexture, v_UV).r;
        out_HDRColor = vec4(vec3(v), 1.0);
        return;
    } else if (u_GBufferDebug == 31) {
        // Final SSRT ambient occlusion (ao_texture): white=open, black=occluded.
        float v = texture(u_AOTexture, v_UV).r;
        out_HDRColor = vec4(vec3(v), 1.0);
        return;
    }
    float sky_intensity = max(u_SunColor_Ambient.w * 3.5, 0.12);
    vec3 sky_color = mix(vec3(0.25, 0.35, 0.50), u_SunColor_Ambient.rgb * 0.5 + vec3(0.15), 0.4);
    vec3 ground_color = vec3(0.18, 0.19, 0.18);
    vec3 hemisphere_light = mix(ground_color, sky_color, clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * sky_intensity;
    vec3 diffuse_energy = (vec3(1.0) - FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness))
                        * (1.0 - metallic);
    vec2 irradiance_uv = vec2(atan(N.z, N.x) / (2.0 * PI) + 0.5,
                              asin(clamp(N.y, -0.999, 0.999)) / PI + 0.5);
    vec3 diffuse_irradiance = u_IBLEnabled == 1
        ? texture(u_IrradianceMap, irradiance_uv).rgb
        : hemisphere_light;
    vec3 ambient_base = diffuse_irradiance * albedo * diffuse_energy;

    // Image-Based Lighting (IBL) & Screen-Space Reflections (SSR)
    vec3 indirect_specular = vec3(0.0);

    if (u_IBLEnabled == 1) {
        vec3 R = reflect(-V, N);
        float NdotV = max(dot(N, V), 0.0);

        // Sample Split-Sum BRDF LUT. `rg` is the split-sum term, `b` is the
        // directional albedo used by the multi-scatter energy compensation.
        vec3 brdf = texture(u_BRDFLUT, vec2(NdotV, roughness)).rgb;

        // Sample Prefiltered Environment Map
        vec2 env_uv = vec2(atan(R.z, R.x) / (2.0 * PI) + 0.5, asin(clamp(R.y, -0.999, 0.999)) / PI + 0.5);
        float max_env_lod = float(max(textureQueryLevels(u_EnvironmentMap) - 1, 0));
        vec3 env_radiance = textureLod(u_EnvironmentMap, env_uv, roughness * max_env_lod).rgb;

        // Sample SSR
        if (u_SSREnabled == 1) {
            vec4 ssr_sample = texture(u_SSRTexture, v_UV);
            if (isnan(ssr_sample.r) || isnan(ssr_sample.g) || isnan(ssr_sample.b) ||
                isinf(ssr_sample.r) || isinf(ssr_sample.g) || isinf(ssr_sample.b) ||
                isnan(ssr_sample.a) || isinf(ssr_sample.a)) {
                ssr_sample = vec4(0.0);
            }
            vec3 safe_ssr_rgb = clamp(ssr_sample.rgb, vec3(0.0), vec3(30.0));
            float safe_ssr_a = clamp(ssr_sample.a, 0.0, 1.0);
            if (safe_ssr_a > 0.001) {
                // Blend SSR with environment IBL, ensuring local reflections never create a dark void below ambient sky fill
                vec3 blended_refl = mix(env_radiance, safe_ssr_rgb, safe_ssr_a);
                env_radiance = max(blended_refl, env_radiance * (1.0 - safe_ssr_a * 0.70));
            }
        }

        // Specular occlusion (Lagarde / Frostbite) prevents specular light leaks into deep crevices
        float spec_ao = clamp(ao * ao + sqrt(NdotV) * (1.0 - ao), 0.0, 1.0);

        vec3 F_env = FresnelSchlickRoughness(NdotV, F0, roughness);
        indirect_specular = env_radiance * (F_env * brdf.x + brdf.y) * spec_ao * energy_comp;
    }

    // Total combine with Multi-Bounce color-preserving Ambient Occlusion on diffuse
    vec3 bounce_ao = MultiBounceAO(ao, albedo);
    vec3 ambient = (ambient_base + indirect_diffuse) * bounce_ao + indirect_specular;
    vec3 total_lit = direct_sun + point_lights_accum + spot_lights_accum + ambient + emissive;

    // Guaranteed NaN / Inf fallback to ambient floor
    if (isnan(total_lit.r) || isnan(total_lit.g) || isnan(total_lit.b) ||
        isinf(total_lit.r) || isinf(total_lit.g) || isinf(total_lit.b)) {
        total_lit = max(ambient_base, vec3(0.05));
    }
    // Sanity clamp only. The previous 100.0 ceiling clipped real HDR headroom
    // (sun disc, speculars, emissives) before bloom and tone-mapping could use
    // it. 1e4 still catches runaway values without touching legitimate radiance.
    total_lit = clamp(total_lit, vec3(0.0), vec3(1e4));

    out_HDRColor = vec4(total_lit, 1.0);
}
