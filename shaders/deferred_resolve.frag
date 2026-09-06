#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_HDRColor;

// G-Buffer Inputs
layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;          // Reversed-Z Depth32F
layout (binding = 3) uniform sampler2D u_ShadowAtlas;

// Phase 5 Advanced Inputs
layout (binding = 4) uniform sampler2D u_AOTexture;            // GTAO / SSAO
layout (binding = 5) uniform sampler2D u_SSGITexture;          // Screen-Space Global Illumination
layout (binding = 6) uniform sampler2D u_BRDFLUT;              // Split-sum 2D BRDF LUT
layout (binding = 7) uniform sampler2D u_EnvironmentMap;       // Prefiltered HDR Environment Map
layout (binding = 8) uniform sampler2D u_SSRTexture;           // Screen-Space Reflections
layout (binding = 9) uniform sampler3D u_LPVVolume;            // 3D Light Propagation Volume

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

// SSBO 3: Dynamic Clustered Local Point Lights
struct PointLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 color_intensity; // rgb = color, w = intensity
};

layout (std430, binding = 3) buffer PointLightBuffer {
    PointLightData u_PointLights[];
};

// Configurable quality & feature uniforms
uniform int u_PCF_Samples = 24;  // 16 to 32 samples (Default: 24)
uniform int u_CascadeCount = 4;  // 1 to 4
uniform int u_GBufferDebug = 0;  // 0=Off, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas, 6=ShadowMask

// AAA Shadow Parametrization
uniform int u_ShadowMode = 2;          // 0=Hard, 1=PCF, 2=PCSS (Default: PCSS)
uniform float u_ShadowSoftness = 1.2;  // Penumbra scale & PCF radius
uniform float u_ShadowBias = 0.0015;   // Base shadow depth bias
uniform float u_ShadowNormalBias = 0.0010; // Normal offset bias scale
uniform vec2 u_ShadowOffset = vec2(0.0);   // Manual alignment offset (delta from calibrated base)

// Calibrated base light-space alignment offset
const vec2 CALIBRATED_SHADOW_OFFSET = vec2(-0.0004, 0.0011);

// Phase 5 Toggles & Settings
uniform int u_AOEnabled = 1;
uniform int u_GIEnabled = 3;     // 0=Off, 1=SSGI, 2=LPV, 3=Hybrid
uniform int u_IBLEnabled = 1;
uniform int u_SSREnabled = 1;
uniform int u_PointLightCount = 0;
uniform vec3 u_LPV_Min = vec3(-32.0, -2.0, -32.0);
uniform vec3 u_LPV_Size = vec3(64.0, 32.0, 64.0);

const float PI = 3.14159265358979323846;

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

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 0.0000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Linearize depth from Reversed-Z buffer to positive metric view-space depth in meters
float LinearizeDepth(float depth, vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 vpos = u_InvProjection * clip;
    return -vpos.z / max(vpos.w, 0.000001);
}

// Bilinear Percentage-Closer Filter (PCF) over 4 discrete shadow texels
// Anti-aliases hard shadow boundaries and provides pristine sub-texel smoothness without blur
float SampleShadowBilinear(vec2 uv, float current_depth, float bias, vec2 uv_min, vec2 uv_max) {
    vec2 full_size = vec2(textureSize(u_ShadowAtlas, 0));
    vec2 texel_size = 1.0 / full_size;
    vec2 tex_coord = uv * full_size - 0.5;
    vec2 base = floor(tex_coord);
    vec2 f = fract(tex_coord);

    vec2 uv00 = clamp((base + vec2(0.5, 0.5)) * texel_size, uv_min, uv_max);
    vec2 uv10 = clamp(uv00 + vec2(texel_size.x, 0.0), uv_min, uv_max);
    vec2 uv01 = clamp(uv00 + vec2(0.0, texel_size.y), uv_min, uv_max);
    vec2 uv11 = clamp(uv00 + texel_size, uv_min, uv_max);

    float s00 = (current_depth - bias <= texture(u_ShadowAtlas, uv00).r) ? 1.0 : 0.0;
    float s10 = (current_depth - bias <= texture(u_ShadowAtlas, uv10).r) ? 1.0 : 0.0;
    float s01 = (current_depth - bias <= texture(u_ShadowAtlas, uv01).r) ? 1.0 : 0.0;
    float s11 = (current_depth - bias <= texture(u_ShadowAtlas, uv11).r) ? 1.0 : 0.0;

    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

// Evaluates shadow occlusion for a single cascade quadrant
float SampleSingleCascade(
    int cascade,
    vec3 world_pos,
    vec3 N,
    vec3 L,
    float cos_theta,
    float slope,
    float ign_phi,
    float softness
) {
    float normal_offset = (u_ShadowNormalBias * slope) * (float(cascade) * 0.35 + 1.0);
    vec3 biased_pos = world_pos + N * normal_offset;

    vec4 light_space_pos = u_LightViewProjection[cascade] * vec4(biased_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // Remap NDC XY [-1, 1] to UV [0, 1]; Z is already in [0, 1] under GL_ZERO_TO_ONE clip control
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;
    proj_coords.xy += CALIBRATED_SHADOW_OFFSET + u_ShadowOffset;

    // Bounds check
    if (proj_coords.z < 0.0 || proj_coords.z > 1.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    vec2 atlas_offset = vec2(float(cascade % 2) * 0.5, float(cascade / 2) * 0.5);
    vec2 uv = proj_coords.xy * 0.5 + atlas_offset;
    vec2 uv_min = atlas_offset + vec2(0.0005);
    vec2 uv_max = atlas_offset + vec2(0.4995);

    float current_depth = proj_coords.z;
    float base_bias = max(u_ShadowBias, 0.0);
    float bias = base_bias * (1.0 + slope * 0.5);

    // MODE 0: ANTI-ALIASED HARD SHADOWS (4-Tap Bilinear PCF)
    if (u_ShadowMode == 0) {
        return SampleShadowBilinear(uv, current_depth, bias, uv_min, uv_max);
    }

    float cascade_scale = float(cascade) * 0.65 + 1.0;

    // MODE 1: UNIFORM PCF (Vogel Disk)
    if (u_ShadowMode == 1) {
        float filter_radius = (0.0009 * softness) / cascade_scale;
        int samples = clamp(u_PCF_Samples, 8, 32);
        float shadow = 0.0;
        for (int i = 0; i < samples; ++i) {
            vec2 offset = VogelDiskSample(i, samples, ign_phi) * filter_radius;
            vec2 sample_uv = clamp(uv + offset, uv_min, uv_max);
            shadow += SampleShadowBilinear(sample_uv, current_depth, bias, uv_min, uv_max);
        }
        return shadow / float(samples);
    }

    // MODE 2: AAA DIRECTIONAL PCSS (Contact-Hardening Soft Shadows via Vogel Disk)
    vec2 full_size = vec2(textureSize(u_ShadowAtlas, 0));
    vec2 texel_size = 1.0 / full_size;

    // Step 1: Blocker Search (Vogel Spiral)
    float search_radius = clamp(texel_size.x * 2.8 * softness, texel_size.x * 1.5, 0.0012) / cascade_scale;
    float blocker_depth_sum = 0.0;
    int blocker_count = 0;
    int blocker_samples = 16;

    for (int i = 0; i < blocker_samples; ++i) {
        vec2 offset = VogelDiskSample(i, blocker_samples, ign_phi) * search_radius;
        vec2 sample_uv = clamp(uv + offset, uv_min, uv_max);
        float d = texture(u_ShadowAtlas, sample_uv).r;
        if (d < current_depth - bias) {
            blocker_depth_sum += d;
            blocker_count++;
        }
    }

    // Fully illuminated if no blockers found in search footprint
    if (blocker_count == 0) {
        return 1.0;
    }

    // Step 2: Physical Directional Penumbra Estimation
    float avg_blocker_depth = blocker_depth_sum / float(blocker_count);
    float depth_diff = max(current_depth - avg_blocker_depth, 0.0);
    float penumbra_ratio = clamp(depth_diff * 22.0 * softness, 0.0, 1.0);

    // Immediate contact hardening: at contact points, bypass wide filtering for razor-sharp edge
    if (penumbra_ratio < 0.03) {
        return SampleShadowBilinear(uv, current_depth, bias, uv_min, uv_max);
    }

    // Step 3: Filtered PCF with contact-hardening penumbra radius
    float min_filter_radius = texel_size.x * 0.75;
    float max_filter_radius = (0.0018 * softness) / cascade_scale;
    float filter_radius = mix(min_filter_radius, max_filter_radius, penumbra_ratio);

    int filter_samples = clamp(u_PCF_Samples, 12, 32);
    float pcss_shadow = 0.0;
    for (int i = 0; i < filter_samples; ++i) {
        vec2 offset = VogelDiskSample(i, filter_samples, ign_phi) * filter_radius;
        vec2 sample_uv = clamp(uv + offset, uv_min, uv_max);
        pcss_shadow += SampleShadowBilinear(sample_uv, current_depth, bias, uv_min, uv_max);
    }
    return pcss_shadow / float(filter_samples);
}

// Cascaded Shadow Map Evaluation (CSM) supporting HARD, PCF, and PCSS with smooth split blending
float CalculateCSMShadow(vec3 world_pos, vec3 N, vec3 L, float view_depth) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) {
        return 0.0;
    }

    int cascade = 0;
    for (int i = 0; i < u_CascadeCount - 1; ++i) {
        if (view_depth > u_CascadeSplits[i]) {
            cascade = i + 1;
        }
    }

    float cos_theta = clamp(NdotL, 0.0, 1.0);
    float slope = clamp(sqrt(max(1.0 - cos_theta * cos_theta, 0.0)) / max(cos_theta, 0.001), 0.0, 3.5);

    // Interleaved Gradient Noise rotation for smooth blue-noise spatial distribution
    float ign_phi = InterleavedGradientNoise(gl_FragCoord.xy) * 6.2831853;
    float softness = clamp(u_ShadowSoftness, 0.1, 4.0);

    float shadow = SampleSingleCascade(cascade, world_pos, N, L, cos_theta, slope, ign_phi, softness);

    // Smooth cascade split transition blending to eliminate seams
    if (cascade < u_CascadeCount - 1) {
        float split_val = u_CascadeSplits[cascade];
        float blend_range = split_val * 0.15;
        float blend_start = split_val - blend_range;
        if (view_depth > blend_start) {
            float blend_t = clamp((view_depth - blend_start) / blend_range, 0.0, 1.0);
            float next_shadow = SampleSingleCascade(cascade + 1, world_pos, N, L, cos_theta, slope, ign_phi, softness);
            shadow = mix(shadow, next_shadow, blend_t);
        }
    }

    return shadow;
}

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
    }

    // Sky gradient for background pixels
    if (raw_depth <= 0.000001) {
        if (u_GBufferDebug > 0) {
            out_HDRColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        vec3 sky_zenith = vec3(0.08, 0.22, 0.45);
        vec3 sky_horizon = vec3(0.65, 0.75, 0.88);
        vec3 sky = mix(sky_horizon, sky_zenith, pow(v_UV.y, 1.5));

        vec3 ray_ndc = vec3(v_UV * 2.0 - 1.0, 1.0);
        vec4 ray_view = u_InvProjection * vec4(ray_ndc, 1.0);
        ray_view.z = -1.0;
        ray_view.w = 0.0;
        vec3 ray_dir = normalize((u_InvView * ray_view).xyz);
        float sun_dot = max(dot(ray_dir, -u_SunDirection_Intensity.xyz), 0.0);
        sky += vec3(1.0, 0.9, 0.7) * pow(sun_dot, 256.0) * 8.0;

        out_HDRColor = vec4(sky, 1.0);
        return;
    }

    // Reconstruct world position from Reversed-Z depth
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= view_pos.w;
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    // Read G-Buffer parameters
    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    vec4 normal_metal = texture(u_GBufferNormalMetallic, v_UV);

    vec3 albedo = albedo_rough.rgb;
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

    float shadow = CalculateCSMShadow(world_pos.xyz, N, L, -view_pos.z);

    if (u_GBufferDebug == 6 || u_GBufferDebug == 7 || u_GBufferDebug == 8) {
        out_HDRColor = vec4(vec3(shadow), 1.0);
        return;
    }

    vec3 direct_sun = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

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
                point_lights_accum += (pl_diff + pl_spec) * pl_rad * pl_NdotL * atten;
            }
        }
    }

    // Global Illumination (SSGI + LPV)
    vec3 indirect_diffuse = vec3(0.0);

    // 1. LPV (Volumetric 3D Indirect Diffuse Bounce)
    if (u_GIEnabled == 2 || u_GIEnabled == 3) {
        vec3 lpv_uvw = clamp((world_pos.xyz - u_LPV_Min) / u_LPV_Size, vec3(0.0), vec3(1.0));
        vec4 lpv_sample = texture(u_LPVVolume, lpv_uvw);
        indirect_diffuse += lpv_sample.rgb * albedo * (1.0 - metallic);
    }

    // 2. SSGI (Screen-Space Near-Field Indirect Diffuse Bounce & Contact Color Bleed)
    if (u_GIEnabled == 1 || u_GIEnabled == 3) {
        vec4 ssgi_sample = texture(u_SSGITexture, v_UV);
        indirect_diffuse += ssgi_sample.rgb * albedo * (1.0 - metallic);
    }

    // Ambient Sky / Ground Foundation
    vec3 sky_ambient = vec3(0.18, 0.24, 0.38) * (u_SunColor_Ambient.w * 2.5);
    vec3 ground_ambient = vec3(0.22, 0.24, 0.22) * (u_SunColor_Ambient.w * 1.8);
    vec3 hemisphere_light = mix(ground_ambient, sky_ambient, clamp(N.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 ambient_base = hemisphere_light * albedo * (vec3(1.0) - F0) * (1.0 - metallic);

    // Image-Based Lighting (IBL) & Screen-Space Reflections (SSR)
    vec3 indirect_specular = vec3(0.0);

    if (u_IBLEnabled == 1) {
        vec3 R = reflect(-V, N);
        float NdotV = max(dot(N, V), 0.0);

        // Sample Split-Sum BRDF LUT
        vec2 brdf = texture(u_BRDFLUT, vec2(NdotV, roughness)).rg;

        // Sample Prefiltered Environment Map
        vec2 env_uv = vec2(atan(R.z, R.x) / (2.0 * PI) + 0.5, asin(clamp(R.y, -0.999, 0.999)) / PI + 0.5);
        vec3 env_radiance = textureLod(u_EnvironmentMap, env_uv, roughness * 5.0).rgb;

        // Sample SSR
        if (u_SSREnabled == 1) {
            vec4 ssr_sample = texture(u_SSRTexture, v_UV);
            env_radiance = mix(env_radiance, ssr_sample.rgb, ssr_sample.a);
        }

        vec3 F_env = FresnelSchlickRoughness(NdotV, F0, roughness);
        indirect_specular = env_radiance * (F_env * brdf.x + brdf.y);
    }

    // Total combine with Ambient Occlusion (SSCS applied directly to direct sun, not doubled here)
    vec3 ambient = (ambient_base + indirect_diffuse + indirect_specular) * ao;
    vec3 total_lit = direct_sun + point_lights_accum + ambient;

    // Atmospheric Fog
    float dist = length(world_pos.xyz - u_CameraPos_Time.xyz);
    float fog_factor = 1.0 - exp(-dist * u_FogColor_Density.w);
    vec3 final_color = mix(total_lit, u_FogColor_Density.rgb, fog_factor);

    out_HDRColor = vec4(final_color, 1.0);
}
