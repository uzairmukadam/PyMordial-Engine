#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_HDRColor;

// G-Buffer Inputs
layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;          // Reversed-Z Depth32F
layout (binding = 3) uniform sampler2D u_ShadowAtlas;

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

// Configurable quality uniforms
uniform int u_PCF_Samples;       // 4, 8, 16
uniform int u_SSCS_Enabled;      // 0 or 1
uniform int u_SSCS_Steps;        // 8 to 16
uniform float u_SSCS_Thickness;  // 0.05
uniform int u_CascadeCount;      // 1 to 4
uniform int u_GBufferDebug;      // 0=Off, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas

const float PI = 3.14159265358979323846;

// 16-tap Poisson Disk Distribution
const vec2 POISSON_16[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),  vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),  vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),  vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),  vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),   vec2(0.14383161, -0.14100790)
);

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

// Cascaded Shadow Map Evaluation (CSM)
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
    // Normal-offset bias prevents self-shadow acne on curved surfaces (spheres/characters)
    float normal_offset = 0.02 * (1.0 - cos_theta);
    vec3 biased_pos = world_pos + N * normal_offset;

    vec4 light_space_pos = u_LightViewProjection[cascade] * vec4(biased_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    // Offset in 2x2 atlas: cascade 0 = (0,0), 1 = (0.5, 0), 2 = (0, 0.5), 3 = (0.5, 0.5)
    vec2 atlas_offset = vec2(float(cascade % 2) * 0.5, float(cascade / 2) * 0.5);
    vec2 uv = proj_coords.xy * 0.5 + atlas_offset;

    float current_depth = proj_coords.z;
    // Slope-scaled depth bias based on true surface normal
    float bias = max(0.003 * (1.0 - cos_theta), 0.0008);

    float shadow = 0.0;
    float filter_radius = 0.0008;

    // Dither Poisson disc rotation using screen-space coordinates
    float angle = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);

    int samples = clamp(u_PCF_Samples, 4, 16);
    for (int i = 0; i < samples; ++i) {
        vec2 offset = rot * POISSON_16[i] * filter_radius;
        vec2 sample_uv = uv + offset;
        float pcf_depth = texture(u_ShadowAtlas, sample_uv).r;
        shadow += (current_depth - bias <= pcf_depth) ? 1.0 : 0.0;
    }
    return shadow / float(samples);
}

// Screen-Space Contact Shadows (SSCS)
float CalculateSSCS(vec3 world_pos, vec3 light_dir, float view_depth) {
    if (u_SSCS_Enabled == 0 || view_depth > 10.0) return 1.0;

    vec4 ray_origin_clip = u_ViewProjection * vec4(world_pos, 1.0);
    vec3 ray_origin = ray_origin_clip.xyz / ray_origin_clip.w;
    ray_origin = ray_origin * 0.5 + 0.5;

    vec4 ray_end_clip = u_ViewProjection * vec4(world_pos + light_dir * 0.35, 1.0);
    vec3 ray_end = ray_end_clip.xyz / ray_end_clip.w;
    ray_end = ray_end * 0.5 + 0.5;

    vec3 ray_step = (ray_end - ray_origin) / float(u_SSCS_Steps);
    vec3 curr_pos = ray_origin + ray_step;

    float shadow = 1.0;
    for (int i = 0; i < u_SSCS_Steps; ++i) {
        if (curr_pos.x < 0.0 || curr_pos.x > 1.0 || curr_pos.y < 0.0 || curr_pos.y > 1.0) break;

        float sampled_depth = texture(u_GBufferDepth, curr_pos.xy).r;
        // In Reversed-Z, closer depths are larger: sampled_depth > curr_pos.z means occluded
        if (sampled_depth > curr_pos.z && (sampled_depth - curr_pos.z) < u_SSCS_Thickness) {
            shadow = 0.15;
            break;
        }
        curr_pos += ray_step;
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

    // In Reversed-Z, background clear is 0.0
    if (raw_depth <= 0.000001) {
        if (u_GBufferDebug > 0) {
            out_HDRColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        // Render rich atmospheric sky gradient
        vec3 sky_zenith = vec3(0.08, 0.22, 0.45);
        vec3 sky_horizon = vec3(0.65, 0.75, 0.88);
        vec3 sky = mix(sky_horizon, sky_zenith, pow(v_UV.y, 1.5));

        // Add sun disc glow
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
    float roughness = albedo_rough.a;
    vec3 N = OctahedralDecode(normal_metal.rg);
    float metallic = normal_metal.b;
    float ao = normal_metal.a;

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

    // PBR Cook-Torrance BRDF Evaluation
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

    // Shadow evaluation: CSM + Screen-Space Contact Shadows
    float csm_shadow = CalculateCSMShadow(world_pos.xyz, N, L, -view_pos.z);
    float sscs_shadow = CalculateSSCS(world_pos.xyz, L, -view_pos.z);
    float shadow = min(csm_shadow, sscs_shadow);

    vec3 direct_light = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    // Hemispheric Sky & Ground Bounce Ambient Lighting
    vec3 sky_ambient = vec3(0.20, 0.28, 0.42) * (u_SunColor_Ambient.w * 3.0);
    vec3 ground_ambient = vec3(0.25, 0.28, 0.25) * (u_SunColor_Ambient.w * 2.0);
    vec3 hemisphere_light = mix(ground_ambient, sky_ambient, clamp(N.y * 0.5 + 0.5, 0.0, 1.0));

    vec3 ambient_diffuse = hemisphere_light * albedo * (vec3(1.0) - kS) * (1.0 - metallic) * ao;
    vec3 ambient_specular = hemisphere_light * F0 * ao * (1.0 - roughness * 0.5);
    vec3 ambient = ambient_diffuse + ambient_specular;

    // Volumetric Atmospheric Fog in-scattering
    float dist = length(world_pos.xyz - u_CameraPos_Time.xyz);
    float fog_factor = 1.0 - exp(-dist * u_FogColor_Density.w);
    vec3 final_color = mix(ambient + direct_light, u_FogColor_Density.rgb, fog_factor);

    out_HDRColor = vec4(final_color, 1.0);
}
