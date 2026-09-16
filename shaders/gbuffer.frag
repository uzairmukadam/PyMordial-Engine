#version 450 core

// Multiple Render Targets (MRT)
layout (location = 0) out vec4 out_AlbedoRoughness; // RT0: Albedo (RGB) + Roughness (A)
layout (location = 1) out vec4 out_NormalMetallic;   // RT1: Octahedral Normal (RG) + Metallic (B) + AO (A)
layout (location = 2) out vec2 out_Velocity;         // RT2: Velocity Vectors (RG16F)
layout (location = 3) out vec4 out_DispInfo;         // RT3: Displacement Info (RG=Mesh UV, B=TexLayer, A=DispMode)

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_UV;
in vec4 v_CurrClip;
in vec4 v_PrevClip;
in flat uint v_EntityID;

// Phase 6: TBN Matrix and Tangent-Space Directions
in mat3 v_TBN;
in vec3 v_TangentViewDir;
in vec3 v_TangentSunDir;

// Common Frame Data UBO (binding 0)
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

// SSBO 2: Material Data (two vec4 per entity)
layout (std430, binding = 2) readonly buffer MaterialBuffer {
    vec4 u_MaterialData[];
};

// Phase 6: PBR Texture Array Atlases (sampler2DArray)
layout (binding = 10) uniform sampler2DArray u_DiffuseArray;
layout (binding = 11) uniform sampler2DArray u_NormalArray;
layout (binding = 12) uniform sampler2DArray u_DisplacementArray;
layout (binding = 13) uniform sampler2DArray u_ARMArray;

// Phase 6: POM Uniforms & Radius Culling
uniform int u_POMEnabled = 1;
uniform int u_POMMinSamples = 8;
uniform int u_POMMaxSamples = 64;
uniform float u_POMHeightScale = 0.08;
uniform int u_POMSelfShadow = 1;
uniform float u_DispNearRadius = 120.0;
uniform float u_DispMidRadius = 300.0;
uniform float u_MaterialDispDepth[32];
uniform float u_POMScaleMultiplier = 1.0;

// Displacement modes (matches DisplacementMode enum in texture_atlas.py)
const uint DISP_MODE_NONE = 0u;
const uint DISP_MODE_POM = 1u;
const uint DISP_MODE_SSDM = 2u;
const uint DISP_MODE_TESSELLATION = 3u;

// Octahedral Normal Encoding
vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 OctahedralEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * signNotZero(n.xy);
    return oct * 0.5 + 0.5;
}

// Steep Parallax Occlusion Mapping with 6-iteration binary refinement
// Returns displaced UV and final height at the intersection (1.0 = peak, 0.0 = valley)
vec2 ParallaxOcclusionMap(vec2 uv, vec3 view_dir_ts, float layer_idx, float height_scale, int target_samples, out float out_height) {
    float v_dot_n = clamp(normalize(view_dir_ts).z, 0.0, 1.0);
    int num_steps = int(mix(float(target_samples), float(u_POMMinSamples), v_dot_n));
    num_steps = clamp(num_steps, 16, 64);

    float step_h = 1.0 / float(num_steps);
    float ray_h = 1.0;

    // View vector projected onto surface in tangent space (clamped divisor prevents horizon divergence)
    vec2 view_dir_2d = view_dir_ts.xy / max(abs(view_dir_ts.z), 0.15);
    vec2 delta_uv = view_dir_2d * height_scale * step_h;

    vec2 current_uv = uv;
    float current_height = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;

    // Steep Parallax: step downward from 1.0 toward 0.0 until ray penetrates heightfield
    for (int i = 0; i < u_POMMaxSamples; ++i) {
        if (ray_h <= current_height || i >= num_steps) {
            break;
        }
        current_uv -= delta_uv;
        ray_h -= step_h;
        current_height = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;
    }

    // Binary refinement (6 iterations for sub-texel depth stability)
    vec2 prev_uv = current_uv + delta_uv;
    float prev_ray_h = ray_h + step_h;

    for (int i = 0; i < 6; ++i) {
        vec2 mid_uv = (current_uv + prev_uv) * 0.5;
        float mid_ray_h = (ray_h + prev_ray_h) * 0.5;
        float mid_height = texture(u_DisplacementArray, vec3(mid_uv, layer_idx)).r;

        if (mid_ray_h > mid_height) {
            // Ray is still above the surface, search deeper half
            prev_uv = mid_uv;
            prev_ray_h = mid_ray_h;
        } else {
            // Ray is below or on surface, search upper half
            current_uv = mid_uv;
            ray_h = mid_ray_h;
        }
    }

    // Exact ground-truth surface height at the refined intersection UV
    out_height = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;
    return current_uv;
}

// POM Self-Shadowing: secondary raymarch upward toward sun in tangent space
float POMSelfShadow(vec2 uv, float surface_height, vec3 sun_dir_ts, float layer_idx, float height_scale) {
    if (sun_dir_ts.z <= 0.0) return 0.0; // Facing away from sun

    // At top peak ceiling, no microgeometry can occlude the sun from above
    float total_h = 1.0 - surface_height;
    if (total_h <= 0.005) return 1.0;

    int num_steps = clamp(u_POMMinSamples * 3, 24, 48);
    float step_h = total_h / float(num_steps);

    // Tangent-space sun direction in UV space: fixed physical slope independent of starting height
    vec2 sun_dir_2d = sun_dir_ts.xy / max(sun_dir_ts.z, 0.05);
    vec2 delta_uv = sun_dir_2d * height_scale * step_h;

    vec2 current_uv = uv + delta_uv;
    float current_h = surface_height + step_h;
    float max_occlusion = 0.0;

    for (int i = 0; i < num_steps; ++i) {
        if (current_h >= 1.0) break;

        float h = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;
        if (h > current_h) {
            float diff = h - current_h;
            float weight = 1.0 - (float(i) / float(num_steps));
            max_occlusion = max(max_occlusion, diff * weight);
        }

        current_uv += delta_uv;
        current_h += step_h;
    }

    return clamp(1.0 - max_occlusion * 2.5, 0.0, 1.0);
}

void main() {
    // Read entity material parameters from SSBO 2
    vec4 mat0 = u_MaterialData[v_EntityID * 2];     // [R, G, B, Roughness]
    vec4 mat1 = u_MaterialData[v_EntityID * 2 + 1]; // [Metallic, AO, TexLayerIdx, MatFlags]

    vec3 albedo = mat0.rgb;
    float roughness = clamp(mat0.a, 0.04, 1.0);
    float metallic = clamp(mat1.r, 0.0, 1.0);
    float ao = clamp(mat1.g, 0.0, 1.0);

    float tex_layer = mat1.b;  // Texture array layer index (0 = no texture)
    uint mat_flags = uint(mat1.a);

    bool has_texture = (mat_flags & 1u) != 0u;
    uint disp_mode = (mat_flags >> 1u) & 3u; // 0=None, 1=POM, 2=SSDM, 3=Tessellation

    vec3 N = normalize(v_Normal);
    vec2 final_uv = v_UV;
    float pom_shadow = 1.0;

    // Strict Mutual Exclusivity & Camera Radius Culling: POM runs ONLY when disp_mode == DISP_MODE_POM (1) and within u_DispMidRadius
    if (has_texture && tex_layer > 0.0) {
        float layer_idx = max(0.0, floor(tex_layer + 0.5));
        float cam_dist = length(u_CameraPos_Time.xyz - v_WorldPos);

        // Deactivated outside medium radius
        if (u_POMEnabled == 1 && disp_mode == DISP_MODE_POM && cam_dist <= u_DispMidRadius) {
            // Per-fragment orthonormal TBN matrix for pristine camera tracking on large ground quads
            vec3 T_frag = normalize(v_TBN[0]);
            vec3 B_frag = normalize(v_TBN[1]);
            vec3 N_frag = normalize(v_TBN[2]);
            mat3 TBN_inv = transpose(mat3(T_frag, B_frag, N_frag));

            vec3 view_world = u_CameraPos_Time.xyz - v_WorldPos;
            vec3 view_ts = normalize(TBN_inv * view_world);
            float pom_height;
            int mat_i = clamp(int(layer_idx), 0, 31);
            float mat_depth = u_MaterialDispDepth[mat_i];
            if (mat_depth <= 0.0) mat_depth = 0.035;
            float effective_scale = mat_depth * u_POMScaleMultiplier;

            // Taper sample count between Near and Mid radius
            float dist_factor = clamp((u_DispMidRadius - cam_dist) / max(u_DispMidRadius - u_DispNearRadius, 0.001), 0.0, 1.0);
            int target_samples = int(mix(float(u_POMMinSamples), float(u_POMMaxSamples), dist_factor));

            final_uv = ParallaxOcclusionMap(v_UV, view_ts, layer_idx, effective_scale, target_samples, pom_height);

            // Sun self-shadowing with smooth distance attenuation (eliminates boundary popping)
            if (u_POMSelfShadow == 1 && cam_dist <= u_DispNearRadius) {
                vec3 sun_ts = normalize(TBN_inv * (-u_SunDirection_Intensity.xyz));
                float raw_shadow = POMSelfShadow(final_uv, pom_height, sun_ts, layer_idx, effective_scale);
                float shadow_fade = 1.0 - smoothstep(u_DispNearRadius * 0.70, u_DispNearRadius, cam_dist);
                pom_shadow = mix(1.0, raw_shadow, shadow_fade);
            }
        }

        // Sample PBR texture arrays at final_uv
        vec4 diffuse_sample = texture(u_DiffuseArray, vec3(final_uv, layer_idx));
        albedo = diffuse_sample.rgb;

        // Normal mapping via TBN
        vec3 normal_sample = texture(u_NormalArray, vec3(final_uv, layer_idx)).rgb;
        vec3 tangent_normal = normalize(normal_sample * 2.0 - 1.0);
        N = normalize(v_TBN * tangent_normal);

        // ARM texture: R=AO, G=Roughness, B=Metallic
        vec4 arm_sample = texture(u_ARMArray, vec3(final_uv, layer_idx));
        ao = arm_sample.r;
        roughness = clamp(arm_sample.g, 0.04, 1.0);
        metallic = clamp(arm_sample.b, 0.0, 1.0);
    }

    // Apply POM self-shadow into AO channel
    ao *= pom_shadow;

    // Velocity vector (UV space: curr_uv - prev_uv)
    float curr_inv_w = 1.0 / max(v_CurrClip.w, 1e-6);
    float prev_inv_w = 1.0 / max(v_PrevClip.w, 1e-6);
    vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5;
    vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
    vec2 velocity = curr_uv - prev_uv;

    // Pack into MRT outputs
    out_AlbedoRoughness = vec4(albedo, roughness);
    out_NormalMetallic = vec4(OctahedralEncode(N), metallic, ao);
    out_Velocity = velocity;
    out_DispInfo = vec4(final_uv, tex_layer, float(disp_mode));
}
