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

// SSBO 2: Material Data (two vec4 per entity)
layout (std430, binding = 2) readonly buffer MaterialBuffer {
    vec4 u_MaterialData[];
};

// Phase 6: PBR Texture Array Atlases (sampler2DArray)
layout (binding = 10) uniform sampler2DArray u_DiffuseArray;
layout (binding = 11) uniform sampler2DArray u_NormalArray;
layout (binding = 12) uniform sampler2DArray u_DisplacementArray;
layout (binding = 13) uniform sampler2DArray u_ARMArray;

// Phase 6: POM Uniforms
uniform int u_POMEnabled = 1;
uniform int u_POMMinSamples = 8;
uniform int u_POMMaxSamples = 64;
uniform float u_POMHeightScale = 0.08;
uniform int u_POMSelfShadow = 1;

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

// Steep Parallax Occlusion Mapping with binary refinement
// Returns displaced UV and final height at the intersection
vec2 ParallaxOcclusionMap(vec2 uv, vec3 view_dir_ts, float layer_idx, out float out_height) {
    float v_dot_n = max(dot(normalize(view_dir_ts), vec3(0.0, 0.0, 1.0)), 0.0);
    int num_steps = int(mix(float(u_POMMaxSamples), float(u_POMMinSamples), v_dot_n));

    float layer_depth = 1.0 / float(num_steps);
    float current_layer_depth = 0.0;

    // View vector projected onto surface in tangent space
    vec2 view_dir_2d = view_dir_ts.xy / max(abs(view_dir_ts.z), 0.001);
    vec2 delta_uv = view_dir_2d * u_POMHeightScale / float(num_steps);

    vec2 current_uv = uv;
    float current_height = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;

    // Steep Parallax: step until ray penetrates heightfield
    for (int i = 0; i < u_POMMaxSamples; ++i) {
        if (current_layer_depth >= current_height || i >= num_steps) {
            break;
        }
        current_uv -= delta_uv;
        current_height = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;
        current_layer_depth += layer_depth;
    }

    // Binary refinement (5 iterations)
    vec2 prev_uv = current_uv + delta_uv;
    float prev_height = texture(u_DisplacementArray, vec3(prev_uv, layer_idx)).r;
    float prev_layer_depth = current_layer_depth - layer_depth;

    for (int i = 0; i < 5; ++i) {
        vec2 mid_uv = (current_uv + prev_uv) * 0.5;
        float mid_height = texture(u_DisplacementArray, vec3(mid_uv, layer_idx)).r;
        float mid_layer_depth = (current_layer_depth + prev_layer_depth) * 0.5;

        if (mid_height > mid_layer_depth) {
            prev_uv = mid_uv;
            prev_layer_depth = mid_layer_depth;
        } else {
            current_uv = mid_uv;
            current_layer_depth = mid_layer_depth;
            current_height = mid_height;
        }
    }

    out_height = current_height;
    return current_uv;
}

// POM Self-Shadowing: secondary raymarch toward sun in tangent space
float POMSelfShadow(vec2 uv, float surface_height, vec3 sun_dir_ts, float layer_idx) {
    if (sun_dir_ts.z <= 0.0) return 0.0; // Below horizon

    int num_steps = 16;
    float layer_depth = surface_height / float(num_steps);

    vec2 sun_dir_2d = sun_dir_ts.xy / max(abs(sun_dir_ts.z), 0.001);
    vec2 delta_uv = sun_dir_2d * u_POMHeightScale / float(num_steps);

    float current_layer_depth = surface_height - layer_depth;
    vec2 current_uv = uv + delta_uv;

    float shadow = 0.0;
    for (int i = 0; i < num_steps; ++i) {
        if (current_layer_depth <= 0.0) break;

        float h = texture(u_DisplacementArray, vec3(current_uv, layer_idx)).r;
        if (h > current_layer_depth) {
            shadow = max(shadow, (h - current_layer_depth) / (1.0 - current_layer_depth + 0.001));
        }

        current_uv += delta_uv;
        current_layer_depth -= layer_depth;
    }

    return clamp(1.0 - shadow * 1.5, 0.0, 1.0);
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

    // Strict Mutual Exclusivity: POM runs ONLY when disp_mode == DISP_MODE_POM (1)
    if (has_texture && tex_layer > 0.0) {
        float layer_idx = tex_layer;

        if (u_POMEnabled == 1 && disp_mode == DISP_MODE_POM) {
            vec3 view_ts = normalize(v_TangentViewDir);
            float pom_height;
            final_uv = ParallaxOcclusionMap(v_UV, view_ts, layer_idx, pom_height);

            // Sun self-shadowing
            if (u_POMSelfShadow == 1) {
                vec3 sun_ts = normalize(v_TangentSunDir);
                pom_shadow = POMSelfShadow(final_uv, pom_height, sun_ts, layer_idx);
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
