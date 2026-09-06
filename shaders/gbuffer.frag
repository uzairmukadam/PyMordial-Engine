#version 450 core

// Multiple Render Targets (MRT)
layout (location = 0) out vec4 out_AlbedoRoughness; // RT0: Albedo (RGB) + Roughness (A)
layout (location = 1) out vec4 out_NormalMetallic;   // RT1: Octahedral Normal (RG) + Metallic (B) + AO (A)
layout (location = 2) out vec2 out_Velocity;         // RT2: Velocity Vectors (RG16F)

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_UV;
in vec4 v_CurrClip;
in vec4 v_PrevClip;
in flat uint v_EntityID;

// SSBO 2: Material Data (two vec4 per entity)
layout (std430, binding = 2) readonly buffer MaterialBuffer {
    vec4 u_MaterialData[];
};

// Octahedral Normal Encoding
vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 OctahedralEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * signNotZero(n.xy);
    return oct * 0.5 + 0.5;
}

void main() {
    // Read entity material parameters from SSBO 2
    vec4 mat0 = u_MaterialData[v_EntityID * 2];     // [R, G, B, Roughness]
    vec4 mat1 = u_MaterialData[v_EntityID * 2 + 1]; // [Metallic, AO, AlbedoTexID, NormalTexID]

    vec3 albedo = mat0.rgb;
    float roughness = clamp(mat0.a, 0.04, 1.0);
    float metallic = clamp(mat1.r, 0.0, 1.0);
    float ao = clamp(mat1.g, 0.0, 1.0);

    // Compute screen-space velocity vector (UV space: curr_uv - prev_uv)
    float curr_inv_w = 1.0 / max(v_CurrClip.w, 1e-6);
    float prev_inv_w = 1.0 / max(v_PrevClip.w, 1e-6);
    vec2 curr_uv = (v_CurrClip.xy * curr_inv_w) * 0.5 + 0.5;
    vec2 prev_uv = (v_PrevClip.xy * prev_inv_w) * 0.5 + 0.5;
    vec2 velocity = curr_uv - prev_uv;

    // Pack into MRT outputs
    out_AlbedoRoughness = vec4(albedo, roughness);
    out_NormalMetallic = vec4(OctahedralEncode(normalize(v_Normal)), metallic, ao);
    out_Velocity = velocity;
}
