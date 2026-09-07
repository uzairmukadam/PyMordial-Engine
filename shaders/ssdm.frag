#version 450 core

/*
 * Screen-Space Displacement Mapping (SSDM)
 *
 * Post-G-Buffer pass modifying hardware depth (gl_FragDepth) and maintaining
 * G-Buffer normals based on displacement heightmaps for surfaces with
 * SSDM enabled (disp_mode == 2).
 *
 * Strict Mutual Exclusivity:
 * Reads RT3 (u_GBufferDispInfo). If disp_mode != 2, the pixel is passed
 * through completely untouched.
 */

in vec2 v_UV;

// G-Buffer inputs
layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_GBufferDispInfo; // RT3: RG=Mesh UV, B=TexLayer, A=DispMode

// Displacement texture array
layout (binding = 12) uniform sampler2DArray u_DisplacementArray;

// Frame Data UBO
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

// Output: modified G-Buffer normal + metallic (re-writes RT1)
layout (location = 0) out vec4 out_NormalMetallic;

// SSDM Configuration & Radius Culling
uniform int u_SSDMEnabled = 1;
uniform float u_SSDMScale = 0.05;
uniform float u_SSDMMaxDistance = 30.0;
uniform float u_SSDMTiling = 1.0;
uniform float u_DispMidRadius = 25.0;
uniform float u_MaterialDispDepth[32];
uniform float u_SSDMScaleMultiplier = 1.0;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 OctahedralEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * signNotZero(n.xy);
    return oct * 0.5 + 0.5;
}

void main() {
    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    vec4 orig_nm = texture(u_GBufferNormalMetallic, v_UV);

    // Skip sky (Reversed-Z: 0.0 is far/sky)
    if (raw_depth <= 0.000001) {
        gl_FragDepth = raw_depth;
        out_NormalMetallic = orig_nm;
        return;
    }

    // Read Displacement Info from RT3 (RG=Mesh UV, B=TexLayer, A=DispMode)
    vec4 disp_info = texture(u_GBufferDispInfo, v_UV);
    vec2 mesh_uv = disp_info.xy;
    float tex_layer = disp_info.z;
    int disp_mode = int(disp_info.w + 0.5);

    // Strict Mutual Exclusivity: Only apply SSDM if disp_mode == 2
    if (u_SSDMEnabled == 0 || disp_mode != 2 || tex_layer <= 0.0) {
        gl_FragDepth = raw_depth;
        out_NormalMetallic = orig_nm;
        return;
    }

    // Reconstruct 3D view-space position from raw depth
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos_h = u_InvProjection * clip_pos;
    vec3 view_pos = view_pos_h.xyz / max(view_pos_h.w, 1e-6);

    float linear_z = -view_pos.z;

    // Radius-based deactivation beyond u_DispMidRadius
    if (linear_z > u_DispMidRadius) {
        gl_FragDepth = raw_depth;
        out_NormalMetallic = orig_nm;
        return;
    }

    float max_d = min(u_SSDMMaxDistance, u_DispMidRadius);
    float dist_fade = clamp(1.0 - (linear_z / max_d), 0.0, 1.0);
    if (dist_fade <= 0.001) {
        gl_FragDepth = raw_depth;
        out_NormalMetallic = orig_nm;
        return;
    }

    // Decode G-Buffer world normal and transform to view space
    vec3 N_world = OctahedralDecode(orig_nm.rg);
    vec3 N_view = normalize((u_View * vec4(N_world, 0.0)).xyz);

    // Sample displacement heightfield at this pixel's mesh UV
    float height = texture(u_DisplacementArray, vec3(mesh_uv, tex_layer)).r;

    // Per-material depth lookup
    int mat_i = clamp(int(tex_layer), 0, 31);
    float mat_depth = u_MaterialDispDepth[mat_i];
    if (mat_depth <= 0.0) mat_depth = 0.035;

    // Displace 3D view position along view-space normal
    float disp = (height - 0.5) * mat_depth * u_SSDMScaleMultiplier * dist_fade;
    vec3 displaced_view_pos = view_pos + N_view * disp;

    // Project displaced view position back to clip space
    vec4 displaced_clip = u_Projection * vec4(displaced_view_pos, 1.0);
    float displaced_depth = displaced_clip.z / max(displaced_clip.w, 1e-6);
    gl_FragDepth = clamp(displaced_depth, 0.0, 1.0);

    // Preserve the pristine 4K normal map from G-Buffer RT1
    out_NormalMetallic = orig_nm;
}
