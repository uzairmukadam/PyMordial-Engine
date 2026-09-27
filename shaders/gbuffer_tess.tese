#version 450 core

layout (triangles, fractional_odd_spacing, ccw) in;

in vec3 te_Position[];
in vec3 te_Normal[];
in vec2 te_UV[];
in vec4 te_Tangent[];
in flat uint te_EntityID[];

// Outputs to G-Buffer Fragment (identical interface to gbuffer.vert)
out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_UV;
out vec3 v_ModelPos;
out vec3 v_ModelNormal;
out vec4 v_CurrClip;
out vec4 v_PrevClip;
out flat uint v_EntityID;

out mat3 v_TBN;
out vec3 v_TangentViewDir;
out vec3 v_TangentSunDir;

// Frame Data UBO (binding 0)
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

// Phase 6: Displacement Texture Array & Per-Material Depths
layout (binding = 12) uniform sampler2DArray u_DisplacementArray;
uniform float u_MaterialDispDepth[32];
uniform float u_TessDisplacementScale = 1.0;
uniform mat4 u_PrevViewProjection;

void main() {
    // Barycentric interpolation weights
    vec3 tc = gl_TessCoord;

    vec3 p0 = te_Position[0];
    vec3 p1 = te_Position[1];
    vec3 p2 = te_Position[2];

    vec3 n0 = te_Normal[0];
    vec3 n1 = te_Normal[1];
    vec3 n2 = te_Normal[2];

    vec2 uv0 = te_UV[0];
    vec2 uv1 = te_UV[1];
    vec2 uv2 = te_UV[2];

    vec4 t0 = te_Tangent[0];
    vec4 t1 = te_Tangent[1];
    vec4 t2 = te_Tangent[2];

    vec3 pos = tc.x * p0 + tc.y * p1 + tc.z * p2;
    vec3 norm = normalize(tc.x * n0 + tc.y * n1 + tc.z * n2);
    vec2 uv = tc.x * uv0 + tc.y * uv1 + tc.z * uv2;
    vec4 tang = tc.x * t0 + tc.y * t1 + tc.z * t2;

    uint entity_idx = te_EntityID[0];
    v_EntityID = entity_idx;

    // Read entity material parameters
    vec4 mat1 = u_MaterialData[entity_idx * 2 + 1];
    float tex_layer = mat1.b;

    // Sample displacement heightfield with explicit LOD 0 (GLSL non-fragment requirement)
    if (tex_layer > 0.0) {
        float height = textureLod(u_DisplacementArray, vec3(uv, tex_layer), 0.0).r;
        int layer_idx = clamp(int(tex_layer), 0, 31);
        float mat_depth = u_MaterialDispDepth[layer_idx];
        if (mat_depth <= 0.0) {
            mat_depth = 0.030; // Physical fallback default (3 cm)
        }
        float disp = (height - 0.5) * mat_depth * u_TessDisplacementScale;

        // Seam-healing protection for flat-face primitives (e.g. cubes with split normals)
        // Displacements smoothly fade to 0 at the UV perimeter so adjacent faces meet seamlessly without tears
        bool is_flat_patch = (dot(n0, n1) > 0.999 && dot(n1, n2) > 0.999);
        if (is_flat_patch) {
            float seam = smoothstep(0.0, 0.04, uv.x) * (1.0 - smoothstep(0.96, 1.0, uv.x)) *
                         smoothstep(0.0, 0.04, uv.y) * (1.0 - smoothstep(0.96, 1.0, uv.y));
            disp *= seam;
        }

        pos += norm * disp;
    }

    v_WorldPos = pos;
    v_ModelPos = pos;
    v_ModelNormal = norm;
    v_Normal = norm;
    v_UV = uv;

    v_CurrClip = u_ViewProjection * vec4(pos, 1.0);
    v_PrevClip = (u_PrevViewProjection[3][3] != 0.0) ? (u_PrevViewProjection * vec4(pos, 1.0)) : v_CurrClip;

    // Recompute TBN matrix for the tessellated surface
    vec3 T = normalize(tang.xyz - dot(tang.xyz, norm) * norm);
    vec3 B = cross(norm, T) * tang.w;
    v_TBN = mat3(T, B, norm);

    mat3 TBN_inv = transpose(v_TBN);
    vec3 view_dir = u_CameraPos_Time.xyz - pos;
    v_TangentViewDir = TBN_inv * view_dir;
    v_TangentSunDir = TBN_inv * (-u_SunDirection_Intensity.xyz);

    gl_Position = v_CurrClip;
}
