#version 450 core

layout (triangles, fractional_odd_spacing, ccw) in;

in vec3 te_Position[];
in vec3 te_Normal[];
in vec2 te_UV[];
in flat uint te_EntityID[];

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

layout (std430, binding = 2) readonly buffer MaterialBuffer {
    vec4 u_MaterialData[];
};

layout (binding = 12) uniform sampler2DArray u_DisplacementArray;
uniform float u_MaterialDispDepth[32];
uniform uint u_CascadeIndex;
uniform float u_TessDisplacementScale = 1.0;

void main() {
    vec3 tc = gl_TessCoord;
    vec3 pos = tc.x * te_Position[0] + tc.y * te_Position[1] + tc.z * te_Position[2];
    vec3 norm = normalize(tc.x * te_Normal[0] + tc.y * te_Normal[1] + tc.z * te_Normal[2]);
    vec2 uv = tc.x * te_UV[0] + tc.y * te_UV[1] + tc.z * te_UV[2];

    uint entity_idx = te_EntityID[0];
    float tex_layer = u_MaterialData[entity_idx * 2 + 1].b;

    if (tex_layer > 0.0) {
        float height = textureLod(u_DisplacementArray, vec3(uv, tex_layer), 0.0).r;
        int layer_idx = clamp(int(tex_layer), 0, 31);
        float mat_depth = u_MaterialDispDepth[layer_idx];
        if (mat_depth <= 0.0) {
            mat_depth = 0.030;
        }
        float disp = (height - 0.5) * mat_depth * u_TessDisplacementScale;

        // Seam healing for flat patches
        bool is_flat_patch = (dot(te_Normal[0], te_Normal[1]) > 0.999 && dot(te_Normal[1], te_Normal[2]) > 0.999);
        if (is_flat_patch) {
            float seam = smoothstep(0.0, 0.04, uv.x) * (1.0 - smoothstep(0.96, 1.0, uv.x)) *
                         smoothstep(0.0, 0.04, uv.y) * (1.0 - smoothstep(0.96, 1.0, uv.y));
            disp *= seam;
        }

        pos += norm * disp;
    }

    gl_Position = u_LightViewProjection[u_CascadeIndex] * vec4(pos, 1.0);
}
