#version 450 core

in vec2 v_UV;
out vec4 out_AlbedoRoughness;

layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferDepth;

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

struct DecalData {
    mat4 world_to_decal;
    vec4 color_opacity; // rgb=tint, a=opacity
    vec4 params;        // x=roughness, y=decal_type (0=skidmark, 1=oil, 2=generic), z=reserved, w=reserved
};

layout (std430, binding = 4) readonly buffer DecalBuffer {
    DecalData u_Decals[128];
};

uniform int u_ActiveDecalCount = 0;

void main() {
    vec4 orig_albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    if (u_ActiveDecalCount <= 0) {
        out_AlbedoRoughness = orig_albedo_rough;
        return;
    }

    float raw_depth = texture(u_GBufferDepth, v_UV).r;
    // Under reversed-Z, 0.0 is far plane (sky / distant background)
    if (raw_depth <= 1e-6) {
        out_AlbedoRoughness = orig_albedo_rough;
        return;
    }

    // Reconstruct world position from reversed-Z depth buffer
    vec4 ndc = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_h = u_InvProjection * ndc;
    vec3 view_pos = view_h.xyz / max(abs(view_h.w), 1e-6);
    vec3 world_pos = (u_InvView * vec4(view_pos, 1.0)).xyz;

    vec3 accum_albedo = orig_albedo_rough.rgb;
    float accum_rough = orig_albedo_rough.a;

    int count = min(u_ActiveDecalCount, 128);
    for (int i = 0; i < count; ++i) {
        vec4 local_p4 = u_Decals[i].world_to_decal * vec4(world_pos, 1.0);
        vec3 local_p = local_p4.xyz;

        // Bounding box test: decal volume is [-0.5, 0.5]^3
        if (abs(local_p.x) <= 0.5 && abs(local_p.y) <= 0.5 && abs(local_p.z) <= 0.5) {
            float opacity = u_Decals[i].color_opacity.a;
            if (opacity <= 0.001) continue;

            // Smooth edge falloff along X, Y, Z borders to prevent hard clipping lines
            float edge_x = smoothstep(0.5, 0.38, abs(local_p.x));
            float edge_y = smoothstep(0.5, 0.12, abs(local_p.y)); // Y is vertical projection thickness
            float edge_z = smoothstep(0.5, 0.38, abs(local_p.z));
            float mask = edge_x * edge_y * edge_z * opacity;

            int dtype = int(u_Decals[i].params.y);
            if (dtype == 0) {
                // Tire skidmark pattern: dual longitudinal contact tracks with asphalt micro-grain
                float tread_u = local_p.x + 0.5;
                float groove = smoothstep(0.14, 0.02, abs(tread_u - 0.26)) + smoothstep(0.14, 0.02, abs(tread_u - 0.74));
                float grain = (sin(world_pos.x * 35.0) * cos(world_pos.z * 35.0)) * 0.12 + 0.88;
                mask *= clamp(groove * 0.75 + 0.25, 0.0, 1.0) * grain;
            }

            vec3 decal_col = u_Decals[i].color_opacity.rgb;
            float decal_rough = u_Decals[i].params.x;

            accum_albedo = mix(accum_albedo, decal_col, mask);
            accum_rough = mix(accum_rough, decal_rough, mask);
        }
    }

    out_AlbedoRoughness = vec4(accum_albedo, accum_rough);
}
