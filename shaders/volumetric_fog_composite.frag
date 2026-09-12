#version 450 core
// ============================================================
//  Froxel Volumetric Fog: Fullscreen HDR Composite Pass
//  Samples trilinearly from 3D integrated volume and outputs
//  vec4(InScattering, Transmittance) for analytical hardware blending.
// ============================================================

in vec2 v_UV;
layout (location = 0) out vec4 out_FogColor;

layout (binding = 0) uniform sampler3D u_IntegratedVol;
layout (binding = 1) uniform sampler2D u_GBufferDepth;

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

uniform float u_FogNear = 0.1;
uniform float u_FogFar = 400.0;
uniform float u_GridDepth = 64.0;
uniform int u_DebugMode = 0; // 0=Normal, 1=Inscattering Only, 2=Transmittance Only

// Interleaved Gradient Noise for sub-voxel anti-banding reconstruction
float InterleavedGradientNoise(vec2 screenPos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

void main() {
    float rawDepth = texture(u_GBufferDepth, v_UV).r;
    bool isSky = (rawDepth <= 1e-6);

    float viewDepth;
    if (isSky) {
        // Sky background integrates across the full frustum volume
        viewDepth = u_FogFar;
    } else {
        // Reconstruct view-space depth from Reversed-Z depth
        vec4 clipPos = vec4(v_UV * 2.0 - 1.0, rawDepth, 1.0);
        vec4 viewPos = u_InvProjection * clipPos;
        viewDepth = max(-viewPos.z / max(viewPos.w, 1e-6), u_FogNear);
    }

    // Continuous logarithmic depth coordinate in [0, 1] with sub-voxel spatial dithering
    float logNear = log(u_FogNear);
    float logFar = log(u_FogFar);
    float baseW = clamp((log(max(viewDepth, u_FogNear)) - logNear) / max(logFar - logNear, 1e-5), 0.0, 1.0);
    float dither = (InterleavedGradientNoise(gl_FragCoord.xy) - 0.5) / max(u_GridDepth, 16.0);
    float w = clamp(baseW + dither, 0.0, 1.0);

    // Trilinear hardware lookup into the integrated froxel volume
    vec4 fogSample = texture(u_IntegratedVol, vec3(v_UV, w));
    vec3 inScattering = fogSample.rgb;
    float transmittance = clamp(fogSample.a, 0.0, 1.0);

    // Atmospheric preservation for the sky:
    // Ground fog should NOT black out the sky dome; the sky retains its luminosity
    // while receiving volumetric sun god rays and horizon haze.
    if (isSky) {
        transmittance = max(transmittance, 0.70);
    }

    if (u_DebugMode == 1) {
        // Debug In-scattering only: zero transmittance multiplier hides scene geometry
        out_FogColor = vec4(inScattering, 0.0);
    } else if (u_DebugMode == 2) {
        // Debug Transmittance only: grayscale optical depth
        out_FogColor = vec4(vec3(transmittance), 0.0);
    } else {
        // Standard physical composite:
        // Result = FragColor.rgb * 1.0 + DstColor.rgb * FragColor.a
        out_FogColor = vec4(inScattering, transmittance);
    }
}
