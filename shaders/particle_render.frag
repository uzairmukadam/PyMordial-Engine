#version 430 core

#include "common/frame_data.glsl"

in vec2  v_UV;
in vec4  v_Color;
in float v_LifeFraction;
in vec3  v_WorldPos;
in float v_ViewDepth;
flat in int v_Mode;

layout(location = 0) out vec4 out_Color;

layout(binding = 0) uniform sampler2D u_DepthTexture; // Reversed-Z 32F
uniform float u_SoftParticleRadius;                   // Feathering distance (meters)
uniform float u_SunScatterIntensity;                  // Sun forward glint multiplier

void main() {
    // 1. Soft Circular Disc Falloff
    float distSq = dot(v_UV, v_UV);
    if (distSq > 1.0) {
        discard;
    }
    // Smooth gaussian-style edge
    float shapeAlpha = 1.0 - smoothstep(0.1, 1.0, distSq);

    // 2. Soft Depth Feathering (Z-Feathering against G-Buffer Reversed-Z Depth)
    vec2 screenUV = gl_FragCoord.xy / u_ScreenSize_Jitter.xy;
    float rawDepth = texture(u_DepthTexture, screenUV).r;

    float softAlpha = 1.0;
    if (rawDepth > 1e-6) {
        vec4 clipPos = vec4(screenUV * 2.0 - 1.0, rawDepth, 1.0);
        vec4 viewPos = u_InvProjection * clipPos;
        float sceneViewDepth = max(-viewPos.z / max(viewPos.w, 1e-6), 0.05);

        float depthDelta = sceneViewDepth - v_ViewDepth;
        if (depthDelta < -0.02) {
            discard; // Occluded behind solid geometry
        }
        float softRadius = max(u_SoftParticleRadius, 0.01);
        softAlpha = clamp(depthDelta / softRadius, 0.0, 1.0);
    }

    // 3. Smooth Lifetime Envelope (Fade In -> Plateau -> Fade Out)
    float lifeIn = smoothstep(1.0, 0.85, v_LifeFraction);
    float lifeOut = smoothstep(0.0, 0.15, v_LifeFraction);
    float lifeAlpha = min(lifeIn, lifeOut);

    // 4. Lighting & Scattering Model
    vec3 litColor;
    if (v_Mode == 1) {
        // Embers / Sparks: High-Dynamic-Range emissive glow with temporal flicker
        float flicker = 1.2 + 0.45 * sin(u_CameraPos_Time.w * 16.0 + v_WorldPos.x * 25.0 + v_WorldPos.y * 35.0);
        litColor = v_Color.rgb * flicker * 3.5;
    } else if (v_Mode == 2) {
        // Fireflies: Bioluminescent pulsing glow
        float pulse = 0.8 + 0.6 * sin(u_CameraPos_Time.w * 3.5 + v_WorldPos.x * 6.0);
        litColor = v_Color.rgb * pulse * 2.5;
    } else {
        // Dust Motes: Henyey-Greenstein Mie forward scattering glint
        vec3 viewDir = normalize(v_WorldPos - u_CameraPos_Time.xyz);
        vec3 toSun = -u_SunDirection_Intensity.xyz; // points towards sun
        float cosTheta = dot(viewDir, toSun);

        // Forward scattering lobe (g = 0.65)
        const float g = 0.65;
        const float g2 = g * g;
        float hgPhase = (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.01), 1.5);

        float sunGlint = hgPhase * u_SunScatterIntensity;
        vec3 sunLight = u_SunColor_Ambient.rgb * (u_SunDirection_Intensity.w * (0.2 + sunGlint));
        vec3 ambient = u_SunColor_Ambient.rgb * (u_SunColor_Ambient.w * 2.5);

        litColor = (ambient + sunLight) * v_Color.rgb;
    }

    float finalAlpha = shapeAlpha * softAlpha * lifeAlpha * v_Color.a;
    if (finalAlpha <= 0.005) {
        discard;
    }

    out_Color = vec4(litColor, finalAlpha);
}
