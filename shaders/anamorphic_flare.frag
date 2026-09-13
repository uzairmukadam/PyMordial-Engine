#version 450 core

in vec2 v_UV;
layout(location = 0) out vec4 out_FlareColor;

layout(binding = 0) uniform sampler2D u_SceneHDR;

uniform float u_FlareThreshold;       // Luminance cutoff (e.g. 1.8)
uniform float u_StreakIntensity;      // Anamorphic horizontal streak brightness
uniform float u_StreakWidth;          // Horizontal streak spread (pixels)
uniform float u_GhostIntensity;       // Optical ghost reflections brightness
uniform float u_HaloIntensity;        // Lens ring halo brightness
uniform vec3  u_FlareTint;            // Base flare tint (e.g. vec3(0.25, 0.65, 1.0) for anamorphic cyan/blue)
uniform vec2  u_ScreenSize;           // Window resolution (width, height)

float getLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 sampleThreshold(vec2 uv) {
    vec3 col = texture(u_SceneHDR, uv).rgb;
    float lum = getLuminance(col);
    float excess = max(0.0, lum - u_FlareThreshold);
    // Soft knee curve for natural highlight transition
    float factor = excess / max(lum, 1e-4);
    return col * factor;
}

void main() {
    vec2 texelSize = 1.0 / u_ScreenSize;
    vec3 totalFlare = vec3(0.0);

    // -------------------------------------------------------------------------
    // 1. Anamorphic Horizontal Streak with Chromatic Dispersion
    // -------------------------------------------------------------------------
    if (u_StreakIntensity > 0.001) {
        vec3 streak = vec3(0.0);
        const int STREAK_TAPS = 16;
        float halfWidth = u_StreakWidth * texelSize.x;

        for (int i = -STREAK_TAPS; i <= STREAK_TAPS; ++i) {
            float offsetNorm = float(i) / float(STREAK_TAPS);
            float weight = exp(-4.0 * offsetNorm * offsetNorm);

            // Chromatic dispersion along horizontal streak (red outer, blue inner)
            float xOffset = offsetNorm * halfWidth;
            float r = sampleThreshold(v_UV + vec2(xOffset * 1.04, 0.0)).r;
            float g = sampleThreshold(v_UV + vec2(xOffset, 0.0)).g;
            float b = sampleThreshold(v_UV + vec2(xOffset * 0.96, 0.0)).b;

            streak += vec3(r, g, b) * weight;
        }

        streak /= float(STREAK_TAPS * 2 + 1) * 0.4;
        totalFlare += streak * u_StreakIntensity * u_FlareTint;
    }

    // -------------------------------------------------------------------------
    // 2. Optical Ghost Reflections through Lens Axis
    // -------------------------------------------------------------------------
    if (u_GhostIntensity > 0.001) {
        vec2 opticalCenter = vec2(0.5, 0.5);
        vec2 toCenter = opticalCenter - v_UV;

        vec3 ghosts = vec3(0.0);
        const int GHOST_COUNT = 4;
        const float ghostScales[4] = float[4](-0.8, 0.4, -0.3, 0.7);
        const vec3 ghostColors[4] = vec3[4](
            vec3(0.2, 0.6, 1.0),  // Cyan
            vec3(0.9, 0.4, 0.2),  // Amber
            vec3(0.4, 0.9, 0.3),  // Emerald
            vec3(0.3, 0.5, 0.95)  // Deep Blue
        );

        for (int i = 0; i < GHOST_COUNT; ++i) {
            vec2 ghostUV = v_UV + toCenter * ghostScales[i];
            // Edge falloff for ghost discs
            float distToEdge = distance(ghostUV, opticalCenter);
            float ghostFalloff = smoothstep(0.75, 0.2, distToEdge);

            vec3 ghostSample = sampleThreshold(ghostUV);
            ghosts += ghostSample * ghostFalloff * ghostColors[i];
        }

        totalFlare += ghosts * u_GhostIntensity;
    }

    // -------------------------------------------------------------------------
    // 3. Lens Ring Halo
    // -------------------------------------------------------------------------
    if (u_HaloIntensity > 0.001) {
        vec2 opticalCenter = vec2(0.5, 0.5);
        vec2 haloVec = opticalCenter - v_UV;
        float haloDist = length(haloVec);
        vec2 haloUV = v_UV + normalize(haloVec) * 0.45;

        // Thin circular ring profile
        float ring = smoothstep(0.05, 0.0, abs(haloDist - 0.45));
        vec3 haloSample = sampleThreshold(haloUV);

        totalFlare += haloSample * ring * u_HaloIntensity * vec3(0.4, 0.75, 1.0);
    }

    out_FlareColor = vec4(totalFlare, 1.0);
}
