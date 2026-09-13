#version 450 core

#include "common/frame_data.glsl"

in vec2 v_UV;
layout(location = 0) out vec4 out_HDRColor;

layout(binding = 0) uniform sampler2D u_SceneHDR;
layout(binding = 1) uniform sampler2D u_DepthTexture; // Reversed-Z 32F

// Depth of Field Uniforms
uniform int   u_DoFEnabled;            // 0 = Off, 1 = On
uniform float u_FocusDistance;         // In-focus distance in meters (e.g. 5.0)
uniform float u_FocalLength;          // Lens focal length in mm (e.g. 50.0)
uniform float u_ApertureFStop;         // Aperture f-number (e.g. 1.4, 2.8)
uniform float u_MaxCoCRadius;          // Maximum blur radius in pixels (e.g. 24.0)
uniform int   u_BokehShape;            // 0 = Circular, 1 = Hexagonal, 2 = Anamorphic
uniform float u_AnamorphicRatio;       // Horizontal stretch ratio (e.g. 2.0 for anamorphic)

// Motion Blur Uniforms
uniform int   u_MotionBlurEnabled;     // 0 = Off, 1 = On
uniform int   u_MotionBlurSamples;     // Number of directional blur taps (e.g. 8 to 24)
uniform float u_MotionBlurIntensity;   // Shutter speed / blur scale factor
uniform float u_MaxMotionRadius;       // Max motion vector in pixels
uniform mat4  u_PrevViewProjection;    // Previous frame VP matrix

// -----------------------------------------------------------------------------
// Golden Spiral Disc Kernel Generator for Bokeh Sampling
// -----------------------------------------------------------------------------
const int BOKEH_SAMPLES = 20;

vec2 getBokehOffset(int index, int total, float shapeModifier, int shapeType) {
    float theta = float(index) * 2.39996323; // Golden angle (approx 137.5 deg)
    float r = sqrt(float(index) + 0.5) / sqrt(float(total));

    vec2 offset = vec2(cos(theta), sin(theta)) * r;

    if (shapeType == 1) {
        // Hexagonal bokeh diaphragm: fold angles into 60 degree sectors
        float hexAngle = mod(theta, 1.04719755) - 0.52359877;
        float hexScale = cos(0.52359877) / cos(hexAngle);
        offset *= hexScale;
    } else if (shapeType == 2) {
        // Anamorphic 2x horizontal stretch
        offset.x *= shapeModifier;
    }

    return offset;
}

float linearizeDepth(float rawDepth, vec2 uv) {
    if (rawDepth <= 1e-6) {
        return 2000.0; // Sky far background
    }
    vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
    vec4 viewPos = u_InvProjection * clip;
    return max(-viewPos.z / max(viewPos.w, 1e-6), 0.05);
}

float getLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 texelSize = 1.0 / u_ScreenSize_Jitter.xy;
    float centerRawDepth = texture(u_DepthTexture, v_UV).r;
    float centerDepth = linearizeDepth(centerRawDepth, v_UV);
    vec3 baseColor = texture(u_SceneHDR, v_UV).rgb;

    vec3 dofColor = baseColor;

    // -------------------------------------------------------------------------
    // 1. Physical Bokeh Depth of Field
    // -------------------------------------------------------------------------
    if (u_DoFEnabled == 1) {
        // Physical thin-lens circle of confusion formula
        float F = u_FocalLength * 0.001;               // mm to meters
        float P = max(u_FocusDistance, F + 0.01);       // Focus distance in meters
        float N = max(u_ApertureFStop, 0.5);            // f-stop

        float D = max(centerDepth, F + 0.01);           // Object depth in meters
        float cocWorld = abs((F * F / (N * (P - F))) * ((D - P) / D));

        // Project world CoC to pixel radius on the sensor (assume 35mm full-frame sensor height = 24mm)
        float sensorHeight = 0.024;
        float cocPixels = (cocWorld / sensorHeight) * u_ScreenSize_Jitter.y;
        cocPixels = clamp(cocPixels, 0.0, u_MaxCoCRadius);

        if (cocPixels > 0.6) {
            vec3 accumColor = vec3(0.0);
            float accumWeight = 0.0;

            for (int i = 0; i < BOKEH_SAMPLES; ++i) {
                vec2 sampleOffset = getBokehOffset(i, BOKEH_SAMPLES, u_AnamorphicRatio, u_BokehShape);
                vec2 sampleUV = v_UV + sampleOffset * (cocPixels * texelSize);

                vec3 tapColor = texture(u_SceneHDR, sampleUV).rgb;
                float tapRawDepth = texture(u_DepthTexture, sampleUV).r;
                float tapDepth = linearizeDepth(tapRawDepth, sampleUV);

                // Depth weighting to prevent background bleeding onto sharp foreground
                float depthWeight = 1.0;
                if (tapDepth < centerDepth - 0.2) {
                    depthWeight = clamp((tapDepth - (centerDepth - 0.6)) / 0.4, 0.1, 1.0);
                }

                // Optical HDR highlight boost for crisp bokeh disc definition
                float lum = getLuminance(tapColor);
                float highlightWeight = 1.0 + max(0.0, lum - 1.0) * 1.5;

                float totalTapWeight = depthWeight * highlightWeight;
                accumColor += tapColor * totalTapWeight;
                accumWeight += totalTapWeight;
            }

            dofColor = accumColor / max(accumWeight, 1e-4);
        }
    }

    vec3 finalOpticsColor = dofColor;

    // -------------------------------------------------------------------------
    // 2. Camera Velocity Motion Blur
    // -------------------------------------------------------------------------
    if (u_MotionBlurEnabled == 1 && centerRawDepth > 1e-6) {
        // Reconstruct current 3D world position from depth
        vec4 clipPos = vec4(v_UV * 2.0 - 1.0, centerRawDepth, 1.0);
        vec4 viewPos = u_InvProjection * clipPos;
        viewPos /= max(viewPos.w, 1e-6);
        vec4 worldPos = u_InvView * viewPos;

        // Project world position with previous frame's View-Projection matrix
        vec4 prevClipPos = u_PrevViewProjection * vec4(worldPos.xyz, 1.0);
        vec2 prevUV = (prevClipPos.xy / max(prevClipPos.w, 1e-6)) * 0.5 + 0.5;

        // Screen-space motion vector
        vec2 motionVec = (v_UV - prevUV) * u_MotionBlurIntensity;

        // Clamp maximum motion length in pixel space
        float motionLen = length(motionVec / texelSize);
        if (motionLen > u_MaxMotionRadius) {
            motionVec *= (u_MaxMotionRadius / max(motionLen, 1e-4));
        }

        // Only blur if there is noticeable subpixel motion
        if (motionLen > 0.75) {
            int samples = clamp(u_MotionBlurSamples, 4, 32);
            vec3 mbAccum = vec3(0.0);
            float mbWeight = 0.0;

            for (int i = 0; i < samples; ++i) {
                float t = float(i) / float(samples - 1) - 0.5; // [-0.5, 0.5] centered
                vec2 tapUV = v_UV + motionVec * t;

                vec3 tapCol = texture(u_SceneHDR, tapUV).rgb;
                float tapRawDepth = texture(u_DepthTexture, tapUV).r;
                float tapDepth = linearizeDepth(tapRawDepth, tapUV);

                // Depth-aware weighting to prevent bleeding behind foreground geometry
                float dw = clamp(1.0 - abs(tapDepth - centerDepth) / max(centerDepth * 0.25, 0.5), 0.1, 1.0);

                mbAccum += tapCol * dw;
                mbWeight += dw;
            }

            finalOpticsColor = mbAccum / max(mbWeight, 1e-4);
        }
    }

    out_HDRColor = vec4(finalOpticsColor, 1.0);
}
