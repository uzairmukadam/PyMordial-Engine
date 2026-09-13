#version 450 core

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec3 v_Tangent;
in vec4 v_ClipPos;
in vec2 v_UV;
in float v_WaveCrest;

layout(location = 0) out vec4 out_HDRColor;

layout(binding = 0) uniform sampler2D u_OpaqueSceneColor;
layout(binding = 1) uniform sampler2D u_DepthTexture; // Reversed-Z 32F

layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;

    vec4 u_CameraPos_Time;         // xyz = camera world pos, w = total elapsed time
    vec4 u_ScreenSize_Jitter;      // xy = width/height, zw = subpixel jitter (TAA)

    vec4 u_SunDirection_Intensity; // xyz = normalized sun dir, w = sun lux
    vec4 u_SunColor_Ambient;        // rgb = sun light color, w = ambient factor

    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;

    vec4 u_FogColor_Density;
    vec4 u_FogParams;
};

// Water Optical & Aesthetic Uniforms
uniform vec3  u_WaterColorShallow;     // Turquoise / emerald shallow color
uniform vec3  u_WaterColorDeep;        // Navy / abyss deep water color
uniform vec3  u_ExtinctionCoeff;       // Beer-Lambert absorption per meter (R, G, B)
uniform float u_RefractionStrength;    // Screen-space refraction offset magnitude
uniform float u_Roughness;             // Surface microfacet roughness (GGX)
uniform vec3  u_FoamColor;             // Shoreline / crest foam color
uniform float u_FoamThreshold;         // Optical depth threshold for shoreline foam (m)
uniform float u_FoamScale;             // Spatial frequency for foam noise
uniform float u_FoamIntensity;         // Contact & crest foam opacity multiplier
uniform float u_WaterClarity;          // Overall clarity / light penetration scale
uniform float u_Time;
uniform int   u_RefractionEnabled;     // 0 = Off, 1 = On
uniform int   u_FoamEnabled;           // 0 = Off, 1 = On

const float PI = 3.14159265359;

// -----------------------------------------------------------------------------
// Linear View-Space Depth Reconstruction from Reversed-Z
// -----------------------------------------------------------------------------
float linearizeDepth(float rawDepth, vec2 uv) {
    if (rawDepth <= 1e-6) {
        return 2000.0; // Sky / infinite background
    }
    vec4 clip = vec4(uv * 2.0 - 1.0, rawDepth, 1.0);
    vec4 viewPos = u_InvProjection * clip;
    return max(-viewPos.z / max(viewPos.w, 1e-6), 0.05);
}

// -----------------------------------------------------------------------------
// Procedural High-Frequency Micro-Ripple Perturbation
// -----------------------------------------------------------------------------
vec3 getMicroRipples(vec2 worldXZ, float time) {
    vec2 p1 = worldXZ * 1.8 + vec2(time * 0.45, time * 0.35);
    vec2 p2 = worldXZ * 3.5 - vec2(time * 0.60, -time * 0.40);

    float h1 = sin(p1.x * 2.0 + p1.y * 1.5) * cos(p1.y * 2.2 - p1.x * 0.8);
    float h2 = sin(p2.x * 3.1 - p2.y * 2.4) * cos(p2.y * 1.8 + p2.x * 2.6);

    float dhdx = 0.04 * (2.0 * cos(p1.x * 2.0 + p1.y * 1.5) + 3.1 * cos(p2.x * 3.1 - p2.y * 2.4));
    float dhdz = 0.04 * (1.5 * cos(p1.x * 2.0 + p1.y * 1.5) - 2.4 * cos(p2.x * 3.1 - p2.y * 2.4));

    return normalize(vec3(-dhdx, 1.0, -dhdz));
}

// -----------------------------------------------------------------------------
// Procedural Animated Foam Noise
// -----------------------------------------------------------------------------
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbmFoam(vec2 p) {
    float v = 0.0;
    float a = 0.55;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 3; ++i) {
        v += a * noise2D(p);
        p = rot * p * 2.05 + vec2(0.15, 0.22);
        a *= 0.5;
    }
    return v;
}

// -----------------------------------------------------------------------------
// Cook-Torrance Microfacet Specular GGX
// -----------------------------------------------------------------------------
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom + 1e-6);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float ggx1 = NdotV / (NdotV * (1.0 - k) + k + 1e-6);
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k + 1e-6);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// -----------------------------------------------------------------------------
// Main Water Fragment Shading
// -----------------------------------------------------------------------------
void main() {
    vec2 screenUV = gl_FragCoord.xy / vec2(textureSize(u_DepthTexture, 0));
    screenUV = clamp(screenUV, vec2(0.0001), vec2(0.9999));

    // 1. Exact Reversed-Z Depth Testing (Near=1.0, Far=0.0)
    float rawFloorDepth = texture(u_DepthTexture, screenUV).r;
    float rawWaterDepth = gl_FragCoord.z;

    // If opaque scene geometry is closer to camera than water surface (larger raw depth in Reversed-Z), discard
    if (rawFloorDepth > rawWaterDepth + 0.00005) {
        discard;
    }

    vec3 camPos = u_CameraPos_Time.xyz;
    float floorDepthView = linearizeDepth(rawFloorDepth, screenUV);
    float waterDepthView = linearizeDepth(rawWaterDepth, screenUV);

    // Optical path thickness of water column (in meters)
    float opticalWaterDepth = max(floorDepthView - waterDepthView, 0.0);

    // 2. Normal Perturbation (Gerstner Normal + High-Frequency Micro-Ripples)
    vec3 baseN = normalize(v_Normal);
    vec3 rippleN = getMicroRipples(v_WorldPos.xz, u_Time);
    // Blend geometric wave normal with micro-ripples
    vec3 N = normalize(vec3(baseN.x + rippleN.x * 0.35, baseN.y, baseN.z + rippleN.z * 0.35));

    vec3 V = normalize(camPos - v_WorldPos);
    vec3 L = normalize(u_SunDirection_Intensity.xyz);
    vec3 H = normalize(V + L);

    float NdotV = max(dot(N, V), 0.001);
    float NdotL = max(dot(N, L), 0.0);

    // 3. Screen-Space Refraction with Depth Disparity Protection
    vec2 refractUV = screenUV;
    if (u_RefractionEnabled == 1) {
        float depthDampener = clamp(opticalWaterDepth * 0.8, 0.05, 1.0);
        vec2 uvOffset = N.xz * u_RefractionStrength * depthDampener;
        vec2 testUV = clamp(screenUV + uvOffset, vec2(0.001), vec2(0.999));

        float candidateRawDepth = texture(u_DepthTexture, testUV).r;
        // In Reversed-Z, candidate depth behind water means candidateRawDepth <= rawWaterDepth + 0.0001
        if (candidateRawDepth <= rawWaterDepth + 0.0001) {
            refractUV = testUV;
            float candidateDepthView = linearizeDepth(candidateRawDepth, testUV);
            opticalWaterDepth = max(candidateDepthView - waterDepthView, 0.0);
        }
    }

    vec3 floorColor = texture(u_OpaqueSceneColor, refractUV).rgb;

    // 4. Beer-Lambert Volume Absorption & Depth Extinction
    // Red light is absorbed within 1-2m, green penetrates deeper, blue deepest
    vec3 extinction = u_ExtinctionCoeff / max(u_WaterClarity, 0.1);
    vec3 transmittance = exp(-extinction * opticalWaterDepth);

    // Rich shallow-to-deep water gradient
    float depthT = clamp(exp(-opticalWaterDepth * 0.35), 0.0, 1.0);
    vec3 waterBodyColor = mix(u_WaterColorDeep, u_WaterColorShallow, depthT);
    // Subtle shallow absorption blend so shallow water has a tropical turquoise tint
    float absorptionFactor = clamp(1.0 - (transmittance.r * 0.3 + transmittance.g * 0.5 + transmittance.b * 0.2), 0.12, 1.0);
    vec3 refractedWaterColor = mix(floorColor * transmittance, waterBodyColor, absorptionFactor);

    // 5. Fresnel Reflectance (Water IOR = 1.333 -> F0 = 0.02)
    vec3 F0 = vec3(0.02);
    // Specular Fresnel (half-angle)
    vec3 F_spec = fresnelSchlick(max(dot(H, V), 0.0), F0);
    // Environment Reflection Fresnel (view-angle)
    vec3 F_env = fresnelSchlick(NdotV, F0);
    vec3 F_surface = clamp(F_env, vec3(0.05), vec3(0.98));

    // 6. Cook-Torrance Sun Specular Glints
    float rough = max(u_Roughness, 0.04);
    float D = distributionGGX(N, H, rough);
    float G = geometrySmith(N, V, L, rough);
    vec3 numerator = D * G * F_spec;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specularGlint = (numerator / denominator) * u_SunColor_Ambient.rgb * u_SunDirection_Intensity.w * NdotL;

    // 7. Sky & Ambient Reflection Approximation
    vec3 R = reflect(-V, N);
    float skyZenithFactor = clamp(R.y, 0.0, 1.0);
    vec3 skyHorizonCol = u_SunColor_Ambient.rgb * 0.4 + vec3(0.40, 0.55, 0.70) * 0.6;
    vec3 skyZenithCol = vec3(0.12, 0.25, 0.55) * (u_SunColor_Ambient.w + 0.2);
    vec3 skyReflection = mix(skyHorizonCol, skyZenithCol, pow(skyZenithFactor, 0.6));

    // Sun disc atmospheric specular in reflection
    float RdotL = max(dot(R, L), 0.0);
    skyReflection += u_SunColor_Ambient.rgb * pow(RdotL, 64.0) * 1.5;

    // Composite surface reflection with refracted underwater light via Fresnel
    vec3 surfaceColor = mix(refractedWaterColor, skyReflection, F_surface) + specularGlint;

    // 8. Shoreline & Wave Crest Contact Foam
    if (u_FoamEnabled == 1) {
        // Shoreline contact factor (strongest at boundary where depth -> 0)
        float shoreFactor = clamp(1.0 - opticalWaterDepth / max(u_FoamThreshold, 0.01), 0.0, 1.0);
        shoreFactor = pow(shoreFactor, 1.5);

        // Animated noise texture coordinates
        vec2 foamCoord = v_WorldPos.xz * u_FoamScale + vec2(u_Time * 0.12, u_Time * 0.08);
        float foamNoise = fbmFoam(foamCoord);

        // Wave crest foam from vertex Gerstner evaluation
        float crestFoam = smoothstep(0.65, 0.95, v_WaveCrest) * 0.85;

        // Composite foam mask
        float totalFoam = clamp((shoreFactor * 1.2 + crestFoam) * foamNoise * 1.5 * u_FoamIntensity, 0.0, 1.0);
        float foamEdge = smoothstep(0.25, 0.65, totalFoam);

        surfaceColor = mix(surfaceColor, u_FoamColor * (u_SunColor_Ambient.rgb * 0.5 + 0.5), foamEdge);
    }

    out_HDRColor = vec4(surfaceColor, 1.0);
}
