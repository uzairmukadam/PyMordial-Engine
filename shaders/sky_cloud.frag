#version 450 core

in vec2 v_UV;
out vec4 out_Cloud;

layout (binding = 0) uniform sampler2D u_DepthTexture;
layout (binding = 1) uniform sampler2D u_CloudNoise;

uniform mat4 u_InvProjection;
uniform mat4 u_InvView;
uniform vec3 u_CameraPosition;
uniform vec3 u_SunDirection;
uniform vec3 u_SunColor;
uniform vec3 u_MoonColor;
uniform float u_MoonIntensity;
uniform float u_Time;
uniform float u_CloudCoverage;
uniform float u_CloudOpacity;
uniform float u_CloudAltitude;
uniform float u_CloudScale;
uniform float u_CloudLayerSeparation;
uniform float u_CloudSecondaryOpacity;
uniform vec2 u_CloudWind;

void SampleCloudLayer(vec3 rayDir, float altitude, float worldPeriod, out vec4 layerNoise, out float shape) {
    float layerDistance = (altitude - u_CameraPosition.y) / rayDir.y;
    vec2 layerPos = u_CameraPosition.xz + rayDir.xz * layerDistance;
    vec2 uv = layerPos / worldPeriod + u_CloudWind * (u_Time / worldPeriod);
    vec4 initial = texture(u_CloudNoise, uv);
    vec2 warp = (initial.ba - 0.5) * 0.24;
    vec2 warpUv = uv + warp;
    vec4 primary = texture(u_CloudNoise, warpUv);

    const mat2 rotation = mat2(0.819152, -0.573576, 0.573576, 0.819152);
    vec2 transformedUv = rotation * uv * 1.371 + vec2(0.319, 0.173);
    vec4 secondary = texture(u_CloudNoise, transformedUv);
    float erosion = secondary.g;

    shape = mix(primary.r, secondary.r, 0.27);
    shape += (secondary.b - 0.5) * 0.10;
    shape += (initial.r - 0.5) * 0.08;
    shape -= max(0.0, erosion - 0.63) * 0.16;
    layerNoise = vec4(primary.r, erosion, secondary.r, initial.g);
}

vec2 EvaluateCloudDensity(float shape, float cellular, float coverage, float erosion) {
    float threshold = mix(0.76, 0.39, clamp(coverage, 0.0, 1.0));
    float broad = smoothstep(threshold - 0.075, threshold + 0.075, shape);
    float cellShape = smoothstep(0.12, 0.77, cellular);
    float density = mix(broad, broad * (0.45 + 0.55 * cellShape), 0.30);
    float edge = 1.0 - smoothstep(0.02, 0.23, abs(density - 0.5));
    density = clamp(density - max(0.0, erosion - 0.70) * 0.20, 0.0, 1.0);
    return vec2(density, edge);
}

void main() {
    if (texture(u_DepthTexture, v_UV).r > 0.000001) {
        discard;
    }

    vec3 rayNdc = vec3(v_UV * 2.0 - 1.0, 1.0);
    vec4 rayView = u_InvProjection * vec4(rayNdc, 1.0);
    rayView.z = -1.0;
    rayView.w = 0.0;
    vec3 rayDir = normalize((u_InvView * rayView).xyz);
    if (rayDir.y <= 0.0) {
        discard;
    }

    float tileWorldSize = max(u_CloudScale, 1000.0) * 8.0;
    vec4 lowerNoise;
    vec2 lowerDensity = vec2(0.0);
    float lowerAlpha = 0.0;
    if (u_CameraPosition.y < u_CloudAltitude) {
        float lowerShape;
        SampleCloudLayer(
            rayDir,
            u_CloudAltitude,
            tileWorldSize,
            lowerNoise,
            lowerShape
        );
        lowerDensity = EvaluateCloudDensity(lowerShape, lowerNoise.y, u_CloudCoverage, lowerNoise.w);
        lowerAlpha = lowerDensity.x * clamp(u_CloudOpacity, 0.0, 1.0);
    } else {
        lowerNoise = vec4(0.0);
    }

    float upperAltitude = u_CloudAltitude + max(u_CloudLayerSeparation, 1000.0);
    float upperAlpha = 0.0;
    vec4 upperNoise = vec4(0.0);
    vec2 upperDensity = vec2(0.0);
    if (u_CameraPosition.y < upperAltitude) {
        float upperShape;
        SampleCloudLayer(
            rayDir,
            upperAltitude,
            tileWorldSize * 1.83,
            upperNoise,
            upperShape
        );
        upperDensity = EvaluateCloudDensity(
            upperShape,
            upperNoise.y,
            u_CloudCoverage * 0.52,
            upperNoise.w
        );
        upperAlpha = upperDensity.x * clamp(u_CloudSecondaryOpacity, 0.0, 0.8);
    }
    float alpha = 1.0 - (1.0 - lowerAlpha) * (1.0 - upperAlpha);
    if (alpha < 0.005) {
        discard;
    }

    vec3 sunDirection = normalize(u_SunDirection);
    float sunElevation = sunDirection.y;
    float daylight = smoothstep(-0.10, 0.18, sunElevation);
    float twilight = (1.0 - smoothstep(0.02, 0.30, sunElevation))
        * smoothstep(-0.16, -0.02, sunElevation);
    float horizon = 1.0 - smoothstep(0.015, 0.28, rayDir.y);
    float viewSun = max(dot(-rayDir, sunDirection), 0.0);
    float forwardScatter = pow(viewSun, 10.0) * daylight;

    vec3 dayTint = mix(vec3(0.76, 0.81, 0.91), vec3(0.97, 0.985, 1.0), daylight);
    vec3 nightTint = u_MoonColor * (0.08 + 0.15 * u_MoonIntensity);
    vec3 horizonTint = mix(dayTint, u_SunColor, twilight * 0.85);
    vec3 baseTint = mix(nightTint, horizonTint, daylight);
    vec3 lowerTint = baseTint * (0.48 + 0.52 * lowerNoise.x);
    lowerTint += u_SunColor * (forwardScatter * 0.55 + lowerDensity.y * daylight * 0.18);
    lowerTint = mix(lowerTint, horizonTint, horizon * 0.40);
    vec3 upperTint = baseTint * (0.72 + 0.28 * upperNoise.x);
    upperTint += u_SunColor * forwardScatter * 0.25;
    vec3 cloudColor = (lowerTint * lowerAlpha * (1.0 - upperAlpha)
        + upperTint * upperAlpha) / max(alpha, 1e-4);

    out_Cloud = vec4(cloudColor, alpha);
}
