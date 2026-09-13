#version 450 core

layout(location = 0) in vec3 in_position;

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

// Water Mesh & Simulation Uniforms
uniform mat4  u_Model;
uniform float u_WaterHeight;
uniform float u_WaveAmplitude;
uniform float u_WaveSpeed;
uniform float u_WaveSteepness;
uniform float u_Time;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec3 v_Tangent;
out vec4 v_ClipPos;
out vec2 v_UV;
out float v_WaveCrest;

// -----------------------------------------------------------------------------
// 4-Octave Gerstner Wave Displacement & Analytic Derivative Evaluation
// -----------------------------------------------------------------------------
struct WaveOctave {
    vec2  direction;
    float wavelength;
    float amplitude;
    float speed;
};

void evaluateGerstnerWaves(
    vec2 xz0,
    float time,
    float ampScale,
    float speedScale,
    float steepness,
    out vec3 displacedPos,
    out vec3 normal,
    out vec3 tangent,
    out float crestFactor
) {
    // 4 parameterized wave octaves (direction, wavelength, base amp, speed)
    WaveOctave octaves[4];
    octaves[0] = WaveOctave(normalize(vec2(1.0, 0.4)),  18.0, 0.35, 1.1);
    octaves[1] = WaveOctave(normalize(vec2(-0.6, 0.8)), 10.0, 0.20, 1.3);
    octaves[2] = WaveOctave(normalize(vec2(0.2, -1.0)),  4.5, 0.10, 1.8);
    octaves[3] = WaveOctave(normalize(vec2(-0.8, -0.5)), 2.2, 0.05, 2.4);

    vec3 disp = vec3(0.0);
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 T = vec3(1.0, 0.0, 0.0);

    float sumQ = 0.0;
    float peakCrest = 0.0;

    for (int i = 0; i < 4; ++i) {
        float A = octaves[i].amplitude * ampScale;
        if (A <= 1e-4) continue;

        vec2 D = octaves[i].direction;
        float L = octaves[i].wavelength;
        float k = 6.28318530718 / max(L, 0.1); // 2 * pi / wavelength
        float w = sqrt(9.81 * k);              // dispersion relation for deep water
        float phase = octaves[i].speed * speedScale * w * time;
        float theta = k * dot(D, xz0) + phase;

        // Crest steepness (Q): prevent self-intersection loops
        float Q = (steepness * 0.75) / (max(k * A * 4.0, 0.001) + 1.0);
        sumQ += Q;

        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        // Trochoidal horizontal displacement & vertical elevation
        disp.x += Q * A * D.x * cosTheta;
        disp.y += A * sinTheta;
        disp.z += Q * A * D.y * cosTheta;

        // Derivatives for normal and tangent
        float WA = k * A;
        N.x -= D.x * WA * cosTheta;
        N.z -= D.y * WA * cosTheta;
        N.y -= Q * WA * sinTheta;

        T.x -= Q * D.x * D.x * WA * sinTheta;
        T.y += D.x * WA * cosTheta;
        T.z -= Q * D.x * D.y * WA * sinTheta;

        // Wave crest concentration metric for foam
        peakCrest += max(sinTheta, 0.0) * (A / max(octaves[0].amplitude * ampScale, 0.001));
    }

    displacedPos = vec3(xz0.x + disp.x, u_WaterHeight + disp.y, xz0.y + disp.z);
    normal = normalize(N);
    tangent = normalize(T);
    crestFactor = clamp(peakCrest * 0.5, 0.0, 1.0);
}

void main() {
    v_UV = in_position.xz;

    // Evaluate world position starting from model transform
    vec4 worldOrigin = u_Model * vec4(in_position, 1.0);
    vec2 xz0 = worldOrigin.xz;

    vec3 worldPos;
    vec3 normal;
    vec3 tangent;
    float crest;

    evaluateGerstnerWaves(
        xz0,
        u_Time,
        u_WaveAmplitude,
        u_WaveSpeed,
        u_WaveSteepness,
        worldPos,
        normal,
        tangent,
        crest
    );

    v_WorldPos = worldPos;
    v_Normal = normal;
    v_Tangent = tangent;
    v_WaveCrest = crest;

    v_ClipPos = u_ViewProjection * vec4(worldPos, 1.0);
    gl_Position = v_ClipPos;
}
