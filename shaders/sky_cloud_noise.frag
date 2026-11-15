#version 450 core

in vec2 v_UV;
out vec4 out_Noise;

float HashNoise(ivec2 cell, int period) {
    ivec2 wrapped = ivec2(
        (cell.x % period + period) % period,
        (cell.y % period + period) % period
    );
    return fract(sin(dot(vec2(wrapped), vec2(127.1, 311.7))) * 43758.5453);
}

float ValueNoise(vec2 uv, int grid) {
    vec2 p = uv * float(grid);
    ivec2 cell = ivec2(floor(p));
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = HashNoise(cell, grid);
    float b = HashNoise(cell + ivec2(1, 0), grid);
    float c = HashNoise(cell + ivec2(0, 1), grid);
    float d = HashNoise(cell + ivec2(1, 1), grid);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float WorleyNoise(vec2 uv, int grid) {
    vec2 p = uv * float(grid);
    ivec2 cell = ivec2(floor(p));
    float nearest = 10.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 neighbor = cell + ivec2(x, y);
            vec2 feature = vec2(
                HashNoise(neighbor, grid),
                HashNoise(neighbor + ivec2(37, 17), grid)
            );
            vec2 delta = vec2(x, y) + feature - fract(p);
            nearest = min(nearest, dot(delta, delta));
        }
    }
    return sqrt(nearest) * 0.72;
}

void main() {
    float base = ValueNoise(v_UV, 5) * 0.32
        + ValueNoise(v_UV, 9) * 0.24
        + ValueNoise(v_UV, 17) * 0.17
        + ValueNoise(v_UV, 33) * 0.12
        + ValueNoise(v_UV, 65) * 0.08
        + ValueNoise(v_UV, 129) * 0.05
        + ValueNoise(v_UV, 257) * 0.02;
    float cellular = WorleyNoise(v_UV, 17) * 0.52
        + WorleyNoise(v_UV, 37) * 0.31
        + WorleyNoise(v_UV, 73) * 0.17;
    float warp_x = ValueNoise(v_UV + vec2(0.193, 0.317), 3) * 0.62
        + ValueNoise(v_UV + vec2(0.417, 0.113), 7) * 0.38;
    float warp_y = ValueNoise(v_UV + vec2(0.271, 0.491), 4) * 0.58
        + ValueNoise(v_UV + vec2(0.071, 0.239), 9) * 0.42;
    out_Noise = vec4(base, cellular, warp_x, warp_y);
}
