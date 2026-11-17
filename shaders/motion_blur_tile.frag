#version 450 core

// Motion Blur Pass 1: per-tile maximum velocity (McGuire et al. 2012).
// Each fragment of the small tile texture scans its tile footprint in the
// G-buffer velocity buffer and writes the largest velocity vector found.

in vec2 v_UV;
layout(location = 0) out vec4 out_TileMax;

layout(binding = 0) uniform sampler2D u_VelocityTexture;

uniform ivec2 u_TileSize;    // tile footprint in pixels
uniform vec2  u_ScreenSize;  // full-resolution screen size in pixels

void main() {
    ivec2 tile = ivec2(gl_FragCoord.xy);
    ivec2 base = tile * u_TileSize;
    ivec2 screen = ivec2(u_ScreenSize);

    vec2  maxVel  = vec2(0.0);
    float maxMag2 = 0.0;

    for (int y = 0; y < u_TileSize.y; ++y) {
        int py = base.y + y;
        if (py >= screen.y) { break; }
        for (int x = 0; x < u_TileSize.x; ++x) {
            int px = base.x + x;
            if (px >= screen.x) { break; }
            vec2 uv = (vec2(float(px), float(py)) + 0.5) / u_ScreenSize;
            vec2 v = texture(u_VelocityTexture, uv).xy;
            float m2 = dot(v, v);
            if (m2 > maxMag2) {
                maxMag2 = m2;
                maxVel = v;
            }
        }
    }

    // rg = max velocity vector (UV space), b = max magnitude (UV space)
    out_TileMax = vec4(maxVel, sqrt(maxMag2), 1.0);
}
