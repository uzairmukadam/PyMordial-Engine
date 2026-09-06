#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_Color;

layout (binding = 0) uniform sampler2D u_CurrentSMAA;
layout (binding = 1) uniform sampler2D u_HistoryFrame;
layout (binding = 2) uniform sampler2D u_VelocityTexture;
layout (binding = 3) uniform sampler2D u_GBufferDepth;

uniform vec2 u_InverseScreenSize;
uniform float u_TemporalWeight = 0.50; // 0.50 for 2x, 0.75 for 4x

void main() {
    vec4 current = texture(u_CurrentSMAA, v_UV);

    // Fetch velocity
    vec2 velocity = texture(u_VelocityTexture, v_UV).xy;
    vec2 prev_uv = v_UV - velocity;

    // Disocclusion / boundary rejection
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        out_Color = current;
        return;
    }

    vec4 history = texture(u_HistoryFrame, prev_uv);

    // 3x3 Neighborhood bounding box in RGB to eliminate ghosting
    vec3 c_min = current.rgb;
    vec3 c_max = current.rgb;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec3 neighbor = texture(u_CurrentSMAA, v_UV + vec2(float(dx), float(dy)) * u_InverseScreenSize).rgb;
            c_min = min(c_min, neighbor);
            c_max = max(c_max, neighbor);
        }
    }

    // Clamp history into local neighborhood
    vec3 history_clamped = clamp(history.rgb, c_min, c_max);

    // Motion velocity damping: drop temporal weight during rapid motion
    float speed = length(velocity / max(u_InverseScreenSize, vec2(1e-5)));
    float weight = clamp(u_TemporalWeight - speed * 0.05, 0.10, u_TemporalWeight);

    vec3 resolved = mix(current.rgb, history_clamped, weight);
    out_Color = vec4(resolved, 1.0);
}
