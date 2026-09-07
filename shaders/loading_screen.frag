#version 450 core

in vec2 v_UV;
out vec4 fragColor;

uniform vec2 u_Resolution;
uniform float u_Time;
uniform float u_Progress;
uniform float u_FadeAlpha;
uniform sampler2D u_TextTexture;

void main() {
    // Pure cinematic black background
    vec3 bg = vec3(0.0);

    // Exact pixel coordinates across current window resolution
    vec2 fragCoord = v_UV * u_Resolution;

    // Scale icon appropriately with display height (optimized for 600p/720p up to 4K)
    float scale = clamp(u_Resolution.y / 1080.0, 0.70, 1.8);
    float margin_x = 56.0 * scale;
    float margin_y = 56.0 * scale;

    // Icon center positioned precisely in the bottom-right corner
    vec2 iconCenter = vec2(u_Resolution.x - margin_x, margin_y);
    vec2 p = fragCoord - iconCenter;
    float dist = length(p);

    // Radii in screen pixels
    float r_outer = 19.0 * scale;
    float r_inner = 15.0 * scale;
    float r_mid = (r_outer + r_inner) * 0.5;
    float ring_w = (r_outer - r_inner) * 0.5;

    // Render loading animation in the bottom-right corner region
    if (dist < 60.0 * scale) {
        float angle = atan(p.y, p.x); // -PI to +PI

        // 1. Faint circular background track
        float track = smoothstep(ring_w + 0.8, ring_w - 0.8, abs(dist - r_mid)) * 0.15;
        bg += vec3(0.35, 0.45, 0.55) * track;

        // 2. High-speed smooth rotating spinner arc
        float rot_angle = mod(angle - u_Time * 4.2, 6.2831853);
        // Tapered sweep of ~270 degrees
        float sweep = smoothstep(0.2, 4.2, rot_angle) * (1.0 - smoothstep(4.2, 4.8, rot_angle));
        float arc = smoothstep(ring_w + 1.0, ring_w - 1.0, abs(dist - r_mid)) * sweep;

        // Glowing leading tip of the spinner
        float tip = smoothstep(0.35, 0.0, abs(rot_angle - 4.5)) * smoothstep(ring_w + 1.5, 0.0, abs(dist - r_mid));

        vec3 cyan_accent = vec3(0.22, 0.72, 1.0);
        vec3 white_glow = vec3(1.0, 1.0, 1.0);

        bg += arc * cyan_accent * 1.9;
        bg += tip * white_glow * 2.2;

        // 3. Inner progress ring indicating actual asset load completion
        float r_prog = 10.0 * scale;
        float prog_w = 1.6 * scale;
        float prog_angle = clamp(u_Progress, 0.0, 1.0) * 6.2831853;
        float cur_angle = mod(1.5707963 - angle, 6.2831853);
        float in_prog = (cur_angle <= prog_angle) ? 1.0 : 0.0;
        float prog_ring = smoothstep(prog_w + 0.8, prog_w - 0.8, abs(dist - r_prog)) * in_prog;
        bg += prog_ring * cyan_accent * 1.3;

        // 4. Central pulsing energy core
        float pulse = 0.5 + 0.5 * sin(u_Time * 4.5);
        float core = smoothstep(3.5 * scale, 0.5 * scale, dist) * (0.6 + 0.4 * pulse);
        bg += core * white_glow * 1.6;

        // 5. Subtle ambient cyan bloom halo
        float halo = smoothstep(38.0 * scale, 6.0 * scale, dist) * 0.22;
        bg += cyan_accent * halo;
    }

    // 6. Sample crisp typography overlay
    vec4 text_overlay = texture(u_TextTexture, v_UV);
    bg = mix(bg, text_overlay.rgb, text_overlay.a);

    // 7. Modulate overall alpha for fade-out transitions
    fragColor = vec4(bg, clamp(u_FadeAlpha, 0.0, 1.0));
}
