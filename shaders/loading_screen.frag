#version 450 core

in vec2 v_UV;
out vec4 fragColor;

uniform vec2 u_Resolution;
uniform float u_Time;
uniform float u_Progress;
uniform float u_FadeAlpha;
uniform sampler2D u_TextTexture;

// Rounded box SDF
float sdRoundedBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    float aspect = u_Resolution.x / max(u_Resolution.y, 1.0);
    vec2 uv_aspect = vec2(v_UV.x * aspect, v_UV.y);

    // 1. Deep cinematic glassmorphic background with vignette
    float dist_center = length(v_UV - vec2(0.5, 0.5));
    float vignette = smoothstep(0.95, 0.20, dist_center);
    vec3 col_center = vec3(0.045, 0.075, 0.125);
    vec3 col_edge = vec3(0.012, 0.018, 0.028);
    vec3 bg = mix(col_edge, col_center, vignette);

    // Subtle undulating ambient wave patterns
    float wave1 = sin(v_UV.x * 12.0 + u_Time * 1.2) * cos(v_UV.y * 10.0 - u_Time * 0.8) * 0.02;
    float wave2 = sin(v_UV.x * 24.0 - u_Time * 0.6 + v_UV.y * 16.0) * 0.015;
    bg += vec3(0.02, 0.08, 0.15) * max(0.0, wave1 + wave2);

    // Subtle background hex/dot energy grid
    vec2 grid_uv = v_UV * vec2(aspect * 35.0, 35.0);
    vec2 grid_cell = fract(grid_uv) - 0.5;
    float dot_grid = smoothstep(0.08, 0.02, length(grid_cell)) * 0.025;
    bg += vec3(0.2, 0.6, 1.0) * dot_grid;

    // 2. High-Tech Rotating Energy Rings at (center_x, center_y = 0.56)
    vec2 glyph_center = vec2(0.5 * aspect, 0.56);
    vec2 p_glyph = uv_aspect - glyph_center;
    float r_glyph = length(p_glyph);
    float angle = atan(p_glyph.y, p_glyph.x);

    // Outer counter-rotating ring
    float arc1 = 0.5 + 0.5 * sin(angle * 5.0 + u_Time * 2.2);
    float ring1 = smoothstep(0.005, 0.0, abs(r_glyph - 0.090)) * arc1;
    
    // Inner counter-rotating segmented ring
    float arc2 = 0.5 + 0.5 * cos(angle * 8.0 - u_Time * 3.0);
    float ring2 = smoothstep(0.004, 0.0, abs(r_glyph - 0.075)) * arc2;

    // Central energy pulse core
    float pulse = 0.5 + 0.5 * sin(u_Time * 4.0);
    float core = smoothstep(0.035, 0.005, r_glyph) * (0.6 + 0.4 * pulse);
    float core_inner = smoothstep(0.012, 0.002, r_glyph);

    vec3 cyan_glow = vec3(0.15, 0.65, 0.98);
    vec3 core_glow = vec3(0.40, 0.88, 1.0);
    vec3 jewel_glow = vec3(0.95, 0.98, 1.0);

    bg += ring1 * cyan_glow * 1.8;
    bg += ring2 * core_glow * 1.5;
    bg += core * cyan_glow * 2.2;
    bg += core_inner * jewel_glow * 2.0;

    // Soft radial halo around central glyph
    float halo = smoothstep(0.22, 0.0, r_glyph) * 0.18;
    bg += cyan_glow * halo;

    // 3. Glassmorphic Glowing Progress Bar at bottom
    // Track bounds in normalized UV
    float bar_x0 = 0.25;
    float bar_x1 = 0.75;
    float bar_w = bar_x1 - bar_x0;
    float bar_y = 0.26;
    float bar_h = 0.014;

    vec2 p_bar = vec2(v_UV.x - 0.5, v_UV.y - bar_y);
    float bar_d = sdRoundedBox(p_bar, vec2(bar_w * 0.5, bar_h * 0.5), bar_h * 0.5);

    if (bar_d < 0.01) {
        // Track background
        float track_alpha = smoothstep(0.001, -0.001, bar_d);
        vec3 track_col = vec3(0.06, 0.10, 0.16);
        
        // Track border highlight
        float border = smoothstep(0.002, 0.0, abs(bar_d));
        track_col = mix(track_col, vec3(0.18, 0.35, 0.55), border * 0.8);

        // Progress fill
        float norm_x = clamp((v_UV.x - bar_x0) / bar_w, 0.0, 1.0);
        float clamped_progress = clamp(u_Progress, 0.0, 1.0);

        if (norm_x <= clamped_progress && bar_d < 0.0) {
            // Fill gradient (deep electric cyan to glowing neon azure)
            vec3 fill_col = mix(vec3(0.12, 0.52, 0.95), vec3(0.25, 0.88, 1.0), norm_x);
            
            // Subtle traveling shimmer highlight
            float shimmer = sin(norm_x * 16.0 - u_Time * 5.0) * 0.12;
            fill_col += vec3(shimmer);

            // Intense glowing leading head
            float head_dist = abs(norm_x - clamped_progress);
            float head_glow = smoothstep(0.04, 0.0, head_dist);
            fill_col = mix(fill_col, vec3(1.0, 1.0, 1.0), head_glow * 0.75);

            track_col = fill_col;
        }

        // Bloom halo around the leading tip
        float head_x = bar_x0 + clamped_progress * bar_w;
        float d_head = length(vec2((v_UV.x - head_x) * aspect, v_UV.y - bar_y));
        float tip_bloom = smoothstep(0.04, 0.0, d_head) * 0.45;
        bg += cyan_glow * tip_bloom;

        bg = mix(bg, track_col, track_alpha);
    }

    // 4. Sample typography & status text overlay
    vec4 text_overlay = texture(u_TextTexture, v_UV);
    bg = mix(bg, text_overlay.rgb, text_overlay.a);

    // 5. Modulate overall alpha for smooth fade-in / fade-out
    fragColor = vec4(bg, clamp(u_FadeAlpha, 0.0, 1.0));
}
