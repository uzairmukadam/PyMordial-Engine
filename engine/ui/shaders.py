"""Native ModernGL UI GLSL Shaders for PyMordial Engine.

Implements GPU-accelerated 2D screen-space UI rendering with:
- Analytic Signed Distance Field (SDF) rounded rectangles with antialiased borders
- 2-stop vertical linear gradients and ambient glows
- Antialiased font glyph and texture atlas sampling
"""

UI_VERT_GLSL = """#version 450 core

// Per-quad uniforms
uniform vec2 u_ScreenSize; // [screen_w, screen_h] in pixels

// Vertex attributes:
// in_Position: [x, y, w, h] in screen pixels (top-left origin)
// in_UV: [u, v, u_w, v_h]
// in_ColorTop: RGBA
// in_ColorBottom: RGBA
// in_BorderColor: RGBA
// in_Params: [corner_radius, border_width, texture_mode (0=flat/grad, 1=tex, 2=font), glow_radius]

in vec4 in_Position;
in vec4 in_UV;
in vec4 in_ColorTop;
in vec4 in_ColorBottom;
in vec4 in_BorderColor;
in vec4 in_Params;

out vec2 v_ScreenPos;
out vec2 v_LocalPos; // [0, 0] to [w, h]
out vec2 v_Size;     // [w, h]
out vec2 v_UV;
out vec4 v_ColorTop;
out vec4 v_ColorBottom;
out vec4 v_BorderColor;
out vec4 v_Params;

void main() {
    vec2 rect_pos = in_Position.xy;
    vec2 rect_size = in_Position.zw;
    vec2 uv_min = in_UV.xy;
    vec2 uv_size = in_UV.zw;

    // Generate quad vertices from gl_VertexID (0..3)
    vec2 corner = vec2(0.0);
    if (gl_VertexID == 0) corner = vec2(0.0, 0.0);
    else if (gl_VertexID == 1) corner = vec2(1.0, 0.0);
    else if (gl_VertexID == 2) corner = vec2(0.0, 1.0);
    else if (gl_VertexID == 3) corner = vec2(1.0, 1.0);

    vec2 pixel_pos = rect_pos + corner * rect_size;
    v_ScreenPos = pixel_pos;
    v_LocalPos = corner * rect_size;
    v_Size = rect_size;
    v_UV = uv_min + corner * uv_size;
    v_ColorTop = in_ColorTop;
    v_ColorBottom = in_ColorBottom;
    v_BorderColor = in_BorderColor;
    v_Params = in_Params;

    // Convert pixel coordinates (top-left 0,0) to NDC [-1, 1] (bottom-left -1,-1)
    vec2 ndc = (pixel_pos / u_ScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for OpenGL
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

UI_FRAG_GLSL = """#version 450 core

in vec2 v_ScreenPos;
in vec2 v_LocalPos;
in vec2 v_Size;
in vec2 v_UV;
in vec4 v_ColorTop;
in vec4 v_ColorBottom;
in vec4 v_BorderColor;
in vec4 v_Params;

out vec4 fragColor;

uniform sampler2D u_Texture;

// Analytic signed distance function to a rounded rectangle
float sdRoundedBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    float radius = min(v_Params.x, min(v_Size.x, v_Size.y) * 0.5);
    float border_width = v_Params.y;
    int tex_mode = int(v_Params.z + 0.5);
    float glow_radius = v_Params.w;

    // SDF evaluation centered in box
    vec2 half_size = v_Size * 0.5;
    vec2 p = v_LocalPos - half_size;
    float dist = sdRoundedBox(p, half_size, radius);

    // Antialiased edge factor
    float afwidth = fwidth(dist);
    float alpha = 1.0 - smoothstep(-afwidth, afwidth, dist);

    if (alpha <= 0.001) {
        discard;
    }

    // Vertical linear gradient interpolation
    float grad_t = clamp(v_LocalPos.y / max(1.0, v_Size.y), 0.0, 1.0);
    vec4 fill_color = mix(v_ColorTop, v_ColorBottom, grad_t);

    if (tex_mode == 1) {
        // Texture color modulated with fill color
        vec4 tex_sample = texture(u_Texture, v_UV);
        fill_color = tex_sample * fill_color;
    } else if (tex_mode == 2) {
        // Font glyph alpha channel
        float text_alpha = texture(u_Texture, v_UV).a;
        fill_color.a *= text_alpha;
    }

    // Border stroke evaluation
    if (border_width > 0.0) {
        float inner_dist = dist + border_width;
        float border_factor = smoothstep(-afwidth, afwidth, inner_dist);
        fill_color = mix(fill_color, v_BorderColor, border_factor * v_BorderColor.a);
    }

    fragColor = vec4(fill_color.rgb, fill_color.a * alpha);
}
"""
