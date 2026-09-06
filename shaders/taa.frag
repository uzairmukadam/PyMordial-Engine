#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_ResolvedColor;

layout (binding = 0) uniform sampler2D u_CurrentFrame;
layout (binding = 1) uniform sampler2D u_HistoryFrame;
layout (binding = 2) uniform sampler2D u_GBufferDepth;
layout (binding = 3) uniform sampler2D u_VelocityTexture;

layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;

    vec4 u_CameraPos_Time;
    vec4 u_ScreenSize_Jitter;

    vec4 u_SunDirection_Intensity;
    vec4 u_SunColor_Ambient;

    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;

    vec4 u_FogColor_Density;
    vec4 u_FogParams;
};

uniform mat4 u_PrevViewProjection;
uniform float u_Feedback = 0.92;   // Base temporal accumulation factor
uniform float u_Sharpness = 0.35;  // Subtle contrast-adaptive unsharp mask

vec3 RGBToYCoCg(vec3 c) {
    return vec3(
        c.r * 0.25 + c.g * 0.5 + c.b * 0.25,
        c.r * 0.5 - c.b * 0.5,
        -c.r * 0.25 + c.g * 0.5 - c.b * 0.25
    );
}

vec3 YCoCgToRGB(vec3 c) {
    return vec3(
        c.x + c.y - c.z,
        c.x + c.z,
        c.x - c.y - c.z
    );
}

// Perceptual luma for HDR tone-weighting
float Luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// 5-tap Catmull-Rom bicubic history filter (eliminates texture blur)
vec4 SampleCatmullRom5(sampler2D tex, vec2 uv, vec2 tex_size) {
    vec2 sample_pos = uv * tex_size;
    vec2 tc = floor(sample_pos - 0.5) + 0.5;
    vec2 f = sample_pos - tc;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    vec2 w12 = w1 + w2;
    vec2 tc12 = tc + (w2 / max(w12, vec2(1e-5)));

    vec2 tc0 = tc - 1.0;
    vec2 tc3 = tc + 2.0;

    vec4 c12 = texture(tex, vec2(tc12.x, tc12.y) / tex_size);
    vec4 c0 = texture(tex, vec2(tc12.x, tc0.y) / tex_size);
    vec4 c1 = texture(tex, vec2(tc0.x, tc12.y) / tex_size);
    vec4 c2 = texture(tex, vec2(tc3.x, tc12.y) / tex_size);
    vec4 c3 = texture(tex, vec2(tc12.x, tc3.y) / tex_size);

    float weight_center = w12.x * w12.y;
    float weight_top = w12.x * w0.y;
    float weight_left = w0.x * w12.y;
    float weight_right = w3.x * w12.y;
    float weight_bottom = w12.x * w3.y;

    vec4 result = c12 * weight_center + c0 * weight_top + c1 * weight_left + c2 * weight_right + c3 * weight_bottom;
    float sum = weight_center + weight_top + weight_left + weight_right + weight_bottom;
    return max(result / max(sum, 1e-5), vec4(0.0));
}

// Karis / PlayStation AABB Line Segment Clipping in YCoCg
vec3 ClipToAABB(vec3 aabb_min, vec3 aabb_max, vec3 p, vec3 q) {
    vec3 r = q - p;
    vec3 rmax = aabb_max - p;
    vec3 rmin = aabb_min - p;

    const float eps = 1e-7;
    vec3 tmax = vec3(1.0);
    vec3 tmin = vec3(0.0);

    if (abs(r.x) > eps) {
        float t1 = rmin.x / r.x;
        float t2 = rmax.x / r.x;
        tmax.x = max(t1, t2);
        tmin.x = min(t1, t2);
    }
    if (abs(r.y) > eps) {
        float t1 = rmin.y / r.y;
        float t2 = rmax.y / r.y;
        tmax.y = max(t1, t2);
        tmin.y = min(t1, t2);
    }
    if (abs(r.z) > eps) {
        float t1 = rmin.z / r.z;
        float t2 = rmax.z / r.z;
        tmax.z = max(t1, t2);
        tmin.z = min(t1, t2);
    }

    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);

    if (t_enter <= t_exit && t_enter >= 0.0 && t_enter <= 1.0) {
        return p + r * t_enter;
    }
    return clamp(q, aabb_min, aabb_max);
}

void main() {
    vec2 texel_size = 1.0 / u_ScreenSize_Jitter.xy;
    vec4 current_sample = texture(u_CurrentFrame, v_UV);

    // 1. Find closest depth in 3x3 neighborhood (Reversed-Z: max depth is closest to camera)
    float closest_depth = -1.0;
    vec2 closest_offset = vec2(0.0);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 off = vec2(float(dx), float(dy)) * texel_size;
            float d = texture(u_GBufferDepth, v_UV + off).r;
            if (d > closest_depth) {
                closest_depth = d;
                closest_offset = off;
            }
        }
    }

    // 2. Fetch velocity from closest depth neighbor (dilates moving foreground edges)
    vec2 velocity = texture(u_VelocityTexture, v_UV + closest_offset).xy;

    // Fallback camera reprojection if velocity is zero or on background/sky
    if (closest_depth <= 1e-6 || length(velocity) <= 1e-6) {
        vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, max(closest_depth, 0.0), 1.0);
        vec4 view_pos = u_InvProjection * clip_pos;
        view_pos /= max(view_pos.w, 1e-6);
        vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

        vec4 prev_clip = u_PrevViewProjection * world_pos;
        vec2 prev_uv = (prev_clip.xy / max(prev_clip.w, 1e-6)) * 0.5 + 0.5;
        if (length(prev_uv - v_UV) > 1e-6) {
            velocity = v_UV - prev_uv;
        }
    }

    // Previous UV where this surface was located
    vec2 prev_uv = v_UV - velocity;

    // Disocclusion / boundary check
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        out_ResolvedColor = current_sample;
        return;
    }

    // 3. Sample history buffer with 5-tap Catmull-Rom filter
    vec4 history_sample = SampleCatmullRom5(u_HistoryFrame, prev_uv, u_ScreenSize_Jitter.xy);

    // 4. Compute 3x3 neighborhood statistics in YCoCg color space
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    vec3 box_min = vec3(1e6);
    vec3 box_max = vec3(-1e6);
    vec3 cross_blur = vec3(0.0);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec3 neighbor = texture(u_CurrentFrame, v_UV + vec2(float(dx), float(dy)) * texel_size).rgb;
            vec3 ycocg = RGBToYCoCg(neighbor);

            box_min = min(box_min, ycocg);
            box_max = max(box_max, ycocg);
            m1 += ycocg;
            m2 += ycocg * ycocg;

            if (abs(dx) + abs(dy) <= 1) {
                cross_blur += neighbor;
            }
        }
    }

    // Variance-based clipping box (1.25 sigma)
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    vec3 var_min = max(box_min, mean - 1.25 * sigma);
    vec3 var_max = min(box_max, mean + 1.25 * sigma);

    // 5. Line clipping: clip history color along ray to current neighborhood center
    vec3 current_ycocg = RGBToYCoCg(current_sample.rgb);
    vec3 history_ycocg = RGBToYCoCg(history_sample.rgb);
    vec3 clipped_ycocg = ClipToAABB(var_min, var_max, current_ycocg, history_ycocg);
    vec3 history_clipped = max(YCoCgToRGB(clipped_ycocg), vec3(0.0));

    // 6. Luminance-Weighted Accumulation (Karis Weighting) to eliminate fireflies and specular jitter
    float w_curr = 1.0 / (1.0 + max(Luma(current_sample.rgb), 0.0));
    float w_hist = 1.0 / (1.0 + max(Luma(history_clipped), 0.0));

    // Dynamic feedback: responsive during fast motion, stable when stationary
    float speed = length(velocity * u_ScreenSize_Jitter.xy);
    float feedback = clamp(u_Feedback - speed * 0.04, 0.70, u_Feedback);

    vec3 blended_color = (current_sample.rgb * w_curr * (1.0 - feedback) + history_clipped * w_hist * feedback)
                       / max(w_curr * (1.0 - feedback) + w_hist * feedback, 1e-5);

    // 7. Subtle adaptive unsharp mask to maintain razor-sharp texture details
    cross_blur /= 5.0;
    vec3 detail = current_sample.rgb - cross_blur;
    vec3 final_color = blended_color + detail * u_Sharpness;

    out_ResolvedColor = vec4(max(final_color, vec3(0.0)), 1.0);
}
