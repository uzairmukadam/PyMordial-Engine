#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_ResolvedColor;

layout (binding = 0) uniform sampler2D u_CurrentFrame;
layout (binding = 1) uniform sampler2D u_HistoryFrame;
layout (binding = 2) uniform sampler2D u_GBufferDepth;

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
uniform float u_Feedback = 0.90; // 90% history, 10% current

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

void main() {
    vec4 current_sample = texture(u_CurrentFrame, v_UV);
    float raw_depth = texture(u_GBufferDepth, v_UV).r;

    // If sky/void, simple blend
    if (raw_depth <= 0.000001) {
        vec4 hist = texture(u_HistoryFrame, v_UV);
        out_ResolvedColor = mix(current_sample, hist, u_Feedback);
        return;
    }

    // Reconstruct world position and project to previous frame clip space
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= view_pos.w;
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    vec4 prev_clip = u_PrevViewProjection * world_pos;
    vec3 prev_ndc = prev_clip.xyz / prev_clip.w;
    vec2 prev_uv = prev_ndc.xy * 0.5 + 0.5;

    // Check if reprojected UV is outside the screen
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        out_ResolvedColor = current_sample;
        return;
    }

    vec4 history_sample = texture(u_HistoryFrame, prev_uv);

    // Compute 3x3 neighborhood AABB bounding box in YCoCg space
    vec2 texel_size = 1.0 / u_ScreenSize_Jitter.xy;
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    vec3 aabb_min = vec3(1e6);
    vec3 aabb_max = vec3(-1e6);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec3 c = texture(u_CurrentFrame, v_UV + vec2(float(dx), float(dy)) * texel_size).rgb;
            vec3 ycocg = RGBToYCoCg(c);
            aabb_min = min(aabb_min, ycocg);
            aabb_max = max(aabb_max, ycocg);
            m1 += ycocg;
            m2 += ycocg * ycocg;
        }
    }

    // Variance-based clipping box
    vec3 mu = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3(0.0)));
    vec3 var_min = mu - 1.5 * sigma;
    vec3 var_max = mu + 1.5 * sigma;

    aabb_min = max(aabb_min, var_min);
    aabb_max = min(aabb_max, var_max);

    // Clamp history into neighborhood bounding box
    vec3 history_ycocg = RGBToYCoCg(history_sample.rgb);
    vec3 clamped_history = clamp(history_ycocg, aabb_min, aabb_max);
    vec3 history_filtered = YCoCgToRGB(clamped_history);

    // Dynamic blend weight: more responsive when velocity/disocclusion is detected
    float blend = clamp(1.0 - u_Feedback, 0.05, 0.25);
    vec3 final_color = mix(history_filtered, current_sample.rgb, blend);

    out_ResolvedColor = vec4(final_color, 1.0);
}
