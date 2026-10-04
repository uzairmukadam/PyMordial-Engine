#version 450 core

in vec2 v_UV;
layout (location = 0) out float out_AO;

layout (binding = 0) uniform sampler2D u_GBufferDepth;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;

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

uniform float u_Radius = 0.35;
uniform float u_Intensity = 1.2;
uniform float u_Power = 1.5;
uniform int u_Directions = 3;
uniform int u_Steps = 4;

const float PI = 3.14159265358979323846;
const float HALF_PI = 1.5707963267948966;

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 GetViewPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = u_InvProjection * clip;
    return view.xyz / view.w;
}

// Jorge Jimenez's Interleaved Gradient Noise for spatial dithering
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

// Fast accurate acos approximation
float FastACos(float inX) {
    float x = abs(inX);
    float res = -0.156583 * x + HALF_PI;
    res *= sqrt(max(0.0, 1.0 - x));
    return (inX >= 0.0) ? res : PI - res;
}

void main() {
    float center_depth = texture(u_GBufferDepth, v_UV).r;
    if (center_depth <= 0.000001) {
        out_AO = 1.0;
        return;
    }

    vec3 view_pos = GetViewPos(v_UV, center_depth);
    vec3 world_normal = OctahedralDecode(texture(u_GBufferNormalMetallic, v_UV).rg);
    vec3 view_normal = normalize((u_View * vec4(world_normal, 0.0)).xyz);
    vec3 view_dir = normalize(-view_pos);

    float noise = InterleavedGradientNoise(gl_FragCoord.xy);

    // Physically scaled screen radius in UV units
    float view_z = max(-view_pos.z, 0.1);
    float proj_scale = u_Projection[1][1] * 0.5;
    vec2 screen_radius_uv = vec2(
        (u_Radius * proj_scale / view_z) * (u_ScreenSize_Jitter.y / u_ScreenSize_Jitter.x),
        (u_Radius * proj_scale / view_z)
    );
    vec2 min_radius_uv = 4.0 / u_ScreenSize_Jitter.xy;
    vec2 max_radius_uv = vec2(0.20);
    screen_radius_uv = clamp(screen_radius_uv, min_radius_uv, max_radius_uv);

    int num_dirs = clamp(u_Directions, 1, 6);
    int num_steps = clamp(u_Steps, 2, 8);

    float visibility = 0.0;

    for (int d = 0; d < num_dirs; ++d) {
        float phi = (float(d) + noise) * (PI / float(num_dirs));
        vec2 dir_2d = vec2(cos(phi), sin(phi));

        vec3 dir_view = vec3(dir_2d.x, dir_2d.y, 0.0);
        vec3 ortho_dir_view = dir_view - dot(dir_view, view_dir) * view_dir;
        vec3 axis_vec = normalize(cross(ortho_dir_view, view_dir));
        vec3 proj_normal = view_normal - axis_vec * dot(view_normal, axis_vec);

        float proj_norm_len = length(proj_normal);
        if (proj_norm_len < 0.001) continue;

        float sign_norm = sign(dot(ortho_dir_view, proj_normal));
        float cos_norm = clamp(dot(proj_normal, view_dir) / proj_norm_len, 0.0, 1.0);
        float n = sign_norm * FastACos(cos_norm);

        // Natural unoccluded horizon target limits defined by tangent plane
        float low_horizon_cos0 = cos(n + HALF_PI);
        float low_horizon_cos1 = cos(n - HALF_PI);

        float horizon_cos0 = low_horizon_cos0;
        float horizon_cos1 = low_horizon_cos1;

        for (int s = 1; s <= num_steps; ++s) {
            float alpha = (float(s) - 0.5 + noise * 0.5) / float(num_steps);

            // Step along positive side
            vec2 uv0 = v_UV + dir_2d * screen_radius_uv * alpha;
            if (uv0.x >= 0.0 && uv0.x <= 1.0 && uv0.y >= 0.0 && uv0.y <= 1.0) {
                float d0 = texture(u_GBufferDepth, uv0).r;
                if (d0 > 0.000001) {
                    vec3 p0 = GetViewPos(uv0, d0);
                    vec3 delta0 = p0 - view_pos;
                    float dist0 = length(delta0);
                    if (dist0 > 0.002 && dist0 < u_Radius && dot(delta0, view_normal) > 0.005) {
                        float dist_ratio0 = dist0 / u_Radius;
                        float weight0 = clamp(1.0 - dist_ratio0 * dist_ratio0, 0.0, 1.0);
                        float shc0 = dot(delta0 / dist0, view_dir);
                        shc0 = mix(low_horizon_cos0, shc0, weight0);
                        horizon_cos0 = max(horizon_cos0, shc0);
                    }
                }
            }

            // Step along negative side
            vec2 uv1 = v_UV - dir_2d * screen_radius_uv * alpha;
            if (uv1.x >= 0.0 && uv1.x <= 1.0 && uv1.y >= 0.0 && uv1.y <= 1.0) {
                float d1 = texture(u_GBufferDepth, uv1).r;
                if (d1 > 0.000001) {
                    vec3 p1 = GetViewPos(uv1, d1);
                    vec3 delta1 = p1 - view_pos;
                    float dist1 = length(delta1);
                    if (dist1 > 0.002 && dist1 < u_Radius && dot(delta1, view_normal) > 0.005) {
                        float dist_ratio1 = dist1 / u_Radius;
                        float weight1 = clamp(1.0 - dist_ratio1 * dist_ratio1, 0.0, 1.0);
                        float shc1 = dot(delta1 / dist1, view_dir);
                        shc1 = mix(low_horizon_cos1, shc1, weight1);
                        horizon_cos1 = max(horizon_cos1, shc1);
                    }
                }
            }
        }

        // Compute horizon angles
        float h0 = -FastACos(clamp(horizon_cos1, -1.0, 1.0));
        float h1 = FastACos(clamp(horizon_cos0, -1.0, 1.0));

        h0 = n + clamp(h0 - n, -HALF_PI, HALF_PI);
        h1 = n + clamp(h1 - n, -HALF_PI, HALF_PI);

        // Jimenez analytical slice integral
        float iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;

        visibility += proj_norm_len * (iarc0 + iarc1);
    }

    visibility /= float(num_dirs);
    float ao = clamp(pow(clamp(visibility, 0.0, 1.0), u_Power * u_Intensity), 0.0, 1.0);

    // Distance fade-out: Screen-space ambient occlusion represents localized contact crevices;
    // fade smoothly to unoccluded (1.0) beyond 70m to eliminate far-distance screen-space noise
    float dist_fade = clamp((view_z - 70.0) / 50.0, 0.0, 1.0);
    out_AO = mix(ao, 1.0, dist_fade);
}
