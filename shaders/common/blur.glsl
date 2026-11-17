// ============================================================================
//  Canonical Blur Library
//  Shared Gaussian kernels, separable / Kawase filters and edge-aware
//  bilateral reconstruction used by ambient occlusion, screen-space
//  reflections, screen-space GI and the bloom pyramid.
//
//  Consumers declare their own samplers and uniforms; this include only
//  provides pure functions so it can be dropped into any fragment shader.
// ============================================================================

#ifndef BLUR_GLSL
#define BLUR_GLSL

// ----------------------------------------------------------------------------
//  Kernel
// ----------------------------------------------------------------------------

// Normalised 5-tap Gaussian kernel (sigma ~= 1.0 texel).
const float GAUSSIAN_5[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

// Continuous 1D Gaussian weight for an arbitrary tap offset.
float GaussianWeight1D(float offset, float sigma) {
    float s = max(sigma, 1e-4);
    return exp(-(offset * offset) / (2.0 * s * s));
}

// Kernel description shared by the separable and bilateral entry points.
struct BlurParams {
    vec2 direction;  // texel-space step direction, e.g. vec2(1.0 / width, 0.0)
    float radius;    // tap spacing multiplier (1.0 = one texel per tap)
    float sigma;     // spatial falloff in texels
};

// ----------------------------------------------------------------------------
//  G-Buffer Reconstruction
// ----------------------------------------------------------------------------

// Reconstructs linear view-space depth from a raw depth-buffer sample.
float LinearizeViewDepth(float raw_depth, vec2 uv, mat4 inv_proj) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = inv_proj * clip;
    return -view.z / max(abs(view.w), 1e-6);
}

// Reconstructs the full view-space position from a raw depth-buffer sample.
vec3 ViewPosFromDepth(vec2 uv, float raw_depth, mat4 inv_proj) {
    vec4 clip = vec4(uv * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view = inv_proj * clip;
    return view.xyz / max(abs(view.w), 1e-6);
}

// Decodes an octahedral-encoded unit normal stored in the G-Buffer.
vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// ----------------------------------------------------------------------------
//  Plain Filters
// ----------------------------------------------------------------------------

// Plain separable 5-tap Gaussian. Call twice (horizontal then vertical).
vec4 SeparableBlur(sampler2D tex, vec2 uv, vec2 direction, float radius) {
    vec4 sum = texture(tex, uv) * GAUSSIAN_5[0];
    float total = GAUSSIAN_5[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = direction * (float(i) * radius);
        sum += texture(tex, uv + offset) * GAUSSIAN_5[i];
        sum += texture(tex, uv - offset) * GAUSSIAN_5[i];
        total += 2.0 * GAUSSIAN_5[i];
    }
    return sum / total;
}

// Sigma-driven separable Gaussian for callers that need a wider kernel.
vec4 SeparableBlur(sampler2D tex, vec2 uv, BlurParams p) {
    vec4 sum = texture(tex, uv);
    float total = 1.0;
    for (int i = 1; i < 5; ++i) {
        float w = GaussianWeight1D(float(i) * p.radius, p.sigma);
        vec2 offset = p.direction * (float(i) * p.radius);
        sum += (texture(tex, uv + offset) + texture(tex, uv - offset)) * w;
        total += 2.0 * w;
    }
    return sum / total;
}

// Cheap dual-filter (Kawase) blur: four diagonal taps at `offset` texels.
// Ideal for low-resolution features where a full Gaussian is too expensive.
vec4 KawaseBlur(sampler2D tex, vec2 uv, vec2 texel, float offset) {
    vec2 o = texel * offset;
    vec4 sum = texture(tex, uv + vec2(-o.x, -o.y));
    sum += texture(tex, uv + vec2(o.x, -o.y));
    sum += texture(tex, uv + vec2(-o.x, o.y));
    sum += texture(tex, uv + vec2(o.x, o.y));
    return sum * 0.25;
}

// Exact 3x3 tent ((1,2,1)^2 / 16) using nine point samples.
//
// Mathematically identical to KawaseBlur(tex, uv, texel, 0.5), but the taps land
// on texel centres so no hardware bilinear interpolation happens. Use this when
// the source is a half-float target and bit-exact reproduction matters.
vec4 TentBlur3x3(sampler2D tex, vec2 uv, vec2 texel) {
    vec4 sum = texture(tex, uv) * 4.0;
    sum += (texture(tex, uv + vec2(-texel.x, 0.0)) + texture(tex, uv + vec2(texel.x, 0.0))) * 2.0;
    sum += (texture(tex, uv + vec2(0.0, -texel.y)) + texture(tex, uv + vec2(0.0, texel.y))) * 2.0;
    sum += texture(tex, uv + vec2(-texel.x, -texel.y));
    sum += texture(tex, uv + vec2(texel.x, -texel.y));
    sum += texture(tex, uv + vec2(-texel.x, texel.y));
    sum += texture(tex, uv + vec2(texel.x, texel.y));
    return sum / 16.0;
}

// ----------------------------------------------------------------------------
//  Edge-Aware Bilateral Accumulation
// ----------------------------------------------------------------------------

// Accumulation flags for BilateralAccumulate().
#define BLUR_USE_NORMAL   1  // reject taps whose normal disagrees with the centre
#define BLUR_METRIC_PLANE 2  // tangent-plane distance instead of view-depth delta
#define BLUR_WEIGHT_ALPHA 4  // weight taps by their own alpha (confidence)
#define BLUR_CLAMP_UV     8  // discard taps that leave the [0, 1] screen range
#define BLUR_ALPHA_BY_KERNEL 16  // normalise alpha by the full kernel weight

// Shared 5-tap bilateral accumulation core.
//
// `rgb_sum` receives sum(sample.rgb * edge_w * alpha_w) and `alpha_sum`
// sum(sample.a * edge_w), where `alpha_w` is the sample alpha when
// BLUR_WEIGHT_ALPHA is set and 1.0 otherwise. `total_w` receives the sum of the
// accepted tap weights and `kernel_w` the sum of every Gaussian weight in the
// kernel (accepted or not), so callers can pick their own normalisation.
//
// `normal_to_view` rotates decoded world-space normals into the space of the
// reconstructed view positions; pass mat3(1.0) when the metric is depth-based.
void BilateralAccumulate(sampler2D tex, sampler2D depth, sampler2D normal,
                         vec2 uv, vec2 direction, float radius,
                         float sigma_z, float normal_power, mat4 inv_proj,
                         int flags, mat3 normal_to_view,
                         out vec3 rgb_sum, out float alpha_sum,
                         out float total_w, out float kernel_w) {
    bool use_normal = (flags & BLUR_USE_NORMAL) != 0;
    bool plane_metric = (flags & BLUR_METRIC_PLANE) != 0;
    bool weight_alpha = (flags & BLUR_WEIGHT_ALPHA) != 0;
    bool clamp_uv = (flags & BLUR_CLAMP_UV) != 0;

    float center_raw = texture(depth, uv).r;
    vec4 center = texture(tex, uv);
    float center_z = LinearizeViewDepth(center_raw, uv, inv_proj);
    vec3 center_pos = ViewPosFromDepth(uv, center_raw, inv_proj);
    vec3 center_n = use_normal
        ? normalize(normal_to_view * OctahedralDecode(texture(normal, uv).rg))
        : vec3(0.0, 0.0, 1.0);

    float two_sigma_sq = 2.0 * sigma_z * sigma_z;
    float center_alpha_w = weight_alpha ? center.a : 1.0;

    rgb_sum = center.rgb * (center_alpha_w * GAUSSIAN_5[0]);
    alpha_sum = center.a * GAUSSIAN_5[0];
    total_w = center_alpha_w * GAUSSIAN_5[0];
    kernel_w = GAUSSIAN_5[0];

    for (int i = 1; i < 5; ++i) {
        float w = GAUSSIAN_5[i];
        vec2 offset = direction * (float(i) * radius);
        kernel_w += 2.0 * w;

        for (int s = 0; s < 2; ++s) {
            vec2 uv_s = (s == 0) ? uv + offset : uv - offset;
            if (clamp_uv && (uv_s.x < 0.0 || uv_s.x > 1.0 || uv_s.y < 0.0 || uv_s.y > 1.0)) {
                continue;
            }

            float raw = texture(depth, uv_s).r;
            if (raw <= 0.000001) {
                continue;
            }

            vec4 sample_c = texture(tex, uv_s);
            if (weight_alpha && sample_c.a <= 0.0001) {
                continue;
            }

            float diff;
            if (plane_metric) {
                vec3 pos = ViewPosFromDepth(uv_s, raw, inv_proj);
                diff = abs(dot(center_n, pos - center_pos));
            } else {
                diff = LinearizeViewDepth(raw, uv_s, inv_proj) - center_z;
            }

            float edge_w = w * exp(-(diff * diff) / two_sigma_sq);
            if (use_normal) {
                vec3 n_s = normalize(normal_to_view * OctahedralDecode(texture(normal, uv_s).rg));
                edge_w *= pow(max(dot(center_n, n_s), 0.0), normal_power);
            }

            float alpha_w = weight_alpha ? sample_c.a : 1.0;
            rgb_sum += sample_c.rgb * (edge_w * alpha_w);
            alpha_sum += sample_c.a * edge_w;
            total_w += edge_w * alpha_w;
        }
    }
}

// ----------------------------------------------------------------------------
//  Bilateral Entry Points
// ----------------------------------------------------------------------------

// Depth-aware bilateral blur. Returns the normalised weighted average, so a
// single-channel source (ambient occlusion) can be read from `.r`.
vec4 BilateralBlurDepth(sampler2D tex, sampler2D depth, vec2 uv, vec2 direction,
                        float radius, float sigma_z, mat4 inv_proj) {
    vec3 rgb_sum;
    float alpha_sum;
    float total_w;
    float kernel_w;
    BilateralAccumulate(tex, depth, depth, uv, direction, radius,
                        sigma_z, 1.0, inv_proj, 0, mat3(1.0),
                        rgb_sum, alpha_sum, total_w, kernel_w);
    float inv = 1.0 / max(total_w, 1e-4);
    return vec4(rgb_sum * inv, alpha_sum * inv);
}

// Depth + normal aware bilateral blur with an explicit metric and normal space.
//
// `flags` combines the BLUR_* bits above. With BLUR_ALPHA_BY_KERNEL the returned
// alpha is the accepted-weight fraction of the whole kernel (a confidence
// value); otherwise the full RGBA is a plain weighted average.
vec4 BilateralBlurDepthNormal(sampler2D tex, sampler2D depth, sampler2D normal,
                              vec2 uv, vec2 direction, float radius,
                              float sigma_z, float normal_power, mat4 inv_proj,
                              int flags, mat3 normal_to_view) {
    vec3 rgb_sum;
    float alpha_sum;
    float total_w;
    float kernel_w;
    BilateralAccumulate(tex, depth, normal, uv, direction, radius,
                        sigma_z, normal_power, inv_proj, flags,
                        normal_to_view, rgb_sum, alpha_sum, total_w, kernel_w);
    if ((flags & BLUR_ALPHA_BY_KERNEL) != 0) {
        return vec4(rgb_sum / max(total_w, 1e-4), alpha_sum / max(kernel_w, 1e-4));
    }
    float inv = 1.0 / max(total_w, 1e-4);
    return vec4(rgb_sum * inv, alpha_sum * inv);
}

// Depth + normal aware bilateral blur using the view-depth metric and
// world-space normals. Returns the normalised weighted average of the sample.
vec4 BilateralBlurDepthNormal(sampler2D tex, sampler2D depth, sampler2D normal,
                              vec2 uv, vec2 direction, float radius,
                              float sigma_z, float normal_power, mat4 inv_proj) {
    return BilateralBlurDepthNormal(tex, depth, normal, uv, direction, radius,
                                    sigma_z, normal_power, inv_proj,
                                    BLUR_USE_NORMAL, mat3(1.0));
}

#endif // BLUR_GLSL
