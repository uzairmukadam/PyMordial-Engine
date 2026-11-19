// ============================================================================
//  Cascaded Shadow Map Sampling Library
//
//  Shared by the deferred resolve and the direct-radiance pass so both compute
//  identical directional-light visibility. Consumers must declare the uniforms
//  and samplers this file references:
//
//      sampler2D       u_ShadowAtlas
//      sampler2DShadow u_ShadowAtlasCmp
//      sampler2D       u_SpotShadowAtlas
//      sampler2DShadow u_SpotShadowAtlasCmp
//      FrameData UBO   (u_LightViewProjection, u_CascadeSplits)
//      u_CascadeCount, u_CascadeAtlasRect[4]
//      u_ShadowBias, u_ShadowNormalBias, u_ShadowBiasSlopeScale,
//      u_ShadowBiasMinScale, u_ShadowBiasMaxScale
//      u_ShadowFilterMode, u_PCF*, u_PCSS*, u_SunAngularRadius
//      u_SpotShadowEnabled, u_SpotLightViewProjection[4]
//
//  Nothing here declares a uniform, so the include can be dropped into any
//  fragment shader that already owns those bindings.
// ============================================================================

#ifndef CSM_SAMPLING_GLSL
#define CSM_SAMPLING_GLSL
// ---------------------------------------------------------------------------
//  Hard shadow sampling
//
//  One receiver setup (SetupCascade) and one hardware comparison tap
//  (ShadowCmpTap). There is no penumbra estimation: the edge is the geometric
//  silhouette of the caster, resolved across a screen pixel so it does not
//  stair-step at high resolution.
// ---------------------------------------------------------------------------

// Everything a filter needs to know about one cascade for one receiver pixel.
struct ShadowCascade {
    vec4 rect;            // atlas rect: xy = UV offset, zw = UV scale
    vec2 uv_min;          // inset atlas bounds (bilinear taps must not bleed)
    vec2 uv_max;
    vec2 uv;              // receiver atlas UV
    float depth;          // receiver shadow-space depth
    vec2 depth_gradient;  // d(depth) / d(atlas UV)
    float texel_uv;       // one atlas texel in atlas UV
    float cascade_texels; // texels across this cascade's rect
    float world_texel;    // one cascade texel in world units
    float screen_texels;  // cascade texels covered by one screen pixel
    float inv_rad;        // 1 / orthographic radius
    float inv_far;        // 1 / depth range
    float bias;           // depth bias in shadow depth units
    float slope;          // tan(acos(N.L)), clamped
    bool in_bounds;
    float edge_fade;      // 0 interior .. 1 at the rect border
};

ShadowCascade SetupCascade(
    int cascade,
    vec3 world_pos,
    vec3 N,
    float slope,
    vec3 world_dx,
    vec3 world_dy
) {
    ShadowCascade c;
    c.rect = u_CascadeAtlasRect[cascade];
    vec2 atlas_scale = max(c.rect.zw, vec2(1e-6));
    c.uv_min = ShadowAtlasUvMin(c.rect, 0.001);
    c.uv_max = ShadowAtlasUvMax(c.rect, 0.001);

    vec2 full_size = vec2(textureSize(u_ShadowAtlas, 0));
    c.texel_uv = 1.0 / max(full_size.x, 1.0);
    c.cascade_texels = max(atlas_scale.x * full_size.x, 1.0);

    // Metric scale recovered from the light projection matrix: the xy rows are
    // scaled by 1/radius and the z row by 1/depth_range.
    c.inv_rad = length(vec3(
        u_LightViewProjection[cascade][0].x,
        u_LightViewProjection[cascade][1].x,
        u_LightViewProjection[cascade][2].x
    ));
    c.inv_far = length(vec3(
        u_LightViewProjection[cascade][0].z,
        u_LightViewProjection[cascade][1].z,
        u_LightViewProjection[cascade][2].z
    ));
    c.world_texel = 2.0 / max(c.inv_rad * c.cascade_texels, 1e-6);

    // Normal-offset bias: push the receiver along its normal by a fraction of a
    // texel footprint. This is what removes acne on sloped surfaces without
    // needing a depth bias large enough to detach the shadow from its caster.
    float normal_offset = u_ShadowNormalBias * slope + c.world_texel * slope * 0.5;
    vec3 biased_pos = world_pos + N * normal_offset;

    vec4 light_space_pos = u_LightViewProjection[cascade] * vec4(biased_pos, 1.0);
    vec3 proj = light_space_pos.xyz / max(light_space_pos.w, 1e-6);
    // Remap NDC XY [-1, 1] to UV [0, 1]; Z is already in [0, 1] under GL_ZERO_TO_ONE clip control
    proj.xy = proj.xy * 0.5 + 0.5;

    c.in_bounds = proj.z >= 0.0 && proj.z <= 1.0 &&
                  proj.x >= 0.0 && proj.x <= 1.0 &&
                  proj.y >= 0.0 && proj.y <= 1.0;
    c.uv = ShadowAtlasUv(proj.xy, c.rect);
    c.depth = proj.z;

    // Receiver-plane depth gradient in atlas UV. Exact for planar receivers,
    // and the reason a per-tap reference depth can replace a fat constant bias.
    vec3 light_dx = (u_LightViewProjection[cascade] * vec4(world_dx, 0.0)).xyz;
    vec3 light_dy = (u_LightViewProjection[cascade] * vec4(world_dy, 0.0)).xyz;
    vec2 uv_dx = light_dx.xy * 0.5 * atlas_scale;
    vec2 uv_dy = light_dy.xy * 0.5 * atlas_scale;

    // How many cascade texels one screen pixel spans. The edge filter must be
    // at least this wide, otherwise the silhouette is thinner than a pixel and
    // reads as a hard stair-step no matter how many taps are taken.
    c.screen_texels = max(length(uv_dx) * c.cascade_texels, 1.0);
    float uv_det = uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x;
    c.depth_gradient = vec2(0.0);
    if (abs(uv_det) > 1e-12) {
        c.depth_gradient.x = (light_dx.z * uv_dy.y - light_dy.z * uv_dx.y) / uv_det;
        c.depth_gradient.y = (uv_dx.x * light_dy.z - uv_dy.x * light_dx.z) / uv_det;
    }

    // Slope-scaled depth bias, floored by a fraction of a texel so grazing
    // surfaces never self-shadow.
    float base_bias_world = max(u_ShadowBias, 0.0);
    float bias_world = clamp(
        base_bias_world * (1.0 + slope * u_ShadowBiasSlopeScale),
        base_bias_world * u_ShadowBiasMinScale,
        base_bias_world * u_ShadowBiasMaxScale
    );
    bias_world = max(bias_world, c.world_texel * min(slope, 3.5) * 0.05);
    c.bias = bias_world * c.inv_far;
    c.slope = slope;

    // Border fade measured in cascade texels, not as a fraction of the cascade.
    // A fractional band is a huge bright ring on a wide cascade.
    vec2 edge_dist = min(proj.xy, 1.0 - proj.xy);
    float edge_texels = min(edge_dist.x, edge_dist.y) * c.cascade_texels;
    c.edge_fade = 1.0 - smoothstep(0.0, 16.0, edge_texels);
    return c;
}

// One hardware comparison tap. The comparison sampler returns the bilinear
// filtered 2x2 PCF result, so this single fetch resolves four texels. The
// reference depth is the receiver plane evaluated at the tap, which keeps the
// bias tight enough that shadows stay welded to their casters.
float ShadowCmpTap(ShadowCascade c, vec2 uv) {
    vec2 tap = clamp(uv, c.uv_min, c.uv_max);
    float ref_depth = ShadowPlaneDepth(c.depth, c.depth_gradient, tap, c.uv) - c.bias;
    return texture(u_ShadowAtlasCmp, vec3(tap, ref_depth));
}

// Hard shadow: a crisp silhouette resolved across one screen pixel rather than
// one shadow texel. A single comparison tap is anti-aliased over a texel, and
// at 1080p a texel spans several pixels, so a lone tap stair-steps. Four taps
// at pixel scale keep the edge sharp while removing the staircase.
float SampleHardShadow(ShadowCascade c) {
    float radius = c.screen_texels * 0.5 * c.texel_uv;
    float phi = ShadowTapPhase(c.uv, c.cascade_texels);
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += ShadowCmpTap(c, c.uv + VogelDiskSample(i, 4, phi) * radius);
    }
    return sum * 0.25;
}

// PCF: a Vogel disk of comparison taps with a radial weight.
//
// The radius is authored in cascade texels, but it is floored at the screen
// footprint. A filter narrower than one screen pixel still stair-steps no
// matter how many taps it takes, because the texel grid underneath stays
// visible; the floor is what makes the tap budget actually buy smoothness.
//
// ``u_PCFFalloff`` shapes the kernel: 1.0 is a linear cone (crisp centre, wide
// soft skirt), 2.0 is a smooth bell, and higher values tighten the penumbra
// toward the centre tap.
float SamplePCFShadow(ShadowCascade c) {
    int taps = clamp(u_PCFSamples, 4, 64);
    float radius_texels = max(u_PCFRadius, c.screen_texels * 0.5);
    float radius = radius_texels * c.texel_uv;
    float falloff = max(u_PCFFalloff, 0.25);
    float phi = ShadowTapPhase(c.uv, c.cascade_texels);

    float sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < taps; ++i) {
        // Normalised squared radius: 0 at the centre, 1 at the rim.
        float r2 = (float(i) + 0.5) / float(taps);
        float w = pow(1.0 - r2, falloff);
        sum += ShadowCmpTap(c, c.uv + VogelDiskSample(i, taps, phi) * radius) * w;
        weight_sum += w;
    }
    return sum / max(weight_sum, 1e-5);
}

// PCSS: blocker search -> penumbra estimate -> disk filter.
//
// Unlike PCF, the filter radius is not authored directly. It is derived from
// the average distance between the receiver and the blockers found around it,
// which is the geometric projection of the light source. That is what makes
// contact shadows stay sharp while only elevated casters soften.
//
// The search radius is in cascade texels and is deliberately wider than the
// filter: a blocker outside the search footprint cannot contribute to the
// penumbra, so the search must cover the largest gap the light can project.
float SamplePCSSShadow(ShadowCascade c) {
    int search_taps = clamp(u_PCSSSearchSamples, 4, 64);
    float search_radius = max(u_PCSSSearchRadius, c.screen_texels * 0.5) * c.texel_uv;
    float search_phi = ShadowTapPhase(c.uv, c.cascade_texels);

    // Step 1: blocker search. The centre tap is included explicitly so a
    // receiver directly under an occluder never reports "no blockers" because
    // the coarse taps straddled it.
    //
    // Two estimates are accumulated:
    //
    //   * ``avg_blocker`` - the classic PCSS average. Stable, but wrong when two
    //     shadows overlap: a near caster (small gap, sharp) and a far caster
    //     (large gap, soft) average to a medium gap, softening the near shadow.
    //   * ``close_blocker`` - a closeness-weighted average. A pure max would be
    //     the sharpest estimate, but a max over noisy taps is high-variance:
    //     one outlier tap dominates the whole pixel, which is exactly what makes
    //     a PCSS penumbra look blotchy. Weighting each blocker by how close it
    //     is keeps the near caster dominant while averaging away the noise.
    float blocker_sum = 0.0;
    int blocker_count = 0;
    float close_blocker_sum = 0.0;
    float close_weight_sum = 0.0;

    // Closeness weight: 1.0 for a blocker touching the receiver, falling off
    // with the gap measured in cascade texels. The 0.05 scale means a blocker
    // 20 texels further away contributes half as much.
    float inv_far = max(c.inv_far, 1e-6);
    float inv_world_texel = 1.0 / max(c.world_texel, 1e-6);

    float center_depth = texture(u_ShadowAtlas, c.uv).r;
    if (center_depth < c.depth - c.bias) {
        blocker_sum += center_depth;
        blocker_count++;
        float gap_texels = max(c.depth - center_depth, 0.0) * inv_far * inv_world_texel;
        float w = 1.0 / (1.0 + gap_texels * 0.05);
        close_blocker_sum += center_depth * w;
        close_weight_sum += w;
    }
    for (int i = 0; i < search_taps; ++i) {
        vec2 tap = clamp(c.uv + VogelDiskSample(i, search_taps, search_phi) * search_radius, c.uv_min, c.uv_max);
        float stored = texture(u_ShadowAtlas, tap).r;
        float ref_depth = ShadowPlaneDepth(c.depth, c.depth_gradient, tap, c.uv) - c.bias;
        if (stored < ref_depth) {
            blocker_sum += stored;
            blocker_count++;
            float gap_texels = max(c.depth - stored, 0.0) * inv_far * inv_world_texel;
            float w = 1.0 / (1.0 + gap_texels * 0.05);
            close_blocker_sum += stored * w;
            close_weight_sum += w;
        }
    }
    if (blocker_count == 0) {
        return 1.0;
    }

    // Step 2: penumbra from the receiver-to-blocker gap. Shadow-space depth is
    // already measured along the light axis, so the gap converts to world units
    // by dividing by the cascade's depth scale.
    //
    // ``u_PCSSClosestBlockerWeight`` blends between the average gap (0.0, the
    // classic PCSS estimate) and the closeness-weighted gap (1.0). The default
    // is 1.0 so overlapping shadows keep the near caster's sharpness instead of
    // inheriting the far caster's softness.
    float avg_blocker = blocker_sum / float(blocker_count);
    float close_blocker = close_blocker_sum / max(close_weight_sum, 1e-6);
    float avg_gap = max(c.depth - avg_blocker, 0.0) / inv_far;
    float close_gap = max(c.depth - close_blocker, 0.0) / inv_far;
    float gap_world = mix(avg_gap, close_gap, clamp(u_PCSSClosestBlockerWeight, 0.0, 1.0));
    // Penumbra width. With a physical sun angular radius the penumbra is the
    // geometric projection of the light disc: gap * tan(angular_radius). That
    // is what makes contact hardening automatic -- a caster touching the
    // receiver has gap 0 and therefore a perfectly sharp edge. The authored
    // ``u_PCSSLightSize`` factor remains as the fallback for stylised lights.
    float penumbra_world = (u_SunAngularRadius > 0.0)
        ? gap_world * u_SunAngularRadius
        : gap_world * max(u_PCSSLightSize, 1e-5);
    // world_texel is metres per cascade texel, so this is a plain unit
    // conversion. Deriving it from inv_rad * cascade_texels instead would be
    // 2x too large, because the cascade spans 2 * radius in world space.
    float penumbra_texels = penumbra_world / max(c.world_texel, 1e-6);

    // The filter may never outgrow the region the blocker search actually
    // covered, otherwise the penumbra is unanchored and reads as a blurry blob.
    // It is also never narrower than a screen pixel, so a contact shadow still
    // resolves its edge instead of stair-stepping.
    float min_radius = max(1.0, c.screen_texels * 0.5);
    float max_radius = max(min(u_PCSSMaxRadius, u_PCSSSearchRadius * 0.9), min_radius);
    float radius_texels = clamp(penumbra_texels, min_radius, max_radius);
    float radius = radius_texels * c.texel_uv;

    // Step 3: disk filter over the estimated penumbra.
    //
    // Two things make this read as a smooth penumbra rather than a blurry blob:
    //
    //   * A Gaussian falloff instead of ``pow(1 - r2, n)``. The polynomial form
    //     has a hard rim at r2 = 1, which shows up as a visible ring where the
    //     filter stops. A Gaussian decays to zero smoothly, so the penumbra
    //     fades out instead of ending.
    //   * A tap count that scales with the radius. A fixed budget spread over a
    //     wide penumbra under-samples it into visible banding; scaling keeps
    //     taps-per-texel roughly constant.
    //
    // A second, decorrelated phase keeps the search and filter disks from
    // sharing tap directions, which would correlate their error.
    float radius_texels_clamped = max(radius_texels, 1.0);
    int taps = clamp(
        int(radius_texels_clamped * radius_texels_clamped * 0.35),
        clamp(u_PCSSSamples, 4, 64),
        96
    );
    float falloff = max(u_PCSSFalloff, 0.25);
    float filter_phi = search_phi + 2.39996323;

    // Gaussian sigma chosen so the kernel reaches ~zero at the disk rim.
    float sigma = max(radius_texels_clamped * 0.5, 1e-3);
    float inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    float sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < taps; ++i) {
        vec2 offset = VogelDiskSample(i, taps, filter_phi) * radius;
        // VogelDiskSample returns a unit-disk point, so its squared length is
        // the normalised squared radius used by the Gaussian.
        float r2 = dot(offset, offset) / max(radius * radius, 1e-12);
        float dist_texels = sqrt(r2) * radius_texels_clamped;
        float w = exp(-dist_texels * dist_texels * inv_two_sigma_sq);
        // Sharpen the tail for high falloff values so the parameter still
        // controls how tight the penumbra reads.
        w = pow(w, falloff * 0.5);
        sum += ShadowCmpTap(c, c.uv + offset) * w;
        weight_sum += w;
    }
    return sum / max(weight_sum, 1e-5);
}

// Dispatches one cascade through the active filter tier.
//
// PCSS is the narrowest tier (cascade 1 by default) because it is the most
// expensive and its variable penumbra is most visible in the near field. PCF
// covers the remaining near cascades, and the far cascades keep the hard path.
float SampleCascadeShadow(int cascade, ShadowCascade c) {
    if (u_ShadowFilterMode == 2 && cascade < u_PCSS_CascadeLimit) {
        return SamplePCSSShadow(c);
    }
    if (u_ShadowFilterMode >= 1 && cascade < u_PCFCascadeLimit) {
        return SamplePCFShadow(c);
    }
    return SampleHardShadow(c);
}

// Cascaded shadow evaluation with overlapped, validity-aware split blending.
// ``out_cascade`` / ``out_coverage`` feed the cascade debug visualizations.
float CalculateCSMShadow(
    vec3 world_pos,
    vec3 N,
    vec3 L,
    float view_depth,
    out int out_cascade,
    out float out_coverage
) {
    out_cascade = 0;
    out_coverage = 0.0;
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) {
        return 0.0;
    }

    float max_shadow_dist = u_CascadeSplits[u_CascadeCount - 1];
    if (view_depth >= max_shadow_dist) {
        return 1.0;
    }

    int cascade = 0;
    for (int i = 0; i < u_CascadeCount - 1; ++i) {
        if (view_depth > u_CascadeSplits[i]) {
            cascade = i + 1;
        }
    }
    out_cascade = cascade;

    float cos_theta = clamp(NdotL, 0.0, 1.0);
    float slope = clamp(sqrt(max(1.0 - cos_theta * cos_theta, 0.0)) / max(cos_theta, 0.001), 0.0, 3.5);

    vec3 world_dx = dFdx(world_pos);
    vec3 world_dy = dFdy(world_pos);

    ShadowCascade c0 = SetupCascade(cascade, world_pos, N, slope, world_dx, world_dy);
    float shadow = c0.in_bounds ? SampleCascadeShadow(cascade, c0) : 1.0;
    bool in_bounds = c0.in_bounds;
    out_coverage = in_bounds ? (1.0 - c0.edge_fade) : 0.0;

    // A receiver can fall outside its selected cascade's rect while still being
    // inside that cascade's depth range: the rect is fitted to the frustum
    // slice, so a receiver near the slice border can project just past it.
    // Walking forward to the first cascade that does cover the pixel keeps the
    // shadow continuous instead of punching a lit hole at the rect border.
    if (!in_bounds) {
        for (int next = cascade + 1; next < u_CascadeCount; ++next) {
            ShadowCascade cn = SetupCascade(next, world_pos, N, slope, world_dx, world_dy);
            if (cn.in_bounds) {
                shadow = SampleCascadeShadow(next, cn);
                in_bounds = true;
                out_cascade = next;
                out_coverage = 1.0 - cn.edge_fade;
                cascade = next;
                break;
            }
        }
        if (!in_bounds) {
            return 1.0;
        }
    }

    float split_val = u_CascadeSplits[cascade];
    float blend_range = max(split_val * 0.20, 0.1);
    bool has_next = cascade < u_CascadeCount - 1;

    // Border fade: near the rect edge the filter taps start clamping against the
    // border, which shows up as a hard band. Hand off to the next cascade there
    // instead. This reuses the single next-cascade sample the split blend already
    // takes, so it costs no extra texture fetches.
    float blend_t = 0.0;
    if (has_next) {
        blend_t = smoothstep(split_val - blend_range, split_val, view_depth);
        blend_t = max(blend_t, 1.0 - out_coverage);
    }

    if (has_next && blend_t > 0.0) {
        ShadowCascade c2 = SetupCascade(cascade + 1, world_pos, N, slope, world_dx, world_dy);
        if (c2.in_bounds) {
            float next_shadow = SampleCascadeShadow(cascade + 1, c2);
            shadow = mix(shadow, next_shadow, blend_t);
        }
    } else if (!has_next) {
        // Final cascade: smooth distance fade-out towards max shadow distance
        float fade_range = max_shadow_dist * 0.15;
        float fade_start = max_shadow_dist - fade_range;
        if (view_depth > fade_start) {
            float fade_t = clamp((view_depth - fade_start) / max(fade_range, 0.001), 0.0, 1.0);
            shadow = mix(shadow, 1.0, fade_t);
        }
    }

    return shadow;
}



#endif // CSM_SAMPLING_GLSL
