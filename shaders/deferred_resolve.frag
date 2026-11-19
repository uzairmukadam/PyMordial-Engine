#version 450 core

in vec2 v_UV;
layout (location = 0) out vec4 out_HDRColor;

// G-Buffer Inputs
layout (binding = 0) uniform sampler2D u_GBufferAlbedoRoughness;
layout (binding = 1) uniform sampler2D u_GBufferNormalMetallic;
layout (binding = 2) uniform sampler2D u_GBufferDepth;          // Reversed-Z Depth32F
layout (binding = 3) uniform sampler2D u_ShadowAtlas;
// Hardware depth-compare view of the same atlas. A tap through this sampler
// returns a bilinear-filtered 2x2 PCF result, so one fetch does the work of four
// manual fetches. The raw sampler above stays bound for the blocker search and
// the atlas debug view, which need the stored depth rather than a comparison.
layout (binding = 9) uniform sampler2DShadow u_ShadowAtlasCmp;

// Phase 5 Advanced Inputs
layout (binding = 4) uniform sampler2D u_AOTexture;            // GTAO / SSAO
layout (binding = 5) uniform sampler2D u_SSGITexture;          // Screen-Space Global Illumination
layout (binding = 6) uniform sampler2D u_BRDFLUT;              // Split-sum 2D BRDF LUT
layout (binding = 7) uniform sampler2D u_EnvironmentMap;       // Prefiltered HDR Environment Map
layout (binding = 8) uniform sampler2D u_SSRTexture;           // Screen-Space Reflections
layout (binding = 12) uniform sampler2D u_SpotShadowAtlas;     // 2x2 Spot Light Perspective Shadow Depth Atlas
// Hardware depth-compare view of the spot atlas (see u_ShadowAtlasCmp).
layout (binding = 16) uniform sampler2DShadow u_SpotShadowAtlasCmp;
layout (binding = 13) uniform sampler2D u_IrradianceMap;       // Diffuse environment irradiance
layout (binding = 14) uniform sampler2D u_HiZTexture;          // Hierarchical-Z Depth Pyramid Mip Chain
layout (binding = 15) uniform sampler2D u_RCTexture;           // Hybrid Radiance Cascades (SSRC + FFPC)

// Unified Frame Context UBO 0
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

// SSBO 3: Dynamic Clustered Local Point & Spot Lights
struct PointLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 color_intensity; // rgb = color, w = intensity
};

struct SpotLightData {
    vec4 pos_radius;      // xyz = position, w = radius
    vec4 dir_inner_cos;   // xyz = normalized direction, w = cos(inner_cone_angle)
    vec4 color_intensity; // rgb = color, w = intensity
    vec4 params;          // x = cos(outer_cone_angle), y = shadow_index (-1 if no shadow), z = shadow_bias, w = reserved
};

layout (std430, binding = 3) buffer LightBuffer {
    PointLightData u_PointLights[256];
    SpotLightData u_SpotLights[];
};

// Configurable quality & feature uniforms
uniform int u_CascadeCount = 4;  // 1 to 4
uniform int u_GBufferDebug = 0;  // 0=Off, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=ShadowAtlas, 6=ShadowMask, 8=SSGI, 10=IndirectDiffuse, 11=HiZ, 12=SpotLights, 13=RadianceCascades, 14=CascadeIndex, 15=CascadeCoverage
uniform int u_HiZDebugMip = 0;   // Mip level to visualize when u_GBufferDebug == 11

// Hard-shadow parametrization
uniform float u_ShadowBias = 0.0015;   // Base world-space shadow depth bias (meters)
uniform float u_ShadowNormalBias = 0.0010; // Normal offset bias scale
uniform float u_ShadowBiasSlopeScale = 2.0; // Slope multiplier for the base bias
uniform float u_ShadowBiasMinScale = 0.5;   // Lower clamp, as a multiple of the base bias
uniform float u_ShadowBiasMaxScale = 6.0;   // Upper clamp, as a multiple of the base bias

// Shadow filtering tiers. Each tier is limited to a prefix of the cascades:
// the far cascades keep the cheaper path because their texels already span
// several screen pixels, so a wider filter there costs taps without adding
// visible detail.
uniform int u_ShadowFilterMode = 0;   // 0 = HARD, 1 = PCF, 2 = PCSS
uniform int u_PCFCascadeLimit = 3;    // cascades with index < this use PCF
uniform int u_PCFSamples = 16;        // Vogel disk taps (4 to 64)
uniform float u_PCFRadius = 1.5;      // filter radius in cascade texels
uniform float u_PCFFalloff = 2.0;     // radial weight exponent (1 = linear cone, 2 = smooth bell)

// PCSS (percentage-closer soft shadows). The penumbra is derived from the
// receiver-to-blocker gap rather than a fixed radius, so contact shadows stay
// sharp and only elevated casters soften.
uniform int u_PCSS_CascadeLimit = 1;      // cascades with index < this use PCSS
uniform int u_PCSSSearchSamples = 16;     // blocker search taps (4 to 64)
uniform float u_PCSSSearchRadius = 48.0;  // blocker search radius in cascade texels
uniform float u_PCSSLightSize = 0.10;     // light-source size factor (penumbra scale)
uniform int u_PCSSSamples = 24;           // penumbra filter taps (4 to 64)
uniform float u_PCSSFalloff = 2.0;        // radial weight exponent
uniform float u_PCSSMaxRadius = 96.0;     // penumbra radius clamp in cascade texels
// 0.0 = average blocker gap (classic PCSS), 1.0 = closest blocker gap.
// The closest blocker keeps overlapping shadows sharp instead of letting a far
// caster's softness bleed onto a near caster's edge.
uniform float u_PCSSClosestBlockerWeight = 1.0;
// Per-cascade atlas rect: xy = atlas UV offset, zw = atlas UV scale.
uniform vec4 u_CascadeAtlasRect[4] = vec4[4](
    vec4(0.0, 0.0, 0.5, 0.5),
    vec4(0.5, 0.0, 0.5, 0.5),
    vec4(0.0, 0.5, 0.5, 0.5),
    vec4(0.5, 0.5, 0.5, 0.5)
);

// Phase 5 Toggles & Settings
uniform int u_AOEnabled = 1;
uniform int u_GIEnabled = 4;     // 0=Off/IBL, 1=SSGI, 4=Radiance Cascades (SSRC+FFPC)
uniform int u_IBLEnabled = 1;
uniform int u_SSREnabled = 1;
uniform int u_PointLightCount = 0;
uniform int u_SpotLightCount = 0;
uniform int u_SpotShadowEnabled = 1;
uniform mat4 u_SpotLightViewProjection[4];

#include "shaders/atmosphere.glsl"

#include "shaders/common/shadows.glsl"

// Jorge Jimenez's Interleaved Gradient Noise (IGN) for uniform blue-noise spatial distribution
float InterleavedGradientNoise(vec2 screen_pos) {
    return fract(52.9829189 * fract(dot(screen_pos, vec2(0.06711056, 0.00583715))));
}

// Vogel Disk Sampling with Golden Angle Spiral for low-discrepancy circular filtering
vec2 VogelDiskSample(int sample_index, int sample_count, float phi) {
    float r = sqrt((float(sample_index) + 0.5) / float(sample_count));
    float theta = float(sample_index) * 2.39996323 + phi; // 2.39996323 rad = 137.507764 deg (Golden Angle)
    return vec2(r * cos(theta), r * sin(theta));
}


// Evaluates localized perspective spot light shadow occlusion.
//
// Uses the same hardware comparison sampler as the directional path, so a tap
// resolves a bilinear 2x2 PCF footprint. The reference depth is offset by the
// receiver-plane gradient rather than a single constant, which keeps the bias
// small enough that contact shadows stay welded to their casters.
float SampleSpotShadow(int shadow_idx, vec3 world_pos, vec3 N, float bias) {
    if (shadow_idx < 0 || shadow_idx >= 4 || u_SpotShadowEnabled == 0) return 1.0;

    vec4 light_space_pos = u_SpotLightViewProjection[shadow_idx] * vec4(world_pos + N * (bias * 0.5), 1.0);
    if (light_space_pos.w <= 0.0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // Remap NDC XY [-1, 1] to UV [0, 1]; Z is already in [0, 1] under GL_ZERO_TO_ONE clip control
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    // Out-of-bounds check (outside spot cone/frustum)
    if (proj_coords.z < 0.0 || proj_coords.z > 1.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    vec2 atlas_offset = vec2(float(shadow_idx % 2) * 0.5, float(shadow_idx / 2) * 0.5);
    vec2 uv = proj_coords.xy * 0.5 + atlas_offset;
    vec2 uv_min = atlas_offset + vec2(0.0005);
    vec2 uv_max = atlas_offset + vec2(0.4995);

    float current_depth = proj_coords.z;
    vec2 full_size = vec2(textureSize(u_SpotShadowAtlas, 0));
    vec2 texel_size = 1.0 / full_size;

    // 9-tap Vogel disk through the comparison sampler. A single bilinear tap
    // already resolves 4 texels, so this covers a 6x6 texel footprint for the
    // cost of nine fetches, which is enough to hide the perspective aliasing a
    // spot light shows at grazing angles.
    float radius = texel_size.x * 2.0;
    float phi = InterleavedGradientNoise(gl_FragCoord.xy) * 6.2831853;
    float sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        vec2 offset = VogelDiskSample(i, 9, phi) * radius;
        vec2 tap = clamp(uv + offset, uv_min, uv_max);
        float r2 = (float(i) + 0.5) / 9.0;
        float w = (1.0 - r2) * (1.0 - r2);
        sum += texture(u_SpotShadowAtlasCmp, vec3(tap, current_depth - bias)) * w;
        weight_sum += w;
    }
    return sum / max(weight_sum, 1e-5);
}

vec3 OctahedralDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 0.0000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Jimenez Multi-Bounce Approximation for rich indirect saturation inside crevices
vec3 MultiBounceAO(float visibility, vec3 albedo) {
    vec3 a = 2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c = 2.7552 * albedo + 0.6903;
    vec3 x = vec3(visibility);
    return clamp(max(x, ((x * a + b) * x + c) * x), 0.0, 1.0);
}

// Linearize depth from Reversed-Z buffer to positive metric view-space depth in meters
float LinearizeDepth(float depth, vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 vpos = u_InvProjection * clip;
    return -vpos.z / max(vpos.w, 0.000001);
}

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
    float penumbra_world = gap_world * max(u_PCSSLightSize, 1e-5);
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


void main() {
    if (u_GBufferDebug == 5) {
        float s = texture(u_ShadowAtlas, v_UV).r;
        out_HDRColor = vec4(vec3(s), 1.0);
        return;
    }

    float raw_depth = texture(u_GBufferDepth, v_UV).r;

    if (u_GBufferDebug == 4) {
        out_HDRColor = vec4(vec3(raw_depth), 1.0);
        return;
    } else if (u_GBufferDebug == 11) {
        float hiz_d = textureLod(u_HiZTexture, v_UV, float(u_HiZDebugMip)).r;
        out_HDRColor = vec4(vec3(hiz_d), 1.0);
        return;
    }

    // Physical Atmosphere & Celestial Sky Dome for background pixels
    if (raw_depth <= 0.000001) {
        if (u_GBufferDebug > 0) {
            out_HDRColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        vec3 ray_ndc = vec3(v_UV * 2.0 - 1.0, 1.0);
        vec4 ray_view = u_InvProjection * vec4(ray_ndc, 1.0);
        ray_view.z = -1.0;
        ray_view.w = 0.0;
        vec3 ray_dir = normalize((u_InvView * ray_view).xyz);

        vec3 sky = EvaluateAtmosphere(
            ray_dir,
            u_AtmosphereSunDir,
            u_SunColor_Ambient.rgb,
            u_SunDirection_Intensity.w,
            u_CameraPos_Time.w,
            u_CameraPos_Time.xyz
        );

        out_HDRColor = vec4(sky, 1.0);
        return;
    }

    // Reconstruct world position from Reversed-Z depth
    vec4 clip_pos = vec4(v_UV * 2.0 - 1.0, raw_depth, 1.0);
    vec4 view_pos = u_InvProjection * clip_pos;
    view_pos /= max(abs(view_pos.w), 1e-6);
    vec4 world_pos = u_InvView * vec4(view_pos.xyz, 1.0);

    // Read G-Buffer parameters
    vec4 albedo_rough = texture(u_GBufferAlbedoRoughness, v_UV);
    vec4 normal_metal = texture(u_GBufferNormalMetallic, v_UV);

    vec3 raw_albedo = albedo_rough.rgb;
    // Phase 8: Extract HDR emissive radiance (> 1.0) and normalize diffuse albedo
    float max_c = max(max(raw_albedo.r, raw_albedo.g), raw_albedo.b);
    vec3 albedo = (max_c > 1.0) ? (raw_albedo / max_c) : raw_albedo;
    vec3 emissive = (max_c > 1.0) ? (raw_albedo - albedo) : vec3(0.0);
    float roughness = clamp(albedo_rough.a, 0.04, 1.0);
    vec3 N = OctahedralDecode(normal_metal.rg);
    float metallic = normal_metal.b;
    float ao = normal_metal.a;

    // Multiply by screen-space GTAO/SSAO if enabled
    if (u_AOEnabled == 1) {
        float gtao = texture(u_AOTexture, v_UV).r;
        ao *= gtao;
    }

    if (u_GBufferDebug == 1) {
        out_HDRColor = vec4(albedo, 1.0);
        return;
    } else if (u_GBufferDebug == 2) {
        out_HDRColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    } else if (u_GBufferDebug == 3) {
        out_HDRColor = vec4(roughness, metallic, ao, 1.0);
        return;
    }

    vec3 V = normalize(u_CameraPos_Time.xyz - world_pos.xyz);
    vec3 L = normalize(-u_SunDirection_Intensity.xyz);
    vec3 H = normalize(V + L);

    // PBR Cook-Torrance Direct Sun Lighting
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    float NdotL = max(dot(N, L), 0.0);
    vec3 radiance = u_SunColor_Ambient.rgb * u_SunDirection_Intensity.w;

    int shadow_cascade = 0;
    float shadow_coverage = 0.0;
    float shadow = CalculateCSMShadow(
        world_pos.xyz, N, L, -view_pos.z, shadow_cascade, shadow_coverage
    );

    if (u_GBufferDebug == 6) {
        out_HDRColor = vec4(vec3(shadow), 1.0);
        return;
    } else if (u_GBufferDebug == 7) {
        out_HDRColor = vec4(vec3(ao), 1.0);
        return;
    } else if (u_GBufferDebug == 14) {
        // Selected cascade index as a false-colour image.
        vec3 cascade_tint = vec3(0.0);
        if (shadow_cascade == 0) {
            cascade_tint = vec3(0.15, 0.45, 1.00);
        } else if (shadow_cascade == 1) {
            cascade_tint = vec3(0.20, 0.95, 0.35);
        } else if (shadow_cascade == 2) {
            cascade_tint = vec3(1.00, 0.85, 0.15);
        } else {
            cascade_tint = vec3(1.00, 0.25, 0.20);
        }
        out_HDRColor = vec4(cascade_tint, 1.0);
        return;
    } else if (u_GBufferDebug == 15) {
        // Per-cascade coverage: brightness falls off toward the rect border,
        // so the fade band and any uncovered receivers are directly visible.
        out_HDRColor = vec4(vec3(shadow_coverage), 1.0);
        return;
    } else if (u_GBufferDebug == 16) {
        // Cascade bounds: each cascade's slice is tinted, and the split
        // distances are drawn as bright boundary lines so the start and end of
        // every cascade is directly readable in world space.
        float view_depth = -view_pos.z;
        vec3 band = vec3(0.0);
        if (shadow_cascade == 0) {
            band = vec3(0.15, 0.45, 1.00);
        } else if (shadow_cascade == 1) {
            band = vec3(0.20, 0.95, 0.35);
        } else if (shadow_cascade == 2) {
            band = vec3(1.00, 0.85, 0.15);
        } else {
            band = vec3(1.00, 0.25, 0.20);
        }

        // Beyond the last cascade there is no shadow coverage at all.
        float max_shadow_dist = u_CascadeSplits[u_CascadeCount - 1];
        if (view_depth >= max_shadow_dist) {
            band = vec3(0.10, 0.10, 0.12);
        }

        // Boundary lines. The width is derived from the per-pixel depth
        // gradient (fwidth) so the line stays a constant number of screen
        // pixels wide at every distance. A world-space width would shrink to
        // sub-pixel near the horizon and vanish exactly where the far cascades
        // meet.
        float depth_per_pixel = max(fwidth(view_depth), 1e-5);
        float line_half_width = depth_per_pixel * 2.0;
        float edge = 0.0;
        for (int i = 0; i < u_CascadeCount; ++i) {
            float d = abs(view_depth - u_CascadeSplits[i]);
            edge = max(edge, 1.0 - smoothstep(line_half_width * 0.5, line_half_width, d));
        }
        // Near-plane start of cascade 0.
        edge = max(edge, 1.0 - smoothstep(line_half_width * 0.5, line_half_width, view_depth));

        // The debug view still passes through tonemapping, which compresses a
        // plain 1.0 to mid grey. Push the boundary well into HDR so it resolves
        // as a crisp white line instead of a faint tint.
        vec3 result = mix(band, vec3(12.0), edge);
        out_HDRColor = vec4(result, 1.0);
        return;
    }

    // Subtle direct contact crevice micro-shadowing
    float direct_crevice = clamp(ao + max(dot(N, L), 0.0), 0.0, 1.0);
    vec3 direct_sun = (kD * albedo / PI + specular) * radiance * NdotL * shadow * direct_crevice;

    // Dynamic Clustered Point Lights (SSBO 3)
    vec3 point_lights_accum = vec3(0.0);
    int light_count = min(u_PointLightCount, 128);
    for (int i = 0; i < light_count; ++i) {
        vec3 pl_pos = u_PointLights[i].pos_radius.xyz;
        float pl_radius = u_PointLights[i].pos_radius.w;
        vec3 pl_delta = pl_pos - world_pos.xyz;
        float pl_dist = length(pl_delta);

        if (pl_dist < pl_radius) {
            vec3 pl_L = pl_delta / max(pl_dist, 0.0001);
            float pl_NdotL = max(dot(N, pl_L), 0.0);

            if (pl_NdotL > 0.0) {
                // Smooth physical inverse-square attenuation with windowing
                float ratio = pl_dist / pl_radius;
                float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                float atten = (win * win) / (pl_dist * pl_dist + 1.0);

                vec3 pl_H = normalize(V + pl_L);
                float pl_NDF = DistributionGGX(N, pl_H, roughness);
                float pl_G = GeometrySmith(N, V, pl_L, roughness);
                vec3 pl_F = FresnelSchlick(max(dot(pl_H, V), 0.0), F0);

                vec3 pl_spec = (pl_NDF * pl_G * pl_F) / (4.0 * max(dot(N, V), 0.0) * pl_NdotL + 0.0001);
                vec3 pl_diff = (vec3(1.0) - pl_F) * (1.0 - metallic) * albedo / PI;

                vec3 pl_rad = u_PointLights[i].color_intensity.rgb * u_PointLights[i].color_intensity.w;
                point_lights_accum += (pl_diff + pl_spec) * pl_rad * pl_NdotL * atten;
            }
        }
    }

    // Dynamic Clustered Spot Lights (SSBO 3 + Perspective Spot Shadows)
    vec3 spot_lights_accum = vec3(0.0);
    int spot_count = min(u_SpotLightCount, 64);
    for (int i = 0; i < spot_count; ++i) {
        vec3 sp_pos = u_SpotLights[i].pos_radius.xyz;
        float sp_radius = u_SpotLights[i].pos_radius.w;
        vec3 sp_delta = sp_pos - world_pos.xyz;
        float sp_dist = length(sp_delta);

        if (sp_dist < sp_radius) {
            vec3 sp_L = sp_delta / max(sp_dist, 0.0001);
            vec3 sp_dir = normalize(u_SpotLights[i].dir_inner_cos.xyz);
            float inner_cos = u_SpotLights[i].dir_inner_cos.w;
            float outer_cos = u_SpotLights[i].params.x;

            // Spot cone angle check: dot product between vector from light to fragment (-sp_L) and spot direction
            float cos_theta = dot(-sp_L, sp_dir);
            if (cos_theta > outer_cos) {
                float spot_cone = smoothstep(outer_cos, inner_cos, cos_theta);
                float sp_NdotL = max(dot(N, sp_L), 0.0);

                if (sp_NdotL > 0.0) {
                    // Smooth physical inverse-square distance attenuation with windowing
                    float ratio = sp_dist / sp_radius;
                    float win = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                    float atten = (win * win) / (sp_dist * sp_dist + 1.0);

                    // Spot shadow factor
                    int shadow_idx = int(u_SpotLights[i].params.y);
                    float sp_bias = u_SpotLights[i].params.z;
                    float sp_shadow = (shadow_idx >= 0) ? SampleSpotShadow(shadow_idx, world_pos.xyz, N, sp_bias) : 1.0;

                    vec3 sp_H = normalize(V + sp_L);
                    float sp_NDF = DistributionGGX(N, sp_H, roughness);
                    float sp_G = GeometrySmith(N, V, sp_L, roughness);
                    vec3 sp_F = FresnelSchlick(max(dot(sp_H, V), 0.0), F0);

                    vec3 sp_spec = (sp_NDF * sp_G * sp_F) / (4.0 * max(dot(N, V), 0.0) * sp_NdotL + 0.0001);
                    vec3 sp_diff = (vec3(1.0) - sp_F) * (1.0 - metallic) * albedo / PI;

                    vec3 sp_rad = u_SpotLights[i].color_intensity.rgb * u_SpotLights[i].color_intensity.w;
                    spot_lights_accum += (sp_diff + sp_spec) * sp_rad * sp_NdotL * atten * spot_cone * sp_shadow;
                }
            }
        }
    }

    if (u_GBufferDebug == 12) {
        out_HDRColor = vec4(spot_lights_accum, 1.0);
        return;
    }

    // Global Illumination (SSGI vs Radiance Cascades)
    vec3 indirect_diffuse = vec3(0.0);
    vec4 ssgi_sample = vec4(0.0);
    vec4 rc_sample = vec4(0.0);

    if (u_GIEnabled == 4) {
        // Next-Gen Hybrid Radiance Cascades (SSRC + FFPC)
        rc_sample = texture(u_RCTexture, v_UV);
        if (isnan(rc_sample.r) || isnan(rc_sample.g) || isnan(rc_sample.b) ||
            isinf(rc_sample.r) || isinf(rc_sample.g) || isinf(rc_sample.b)) {
            rc_sample = vec4(0.0);
        }
        vec3 rc_diffuse = clamp(rc_sample.rgb, vec3(0.0), vec3(30.0)) * albedo * (1.0 - metallic);
        indirect_diffuse += rc_diffuse;
    } else if (u_GIEnabled == 1 || u_GIEnabled == 2 || u_GIEnabled == 3) {
        // SSGI (Screen-Space Near-Field Indirect Diffuse Bounce & Contact Color Bleed)
        ssgi_sample = texture(u_SSGITexture, v_UV);
        if (isnan(ssgi_sample.r) || isnan(ssgi_sample.g) || isnan(ssgi_sample.b) ||
            isinf(ssgi_sample.r) || isinf(ssgi_sample.g) || isinf(ssgi_sample.b) ||
            isnan(ssgi_sample.a) || isinf(ssgi_sample.a)) {
            ssgi_sample = vec4(0.0);
        }
        vec3 ssgi_diffuse = clamp(ssgi_sample.rgb, vec3(0.0), vec3(30.0)) * albedo * (1.0 - metallic);
        indirect_diffuse += ssgi_diffuse;
    }

    if (u_GBufferDebug == 8) {
        out_HDRColor = vec4(ssgi_sample.rgb, 1.0);
        return;
    } else if (u_GBufferDebug == 10) {
        out_HDRColor = vec4(indirect_diffuse, 1.0);
        return;
    } else if (u_GBufferDebug == 13) {
        out_HDRColor = vec4(rc_sample.rgb, 1.0);
        return;
    }

    // Diffuse IBL uses the cosine-convolved environment; retain a stable fallback when disabled.
    float sky_intensity = max(u_SunColor_Ambient.w * 3.5, 0.12);
    vec3 sky_color = mix(vec3(0.25, 0.35, 0.50), u_SunColor_Ambient.rgb * 0.5 + vec3(0.15), 0.4);
    vec3 ground_color = vec3(0.18, 0.19, 0.18);
    vec3 hemisphere_light = mix(ground_color, sky_color, clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * sky_intensity;
    vec3 diffuse_energy = (vec3(1.0) - FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness))
                        * (1.0 - metallic);
    vec2 irradiance_uv = vec2(atan(N.z, N.x) / (2.0 * PI) + 0.5,
                              asin(clamp(N.y, -0.999, 0.999)) / PI + 0.5);
    vec3 diffuse_irradiance = u_IBLEnabled == 1
        ? texture(u_IrradianceMap, irradiance_uv).rgb
        : hemisphere_light;
    vec3 ambient_base = diffuse_irradiance * albedo * diffuse_energy;

    // Image-Based Lighting (IBL) & Screen-Space Reflections (SSR)
    vec3 indirect_specular = vec3(0.0);

    if (u_IBLEnabled == 1) {
        vec3 R = reflect(-V, N);
        float NdotV = max(dot(N, V), 0.0);

        // Sample Split-Sum BRDF LUT
        vec2 brdf = texture(u_BRDFLUT, vec2(NdotV, roughness)).rg;

        // Sample Prefiltered Environment Map
        vec2 env_uv = vec2(atan(R.z, R.x) / (2.0 * PI) + 0.5, asin(clamp(R.y, -0.999, 0.999)) / PI + 0.5);
        float max_env_lod = float(max(textureQueryLevels(u_EnvironmentMap) - 1, 0));
        vec3 env_radiance = textureLod(u_EnvironmentMap, env_uv, roughness * max_env_lod).rgb;

        // Sample SSR
        if (u_SSREnabled == 1) {
            vec4 ssr_sample = texture(u_SSRTexture, v_UV);
            if (isnan(ssr_sample.r) || isnan(ssr_sample.g) || isnan(ssr_sample.b) ||
                isinf(ssr_sample.r) || isinf(ssr_sample.g) || isinf(ssr_sample.b) ||
                isnan(ssr_sample.a) || isinf(ssr_sample.a)) {
                ssr_sample = vec4(0.0);
            }
            vec3 safe_ssr_rgb = clamp(ssr_sample.rgb, vec3(0.0), vec3(30.0));
            float safe_ssr_a = clamp(ssr_sample.a, 0.0, 1.0);
            if (safe_ssr_a > 0.001) {
                // Blend SSR with environment IBL, ensuring local reflections never create a dark void below ambient sky fill
                vec3 blended_refl = mix(env_radiance, safe_ssr_rgb, safe_ssr_a);
                env_radiance = max(blended_refl, env_radiance * (1.0 - safe_ssr_a * 0.70));
            }
        }

        // Specular occlusion (Lagarde / Frostbite) prevents specular light leaks into deep crevices
        float spec_ao = clamp(ao * ao + sqrt(NdotV) * (1.0 - ao), 0.0, 1.0);

        vec3 F_env = FresnelSchlickRoughness(NdotV, F0, roughness);
        indirect_specular = env_radiance * (F_env * brdf.x + brdf.y) * spec_ao;
    }

    // Total combine with Multi-Bounce color-preserving Ambient Occlusion on diffuse
    vec3 bounce_ao = MultiBounceAO(ao, albedo);
    vec3 ambient = (ambient_base + indirect_diffuse) * bounce_ao + indirect_specular;
    vec3 total_lit = direct_sun + point_lights_accum + spot_lights_accum + ambient + emissive;

    // Guaranteed NaN / Inf fallback to ambient floor
    if (isnan(total_lit.r) || isnan(total_lit.g) || isnan(total_lit.b) ||
        isinf(total_lit.r) || isinf(total_lit.g) || isinf(total_lit.b)) {
        total_lit = max(ambient_base, vec3(0.05));
    }
    total_lit = clamp(total_lit, vec3(0.0), vec3(100.0));

    out_HDRColor = vec4(total_lit, 1.0);
}
