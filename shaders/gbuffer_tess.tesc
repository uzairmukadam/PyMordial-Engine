#version 450 core

layout (vertices = 3) out;

in vec3 tc_Position[];
in vec3 tc_Normal[];
in vec2 tc_UV[];
in vec4 tc_Tangent[];
in flat uint tc_EntityID[];

out vec3 te_Position[];
out vec3 te_Normal[];
out vec2 te_UV[];
out vec4 te_Tangent[];
out flat uint te_EntityID[];

// Common Frame Data UBO (binding 0)
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

// 3-Tier Camera Radius Tessellation Uniforms
uniform int u_TessEnabled = 1;
uniform int u_FrustumCullEnabled = 1;
uniform float u_DispNearRadius = 8.0;   // High quality radius boundary
uniform float u_DispMidRadius = 25.0;   // Medium quality radius boundary (tessellation drops to 1.0 beyond)
uniform float u_TessMaxLevel = 24.0;    // Highest quality tessellation level (near)
uniform float u_TessMedLevel = 8.0;     // Medium quality tessellation level (mid)

float calc_edge_level(vec3 p0, vec3 p1) {
    vec3 mid = (p0 + p1) * 0.5;
    float dist = length(mid - u_CameraPos_Time.xyz);

    if (dist <= u_DispNearRadius) {
        return u_TessMaxLevel;
    } else if (dist <= u_DispMidRadius) {
        float t = (dist - u_DispNearRadius) / max(u_DispMidRadius - u_DispNearRadius, 0.001);
        return mix(u_TessMaxLevel, u_TessMedLevel, t);
    } else {
        return 1.0; // Outside medium radius: base mesh only
    }
}

// Conservative GPU Frustum Culling with Polygon Clipping Safety:
// Evaluates the patch against the 6 camera frustum planes in View Space.
// A patch is ONLY culled if ALL 3 control points are strictly outside the SAME frustum plane,
// taking into account maximum possible displacement padding (1.5 meters).
// If a patch even partially intersects the frustum (or crosses the near plane), it is NEVER culled.
// Instead, it proceeds to evaluation and is seamlessly clipped by the GPU's fixed-function
// hardware polygon clipper. This ensures models never disappear or pop when partially in view.
bool is_patch_culled(vec3 p0, vec3 p1, vec3 p2) {
    // Transform patch vertices to View Space
    vec3 v0 = (u_View * vec4(p0, 1.0)).xyz;
    vec3 v1 = (u_View * vec4(p1, 1.0)).xyz;
    vec3 v2 = (u_View * vec4(p2, 1.0)).xyz;

    // Conservative padding for displacement (1.5 meters)
    const float pad = 1.5;
    const float near_dist = 0.1;

    // 1. Near Plane Test:
    // In OpenGL view space, camera looks down -Z.
    // Near plane is at z = -near_dist (-0.1). Points in front have z < -near_dist.
    // If ALL 3 vertices are behind the camera (z > -near_dist + pad), the patch is behind the camera.
    if (v0.z > -near_dist + pad && v1.z > -near_dist + pad && v2.z > -near_dist + pad) {
        return true;
    }

    // If ANY vertex crosses or is behind the camera plane (z > -near_dist),
    // the patch spans the near clipping plane. DO NOT cull it against side planes in view space;
    // preserve it so hardware polygon clipping can slice it cleanly!
    if (v0.z > -near_dist || v1.z > -near_dist || v2.z > -near_dist) {
        return false;
    }

    // 2. Side Planes (Left, Right, Bottom, Top):
    // Projection matrix scaling terms:
    float p00 = u_Projection[0][0]; // cot(fovy/2) / aspect
    float p11 = u_Projection[1][1]; // cot(fovy/2)

    float lenX = sqrt(p00 * p00 + 1.0);
    float lenY = sqrt(p11 * p11 + 1.0);

    // Inward-pointing normalized plane normals in view space (camera looks down -Z)
    vec3 nLeft   = vec3( p00,  0.0, -1.0) / lenX;
    vec3 nRight  = vec3(-p00,  0.0, -1.0) / lenX;
    vec3 nBottom = vec3( 0.0,  p11, -1.0) / lenY;
    vec3 nTop    = vec3( 0.0, -p11, -1.0) / lenY;

    // Signed distance to plane: dot(normal, v). Negative = outside.
    // Left Plane
    if (dot(nLeft, v0) < -pad && dot(nLeft, v1) < -pad && dot(nLeft, v2) < -pad) return true;
    // Right Plane
    if (dot(nRight, v0) < -pad && dot(nRight, v1) < -pad && dot(nRight, v2) < -pad) return true;
    // Bottom Plane
    if (dot(nBottom, v0) < -pad && dot(nBottom, v1) < -pad && dot(nBottom, v2) < -pad) return true;
    // Top Plane
    if (dot(nTop, v0) < -pad && dot(nTop, v1) < -pad && dot(nTop, v2) < -pad) return true;

    // 3. Far Plane Test:
    const float far_dist = 1000.0;
    if (-v0.z > far_dist + pad && -v1.z > far_dist + pad && -v2.z > far_dist + pad) {
        return true;
    }

    // Patch intersects or is inside the frustum:
    // Retain patch; GPU hardware polygon clipper will clip any portions outside the viewport!
    return false;
}

void main() {
    // Pass-through per-vertex attributes
    te_Position[gl_InvocationID] = tc_Position[gl_InvocationID];
    te_Normal[gl_InvocationID] = tc_Normal[gl_InvocationID];
    te_UV[gl_InvocationID] = tc_UV[gl_InvocationID];
    te_Tangent[gl_InvocationID] = tc_Tangent[gl_InvocationID];
    te_EntityID[gl_InvocationID] = tc_EntityID[gl_InvocationID];

    // Compute dynamic tessellation levels on first invocation
    if (gl_InvocationID == 0) {
        if (u_TessEnabled == 1) {
            // Conservative frustum culling
            if (u_FrustumCullEnabled == 1 && is_patch_culled(tc_Position[0], tc_Position[1], tc_Position[2])) {
                gl_TessLevelOuter[0] = 0.0;
                gl_TessLevelOuter[1] = 0.0;
                gl_TessLevelOuter[2] = 0.0;
                gl_TessLevelInner[0] = 0.0;
            } else {
                float l0 = calc_edge_level(tc_Position[1], tc_Position[2]);
                float l1 = calc_edge_level(tc_Position[2], tc_Position[0]);
                float l2 = calc_edge_level(tc_Position[0], tc_Position[1]);

                gl_TessLevelOuter[0] = l0;
                gl_TessLevelOuter[1] = l1;
                gl_TessLevelOuter[2] = l2;
                gl_TessLevelInner[0] = (l0 + l1 + l2) * 0.333333;
            }
        } else {
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = 1.0;
            gl_TessLevelOuter[2] = 1.0;
            gl_TessLevelInner[0] = 1.0;
        }
    }
}
