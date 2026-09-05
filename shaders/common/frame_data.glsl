// Frame Context Uniform Buffer Object (layout std140, binding = 0)
// Bound by all engine shaders; updated once per frame.

layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;

    vec4 u_CameraPos_Time;         // xyz = camera world pos, w = total elapsed time
    vec4 u_ScreenSize_Jitter;      // xy = width/height, zw = subpixel jitter (TAA)

    // Directional Sunlight & Sky
    vec4 u_SunDirection_Intensity; // xyz = normalized sun dir, w = sun lux
    vec4 u_SunColor_Ambient;        // rgb = sun light color, w = ambient factor

    // Cascaded Shadow Maps (CSM)
    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;          // Far split distances for cascades 0..3

    // Volumetric Atmospheric Fog
    vec4 u_FogColor_Density;        // rgb = fog color, w = fog density
    vec4 u_FogParams;               // x = height falloff, y = max distance, zw = unused
};
