# PyMordial Engine: Technical Feature Specification & Architecture Reference

PyMordial Engine is a high-performance, modular 3D game engine built in Python 3.11+ using ModernGL (OpenGL 4.5 Core Profile), NumPy, Rapier physics, and PyGame-CE. Designed from the ground up for AAA-grade visual fidelity, zero-allocation runtime performance, and clean isolation between core engine infrastructure and project-specific gameplay logic.

---

## 1. Engine Core & Architecture

### 1.1 Fixed-Timestep Game Loop (`engine.core.loop.EngineLoop`)
- **Semi-implicit Euler integration** with decoupled simulation (`fixed_update`) and variable presentation (`update`, `render`).
- **Configurable fixed simulation tick rate** (default 60 Hz).
- **Accumulator clamp & max frame step caps** to prevent spiral-of-death under CPU stalls.
- **Interpolation alpha (`0.0 <= alpha <= 1.0`)** passed to rendering passes for jitter-free visual smoothing between physics ticks.

### 1.2 Cache-Aligned High-Throughput ECS (`engine.core.ecs.EntityManager`)
- **Contiguous NumPy structured arrays** acting as Structure-of-Arrays (SoA) for transforms and PBR material flags.
- **Direct GPU SSBO DMA transfers**: Transform matrices (`mat4x4`, 64 bytes) and material indices/tiling flags (`uvec4`, 16 bytes) are written directly from memory views to OpenGL Shader Storage Buffer Objects without intermediate Python object instantiation.
- **Zero-allocation entity spawning and despawning** via free-list recycling index pools.

### 1.3 Event Bus (`engine.events`)
- Fast, type-safe synchronous publisher-subscriber event dispatcher.
- Built-in event types for window resize, focus changes, input actions, and custom project triggers.

---

## 2. Rendering Pipeline (`engine.gfx.pipeline.RenderPipeline`)

The rendering architecture implements a state-of-the-art deferred rendering pipeline with hardware-accelerated indirect drawing and comprehensive physical lighting.

### 2.1 Multi-Draw Indirect (MDI) & MegaBuffer
- **Unified MegaBuffer (`engine.gfx.mega_buffer.MegaBuffer`)**: Single giant vertex and index buffer storing all static and dynamic project geometry.
- **Zero Draw-Call Overhead**: Entire scenes are dispatched in a single GPU call via `glMultiDrawElementsIndirect`.
- **Custom 32-Byte Vertex Layout (`engine.assets.mesh_format.VERTEX_DTYPE`)**:
  - Position: `float32[3]` (12 bytes, offset 0)
  - Normal: `float16[4]` (8 bytes, offset 12) — `W` carries the bitangent sign
  - UV0: `float16[2]` (4 bytes, offset 20)
  - Tangent: `float16[4]` (8 bytes, offset 24) — `W` carries handedness
  - Cache-aligned to 32 bytes (two vertices per 64-byte L1 cache line) for peak GPU memory fetch coalescing.
  - The layout is asserted at import time (`VERTEX_DTYPE.itemsize == 32`), so any change that breaks the stride fails fast rather than silently corrupting geometry.

### 2.2 Reversed-Z Floating Point Depth
- Depth range: `1.0` (near plane) to `0.0` (far plane / infinity).
- `GL_GREATER` depth comparison eliminating Z-fighting across vast outdoor landscapes.

### 2.3 Cascaded Shadow Maps (CSM) (`engine.gfx.shadow_csm.CascadedShadowMap`)
- Up to 4 shadow cascades with logarithmic/linear split blending.
- Stable frustum-slice radii with overlapping receiver regions and texel-snapped light-space bounds reduce cascade shimmer and popping.
- Scene AABBs upstream of each receiver slice extend cascade depth ranges so relevant off-camera casters remain visible.
- Texel-aware slope bias, PCF/PCSS sampling, and depth-aware blending across cascade boundaries reduce distant acne and filtering seams.
- **Split modes** (`RenderConfig.csm_split_mode`): `PSSM` (default), `GEOMETRIC`, `UNIFORM`. The default blend is automatically reduced so adjacent cascade spans never exceed a 2.5x ratio, which keeps texel density from dropping in visible steps.
- **Per-cascade distance** (`RenderConfig.csm_cascade_distance_scale`): a 4-tuple of far-distance multipliers so each cascade's reach can be tuned independently without changing the overall shadow distance.
- **Filter modes** (`RenderConfig.shadow_mode`): `HARD`, `PCF`, `PCSS`, `PCSS_ULTRA`. PCSS derives its penumbra from the physical solar angular radius (`RenderConfig.sun_angular_radius`, default 0.00465 rad) rather than an arbitrary constant, and rotates its Vogel disk per pixel so banding becomes TAA-resolvable noise. One cascade texel is `1 / atlas_size` in atlas UV — using `2 / atlas_size` doubles every filter radius and is the classic cause of blurry, banded PCSS.
- **Per-cascade atlas allocation** (`RenderConfig.csm_nonuniform_atlas`, default off): near cascades can claim a larger share of the atlas. When disabled the rects reproduce the legacy 2x2 quadrant layout exactly.
- **Debug views**: `debug_gbuffer` 14 shows the selected cascade index as false colour; 15 shows per-cascade coverage.

### 2.4 Deferred G-Buffer MRT Layout (`engine.gfx.g_buffer.GBuffer`)
- **RT0 (RGBA16F)**: Albedo RGB + Roughness A (HDR emissive capable).
- **RT1 (RGBA16F)**: Octahedral Normal RG + Metallic B + Ambient Occlusion A.
- **RT2 (RG16F)**: Screen-space velocity / motion vectors.
- **RT3 (RGBA16F)**: Mesh UV (RG) + Texture Layer Index (B) + Displacement Mode (A).
- **RT4 (R32F)**: Stable entity surface ID for temporal and GI self-hit rejection.
- **Depth (DEPTH32F)**: High-precision Reversed-Z depth buffer.

### 2.5 Material Atlas & Parallax Occlusion Mapping (POM)
- **2D Texture Array Atlas (`engine.gfx.texture_atlas.TextureArrayAtlas`)**: Bundles up to 32 4K PBR material sets (Albedo, Normal/Roughness/Metallic, Displacement/Height) into a single texture array.
- **Steep Parallax Occlusion Mapping (POM)**: Raymarches displacement layers in tangent space with linear search + secant root-finding and POM self-shadowing.

### 2.6 Global Illumination & Lighting Passes
- **Ground Truth Ambient Occlusion (GTAO)**: High-performance horizon-based occlusion with spatial bilateral blur filter. Renders at full resolution by default (`RenderConfig.ao_resolution_scale = 1.0`) so its screen-space radius matches the actual viewport; a downscaled target must still be told the full render size or the radius clamp silently shrinks the effect.
- **Screen-Space Global Illumination (SSGI)**: Real-time indirect diffuse bounce from on-screen surfaces.
- **Radiance Cascades (SSRC + FFPC)**: Half-resolution cosine-weighted screen-space ray marching gathers neighboring surfaces' emissive, sun-lit, and ambient radiance for colored diffuse bounce. G-buffer entity IDs reject same-object hits on the same local surface to reduce self-bleed while preserving bounce from other faces and objects. Normal/depth-aware filtering and motion-reprojected temporal accumulation reduce noise and preserve edges; temporal confidence is reduced for fast screen-space motion and history depth is checked in the previous camera view to limit ghosting. History is invalidated when RC is disabled. Screen-space visibility remains limited to on-screen geometry; mutually exclusive with the classic SSGI path.
- **GI Mode Selection (`GIMode`)**: Runtime-switchable global illumination pipeline via `RenderConfig.gi_mode` — supports `OFF`, `IBL`, `SSGI`, and `RADIANCE_CASCADES` (SSRC + FFPC).
- **Clustered Forward+ Local Lights (`engine.gfx.passes.clustered_lights.ClusteredLightingPass`)**: Computes 3D screen frustum grid clusters to evaluate hundreds of dynamic point and spot lights simultaneously.
- **Screen-Space Reflections (SSR)**: Full-resolution high/ultra/cinematic view-space reflection tracing with tight depth intersection and subpixel refinement. Rough surfaces use depth/normal-aware reflection reconstruction, and missing screen-space hits fade into prefiltered HDR IBL. SSR remains limited to camera-visible geometry. `RenderConfig.ssr_mode` selects `HIZ` (default, hierarchical-Z accelerated march that strides empty space in coarse mips and refines on contact) or `LINEAR` (legacy fixed-step march). Both modes share the same hit validation, so they agree on which hits are accepted.
- **Image-Based Lighting (IBL)**: Split-sum GGX specular lighting with a runtime sun-driven linear HDR sky, solid-angle-aware roughness prefiltering across the full mip chain, cosine-convolved diffuse irradiance, and a BRDF integration LUT.
- **Atmosphere & Sky (`engine.gfx.atmosphere.AtmosphereSystem`)**: Configurable Hillaire Rayleigh/Mie/ozone scattering, planetary dimensions and scale heights, day/night lighting, sun and moon discs, procedural clouds, and a separate star-and-nebula space mode. Built-in presets cover Earth day/sunset/night, two alien atmospheres, and outer space; `AtmosphereConfig` can be customized directly per project.
- **Solar Kinematics**: `compute_sun_vector` derives the sun from a direct East-North-Up vector, so the arc is continuous through solar noon (an azimuth-only formulation degenerates there and snaps by 90 degrees). `AtmosphereConfig.declination_deg` sets the seasonal tilt; `latitude_deg` and `azimuth_deg` set the observer position and heading.
- **Global Light System (`engine.modules.atmosphere.global_light.GlobalLightSystem`)**: Single source of truth for `sun_dir`/`sun_lux`/`moon_dir`/`moon_lux`/`moon_phase`/`ambient` plus `primary_dir`/`primary_lux`, which pick whichever body is above the horizon. Shadows, fog, particles, and clouds all read this one surface. The moon orbits on its own hour angle (`moon_orbit_offset_hours`) and its own orbital plane (`moon_declination_deg`, `moon_azimuth_offset_deg`), so it does not simply retrace the sun's arc; its intensity is modulated by the sun-moon separation. All three lunar parameters default to values that preserve the historical antipodal moon, so projects opt in to an independent lunar path.
- **Sky Tuning**: `AtmosphereConfig.sky_exposure` and `sky_saturation` scale the sky radiance; the day/night blend is driven by sun transmittance so twilight passes through blue, orange, and deep blue instead of snapping.
- **Volumetric Fog**: Froxel raymarch through anisotropic participating media. Shared phase functions, noise, and CSM sampling live in `shaders/common/volumetrics.glsl` so fog, particles, and clouds use one canonical implementation.
- **Shared Blur Library**: `shaders/common/blur.glsl` provides Gaussian, Kawase, tent, and depth/normal-aware bilateral kernels. `engine.gfx.passes.blur_pass.BlurPass` owns the ping-pong pair so low-resolution features (AO, SSR, SSGI) share one implementation.

### 2.7 Entity Shapes & Water Surfaces
- `EntityManager.create_entity` supports the built-in `cube`, `sphere`, `plane`, `capsule`, `cylinder`, `cone`, and `water` shapes. `dimensions` sets local mesh size in meters; `shape`, `position`, `rotation`, `scale`, and material values can also be edited later through `TransformProxy`.
- Water is an ECS surface rather than an implicit graphics-preset plane. Create and size each surface explicitly:
  ```python
  water_id = app.ecs.create_entity(
      shape="water",
      position=(0.0, 0.5, 0.0),
      dimensions=(80.0, 0.0, 40.0),
      is_static=True,
  )
  app.ecs.get_transform(water_id).position = (4.0, 0.75, -2.0)
  ```
- Water has no global enable toggle: each water entity in the scene is rendered. Graphics presets configure shared optical quality only; they do not create or remove surfaces.
- Wave, refraction, foam, clarity, roughness, and water colors are authored per surface with `WaterProperties` from `engine.core`. Pass `water=WaterProperties(...)` to `create_entity(shape="water", ...)`, or edit the result through `app.ecs.get_transform(water_id).water`.

### 2.8 Anti-Aliasing & Post-Processing
- **Temporal Anti-Aliasing (TAA)**: Subpixel Halton jittering with 3x3 depth-dilated and magnitude-dilated velocity fetching, octahedral-normal history rejection, tonemapped YCoCg variance clipping, Catmull-Rom history reprojection, and contrast-adaptive sharpening. Tunable via `taa_feedback`, `taa_max_feedback`, `taa_min_feedback`, `taa_sharpness`, `taa_variance_gamma`.
- **Subpixel Morphological Anti-Aliasing (SMAA 1X / 2X / 4X)** & **Fast Approximate Anti-Aliasing (FXAA)**. SMAA detects both orthogonal and diagonal edges and resolves history with the same jitter-aware reprojection TAA uses.
- **Tone-Mapping**: ACES filmic curve, AgX, Reinhard, Uncharted 2, and neutral exposure operators.
- **Camera Optics**: Physical bokeh depth of field, bloom, chromatic aberration, and lens distortion.
- **Motion Blur**: Velocity-buffer driven (reads G-buffer RT2), 2-pass tile-max reconstruction (McGuire et al.) with depth-aware weighting and velocity dilation. Runs after TAA so the temporal filter accumulates a sharp image before the blur is applied once.
- **Lens Flare**: `RenderConfig.lens_flare_mode` selects `ANAMORPHIC` (horizontal streak, optical ghosts, halo ring) or `SOLAR` (sun-anchored disc bloom, aperture starburst, streak, ghosts, halo, with a depth-buffer occlusion test).

---

## 3. Physics & Kinematic Character Motor (`engine.physics`)

Powered by Rapier (via `rapier2d` / `rapier-rs` C bindings):
- **Rigid Body Dynamics**: Full support for Dynamic, Kinematic (position- and velocity-based), and Fixed collision bodies.
- **Collider Primitives**: Cuboids, spheres, capsules, convex hulls, and arbitrary triangle meshes.
- **Kinematic Character Controller (KCC)**:
  - Non-penetrating ground clamping and stair stepping.
  - Slope sliding with configurable maximum climb angle.
  - Air acceleration, coyote time, jump buffering, and variable jump cut.
  - Inertia and momentum preservation when launching off moving platforms.
- **Spatial Queries**: Continuous Raycasting (`cast_ray`), Shape Sweeping, and Contact Pair reporting.
- **Pick-and-Carry / Throw Grabber**: Physics-based object manipulation with distance clamping, velocity damping, and impulse release.

---

## 4. 3D Spatialized Audio Engine (`engine.audio.AudioEngine`)

- Hardware-accelerated OpenAL / PyGame audio mixer backing.
- **Spatial 3D Sound Cues (`SoundCue`)**:
  - Distance attenuation (linear, logarithmic, inverse clamp).
  - Stereo panning computed relative to camera orientation.
  - Dynamic pitch variance and volume randomization.
  - Looping audio emitters for ambient environmental loops and vehicle engines.

---

## 5. Virtual File System (VFS) & Asset Pipeline (`engine.assets`)

- **Contiguous `.pak` Container (`PakWriter` / `PakArchive`)**: Single-file archive with 64-bit FNV-1a hash table for O(1) asset lookups.
- **Zero-Copy Memory-Mapped Streaming (`mmap`)**: Enables immediate zero-copy data uploading straight to GPU textures and vertex buffers.
- **Asset Cooker CLI (`engine.assets.cooker`)**: Offline tool for compressing and baking textures (BC7/DXT5), packing PBR materials, and optimizing 3D meshes into cache-aligned `.pm_mesh` files.

---

## 6. Modular Project Architecture (`engine.app`)

PyMordial enforces a clean decoupling between the core engine and games:
- **`ProjectConfig`**: Strongly-typed project specification (title, resolution, physics frequency, quality preset, headless flag, engine-module selection).
- **`ProjectApp`**: The lightweight application host managing window lifecycle, subsystem initialization, and main game loop.
- **`ProjectModule`**: Lifecycle plugin protocol (`on_attach`, `on_fixed_update`, `on_update`, `on_ui`, `on_event`, `on_detach`).
- **`BaseWorldBuilder` / `DefaultWorldBuilder`**: Extensible world generation contracts allowing projects to define procedural terrains, city grids, or use default tiled POM planes.

### 6.1 Module metadata (`engine.app.module_manifest`)

Modules declare identity and dependencies with a typed `ModuleManifest` exposed as a
`MODULE_MANIFEST` class attribute (or a `get_module_manifest()` callable):

```python
from engine.app.module_manifest import ModuleRole, build_manifest


class MyModule(ProjectModule):
    MODULE_MANIFEST = build_manifest(
        "my_game.combat",
        version="1.0.0",
        role=ModuleRole.GAMEPLAY,
        entry="projects.my_game.modules.combat:CombatModule",
        dependencies=["my_game.core"],
        capabilities=("gameplay.combat",),
    )
```

`validate_manifest()` is side-effect free — it never imports or executes module code — and
checks id format, semver, API major compatibility, role, entry shape, and dependency
declarations. Modules without metadata remain supported as legacy gameplay modules.

### 6.2 Module host (`engine.app.module_registry`)

`ModuleRegistry` resolves a deterministic dependency order (Kahn topological sort, id
tie-break) and reports missing dependencies, duplicate ids, and cycles with the cycle path.
`ModuleHost` attaches in dependency order, detaches in reverse, and **rolls back** every
already-attached module plus all host-owned event subscriptions if any module fails:

```python
registry = ModuleRegistry()
registry.register_instance(SpinnerModule())
app.load_modules(registry)  # raises ModuleAttachError after a clean rollback
```

### 6.3 Render extension graph (`engine.gfx.render_graph`, `render_feature`, `render_services`)

`RenderPass` declares `inputs`, `outputs`, `after`, `before`, and `capabilities`.
`RenderGraph.validate()` rejects unknown ordering references, duplicate outputs, missing
inputs, and cycles before any GPU work runs. `RenderPipeline` describes its own 19 built-in
stages as a validated graph and hosts module features through `register_feature()`.
`RenderServices` is the only renderer surface a feature may touch; it exposes capability
queries and texture-unit reservations, with the core deferred-resolve units marked reserved.

### 6.4 Engine modules (`engine.modules`)

Official reusable capabilities. `pymordial.atmosphere` is the first extraction and is
enabled by default. Projects configure or disable it declaratively:

```python
ProjectConfig(
    atmosphere_enabled=False,  # flat-sky fallback
    atmosphere_settings={"preset": "EARTH_SUNSET"},  # or configure
)
```

The pipeline consumes the `EnvironmentProvider` contract (`engine.gfx.environment`) and
never imports a specific atmosphere implementation at module scope.

---

## 7. Dual-Target Release Packaging (`engine.tools.packager`)

A single build command bundles complete projects into standalone release distributions:
- **`--target pyinstaller`**: Beginner-friendly single-folder distribution requiring zero C++ compiler toolchains.
- **`--target nuitka`**: Advanced native Ahead-of-Time (AOT) C++ compilation for peak CPU execution speed and binary obfuscation.
- **Release Optimizations**: Strips debug overlays (`debug_overlay`), tree-shakes unused modules, and archives assets into uncompressed/compressed `game.pak` containers.
