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
  - Position: `float32[3]` (12 bytes)
  - UV0: `float32[2]` (8 bytes)
  - Normal: `int16[4]` normalized (8 bytes)
  - Tangent: `int8[4]` normalized with bitangent sign (4 bytes)
  - Cache-aligned to 32 bytes for peak GPU memory fetch coalescing.

### 2.2 Reversed-Z Floating Point Depth
- Depth range: `1.0` (near plane) to `0.0` (far plane / infinity).
- `GL_GREATER` depth comparison eliminating Z-fighting across vast outdoor landscapes.

### 2.3 Cascaded Shadow Maps (CSM) (`engine.gfx.shadow_csm.CascadedShadowMap`)
- Up to 4 shadow cascades with logarithmic/linear split blending.
- Percentage-Closer Filtering (PCF) with variable sample kernels (16-tap Poisson disc, 5x5, 7x7).
- Shadow bounding spheres fitted tightly to camera frustum slices with texel snapping to prevent edge shimmer during camera translation.

### 2.4 Deferred G-Buffer MRT Layout (`engine.gfx.g_buffer.GBuffer`)
- **RT0 (RGBA16F)**: Albedo RGB + Roughness A.
- **RT1 (RGBA16F)**: Octahedron/World Normal RGB + Metallic A.
- **RT2 (RGBA16F)**: Motion Vectors (Screen-space 2D velocity) + Material Layer ID.
- **RT3 (R11G11B10F)**: Emissive RGB + Ambient Occlusion.
- **Depth (DEPTH32F)**: High-precision Reversed-Z depth buffer.

### 2.5 Material Atlas & Parallax Occlusion Mapping (POM)
- **2D Texture Array Atlas (`engine.gfx.texture_atlas.TextureArrayAtlas`)**: Bundles up to 32 4K PBR material sets (Albedo, Normal/Roughness/Metallic, Displacement/Height) into a single texture array.
- **Steep Parallax Occlusion Mapping (POM)**: Raymarches displacement layers in tangent space with linear search + secant root-finding and POM self-shadowing.

### 2.6 Global Illumination & Lighting Passes
- **Ground Truth Ambient Occlusion (GTAO)**: High-performance horizon-based occlusion with spatial bilateral blur filter.
- **Screen-Space Global Illumination (SSGI)**: Real-time indirect diffuse bounce from on-screen surfaces.
- **Light Propagation Volumes (LPV)**: 3D spatial grid for low-frequency multi-bounce indirect lighting.
- **Radiance Cascades (SSRC + FFPC)**: Screen-Space Radiance Cascades and Far-Field Probe Cascades for high-quality, multi-scale global illumination. Interval raymarching computes per-cascade radiance at descending angular resolutions, merged via bilateral/temporal resolve for noise-free indirect lighting. Mutually exclusive with the classic SSGI/LPV path.
- **GI Mode Selection (`GIMode`)**: Runtime-switchable global illumination pipeline via `RenderConfig.gi_mode` — supports `OFF`, `SSGI`, `LPV`, `HYBRID` (SSGI + LPV), and `RADIANCE_CASCADES` (SSRC + FFPC).
- **Clustered Forward+ Local Lights (`engine.gfx.passes.clustered_lights.ClusteredLightingPass`)**: Computes 3D screen frustum grid clusters to evaluate hundreds of dynamic point and spot lights simultaneously.
- **Screen-Space Reflections (SSR)**: Hi-Z traced glossy/rough reflections with edge fade and fallback to IBL.
- **Image-Based Lighting (IBL)**: Split-sum approximation using prefiltered environment cubemaps and BRDF integration LUT.
- **Atmosphere & Sky (`engine.gfx.atmosphere.AtmosphereSystem`)**: Physically-based Rayleigh and Mie atmospheric scattering with dynamic sun and moon positions.
- **Volumetric Fog**: Light-shaft raymarching through anisotropic participating media.

### 2.7 Anti-Aliasing & Post-Processing
- **Temporal Anti-Aliasing (TAA)**: Subpixel Halton jittering with neighborhood color clamping to prevent ghosting.
- **Subpixel Morphological Anti-Aliasing (SMAA 1X / 2X / 4X)** & **Fast Approximate Anti-Aliasing (FXAA)**.
- **Tone-Mapping**: ACES filmic curve, AgX, Reinhard, Uncharted 2, and neutral exposure operators.
- **Camera Optics**: Physical bokeh depth of field, bloom, chromatic aberration, and lens distortion.

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
- **`ProjectConfig`**: Strongly-typed project specification (title, resolution, physics frequency, quality preset, headless flag).
- **`ProjectApp`**: The lightweight application host managing window lifecycle, subsystem initialization, and main game loop.
- **`ProjectModule`**: Lifecycle plugin protocol (`on_attach`, `on_fixed_update`, `on_update`, `on_ui`, `on_event`, `on_detach`).
- **`BaseWorldBuilder` / `DefaultWorldBuilder`**: Extensible world generation contracts allowing projects to define procedural terrains, city grids, or use default tiled POM planes.

---

## 7. Dual-Target Release Packaging (`engine.tools.packager`)

A single build command bundles complete projects into standalone release distributions:
- **`--target pyinstaller`**: Beginner-friendly single-folder distribution requiring zero C++ compiler toolchains.
- **`--target nuitka`**: Advanced native Ahead-of-Time (AOT) C++ compilation for peak CPU execution speed and binary obfuscation.
- **Release Optimizations**: Strips debug overlays (`debug_overlay`), tree-shakes unused modules, and archives assets into uncompressed/compressed `game.pak` containers.
