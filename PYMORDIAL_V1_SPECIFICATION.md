# PyMordial Game Engine V1: Architecture, Systems & Implementation Specification

> **Status**: Approved Architectural Master Blueprint  
> **Target Runtime**: Python 3.11+ / OpenGL 4.5 Core / C++20 / Rust (Rapier3D)  
> **Intended Use**: Foundational specification document for scaffolding the new PyMordial V2 repository.

---

## 1. Executive Summary & Vision

**PyMordial V1** is a data-oriented, hybrid-tier 3D game engine engineered to bridge two historically disjoint game development paradigms:

1. **For Indie Prototypers and Gameplay Designers**: A modern, high-level Python workflow featuring an intuitive ImGui docking editor, declarative prefab composition, auto-reflecting entity inspectors, and expressive component scripting (`EntityBehavior`).
2. **For Advanced and Engine Systems Engineers**: A zero-overhead, data-oriented execution core. All transforms, bounding hierarchies, material parameters, and physics states reside in contiguous, C-aligned memory tables (NumPy float32 arrays) that map directly into OpenGL Uniform Buffer Objects (UBOs) and Shader Storage Buffer Objects (SSBOs). Advanced developers can bypass the Python runtime entirely by dropping in native C++20 / SIMD plugins via `pybind11`/`nanobind` that mutate raw memory buffers with zero copy overhead and inject custom OpenGL compute passes.

### Primary Engineering Objectives
* **Deterministic Single-Process Tick**: Elimination of multi-process IPC, `mmap` serialization bottlenecks, and OS synchronization drift.
* **Tightly Coupled ModernGL 4.5 Pipeline**: Consolidation of disparate render passes into a unified Frame Context (UBO 0), Multi-Draw Indirect (MDI) geometry submission, integrated Cascaded Shadow Maps (CSM), Screen-Space Contact Shadows (SSCS), and Clustered PBR lighting.
* **Zero-Allocation Runtime Loops**: Strict prohibition of temporary Python dictionary, list, or tuple allocations within per-frame tick and render loops.
* **Strict Offline Asset Cooking**: Elimination of runtime `.gltf`/`.obj`/`.png` parsing in favor of memory-mapped, 32-byte cache-aligned binary formats (`.pm_mesh`, `.pm_tex`, `.pm_mat`, `.pak`).

---

## 2. Core Architectural Pillars

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   PyMordial V1 Engine Host (Python)                      │
├──────────────────────────────────────────────────────────────────────────┤
│ - Pygame-ce Window & Input Polling                                       │
│ - Deterministic Fixed-Accumulator Tick Loop (60 Hz)                      │
│ - Dense-to-Sparse Entity ID Allocator (100,000 entities)                 │
└─────────────────────┬───────────────────────────────┬────────────────────┘
                      │                               │
                      ▼                               ▼
    ┌───────────────────────────────────┐   ┌──────────────────────────────┐
    │     Contiguous Memory Tables      │   │  Rapier3D Physics (Rust)     │
    │  - WorldTransforms (Nx16 float32) │   │  - In-Thread Synchronous Step│
    │  - RigidBodyState  (Nx7  float32) │<─>│  - Releases Python GIL       │
    │  - MaterialData    (Nx8  float32) │   │  - Kinematic Capsule Sweeps  │
    └─────────────────┬─────────────────┘   └──────────────────────────────┘
                      │
     ┌────────────────┴────────────────────────┐
     │ Zero-Copy Pointer (py::buffer_info)      │ Direct GPU SSBO Upload
     ▼                                         ▼
┌───────────────────────────┐         ┌────────────────────────────────────┐
│ C++20 Native Plugins      │         │ OpenGL 4.5 MDI Render Pipeline     │
│ - SIMD Crowd / Boids      │         │ - UBO 0: FrameUniforms             │
│ - JPS+ 3D Pathfinding     │         │ - SSBO 1: WorldTransforms          │
│ - Custom Compute Passes   │         │ - Multi-Draw Indirect Mega-VBO/IBO │
└───────────────────────────┘         └─────────────────┬──────────────────┘
                                                        │
                                                        ▼
                                      ┌────────────────────────────────────┐
                                      │ Offscreen HDR Scene FBO            │
                                      │  - Color0: HDR16F (Resolved Scene) │
                                      │  - Color1: Normal + Roughness      │
                                      │  - Depth:  Reversed-Z 32F          │
                                      └─────────────────┬──────────────────┘
                                                        │
                                                        ▼
                                      ┌────────────────────────────────────┐
                                      │ Editor Viewport ImGui Texture      │
                                      │ (Zero-copy GPU texture handle)     │
                                      └────────────────────────────────────┘
```

### 2.1. Single-Process Deterministic Loop with GIL Release
The engine orchestrates an explicit tick sequence. Rapier3D (compiled in native Rust) releases the Python GIL during simulation stepping, allowing Rust to execute multithreaded Rayon work while eliminating multi-process latency:

```python
FIXED_DT = 1.0 / 60.0
MAX_ACCUMULATOR = 0.20

while running:
    frame_time = min(clock.tick(), MAX_ACCUMULATOR)
    accumulator += frame_time

    # 1. Input Processing Phase
    input_manager.poll_events()

    # 2. Fixed Timestep Phase (Physics & Native Simulation)
    while accumulator >= FIXED_DT:
        physics_manager.cache_previous_state()   # Snapshot prev_state for interpolation
        physics_manager.step_simulation(FIXED_DT)# Releases GIL to Rust Rayon pool
        physics_manager.sync_to_ecs()           # Copy Rapier transforms to Curr_State
        script_manager.tick_fixed_update(FIXED_DT)
        plugin_manager.tick_native_fixed(FIXED_DT)
        accumulator -= FIXED_DT

    # 3. Variable Timestep Phase (Sub-Frame Interpolation & Gameplay)
    alpha = accumulator / FIXED_DT
    physics_manager.interpolate_render_transforms(alpha) # NLERP prev/curr -> WorldTransforms
    script_manager.tick_update(frame_time)
    plugin_manager.tick_native_update(frame_time)

    # 4. Render Phase (GPU Submission)
    render_pipeline.render_frame(camera)

    # 5. Editor / UI Phase
    editor_ui.render(viewport_texture=render_pipeline.output_texture_id)
    window.swap_buffers()
```

### 2.2. Contiguous Flat Memory Hierarchy
All scene data is stored in pre-allocated, flat C-contiguous NumPy arrays for a configurable entity ceiling ($N = 100{,}000$):

* **`WorldTransforms` Table**:
  * Shape: `(MAX_ENTITIES, 16)`, Type: `np.float32`, Memory: C-Contiguous Row-Major.
  * Mirrors directly into `layout(std430, binding = 1) readonly buffer TransformBuffer { mat4 u_WorldTransforms[]; };`.
* **`RigidBodyState` Double Buffer**:
  * Shape: `(2, MAX_ENTITIES, 7)`, Type: `np.float32`.
  * Layout: $[P_x, P_y, P_z, Q_x, Q_y, Q_z, Q_w]$. Index 0 = Previous Tick, Index 1 = Current Tick.
  * Used for vectorized **NLERP** interpolation into `WorldTransforms`.
* **`MaterialData` Table**:
  * Shape: `(MAX_ENTITIES, 8)`, Type: `np.float32`.
  * Layout: $[R, G, B, \text{Roughness}, \text{Metallic}, \text{AO}, \text{AlbedoTexID}, \text{NormalTexID}]$.
  * Mirrors directly into `layout(std430, binding = 2) readonly buffer MaterialBuffer`.

### 2.3. Dense-to-Sparse Entity Indexing & Free-List Pool
To guarantee that active entities remain densely packed for GPU sub-buffer streaming (`glBufferSubData`) while maintaining stable external 32-bit Entity IDs:
* **Sparse Array**: Maps stable `EntityID` $\to$ `DenseIndex`.
* **Dense Array**: Maps `DenseIndex` $\to$ stable `EntityID`.
* **Free-List**: Array of recycled `EntityID`s for $O(1)$ allocation and deallocation.
* When entity $E$ is destroyed, the last dense entity is swapped into $E$'s slot, updating the sparse table ($O(1)$ removal, zero memory holes in GPU buffers).

---

## 3. Tightly Coupled Graphics Pipeline (ModernGL / OpenGL 4.5 Core)

Rather than maintaining disconnected rendering modules, the pipeline is tightly integrated around a **Unified Frame Context (UBO 0)**, **Multi-Draw Indirect (MDI)**, and a **Consolidated Deferred Lighting & Shadow Resolve**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         PASS 1: SHADOW ATLAS PASS                        │
│ - Cascaded Shadow Maps (CSM): 4 Cascades packed into 4096x4096 Depth FBO │
│ - Tight bounding boxes fitted to camera frustum split planes             │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        PASS 2: G-BUFFER GEOMETRY (MDI)                   │
│ - Bound: Single Shared Mega-VBO/IBO                                      │
│ - Submit: glMultiDrawElementsIndirect(GL_TRIANGLES, ...) (1-2 draw calls)│
│ - MRT Output:                                                            │
│   • RT0 (RGBA8):     Albedo (RGB) + Roughness (A)                        │
│   • RT1 (RGB10_A2):  Octahedral Encoded Normal (RG) + Metallic (B)       │
│   • RT2 (RG16F):     Velocity Vectors                                    │
│   • Depth (DEPTH32F):Reversed-Z Floating Point Depth                     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              PASS 3: TIGHTLY COUPLED LIGHTING & SHADOW RESOLVE           │
│ - Reads: Depth32F, RT0, RT1 once                                         │
│ - Evaluates:                                                             │
│   1. Clustered Point & Spot Lights (Frustum tiles: 16x9x24 bins)         │
│   2. Directional Sunlight with Cook-Torrance PBR BRDF                    │
│   3. 4-Cascade CSM Shadows with 16-tap Poisson PCF Filtering             │
│   4. Screen-Space Contact Shadows (SSCS): 12-step depth raymarch (<10m)   │
│   5. Integrated Volumetric Fog In-Scattering                             │
│ - Output: Single HDR16F Scene Color Texture                              │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  PASS 4: POST-PROCESS & EDITOR RESOLVE                   │
│ - Downsample / Upsample Bloom Pyramid (13-tap tent filter)               │
│ - ACES / AgX Tonemapping & Color Grading LUT                             │
│ - Viewport Blit: Color texture passed directly to ImGui Viewport Dock    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.1. Unified Frame Context UBO (`layout(std140, binding = 0)`)
Every engine shader binds to UBO binding point 0. It is updated exactly once per frame from Python via `ubo.write(...)`:

```glsl
layout (std140, binding = 0) uniform FrameData {
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    mat4 u_InvProjection;
    mat4 u_InvView;
    
    vec4 u_CameraPos_Time;        // xyz = camera world pos, w = total elapsed time
    vec4 u_ScreenSize_Jitter;     // xy = width/height, zw = subpixel jitter (TAA)
    
    // Directional Sunlight & Sky
    vec4 u_SunDirection_Intensity;// xyz = normalized sun dir, w = sun lux
    vec4 u_SunColor_Ambient;       // rgb = sun light color, w = ambient factor
    
    // Cascaded Shadow Maps (CSM)
    mat4 u_LightViewProjection[4];
    vec4 u_CascadeSplits;         // Far split distances for cascades 0..3
    
    // Volumetric Atmospheric Fog
    vec4 u_FogColor_Density;       // rgb = fog color, w = fog density
    vec4 u_FogParams;              // x = height falloff, y = max distance, zw = unused
};
```

### 3.2. Multi-Draw Indirect (MDI) Architecture
* **Single Mega-VBO/IBO**: All meshes loaded by the asset system are packed into a single GPU buffer allocation.
* **Indirect Buffer**: Contains an array of `DrawElementsIndirectCommand` structs:
  ```cpp
  struct DrawElementsIndirectCommand {
      uint32_t count;         // Index count
      uint32_t instanceCount; // 1 for single entity, M for instanced foliage
      uint32_t firstIndex;    // Byte offset into IBO
      int32_t  baseVertex;     // Base vertex offset into VBO
      uint32_t baseInstance;   // Index into WorldTransforms & Material SSBOs
  };
  ```
* **Execution**: Executed via `glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, drawCount, 0)` through ModernGL's context.

### 3.3. Screen-Space Contact Shadows (SSCS)
Integrated directly into the deferred lighting resolve shader:
* For any pixel receiving directional sunlight where distance $< 10.0\text{ m}$:
* Raymarches in view space toward the light direction for up to 12 steps.
* Eliminates shadow acne and grounding detachment under character feet, vehicle tires, and small debris.

---

## 4. Physics & Spatial Kinematics (Rapier3D)

* **Direct Python-Rust Binding**: Built using official `rapier3d-py` or compiled submodule using `maturin`.
* **In-Thread Execution**: Executes directly on the main thread during fixed update steps (`FIXED_DT = 1.0 / 60.0`). The Python GIL is automatically released during `world.step()`.
* **Kinematic Capsule Controller**:
  * Capsule shape: Radius $0.4\text{ m}$, Total Height $1.8\text{ m}$.
  * Built-in grounding check (downward raycast + sphere cast $0.05\text{ m}$ below capsule base).
  * Slope slide angle cutoff ($45^\circ$).
  * Dynamic step snapping for curbs and stairs ($0.3\text{ m}$ maximum step height).
* **Double-Buffered State Extraction**:
  * Direct batch extraction of position (`float32[3]`) and quaternion (`float32[4]`) from active Rapier rigid bodies into the `RigidBodyState[1]` table.

---

## 5. Offline Asset Baking & Virtual File System (VFS)

Raw assets (`.gltf`, `.fbx`, `.png`, `.wav`) are **never parsed at runtime**. The offline CLI cooker (`python -m engine.assets.cooker`) compiles assets into fast, binary, memory-mappable structures.

### 5.1. 32-Byte Cache-Aligned Binary Mesh Format (`.pm_mesh`)

```
┌────────────────────────────────────────────────────────────────────────┐
│ Header: Magic (0x504D4D53) | VertCount (u32) | IndexCount (u32)        │
│ Bounding Sphere Center (3xf32) | Bounding Sphere Radius (f32)          │
│ Bounding Box Min (3xf32) | Bounding Box Max (3xf32)                    │
├────────────────────────────────────────────────────────────────────────┤
│ Interleaved Vertex Array (32 bytes per vertex):                        │
│ • Position: 3 x float32 (12 bytes)                                     │
│ • Normal:   4 x int16_snorm (8 bytes, octahedral tangent sign in W)    │
│ • UV:       2 x float16 (4 bytes)                                      │
│ • Tangent:  4 x int16_snorm (8 bytes)                                  │
├────────────────────────────────────────────────────────────────────────┤
│ Index Array: uint32[IndexCount] (Contiguous triangle indices)          │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2. Compressed GPU Texture Format (`.pm_tex`)
* **Albedo / Roughness / Metallic / AO**: Pre-compressed to **BC7** format via KTX / ISPC texture compressor.
* **Normal Maps**: Pre-compressed to **BC5** format (dual-channel tangent space $X, Y$; $Z$ reconstructed in GLSL via $Z = \sqrt{1 - X^2 - Y^2}$).
* Direct upload to GPU via `glCompressedTexImage2D` with pre-generated mipmap chains.

### 5.3. Virtual File System (`.pak`)
* Bundles all compiled `.pm_mesh`, `.pm_tex`, and `.pm_mat` files into a single contiguous archive with an uncompressed directory header and optional Zstandard compression for distribution.

---

## 6. High-Performance C++ Native Plugin ABI (`pymordial.h`)

The engine provides an official, header-only C++20 SDK (`pymordial.h`) based on `pybind11`/`nanobind`. Developers write performance-critical gameplay code (e.g., flocking boids, custom solvers, procedural mesh generation) that executes at bare-metal speed:

```cpp
// pymordial.h — Native Plugin Interface
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pymordial {

struct EngineMemoryView {
    float* transforms;     // Nx16 float32 (WorldTransforms)
    float* prev_physics;   // Nx7  float32 (Position + Quaternion)
    float* curr_physics;   // Nx7  float32 (Position + Quaternion)
    float* materials;      // Nx8  float32 (PBR params)
    uint32_t entity_count;
    float delta_time;
};

class PluginBase {
public:
    virtual ~PluginBase() = default;
    virtual void OnInit() {}
    virtual void OnFixedUpdate(const EngineMemoryView& mem) {}
    virtual void OnUpdate(const EngineMemoryView& mem) {}
    virtual void OnShutdown() {}
};

} // namespace pymordial
```

---

## 7. Standalone Editor Architecture (`editor/`)

The PyMordial Editor is completely isolated from the runtime core. It consumes the engine as a headless client and renders inside Dear ImGui (`imgui-bundle` with docking support).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PyMordial V1 Editor                                                [ _ ][ □ ][ X ]     │
├──────────────┬──────────────────────────────────────────┬──────────────────────────────┤
│ 📂 Outliner  │ 🎮 Viewport Dock (Offscreen FBO)         │ ⚙️ Inspector                 │
│ ├─ SceneRoot │   ┌──────────────────────────────────┐   │ Entity: Player               │
│ │  ├─ Sun    │   │                                  │   │ UUID: ent_001a4f             │
│ │  ├─ Camera │   │ [Translate][Rotate][Scale] (Gizmo)│  ├──────────────────────────────┤
│ │  ├─ Player │   │                                  │   │ Transform                    │
│ │  └─ Level  │   │       3D Scene Rendered to       │   │  Pos: [ 0.00,  1.00,  0.00 ] │
│ │            │   │       ImGui Texture Widget       │   │  Rot: [ 0.00,  0.00,  0.00 ] │
│              │   │                                  │   │  Scl: [ 1.00,  1.00,  1.00 ] │
│              │   │                                  │   ├──────────────────────────────┤
│              │   │ View: Lit | 60 FPS | 1.2 ms      │   │ RigidBody (Capsule)          │
│              │   └──────────────────────────────────┘   │  Mass: 70.0 kg               │
│              │ [▶ Play (PIE)] [⏸ Pause] [⏹ Stop]        │  Kinematic: True             │
├──────────────┴──────────────────────────────────────────┴──────────────────────────────┤
│ 📁 Content Browser: assets/models/ | assets/materials/ | assets/prefabs/               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Editor Features:
1. **Zero-Copy ImGui Viewport**: The engine renders into an offscreen HDR FBO; the editor displays the color attachment texture directly using `imgui.image(texture_id, (width, height))`.
2. **6-DOF Viewport Camera**: Smooth WASD fly-cam (holding Right Mouse), Alt+Left Mouse orbit, Alt+Middle Mouse pan, and 'F' key focus frame.
3. **3D Transform Gizmos**: Interactive Translate (W), Rotate (E), and Scale (R) handles with grid and surface snapping.
4. **Instant Play-In-Editor (PIE)**: Because memory is flat and contiguous, toggling Play mode creates a fast `memcpy` snapshot of the NumPy buffers. Pressing `Esc` restores the exact pre-simulation state instantaneously without reloading assets or recompiling shaders.

---

## 8. Complete Directory Structure & Repository Layout

```text
PyMordial-Engine/
├── CMakeLists.txt                # Root CMake for C++ plugins and SDK
├── pyproject.toml                # Project packaging and dependencies
├── README.md                     # Repository documentation & quickstart
│
├── engine/                       # Core engine package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ecs.py                # Flat NumPy memory tables (Transforms, Physics, Materials)
│   │   ├── entity_pool.py        # Sparse-Dense ID mapping & O(1) Free-List allocator
│   │   ├── loop.py               # Deterministic fixed-accumulator main tick loop
│   │   ├── input.py              # Semantic action/axis input manager
│   │   └── math_utils.py         # Vectorized quaternion & matrix routines
│   │
│   ├── gfx/
│   │   ├── __init__.py
│   │   ├── context.py            # ModernGL context initialization & display surface
│   │   ├── frame_context.py      # UBO 0 (FrameData) allocation & synchronization
│   │   ├── mega_buffer.py        # Shared Mega-VBO/IBO allocator
│   │   ├── mdi.py                # Multi-Draw Indirect command batcher & submitter
│   │   ├── g_buffer.py           # MRT Framebuffer allocation (HDR16F, Normal, Depth32F)
│   │   ├── shadow_csm.py         # Cascaded Shadow Maps atlas manager (4 cascades)
│   │   ├── clustered_light.py    # 3D Frustum cluster light binning (Point/Spot)
│   │   ├── post_process.py       # Bloom pyramid, tonemapping (ACES/AgX), depth of field
│   │   └── pipeline.py           # Unified master render graph & pass coordinator
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── rapier_world.py       # Synchronous Rapier3D integration (in-thread, GIL released)
│   │   ├── character_motor.py    # Kinematic capsule controller (grounding, step snapping)
│   │   └── queries.py            # Raycast, shape sweep, and overlap queries
│   │
│   ├── assets/
│   │   ├── __init__.py
│   │   ├── cooker.py             # CLI Asset Baker (.gltf/.png -> .pm_mesh/.pm_tex)
│   │   ├── mesh_format.py        # 32-byte binary .pm_mesh parser/writer
│   │   ├── texture_format.py     # BC7/BC5 compressed .pm_tex parser/writer
│   │   ├── vfs.py                # Virtual File System (.pak archive reader)
│   │   └── resource_cache.py     # Fast memory-mapped binary loader
│   │
│   └── plugins/
│       ├── __init__.py
│       ├── plugin_host.py        # Dynamic library loader (.so / .dll)
│       └── include/
│           └── pymordial.h       # C++20 Header-Only Native Plugin SDK
│
├── editor/                       # Standalone Level & World Editor
│   ├── __init__.py
│   ├── main.py                   # Editor entry point (python -m editor)
│   ├── app.py                    # Master Editor lifecycle & state machine
│   ├── camera.py                 # 6-DOF Fly/Orbit/Pan editor camera
│   ├── gizmos.py                 # 3D Translate/Rotate/Scale viewport gizmos
│   ├── selection.py              # Entity mouse ray-picking & multi-select
│   ├── history.py                # Undo / Redo Command Pattern (Ctrl+Z / Ctrl+Y)
│   └── ui/
│       ├── main_dockspace.py     # Main docking layout & dark styling
│       ├── viewport_panel.py     # Offscreen ModernGL FBO Viewport widget
│       ├── outliner_panel.py     # Hierarchy Tree & Scene Graph panel
│       ├── inspector_panel.py    # Component Property Inspector
│       └── content_browser.py    # Thumbnail asset shelf & drag-and-drop spawner
│
├── shaders/                      # Consolidated, high-performance GLSL 4.5 shaders
│   ├── common/
│   │   ├── frame_data.glsl       # UBO 0 shared struct definition
│   │   └── pbr_brdf.glsl         # Cook-Torrance GGX PBR evaluation functions
│   ├── gbuffer.vert              # MDI geometry vertex shader (reads SSBO 1 & 2)
│   ├── gbuffer.frag              # MRT output (Albedo, Octahedral Normal, Velocity)
│   ├── csm_depth.vert            # Shadow pass vertex shader
│   ├── csm_depth.frag            # Shadow pass fragment depth output
│   ├── deferred_resolve.frag     # Consolidated Lighting + CSM + SSCS + Fog
│   └── post_process.frag         # Downsample, Bloom, Tonemap, Color Grading
│
├── runtime/                      # Standalone Game Player
│   ├── __init__.py
│   └── boot.py                   # Minimal headless or windowed game runtime launcher
│
├── sdk/                          # C++ Developer Starter Kit
│   └── plugin_template/
│       ├── CMakeLists.txt
│       └── src/
│           └── plugin_main.cpp   # Starter template for C++ plugins
│
└── tests/                        # Comprehensive test suite
    ├── test_ecs_memory.py        # Flat buffer contiguity & dense-sparse map tests
    ├── test_loop_determinism.py  # Fixed accumulator timing verification
    ├── test_asset_cooker.py      # Mesh & texture binary packing verification
    └── test_physics_sync.py      # Rapier transform synchronization tests
```

---

## 9. Phased Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYMORDIAL V1 IMPLEMENTATION ROADMAP                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Foundation, Contiguous Memory & Deterministic Loop                │
│  • Initialize Pygame-ce window with ModernGL 4.5 Core Profile context.      │
│  • Build `engine/core/ecs.py` with flat `WorldTransforms` (Nx16 float32)     │
│    and double-buffered `RigidBodyState` (Nx7 float32) memory tables.        │
│  • Implement Sparse-Dense entity pool with O(1) Free-List recycling.        │
│  • Build deterministic fixed-accumulator tick loop with alpha interpolation.│
│                                                                             │
│  Phase 2: Tightly Coupled Rendering Core (MDI & Unified UBO)                │
│  • Implement `FrameData` UBO 0 (View, Proj, InvProj, Sun, Fog, Splits).     │
│  • Build `MegaBuffer` (shared VBO/IBO) and `MultiDrawIndirect` (MDI).       │
│  • Create MRT G-Buffer (HDR16F Albedo, Octahedral Normal, Reversed-Z Depth).│
│  • Implement 4-Cascade CSM atlas and Screen-Space Contact Shadows (SSCS).   │
│  • Build Consolidated Lighting & Fog Resolve pass.                          │
│                                                                             │
│  Phase 3: Synchronous Physics & Character Controller                        │
│  • Integrate Rapier3D directly in main thread loop (GIL released in Rust).  │
│  • Double-buffered state extraction into `RigidBodyState`.                  │
│  • Build Kinematic Capsule Character Controller with grounding and slope.   │
│                                                                             │
│  Phase 4: Offline Asset Cooker & Virtual File System (VFS)                  │
│  • Build `cooker.py` CLI tool: `.gltf`/`.glb` -> 32-byte `.pm_mesh`.        │
│  • Implement BC7/BC5 texture compression pipeline -> `.pm_tex`.             │
│  • Build `.pak` archive reader with memory-mapped streaming.                │
│                                                                             │
│  Phase 5: C++ Native Plugin ABI (`pymordial.h`)                             │
│  • Build header-only `pymordial.h` using `pybind11`.                        │
│  • Create `plugin_host.py` dynamic library loader (.so / .dll).             │
│  • Test sample C++ plugin modifying `WorldTransforms` via SIMD.             │
│                                                                             │
│  Phase 6: Standalone ImGui Docking Editor & 3D Gizmos                       │
│  • Implement ImGui docking window with dark slate theme.                    │
│  • Embed offscreen ModernGL FBO Viewport widget (`imgui.image`).            │
│  • Build 6-DOF Fly/Orbit camera and 3D Transform Gizmos (W, E, R).          │
│  • Build Scene Outliner, Component Inspector, and Content Browser.          │
│  • Implement Play-In-Editor (PIE) with zero-reload memcpy state snapshots.  │
│                                                                             │
│  Phase 7: Packaging, Freezing & Standalone Release                          │
│  • Build standalone game bootloader in `runtime/boot.py`.                   │
│  • Configure PyInstaller / Nuitka standalone release compilation script.    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Development Directives & Engineering Constraints

When developing or pair-programming on PyMordial V1 modules, the following rules must be strictly enforced:

1. **Zero Python Allocations in Hot Paths**:
   Inside `tick_update`, `tick_fixed_update`, or `render_frame`, **never** instantiate temporary Python `dict`, `list`, or `tuple` objects. Pre-allocate scratch arrays and buffers during initialization.
2. **Strict C-Contiguity Enforcement**:
   All NumPy slices passing to OpenGL or C++ plugins must be explicitly verified with `assert arr.flags['C_CONTIGUOUS']` and `dtype=np.float32`.
3. **Keep Physics Synchronous**:
   Never wrap Rapier3D queries or steps in asynchronous event queues, background threads, or multi-process pipes. Stepping and raycasting must occur directly in the main thread tick.
4. **Absolute Decoupling of Editor and Runtime**:
   The engine core in `engine/` must be 100% operational without importing or loading `imgui` or the `editor/` package. The editor is strictly an external consumer that passes an offscreen FBO texture to a GUI widget.
5. **Single Source of Truth for Transformations**:
   Entity positions, rotations, and scales live exclusively in the flat memory tables. No component object may store independent transform state that requires manual synchronization.
