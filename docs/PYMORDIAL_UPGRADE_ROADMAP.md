# PyMordial Engine: Master Upgrade Roadmap & Beta Transition Blueprint
*(Inspired by Guerrilla Games' Decima Engine — Tailored for Python 3.11+, ModernGL / OpenGL 4.5 Core Profile, and Rapier3D)*

---

## 1. Executive Summary & Strategic Objectives

PyMordial Engine is a high-performance 3D engine built on **Python 3.11+**, **ModernGL / OpenGL 4.5 Core Profile**, and **Rapier3D**. Its current alpha foundation features Multi-Draw Indirect (MDI) via a unified `MegaBuffer`, zero-allocation contiguous NumPy Structure-of-Arrays (SoA) ECS mirroring directly into OpenGL SSBOs, Reversed-Z floating-point depth, Radiance Cascades, Clustered Forward+ local lighting, and decoupled fixed-timestep physics.

The purpose of this Master Roadmap is to transition PyMordial Engine from an exploratory **Alpha Base** to a production-grade, highly cohesive **Beta Architecture**:
1. **Decima Engine Inspiration**: Replicate the core strengths of Guerrilla's Decima Engine (*Horizon Zero Dawn / Forbidden West*, *Death Stranding*)—vast living open worlds, procedural clutter/foliage, seamless LOD, rock-solid frame pacing, and physical atmosphere.
2. **Strict Tech Stack Discipline**:
   - **Zero-Allocation Inner Loop**: No dynamic object creation in `on_fixed_update`, `on_update`, or `on_render`; use pre-allocated scratch arrays and direct memoryviews.
   - **ModernGL / OpenGL 4.5 Core Profile**: Persistent mapped buffers, compute shaders, SSBOs, and indirect GPU dispatches.
   - **Rapier3D Physics**: Decoupled 60 Hz fixed simulation ticks with interpolation alpha ($\alpha$) for sub-frame visual rendering.
3. **Multi-Tier Hardware Scalability (The Counterpart Principle)**:
   - **Target Benchmark (Mid-to-High Systems)**: Stunning AAA visual fidelity (Radiance Cascades, PCSS soft shadows, GTAO, Froxel fog, TAA).
   - **Steam Deck & Steam Machine Target (15W TDP, 60 FPS)**: Highly efficient, bandwidth-conscious fallbacks that preserve visual depth at $1280 \times 800$.
   - **Low-End PC & Integrated GPU Fallback**: Clean, artifact-free presentation using lightweight counterparts rather than degraded or broken rendering.
4. **Decisive System Pruning**: Strip redundant prototypes, unmaintained experiments, and bandwidth-heavy dead weight (HBAO, LPV, SSCS, SMAA 2X/4X) to establish clean, predictable boundaries before building post-beta developer tools and distribution pipelines.

---

## 2. Multi-Tier Hardware Scaling Matrix (The Counterpart Principle)

Rather than stripping low-fidelity features, PyMordial maintains **mathematically sound, low-end counterparts** for every high-end feature. This guarantees high framerates and visual coherence from integrated GPUs up to high-end dedicated graphics cards.

```
┌────────────────────────────┬─────────────────────────────┬──────────────────────────────┐
│ High / Ultra Tier (Mid+)   │ Steam Deck / Medium Tier    │ Low-End / Integrated GPU     │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Radiance Cascades (SSRC)   │ SSGI (Half-res, 8 steps)    │ Baked IBL + Split-Sum Env    │
│ GTAO (Full-res, 2 slices)  │ GTAO (Half-res bilateral)   │ SSAO (Fast 8-sample disc)    │
│ PCSS Variable Penumbra     │ Vogel PCF (16 taps)         │ Hard / 4-tap PCF (2 cascades)│
│ TAA + CAS Sharpening       │ TAA (Minimal jitter)        │ FXAA (Single-pass edge)      │
│ Froxel Volumetric Fog 3D   │ Low-res Froxel (80x45x32)   │ Analytic Height/Distance Fog │
│ Full Hi-Z Occlusion + MDI  │ Hi-Z Occlusion + MDI        │ Vectorized CPU Frustum Cull  │
│ Hardware Tessellation/POM  │ POM (8-16 linear steps)     │ Standard Tangent Normal Maps │
│ 2-Bone IK Foot Clamping    │ 2-Bone IK Foot Clamping     │ Ground Ray Snapping          │
└────────────────────────────┴─────────────────────────────┴──────────────────────────────┘
```

### Steam Deck & Steam Machine Target Profile ($1280 \times 800$, 15W TDP)

| Pipeline Pass | Steam Deck Budget Configuration | Frame Time Target |
|---|---|---|
| **G-Buffer MRT Pass** | $1280 \times 800$ Native, MDI MegaBuffer, Albedo/Normal/Motion/Material | $< 2.5\text{ ms}$ |
| **Cascaded Shadows** | 3 Cascades ($2048$ Atlas), 16-tap Vogel PCF, Per-Cascade Frustum Culling | $< 2.2\text{ ms}$ |
| **Deferred Lighting** | Clustered Local Lights + Half-Res SSGI (or IBL + Half-Res GTAO) | $< 3.5\text{ ms}$ |
| **Ambient Occlusion** | Half-Resolution GTAO with spatial bilateral blur | $< 0.8\text{ ms}$ |
| **Post-Processing** | TAA ($0.85$ Feedback, $0.35$ Sharpness), ACES/AgX Tonemap, Bloom | $< 1.2\text{ ms}$ |
| **Atmosphere & Fog** | Low-Res Froxel ($80 \times 45 \times 32$), Directional sun in-scattering | $< 1.1\text{ ms}$ |
| **Rapier3D Physics** | Kinematic Motor + Capsule Casts @ 60 Hz fixed tick | $< 1.5\text{ ms}$ |
| **Frame Total** | **Locked 60 FPS (Rock-solid frame pacing within 15W budget)** | **$\approx 12.8\text{ ms}$** |

---

## 3. Subsystem Architecture & Upgrades (Decima-Inspired)

---

### 3.1 Visual Features & World Generation

#### 3.1.1 Decima-Style Procedural Foliage & Clutter Scatter
* **Alpha State**: Every rendered mesh requires an ECS entity slot and an entry in `world_transforms` SSBO. Scattering tens of thousands of grass blades or pebbles creates severe Python CPU memory and iteration overhead.
* **Decima Architectural Approach**: Decima generates foliage, surface clutter, and rocks procedurally in GPU compute shaders based on density/biome splatmaps, surface slope, and elevation rules.
* **PyMordial Technical Implementation**:
  - **GPU Indirect Scatter Compute Pass**: A compute shader evaluates Blue Noise / Poisson disc distributions across active terrain sectors. Surviving instances write instance transforms directly into a dedicated dynamic SSBO or indirect buffer.
  - **Procedural Wind Displacement**: A shared wind texture / analytic Gerstner wind vector evaluated in vertex shaders (`foliage.vert`) displacing foliage vertices based on height above ground and vertex color ambient bend masks.
  - **Zero CPU Cost**: Foliage instances are never tracked as Python ECS entities, keeping Python memory usage flat and zero-allocation.

#### 3.1.2 Quadtree Continuous Distance LOD (CDLOD) Terrain
* **Alpha State**: Scenes rely on imported static meshes or flat planes with tessellation displacement (`gbuffer_tess.tesc/tese`).
* **Decima Architectural Approach**: Large-scale continuous terrain using hierarchical quadtree streaming with seamless morphological LOD transition ($\alpha$-morph) to eliminate edge cracks without CPU stitch meshes.
* **PyMordial Technical Implementation**:
  - Quadtree nodes represent chunk extents ($64 \times 64$ m, $128 \times 128$ m, etc.) centered around the camera.
  - Heightmap and normal data bound as 2D texture arrays or virtual textures.
  - Vertex shader computes concentric spherical morphing factors:
    $$\alpha = \text{clamp}\left(\frac{\|\mathbf{x}_{\text{cam}} - \mathbf{x}\| - r_{\text{inner}}}{r_{\text{outer}} - r_{\text{inner}}}, 0.0, 1.0\right)$$
    snapping odd vertices to neighbor vertices to eliminate popping.
  - Backed in Rapier by localized `HeightField` colliders generated only in active player vicinity.

#### 3.1.3 Decima Volumetric Cloudscapes & Weather Simulation
* **Alpha State**: Real-time atmospheric Rayleigh and Mie scattering (`engine.gfx.atmosphere.AtmosphereSystem` and `hillaire_*.frag`) and single-scattering volumetric fog slices (`volumetric_fog_inject.comp`).
* **Decima Architectural Approach**: Layered 3D raymarched volumetric cloudscapes (Worley and curl noise) featuring multi-scattering approximations, dynamic coverage maps, and rain shafts.
* **PyMordial Technical Implementation**:
  - 3D Noise Texture ($128 \times 128 \times 128$ RGBA8) containing low-frequency Perlin-Worley noise for cloud shape and high-frequency curl noise for billowing erosion.
  - Quarter-resolution compute raymarch pass (`cloud_raymarch.comp`) evaluated at $1/4$ screen resolution, temporally resolved with neighborhood history clamping, and composited into the deferred resolve.
  - Forward scattering / silver-lining approximation:
    $$P_{\text{Henyey-Greenstein}}(\theta, g_1, g_2) = w \cdot \Phi(g_1) + (1 - w) \cdot \Phi(g_2)$$

---

### 3.2 Graphics & Rendering Architecture

#### 3.2.1 GPU-Driven Occlusion Culling via Hi-Z Depth (Decima Two-Phase Culling)
* **Alpha State**: Frustum culling is currently performed on CPU in Python via vectorized NumPy operations (`engine.gfx.culling.FrustumCuller`), followed by a Python loop filtering contiguous MDI draw batches.
* **Decima Architectural Approach**: Decima completely eliminates CPU-side draw culling. A compute shader takes the previous frame's Hi-Z depth pyramid, projects object bounding boxes to screen space, selects the corresponding depth mip level, and occludes hidden geometry before drawing.
* **PyMordial Technical Implementation**:
  - Utilize PyMordial's existing [`hiz_pass.py`](file:///d:/Projects/PyMordial-Engine/engine/gfx/passes/hiz_pass.py) and [`hiz_downsample.comp`](file:///d:/Projects/PyMordial-Engine/shaders/hiz_downsample.comp).
  - Add compute shader `cull_indirect.comp`:
    1. Read bounding box from `AABB` SSBO.
    2. Frustum test against camera planes.
    3. Hi-Z depth test: Find screen-space footprint $\max(\Delta x, \Delta y)$, sample Hi-Z mip $\lceil \log_2(\text{size}) \rceil$.
    4. Surviving instances write to an indexed instance list or write directly to a `GL_DRAW_INDIRECT_BUFFER` using `atomicAdd`.
  - Dispatched via `glMultiDrawElementsIndirect` with zero Python iteration overhead.

#### 3.2.2 Cascaded Shadow Map (CSM) Per-Cascade Culling & Static Cache
* **Alpha State**: `pipeline.py` loads all non-tessellated shadow casters into `shadow_mdi` unconditionally across all shadow cascades, causing heavy vertex overdraw.
* **Decima Architectural Approach**: Decima strictly culls geometry per shadow cascade and partitions static environment geometry into cached shadow atlas pages that only redraw when sun position shifts significantly.
* **PyMordial Technical Implementation**:
  - Test entity AABBs against each cascade's light-space orthographic projection bounds.
  - Staggered static cascade updates: Cascade 0 (near) updates every frame; Cascade 3 (far) updates every 4–8 frames using `ecs.is_static`.
  - Sloped Depth Bias:
    $$\text{bias} = \text{clamp}\left(\text{bias}_{\text{base}} \cdot \tan(\arccos(N \cdot L)), \text{bias}_{\text{min}}, \text{bias}_{\text{max}}\right)$$

#### 3.2.3 Global Illumination (GI) Consolidation
* **What to Keep**:
  - **High/Ultra**: **Radiance Cascades (SSRC + FFPC)**. Multi-interval raymarching with probe cascades for rich, multi-bounce indirect diffuse light.
  - **Steam Deck / Medium**: **Screen-Space GI (SSGI)** at half-resolution (8 ray steps). Great real-time bounce for on-screen geometry with low arithmetic cost on RDNA2 APUs.
  - **Low-End**: **IBL Split-Sum** with prefiltered environment radiance map. Instant indirect ambient fill at essentially zero GPU compute cost.
* **What to Remove**:
  - **Remove 3D Light Propagation Volumes (`lpv_pass.py`, `lpv_propagate.comp`)**:
    - *Why*: LPV requires huge 3D volume textures ($32 \times 16 \times 32$), consumes excessive VRAM memory bandwidth, and leaks light through thin walls and floors. Removing LPV saves significant code complexity without sacrificing visual quality at any tier.

#### 3.2.4 Ambient Occlusion (AO) Consolidation
* **What to Keep**:
  - **High/Ultra**: **GTAO (Ground Truth Ambient Occlusion)** with 2-direction horizon integration and spatial bilateral blur.
  - **Steam Deck / Low-End**: **SSAO (Screen-Space Ambient Occlusion)** via [`ssao.frag`](file:///d:/Projects/PyMordial-Engine/shaders/ssao.frag) using an 8-sample Poisson disc or half-resolution GTAO.
* **What to Remove**:
  - **Remove HBAO (`hbao.frag`)**: HBAO sits in an awkward middle ground—it is heavier than SSAO, less accurate than GTAO, and requires an extra compile target.
  - **Purge Screen-Space Contact Shadows (SSCS)**: GTAO with small radius provides superior sub-centimeter contact darkening without the raymarching artifacts, false self-shadowing, or camera-edge popping inherent to SSCS.

---

### 3.3 Post-Processing & Camera Optics

#### 3.3.1 Temporal Anti-Aliasing (TAA) Standard & Depth-Dilated Motion Vectors
* **What to Keep**:
  - **High / Medium / Steam Deck**: **TAA (Temporal Anti-Aliasing)** with depth-dilated velocity sampling and subpixel Halton jittering. On Steam Deck's 800p screen, TAA is crucial to stabilize high-frequency specular shimmer and resolve stochastic samples.
  - **Low-End / Integrated GPU**: **FXAA (Fast Approximate Anti-Aliasing)**. Single-pass, zero history buffers, zero jittering ($< 0.15\text{ ms}$).
* **What to Remove**:
  - **Remove SMAA 2X / 4X (`smaa_pass.py`)**: Requires multi-sample temporal reprojection and 4 separate ping-pong textures. In deferred pipelines, it consumes far too much memory bandwidth compared to TAA. (Keep simple SMAA 1X or FXAA for non-temporal users).
* **Depth-Dilated Velocity Fetching**:
  - In `taa.frag`, sample a $3 \times 3$ neighborhood and choose the velocity of the pixel with the closest depth to the camera. This completely prevents background ghosting behind moving foreground silhouettes.

#### 3.3.2 Velocity Tile-Max Motion Blur (McGuire et al.)
* **Alpha State**: Motion vectors are written to G-Buffer RT2, but post-processing currently lacks cinematic camera and object motion blur.
* **Decima Architectural Approach**: Decima features high-quality camera and per-object motion blur computed via a 2-pass velocity tile-max filter to avoid ghosting and boundary bleeding.
* **PyMordial Technical Implementation**:
  - Downsample velocity buffer into tile-max texture ($16 \times 16$ or $32 \times 32$ pixels per tile).
  - Neighbor-max search pass in compute shader to find dominant velocity.
  - Directional blur reconstruction gathering variable samples along tile velocity.

#### 3.3.3 Decima-Grade Film Grain & Tonemapping
* **Implementation**: Luminance-modulated Simplex/blue-noise grain:
  $$I_{\text{final}} = I_{\text{linear}} \cdot (1.0 + \text{grain\_noise} \cdot (1.0 - L)^{1.5} \cdot \sigma)$$
* Integrated **AgX** and **ACES** tonemapping curves preventing color skewing in overexposed highlights.

---

### 3.4 Physics & Motion Dynamics (Rapier3D Integration)

#### 3.4.1 Procedural Two-Bone IK Foot Clamping & Terrain Alignment
* **Alpha State**: `CharacterMotor` performs capsule shape casting for non-penetrating movement, stair stepping, and slope sliding. However, character feet do not conform to uneven terrain or boulders.
* **Decima Architectural Approach**: Aloy and creatures dynamically adapt their stance to slopes, rocks, and uneven surfaces using analytical two-bone inverse kinematics (hips, knees, ankles) and pelvic tilt adjustment.
* **PyMordial Technical Implementation**:
  - In `on_fixed_update`: Cast downward rays from virtual left and right hip sockets into Rapier:
    `hit_l = world.cast_ray(origin_l, down_dir, max_dist=1.5)`
    `hit_r = world.cast_ray(origin_r, down_dir, max_dist=1.5)`
  - Compute pelvic offset:
    $$\Delta y_{\text{pelvis}} = \min(h_l, h_r)$$
  - Solve analytical 2-bone cosine IK:
    $$\cos(\theta_{\text{knee}}) = \frac{L_1^2 + L_2^2 - d_{\text{target}}^2}{2 L_1 L_2}$$
  - Write adjusted foot/pelvis transforms to `world_transforms` SSBO with zero allocation.
  - *Low-End Fallback*: Direct ground ray snapping without full knee trigonometry.

#### 3.4.2 Dynamic Secondary Physics (Cloth, Hair, Antenna Sway)
* **Implementation**: Lightweight vectorized Position-Based Dynamics (PBD) / Verlet particle chains running alongside kinematic characters for fluid, secondary mechanical and clothing motion.
  $$\mathbf{x}_{i}(t + \Delta t) = 2\mathbf{x}_{i}(t) - \mathbf{x}_{i}(t - \Delta t) + \mathbf{a} \Delta t^2$$
* Output transforms directly injected into instance SSBOs.

#### 3.4.3 Rapier HeightField Terrain Colliders
* **Implementation**: Wrap Rapier's native `Collider.heightfield` with 2D elevation grid data matching the active CDLOD terrain chunks. Enables sub-millisecond ground clamping and physics collision across square kilometers of terrain.

---

### 3.5 Storage, Virtual File System (VFS) & Asset Streaming

#### 3.5.1 BC7 / BC5 GPU Hardware Texture Compression Pipeline
* **Alpha State**: Textures are cooked via `engine.assets.cooker.py` using NumPy arrays and PNG/RAW formatting.
* **Decima Architectural Approach**: All albedo, normal, and displacement textures are streamed as hardware-compressed Block Compression formats (BC7 for high-fidelity albedo, BC5 for 2-channel tangent normals, BC4 for roughness/height).
* **PyMordial Technical Implementation**:
  - Integrate native BC7/BC5 cooker utilities (`ispc_texcomp` or `nvcompress` via Python ctypes/subprocess).
  - ModernGL native loading of compressed texture data directly into `TextureArrayAtlas`:
    - Reduces GPU VRAM footprint by **75%** (from 64 MB down to 16 MB per 4K PBR layer).
    - Drastically cuts disk I/O time during open-world streaming on Steam Deck and low-end storage.

#### 3.5.2 Chunk-Based Asynchronous Streaming & Prefetching
* **Implementation**:
  - **Asynchronous Disk Queue (`engine.assets.streaming_queue.AssetStreamingQueue`)**: Background Python worker threads fetch uncooked or pre-cooked byte blocks into pinned memory.
  - **OpenGL Buffer Upload Subsystem**: Leverage Persistent Mapped Buffers (`GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT`) or ring-buffered PBOs limited to a fixed upload budget per frame (e.g., max 8 MB per frame) to guarantee smooth 60 FPS frame pacing.

#### 3.5.3 Zero-Copy Packed Binary Scene Graph
* **Implementation**: Pre-pack scene entity tables directly into `.pm_scene` flat binary files matching the exact SoA layout of `rigid_body_state` and `material_data`. Loaded via `np.fromfile()` into pre-allocated ECS memory blocks in $< 2\text{ ms}$.

---

### 3.6 3D Spatialized Audio Subsystem

#### 3.6.1 Distance-Based Air Absorption & High-Frequency Low-Pass Filtering
* **Implementation**: 1-pole Low-Pass Filter (LPF) cutoff attenuating high frequencies ($> 4\text{ kHz}$) over long outdoor distances:
  $$f_{\text{cutoff}}(d) = f_{\max} \cdot \left(\frac{1}{1 + \alpha_{\text{air}} \cdot d}\right)$$
  Audio cues in the distance sound muffled and bass-heavy, creating realistic open-world depth.

#### 3.6.2 Physics-Driven Acoustic Occlusion & Geometry Obstruction
* **Implementation**: Real-time Rapier raycasts between listener and sound emitters against physics collision geometry. When obstructed by walls or terrain, apply low-pass filtering and volume dampening.

#### 3.6.3 Dynamic Biome Ambience & Wind Convolution
* **Implementation**: Procedural ambience manager sampling weather wind speed and player height, blending between rustling leaf layers, distant mountain gusts, and rain reverberation.

---

## 4. Systems Consolidation: Summary of Pruning Decisions

```
[SYSTEM]                     [RETAINED FOR PRODUCTION]                      [PRUNED / DEPRECATED]
Ambient Occlusion     ──>    GTAO (High/Deck) + SSAO (Low)          ──>     HBAO (hbao.frag), SSCS
Global Illumination   ──>    Radiance Cascades, SSGI, IBL           ──>     3D LPV (lpv_pass.py, lpv_propagate.comp)
Anti-Aliasing         ──>    TAA (High/Deck) + FXAA (Low)           ──>     SMAA 2X/4X multi-sample buffers
Shadow Mapping        ──>    PCSS, Vogel PCF, Hard PCF              ──>     SSCS uniforms / raymarching
Scatter / Clutter     ──>    GPU-Driven Indirect Scatter            ──>     Individual ECS entities for foliage
```

---

## 5. Clean Beta Codebase Architecture

```
PyMordial-Engine/
├── engine/
│   ├── app/                 # Application host & lifecycle (EngineLoop)
│   ├── assets/              # VFS, BC7/BC5 texture loader, MegaBuffer mesh packaging
│   ├── audio/               # 3D spatialized audio, distance LPF, Rapier occlusion raycasts
│   ├── camera/              # Camera matrices, Frustum extraction, Halton subpixel jitter
│   ├── config/              # Central CVars, settings TOML serialization
│   ├── core/                # EngineLoop, SoA EntityManager, EntityPool, Math utilities
│   ├── debug/               # Central Profiler, DebugDraw, F1-F3 Subsystem (Dev-only)
│   ├── gfx/
│   │   ├── passes/          # GTAO, SSAO, Radiance Cascades, SSGI, Clustered Lights, TAA, Fog
│   │   ├── pipeline.py      # Master Render Graph Coordinator
│   │   ├── mega_buffer.py   # Unified vertex/index allocations & MDI dispatch
│   │   ├── g_buffer.py      # High-efficiency MRT targets & Reversed-Z depth
│   │   └── shadow_csm.py    # Per-cascade culled shadow mapping
│   ├── physics/             # Rapier3D integration, CharacterMotor, Heightfield, Raycasts, IK
│   ├── ui/                  # Scaled 1280x720 responsive UI canvas
│   └── world/               # World sectors, chunked streaming, procedural foliage scatter
└── projects/                # Game projects (isolated, consuming ProjectModule)
```

---

## 6. Post-Beta Foundation: Development Tools & Distribution Pipeline

Once Beta stability and multi-tier visual performance are achieved, development transitions into Game Creator Tooling and Automated Distribution:

### 6.1 Editor & Developer Tooling
1. **In-Engine World & Entity Inspector**: Visual manipulation of transforms, light parameters, and material layers rendered on the scaled $1280 \times 720$ UI canvas.
2. **CLI Asset Cooking Pipeline (`engine.tools.cooker`)**: One-command compilation of raw OBJ/GLTF/PNG assets into optimized `.pm_mesh` and hardware-compressed `.pm_tex_bc7` formats.
3. **Live Material & Shader Hot-Reloading**: File watcher recompiling ModernGL shader programs on disk modification without restarting the simulation.

### 6.2 Distribution & Packaging Pipeline
1. **Unified Packager (`engine.tools.packager`)**:
   - Automated dual-target packaging via **Nuitka** (C-compiled binary for maximum execution speed) or **PyInstaller**.
   - Strip all developer overlays and hotkeys (`F1-F3`, `DebugDraw`, `SystemMonitor`) in `--release` builds.
   - Tree-shake unused gameplay modules and pack assets into zero-copy, memory-mapped `game.pak` archives.
2. **Steam & Standalone Distribution Ready**:
   - Native Steam Deck / SteamOS controller configuration out of the box.
   - Self-contained executable requiring zero Python or external dependency installation from the end user.

---

## 7. Phased Implementation Roadmap

| Phase | Milestone | Subsystem | Action Items | Hardware Impact |
|---|---|---|---|---|
| **Phase 1** | **Alpha Cleanup & Pruning** | Graphics / Shaders | • Implement Per-Cascade Frustum Culling in `shadow_csm.py`<br>• Clean up `RenderConfig` (purge SSCS & unused aliases)<br>• Prune `hbao.frag` and retire `lpv_pass.py` / `lpv_propagate.comp`<br>• Update `QUALITY_PRESETS` for High, Steam Deck, and Low tiers | 40–60% vertex reduction on shadows; eliminates VRAM bandwidth leaks on Steam Deck |
| **Phase 2** | **Decima Beta Visuals** | Graphics / World / Assets | • Add GPU-Driven Hi-Z Occlusion Culling (`cull_indirect.comp`)<br>• Implement BC7/BC5 hardware texture compression pipeline<br>• Add Depth-Dilated Velocity Fetching to `taa.frag`<br>• Sloped Depth Bias for CSM shadows | Massively scales instance counts to 100k+; cuts VRAM footprint by 75% |
| **Phase 3** | **Motion & World Dynamics** | Physics / World / Audio | • Add Procedural 2-Bone IK Foot Clamping in `CharacterMotor`<br>• Implement GPU-driven foliage scatter with Gerstner wind sway<br>• Add Distance LPF air absorption and Rapier acoustic occlusion in `AudioEngine` | High-fidelity terrain traversal and immersive spatialized Decima audio |
| **Phase 4** | **Terrain & Cloudscapes** | World / Graphics | • CDLOD Quadtree terrain manager backed by Rapier `HeightField`<br>• 3D Perlin-Worley raymarched volumetric cloudscapes pass | Seamless open-world landscape streaming without memory stalls |
| **Phase 5** | **Tooling & Distribution** | Tools / Packaging | • In-engine UI Entity Inspector<br>• Live shader hot-reloader<br>• Automated Nuitka `--release` packager with `game.pak` asset archiving | Production-ready developer workflow and standalone game distribution |
