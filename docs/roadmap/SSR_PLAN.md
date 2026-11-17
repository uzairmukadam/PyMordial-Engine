# PyMordial Engine: Hi-Z Screen-Space Reflections (SSR) Design

> **Status: DONE — Hi-Z accelerated variant implemented.** `SSRPass` supports both `RenderConfig.ssr_mode = "HIZ"` (default) and `"LINEAR"` (legacy). The Hi-Z march strides empty space in coarse mips and refines on contact, sharing the linear path's hit validation so both modes agree on accepted hits. Hi-Z empty-space skipping is also wired into SSGI. See [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) for scheduling.

Here is a production-ready, highly efficient design for a custom **Hierarchical Screen-Space Reflection (Hi-Z SSR)** pipeline.

By combining **roughness masking**, **per-pixel normal evaluation**, and a **Hi-Z depth pyramid**, this architecture effortlessly handles both flat and curved surfaces while avoiding the performance bottlenecks of naive ray marching.

---

### The Architecture: 5-Stage Hi-Z SSR Pipeline

#### 1. Input Preparation & Roughness Masking (The G-Buffer Pass)

Instead of tracing reflections for every single pixel, you cull unnecessary work upfront using your PBR data:

* **Inputs Required:** Scene Color, Screen-Space Normals (with micro-detail normal maps applied), Roughness, and Depth.
* **The Masking Logic:** Create a binary or gradient reflection mask based on the roughness threshold (e.g., pixels with `Roughness > 0.8` are completely skipped). For valid pixels, compute the reflection vector $R$ using the view vector $V$ and the per-pixel normal $N$:

$$R = reflect(-V, N)$$


* *Why this handles curved surfaces:* Because $N$ is evaluated *per-pixel* from the normal map, a curved car fender or a bumpy rock wall will naturally scatter reflection vectors in different directions, completely eliminating the need for a global "buffer flip."

#### 2. Hi-Z Depth Pyramid Generation (The Performance Engine)

To make ray marching fast and prevent it from missing thin geometry, you generate a **Hi-Z (Hierarchical-Z) buffer**:

* At the start of the frame, take your main depth buffer and downscale it into a mip chain (Mip 0 = full res, Mip 1 = half res, etc.).
* Each higher mip level stores the **furthest depth** (conservative depth) of a $2\times2$ block of pixels from the level below.
* **Why it matters:** When a ray travels across the screen, it can start by taking massive steps through coarse mip levels (skipping empty space quickly) and automatically drop down to finer mip levels when it gets close to an object. This turns an $O(N)$ linear search into a lightning-fast logarithmic search.

#### 3. Hierarchical Screen-Space Ray Marching

This is where your ray looks for an intersection in the depth buffer:

* Project the reflection vector $R$ into screen space.
* March along the ray in screen coordinates using the Hi-Z pyramid.
* **The Intersection Test:** At each step, check the ray's depth against the depth stored in the Hi-Z mip level. If the ray is in front of the surface, take a big step forward. If it crosses behind the surface depth, step back a level to find the exact intersection pixel.
* *Handling Off-Frustum Data:* To address your idea of utilizing surfaces just outside the visible frustum, you can pad your Hi-Z generation bounds slightly beyond the screen edges (e.g., a 10% render border). This allows reflection rays to "peek" just outside the viewport before fading out.

#### 4. Validation and Artifact Reduction

Screen-space ray marching is prone to false hits and light-leaking. Apply three safety checks when a hit is found:

* **Normal Consistency Check:** Compare the normal of the hit pixel with your reflection vector. If they point in opposite directions, it means the ray hit the *backside* of an object or an occluded surface. Discard the hit.
* **Thickness Test:** Ensure the distance between the ray depth and the surface depth is within a reasonable threshold (e.g., the object isn't infinitely thin). This prevents reflections from bleeding through walls.
* **Screen-Edge Fade:** Smoothly fade the reflection alpha to 0 as the ray approaches the absolute edge of the screen buffer, blending gracefully into your fallback environment probes.

#### 5. Temporal Denoising (The Secret to High Quality)

Raw screen-space reflections look noisy because pixels change frame-by-frame. To make it look clean:

* Run a **Temporal Accumulation pass** that blends the current frame's reflections with the previous frame's reprojected reflections using velocity vectors.
* Follow it up with a small, edge-aware bilateral blur or a spatial denoiser to smooth out any remaining grain without smudging sharp reflections on glossy surfaces.

---

### Why This Design Wins

* **Flat & Curved Unified:** Because it relies on per-pixel reflection vectors and ray marching rather than geometry-specific mirroring, a flat floor and a complex curved sculpture use the exact same mathematical code path.
* **Performance Scalability:** Hi-Z acceleration means you can afford higher step counts and finer precision without tanking your frame rate.
* **Realistic PBR Integration:** Roughness masking ensures that rough surfaces output soft, wide, glossy reflection blobs instead of sharp, distracting mirrors.