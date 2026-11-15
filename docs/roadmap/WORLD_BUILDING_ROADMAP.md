# PyMordial Engine: World Building & Streaming Roadmap

> **Status: ANALYSIS — gap assessment.** This document evaluates what the engine has today against what open-world support requires. It is a problem statement, not a plan of record; Horizon 4 in [MASTER_FUTURE_PLAN.md](./MASTER_FUTURE_PLAN.md) is the authoritative schedule.

The engine has useful foundations—contiguous ECS storage, static/dynamic transform uploads, frustum/Hi-Z culling, a render graph, a VFS, and a background asset loader. But those features don't yet add up to open-world support: the world-building interface is basic, rendering still has significant CPU-side work, and there's no world-cell streaming system.

### Biggest gaps

| Area | Current foundation | What larger levels need |
|---|---|---|
| **World and streaming** | `BaseWorldBuilder` builds and tears down a world as a whole; `BackgroundAssetLoader` primarily handles material decoding. | Divide levels into cells/sectors with load, activate, deactivate, and unload stages. Stream cells around the player asynchronously, prioritize nearby/visible cells, and avoid frame spikes during GPU upload. |
| **Entity and scene data** | `EntityManager` uses dense arrays and supports a static/dynamic split. | Organize static entities spatially by cell, not just as one global set. Add cell-level entity and draw-batch ownership, efficient unloads, and dirty-region updates. Revisit fixed capacities: the renderer currently reserves SSBOs for 100,000 entities. |
| **Rendering and draw submission** | `FrustumCuller` provides vectorized visibility tests, and `Hi-Z culling` exists. | Make the draw path scale with visible work: add real GPU-driven indirect submission, spatial/HLOD culling, and mesh LODs. A key implementation detail: `MultiDrawIndirect.submit` currently loops over commands and calls `vao.render` for each one; it isn’t issuing a hardware multi-draw call. That can become a CPU bottleneck in a dense city. |
| **Geometry and terrain** | `MegaBuffer` packs meshes together, and `DefaultWorldBuilder` can generate ground meshes. | Add terrain tiles with LOD and seam handling, reusable building/prop instances, HLOD for distant blocks, and collision proxies. Avoid rebuilding the whole shared buffer whenever streaming adds geometry. |
| **Materials and textures** | `TextureArrayAtlas` provides cached texture arrays, while `ResourceCache` caches loaded assets. | Add a residency budget, background mip/texture streaming, and eviction. The atlas has a 32-layer limit and fixed resolution, so it won’t be a sufficient general strategy for a large varied world without extension or an alternative texture system. |
| **Lighting and shadows** | The renderer has CSM, local lights, and several GI options in `quality_presets.py`. | Select lights and shadow casters by nearby cells, add distance/importance LOD for shadows, and establish a scalable outdoor GI strategy. Screen-space effects such as RC and SSR can’t provide complete off-screen or distant-world lighting by themselves. |
| **Physics and gameplay** | `PhysicsManager` wraps Rapier and synchronizes physics with the ECS. | Load collision only for active nearby regions; use simplified static collision for buildings and terrain. Navigation, AI, and audio also need region-aware data and activation rather than assuming the whole level is active. |
| **Large-world coordinates** | Transforms are stored as floats in the ECS and sent to GPU buffers. | Use cell-local coordinates and origin rebasing before world distances make float precision visible in camera movement, physics, or shadows. |

### Recommended order

1. **Profile and set budgets first.** Measure CPU and GPU frame time separately, along with draw calls, visible entities, resident asset memory, and streaming spikes. The current `SystemMonitor` tracks useful CPU/frame metrics, but GPU pass timings and streaming/resource residency need to be added.
2. **Build a small world-partition prototype.** Stream a handful of neighboring cells in and out, including their entities, render batches, collision, and assets. Test rapid travel and unloading—not just initial loading.
3. **Fix the renderer’s scaling path.** Implement true indirect batching, then add per-cell culling and LOD/HLOD. Measure CPU submission cost before increasing scene density.
4. **Create a representative city-block test map.** Include repeated buildings and props, unique hero assets, terrain, lights, shadows, and collision. Set frame-time and memory targets on the hardware you care about.
5. **Only then expand the content and streaming distance.** Use the measurements to decide where texture streaming, shadow changes, physics deactivation, or origin rebasing becomes necessary.

I would **not** start by raising entity capacity or building a huge city in one loaded scene. The most valuable first milestone is a small streamed district that proves load/unload behavior and stable frame times. That will reveal which subsystem actually needs the next investment.