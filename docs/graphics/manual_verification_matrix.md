# Manual Verification Matrix — Phase 2

> **Documentation-only.** This file defines a small, deterministic scene set and the exact
> procedure to verify each graphics feature from
> [feature_inventory.md](./feature_inventory.md).
>
> **Rules for executing this matrix**
> 1. Use the **same** resolution (1280×720), quality preset, warm-up (60 frames), and sample count
>    for every A/B comparison.
> 2. Record **frame time in ms**, never FPS, for any performance claim.
> 3. Do **not** mark a feature PASS because it compiles or produces *some* output. PASS requires the
>    stated expected behaviour to be observed.
> 4. Capture a screenshot for every FAIL and every PASS WITH ISSUES.
> 5. Change **one** variable per test. If two features are toggled together, the result is not
>    attributable.

---

## 1. Scene Set

Nine scenes. Each isolates a lighting/material/geometry condition. None is a "kitchen sink".

| ID | Name | Purpose | Covers |
|---|---|---|---|
| **S1** | Indoor Box Room | diffuse bounce, occlusion, local lights | indoor diffuse, AO, GI color bleed, Hi-Z occlusion, point/spot lights |
| **S2** | Material Grid | PBR response | rough vs metallic, energy compensation, flat sky |
| **S3** | Detail Wall | high-frequency geometry | POM, tessellation, SSDM, specular AA, thin geometry, AA |
| **S4** | Moving Objects | temporal correctness | motion vectors, TAA, motion blur, disocclusion |
| **S5** | Mirror Floor | reflections + optics | SSR, IBL, DoF, reflective materials |
| **S6** | Outdoor Vista | sun, sky, shadows, depth | directional light, CSM, atmosphere, fog, tonemap, exposure, day/night |
| **S7** | Emissive & Lights | emissive + particles | bloom, particles, spot shadows, emissive surfaces |
| **S8** | Contact Crevices | near-contact AO/GI | AO, GI contact darkening, shadow boundaries |
| **S9** | Water Chambers | water + refraction | water, refraction, transparency |

### 1.1 Scene construction (deterministic)

All scenes are built with the existing public API only. No engine change is required.

```python
# Common preamble (all scenes)
from engine.app.project_app import ProjectApp
from engine.app.config import ProjectConfig
from engine.core import WaterProperties

cfg = ProjectConfig(
    title="verify", width=1280, height=720,
    quality_preset="ultra", enable_debug=True, headless=False,
    max_frames=None,
)
app = ProjectApp(cfg)
```

#### S1 — Indoor Box Room
```python
# 12x4x12 room, open ceiling, one point light, one spot light
add_box(app, (0, -0.1, 0), (12, 0.2, 12), (0.55, 0.52, 0.48), 0.85)   # floor
add_box(app, (0, 2.0, -6), (12, 4, 0.2), (0.70, 0.68, 0.64), 0.80)    # back wall
add_box(app, (-6, 2.0, 0), (0.2, 4, 12), (0.70, 0.68, 0.64), 0.80)    # left wall
add_box(app, (6, 2.0, 0), (0.2, 4, 12), (0.70, 0.68, 0.64), 0.80)     # right wall
# Coloured panels for GI colour bleed
add_box(app, (-5.85, 1.6, 0), (0.1, 2.4, 3.0), (0.85, 0.15, 0.12), 0.75)  # red
add_box(app, (5.85, 1.6, 0), (0.1, 2.4, 3.0), (0.12, 0.35, 0.85), 0.75)   # blue
# Occluder for Hi-Z culling
add_box(app, (0, 1.5, 2.0), (3.0, 3.0, 0.2), (0.60, 0.60, 0.60), 0.8)
app.pipeline.add_point_light((0.0, 3.0, 0.0), radius=9.0, color=(1.0, 0.95, 0.85), intensity=6.0)
app.pipeline.add_spot_light((0.0, 3.8, -4.0), (0.0, -1.0, 0.3), radius=12.0,
                            inner_cone_angle=0.35, outer_cone_angle=0.55,
                            color=(1.0, 0.9, 0.8), intensity=8.0, cast_shadow=True)
```

#### S2 — Material Grid
```python
# 5x5 grid of spheres: roughness 0.05..0.95 across X, metallic 0..1 across Z
for i in range(5):
    for j in range(5):
        app.ecs.create_entity(
            shape="sphere", position=((i-2)*2.0, 1.0, (j-2)*2.0),
            dimensions=(1.0, 1.0, 1.0),
            color=(0.85, 0.85, 0.88),
            roughness=0.05 + i * 0.225,
            metallic=j * 0.25,
            is_static=True,
        )
add_box(app, (0, -0.1, 0), (14, 0.2, 14), (0.35, 0.35, 0.38), 0.9)
# Flat sky for isolation
cfg.atmosphere_enabled = False   # set BEFORE ProjectApp(cfg)
```

#### S3 — Detail Wall
```python
# Brick/concrete wall with POM + tessellation materials, plus thin geometry
add_box(app, (0, 2.0, -4), (10, 4, 0.3), (0.6, 0.55, 0.5), 0.8, material_id="brick_wall")
add_box(app, (0, -0.1, 0), (10, 0.2, 10), (0.5, 0.5, 0.5), 0.85, material_id="concrete_floor")
# Thin geometry: 20 vertical slats 2cm thick
for k in range(20):
    add_box(app, (-4.5 + k*0.47, 1.5, 0.0), (0.02, 3.0, 0.02), (0.9, 0.9, 0.9), 0.4)
```

#### S4 — Moving Objects
```python
# 8 dynamic cubes orbiting a static pillar; camera static
add_box(app, (0, 1.5, 0), (1.0, 3.0, 1.0), (0.7, 0.7, 0.7), 0.6)
for k in range(8):
    e = app.ecs.create_entity(shape="cube", position=(3.0, 1.0, 0.0),
                              dimensions=(0.6, 0.6, 0.6),
                              color=(0.9, 0.4, 0.2), roughness=0.4, metallic=0.2)
    app.physics.create_body(e, body_type="kinematic", position=(3.0, 1.0, 0.0))
# Drive them in a module's on_fixed_update with a deterministic angle from fixed_tick_count
```

#### S5 — Mirror Floor
```python
add_box(app, (0, -0.1, 0), (20, 0.2, 20), (0.9, 0.9, 0.92), 0.03, metallic=1.0)  # mirror
# Foreground and background objects for DoF separation
add_box(app, (0, 1.0, 2.0), (1.0, 2.0, 1.0), (0.8, 0.2, 0.2), 0.5)
add_box(app, (0, 3.0, -18.0), (4.0, 6.0, 4.0), (0.2, 0.4, 0.8), 0.5)
```

#### S6 — Outdoor Vista
```python
add_box(app, (0, -0.1, 0), (400, 0.2, 400), (0.35, 0.42, 0.28), 0.9)   # ground
# Depth range: near pillar, mid buildings, far ridge
add_box(app, (2, 2.0, 4.0), (1.0, 4.0, 1.0), (0.7, 0.7, 0.7), 0.7)
for k in range(6):
    add_box(app, (-40 + k*16, 8.0, -60.0), (10, 16, 10), (0.55, 0.55, 0.6), 0.8)
add_box(app, (0, 20.0, -300.0), (200, 40, 10), (0.4, 0.45, 0.5), 0.9)  # far ridge
# Atmosphere preset EARTH_DAY; time_of_day swept for day/night
```

#### S7 — Emissive & Lights
```python
add_box(app, (0, -0.1, 0), (16, 0.2, 16), (0.25, 0.25, 0.28), 0.9)
# Emissive panels
for k in range(4):
    app.ecs.create_entity(shape="cube", position=(-4.5 + k*3.0, 2.0, -5.0),
                          dimensions=(1.5, 1.5, 0.1),
                          color=(1.0, 0.85, 0.6), emissive_intensity=12.0, is_static=True)
# Spot light with shadow
app.pipeline.add_spot_light((0, 5.0, 0), (0, -1, 0), radius=14.0,
                            inner_cone_angle=0.3, outer_cone_angle=0.5,
                            color=(1, 1, 1), intensity=10.0, cast_shadow=True)
# Particle emitter
app.pipeline.particle_pass.add_emitter(mode="EMBERS", center=(0, 1.0, 0),
                                       half_extents=(3, 1, 3), count=512)
```

#### S8 — Contact Crevices
```python
add_box(app, (0, -0.1, 0), (10, 0.2, 10), (0.6, 0.6, 0.6), 0.85)
# Stacked boxes with tight gaps (2cm) to stress AO/GI contact
for k in range(6):
    add_box(app, (0, 0.5 + k*1.0, 0), (1.0, 1.0, 1.0), (0.7, 0.7, 0.7), 0.8)
    add_box(app, (1.02, 0.5 + k*1.0, 0), (1.0, 1.0, 1.0), (0.7, 0.7, 0.7), 0.8)
# A sphere resting in a corner
app.ecs.create_entity(shape="sphere", position=(4.5, 0.5, 4.5),
                      dimensions=(1.0, 1.0, 1.0), color=(0.8, 0.8, 0.8), roughness=0.7)
```

#### S9 — Water Chambers
```python
# Reuse examples/03_water_lighting_lab.py as the canonical S9 scene.
# Run: python examples/03_water_lighting_lab.py
```

---

## 2. Camera Conditions

| Condition | Definition | Used by |
|---|---|---|
| **C-static** | camera fixed, no input | S1, S2, S3, S7, S8 |
| **C-pan-slow** | yaw 10°/s for 5 s | S3, S6 (shadow crawl) |
| **C-pan-fast** | yaw 120°/s for 1 s | S4, S5 (motion blur) |
| **C-dolly-near** | move from 20 m to 2 m from target | S3 (tessellation radius) |
| **C-dolly-far** | move from 2 m to 400 m | S6 (cascade transitions) |
| **C-graze** | camera 5° above surface plane | S3 (POM), S5 (SSR) |
| **C-look-up** | pitch +60° | S6 (clouds, sky) |
| **C-look-down** | pitch −60° | S5 (mirror floor) |
| **C-occluded** | camera behind a wall looking at geometry behind it | S1 (Hi-Z culling) |
| **C-disocclude** | fast lateral move revealing previously hidden geometry | S4 (TAA/SSGI ghosting) |

---

## 3. Feature → Scene → Procedure Matrix

Each row is one verification. "Toggle" means the exact config field to change.

### 3.1 Geometry

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| G1 | MDI base geometry | S1 | C-static | render | all boxes at correct transforms | no collapsed/missing meshes |
| G2 | MDI fallback | S1 | C-static | force `hardware_mdi=False` (code) | identical image | pixel-diff ≈ 0 |
| G3 | Tessellation | S3 | C-dolly-near | toggle `tess_enabled` | relief appears near | relief visible < 120 m, flat > 300 m, no cracks |
| G4 | Wireframe | S1 | C-static | toggle `wireframe` | edges visible | toggles cleanly |

### 3.2 Material / PBR

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| M1 | PBR base | S2 | C-static | render | rough vs metal distinct | metal tints specular; rough broadens highlight |
| M2 | Energy compensation | S2 | C-static | toggle `energy_compensation_enabled` | rough metal brightness preserved | rough metal luminance within 10% of smooth |
| M3 | Specular AA | S3 | C-pan-slow | toggle `specular_aa_enabled` | no specular crawl | no shimmer on slats |
| M4 | POM | S3 | C-graze | toggle `pom_enabled` | bricks recessed | relief stable, no swimming |
| M5 | SSDM | S3 | C-dolly-far | toggle `ssdm_enabled` | relief at mid distance | relief 30–300 m, silhouettes unchanged |
| M6 | Material atlas | S3 | C-static | render | correct material per surface | brick vs concrete distinct |

### 3.3 Direct Lighting

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| L1 | Directional sun | S6 | C-static | render | sunlit scene | shadows align with sun |
| L2 | Point lights | S1 | C-static | render | local illumination | inverse-square falloff |
| L3 | Spot lights | S7 | C-static | render | cone falloff | smooth cone edge |
| L4 | Spot shadows | S7 | C-static | toggle `spot_shadows_enabled` | shadow from spot | shadow attached to caster |

### 3.4 IBL

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| I1 | Split-sum IBL | S5 | C-static | toggle `ibl_enabled` | ambient from sky | metal reflects sky; ambient not black |
| I2 | `gi_mode="IBL"` | S5 | C-static | set `gi_mode="IBL"` then `"OFF"` | identical GI | GI_TOTAL debug view identical |

### 3.5 Shadows

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| SH1 | CSM | S6 | C-dolly-far | render | stable shadows | no cascade pop |
| SH2 | Hard filter | S6 | C-static | `shadow_filter_mode="HARD"` | binary edges | no gradient |
| SH3 | PCF | S6 | C-static | `shadow_filter_mode="PCF"` | soft edges | smooth penumbra, no banding |
| SH4 | PCSS | S6 | C-static | `shadow_filter_mode="PCSS"` | contact hardening | penumbra widens with distance |
| SH5 | Stabilization | S6 | C-pan-slow | render | no crawl | shadows static |
| SH6 | Legacy removal | — | — | grep shader | absent | `u_SunAngularRadius` **present** (known fail) |

### 3.6 AO

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| A1 | SSAO | S8 | C-static | `ao_mode="SSAO"` | crease darkening | no halo on silhouettes |
| A2 | GTAO legacy | S8 | C-static | `ao_mode="GTAO"`, `ao_backend="LEGACY"` | cleaner AO | contact darkening correct |
| A3 | GTAO Hi-Z | S8 | C-static | `ao_backend="HIZ_BOUNDS"` | ≈ legacy | A/B no gross difference |
| A4 | AO debug views | S1 | C-static | set `ao_debug_view` 1–4 (F2, GTAO+HIZ_BOUNDS) | debug images | each mode distinct |

### 3.7 GI

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| GI1 | SSGI legacy | S1 | C-static | `gi_mode="SSGI"`, `ssgi_backend="LEGACY"` | color bleed | red/blue bleed visible |
| GI2 | SSGI Hi-Z | S1 | C-static | `ssgi_backend="HIZ_BOUNDS"` | ≈ legacy | A/B no gross difference |
| GI3 | SSGI debug | S1 | C-static | set `ssgi_debug_view` 1–8 (F2, HIZ_BOUNDS) | debug images | each mode distinct |
| GI4 | RC legacy | S1 | C-static | `gi_mode="RADIANCE_CASCADES"`, `rc_backend="LEGACY"` | indirect diffuse | no light leaks through walls |
| GI5 | RC cascades | S1 | C-static | `rc_backend="CASCADES"` | smooth irradiance | contact detail correct |
| GI6 | RC debug | S1 | C-static | set `rc_debug_view` (F2, CASCADES) | debug images | each mode distinct |

### 3.8 Reflections

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| R1 | SSR legacy | S5 | C-graze | `ssr_backend="LEGACY"` | reflections | tracks geometry, no edge smear |
| R2 | SSR Hi-Z | S5 | C-graze | `ssr_backend="HIZ_BOUNDS"` | ≈ legacy | A/B no gross difference |
| R3 | SSR march mode | S5 | C-graze | `ssr_mode="LINEAR"` vs `"HIZ"` (F2 "SSR March Mode") | similar result | Hi-Z faster, similar image |
| R4 | SSR debug | S5 | C-graze | set `ssr_debug_view` (F2, HIZ_BOUNDS only) | debug images | each mode distinct |

### 3.9 Sky / Atmosphere

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| SK1 | Hillaire sky | S6 | C-look-up | render | blue zenith, warm horizon | plausible at 3 times of day |
| SK2 | Space mode | S6 | C-look-up | preset `SPACE` | stars, no gradient | stars visible |
| SK3 | Clouds | S6 | C-look-up | toggle `cloud_enabled` | drifting clouds | drift over 10 s, occlude sun |
| SK4 | Presets | S6 | C-static | cycle 6 presets | distinct moods | each visibly different |
| SK5 | Flat fallback | S2 | C-static | `atmosphere_enabled=False` | uniform sky | sky is one color |

### 3.10 Volumetrics

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| V1 | Froxel fog | S6 | C-static | toggle `volumetric_fog_enabled` | haze + shafts | haze increases with distance |
| V2 | Fog resolution | S6 | C-static | cycle `fog_resolution` LOW→CINEMATIC | less banding | banding visibly reduced |
| V3 | Fog debug | S6 | C-static | cycle `fog_debug_mode` 0–2 | isolated terms | each mode isolates its term |

### 3.11 Transparency / Water

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| T1 | Particle blend | S7 | C-static | cycle `particle_mode` | correct blend | additive glows; alpha no hard clip |
| T2 | Water refraction | S9 | C-static | render | refraction | distortion visible, foam at contact |
| T3 | Water waves | S9 | C-static | observe 10 s | animated waves | smooth animation, no z-fight |

### 3.12 Terrain

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| TR1 | Grid terrain | S6 | C-static | render | flat ground | ground renders |
| TR2 | Heightmap terrain | — | — | — | absent | confirm absent (documentation) |

### 3.13 Particles

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| P1 | Compute sim | S7 | C-static | render | animated particles | drift with turbulence |
| P2 | DUST_MOTES | S7 | C-static | `particle_mode="DUST_MOTES"` | drifting motes | visible, slow drift |
| P3 | EMBERS | S7 | C-static | `particle_mode="EMBERS"` | rising embers | rise, additive glow |
| P4 | FIREFLIES | S7 | C-static | `particle_mode="FIREFLIES"` | drifting points | visible |

### 3.14 Anti-Aliasing

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| AA1 | TAA | S3 | C-static then C-pan-slow | `aa_mode="TAA"` | stable edges | no shimmer static; no ghost after pan |
| AA2 | FXAA | S3 | C-static | `aa_mode="FXAA"` | softened edges | smoother, no gross blur |
| AA3 | SMAA | S3 | C-static | `aa_mode="SMAA_1X"` | smooth + sharp | detail preserved |
| AA4 | No AA | S3 | C-static | `aa_mode="OFF"` | aliased | baseline |
| AA5 | AA exclusivity | S3 | C-static | set `aa_mode="FXAA"` then force `taa_enabled=True` | only FXAA runs | `feature_state()["aa"]["active_stages"] == ("FXAA",)` |

### 3.15 Resolution Scaling

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| RS1 | AO scale | S8 | C-static | `ao_resolution_scale` 1.0→0.5 | softer AO | visible softening, no blockiness |
| RS2 | SSGI scale | S1 | C-static | `ssgi_resolution_scale` 1.0→0.5 | softer GI | visible softening |
| RS3 | RC scale | S1 | C-static | `rc_resolution_scale` 1.0→0.5 | softer GI | visible softening |
| RS4 | SSR scale | S5 | C-graze | `ssr_resolution_scale` 1.0→0.5 | softer reflections | visible softening |
| RS5 | Flare scale | S6 | C-static | `lens_flare_resolution_scale` 1.0→0.5 | softer flare | visible softening |

### 3.16 Tone Mapping

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| TM1 | ACES | S6 | C-static | `tonemap_mode="ACES"` | filmic | highlight roll-off |
| TM2 | AgX | S6 | C-static | `tonemap_mode="AgX"` | desaturating highlights | distinct from ACES |
| TM3 | Neutral | S6 | C-static | `tonemap_mode="Neutral"` | preserving | distinct |
| TM4 | Reinhard | S6 | C-static | `tonemap_mode="Reinhard"` | flat | distinct, clips |

### 3.17 Exposure

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| E1 | Manual | S6 | C-static | `auto_exposure_enabled=False` | constant brightness | stable |
| E2 | Auto | S6 | C-static | `auto_exposure_enabled=True`, sweep time_of_day | adaptation | smooth, no pumping |

### 3.18 Bloom

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| B1 | Bloom | S7 | C-static | toggle `bloom_enabled` | glow on emissive | bloom only on highlights |

### 3.19 Color Grading

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| CG1 | White balance | S6 | C-static | toggle `white_balance_enabled` (F2 Diagnostics) | warm/cool shift | visible shift |
| CG2 | Contrast | S6 | C-static | `contrast` 1.0→1.5 | punchier | visible |
| CG3 | Saturation | S6 | C-static | `saturation` 1.0→0.0 | greyscale | visible |
| CG4 | CDL | S6 | C-static | `lift`/`gamma`/`gain` | graded | visible |
| CG5 | Chromatic aberration | S6 | C-static | toggle | edge fringing | visible at edges |
| CG6 | Vignette | S6 | C-static | toggle | darkened corners | visible |
| CG7 | Film grain | S6 | C-static | toggle | noise | visible |

### 3.20 Depth of Field

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| D1 | DoF circular | S5 | C-static | `dof_enabled=True`, `dof_bokeh_shape="CIRCULAR"` | round bokeh | round highlights |
| D2 | DoF hexagonal | S5 | C-static | `"HEXAGONAL"` | hex bokeh | hex highlights |
| D3 | DoF anamorphic | S5 | C-static | `"ANAMORPHIC"` | oval bokeh | oval highlights |

### 3.21 Motion Blur

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| MB1 | Motion blur | S4 | C-pan-fast | toggle `motion_blur_enabled` | directional blur | blur along motion; static sharp |

### 3.22 Debug Visualization

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| D1 | 24 debug modes | S1–S9 | C-static | cycle `debug_gbuffer` 0–23 | each buffer | 24 distinct plausible images |
| D2 | Debug wireframes | S1 | C-static | `pipeline.debug.line(...)` | gizmos | visible |
| D3 | F1/F2/F3 | any | — | press keys | panels | toggle correctly |

### 3.23 Temporal

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| TP1 | TAA history | S4 | C-disocclude | `aa_mode="TAA"` | no ghost | clean after 5 frames |
| TP2 | SSGI history | S1 | C-disocclude | `gi_mode="SSGI"` | no ghost | clean after 5 frames |
| TP3 | RC history | S1 | C-disocclude | `gi_mode="RADIANCE_CASCADES"` | no ghost | clean after 5 frames |
| TP4 | RC cascades temporal | S1 | C-disocclude | `rc_temporal_enabled=True` (F2, CASCADES) | no ghost | clean after 5 frames |
| TP5 | Motion vectors | S4 | C-pan-fast | render | correct velocity | blur direction matches |

### 3.24 Culling

| # | Feature | Scene | Camera | Procedure | Expected | Pass criteria |
|---|---|---|---|---|---|---|
| C1 | Frustum culling | S6 | C-static → rotate 180° | toggle `frustum_cull_enabled` | draw count drops | off-screen not drawn |
| C2 | Hi-Z culling | S1 | C-occluded | toggle `hiz_cull_enabled` | occluded culled | draw count drops, no popping |
| C3 | Mesh LOD | — | — | — | absent | confirm absent |

---

## 4. Performance Testability

Only features marked "perf testable" in the inventory may be used for performance claims. For each:

1. Warm up 60 frames.
2. Record 300 frames of `GpuPassProfiler.get_ms(<pass>)` and `SystemMonitor` CPU stage times.
3. Report **median** and **p95** frame time in ms, plus the specific pass ms.
4. Toggle exactly one parameter; re-measure with the same scene/camera/resolution.
5. Never use `glFinish`; the profiler already reads older frames.

| Feature | Pass to measure | Parameter to sweep |
|---|---|---|
| Tessellation | G-Buffer | `tess_max_level` 8 → 24 |
| POM | G-Buffer | `pom_max_samples` 16 → 64 |
| CSM | CSM | `shadow_resolution` 2048 → 8192 |
| PCF | Deferred resolve | `pcf_samples` 8 → 32 |
| PCSS | Deferred resolve | `pcss_samples` 12 → 32 |
| AO | AO | `ao_resolution_scale` 1.0 → 0.5 |
| SSGI | SSGI total | `ssgi_steps` 8 → 32 |
| RC | RC total | `rc_ray_count` 8 → 32 |
| SSR | SSR | `ssr_steps` 24 → 64 |
| Volumetric fog | Volumetric fog | `fog_resolution` LOW → CINEMATIC |
| Bloom | Bloom | `bloom_iterations` 3 → 6 |
| Auto-exposure | Exposure | on/off |
| Hi-Z culling | G-Buffer | `hiz_cull_enabled` on/off |
| TAA | TAA | on/off |

---

## 5. Known-Artifact Watch List

These are the artifacts most likely to appear. Record them explicitly.

| Artifact | Where to look | Likely cause |
|---|---|---|
| Shadow acne | S6, S8 | bias too low |
| Peter-panning | S6 | bias too high |
| Cascade popping | S6 C-dolly-far | split mismatch |
| AO halo | S8 silhouettes | depth discontinuity |
| GI light leak | S1 through walls | thickness too high |
| GI ghosting | S1 C-disocclude | temporal blend too high |
| TAA ghost trail | S4 C-disocclude | feedback too high |
| SSR edge smear | S5 screen edge | no fallback |
| POM swimming | S3 C-graze | UV step too coarse |
| Tessellation cracks | S3 patch borders | missing edge factor |
| Froxel banding | S6 LOW fog | grid too coarse |
| Bloom halo | S7 | threshold too low |
| DoF CoC discontinuity | S5 depth edges | no CoC smoothing |
| Motion blur ghost | S4 disocclusion | no velocity dilation |
| Specular shimmer | S3 C-pan-slow | specular AA off |

---

## 6. Files Inspected

`engine/gfx/pipeline.py`, `engine/gfx/quality_presets.py`, `engine/gfx/passes/*.py`,
`engine/modules/atmosphere/config.py`, `engine/debug/engine_tweaks.py`,
`engine/debug/debug_menu.py`, `engine/core/ecs.py`, `engine/core/water.py`,
`examples/03_water_lighting_lab.py`, `examples/06_graphics_benchmark.py`, `examples/_common.py`.

## 7. Files Changed

None. Only `docs/graphics/*.md` created.

## 8. Tests Performed

Static analysis only. No scene was executed in this phase.

## 9. Results

- 9 deterministic scenes defined, each isolating one condition.
- 10 camera conditions defined.
- **~95 verification rows** across 24 feature categories.
- 14 performance sweeps defined with explicit pass names.

## 10. Known Limitations

- Scene snippets use `add_box` from `examples/_common.py`; a standalone harness must replicate it
  (it is a thin wrapper over `ecs.create_entity` + optional physics).
- S4 requires a small module to drive kinematic motion deterministically; the snippet shows the
  intent, not the full module.
- S9 reuses an existing example rather than defining a new scene, to avoid duplicating 5 chambers.
- "PASS" here means the *stated expected behaviour* was observed, not that the feature is
  production-ready.

## 11. Recommended Next Action

Execute in this order (cheapest signal first):

1. **S2 + S6** — validates PBR, tonemap, exposure, sky, shadows. Catches gross regressions.
2. **S1 + S8** — validates AO and GI, the most artifact-prone systems.
3. **S3 + S5** — validates displacement, AA, reflections, DoF.
4. **S4 + S7 + S9** — validates temporal, particles, water.

Record results in [feature_status_checklist.md](./feature_status_checklist.md).
