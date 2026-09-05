"""Interactive 3D Graphics Validation Demo for PyMordial Engine Phase 2.

Showcases:
- ModernGL 4.5 Core Profile 3D rendering on RTX GPU
- Multi-Draw Indirect (MDI) geometry submission from shared Mega-Buffer
- Reversed-Z 32F floating-point depth buffer (zero Z-fighting)
- Cook-Torrance GGX PBR BRDF (Roughness, Metallic, Fresnel)
- 4-Cascade Cascaded Shadow Maps (CSM) with 16-tap Poisson PCF
- Screen-Space Contact Shadows (SSCS) under objects (<10m)
- Real-time Graphics Quality Presets (1=Low, 2=Med, 3=High, 4=Ultra, 5=Cinematic)
- ACES / AgX Filmic tonemapping
- Native Rust Rapier3D physics integrated with 3D MDI rendering
"""

import sys
from pathlib import Path

import math
import time
import random
import pygame

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.core.ecs import EntityManager  # noqa: E402
from engine.core.loop import EngineLoop  # noqa: E402
from engine.physics.rapier_world import PhysicsManager  # noqa: E402
from engine.gfx import (  # noqa: E402
    RenderContext,
    RenderPipeline,
    GraphicsQuality,
    get_quality_preset,
    HudOverlay,
)


def main():
    width, height = 1280, 720
    render_ctx = RenderContext(
        width=width,
        height=height,
        title="PyMordial Engine — Phase 2 3D PBR Validation",
        hidden=False,
    )
    pipeline = RenderPipeline(render_ctx)
    hud = HudOverlay(render_ctx.ctx, width, height)

    ecs = EntityManager(max_entities=20_000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
    loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)

    # 1. Ground Platform Box (Entity 0)
    ground_id = ecs.create_entity(
        position=(0.0, -1.0, 0.0),
        scale=(30.0, 0.5, 30.0),
        color=(0.35, 0.40, 0.45),
        roughness=0.3,
        metallic=0.1,
    )
    physics.create_body(ground_id, body_type="fixed", position=(0.0, -1.0, 0.0))
    physics.attach_box_collider(ground_id, half_x=15.0, half_y=0.25, half_z=15.0)

    # Palette of rich PBR material colors
    PALETTE = [
        ((0.95, 0.75, 0.25), 0.15, 0.90),  # Polished Gold
        ((0.90, 0.92, 0.95), 0.20, 0.90),  # Silver / Chrome
        ((0.85, 0.20, 0.15), 0.35, 0.05),  # Ruby Plastic
        ((0.15, 0.55, 0.85), 0.20, 0.10),  # Cobalt Gloss
        ((0.25, 0.85, 0.45), 0.40, 0.00),  # Emerald Matte
        ((0.75, 0.30, 0.85), 0.30, 0.50),  # Metallic Violet
        ((0.90, 0.50, 0.20), 0.40, 0.40),  # Copper
    ]

    def spawn_pbr_sphere(x=None, y=None, z=None):
        if ecs.active_count >= 1_500:
            return
        px = random.uniform(-7.0, 7.0) if x is None else x
        py = random.uniform(4.0, 12.0) if y is None else y
        pz = random.uniform(-7.0, 7.0) if z is None else z

        color, roughness, metallic = random.choice(PALETTE)
        radius = 0.5

        ent = ecs.create_entity(
            position=(px, py, pz),
            scale=(radius, radius, radius),
            color=color,
            roughness=roughness,
            metallic=metallic,
        )
        physics.create_body(ent, body_type="dynamic", position=(px, py, pz))
        physics.attach_sphere_collider(ent, radius=radius)
        return ent

    # Spawn initial cluster of 30 PBR physics spheres
    for _ in range(30):
        spawn_pbr_sphere()

    # Pre-simulation snapshot
    saved_snapshot = ecs.snapshot_memory()

    # Wire physics into loop.on_fixed_update
    def fixed_update(dt: float):
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    loop.on_fixed_update = fixed_update

    # Camera Orbit state
    cam_dist = 18.0
    cam_yaw = 45.0
    cam_pitch = 25.0
    cam_target = (0.0, 2.0, 0.0)

    # Initial sun direction
    sun_angle = 0.0
    current_preset = GraphicsQuality.HIGH

    clock = pygame.time.Clock()
    running = True

    # Reset mouse relative delta
    pygame.mouse.get_rel()

    status_message = "Ready: PyMordial Phase 2 Active"
    status_time = time.perf_counter()

    while running:
        raw_dt_ms = clock.tick(0)  # Uncapped framerate
        dt = raw_dt_ms * 0.001

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    for _ in range(8):
                        spawn_pbr_sphere()
                    status_message = f"> Spawned 8 PBR spheres (Total: {ecs.active_count})"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_1:
                    current_preset = GraphicsQuality.LOW
                    pipeline.apply_config(get_quality_preset(current_preset))
                    status_message = f"> Preset changed to: {current_preset.value.upper()}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_2:
                    current_preset = GraphicsQuality.MEDIUM
                    pipeline.apply_config(get_quality_preset(current_preset))
                    status_message = f"> Preset changed to: {current_preset.value.upper()}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_3:
                    current_preset = GraphicsQuality.HIGH
                    pipeline.apply_config(get_quality_preset(current_preset))
                    status_message = f"> Preset changed to: {current_preset.value.upper()}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_4:
                    current_preset = GraphicsQuality.ULTRA
                    pipeline.apply_config(get_quality_preset(current_preset))
                    status_message = f"> Preset changed to: {current_preset.value.upper()}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_5:
                    current_preset = GraphicsQuality.CINEMATIC
                    pipeline.apply_config(get_quality_preset(current_preset))
                    status_message = f"> Preset changed to: {current_preset.value.upper()}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_t:
                    modes = ["ACES", "AgX", "Reinhard"]
                    curr_idx = modes.index(pipeline.config.tonemap_mode)
                    pipeline.config.tonemap_mode = modes[(curr_idx + 1) % len(modes)]
                    status_message = f"> Tonemapping set to: {pipeline.config.tonemap_mode}"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_l:
                    sun_angle += 0.5
                    status_message = f"> Rotated Sun to {sun_angle:.2f} rad"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_p:
                    saved_snapshot = ecs.snapshot_memory()
                    status_message = f"> PIE Snapshot Saved ({ecs.active_count} entities in C-RAM)"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_r:
                    for i in range(ecs.active_count):
                        ent_id = ecs.pool.get_entity_id(i)
                        if ent_id != ground_id:
                            physics.remove_body(ent_id)
                    ecs.restore_snapshot(saved_snapshot)
                    for i in range(ecs.active_count):
                        ent_id = ecs.pool.get_entity_id(i)
                        if ent_id != ground_id:
                            pos = ecs.rigid_body_state[1, i, 0:3]
                            rot = ecs.rigid_body_state[1, i, 3:7]
                            physics.create_body(ent_id, body_type="dynamic", position=pos, rotation=rot)
                            physics.attach_sphere_collider(ent_id, radius=0.5)
                    status_message = f"> PIE Restored via memcpy ({ecs.active_count} entities)"
                    status_time = time.perf_counter()
                elif event.key == pygame.K_c:
                    for i in reversed(range(ecs.active_count)):
                        ent_id = ecs.pool.get_entity_id(i)
                        if ent_id != ground_id:
                            physics.remove_body(ent_id)
                            ecs.destroy_entity(ent_id)
                    status_message = f"> Cleared spheres (Entities: {ecs.active_count})"
                    status_time = time.perf_counter()

            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = max(3.0, min(60.0, cam_dist - event.y * 1.5))

        # Direct hardware mouse drag orbit (natural grab orientation)
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_rel = pygame.mouse.get_rel()
        if mouse_pressed[0]:  # Left mouse button held
            cam_yaw -= mouse_rel[0] * 0.3
            cam_pitch = max(-85.0, min(85.0, cam_pitch - mouse_rel[1] * 0.3))

        # Keyboard Orbit Controls (Arrow keys & WASD)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            cam_yaw -= 60.0 * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            cam_yaw += 60.0 * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            cam_pitch = min(85.0, cam_pitch + 45.0 * dt)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            cam_pitch = max(-85.0, cam_pitch - 45.0 * dt)
        if keys[pygame.K_q]:
            cam_dist = max(3.0, cam_dist - 15.0 * dt)
        if keys[pygame.K_e]:
            cam_dist = min(60.0, cam_dist + 15.0 * dt)

        # Compute camera position from yaw/pitch orbit
        rad_yaw = math.radians(cam_yaw)
        rad_pitch = math.radians(cam_pitch)
        cx = cam_target[0] + cam_dist * math.cos(rad_pitch) * math.sin(rad_yaw)
        cy = cam_target[1] + cam_dist * math.sin(rad_pitch)
        cz = cam_target[2] + cam_dist * math.cos(rad_pitch) * math.cos(rad_yaw)
        camera_pos = (cx, cy, cz)

        # Sun direction vector
        sun_x = math.cos(sun_angle) * 0.6 + 0.2
        sun_z = math.sin(sun_angle) * 0.6 + 0.2
        sun_dir = (sun_x, -0.85, sun_z)

        # Step deterministic simulation (60 Hz fixed tick)
        loop.step_frame(dt)

        # Execute Phase 2 ModernGL 4.5 Core 3D render pipeline
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=camera_pos,
            camera_target=cam_target,
            time_elapsed=loop.elapsed_time,
            sun_dir=sun_dir,
            sun_lux=pipeline.config.sun_intensity,
        )

        # Render sleek on-screen HUD telemetry & controls overlay
        hud.render(
            fps=clock.get_fps(),
            frame_time_ms=raw_dt_ms,
            preset_name=current_preset.value,
            entity_count=ecs.active_count,
            cam_dist=cam_dist,
            cam_yaw=cam_yaw % 360,
            cam_pitch=cam_pitch,
            sun_angle=sun_angle,
            tonemap_mode=pipeline.config.tonemap_mode,
            status_message=status_message,
            status_time=status_time,
        )

        render_ctx.swap_buffers()

    hud.destroy()
    pipeline.destroy()
    render_ctx.destroy()
    pygame.quit()
    print("Phase 2 Demo exited cleanly.")


if __name__ == "__main__":
    main()
