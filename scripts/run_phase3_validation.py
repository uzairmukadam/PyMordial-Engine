"""Interactive 3D Kinematic Character Controller Validation Demo for PyMordial Engine Phase 3.

Showcases:
- Native Rust Rapier3D KinematicCharacterController (KCC) with shape sweeps and CCD
- Camera-relative omnidirectional movement (walk & sprint)
- Variable vertical jump physics and gravity
- Auto-stepping over obstacles (up to 0.3m)
- Slope limit climb checking (walkable <= 45°, blocked/sliding > 45°)
- 3rd-person follow/orbit camera with obstacle raycast occlusion clipping
- PBR deferred rendering of character capsule (1.8m x 0.4m) via Multi-Draw Indirect
- Dynamic physics spheres interaction
- Sleek on-screen telemetry HUD with real-time character motor metrics
"""

from __future__ import annotations
import sys
import os
import argparse
import math
import time
import random
from pathlib import Path
import pygame

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.core.ecs import EntityManager  # noqa: E402
from engine.core.loop import EngineLoop  # noqa: E402
from engine.physics.rapier_world import PhysicsManager  # noqa: E402
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig  # noqa: E402
from engine.core.character_camera import CharacterCamera, CharacterCameraConfig  # noqa: E402
from engine.gfx import (  # noqa: E402
    RenderContext,
    RenderPipeline,
    GraphicsQuality,
    get_quality_preset,
    HudOverlay,
)


def make_quat_rot_x(angle_rad: float) -> tuple[float, float, float, float]:
    """Computes unit quaternion for rotation around local X axis."""
    half = angle_rad * 0.5
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def spawn_static_box(
    ecs: EntityManager,
    physics: PhysicsManager,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    roughness: float = 0.5,
    metallic: float = 0.1,
) -> int:
    """Spawns an aligned static box entity where visual scale exactly matches physical box collider."""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    ent = ecs.create_entity(
        position=position,
        rotation=rotation,
        scale=size,
        color=color,
        roughness=roughness,
        metallic=metallic,
    )
    physics.create_body(ent, body_type="fixed", position=position, rotation=rotation)
    physics.attach_box_collider(ent, half_x=hx, half_y=hy, half_z=hz)
    return ent


def build_arena(
    ecs: EntityManager,
    physics: PhysicsManager,
) -> tuple[list[int], int, list[int]]:
    """Builds the Phase 3 validation obstacle course arena in ECS & Rapier3D.

    Returns:
        (static_cube_ids, character_id, dynamic_sphere_ids)
    """
    cube_ids: list[int] = []

    # 1. Main Ground Platform: top surface flush at y = 0.0, size 50m x 1m x 50m
    ground_id = spawn_static_box(
        ecs,
        physics,
        position=(0.0, -0.5, 0.0),
        size=(50.0, 1.0, 50.0),
        color=(0.20, 0.22, 0.26),
        roughness=0.6,
        metallic=0.1,
    )
    cube_ids.append(ground_id)

    # 2. Stepping Course (Auto-Stepping validation, step height <= 0.3m)
    # Step 1: 0.15m height (y: 0.0 -> 0.15)
    s1 = spawn_static_box(
        ecs,
        physics,
        position=(-6.0, 0.075, -5.0),
        size=(3.0, 0.15, 1.5),
        color=(0.60, 0.55, 0.50),
        roughness=0.4,
        metallic=0.0,
    )
    cube_ids.append(s1)

    # Step 2: 0.30m total height (y: 0.0 -> 0.30, rise 0.15m from step 1)
    s2 = spawn_static_box(
        ecs,
        physics,
        position=(-6.0, 0.15, -7.0),
        size=(3.0, 0.30, 1.5),
        color=(0.65, 0.60, 0.55),
        roughness=0.4,
        metallic=0.0,
    )
    cube_ids.append(s2)

    # Step 3: 0.45m total height (y: 0.0 -> 0.45, rise 0.15m from step 2)
    s3 = spawn_static_box(
        ecs,
        physics,
        position=(-6.0, 0.225, -9.0),
        size=(3.0, 0.45, 1.5),
        color=(0.70, 0.65, 0.60),
        roughness=0.4,
        metallic=0.0,
    )
    cube_ids.append(s3)

    # Tall Barrier (1.2m high - requires jump to clear)
    s4 = spawn_static_box(
        ecs,
        physics,
        position=(-6.0, 0.60, -11.5),
        size=(3.0, 1.20, 1.2),
        color=(0.75, 0.40, 0.20),
        roughness=0.3,
        metallic=0.2,
    )
    cube_ids.append(s4)

    # 3. Slope Limit Ramps
    # Ramp A: 20° gentle incline (Walkable, <= 45°)
    q_ramp_a = make_quat_rot_x(math.radians(-20.0))
    ra = spawn_static_box(
        ecs,
        physics,
        position=(6.0, 0.85, -6.0),
        size=(3.0, 0.2, 6.0),
        rotation=q_ramp_a,
        color=(0.25, 0.65, 0.40),  # Green
        roughness=0.4,
        metallic=0.1,
    )
    cube_ids.append(ra)

    # Ramp B: 52° steep incline (Exceeds 45° slope limit - Blocked/Sliding)
    q_ramp_b = make_quat_rot_x(math.radians(-52.0))
    rb = spawn_static_box(
        ecs,
        physics,
        position=(11.0, 1.8, -6.0),
        size=(3.0, 0.2, 6.0),
        rotation=q_ramp_b,
        color=(0.75, 0.20, 0.25),  # Red
        roughness=0.3,
        metallic=0.2,
    )
    cube_ids.append(rb)

    # 4. Obstacle Columns / Walls (Camera Raycast Clipping & Collision blocking)
    col1 = spawn_static_box(
        ecs,
        physics,
        position=(-4.0, 2.0, 4.0),
        size=(1.2, 4.0, 1.2),
        color=(0.40, 0.45, 0.55),
        roughness=0.2,
        metallic=0.7,
    )
    cube_ids.append(col1)

    col2 = spawn_static_box(
        ecs,
        physics,
        position=(4.0, 2.0, 4.0),
        size=(1.2, 4.0, 1.2),
        color=(0.40, 0.45, 0.55),
        roughness=0.2,
        metallic=0.7,
    )
    cube_ids.append(col2)

    wall = spawn_static_box(
        ecs,
        physics,
        position=(0.0, 2.0, 8.0),
        size=(10.0, 4.0, 0.5),
        color=(0.30, 0.35, 0.45),
        roughness=0.5,
        metallic=0.1,
    )
    cube_ids.append(wall)

    # 5. Player Character Entity (Rendered with capsule mesh primitive)
    char_start_pos = (0.0, 0.92, 0.0)
    char_id = ecs.create_entity(
        position=char_start_pos,
        scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.75, 0.85),  # Cyan hero suit
        roughness=0.25,
        metallic=0.85,
    )

    # 6. Dynamic PBR Physics Spheres
    sphere_ids: list[int] = []
    SPHERE_PALETTE = [
        ((0.95, 0.75, 0.25), 0.15, 0.90),  # Gold
        ((0.90, 0.92, 0.95), 0.20, 0.90),  # Chrome
        ((0.85, 0.20, 0.15), 0.35, 0.05),  # Ruby
        ((0.15, 0.55, 0.85), 0.20, 0.10),  # Cobalt
        ((0.75, 0.30, 0.85), 0.30, 0.50),  # Violet
        ((0.90, 0.50, 0.20), 0.40, 0.40),  # Copper
    ]

    for i in range(16):
        px = random.uniform(-3.0, 3.0)
        py = random.uniform(1.0, 4.0)
        pz = random.uniform(-4.0, -1.0)
        color, roughness, metallic = random.choice(SPHERE_PALETTE)
        radius = 0.45
        sph = ecs.create_entity(
            position=(px, py, pz),
            scale=(radius, radius, radius),
            color=color,
            roughness=roughness,
            metallic=metallic,
        )
        physics.create_body(sph, body_type="dynamic", position=(px, py, pz))
        physics.attach_sphere_collider(sph, radius=radius)
        sphere_ids.append(sph)

    return cube_ids, char_id, sphere_ids


def main():
    parser = argparse.ArgumentParser(description="PyMordial Engine Phase 3 KCC Validation")
    parser.add_argument("--headless", action="store_true", help="Run headlessly without displaying window")
    parser.add_argument("--frames", type=int, default=0, help="Run for N frames then exit (0 = interactive)")
    parser.add_argument("--screenshot", type=str, default="", help="Path to save screenshot after execution")
    args = parser.parse_args()

    width, height = 1280, 720
    is_headless = args.headless or (args.frames > 0 and not os.environ.get("DISPLAY") and sys.platform != "win32")

    if is_headless:
        render_ctx = RenderContext.create_headless(width=width, height=height)
    else:
        render_ctx = RenderContext(
            width=width,
            height=height,
            title="PyMordial Engine — Phase 3 Kinematic Character Controller",
            hidden=False,
        )

    pipeline = RenderPipeline(render_ctx)
    ecs = EntityManager(max_entities=5_000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
    loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)

    # Build obstacle arena
    cube_ids, char_id, sphere_ids = build_arena(ecs, physics)

    # Setup Character Motor
    motor_cfg = CharacterMotorConfig(
        walk_speed=5.0,
        run_speed=10.0,
        jump_force=8.5,
        gravity=-22.0,
        step_height=0.30,
        max_slope_deg=45.0,
    )
    motor = CharacterMotor(
        entity_id=char_id,
        physics=physics,
        ecs=ecs,
        config=motor_cfg,
        initial_position=(0.0, 0.92, 0.0),
    )

    # Setup Follow / Orbit Camera with obstacle collision clipping
    cam_cfg = CharacterCameraConfig(
        distance=6.0,
        min_distance=1.2,
        max_distance=15.0,
        target_offset_y=1.40,  # Focus on character head height
        camera_radius=0.25,
        zoom_speed=18.0,
    )
    cam = CharacterCamera(
        config=cam_cfg,
        initial_yaw_deg=0.0,
        initial_pitch_deg=18.0,
    )

    # Custom HUD Shortcuts
    hud_shortcuts = [
        ("[W, A, S, D]", "Move Character (Cam-Relative)"),
        ("[L-SHIFT]", "Sprint (Hold / Toggle)"),
        ("[SPACE]", "Jump (Variable Physics)"),
        ("[R-Drag / L-Drag]", "Orbit 3rd-Person Camera"),
        ("[Mouse Wheel]", "Zoom Camera Distance"),
        ("[1 .. 5]", "Quality Presets (Low -> Cine)"),
        ("[T]", "Toggle Tonemap Mode"),
        ("[L]", "Rotate Sun Direction"),
        ("[R]", "Reset Player to Spawn"),
        ("[K]", "Spawn 4 Dynamic Spheres"),
        ("[ESC]", "Exit Engine"),
    ]
    hud = HudOverlay(
        render_ctx.ctx,
        screen_width=width,
        screen_height=height,
        panel_width=380,
        panel_height=440,
        title="PYMORDIAL ENGINE — PHASE 3 KCC",
        shortcuts=hud_shortcuts,
    )

    # Cached state
    is_sprinting = False
    sun_angle = 0.0
    current_preset = GraphicsQuality.HIGH
    status_message = "Phase 3 KCC Active: Auto-step <=0.3m, Slope <=45°"
    status_time = time.perf_counter()

    # Input state
    move_fwd = 0.0
    move_side = 0.0
    jump_requested = False

    # Fixed physics update callback
    def fixed_update(dt: float):
        nonlocal jump_requested
        # Update Character Motor using current camera yaw
        motor.update(
            dt=dt,
            move_input=(move_fwd, move_side),
            is_running=is_sprinting,
            jump_requested=jump_requested,
            camera_yaw_deg=cam.yaw_deg,
        )
        jump_requested = False  # Reset one-shot trigger

        # Step Rapier3D physics simulation
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    loop.on_fixed_update = fixed_update

    clock = pygame.time.Clock()
    running = True
    frame_idx = 0
    max_frames = args.frames

    if not is_headless:
        pygame.mouse.get_rel()

    while running:
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        raw_dt_ms = clock.tick(0) if not is_headless else 16.67
        dt = raw_dt_ms * 0.001
        dt = min(0.05, max(0.001, dt))

        # Event handling
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        jump_requested = True
                    elif event.key == pygame.K_r:
                        # Reset character position
                        motor.set_position((0.0, 0.92, 0.0))
                        status_message = "> Reset Character to Spawn (0, 0.92, 0)"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_k:
                        # Spawn dynamic spheres
                        for _ in range(4):
                            px = random.uniform(-2.0, 2.0)
                            py = random.uniform(3.0, 6.0)
                            pz = random.uniform(-2.0, 2.0)
                            sph = ecs.create_entity(
                                position=(px, py, pz),
                                scale=(0.45, 0.45, 0.45),
                                color=(0.95, 0.75, 0.25),
                                roughness=0.2,
                                metallic=0.8,
                            )
                            physics.create_body(sph, body_type="dynamic", position=(px, py, pz))
                            physics.attach_sphere_collider(sph, radius=0.45)
                            sphere_ids.append(sph)
                        status_message = f"> Spawned 4 Spheres (Total: {ecs.active_count})"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_1:
                        current_preset = GraphicsQuality.LOW
                        pipeline.apply_config(get_quality_preset(current_preset))
                        status_message = f"> Quality: {current_preset.value.upper()}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_2:
                        current_preset = GraphicsQuality.MEDIUM
                        pipeline.apply_config(get_quality_preset(current_preset))
                        status_message = f"> Quality: {current_preset.value.upper()}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_3:
                        current_preset = GraphicsQuality.HIGH
                        pipeline.apply_config(get_quality_preset(current_preset))
                        status_message = f"> Quality: {current_preset.value.upper()}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_4:
                        current_preset = GraphicsQuality.ULTRA
                        pipeline.apply_config(get_quality_preset(current_preset))
                        status_message = f"> Quality: {current_preset.value.upper()}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_5:
                        current_preset = GraphicsQuality.CINEMATIC
                        pipeline.apply_config(get_quality_preset(current_preset))
                        status_message = f"> Quality: {current_preset.value.upper()}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_t:
                        modes = ["ACES", "AgX", "Reinhard"]
                        curr_idx = modes.index(pipeline.config.tonemap_mode)
                        pipeline.config.tonemap_mode = modes[(curr_idx + 1) % len(modes)]
                        status_message = f"> Tonemapping: {pipeline.config.tonemap_mode}"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_l:
                        sun_angle += 0.4
                        status_message = f"> Rotated Sun to {sun_angle:.2f} rad"
                        status_time = time.perf_counter()

                elif event.type == pygame.MOUSEWHEEL:
                    cam.handle_zoom(event.y)

            # Continuous Keyboard Input
            keys = pygame.key.get_pressed()
            is_sprinting = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

            move_fwd = 0.0
            move_side = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                move_fwd += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                move_fwd -= 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                move_side += 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                move_side -= 1.0

            # Mouse Orbit Drag (Left or Right click drag)
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_rel = pygame.mouse.get_rel()
            if mouse_buttons[0] or mouse_buttons[2]:
                cam.handle_mouse_orbit(mouse_rel[0], mouse_rel[1])
        else:
            # Headless automated scripted motion for validation
            if frame_idx < 30:
                move_fwd = 1.0  # Walk forward toward steps
                is_sprinting = False
            elif frame_idx < 60:
                move_fwd = 1.0
                is_sprinting = True  # Sprint
            elif frame_idx == 60:
                jump_requested = True
            elif frame_idx < 90:
                move_fwd = 0.5
            else:
                move_fwd = 0.0

        # Step deterministic simulation (60 Hz fixed tick)
        loop.step_frame(dt)

        # Get interpolated character position for camera
        char_dense_idx = ecs.pool.get_dense_index(char_id)
        char_world_pos = ecs.rigid_body_state[1, char_dense_idx, 0:3]

        # Update collision-aware camera
        camera_pos, camera_target = cam.update(
            dt=dt,
            character_pos=char_world_pos,
            physics=physics,
        )

        # Sun direction
        sun_x = math.cos(sun_angle) * 0.6 + 0.2
        sun_z = math.sin(sun_angle) * 0.6 + 0.2
        sun_dir = (sun_x, -0.85, sun_z)

        # Setup MDI draw batches:
        # Batch 1: Static Cubes (Entities 0 .. len(cube_ids) - 1)
        # Batch 2: Character Capsule (Entity char_id)
        # Batch 3: Dynamic Spheres (Entities char_id + 1 .. active_count - 1)
        num_cubes = len(cube_ids)
        num_spheres = len(sphere_ids)
        draw_batches = [
            ("cube", num_cubes, 0),
            ("capsule", 1, num_cubes),
            ("sphere", num_spheres, num_cubes + 1),
        ]

        # Render ModernGL 4.5 Core 3D frame
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=camera_pos,
            camera_target=camera_target,
            time_elapsed=loop.elapsed_time,
            sun_dir=sun_dir,
            sun_lux=pipeline.config.sun_intensity,
            draw_batches=draw_batches,
        )

        # Format Telemetry Lines for Character Motor
        mstate = motor.state
        state_str = "GROUNDED" if mstate.is_grounded else ("JUMPING" if mstate.is_jumping else "IN AIR")
        state_color = (74, 222, 128) if mstate.is_grounded else (250, 204, 21)
        sprint_str = "SPRINT" if is_sprinting else "WALK"
        clip_str = "CLIPPED" if cam.is_clipped else "FREE"
        clip_color = (248, 113, 113) if cam.is_clipped else (148, 163, 184)

        extra_lines = [
            (
                f"KCC: Pos ({char_world_pos[0]:4.1f}, {char_world_pos[1]:4.1f}, {char_world_pos[2]:4.1f})",
                (56, 189, 248),
            ),
            (
                f"State: {state_str:<9} | Mode: {sprint_str:<6} | Spd: {mstate.horizontal_speed:4.1f}m/s",
                state_color,
            ),
            (
                f"Camera: Dist {cam.current_distance:4.1f}m | Obstacle Occlusion: {clip_str}",
                clip_color,
            ),
            (
                "Auto-Step: <=0.30m ACTIVE | Max Slope: <=45° ACTIVE",
                (203, 213, 225),
            ),
        ]

        # Render sleek glassmorphic HUD
        hud.render(
            fps=clock.get_fps() if not is_headless else 60.0,
            frame_time_ms=raw_dt_ms,
            preset_name=current_preset.value,
            entity_count=ecs.active_count,
            cam_dist=cam.current_distance,
            cam_yaw=cam.yaw_deg % 360,
            cam_pitch=cam.pitch_deg,
            sun_angle=sun_angle,
            tonemap_mode=pipeline.config.tonemap_mode,
            status_message=status_message,
            status_time=status_time,
            extra_lines=extra_lines,
        )

        if not is_headless:
            render_ctx.swap_buffers()

    # If screenshot requested, save rendered final texture
    if args.screenshot:
        from PIL import Image

        raw_pixels = pipeline.post_process.final_texture.read()
        img = Image.frombytes("RGBA", (width, height), raw_pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        save_path = Path(args.screenshot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path))
        print(f"Screenshot saved to: {save_path.resolve()}")

    hud.destroy()
    pipeline.destroy()
    render_ctx.destroy()
    if not is_headless:
        pygame.quit()
    print("Phase 3 Validation script completed successfully.")


if __name__ == "__main__":
    main()
