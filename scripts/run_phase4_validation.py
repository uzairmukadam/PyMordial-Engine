"""Interactive Validation Demo for PyMordial Engine Phase 4: Offline Asset Cooker & VFS.

Showcases:
- Offline asset baking: .obj 3D models baked into 32-byte cache-aligned .pm_mesh
- Pre-filtered mipmapped .pm_tex textures
- Packaging into a contiguous .pak archive container
- Virtual File System (VFS) mounting with zero-copy memory-mapped streaming
- Direct ingestion into MegaBuffer with Multi-Draw Indirect (MDI) batch rendering
- Full kinematic character controller (Phase 3) interacting with cooked physical 3D assets
- Live telemetry HUD displaying VFS, mmap streaming, and batch performance metrics
"""

from __future__ import annotations
import sys
import argparse
import math
import time
import random
from pathlib import Path
import pygame
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.core.ecs import EntityManager  # noqa: E402
from engine.core.loop import EngineLoop  # noqa: E402
from engine.physics.rapier_world import PhysicsManager  # noqa: E402
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig  # noqa: E402
from engine.core.character_camera import CharacterCamera, CharacterCameraConfig  # noqa: E402
from engine.assets.cooker import cook_mesh, cook_texture, pack_directory  # noqa: E402
from engine.gfx import (  # noqa: E402
    RenderContext,
    RenderPipeline,
    GraphicsQuality,
    get_quality_preset,
    HudOverlay,
)
from engine.gfx.mega_buffer import MeshAllocation  # noqa: E402


def generate_monolith_obj(filepath: Path) -> None:
    """Generates an intricate, beveled 3D monolith pillar model in Wavefront .obj format."""
    obj_lines = [
        "# PyMordial Phase 4 Cooked Asset: Beveled Monolith Pillar",
        # Base vertices (Y = 0.0 to Y = 0.4)
        "v -1.0 0.0 -1.0",
        "v  1.0 0.0 -1.0",
        "v  1.0 0.0  1.0",
        "v -1.0 0.0  1.0",
        "v -0.8 0.4 -0.8",
        "v  0.8 0.4 -0.8",
        "v  0.8 0.4  0.8",
        "v -0.8 0.4  0.8",
        # Shaft vertices (Y = 0.4 to Y = 3.6, tapering slightly)
        "v -0.6 3.6 -0.6",
        "v  0.6 3.6 -0.6",
        "v  0.6 3.6  0.6",
        "v -0.6 3.6  0.6",
        # Pyramidion apex vertex (Y = 4.2)
        "v  0.0 4.2  0.0",
        # Texture coordinates
        "vt 0.0 0.0",
        "vt 1.0 0.0",
        "vt 1.0 1.0",
        "vt 0.0 1.0",
        "vt 0.5 1.0",
        # Normals
        "vn  0.0 -1.0  0.0",  # 1: Down
        "vn  0.0  1.0  0.0",  # 2: Up
        "vn  0.0  0.0 -1.0",  # 3: North
        "vn  1.0  0.0  0.0",  # 4: East
        "vn  0.0  0.0  1.0",  # 5: South
        "vn -1.0  0.0  0.0",  # 6: West
        "vn  0.0  0.6 -0.8",  # 7: North slant
        "vn  0.8  0.6  0.0",  # 8: East slant
        "vn  0.0  0.6  0.8",  # 9: South slant
        "vn -0.8  0.6  0.0",  # 10: West slant
        # Faces: Bottom (Y = 0)
        "f 1/1/1 2/2/1 3/3/1",
        "f 1/1/1 3/3/1 4/4/1",
        # Base beveled sides (North, East, South, West)
        "f 1/1/7 6/3/7 2/2/7",
        "f 1/1/7 5/4/7 6/3/7",
        "f 2/1/8 7/3/8 3/2/8",
        "f 2/1/8 6/4/8 7/3/8",
        "f 3/1/9 8/3/9 4/2/9",
        "f 3/1/9 7/4/9 8/3/9",
        "f 4/1/10 5/3/10 1/2/10",
        "f 4/1/10 8/4/10 5/3/10",
        # Main Shaft faces (North, East, South, West)
        "f 5/1/3 10/3/3 6/2/3",
        "f 5/1/3 9/4/3 10/3/3",
        "f 6/1/4 11/3/4 7/2/4",
        "f 6/1/4 10/4/4 11/3/4",
        "f 7/1/5 12/3/5 8/2/5",
        "f 7/1/5 11/4/5 12/3/5",
        "f 8/1/6 9/3/6 5/2/6",
        "f 8/1/6 12/4/6 9/3/6",
        # Pyramidion Cap
        "f 9/1/7 13/5/7 10/2/7",
        "f 10/1/8 13/5/8 11/2/8",
        "f 11/1/9 13/5/9 12/2/9",
        "f 12/1/10 13/5/10 9/2/10",
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(obj_lines), encoding="utf-8")


def generate_pattern_texture(filepath: Path) -> None:
    """Generates a procedural high-res texture with color accents for offline baking."""
    w, h = 128, 128
    img = Image.new("RGBA", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            cx = (x % 32) < 16
            cy = (y % 32) < 16
            is_check = cx ^ cy
            r = 180 if is_check else 120
            g = 140 if is_check else 90
            b = 80 if is_check else 50
            pixels[x, y] = (r, g, b, 255)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(filepath)


def bake_asset_package(build_dir: Path) -> Path:
    """Bakes raw source assets into 32-byte .pm_mesh, .pm_tex, and packages into a .pak archive."""
    src_dir = build_dir / "raw_assets"
    cooked_dir = build_dir / "cooked_assets"
    pak_path = build_dir / "level_assets.pak"

    raw_obj = src_dir / "monolith.obj"
    raw_tex = src_dir / "monolith_albedo.png"

    generate_monolith_obj(raw_obj)
    generate_pattern_texture(raw_tex)

    cooked_mesh_path = cooked_dir / "models" / "monolith.pm_mesh"
    cooked_tex_path = cooked_dir / "textures" / "monolith_albedo.pm_tex"

    # 1. Cook raw .obj -> 32-byte cache-aligned .pm_mesh
    cook_mesh(raw_obj, cooked_mesh_path)
    # 2. Cook raw .png -> mipmapped .pm_tex
    cook_texture(raw_tex, cooked_tex_path)

    # 3. Pack cooked directory into contiguous .pak archive container
    pack_directory(cooked_dir, pak_path, compress=False)
    return pak_path


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
    """Spawns a static box where visual scale exactly matches physical half-extents."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Phase 4 Validation")
    parser.add_argument("--headless", action="store_true", help="Run without window display")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames")
    parser.add_argument("--screenshot", type=str, default="", help="Save screenshot to path")
    args = parser.parse_args()

    width, height = 1280, 720
    is_headless = args.headless

    # Bake Phase 4 asset package (.pak)
    build_dir = ROOT_DIR / "build" / "phase4_assets"
    pak_path = bake_asset_package(build_dir)

    # 1. Initialize Display & Engine Core
    if is_headless:
        ctx_wrapper = RenderContext.create_headless(width=width, height=height)
    else:
        ctx_wrapper = RenderContext(
            width=width,
            height=height,
            title="PyMordial Engine — Phase 4: Cooked Assets & VFS Streaming",
            hidden=False,
        )
    pipeline = RenderPipeline(ctx_wrapper)
    ecs = EntityManager(max_entities=1000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-20.0, gravity_z=0.0)
    loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)

    # 2. Mount Cooked .pak Archive via Virtual File System (VFS)
    pipeline.resources.mount_pak(pak_path)

    # 3. Load Cooked 32-Byte .pm_mesh Directly from .pak into GPU MegaBuffer
    alloc_monolith = pipeline.load_cooked_mesh("monolith", "models/monolith.pm_mesh")
    cooked_mesh = pipeline.resources.load_mesh("models/monolith.pm_mesh")

    # 4. Construct Scene Environment
    cube_ids: list[int] = []
    # Ground Platform (50m x 1m x 50m, top surface at y = 0.0)
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

    # Stepping Obstacles
    s1 = spawn_static_box(ecs, physics, position=(-6.0, 0.075, -5.0), size=(3.0, 0.15, 1.5), color=(0.60, 0.55, 0.50))
    s2 = spawn_static_box(ecs, physics, position=(-6.0, 0.150, -7.0), size=(3.0, 0.30, 1.5), color=(0.65, 0.60, 0.55))
    s3 = spawn_static_box(ecs, physics, position=(-6.0, 0.225, -9.0), size=(3.0, 0.45, 1.5), color=(0.70, 0.65, 0.60))
    cube_ids.extend([s1, s2, s3])

    # 5. Spawn Cooked Monolith Entities in World
    monolith_ids: list[int] = []
    monolith_positions = [
        (-4.0, 0.0, 3.0),
        (4.0, 0.0, 3.0),
        (-4.0, 0.0, -3.0),
        (4.0, 0.0, -3.0),
    ]
    for pos in monolith_positions:
        ent = ecs.create_entity(
            position=pos,
            scale=(1.0, 1.0, 1.0),
            color=(0.85, 0.72, 0.45),  # Sandstone gold
            roughness=0.35,
            metallic=0.15,
        )
        # Physics collider for monolith (pillar size ~ 1.6m x 4.2m x 1.6m, center at y = 2.1)
        physics.create_body(ent, body_type="fixed", position=(pos[0], pos[1] + 2.1, pos[2]))
        physics.attach_box_collider(ent, half_x=0.8, half_y=2.1, half_z=0.8)
        monolith_ids.append(ent)

    # 6. Spawn Phase 3 Kinematic Character (Player Capsule)
    char_id = ecs.create_entity(
        position=(0.0, 0.92, 0.0),
        scale=(1.0, 1.0, 1.0),
        color=(0.25, 0.70, 0.95),  # Cyan
        roughness=0.25,
        metallic=0.20,
    )
    motor = CharacterMotor(
        entity_id=char_id,
        physics=physics,
        ecs=ecs,
        config=CharacterMotorConfig(
            walk_speed=6.0,
            run_speed=11.0,
            jump_force=8.0,
            gravity=-20.0,
            step_height=0.30,
        ),
        initial_position=(0.0, 0.92, 0.0),
    )

    # 7. Spawn Dynamic Physics Spheres
    sphere_ids: list[int] = []
    for _ in range(8):
        px = random.uniform(-3.0, 3.0)
        py = random.uniform(3.0, 6.0)
        pz = random.uniform(-3.0, 3.0)
        sph = ecs.create_entity(
            position=(px, py, pz),
            scale=(0.45, 0.45, 0.45),
            color=(0.95, 0.45, 0.25),
            roughness=0.2,
            metallic=0.8,
        )
        physics.create_body(sph, body_type="dynamic", position=(px, py, pz))
        physics.attach_sphere_collider(sph, radius=0.45)
        sphere_ids.append(sph)

    # Camera & Telemetry HUD
    cam = CharacterCamera(
        config=CharacterCameraConfig(
            distance=7.5,
            target_offset_y=1.35,
            camera_radius=0.25,
        ),
        initial_yaw_deg=45.0,
        initial_pitch_deg=22.0,
    )
    hud = HudOverlay(ctx_wrapper.ctx, ctx_wrapper.width, ctx_wrapper.height)

    # Pre-cache MeshAllocation objects for MDI batch rendering
    alloc_cube = pipeline.mega_buffer.allocations["cube"]
    alloc_capsule = pipeline.mega_buffer.allocations["capsule"]
    alloc_sphere = pipeline.mega_buffer.allocations["sphere"]

    num_cubes = len(cube_ids)
    num_monoliths = len(monolith_ids)

    # Zero-allocation MDI batch specification
    # Batch 1: Static Cubes / Floor (Entities 0 .. num_cubes - 1)
    # Batch 2: Cooked 32-Byte Monoliths (Entities num_cubes .. num_cubes + num_monoliths - 1)
    # Batch 3: Character Capsule (Entity char_id)
    # Batch 4: Dynamic Spheres (Entities char_id + 1 .. active_count - 1)
    draw_batches: list[tuple[MeshAllocation, int, int]] = [
        (alloc_cube, num_cubes, 0),
        (alloc_monolith, num_monoliths, num_cubes),
        (alloc_capsule, 1, num_cubes + num_monoliths),
        (alloc_sphere, len(sphere_ids), num_cubes + num_monoliths + 1),
    ]

    # Simulation State
    move_fwd = 0.0
    move_strafe = 0.0
    is_sprinting = False
    jump_requested = False
    sun_angle = 0.8
    current_preset = GraphicsQuality.HIGH

    status_message = f"> VFS Mounted '{pak_path.name}' (mmap zero-copy)"
    status_time = time.perf_counter()

    def fixed_update(dt: float) -> None:
        nonlocal jump_requested
        motor.update(
            dt=dt,
            move_input=(move_fwd, move_strafe),
            is_running=is_sprinting,
            jump_requested=jump_requested,
            camera_yaw_deg=cam.yaw_deg,
        )
        jump_requested = False
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
                        motor.set_position((0.0, 0.92, 0.0))
                        status_message = "> Reset Character to Spawn (0, 0.92, 0)"
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
                        status_message = f"> Rotated Sun Angle ({sun_angle:0.1f} rad)"
                        status_time = time.perf_counter()
                    elif event.key == pygame.K_k:
                        for _ in range(4):
                            px = random.uniform(-3.0, 3.0)
                            py = random.uniform(3.0, 6.0)
                            pz = random.uniform(-3.0, 3.0)
                            sph = ecs.create_entity(
                                position=(px, py, pz),
                                scale=(0.45, 0.45, 0.45),
                                color=(0.95, 0.45, 0.25),
                                roughness=0.2,
                                metallic=0.8,
                            )
                            physics.create_body(sph, body_type="dynamic", position=(px, py, pz))
                            physics.attach_sphere_collider(sph, radius=0.45)
                            sphere_ids.append(sph)
                        status_message = f"> Spawned 4 Spheres (Total: {ecs.active_count})"
                        status_time = time.perf_counter()

                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[2] or pygame.mouse.get_pressed()[0]:
                        cam.handle_mouse_orbit(event.rel[0], event.rel[1])
                elif event.type == pygame.MOUSEWHEEL:
                    cam.handle_zoom(event.y)

            keys = pygame.key.get_pressed()
            move_fwd = (1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0) - (
                1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
            )
            move_strafe = (1.0 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0.0) - (
                1.0 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0.0
            )
            is_sprinting = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        else:
            # Scripted headless simulation motion
            if frame_idx < 30:
                move_fwd = 1.0
                is_sprinting = False
            elif frame_idx < 60:
                move_fwd = 1.0
                is_sprinting = True
            elif frame_idx == 60:
                jump_requested = True
            else:
                move_fwd = 0.0

        # Step deterministic simulation (60 Hz fixed tick)
        loop.step_frame(dt)

        # Get interpolated character position for camera
        char_dense_idx = ecs.pool.get_dense_index(char_id)
        char_world_pos = ecs.rigid_body_state[1, char_dense_idx, 0:3]

        camera_pos, camera_target = cam.update(
            dt=dt,
            character_pos=char_world_pos,
            physics=physics,
        )

        # Directional Sunlight
        sun_x = math.cos(sun_angle) * 0.6 + 0.2
        sun_z = math.sin(sun_angle) * 0.6 + 0.2
        sun_dir = (sun_x, -0.85, sun_z)

        # Update dynamic sphere batch count if needed
        num_spheres = len(sphere_ids)
        if draw_batches[3][1] != num_spheres:
            draw_batches[3] = (alloc_sphere, num_spheres, num_cubes + num_monoliths + 1)

        # Render ModernGL 4.5 Core 3D frame via MDI
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=camera_pos,
            camera_target=camera_target,
            time_elapsed=loop.elapsed_time,
            sun_dir=sun_dir,
            sun_lux=pipeline.config.sun_intensity,
            draw_batches=draw_batches,
        )

        # Format Telemetry Lines for Character Motor & VFS
        mstate = motor.state
        state_str = "GROUNDED" if mstate.is_grounded else ("JUMPING" if mstate.is_jumping else "IN AIR")
        state_color = (74, 222, 128) if mstate.is_grounded else (250, 204, 21)
        sprint_str = "SPRINT" if is_sprinting else "WALK"

        extra_lines = [
            (
                f"VFS: '{pak_path.name}' [mmap zero-copy active]",
                (56, 189, 248),
            ),
            (
                f"Cooked Mesh: 'monolith.pm_mesh' ({cooked_mesh.vertex_count} verts, 32B cache stride)",
                (251, 146, 60),
            ),
            (
                f"Bounds: Sphere Radius {cooked_mesh.bounds.sphere_radius:0.2f}m | MDI Batches: {len(draw_batches)}",
                (234, 179, 8),
            ),
            (
                f"KCC: Pos ({char_world_pos[0]:4.1f}, {char_world_pos[1]:4.1f}, {char_world_pos[2]:4.1f}) | {state_str} ({sprint_str})",
                state_color,
            ),
            (
                f"Entities: {ecs.active_count} | Cam Dist: {cam.current_distance:4.1f}m",
                (148, 163, 184),
            ),
        ]

        # Draw Telemetry HUD
        hud.render(
            fps=1.0 / dt if dt > 0.0 else 60.0,
            frame_time_ms=dt * 1000.0,
            preset_name=current_preset.value,
            entity_count=ecs.active_count,
            cam_dist=cam.current_distance,
            cam_yaw=cam.yaw_deg % 360,
            cam_pitch=cam.pitch_deg,
            sun_angle=sun_angle,
            tonemap_mode=pipeline.config.tonemap_mode,
            status_message=status_message if (time.perf_counter() - status_time < 3.0) else "",
            status_time=status_time,
            extra_lines=extra_lines,
        )

        if not is_headless:
            ctx_wrapper.swap_buffers()

    if args.screenshot:
        raw_pixels = pipeline.post_process.final_texture.read()
        img = Image.frombytes("RGBA", (width, height), raw_pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        save_path = Path(args.screenshot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path))
        print(f"Saved Phase 4 screenshot to: {save_path.resolve()}")

    hud.destroy()
    pipeline.resources.close()
    pipeline.destroy()
    ctx_wrapper.destroy()
    if not is_headless:
        pygame.quit()
    print("Phase 4 Validation script completed successfully.")


if __name__ == "__main__":
    main()
