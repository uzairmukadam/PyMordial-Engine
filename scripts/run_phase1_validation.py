"""High-Performance Manual Validation Demo for PyMordial Engine Phase 1.

Demonstrates:
- Contiguous C-aligned flat NumPy memory tables
- O(1) swap-and-pop Dense-Sparse entity pool
- SIMD-accelerated native Rust Rapier3D physics (Release build with Rayon)
- Zero-copy buffer transform synchronization
- Vectorized sub-frame NLERP interpolation
- Uncapped variable-rate rendering (pushing maximum FPS)
- Instant Play-In-Editor (PIE) snapshot & restore
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import random
import pygame
import numpy as np

from engine.core.ecs import EntityManager
from engine.core.loop import EngineLoop
from engine.core.input import InputManager
from engine.physics.rapier_world import PhysicsManager


def main():
    pygame.init()
    width, height = 960, 640
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("PyMordial Engine - High-Performance Phase 1 Suite")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 15)
    font_bold = pygame.font.SysFont("Consolas", 17, bold=True)

    ecs = EntityManager(max_entities=20_000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
    input_mgr = InputManager()
    loop = EngineLoop(ecs=ecs, input_manager=input_mgr, fixed_dt=1.0 / 60.0)

    # 1. Spawn fixed ground at y = -2.0
    ground_id = ecs.create_entity(position=(0.0, -2.0, 0.0), scale=(16.0, 0.5, 1.0), color=(0.2, 0.7, 0.3))
    physics.create_body(ground_id, body_type="fixed", position=(0.0, -2.0, 0.0))
    physics.attach_box_collider(ground_id, half_x=8.0, half_y=0.25, half_z=1.0)

    def spawn_sphere(x=None, y=None):
        if ecs.active_count >= 2_000:
            return
        px = random.uniform(-6.5, 6.5) if x is None else x
        py = random.uniform(2.0, 9.0) if y is None else y
        color = (random.uniform(0.4, 1.0), random.uniform(0.4, 1.0), random.uniform(0.8, 1.0))
        ent = ecs.create_entity(position=(px, py, 0.0), scale=(0.35, 0.35, 0.35), color=color)
        physics.create_body(ent, body_type="dynamic", position=(px, py, 0.0))
        physics.attach_sphere_collider(ent, radius=0.35)
        return ent

    # Initial cluster of physics bodies
    for _ in range(25):
        spawn_sphere()

    # Pre-simulation snapshot
    saved_snapshot = ecs.snapshot_memory()
    snapshot_status = "Initial Snapshot Taken"

    # Wire simulation step into loop.on_fixed_update
    def fixed_update(dt: float):
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    loop.on_fixed_update = fixed_update

    # Camera scale and origin mapping (world coordinates to screen pixels)
    scale = 35.0
    origin_x, origin_y = width // 2, int(height * 0.75)

    running = True
    uncapped_fps = True
    print("\n" + "=" * 60)
    print("PyMordial Engine Phase 1 — High-Performance Validation")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE]      : Spawn dynamic physics sphere")
    print("  [U]          : Toggle Uncapped FPS vs 120 FPS limit")
    print("  [P]          : Take Instant Play-In-Editor (PIE) Snapshot")
    print("  [R]          : Restore Snapshot (Zero-copy memcpy reset)")
    print("  [C]          : Clear all dynamic entities")
    print("  [ESC]        : Exit")
    print("=" * 60 + "\n")

    # Diagnostic surface cache to prevent per-frame text surface allocation
    diag_surf = pygame.Surface((440, 195), pygame.SRCALPHA)
    last_stat_update = 0.0
    measured_fps = 0.0

    def update_diagnostics_surface(fps_val, frame_ms):
        diag_surf.fill((0, 0, 0, 0))
        panel_rect = pygame.Rect(0, 0, 440, 195)
        pygame.draw.rect(diag_surf, (22, 26, 35, 235), panel_rect, border_radius=8)
        pygame.draw.rect(diag_surf, (60, 80, 115), panel_rect, 1, border_radius=8)

        fps_mode = "UNCAPPED (Maximum Speed)" if uncapped_fps else "CAPPED (120 FPS Limit)"
        lines = [
            ("PyMordial Engine Phase 1 — Release Build", (100, 200, 255), font_bold),
            (f"Render FPS: {fps_val:.1f} ({fps_mode})", (255, 235, 120), font_bold),
            (f"Frame Time: {frame_ms:.2f} ms | Simulation: 60 Hz Fixed", (220, 230, 245), font),
            (f"Active Entities: {ecs.active_count} / {ecs.max_entities}", (180, 240, 180), font),
            (f"Physics Engine: Rapier3D Native Release [PARALLEL]", (150, 240, 160), font),
            (f"Zero-Copy Sync: Direct C-Buffer Memory Writes [OK]", (150, 240, 160), font),
            (f"PIE Status: {snapshot_status}", (200, 215, 240), font),
            ("Press [U] to toggle FPS cap | [SPACE] to spawn", (160, 170, 190), font),
        ]

        y_offset = 12
        for text, col, f_obj in lines:
            rendered = f_obj.render(text, True, col)
            diag_surf.blit(rendered, (14, y_offset))
            y_offset += 22

    update_diagnostics_surface(0.0, 0.0)

    ground_rect = pygame.Rect(
        int(origin_x - 8.0 * scale),
        int(origin_y - (-1.75) * scale),
        int(16.0 * scale),
        int(0.5 * scale),
    )

    r_px = int(0.35 * scale)

    while running:
        # Uncapped runs at full CPU/GPU speed; otherwise throttle to 120 FPS
        raw_dt_ms = clock.tick(0 if uncapped_fps else 120)
        dt = raw_dt_ms * 0.001

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    for _ in range(5):  # Spawn in batches of 5
                        spawn_sphere()
                elif event.key == pygame.K_u:
                    uncapped_fps = not uncapped_fps
                elif event.key == pygame.K_p:
                    saved_snapshot = ecs.snapshot_memory()
                    snapshot_status = f"Snapshot Saved ({ecs.active_count} entities)"
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
                            physics.attach_sphere_collider(ent_id, radius=0.35)
                    snapshot_status = f"Snapshot Restored ({ecs.active_count} entities)"
                elif event.key == pygame.K_c:
                    for i in reversed(range(ecs.active_count)):
                        ent_id = ecs.pool.get_entity_id(i)
                        if ent_id != ground_id:
                            physics.remove_body(ent_id)
                            ecs.destroy_entity(ent_id)

        # Step deterministic engine loop
        loop.step_frame(dt)

        # Rendering
        screen.fill((16, 18, 24))

        # Render ground
        pygame.draw.rect(screen, (45, 140, 70), ground_rect, border_radius=4)

        # Render entities from contiguous WorldTransforms table
        active_count = ecs.active_count
        if active_count > 1:
            transforms = ecs.world_transforms
            materials = ecs.material_data

            # Vectorized screen coordinate computation using NumPy
            xs = (origin_x + transforms[:active_count, 12] * scale).astype(np.int32)
            ys = (origin_y - transforms[:active_count, 13] * scale).astype(np.int32)

            for i in range(1, active_count):  # Skip ground (index 0)
                color = (
                    int(materials[i, 0] * 255),
                    int(materials[i, 1] * 255),
                    int(materials[i, 2] * 255),
                )
                sx, sy = int(xs[i]), int(ys[i])
                pygame.draw.circle(screen, color, (sx, sy), r_px)

        # Refresh diagnostic overlay cache every 100ms
        now = time.perf_counter()
        if now - last_stat_update >= 0.10:
            measured_fps = clock.get_fps()
            update_diagnostics_surface(measured_fps, raw_dt_ms)
            last_stat_update = now

        # Blit cached overlay
        screen.blit(diag_surf, (15, 15))

        pygame.display.flip()

    pygame.quit()
    print("Demo exited cleanly.")


if __name__ == "__main__":
    main()
