"""Micro-benchmark for PyMordial Phase 1 hot paths."""

import time
import numpy as np

from engine.core.ecs import EntityManager
from engine.core.loop import EngineLoop
from engine.physics.rapier_world import PhysicsManager


def benchmark():
    ecs = EntityManager(max_entities=10_000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)

    # Spawn 1,000 physics entities
    entity_count = 1_000
    for i in range(entity_count):
        ent = ecs.create_entity(position=(float(i), 10.0, 0.0))
        physics.create_body(ent, body_type="dynamic", position=(float(i), 10.0, 0.0))
        physics.attach_sphere_collider(ent, radius=0.5)

    print(f"Benchmarking with {entity_count} active physics entities...")

    # 1. Benchmark Physics Step (GIL released in Rust)
    t0 = time.perf_counter()
    steps = 500
    for _ in range(steps):
        physics.step_simulation(1.0 / 60.0)
    t_physics = time.perf_counter() - t0
    print(f"Physics 500 steps: {t_physics*1000:.2f} ms ({t_physics/steps*1000:.3f} ms/step)")

    # 2. Benchmark Physics -> ECS Sync
    t0 = time.perf_counter()
    for _ in range(steps):
        physics.sync_to_ecs(ecs)
    t_sync = time.perf_counter() - t0
    print(f"Physics -> ECS Sync 500 runs: {t_sync*1000:.2f} ms ({t_sync/steps*1000:.3f} ms/sync)")

    # 3. Benchmark ECS Sub-frame NLERP Interpolation
    t0 = time.perf_counter()
    for _ in range(steps):
        ecs.interpolate_render_transforms(0.5)
    t_interp = time.perf_counter() - t0
    print(f"ECS NLERP Interpolation 500 runs: {t_interp*1000:.2f} ms ({t_interp/steps*1000:.3f} ms/interp)")

    # 4. Benchmark Full Engine Loop Step
    loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)
    loop.on_fixed_update = lambda dt: (physics.step_simulation(dt), physics.sync_to_ecs(ecs))

    t0 = time.perf_counter()
    frame_count = 1_000
    for _ in range(frame_count):
        loop.step_frame(1.0 / 60.0)
    t_loop = time.perf_counter() - t0
    effective_fps = frame_count / t_loop
    print(f"Full Loop {frame_count} frames: {t_loop*1000:.2f} ms ({t_loop/frame_count*1000:.3f} ms/frame -> {effective_fps:.1f} FPS for 1,000 entities!)")


if __name__ == "__main__":
    benchmark()
