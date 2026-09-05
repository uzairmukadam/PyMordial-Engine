"""Comprehensive unit tests for the Kinematic Character Controller (KCC),

character motor physics, slope climbing limits, autostep, and collision-aware camera.
"""

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig
from engine.core.character_camera import CharacterCamera, CharacterCameraConfig
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import get_quality_preset, GraphicsQuality


class TestCharacterController:
    def test_kcc_creation_and_grounding(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)

        # Fixed Ground box: center at y = -0.5, half-y = 0.5 (top surface at y = 0.0)
        ground_ent = ecs.create_entity(position=(0.0, -0.5, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_ent, half_x=20.0, half_y=0.5, half_z=20.0)

        # Character entity starting in the air at y = 2.0
        char_ent = ecs.create_entity(position=(0.0, 2.0, 0.0))
        motor = CharacterMotor(char_ent, physics, ecs, initial_position=(0.0, 2.0, 0.0))

        assert not motor.state.is_grounded

        # Step 60 Hz physics for 30 ticks (0.5 seconds of free fall under gravity)
        dt = 1.0 / 60.0
        for _ in range(30):
            motor.update(dt=dt, move_input=(0.0, 0.0))

        # Capsule bottom should be resting on ground at y = 0.0
        # Capsule center = ground_top (0.0) + half_height (0.5) + radius (0.4) + offset (0.02) = 0.92
        transform = physics.world.get_transform(char_ent)
        assert motor.state.is_grounded
        assert abs(transform[1] - 0.92) < 0.02
        assert abs(motor.state.velocity[1]) < 0.01

    def test_kcc_horizontal_movement_and_sprint(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager()

        # Ground box: top at y = 0.0
        ground_ent = ecs.create_entity(position=(0.0, -0.5, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_ent, half_x=50.0, half_y=0.5, half_z=50.0)

        # Character placed directly on ground at y = 0.92
        char_ent = ecs.create_entity(position=(0.0, 0.92, 0.0))
        cfg = CharacterMotorConfig(walk_speed=5.0, run_speed=10.0, acceleration=50.0)
        motor = CharacterMotor(char_ent, physics, ecs, config=cfg, initial_position=(0.0, 0.92, 0.0))

        dt = 1.0 / 60.0

        # Settle grounding
        motor.update(dt=dt, move_input=(0.0, 0.0))
        assert motor.state.is_grounded

        # Walk forward (camera yaw = 0 -> forward is -Z)
        for _ in range(30):
            motor.update(dt=dt, move_input=(1.0, 0.0), is_running=False, camera_yaw_deg=0.0)

        assert abs(motor.state.horizontal_speed - 5.0) < 0.1
        pos = physics.world.get_transform(char_ent)
        assert pos[2] < -1.0  # Moved forward along -Z

        # Now sprint forward
        for _ in range(30):
            motor.update(dt=dt, move_input=(1.0, 0.0), is_running=True, camera_yaw_deg=0.0)

        assert abs(motor.state.horizontal_speed - 10.0) < 0.1

    def test_kcc_jumping_and_air_physics(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager()

        # Ground box: top at y = 0.0
        ground_ent = ecs.create_entity(position=(0.0, -0.5, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_ent, half_x=20.0, half_y=0.5, half_z=20.0)

        char_ent = ecs.create_entity(position=(0.0, 0.92, 0.0))
        cfg = CharacterMotorConfig(jump_force=8.0, gravity=-20.0)
        motor = CharacterMotor(char_ent, physics, ecs, config=cfg, initial_position=(0.0, 0.92, 0.0))

        dt = 1.0 / 60.0
        motor.update(dt=dt, move_input=(0.0, 0.0))
        assert motor.state.is_grounded

        # Trigger jump
        motor.update(dt=dt, move_input=(0.0, 0.0), jump_requested=True)
        assert motor.state.is_jumping
        assert not motor.state.is_grounded
        assert motor.state.velocity[1] > 6.0

        # Follow ascent to apex
        highest_y = 0.0
        for _ in range(25):
            motor.update(dt=dt, move_input=(0.0, 0.0))
            y = physics.world.get_transform(char_ent)[1]
            if y > highest_y:
                highest_y = y

        # Apex should be around y = 0.92 + (8^2 / (2 * 20)) = 0.92 + 1.6 = 2.52
        assert highest_y > 2.2

        # Follow descent back to ground
        for _ in range(40):
            motor.update(dt=dt, move_input=(0.0, 0.0))

        assert motor.state.is_grounded
        assert abs(physics.world.get_transform(char_ent)[1] - 0.92) < 0.02

    def test_kcc_step_snapping_autostep(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager()

        # Ground platform: top at y = 0.0
        ground_ent = ecs.create_entity(position=(0.0, -0.5, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_ent, half_x=20.0, half_y=0.5, half_z=20.0)

        # Stair step at z = -2.0 with height 0.2m (top at y = 0.2m)
        step_ent = ecs.create_entity(position=(0.0, 0.1, -2.0))
        physics.create_body(step_ent, body_type="fixed", position=(0.0, 0.1, -2.0))
        physics.attach_box_collider(step_ent, half_x=5.0, half_y=0.1, half_z=1.0)

        # Character starting on ground at z = 0.0
        char_ent = ecs.create_entity(position=(0.0, 0.92, 0.0))
        cfg = CharacterMotorConfig(walk_speed=4.0, step_height=0.3)
        motor = CharacterMotor(char_ent, physics, ecs, config=cfg, initial_position=(0.0, 0.92, 0.0))

        dt = 1.0 / 60.0

        # Walk straight forward across the step
        for _ in range(60):
            motor.update(dt=dt, move_input=(1.0, 0.0), camera_yaw_deg=0.0)

        pos = physics.world.get_transform(char_ent)
        # Character successfully climbed over the step to z < -2.5
        assert pos[2] < -2.5
        assert motor.state.is_grounded

    def test_kcc_wall_collision_blocked(self):
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager()

        # Ground platform
        ground_ent = ecs.create_entity(position=(0.0, -0.5, 0.0))
        physics.create_body(ground_ent, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_ent, half_x=20.0, half_y=0.5, half_z=20.0)

        # Tall vertical barrier at z = -2.0 of height 3.0m (unclimbable)
        wall_ent = ecs.create_entity(position=(0.0, 1.5, -2.0))
        physics.create_body(wall_ent, body_type="fixed", position=(0.0, 1.5, -2.0))
        physics.attach_box_collider(wall_ent, half_x=5.0, half_y=1.5, half_z=0.2)

        # Character at z = -0.5
        char_ent = ecs.create_entity(position=(0.0, 0.92, -0.5))
        cfg = CharacterMotorConfig(walk_speed=5.0)
        motor = CharacterMotor(char_ent, physics, ecs, config=cfg, initial_position=(0.0, 0.92, -0.5))

        dt = 1.0 / 60.0
        # Walk straight into wall for 60 ticks
        for _ in range(60):
            motor.update(dt=dt, move_input=(1.0, 0.0), camera_yaw_deg=0.0)

        pos = physics.world.get_transform(char_ent)
        # Wall is at z = -2.0, with thickness 0.2 (front surface at -1.8)
        # Capsule radius = 0.4. Character should be stopped at approximately z = -1.38
        assert pos[2] > -1.5
        assert pos[2] < -1.2

    def test_character_camera_obstacle_clipping(self):
        physics = PhysicsManager()

        # Character standing at (0, 0, 0), camera target at (0, 1.35, 0)
        # Place a wall at (0, 1.35, 3.0) directly behind the character
        physics.create_body(10, body_type="fixed", position=(0.0, 1.35, 3.0))
        physics.attach_box_collider(10, half_x=5.0, half_y=5.0, half_z=0.1)

        cam_cfg = CharacterCameraConfig(distance=8.0, min_distance=1.0, camera_radius=0.2)
        # Looking straight ahead (yaw = 0, pitch = 0 -> direction to camera is +Z)
        cam = CharacterCamera(config=cam_cfg, initial_yaw_deg=0.0, initial_pitch_deg=0.0)

        # Update without physics: camera sits at full distance 8.0m
        eye_free, _ = cam.update(dt=0.016, character_pos=(0.0, 0.0, 0.0), physics=None)
        assert abs(eye_free[2] - 8.0) < 0.05

        # Update with physics: wall is hit at dist = 2.9m -> camera pulls in to ~2.7m
        eye_clipped, _ = cam.update(dt=0.016, character_pos=(0.0, 0.0, 0.0), physics=physics)
        assert eye_clipped[2] < 3.0
        assert eye_clipped[2] > 2.0

    def test_capsule_mesh_and_mdi_batch_rendering(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        try:
            cfg = get_quality_preset(GraphicsQuality.LOW)
            pipeline = RenderPipeline(ctx, config=cfg)

            ecs = EntityManager(max_entities=10)
            # Entity 0: Ground Box
            ecs.create_entity(
                position=(0.0, -0.5, 0.0),
                scale=(20.0, 0.5, 20.0),
                color=(0.3, 0.6, 0.3),
            )
            # Entity 1: Character Capsule
            ecs.create_entity(
                position=(0.0, 0.92, 0.0),
                scale=(1.0, 1.0, 1.0),
                color=(0.9, 0.2, 0.2),
                metallic=0.8,
                roughness=0.2,
            )

            # Check capsule allocation exists in mega buffer
            assert "capsule" in pipeline.mega_buffer.allocations
            capsule_alloc = pipeline.mega_buffer.allocations["capsule"]
            assert capsule_alloc.index_count > 0

            # Render frame with custom draw batches
            draw_batches = [
                ("cube", 1, 0),
                ("capsule", 1, 1),
            ]
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.92, 0.0),
                time_elapsed=0.1,
                draw_batches=draw_batches,
            )

            # Verify output rendered to final texture without error
            assert pipeline.output_texture_id > 0
            tex_data = pipeline.post_process.final_texture.read()
            assert len(tex_data) == 320 * 240 * 4
        finally:
            ctx.destroy()

    def test_character_motor_set_position_teleport(self):
        ecs = EntityManager(max_entities=10)
        physics = PhysicsManager()

        char_ent = ecs.create_entity(position=(0.0, 1.0, 0.0))
        motor = CharacterMotor(char_ent, physics, ecs, initial_position=(0.0, 1.0, 0.0))

        # Teleport character to (10.0, 5.0, -15.0)
        motor.set_position((10.0, 5.0, -15.0))

        # Check physics transform
        phys_pos = physics.world.get_transform(char_ent)
        assert abs(phys_pos[0] - 10.0) < 1e-4
        assert abs(phys_pos[1] - 5.0) < 1e-4
        assert abs(phys_pos[2] - (-15.0)) < 1e-4

        # Check ECS state
        dense_idx = ecs.pool.get_dense_index(char_ent)
        assert abs(ecs.rigid_body_state[1, dense_idx, 0] - 10.0) < 1e-4
        assert abs(ecs.rigid_body_state[1, dense_idx, 1] - 5.0) < 1e-4
        assert abs(ecs.rigid_body_state[1, dense_idx, 2] - (-15.0)) < 1e-4


