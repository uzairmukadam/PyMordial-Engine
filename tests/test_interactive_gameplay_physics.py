"""Unit and Integration Tests for Phase 5: Interactive Gameplay Mechanics & Physics.

Tests native Rapier impulse/velocity APIs, dynamic prop management, destructibles
shattering into debris, character impulse pushing, opt-in PhysicsGrabber pick-and-throw,
and surface-aware footstep cadence.
"""

from __future__ import annotations
import math

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig
from engine.physics.props import InteractivePropManager, PropType
from engine.physics.interaction import PhysicsGrabber
from engine.gfx.mega_buffer import MeshAllocation


class TestInteractiveGameplayPhysics:
    """Test suite covering Phase 5 physics and gameplay interaction systems."""

    def test_native_rapier_velocity_and_impulse(self) -> None:
        """Validates native impulse, velocity, torque, damping, and dynamic gravity methods."""
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        ent_id = 42

        # Create body and attach collider so mass properties are computed
        physics.create_body(ent_id, body_type="dynamic", position=(0.0, 5.0, 0.0))
        physics.attach_box_collider(ent_id, 0.5, 0.5, 0.5)

        # 1. Direct velocity setter and getter
        physics.set_linvel(ent_id, (2.0, 4.0, -1.0))
        vx, vy, vz = physics.get_linvel(ent_id)
        assert math.isclose(vx, 2.0, abs_tol=1e-4)
        assert math.isclose(vy, 4.0, abs_tol=1e-4)
        assert math.isclose(vz, -1.0, abs_tol=1e-4)

        # 2. Linear impulse application
        physics.apply_impulse(ent_id, (0.0, 5.0, 3.0))
        vx2, vy2, vz2 = physics.get_linvel(ent_id)
        assert vy2 > vy
        assert vz2 > vz

        # 3. Angular velocity and torque impulse
        physics.set_angvel(ent_id, (0.0, 1.0, 0.0))
        wx, wy, wz = physics.get_angvel(ent_id)
        assert math.isclose(wy, 1.0, abs_tol=1e-4)

        physics.apply_torque_impulse(ent_id, (2.0, 0.0, 0.0))
        wx2, wy2, wz2 = physics.get_angvel(ent_id)
        assert wx2 > 0.5

        # 4. Damping
        physics.set_damping(ent_id, linear=0.5, angular=0.5)

        # 5. Dynamic gravity manipulation
        gx, gy, gz = physics.get_gravity()
        assert math.isclose(gy, -9.81, abs_tol=0.1)

        physics.set_gravity(0.0, -25.0, 0.0)
        gx_new, gy_new, gz_new = physics.get_gravity()
        assert math.isclose(gy_new, -25.0, abs_tol=1e-4)

    def test_prop_manager_spawning_and_simulation(self) -> None:
        """Validates spawning crates, spheres, barrels, and pyramid stacks in ECS and Rapier."""
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        # 1. Spawn crate
        crate_id = props.spawn_crate(position=(0.0, 5.0, 0.0), size=1.0)
        assert props.is_prop(crate_id)
        assert props.active_props_count == 1
        prop_inst = props.get_prop(crate_id)
        assert prop_inst is not None
        assert prop_inst.prop_type == PropType.CRATE
        assert prop_inst.shape_name == "cube"

        # 2. Spawn sphere and barrel
        sphere_id = props.spawn_sphere(position=(-2.0, 5.0, 0.0), radius=0.5)
        barrel_id = props.spawn_barrel(position=(2.0, 5.0, 0.0), half_height=0.5, radius=0.35)
        assert props.active_props_count == 3
        assert props.get_prop(sphere_id).prop_type == PropType.SPHERE
        assert props.get_prop(barrel_id).prop_type == PropType.BARREL

        # 3. Spawn structured pyramid
        pyramid_ids = props.spawn_pyramid(base_center=(0.0, 1.0, 5.0), rows=3, box_size=0.8)
        # Row 0: 3 crates, Row 1: 2 crates, Row 2: 1 crate -> total 6 crates
        assert len(pyramid_ids) == 6
        assert props.active_props_count == 9

        # Step simulation: all props fall under gravity
        for _ in range(10):
            physics.step_simulation(1 / 60)
            physics.sync_to_ecs(ecs)

        t_crate = physics.world.get_transform(crate_id)
        assert t_crate[1] < 5.0  # Crate fell under gravity

        # Clear all
        props.clear_all_props()
        assert props.active_props_count == 0

    def test_prop_manager_destruction_and_debris(self) -> None:
        """Validates destructible crates, damage thresholds, and debris fragment lifecycles."""
        ecs = EntityManager(max_entities=200)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        crate_id = props.spawn_crate(position=(0.0, 2.0, 0.0), size=1.0, health=40.0)

        # 1. Partial damage: health decreases but prop remains
        destroyed = props.damage_prop(crate_id, damage=15.0)
        assert not destroyed
        assert props.get_prop(crate_id).health == 25.0
        assert props.active_props_count == 1

        # 2. Fatal damage: crate shatters into mini debris fragments
        destroyed = props.damage_prop(crate_id, damage=30.0, hit_point=(0.0, 2.0, 0.0), hit_direction=(0.0, 0.0, 1.0))
        assert destroyed
        # Original crate is gone; only debris remain
        crates_remaining = [p for p in props._props.values() if p.prop_type == PropType.CRATE]
        assert len(crates_remaining) == 0

        # Debris pieces spawned
        assert props.active_props_count >= 4
        debris_props = [p for p in props._props.values() if p.prop_type == PropType.DEBRIS]
        assert len(debris_props) >= 4
        for d in debris_props:
            assert d.lifetime is not None
            assert d.lifetime > 0.0
            # Debris received velocity/impulse
            vx, vy, vz = physics.get_linvel(d.entity_id)
            assert (vx * vx + vy * vy + vz * vz) > 0.1

        # 3. Simulate time progression: debris pieces expire and despawn
        # Set lifetimes to small duration
        for d in debris_props:
            d.lifetime = 0.05

        props.update(dt=0.1)
        assert props.active_props_count == 0  # All debris cleanly removed

    def test_character_prop_impulse_push(self) -> None:
        """Validates momentum transfer from walking character into colliding dynamic props."""
        ecs = EntityManager(max_entities=50)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        # Crate placed right in front of character
        crate_id = props.spawn_crate(position=(0.0, 0.5, 1.0), size=0.8)
        initial_vz = physics.get_linvel(crate_id)[2]

        # Character moving forward (+Z direction) at 6.0 m/s
        char_pos = (0.0, 0.5, 0.0)
        char_velocity = (0.0, 0.0, 6.0)

        props.push_props_near_character(
            character_pos=char_pos,
            character_velocity=char_velocity,
            push_radius=1.5,
            push_force=8.0,
        )

        after_vz = physics.get_linvel(crate_id)[2]
        # Crate gained forward velocity along +Z
        assert after_vz > initial_vz + 1.0

    def test_physics_grabber_pick_carry_throw(self) -> None:
        """Validates decoupled PhysicsGrabber raycast acquisition, spring carry, and launch throw."""
        ecs = EntityManager(max_entities=50)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)
        grabber = PhysicsGrabber(physics=physics, prop_manager=props, max_reach=4.0, throw_force=20.0)

        # Crate 2.0m ahead
        crate_id = props.spawn_crate(position=(0.0, 1.5, 2.0), size=0.8)

        # 1. Grab raycast
        cam_pos = (0.0, 1.5, 0.0)
        cam_fwd = (0.0, 0.0, 1.0)
        grabbed = grabber.try_grab(origin=cam_pos, direction=cam_fwd)
        assert grabbed is True
        assert grabber.is_holding is True
        assert grabber.held_entity_id == crate_id
        assert props.get_prop(crate_id).is_grabbed is True

        # 2. Spring carry tracking update
        # Move camera to (0.0, 2.0, 0.0) with forward (1.0, 0.0, 0.0)
        grabber.update(dt=0.016, camera_pos=(0.0, 2.0, 0.0), camera_forward=(1.0, 0.0, 0.0))
        # Held crate should have velocity pulling it towards target (+X)
        vx, vy, vz = physics.get_linvel(crate_id)
        assert vx > 1.0

        # 3. Throw held prop
        thrown_id = grabber.throw_held(direction=(0.0, 0.0, 1.0), speed=25.0)
        assert thrown_id == crate_id
        assert grabber.is_holding is False
        assert props.get_prop(crate_id).is_grabbed is False

        # Thrown crate has high forward velocity
        tvx, tvy, tvz = physics.get_linvel(crate_id)
        assert tvz >= 24.0

    def test_character_motor_footstep_cadence_and_surface(self) -> None:
        """Validates footstep distance cadence accumulator and water surface detection."""
        ecs = EntityManager(max_entities=20)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)

        # Static ground
        ground_id = ecs.create_entity(position=(0.0, 0.0, 0.0))
        physics.create_body(ground_id, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(ground_id, 20.0, 0.5, 20.0)

        char_ent = ecs.create_entity(position=(0.0, 0.92, 0.0))
        cfg = CharacterMotorConfig(walk_speed=5.0, run_speed=10.0)
        motor = CharacterMotor(char_ent, physics, ecs, config=cfg, initial_position=(0.0, 0.92, 0.0))

        footstep_events: list[tuple[tuple[float, float, float], bool]] = []
        motor.on_footstep = lambda pos, in_water: footstep_events.append((pos, in_water))
        motor.walk_stride = 1.0  # Trigger step every 1.0m for test
        motor.water_height = 0.5

        # Move character forward 3.0 meters over several 60 Hz ticks
        dt = 1 / 60
        for _ in range(60):
            motor.update(
                dt=dt,
                move_input=(1.0, 0.0),  # Forward
                is_running=False,
                jump_requested=False,
                camera_yaw_deg=0.0,
            )

        # Should have triggered at least 2 footstep events over 3.0 meters
        assert len(footstep_events) >= 2
        # Feet are at y = 0.92 - 0.5 - 0.4 = 0.02, which is <= water_height (0.5), so in_water should be True!
        assert footstep_events[0][1] is True

        # Now raise ground above water height (water_height = -1.0)
        motor.water_height = -1.0
        footstep_events.clear()
        for _ in range(60):
            motor.update(
                dt=dt,
                move_input=(1.0, 0.0),
                is_running=False,
                jump_requested=False,
                camera_yaw_deg=0.0,
            )
        assert len(footstep_events) >= 2
        assert footstep_events[0][1] is False  # Dry ground

    def test_prop_draw_batches_generation(self) -> None:
        """Validates generation of MDI draw batch tuples for active props."""
        ecs = EntityManager(max_entities=50)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        props.spawn_crate(position=(0.0, 1.0, 0.0))
        props.spawn_sphere(position=(2.0, 1.0, 0.0))
        props.spawn_barrel(position=(-2.0, 1.0, 0.0))

        alloc_cube = MeshAllocation(first_index=0, index_count=36, base_vertex=0, vertex_count=24)
        alloc_sphere = MeshAllocation(first_index=36, index_count=120, base_vertex=24, vertex_count=60)
        alloc_capsule = MeshAllocation(first_index=156, index_count=80, base_vertex=84, vertex_count=40)

        batches = props.get_draw_batches(alloc_cube, alloc_sphere, alloc_capsule)
        assert len(batches) == 3

        # Cube batch
        cube_batches = [b for b in batches if b[0] == alloc_cube]
        sphere_batches = [b for b in batches if b[0] == alloc_sphere]
        capsule_batches = [b for b in batches if b[0] == alloc_capsule]

        assert len(cube_batches) == 1
        assert len(sphere_batches) == 1
        assert len(capsule_batches) == 1

    def test_prop_mass_inertia_and_touch_contact_only(self) -> None:
        """Validates that props have realistic mass and do not trigger push impulses across an air gap."""
        ecs = EntityManager(max_entities=50)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        # 0.8m crate: volume = 0.8^3 = 0.512 m^3. At density 75.0, mass ~ 38.4 kg
        crate_id = props.spawn_crate(position=(0.0, 0.5, 1.2), size=0.8)
        mass = physics.get_mass(crate_id)
        assert mass >= 30.0  # Real weight! Not a 0.5 kg styrofoam feather

        # Character at (0, 0.5, 0), moving forward at 4.0 m/s
        # Contact distance is 0.4 (char) + 0.4 (crate) = 0.8m.
        # At z = 1.2m, distance is 1.2m > 0.84m (air gap of 0.36m).
        char_pos = (0.0, 0.5, 0.0)
        char_vel = (0.0, 0.0, 4.0)

        props.push_props_near_character(
            character_pos=char_pos,
            character_velocity=char_vel,
            push_radius=None,  # Strict touch-only
        )

        # Crate should NOT have moved at all (zero impulse across air gap)
        vz_airgap = physics.get_linvel(crate_id)[2]
        assert abs(vz_airgap) < 1e-4

        # Now place character in actual physical contact (distance = 0.82m <= 0.84m)
        char_contact_pos = (0.0, 0.5, 0.38)
        props.push_props_near_character(
            character_pos=char_contact_pos,
            character_velocity=char_vel,
            push_radius=None,
        )

        vz_contact = physics.get_linvel(crate_id)[2]
        # Should now have cleanly pushed on physical contact
        assert vz_contact > 0.5

    def test_prop_velocity_clamping_to_character_speed(self) -> None:
        """Validates that pushed props never exceed the character's movement speed."""
        ecs = EntityManager(max_entities=50)
        physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)
        props = InteractivePropManager(physics=physics, ecs=ecs)

        crate_id = props.spawn_crate(position=(0.0, 0.5, 0.8), size=0.8)
        char_pos = (0.0, 0.5, 0.0)
        char_speed = 3.5
        char_vel = (0.0, 0.0, char_speed)

        # Repeatedly push over 20 consecutive simulation frames
        for _ in range(20):
            props.push_props_near_character(
                character_pos=char_pos,
                character_velocity=char_vel,
                push_radius=1.2,
                push_force=8.0,
            )

        vz = physics.get_linvel(crate_id)[2]
        # Must not have flown away at 50+ m/s; should be bounded by walking/running speed!
        assert vz <= char_speed + 0.05
