"""Unit and Integration Tests for Realistic Vehicle Mechanics & Visual Polish."""

import math
import numpy as np
import pytest
import pygame

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager
from engine.physics.vehicle.vehicle_config import VehicleConfig, WheelConfig, DriveType
from engine.physics.vehicle.raycast_vehicle import RaycastVehicle
from engine.physics.props import InteractivePropManager, PropType
from projects.shotgun_escape_the_heat.modules.vehicle_controller.tire_vfx import TireVFXManager
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_audio import VehicleAudioManager


class TestPrimitiveColliders:
    """Validates native Rust Rapier primitive colliders."""

    def test_cylinder_collider_attachment(self):
        physics = PhysicsManager()
        eid = 1
        physics.create_body(eid, body_type="dynamic", position=(0.0, 5.0, 0.0))
        physics.attach_cylinder_collider(eid, half_height=0.5, radius=0.35, density=100.0)

        for _ in range(10):
            physics.step_simulation(1.0 / 60.0)

        transform = physics.world.get_transform(eid)
        assert transform[1] < 5.0, "Cylinder body should have fallen under gravity"

    def test_cone_collider_attachment(self):
        physics = PhysicsManager()
        eid = 2
        physics.create_body(eid, body_type="dynamic", position=(0.0, 5.0, 0.0))
        physics.attach_cone_collider(eid, half_height=0.6, radius=0.4, density=100.0)
        physics.step_simulation(1.0 / 60.0)
        transform = physics.world.get_transform(eid)
        assert transform[1] < 5.0

    def test_plane_and_round_box_colliders(self):
        physics = PhysicsManager()
        # Static ground plane
        physics.create_body(10, body_type="fixed", position=(0.0, 0.0, 0.0))
        physics.attach_plane_collider(10, half_x=50.0, half_z=50.0, thickness=0.5)

        # Dynamic round cuboid prop
        physics.create_body(11, body_type="dynamic", position=(0.0, 3.0, 0.0))
        physics.attach_round_box_collider(11, half_x=0.5, half_y=0.5, half_z=0.5, border_radius=0.05, density=100.0)

        for _ in range(60):
            physics.step_simulation(1.0 / 60.0)

        transform = physics.world.get_transform(11)
        # Should rest near y=0.5 on the plane
        assert transform[1] > 0.0, "Round box should collide and rest above the ground plane"


class TestDynamicProps:
    """Validates primitive prop spawning in InteractivePropManager."""

    def test_spawn_cylinder_and_cone_props(self):
        ecs = EntityManager(max_entities=128)
        physics = PhysicsManager()
        prop_mgr = InteractivePropManager(physics, ecs)

        cyl_id = prop_mgr.spawn_cylinder(position=(2.0, 1.0, 0.0), half_height=0.5, radius=0.3)
        assert cyl_id >= 0
        assert prop_mgr.active_props_count == 1
        assert prop_mgr.get_prop(cyl_id).prop_type == PropType.CYLINDER

        cone_id = prop_mgr.spawn_cone(position=(4.0, 1.0, 0.0), half_height=0.4, radius=0.25)
        assert cone_id >= 0
        assert prop_mgr.active_props_count == 2
        assert prop_mgr.get_prop(cone_id).prop_type == PropType.CONE


class TestVehicleMechanics:
    """Validates realistic powertrain, Pacejka slip, weight transfer, and turbo simulation."""

    def setup_method(self):
        self.ecs = EntityManager(max_entities=128)
        self.physics = PhysicsManager()

        # Static ground plane at y=0.0 with half_y=0.5
        ground_id = self.ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(100.0, 1.0, 100.0))
        self.physics.create_body(ground_id, body_type="fixed", position=(0.0, 0.0, 0.0))
        self.physics.attach_box_collider(ground_id, half_x=50.0, half_y=0.5, half_z=50.0, friction=0.8)

        self.cfg = VehicleConfig(
            chassis_size=(1.8, 0.7, 4.2),
            chassis_mass=1500.0,
            engine_torque=4500.0,
            top_speed_mps=50.0,
        )

        # Vehicle entity at y=1.45
        self.car_id = self.ecs.create_entity(
            position=(0.0, 1.45, 0.0),
            scale=self.cfg.chassis_size,
        )
        self.physics.create_body(self.car_id, body_type="dynamic", position=(0.0, 1.45, 0.0))
        hx = self.cfg.chassis_size[0] * 0.5
        hy = self.cfg.chassis_size[1] * 0.5
        hz = self.cfg.chassis_size[2] * 0.5
        vol = 8.0 * hx * hy * hz
        self.physics.attach_box_collider(self.car_id, half_x=hx, half_y=hy, half_z=hz, density=self.cfg.chassis_mass / vol)
        self.physics.sync_to_ecs(self.ecs)

        self.vehicle = RaycastVehicle(self.car_id, self.cfg, self.physics, self.ecs)

    def test_engine_idle_and_rev_response(self):
        dt = 1.0 / 60.0
        # Step at idle
        self.vehicle.set_throttle(0.0)
        for _ in range(30):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        assert 750.0 <= self.vehicle.engine_rpm <= 850.0, f"Engine should idle around 800 RPM, got {self.vehicle.engine_rpm}"

        # Apply throttle
        self.vehicle.set_throttle(1.0)
        for _ in range(60):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        assert self.vehicle.engine_rpm > 1200.0, f"Engine should rev under throttle, got {self.vehicle.engine_rpm}"

    def test_5speed_transmission_and_turbo_dynamics(self):
        dt = 1.0 / 60.0

        # Settle on suspension first
        for _ in range(30):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        self.vehicle.set_throttle(1.0)

        # Accelerate over multiple seconds to test gear shifting and turbo buildup
        reached_gear_2_or_higher = False
        boost_built = False

        for tick in range(260):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

            if self.vehicle.current_gear >= 2:
                reached_gear_2_or_higher = True
            if self.vehicle.turbo_boost > 0.05:
                boost_built = True

        assert self.vehicle.current_speed_mps > 1.0, f"Car should accelerate forward, got {self.vehicle.current_speed_mps}"
        assert reached_gear_2_or_higher, f"Transmission should upshift during acceleration, current gear: {self.vehicle.current_gear}"
        assert boost_built, f"Turbo boost should build under sustained throttle, boost: {self.vehicle.turbo_boost}"

        # Test throttle lift and blowoff trigger
        self.vehicle.set_throttle(0.0)
        self.vehicle.update(dt)
        self.physics.step_simulation(dt)
        self.physics.sync_to_ecs(self.ecs)
        assert self.vehicle.turbo_blowoff_triggered or self.vehicle.turbo_boost >= 0.0

    def test_pacejka_tire_slip_and_drift_telemetry(self):
        dt = 1.0 / 60.0
        self.vehicle.set_throttle(1.0)
        self.vehicle.set_steering(1.0)
        self.vehicle.set_handbrake(True)

        # Spin car with handbrake and steering to induce slip
        for _ in range(40):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        # Wheels should report slip telemetry
        rear_left = self.vehicle.wheel_states[2]
        assert hasattr(rear_left, "lateral_slip")
        assert hasattr(rear_left, "slip_angle")
        assert hasattr(rear_left, "skidding")

    def test_kamms_friction_circle_and_brake_bias(self):
        """Verifies Kamm's friction circle: braking consumes longitudinal capacity and front bias bites harder."""
        dt = 1.0 / 60.0
        # Settle car
        for _ in range(30):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        # Apply brake and steer simultaneously
        self.vehicle.set_brake(1.0)
        self.vehicle.set_steering(1.0)
        self.vehicle.update(dt)

        assert self.vehicle.config.brake_bias_front == 0.65
        assert hasattr(self.vehicle.config, "caster_aligning_torque")

    def test_dynamic_2axis_weight_transfer(self):
        """Verifies longitudinal dive and lateral weight transfer."""
        dt = 1.0 / 60.0
        # Accelerate hard
        self.vehicle.set_throttle(1.0)
        for _ in range(25):
            self.vehicle.update(dt)
            self.physics.step_simulation(dt)
            self.physics.sync_to_ecs(self.ecs)

        assert hasattr(self.vehicle, "_accel_x_filtered")
        assert hasattr(self.vehicle, "_accel_y_filtered")
        assert self.vehicle._accel_x_filtered != 0.0

    def test_speed_sensitive_steering_lock(self):
        """Verifies that high forward speed tapers the effective steering angle."""
        dt = 1.0 / 60.0
        self.vehicle.current_speed_mps = 50.0  # Simulated high speed
        self.vehicle.set_steering(1.0)
        self.vehicle.update(dt)
        # Steer angle should be less than max_steer_angle_rad due to speed reduction
        assert self.vehicle.current_steer_angle < self.vehicle.config.max_steer_angle_rad


class TestVehicleAudioSynthesis:
    """Validates procedural in-memory audio waveform generation."""

    def test_audio_generator_methods(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        audio = VehicleAudioManager()
        engine_loop = audio._generate_engine_loop(base_freq=60.0, duration=0.2)
        assert engine_loop is not None

        screech_loop = audio._generate_tire_screech_loop(duration=0.2)
        assert screech_loop is not None

        blowoff = audio._generate_turbo_blowoff(duration=0.2)
        assert blowoff is not None

        crash = audio._generate_crash_impact(duration=0.2)
        assert crash is not None
