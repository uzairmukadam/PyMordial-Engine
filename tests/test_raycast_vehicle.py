"""Automated Unit Tests for Raycast Vehicle Physics System."""

import pytest

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager
from engine.physics.vehicle import (
    DriveType,
    VehicleConfig,
    RaycastVehicle,
)


@pytest.fixture
def physics_setup():
    """Sets up a minimal ECS, PhysicsManager with a static ground plane, and vehicle chassis."""
    ecs = EntityManager(max_entities=100)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0)

    # 1. Ground Plane entity (fixed box at y = 0.0, half-height = 0.5)
    ground_id = ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(50.0, 1.0, 50.0))
    physics.create_body(ground_id, body_type="fixed", position=(0.0, 0.0, 0.0))
    physics.attach_box_collider(ground_id, half_x=50.0, half_y=0.5, half_z=50.0, friction=0.8)

    # 2. Vehicle Chassis entity (spawned at y = 1.45 to rest on ground y = 0.5 + susp 0.45 + radius 0.35 + offset 0.15)
    config = VehicleConfig()
    chassis_id = ecs.create_entity(position=(0.0, 1.45, 0.0), scale=config.chassis_size)
    physics.create_body(chassis_id, body_type="dynamic", position=(0.0, 1.45, 0.0))
    hx, hy, hz = config.chassis_size[0] * 0.5, config.chassis_size[1] * 0.5, config.chassis_size[2] * 0.5
    vol = 8.0 * hx * hy * hz
    physics.attach_box_collider(chassis_id, half_x=hx, half_y=hy, half_z=hz, density=config.chassis_mass / vol)

    physics.sync_to_ecs(ecs)

    return ecs, physics, chassis_id


def test_vehicle_config_archetype_parametrization():
    """Verifies that vehicle configs can be customized without touching engine code."""
    muscle = VehicleConfig(
        chassis_mass=1500.0,
        engine_torque=3800.0,
        drive_type=DriveType.RWD,
        tire_grip=0.95,
    )
    suv = VehicleConfig(
        chassis_mass=2200.0,
        engine_torque=4500.0,
        drive_type=DriveType.AWD,
        tire_grip=1.1,
    )

    assert muscle.chassis_mass == 1500.0
    assert muscle.drive_type == DriveType.RWD
    assert suv.chassis_mass == 2200.0
    assert suv.drive_type == DriveType.AWD
    assert len(muscle.wheels) == 4


def test_raycast_vehicle_suspension_equilibrium(physics_setup):
    """Verifies that suspension rays detect ground and generate upward normal forces."""
    ecs, physics, chassis_id = physics_setup
    config = VehicleConfig()
    vehicle = RaycastVehicle(chassis_id, config, physics, ecs)

    dt = 1.0 / 60.0
    # Simulate a few steps to let suspension settle
    for _ in range(30):
        vehicle.update(dt)
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    # Check that wheels are grounded and supporting the vehicle
    grounded_count = sum(1 for w in vehicle.wheel_states if w.is_grounded)
    assert grounded_count == 4

    total_upward_force = sum(w.normal_force for w in vehicle.wheel_states)
    # Total normal force should be approximately chassis_mass * gravity (~1500 * 9.81 = 14715 N)
    expected_weight = config.chassis_mass * 9.81
    assert total_upward_force > expected_weight * 0.5


def test_raycast_vehicle_throttle_and_acceleration(physics_setup):
    """Verifies that applying forward throttle accelerates the vehicle forward in -Z."""
    ecs, physics, chassis_id = physics_setup
    config = VehicleConfig(engine_torque=5000.0)
    vehicle = RaycastVehicle(chassis_id, config, physics, ecs)

    dt = 1.0 / 60.0
    # Settle first
    for _ in range(20):
        vehicle.update(dt)
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    # Apply forward throttle
    vehicle.set_throttle(1.0)
    for _ in range(30):
        vehicle.update(dt)
        physics.step_simulation(dt)
        physics.sync_to_ecs(ecs)

    vx, vy, vz = physics.get_linvel(chassis_id)
    # Forward is in -Z direction in PyMordial coordinate system
    assert vz < -0.5, f"Expected negative Z forward velocity, got vz={vz}"
    assert vehicle.current_speed_mps > 0.5


def test_raycast_vehicle_steering_interpolation(physics_setup):
    """Verifies that steering smoothly interpolates towards target angle."""
    ecs, physics, chassis_id = physics_setup
    config = VehicleConfig(steer_speed=5.0, max_steer_angle_rad=0.5)
    vehicle = RaycastVehicle(chassis_id, config, physics, ecs)

    vehicle.set_steering(1.0)
    dt = 1.0 / 60.0

    vehicle.update(dt)
    assert vehicle.current_steer_angle > 0.0
    assert vehicle.current_steer_angle <= config.max_steer_angle_rad

    # Steer full right over several frames
    for _ in range(30):
        vehicle.update(dt)

    assert pytest.approx(vehicle.current_steer_angle, abs=0.01) == config.max_steer_angle_rad


def test_raycast_vehicle_surface_friction_scaling(physics_setup):
    """Verifies that surface_friction_mult scales effective tire grip (e.g. wet road 0.65x)."""
    ecs, physics, chassis_id = physics_setup
    config = VehicleConfig()
    vehicle = RaycastVehicle(chassis_id, config, physics, ecs)

    vehicle.set_surface_friction(0.65)
    assert vehicle.surface_friction_mult == 0.65

    vehicle.set_surface_friction(1.0)
    assert vehicle.surface_friction_mult == 1.0
