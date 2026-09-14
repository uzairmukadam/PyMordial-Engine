"""Unit Tests for Phase 2: Vehicle Collision Damage, Mass Scaling, Armor Mitigation, and Totaled States."""

import pytest
import numpy as np

from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_damage import VehicleDamageController
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_vfx import VehicleVFXManager
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_presets import CLASS_SPECS, CarClass


class DummyPhysics:
    """Mock PhysicsManager providing controllable linear velocity."""
    def __init__(self, linvel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.linvel = linvel

    def get_linvel(self, entity_id: int) -> tuple[float, float, float]:
        return self.linvel


class DummyApp:
    """Minimal ProjectApp stub for unit testing."""
    def __init__(self, linvel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.physics = DummyPhysics(linvel)


def test_damage_velocity_threshold():
    """Verify that collisions below threshold (3.5 m/s) inflict 0% damage."""
    app = DummyApp(linvel=(0.0, 0.0, -20.0))
    ctrl = VehicleDamageController(entity_id=1, chassis_mass=1400.0, armor_factor=1.0)

    # First update seeds initial velocity
    ctrl.update(app, 0.016)
    assert ctrl.damage == 0.0

    # Slight speed reduction (20 -> 18 m/s: delta-V is 2.0 m/s, below threshold 3.5 m/s)
    app.physics.linvel = (0.0, 0.0, -18.0)
    dmg = ctrl.update(app, 0.016)
    assert dmg == 0.0
    assert ctrl.damage == 0.0


def test_armor_mitigation():
    """Verify Stage 3 armor reduces damage by 60% compared to stock Stage 0."""
    # Stage 0: armor_factor = 1.0
    app0 = DummyApp(linvel=(0.0, 0.0, -25.0))
    ctrl0 = VehicleDamageController(entity_id=1, chassis_mass=1400.0, armor_factor=1.0)
    ctrl0.update(app0, 0.016)

    # Full stop collision (25 -> 0 m/s: delta-V is 25 m/s)
    app0.physics.linvel = (0.0, 0.0, 0.0)
    dmg0 = ctrl0.update(app0, 0.016)

    # Stage 3: armor_factor = 0.40
    stage3_armor = CLASS_SPECS[CarClass.STARTER_SEDAN].armor_damage_factors[3]
    assert stage3_armor == 0.40

    app3 = DummyApp(linvel=(0.0, 0.0, -25.0))
    ctrl3 = VehicleDamageController(entity_id=2, chassis_mass=1400.0, armor_factor=stage3_armor)
    ctrl3.update(app3, 0.016)

    app3.physics.linvel = (0.0, 0.0, 0.0)
    dmg3 = ctrl3.update(app3, 0.016)

    assert dmg0 > 0.0
    assert dmg3 > 0.0
    # Stage 3 damage should be exactly 40% of Stage 0 damage (60% reduction)
    assert pytest.approx(dmg3 / dmg0, 0.01) == 0.40


def test_mass_scaling():
    """Verify heavy Enforcer SUV absorbs impact force better than lightweight Compact Tuner."""
    # Tuner (1180 kg)
    app_tuner = DummyApp(linvel=(0.0, 0.0, -20.0))
    ctrl_tuner = VehicleDamageController(entity_id=1, chassis_mass=1180.0, armor_factor=1.0)
    ctrl_tuner.update(app_tuner, 0.016)
    app_tuner.physics.linvel = (0.0, 0.0, 0.0)
    dmg_tuner = ctrl_tuner.update(app_tuner, 0.016)

    # SUV (2250 kg)
    app_suv = DummyApp(linvel=(0.0, 0.0, -20.0))
    ctrl_suv = VehicleDamageController(entity_id=2, chassis_mass=2250.0, armor_factor=1.0)
    ctrl_suv.update(app_suv, 0.016)
    app_suv.physics.linvel = (0.0, 0.0, 0.0)
    dmg_suv = ctrl_suv.update(app_suv, 0.016)

    assert dmg_suv < dmg_tuner
    # Mass ratio check
    ratio = dmg_suv / dmg_tuner
    expected_ratio = 1180.0 / 2250.0
    assert pytest.approx(ratio, 0.02) == expected_ratio


def test_torque_degradation_and_totaled_state():
    """Verify mechanical torque degradation above 50% and stall at 100%."""
    ctrl = VehicleDamageController(entity_id=1, initial_damage=0.0)
    assert ctrl.torque_multiplier == 1.0
    assert ctrl.top_speed_multiplier == 1.0
    assert ctrl.is_totaled is False

    # At 40% damage: still full power
    ctrl.damage = 40.0
    assert ctrl.torque_multiplier == 1.0
    assert ctrl.is_totaled is False

    # At 75% damage: degraded power
    ctrl.damage = 75.0
    assert 0.70 < ctrl.torque_multiplier < 0.90
    assert 0.75 < ctrl.top_speed_multiplier < 0.95
    assert ctrl.is_totaled is False

    # At 100% damage: Totaled engine stall
    ctrl.damage = 100.0
    assert ctrl.is_totaled is True
    assert ctrl.torque_multiplier == 0.0
    assert ctrl.top_speed_multiplier == 0.0


def test_camera_shake_on_high_impulse():
    """Verify high impulse crashes trigger camera shake recoil."""
    app = DummyApp(linvel=(0.0, 0.0, -30.0))
    ctrl = VehicleDamageController(entity_id=1)
    vfx = VehicleVFXManager()
    ctrl.set_on_impact_callback(vfx.trigger_crash_impulse)

    ctrl.update(app, 0.016)
    assert vfx.camera_shake_intensity == 0.0

    # High velocity collision
    app.physics.linvel = (0.0, 0.0, 0.0)
    ctrl.update(app, 0.016)

    assert vfx.camera_shake_intensity > 0.3


def test_damage_repair_and_sync():
    """Verify set_damage and repair logic properly clears damage, bias, and totaled status."""
    ctrl = VehicleDamageController(entity_id=1, initial_damage=100.0)
    assert ctrl.is_totaled is True
    assert ctrl.torque_multiplier == 0.0

    # Setting damage to 0 (repair)
    ctrl.set_damage(0.0)
    assert ctrl.damage == 0.0
    assert ctrl.is_totaled is False
    assert ctrl.torque_multiplier == 1.0
    assert ctrl.steering_pull_bias == 0.0
