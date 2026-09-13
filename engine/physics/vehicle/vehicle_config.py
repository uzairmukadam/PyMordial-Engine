"""Parametrized Vehicle Configuration Data Structures.

Provides pure, domain-agnostic dataclasses for configuring raycast vehicle dynamics
(mass, dimensions, suspension spring-dampers, powertrain, steering, and tire grip).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class DriveType(str, Enum):
    """Powertrain wheel drive configuration."""
    RWD = "rwd"
    FWD = "fwd"
    AWD = "awd"


@dataclass(slots=True)
class WheelConfig:
    """Per-wheel geometric and suspension tuning parameters."""
    # Local position offset relative to chassis center (x: right, y: up, z: forward in vehicle space)
    offset: tuple[float, float, float]
    radius: float = 0.35
    width: float = 0.25
    suspension_rest_length: float = 0.45
    spring_stiffness: float = 35000.0   # N/m
    spring_damping: float = 4000.0      # N*s/m
    max_compression: float = 0.25       # Maximum allowable suspension compression in meters
    is_steerable: bool = False
    is_driven: bool = True
    brake_ratio: float = 1.0            # 1.0 = full brake force applied


@dataclass(slots=True)
class VehicleConfig:
    """Complete physical specification of a vehicle archetype."""
    # Chassis Physical Properties
    chassis_size: tuple[float, float, float] = (1.8, 0.7, 4.2)  # width (X), height (Y), length (Z)
    chassis_mass: float = 1500.0                                 # Mass in kg
    center_of_mass_offset: tuple[float, float, float] = (0.0, -0.2, 0.0)
    linear_damping: float = 0.05
    angular_damping: float = 0.35

    # Wheels (Default: 4 wheels FL, FR, RL, RR)
    wheels: list[WheelConfig] = field(default_factory=lambda: [
        # Front-Left (x: -0.85, z: -1.35 is forward)
        WheelConfig(offset=(-0.85, -0.15, -1.35), is_steerable=True, is_driven=False, brake_ratio=1.0),
        # Front-Right (x: 0.85, z: -1.35 is forward)
        WheelConfig(offset=(0.85, -0.15, -1.35), is_steerable=True, is_driven=False, brake_ratio=1.0),
        # Rear-Left (x: -0.85, z: 1.35 is rear)
        WheelConfig(offset=(-0.85, -0.15, 1.35), is_steerable=False, is_driven=True, brake_ratio=0.8),
        # Rear-Right (x: 0.85, z: 1.35 is rear)
        WheelConfig(offset=(0.85, -0.15, 1.35), is_steerable=False, is_driven=True, brake_ratio=0.8),
    ])

    # Powertrain & Brakes
    drive_type: DriveType = DriveType.RWD
    engine_torque: float = 3200.0       # Peak driving force (N)
    top_speed_mps: float = 55.0         # ~125 mph / 200 km/h
    brake_torque: float = 6500.0        # Braking force (N)
    handbrake_torque: float = 9000.0    # Rear wheel handbrake lock force (N)
    reverse_torque: float = 1800.0

    # Steering & Handling
    max_steer_angle_rad: float = 0.52   # ~30 degrees
    steer_speed: float = 4.5            # Rad/s response speed
    anti_roll_stiffness: float = 12000.0 # Force transfer between left and right wheels
    upright_stiffness: float = 34000.0   # Self-righting torque restoring chassis to level (N*m/rad)
    upright_damping: float = 5200.0      # Damping on roll and pitch oscillations (N*m*s/rad)

    # Tire Friction & Surface Slip
    tire_grip: float = 1.0              # Baseline lateral tire grip scalar
    drift_friction_factor: float = 0.45 # Lateral friction reduction under handbrake/skid
    surface_friction_mult: float = 1.0  # Environmental grip scalar (e.g. 0.65 in rain)
