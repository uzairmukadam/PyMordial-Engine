"""Authoritative Kinematic Character Motor System for PyMordial Engine.

Implements responsive 60 Hz character physics:
- Input-driven horizontal acceleration & deceleration with camera orientation alignment
- Continuous collision sweep & slide against static and dynamic geometry
- Dynamic ground snapping and curb/step climbing (<= 0.3m)
- Slope limit handling (<= 45 deg climb, slide on steep slopes)
- Variable jumping and air control
- Contiguous zero-allocation updates directly into ECS RigidBodyState
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable
import numpy as np

from engine.core.ecs import EntityManager
from engine.physics.rapier_world import PhysicsManager


@dataclass(slots=True)
class CharacterMotorConfig:
    """Tuning parameters for character kinematics and movement feel."""

    walk_speed: float = 6.0          # Normal walking speed in m/s
    run_speed: float = 11.0          # Sprinting speed in m/s
    jump_force: float = 7.5          # Initial vertical velocity upon jump in m/s
    gravity: float = -20.0           # Downward acceleration in m/s^2 (punchy feel)
    acceleration: float = 40.0       # Ground acceleration in m/s^2
    air_acceleration: float = 15.0   # Airborne acceleration in m/s^2
    deceleration: float = 45.0       # Ground friction / deceleration in m/s^2
    capsule_half_height: float = 0.5 # Cylinder half-height in meters
    capsule_radius: float = 0.4      # Hemisphere cap radius in meters
    max_slope_deg: float = 45.0      # Maximum climbable slope angle in degrees
    step_height: float = 0.3         # Maximum vertical stair step / curb height in meters
    snap_to_ground: float = 0.25     # Ground snapping probe distance in meters


class CharacterMotorState:
    """Mutable runtime state of the character motor."""

    __slots__ = (
        "velocity",
        "is_grounded",
        "is_sliding",
        "is_jumping",
        "yaw_deg",
    )

    def __init__(self) -> None:
        self.velocity = np.zeros(3, dtype=np.float32)
        self.is_grounded = False
        self.is_sliding = False
        self.is_jumping = False
        self.yaw_deg = 0.0

    @property
    def horizontal_speed(self) -> float:
        """Current planar speed in meters per second."""
        vx = float(self.velocity[0])
        vz = float(self.velocity[2])
        return math.sqrt(vx * vx + vz * vz)


class CharacterMotor:
    """Coordinates input, kinematics, Rapier3D sweeps, and ECS transform updates."""

    __slots__ = (
        "entity_id",
        "physics",
        "ecs",
        "config",
        "state",
        "distance_traversed",
        "walk_stride",
        "run_stride",
        "water_height",
        "on_footstep",
    )

    def __init__(
        self,
        entity_id: int,
        physics: PhysicsManager,
        ecs: EntityManager,
        config: CharacterMotorConfig | None = None,
        initial_position: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        self.entity_id = entity_id
        self.physics = physics
        self.ecs = ecs
        self.config = config if config is not None else CharacterMotorConfig()
        self.state = CharacterMotorState()
        self.distance_traversed: float = 0.0
        self.walk_stride: float = 1.85
        self.run_stride: float = 1.25
        self.water_height: float = 0.45
        self.on_footstep: Callable[[tuple[float, float, float], bool], None] | None = None

        # Register native kinematic character controller in Rapier world
        self.physics.create_character_controller(
            entity_id=self.entity_id,
            half_height=self.config.capsule_half_height,
            radius=self.config.capsule_radius,
            max_slope_deg=self.config.max_slope_deg,
            step_height=self.config.step_height,
            snap_to_ground=self.config.snap_to_ground,
            position=initial_position,
        )

        # Initialize ECS RigidBodyState position
        dense_idx = self.ecs.pool.get_dense_index(self.entity_id)
        if dense_idx >= 0:
            rbs = self.ecs.rigid_body_state
            for buf_idx in (0, 1):
                rbs[buf_idx, dense_idx, 0] = initial_position[0]
                rbs[buf_idx, dense_idx, 1] = initial_position[1]
                rbs[buf_idx, dense_idx, 2] = initial_position[2]
                rbs[buf_idx, dense_idx, 3] = 0.0
                rbs[buf_idx, dense_idx, 4] = 0.0
                rbs[buf_idx, dense_idx, 5] = 0.0
                rbs[buf_idx, dense_idx, 6] = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        """Current character world position."""
        dense_idx = self.ecs.pool.get_dense_index(self.entity_id)
        if dense_idx >= 0:
            return (
                float(self.ecs.rigid_body_state[1, dense_idx, 0]),
                float(self.ecs.rigid_body_state[1, dense_idx, 1]),
                float(self.ecs.rigid_body_state[1, dense_idx, 2]),
            )
        pos, _ = self.physics.get_transform(self.entity_id)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def set_position(
        self,
        position: tuple[float, float, float] | np.ndarray,
        reset_velocity: bool = True,
    ) -> None:
        """Teleports the character to a target world position."""
        px, py, pz = float(position[0]), float(position[1]), float(position[2])
        # Update Rapier rigid body
        self.physics.set_transform(self.entity_id, (px, py, pz), (0.0, 0.0, 0.0, 1.0))

        # Update ECS RigidBodyState for both buffers (to prevent interpolation smear)
        dense_idx = self.ecs.pool.get_dense_index(self.entity_id)
        if dense_idx >= 0:
            rbs = self.ecs.rigid_body_state
            for buf_idx in (0, 1):
                rbs[buf_idx, dense_idx, 0] = px
                rbs[buf_idx, dense_idx, 1] = py
                rbs[buf_idx, dense_idx, 2] = pz
                rbs[buf_idx, dense_idx, 3] = 0.0
                rbs[buf_idx, dense_idx, 4] = 0.0
                rbs[buf_idx, dense_idx, 5] = 0.0
                rbs[buf_idx, dense_idx, 6] = 1.0
            self.ecs.recompute_matrix(dense_idx)

        if reset_velocity:
            self.state.velocity.fill(0.0)
            self.state.is_grounded = False
            self.state.is_jumping = False
            self.state.is_sliding = False
            self.distance_traversed = 0.0

    def teleport(
        self,
        position: tuple[float, float, float] | np.ndarray,
        reset_velocity: bool = True,
    ) -> None:
        """Teleports the character to a target world position (alias for set_position)."""
        self.set_position(position, reset_velocity=reset_velocity)

    def update(
        self,
        dt: float,
        move_input: tuple[float, float],
        is_running: bool = False,
        jump_requested: bool = False,
        camera_yaw_deg: float = 0.0,
    ) -> None:
        """Executes a 60 Hz character physics step with continuous collision detection.

        Args:
            dt: Fixed simulation timestep in seconds (e.g. 1/60).
            move_input: Tuple of (forward, strafe), each in [-1.0, 1.0].
            is_running: True if sprint modifier is held.
            jump_requested: True if jump action was triggered this tick.
            camera_yaw_deg: Camera orbit horizontal yaw in degrees.
        """
        if dt <= 1e-6:
            return

        fwd_in, strafe_in = move_input[0], move_input[1]
        cfg = self.config
        state = self.state

        # 1. Compute desired movement direction aligned with camera horizontal view
        yaw_rad = math.radians(camera_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Camera forward projected onto horizontal plane: (-sin(yaw), -cos(yaw))
        cam_fwd_x = -sin_yaw
        cam_fwd_z = -cos_yaw
        # Camera right: (cos(yaw), -sin(yaw))
        cam_rgt_x = cos_yaw
        cam_rgt_z = -sin_yaw

        dir_x = cam_fwd_x * fwd_in + cam_rgt_x * strafe_in
        dir_z = cam_fwd_z * fwd_in + cam_rgt_z * strafe_in
        dir_len = math.sqrt(dir_x * dir_x + dir_z * dir_z)

        target_speed = (cfg.run_speed if is_running else cfg.walk_speed) if dir_len > 0.05 else 0.0
        if dir_len > 1.0:
            dir_x /= dir_len
            dir_z /= dir_len

        target_vx = dir_x * target_speed
        target_vz = dir_z * target_speed

        # 2. Horizontal Acceleration / Deceleration
        accel = cfg.acceleration if state.is_grounded else cfg.air_acceleration
        decel = cfg.deceleration if state.is_grounded else (cfg.air_acceleration * 0.5)

        curr_vx = float(state.velocity[0])
        curr_vz = float(state.velocity[2])

        diff_x = target_vx - curr_vx
        diff_z = target_vz - curr_vz
        diff_len = math.sqrt(diff_x * diff_x + diff_z * diff_z)

        if diff_len > 1e-4:
            rate = accel if target_speed > 0.0 else decel
            max_change = rate * dt
            if diff_len <= max_change:
                curr_vx = target_vx
                curr_vz = target_vz
            else:
                inv_diff = 1.0 / diff_len
                curr_vx += diff_x * inv_diff * max_change
                curr_vz += diff_z * inv_diff * max_change

        state.velocity[0] = curr_vx
        state.velocity[2] = curr_vz

        # Update character orientation towards movement direction if moving with smooth angular damping
        if dir_len > 0.1:
            target_yaw = math.degrees(math.atan2(-dir_x, -dir_z))
            diff = (target_yaw - state.yaw_deg + 180.0) % 360.0 - 180.0
            max_turn = 720.0 * dt  # 720 deg/s smooth angular damping
            if abs(diff) <= max_turn:
                state.yaw_deg = target_yaw
            else:
                state.yaw_deg += math.copysign(max_turn, diff)

        # 3. Vertical Kinematics & Jumping
        curr_vy = float(state.velocity[1])

        if state.is_grounded:
            if jump_requested:
                curr_vy = cfg.jump_force
                state.is_grounded = False
                state.is_jumping = True
                self.distance_traversed = 0.0
            else:
                # Slight downward bias to stick firmly to slopes and stairs
                curr_vy = -0.5
                state.is_jumping = False
        else:
            curr_vy += cfg.gravity * dt
            # Clamp terminal falling velocity
            if curr_vy < -45.0:
                curr_vy = -45.0

        state.velocity[1] = curr_vy

        # 4. Submit Desired Translation to Native Rapier KCC
        desired_dx = curr_vx * dt
        desired_dy = curr_vy * dt
        desired_dz = curr_vz * dt

        effective_delta, grounded, sliding = self.physics.move_character(
            self.entity_id,
            (desired_dx, desired_dy, desired_dz),
            dt,
        )

        # 5. Reconcile Grounded State and Velocity
        state.is_grounded = grounded
        state.is_sliding = sliding

        eff_dx, eff_dy, eff_dz = effective_delta
        state.velocity[0] = eff_dx / dt
        state.velocity[2] = eff_dz / dt

        if grounded and not state.is_jumping and eff_dy >= -0.01:
            state.velocity[1] = 0.0
        else:
            state.velocity[1] = eff_dy / dt

        # 6. Synchronize Updated Position & Rotation into Rapier and ECS Contiguous Buffer
        transform = self.physics.world.get_transform(self.entity_id)
        pos_x, pos_y, pos_z = transform[0], transform[1], transform[2]

        half_yaw = math.radians(state.yaw_deg) * 0.5
        qy = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        # Synchronize rotation into Rapier rigid body so sync_to_ecs doesn't clobber it
        self.physics.set_transform(self.entity_id, (pos_x, pos_y, pos_z), (0.0, qy, 0.0, qw))

        dense_idx = self.ecs.pool.get_dense_index(self.entity_id)
        if dense_idx >= 0:
            rbs = self.ecs.rigid_body_state
            rbs[1, dense_idx, 0] = pos_x
            rbs[1, dense_idx, 1] = pos_y
            rbs[1, dense_idx, 2] = pos_z
            rbs[1, dense_idx, 3] = 0.0
            rbs[1, dense_idx, 4] = qy
            rbs[1, dense_idx, 5] = 0.0
            rbs[1, dense_idx, 6] = qw

        # 7. Footstep cadence tracking
        if state.is_grounded and not state.is_jumping:
            h_dist = math.sqrt(eff_dx * eff_dx + eff_dz * eff_dz)
            self.distance_traversed += h_dist
            stride = self.run_stride if is_running else self.walk_stride
            while self.distance_traversed >= stride:
                self.distance_traversed -= stride
                if self.on_footstep is not None:
                    feet_y = pos_y - cfg.capsule_half_height - cfg.capsule_radius
                    in_water = feet_y <= (self.water_height + 0.1)
                    self.on_footstep((pos_x, feet_y, pos_z), in_water)


