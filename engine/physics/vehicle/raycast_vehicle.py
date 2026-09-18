"""Authoritative Raycast Vehicle Physics Controller for PyMordial Engine.

Implements a high-performance, zero-allocation 4-wheel raycast vehicle:
- Downward suspension raycasts with Hooke's Law spring-damping
- Dynamic anti-roll bar force transfer across axles
- Lateral tire friction counteracting slip angle, modulated by surface grip (dry vs wet)
- Longitudinal powertrain traction (RWD, FWD, AWD), braking, and handbrake drift slip
- Contact-point impulse and torque application directly to Rapier rigid body
- Pre-allocated scratch buffers for zero memory allocation in 60 Hz simulation loop
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np

from engine.physics.vehicle.vehicle_config import VehicleConfig, DriveType

if TYPE_CHECKING:
    from engine.core.ecs import EntityManager
    from engine.physics.rapier_world import PhysicsManager

_GEAR_RATIOS = (3.6, 2.1, 1.45, 1.05, 0.82)
_REV_RATIO = 3.5
_FINAL_DRIVE = 3.7


def _quat_to_rotation_matrix(q: np.ndarray, out: np.ndarray) -> None:
    """Converts quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix in place."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    out[0, 0] = 1.0 - (yy + zz)
    out[0, 1] = xy - wz
    out[0, 2] = xz + wy

    out[1, 0] = xy + wz
    out[1, 1] = 1.0 - (xx + zz)
    out[1, 2] = yz - wx

    out[2, 0] = xz - wy
    out[2, 1] = yz + wx
    out[2, 2] = 1.0 - (xx + yy)


class WheelRuntimeState:
    """Mutable runtime state for an individual raycast wheel."""

    __slots__ = (
        "suspension_length",
        "prev_suspension_length",
        "compression",
        "normal_force",
        "is_grounded",
        "hit_point",
        "hit_normal",
        "steer_angle",
        "spin_angle",
        "angular_velocity",
        "world_pos",
        "lateral_slip",
        "longitudinal_slip",
        "slip_ratio",
        "slip_angle",
        "skidding",
    )

    def __init__(self, rest_length: float) -> None:
        self.suspension_length: float = rest_length
        self.prev_suspension_length: float = rest_length
        self.compression: float = 0.0
        self.normal_force: float = 0.0
        self.is_grounded: bool = False
        self.hit_point = np.zeros(3, dtype=np.float32)
        self.hit_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.steer_angle: float = 0.0
        self.spin_angle: float = 0.0
        self.angular_velocity: float = 0.0
        self.world_pos = np.zeros(3, dtype=np.float32)
        self.lateral_slip: float = 0.0
        self.longitudinal_slip: float = 0.0
        self.slip_ratio: float = 0.0
        self.slip_angle: float = 0.0
        self.skidding: bool = False


class RaycastVehicle:
    """Physics-driven raycast vehicle controller executing on fixed simulation ticks."""

    __slots__ = (
        "entity_id",
        "config",
        "physics",
        "ecs",
        "wheel_states",
        # Input states
        "throttle",
        "steering_input",
        "brake_input",
        "handbrake",
        "surface_friction_mult",
        # Telemetry
        "current_speed_mps",
        "current_steer_angle",
        # Realistic Powertrain & Telemetry
        "current_gear",
        "engine_rpm",
        "turbo_boost",
        "turbo_blowoff_triggered",
        "drift_angle_deg",
        "is_drifting",
        "_prev_fwd_speed",
        "_prev_lat_speed",
        "_shift_timer",
        "_accel_x_filtered",
        "_accel_y_filtered",
        "_prev_throttle",
        # Pre-allocated scratch buffers
        "_rot_mat",
        "_chassis_pos",
        "_chassis_rot",
        "_chassis_linvel",
        "_chassis_angvel",
        "_forward_vec",
        "_right_vec",
        "_up_vec",
        "_wheel_attach_world",
        "_wheel_forward",
        "_wheel_right",
        "_contact_point",
        "_contact_vel",
        "_r_vec",
        "_total_force",
        "_total_torque",
        "_impulse_scratch",
        "_torque_scratch",
    )

    def __init__(
        self,
        entity_id: int,
        config: VehicleConfig,
        physics: PhysicsManager,
        ecs: EntityManager,
    ) -> None:
        self.entity_id = entity_id
        self.config = config
        self.physics = physics
        self.ecs = ecs

        # Wheel runtime states
        self.wheel_states: list[WheelRuntimeState] = [
            WheelRuntimeState(w.suspension_rest_length) for w in config.wheels
        ]

        # Driver controls
        self.throttle: float = 0.0
        self.steering_input: float = 0.0
        self.brake_input: float = 0.0
        self.handbrake: bool = False
        self.surface_friction_mult: float = config.surface_friction_mult

        # Telemetry
        self.current_speed_mps: float = 0.0
        self.current_steer_angle: float = 0.0

        # Realistic Powertrain & Telemetry
        self.current_gear: int = 1
        self.engine_rpm: float = 800.0
        self.turbo_boost: float = 0.0
        self.turbo_blowoff_triggered: bool = False
        self.drift_angle_deg: float = 0.0
        self.is_drifting: bool = False
        self._prev_fwd_speed: float = 0.0
        self._prev_lat_speed: float = 0.0
        self._shift_timer: float = 0.0
        self._accel_x_filtered: float = 0.0
        self._accel_y_filtered: float = 0.0
        self._prev_throttle: float = 0.0

        # Pre-allocated buffers for zero allocations in fixed_update
        self._rot_mat = np.zeros((3, 3), dtype=np.float32)
        self._chassis_pos = np.zeros(3, dtype=np.float32)
        self._chassis_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._chassis_linvel = np.zeros(3, dtype=np.float32)
        self._chassis_angvel = np.zeros(3, dtype=np.float32)
        self._forward_vec = np.zeros(3, dtype=np.float32)
        self._right_vec = np.zeros(3, dtype=np.float32)
        self._up_vec = np.zeros(3, dtype=np.float32)
        self._wheel_attach_world = np.zeros(3, dtype=np.float32)
        self._wheel_forward = np.zeros(3, dtype=np.float32)
        self._wheel_right = np.zeros(3, dtype=np.float32)
        self._contact_point = np.zeros(3, dtype=np.float32)
        self._contact_vel = np.zeros(3, dtype=np.float32)
        self._r_vec = np.zeros(3, dtype=np.float32)
        self._total_force = np.zeros(3, dtype=np.float32)
        self._total_torque = np.zeros(3, dtype=np.float32)
        self._impulse_scratch = np.zeros(3, dtype=np.float32)
        self._torque_scratch = np.zeros(3, dtype=np.float32)

    def set_throttle(self, value: float) -> None:
        """Sets forward/reverse throttle [-1.0, 1.0]."""
        self.throttle = max(-1.0, min(1.0, float(value)))

    def set_steering(self, value: float) -> None:
        """Sets steering input [-1.0, 1.0]."""
        self.steering_input = max(-1.0, min(1.0, float(value)))

    def set_brake(self, value: float) -> None:
        """Sets braking force [0.0, 1.0]."""
        self.brake_input = max(0.0, min(1.0, float(value)))

    def set_handbrake(self, value: bool) -> None:
        """Locks rear wheels for drifting."""
        self.handbrake = bool(value)

    @property
    def position(self) -> tuple[float, float, float]:
        """Returns current world position of the chassis center of mass."""
        pool = self.ecs.pool
        dense_idx = pool.get_dense_index(self.entity_id)
        if dense_idx >= 0:
            rb_state = self.ecs.rigid_body_state[1]
            return (float(rb_state[dense_idx, 0]), float(rb_state[dense_idx, 1]), float(rb_state[dense_idx, 2]))
        return (float(self._chassis_pos[0]), float(self._chassis_pos[1]), float(self._chassis_pos[2]))

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Returns current linear velocity of the chassis."""
        vx, vy, vz = self.physics.get_linvel(self.entity_id)
        return (float(vx), float(vy), float(vz))

    def set_surface_friction(self, mult: float) -> None:
        """Scales tire surface grip (e.g. 1.0 for dry road, 0.65 for wet rain road)."""
        self.surface_friction_mult = max(0.1, float(mult))

    def update(self, dt: float) -> None:
        """Fixed simulation step: computes suspension raycasts and wheel traction."""
        if dt <= 0.0:
            return

        pool = self.ecs.pool
        dense_idx = pool.get_dense_index(self.entity_id)
        if dense_idx < 0:
            return

        # 1. Fetch current chassis state from ECS and PhysicsManager
        rb_state = self.ecs.rigid_body_state[1]
        self._chassis_pos[0] = rb_state[dense_idx, 0]
        self._chassis_pos[1] = rb_state[dense_idx, 1]
        self._chassis_pos[2] = rb_state[dense_idx, 2]

        self._chassis_rot[0] = rb_state[dense_idx, 3]
        self._chassis_rot[1] = rb_state[dense_idx, 4]
        self._chassis_rot[2] = rb_state[dense_idx, 5]
        self._chassis_rot[3] = rb_state[dense_idx, 6]

        vx, vy, vz = self.physics.get_linvel(self.entity_id)
        self._chassis_linvel[0] = vx
        self._chassis_linvel[1] = vy
        self._chassis_linvel[2] = vz

        wx, wy, wz = self.physics.get_angvel(self.entity_id)
        self._chassis_angvel[0] = wx
        self._chassis_angvel[1] = wy
        self._chassis_angvel[2] = wz

        # 2. Extract Chassis Orientation Vectors
        _quat_to_rotation_matrix(self._chassis_rot, self._rot_mat)
        # Right (+X)
        self._right_vec[0] = self._rot_mat[0, 0]
        self._right_vec[1] = self._rot_mat[1, 0]
        self._right_vec[2] = self._rot_mat[2, 0]
        # Up (+Y)
        self._up_vec[0] = self._rot_mat[0, 1]
        self._up_vec[1] = self._rot_mat[1, 1]
        self._up_vec[2] = self._rot_mat[2, 1]
        # Forward (-Z in PyMordial coordinate system)
        self._forward_vec[0] = -self._rot_mat[0, 2]
        self._forward_vec[1] = -self._rot_mat[1, 2]
        self._forward_vec[2] = -self._rot_mat[2, 2]

        # Vehicle planar speed along forward and right vectors (direct float arithmetic, zero allocation)
        fwd_speed = float(
            self._chassis_linvel[0] * self._forward_vec[0]
            + self._chassis_linvel[1] * self._forward_vec[1]
            + self._chassis_linvel[2] * self._forward_vec[2]
        )
        lat_speed = float(
            self._chassis_linvel[0] * self._right_vec[0]
            + self._chassis_linvel[1] * self._right_vec[1]
            + self._chassis_linvel[2] * self._right_vec[2]
        )
        self.current_speed_mps = fwd_speed

        # 2-Axis dynamic acceleration tracking (longitudinal + lateral)
        raw_accel_x = (fwd_speed - self._prev_fwd_speed) / dt if dt > 0.0 else 0.0
        raw_accel_y = (lat_speed - self._prev_lat_speed) / dt if dt > 0.0 else 0.0
        self._prev_fwd_speed = fwd_speed
        self._prev_lat_speed = lat_speed

        accel_alpha = min(1.0, dt * 8.0)
        self._accel_x_filtered += (raw_accel_x - self._accel_x_filtered) * accel_alpha
        self._accel_y_filtered += (raw_accel_y - self._accel_y_filtered) * accel_alpha

        # Pitch weight transfer (dive on braking, squat on acceleration)
        wheelbase = 2.7
        track_width = 1.7
        cg_height = 0.38
        pitch_weight_transfer = self.config.chassis_mass * self._accel_x_filtered * (cg_height / wheelbase)
        max_pitch_wt = self.config.chassis_mass * 9.81 * 0.45
        pitch_weight_transfer = max(-max_pitch_wt, min(max_pitch_wt, pitch_weight_transfer))

        # Roll weight transfer (lateral load shift to outside wheels during cornering)
        lat_weight_transfer = self.config.chassis_mass * self._accel_y_filtered * (cg_height / track_width)
        max_roll_wt = self.config.chassis_mass * 9.81 * 0.45
        lat_weight_transfer = max(-max_roll_wt, min(max_roll_wt, lat_weight_transfer))

        # Powertrain 5-speed transmission simulation
        gear_ratios = _GEAR_RATIOS
        rev_ratio = _REV_RATIO
        final_drive = _FINAL_DRIVE

        if self._shift_timer > 0.0:
            self._shift_timer -= dt

        # Reverse vs Forward gear selection
        if self.throttle < -0.05 and fwd_speed < 1.0:
            self.current_gear = -1
        elif self.current_gear == -1 and fwd_speed > -0.5 and self.throttle >= 0.0:
            self.current_gear = 1

        curr_ratio = rev_ratio * final_drive if self.current_gear == -1 else gear_ratios[self.current_gear - 1] * final_drive

        # Calculate wheel-based RPM
        num_wheels = len(self.config.wheels)
        driven_wheel_speed = abs(fwd_speed)
        wheel_rad = max(0.1, self.config.wheels[2].radius if num_wheels > 2 else 0.35)
        rpm_from_speed = (driven_wheel_speed / wheel_rad) * curr_ratio * (60.0 / (2.0 * math.pi))

        # Smooth clutch / torque converter slip blend
        clutch_slip = max(0.0, 1.0 - driven_wheel_speed / 8.0)
        target_rpm = max(800.0, rpm_from_speed + abs(self.throttle) * 2800.0 * clutch_slip)

        # Smooth RPM response
        rpm_rate = 14.0 if abs(self.throttle) > 0.1 else 8.0
        self.engine_rpm += (target_rpm - self.engine_rpm) * min(1.0, dt * rpm_rate)
        self.engine_rpm = max(800.0, min(7200.0, self.engine_rpm))

        # Automatic transmission shifting logic (load-dependent shift threshold)
        if self._shift_timer <= 0.0 and self.current_gear > 0:
            upshift_rpm = 3400.0 + abs(self.throttle) * 2200.0
            if self.engine_rpm > upshift_rpm and self.current_gear < 5 and fwd_speed > 4.0:
                self.current_gear += 1
                self._shift_timer = 0.20
                self.engine_rpm *= 0.68
            elif self.engine_rpm < 2200.0 and self.current_gear > 1 and fwd_speed < 45.0:
                self.current_gear -= 1
                self._shift_timer = 0.15
                self.engine_rpm = min(6500.0, self.engine_rpm * 1.35)

        # Turbo Boost Dynamics
        self.turbo_blowoff_triggered = False
        if abs(self.throttle) > 0.4 and self.engine_rpm > 2200.0:
            self.turbo_boost = min(1.2, self.turbo_boost + dt * 1.8)
        else:
            if self._prev_throttle > 0.5 and abs(self.throttle) < 0.15 and self.turbo_boost > 0.25:
                self.turbo_blowoff_triggered = True
            self.turbo_boost = max(0.0, self.turbo_boost - dt * 3.5)
        self._prev_throttle = self.throttle

        # 3. Smooth steering interpolation with speed-sensitive dynamic lock
        speed_abs = abs(fwd_speed)
        speed_factor = 1.0 / (1.0 + speed_abs * 0.035)
        min_steer_reduction = getattr(self.config, "high_speed_steer_reduction", 0.45)
        effective_max_steer = self.config.max_steer_angle_rad * max(min_steer_reduction, speed_factor)
        target_steer = self.steering_input * effective_max_steer
        steer_diff = target_steer - self.current_steer_angle
        max_steer_step = self.config.steer_speed * dt
        if abs(steer_diff) <= max_steer_step:
            self.current_steer_angle = target_steer
        else:
            self.current_steer_angle += math.copysign(max_steer_step, steer_diff)

        # 4. Suspension Raycasting and Spring Forces
        # Blend local chassis down with world down for stability during body roll
        down_x = -self._up_vec[0] * 0.65
        down_y = -self._up_vec[1] * 0.65 - 0.35
        down_z = -self._up_vec[2] * 0.65
        down_len = math.sqrt(down_x * down_x + down_y * down_y + down_z * down_z)
        if down_len > 1e-6:
            inv_l = 1.0 / down_len
            down_x *= inv_l
            down_y *= inv_l
            down_z *= inv_l

        # Half-height margin to ensure ray origin starts outside chassis box
        chassis_half_y = self.config.chassis_size[1] * 0.5 + 0.05

        for i in range(num_wheels):
            w_cfg = self.config.wheels[i]
            w_state = self.wheel_states[i]
            w_state.prev_suspension_length = w_state.suspension_length

            # Wheel attachment point in world space
            # offset: (ox, oy, oz) where oz is along -Z forward
            ox, oy, oz = w_cfg.offset
            self._wheel_attach_world[0] = (
                self._chassis_pos[0]
                + self._right_vec[0] * ox
                + self._up_vec[0] * oy
                - self._forward_vec[0] * oz  # offset z is forward (+z in vehicle config is rear, -z is front)
            )
            self._wheel_attach_world[1] = (
                self._chassis_pos[1]
                + self._right_vec[1] * ox
                + self._up_vec[1] * oy
                - self._forward_vec[1] * oz
            )
            self._wheel_attach_world[2] = (
                self._chassis_pos[2]
                + self._right_vec[2] * ox
                + self._up_vec[2] * oy
                - self._forward_vec[2] * oz
            )

            # Ray origin starts safely just below the bottom of the chassis collider
            d_to_bottom = max(0.01, chassis_half_y + oy)
            ray_start_offset = d_to_bottom + 0.01

            ray_ox = self._wheel_attach_world[0] + down_x * ray_start_offset
            ray_oy = self._wheel_attach_world[1] + down_y * ray_start_offset
            ray_oz = self._wheel_attach_world[2] + down_z * ray_start_offset

            max_ray_dist = w_cfg.suspension_rest_length + w_cfg.radius

            hit = self.physics.cast_ray_raw(
                ray_ox,
                ray_oy,
                ray_oz,
                down_x,
                down_y,
                down_z,
                max_distance=max_ray_dist,
                solid=True,
            )

            if hit is not None and hit[0] != self.entity_id:
                _hit_ent, dist_from_ray_origin, nx, ny, nz = hit
                effective_dist = dist_from_ray_origin + ray_start_offset
                susp_len = max(0.0, effective_dist - w_cfg.radius)

                if susp_len <= w_cfg.suspension_rest_length + 1e-3:
                    w_state.is_grounded = True
                    w_state.suspension_length = susp_len
                    w_state.compression = min(
                        w_cfg.max_compression,
                        w_cfg.suspension_rest_length - susp_len,
                    )
                    w_state.hit_point[0] = ray_ox + down_x * dist_from_ray_origin
                    w_state.hit_point[1] = ray_oy + down_y * dist_from_ray_origin
                    w_state.hit_point[2] = ray_oz + down_z * dist_from_ray_origin
                    # Guard degenerate normal vector
                    if abs(nx) < 1e-4 and abs(ny) < 1e-4 and abs(nz) < 1e-4:
                        nx, ny, nz = 0.0, 1.0, 0.0
                    w_state.hit_normal[0] = nx
                    w_state.hit_normal[1] = ny
                    w_state.hit_normal[2] = nz

                    # Spring + Damper force (Progressive Bump-Stop + Asymmetric Rebound Damping)
                    v_susp = (w_state.prev_suspension_length - susp_len) / dt

                    bump_threshold = getattr(w_cfg, "bump_stop_threshold", 0.78) * w_cfg.max_compression
                    if w_state.compression > bump_threshold:
                        excess = (w_state.compression - bump_threshold) / max(0.005, w_cfg.max_compression - bump_threshold)
                        bump_mult = 1.0 + (getattr(w_cfg, "bump_stop_stiffness_mult", 3.5) - 1.0) * min(1.0, excess * excess)
                        spring_force = w_cfg.spring_stiffness * bump_mult * w_state.compression
                    else:
                        spring_force = w_cfg.spring_stiffness * w_state.compression

                    rebound_mult = getattr(self.config, "suspension_rebound_damping_factor", 1.40)
                    damper_coeff = w_cfg.spring_damping * (rebound_mult if v_susp < 0.0 else 1.0)
                    damper_force = damper_coeff * v_susp
                    w_state.normal_force = max(0.0, spring_force + damper_force)
                else:
                    w_state.is_grounded = False
                    w_state.suspension_length = w_cfg.suspension_rest_length
                    w_state.compression = 0.0
                    w_state.normal_force = 0.0
            else:
                w_state.is_grounded = False
                w_state.suspension_length = w_cfg.suspension_rest_length
                w_state.compression = 0.0
                w_state.normal_force = 0.0

            # Store current visual wheel world position
            w_state.world_pos[0] = self._wheel_attach_world[0] + down_x * w_state.suspension_length
            w_state.world_pos[1] = self._wheel_attach_world[1] + down_y * w_state.suspension_length
            w_state.world_pos[2] = self._wheel_attach_world[2] + down_z * w_state.suspension_length

            if w_state.is_grounded:
                # Guarantee tire bottom never penetrates below the ground contact surface
                min_center_y = w_state.hit_point[1] + w_cfg.radius
                if w_state.world_pos[1] < min_center_y:
                    w_state.world_pos[1] = min_center_y

            if w_cfg.is_steerable:
                w_state.steer_angle = self.current_steer_angle
            else:
                w_state.steer_angle = 0.0

        # 5. Anti-Roll Bar Stabilization & 2-Axis Dynamic Weight Transfer (Pitch & Roll)
        if num_wheels >= 4:
            f_pitch = pitch_weight_transfer * 0.5
            f_roll = lat_weight_transfer * 0.5

            # Front-Left (0): loses pitch on accel, unloads/loads with roll
            if self.wheel_states[0].is_grounded:
                self.wheel_states[0].normal_force = max(0.0, self.wheel_states[0].normal_force - f_pitch - f_roll)
            # Front-Right (1): loses pitch on accel, opposite roll
            if self.wheel_states[1].is_grounded:
                self.wheel_states[1].normal_force = max(0.0, self.wheel_states[1].normal_force - f_pitch + f_roll)
            # Rear-Left (2): gains pitch on accel, unloads/loads with roll
            if self.wheel_states[2].is_grounded:
                self.wheel_states[2].normal_force = max(0.0, self.wheel_states[2].normal_force + f_pitch - f_roll)
            # Rear-Right (3): gains pitch on accel, opposite roll
            if self.wheel_states[3].is_grounded:
                self.wheel_states[3].normal_force = max(0.0, self.wheel_states[3].normal_force + f_pitch + f_roll)

            # Front Axle ARB
            f_diff = self.wheel_states[0].compression - self.wheel_states[1].compression
            f_arb = f_diff * self.config.anti_roll_stiffness
            if self.wheel_states[0].is_grounded:
                self.wheel_states[0].normal_force = max(0.0, self.wheel_states[0].normal_force + f_arb)
            if self.wheel_states[1].is_grounded:
                self.wheel_states[1].normal_force = max(0.0, self.wheel_states[1].normal_force - f_arb)

            # Rear Axle ARB
            r_diff = self.wheel_states[2].compression - self.wheel_states[3].compression
            r_arb = r_diff * self.config.anti_roll_stiffness
            if self.wheel_states[2].is_grounded:
                self.wheel_states[2].normal_force = max(0.0, self.wheel_states[2].normal_force + r_arb)
            if self.wheel_states[3].is_grounded:
                self.wheel_states[3].normal_force = max(0.0, self.wheel_states[3].normal_force - r_arb)

        # 6. Apply Forces: Suspension, Traction, Braking, and Lateral Tire Friction (Kamm's Circle)
        effective_surface_grip = self.config.tire_grip * self.surface_friction_mult
        fz_nominal = (self.config.chassis_mass * 9.81) / max(1, num_wheels)
        load_sens = getattr(self.config, "tire_load_sensitivity", 0.15)
        brake_bias_f = getattr(self.config, "brake_bias_front", 0.65)
        caster_torque_factor = getattr(self.config, "caster_aligning_torque", 0.08)

        for i in range(num_wheels):
            w_cfg = self.config.wheels[i]
            w_state = self.wheel_states[i]

            if not w_state.is_grounded or w_state.normal_force <= 0.0:
                continue

            fn = w_state.normal_force

            # Tire Load Sensitivity: heavier loaded tires produce slightly diminishing returns
            load_ratio = fn / max(100.0, fz_nominal)
            load_factor = max(0.70, min(1.30, 1.0 - load_sens * (load_ratio - 1.0)))
            tire_grip_eff = effective_surface_grip * load_factor
            f_traction_max = fn * tire_grip_eff

            # A. Suspension Upward Force
            fx_susp = self._up_vec[0] * (fn * 0.25)
            fy_susp = fn
            fz_susp = self._up_vec[2] * (fn * 0.25)

            # B. Wheel Steering Directions
            steer = w_state.steer_angle
            cos_s = math.cos(steer)
            sin_s = math.sin(steer)

            self._wheel_forward[0] = self._forward_vec[0] * cos_s + self._right_vec[0] * sin_s
            self._wheel_forward[1] = self._forward_vec[1] * cos_s + self._right_vec[1] * sin_s
            self._wheel_forward[2] = self._forward_vec[2] * cos_s + self._right_vec[2] * sin_s

            self._wheel_right[0] = self._right_vec[0] * cos_s - self._forward_vec[0] * sin_s
            self._wheel_right[1] = self._right_vec[1] * cos_s - self._forward_vec[1] * sin_s
            self._wheel_right[2] = self._right_vec[2] * cos_s - self._forward_vec[2] * sin_s

            # Contact point relative to chassis center of mass
            self._r_vec[0] = w_state.hit_point[0] - self._chassis_pos[0]
            self._r_vec[1] = w_state.hit_point[1] - self._chassis_pos[1]
            self._r_vec[2] = w_state.hit_point[2] - self._chassis_pos[2]

            # Linear velocity at contact point: v_contact = v_chassis + w x r
            cx = self._chassis_angvel[1] * self._r_vec[2] - self._chassis_angvel[2] * self._r_vec[1]
            cy = self._chassis_angvel[2] * self._r_vec[0] - self._chassis_angvel[0] * self._r_vec[2]
            cz = self._chassis_angvel[0] * self._r_vec[1] - self._chassis_angvel[1] * self._r_vec[0]

            self._contact_vel[0] = self._chassis_linvel[0] + cx
            self._contact_vel[1] = self._chassis_linvel[1] + cy
            self._contact_vel[2] = self._chassis_linvel[2] + cz

            v_forward = float(np.dot(self._contact_vel, self._wheel_forward))
            v_lateral = float(np.dot(self._contact_vel, self._wheel_right))

            # Wheel spin update
            w_state.angular_velocity = v_forward / max(0.05, w_cfg.radius)
            w_state.spin_angle += w_state.angular_velocity * dt

            # Slip angles and slip ratios
            slip_angle = -math.atan2(v_lateral, max(0.5, abs(v_forward)))
            slip_ratio = (w_state.angular_velocity * w_cfg.radius - v_forward) / max(1.0, abs(v_forward))

            w_state.lateral_slip = abs(v_lateral)
            w_state.longitudinal_slip = abs(slip_ratio)
            w_state.slip_ratio = slip_ratio
            w_state.slip_angle = slip_angle
            w_state.skidding = (abs(v_lateral) > 2.0) or (self.handbrake and abs(v_forward) > 2.0) or (abs(slip_ratio) > 0.5)

            # 1. Compute Longitudinal Force (Powertrain Drive, Engine Braking, Foot Braking, Handbrake)
            f_long = 0.0

            # Powertrain Drive
            is_driven_wheel = w_cfg.is_driven
            if self.config.drive_type == DriveType.FWD:
                is_driven_wheel = w_cfg.is_steerable
            elif self.config.drive_type == DriveType.RWD:
                is_driven_wheel = not w_cfg.is_steerable
            elif self.config.drive_type == DriveType.AWD:
                is_driven_wheel = True

            num_driven = 2 if self.config.drive_type in (DriveType.RWD, DriveType.FWD) else num_wheels
            gear_ratio_val = rev_ratio if self.current_gear == -1 else gear_ratios[self.current_gear - 1]
            gear_mult = max(0.50, min(1.35, 0.45 + 0.55 * (gear_ratio_val / 3.6)))
            if self._shift_timer > 0.0:
                gear_mult *= 0.20
            boost_mult = 1.0 + self.turbo_boost * 0.32

            if is_driven_wheel and abs(self.throttle) > 0.01:
                if self.throttle > 0.0:
                    if fwd_speed < self.config.top_speed_mps:
                        speed_ratio = max(0.0, fwd_speed / self.config.top_speed_mps)
                        power_curve = 1.0 - speed_ratio * speed_ratio
                        f_long += (
                            self.throttle
                            * self.config.engine_torque
                            * power_curve
                            * gear_mult
                            * boost_mult
                            * (1.0 / num_driven)
                        )
                else:
                    # Reverse
                    if fwd_speed > -15.0:
                        f_long += self.throttle * self.config.reverse_torque * (1.0 / num_driven)

            # Engine braking when off-throttle
            if abs(self.throttle) < 0.02 and abs(v_forward) > 1.0 and not self.handbrake:
                engine_brake = 350.0 * (self.engine_rpm / 3500.0) * (1.0 / num_wheels)
                f_long -= math.copysign(engine_brake, v_forward)

            # Foot Braking with realistic Front/Rear Brake Bias
            if self.brake_input > 0.01 and abs(v_forward) > 0.05:
                bias_ratio = brake_bias_f if w_cfg.is_steerable else (1.0 - brake_bias_f)
                brake_f = self.brake_input * self.config.brake_torque * (bias_ratio * 2.0 / num_wheels) * w_cfg.brake_ratio
                f_long -= math.copysign(min(brake_f, abs(v_forward) * 2000.0), v_forward)

            # Handbrake on rear wheels
            if not w_cfg.is_steerable and self.handbrake and abs(v_forward) > 0.05:
                hb_f = self.config.handbrake_torque * (1.0 / num_wheels)
                f_long -= math.copysign(min(hb_f, abs(v_forward) * 3000.0), v_forward)

            # Clamp longitudinal force to total traction circle
            f_long = max(-f_traction_max, min(f_traction_max, f_long))

            # 2. Kamm's Friction Circle: Available lateral force budget after longitudinal traction
            f_lat_cap_sq = max(0.0, f_traction_max * f_traction_max - f_long * f_long)
            f_lat_budget = math.sqrt(f_lat_cap_sq)

            # Handbrake specifically induces rear-axle sliding breakaway
            if not w_cfg.is_steerable and self.handbrake:
                f_lat_budget = min(f_lat_budget, fn * tire_grip_eff * self.config.drift_friction_factor)

            # 3. Progressive Pacejka Brush Slip Model
            s_norm = abs(math.tan(slip_angle)) / 0.18
            if s_norm < 1.0:
                pacejka_factor = math.sin(s_norm * 1.570796)
            else:
                # Sliding dynamic friction plateau (~78% of peak)
                pacejka_factor = 1.0 - 0.22 * min(1.0, (s_norm - 1.0) / 2.0)

            max_lat_force = f_lat_budget * pacejka_factor
            desired_lateral_force = -v_lateral * (self.config.chassis_mass / num_wheels) / max(0.01, dt * 2.0)
            f_lat = max(-max_lat_force, min(max_lat_force, desired_lateral_force))

            # Small velocity deadzone to eliminate zero-speed oscillation
            if abs(v_lateral) < 0.08 and abs(v_forward) < 0.1:
                f_lat *= (abs(v_lateral) / 0.08)

            # 4. Counter-Steering Caster Aligning Torque from Pneumatic Trail
            aligning_torque = -f_lat * caster_torque_factor
            if self.is_drifting and w_cfg.is_steerable:
                drift_counter_mult = getattr(self.config, "drift_counter_steer_assist", 0.25)
                aligning_torque -= math.radians(self.drift_angle_deg) * drift_counter_mult * 150.0

            # E. Aggregate Total Contact Force (Suspension + Longitudinal + Lateral)
            f_total_x = fx_susp + self._wheel_forward[0] * f_long + self._wheel_right[0] * f_lat
            f_total_y = fy_susp + self._wheel_forward[1] * f_long + self._wheel_right[1] * f_lat
            f_total_z = fz_susp + self._wheel_forward[2] * f_long + self._wheel_right[2] * f_lat

            # Convert Force to Impulse: J = F * dt
            self._impulse_scratch[0] = f_total_x * dt
            self._impulse_scratch[1] = f_total_y * dt
            self._impulse_scratch[2] = f_total_z * dt

            # Torque Impulse: Tau = r x J + aligning torque on yaw
            rx, ry, rz = self._r_vec[0], self._r_vec[1], self._r_vec[2]
            jx, jy, jz = self._impulse_scratch[0], self._impulse_scratch[1], self._impulse_scratch[2]

            self._torque_scratch[0] = ry * jz - rz * jy
            self._torque_scratch[1] = (rz * jx - rx * jz) + aligning_torque * dt
            self._torque_scratch[2] = rx * jy - ry * jx

            # Apply to Rapier rigid body
            self.physics.apply_impulse(self.entity_id, self._impulse_scratch)
            self.physics.apply_torque_impulse(self.entity_id, self._torque_scratch)

        # 7. Active Upright Self-Righting Stabilizer
        # Rapidly restores chassis roll and pitch back to level after cornering or curb impacts
        grounded_count = sum(1 for w in self.wheel_states if w.is_grounded)
        if grounded_count >= 1 and self._up_vec[1] > 0.20:
            # Cross product: local_up x world_up (0, 1, 0) = (-uz, 0, ux)
            tau_x = -self._up_vec[2] * self.config.upright_stiffness
            tau_z = self._up_vec[0] * self.config.upright_stiffness

            # Roll/pitch angular velocity damping (leaving yaw wy completely free for steering)
            tau_x -= self._chassis_angvel[0] * self.config.upright_damping
            tau_z -= self._chassis_angvel[2] * self.config.upright_damping

            self._torque_scratch[0] = tau_x * dt
            self._torque_scratch[1] = 0.0
            self._torque_scratch[2] = tau_z * dt
            self.physics.apply_torque_impulse(self.entity_id, self._torque_scratch)

        # 8. Road Holding Plant & Aerodynamic Downforce (Gives vehicle heavy, grounded asphalt feel)
        if grounded_count >= 2:
            # Baseline road adhesion plant proportional to chassis mass (plants the vehicle firmly)
            base_plant = self.config.chassis_mass * 4.2
            # Speed-dependent aerodynamic downforce
            aero_downforce = min(5500.0, 0.5 * 1.225 * (fwd_speed * fwd_speed) * 0.65) if fwd_speed > 2.0 else 0.0
            total_downforce = base_plant + aero_downforce
            self._impulse_scratch[0] = 0.0
            self._impulse_scratch[1] = -total_downforce * dt
            self._impulse_scratch[2] = 0.0
            self.physics.apply_impulse(self.entity_id, self._impulse_scratch)

        # 9. Drift Telemetry & Counter-Steering Stabilizer
        v_lat_chassis = float(np.dot(self._chassis_linvel, self._right_vec))
        if fwd_speed > 3.0:
            self.drift_angle_deg = math.degrees(math.atan2(v_lat_chassis, fwd_speed))
            self.is_drifting = (abs(self.drift_angle_deg) > 12.0) and (fwd_speed > 5.0)
        else:
            self.drift_angle_deg = 0.0
            self.is_drifting = False

    def get_wheel_transform(self, wheel_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the world position and orientation quaternion for a visual wheel mesh.

        Returns:
            (pos_xyz, rot_xyzw) as NumPy arrays.
        """
        w_state = self.wheel_states[wheel_idx]
        return w_state.world_pos, self._chassis_rot
