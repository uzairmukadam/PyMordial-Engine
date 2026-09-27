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
        "initialized",
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
        self.initialized: bool = False


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
        taper_rate = getattr(self.config, "high_speed_steer_taper", 0.018)
        speed_factor = 1.0 / (1.0 + speed_abs * taper_rate)
        min_steer_reduction = getattr(self.config, "high_speed_steer_reduction", 0.55)
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

        for i in range(num_wheels):
            w_cfg = self.config.wheels[i]
            w_state = self.wheel_states[i]
            w_state.prev_suspension_length = w_state.suspension_length
            max_droop = getattr(w_cfg, "max_droop", 0.12)
            max_extended_len = w_cfg.suspension_rest_length + max_droop

            # Wheel attachment point in world space
            # offset: (ox, oy, oz) where oz is along -Z forward (+z in vehicle config is rear, -z is front)
            ox, oy, oz = w_cfg.offset
            self._wheel_attach_world[0] = (
                self._chassis_pos[0]
                + self._right_vec[0] * ox
                + self._up_vec[0] * oy
                - self._forward_vec[0] * oz
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

            # Cast ray downwards starting directly from the suspension anchor
            # Extended reach allows continuous contact tracking over potholes, dips, curbs, and irregular terrain
            max_ray_dist = w_cfg.suspension_rest_length + w_cfg.radius + 0.35

            curr_ox = self._wheel_attach_world[0]
            curr_oy = self._wheel_attach_world[1]
            curr_oz = self._wheel_attach_world[2]
            accum_dist = 0.0
            rem_dist = max_ray_dist
            final_hit = None

            # Multi-step raycast: bypass vehicle chassis collider to detect ground/terrain without tunneling
            h = self.physics.cast_ray_raw(
                curr_ox,
                curr_oy,
                curr_oz,
                down_x,
                down_y,
                down_z,
                max_distance=rem_dist,
                solid=False,
            )
            if h is not None:
                hit_ent, hit_d, nx, ny, nz = h
                if hit_ent == self.entity_id:
                    # Ray started inside or hit chassis collider: advance past chassis exit point
                    exit_dist = hit_d
                    step = 0.001
                    rem = rem_dist - exit_dist - step
                    if rem > 0.0:
                        # Cast from 1mm below chassis with solid=True to detect terrain and prevent ground tunneling
                        h2 = self.physics.cast_ray_raw(
                            curr_ox + down_x * (exit_dist + step),
                            curr_oy + down_y * (exit_dist + step),
                            curr_oz + down_z * (exit_dist + step),
                            down_x,
                            down_y,
                            down_z,
                            max_distance=rem,
                            solid=True,
                        )
                        if h2 is not None:
                            if h2[0] != self.entity_id:
                                final_hit = (h2[0], exit_dist + step + h2[1], h2[2], h2[3], h2[4])
                            else:
                                # Stepped through secondary chassis face, advance another 4mm
                                step2 = 0.004
                                rem2 = rem - step2
                                if rem2 > 0.0:
                                    h3 = self.physics.cast_ray_raw(
                                        curr_ox + down_x * (exit_dist + step + step2),
                                        curr_oy + down_y * (exit_dist + step + step2),
                                        curr_oz + down_z * (exit_dist + step + step2),
                                        down_x,
                                        down_y,
                                        down_z,
                                        max_distance=rem2,
                                        solid=True,
                                    )
                                    if h3 is not None and h3[0] != self.entity_id:
                                        final_hit = (h3[0], exit_dist + step + step2 + h3[1], h3[2], h3[3], h3[4])
                else:
                    final_hit = (hit_ent, hit_d, nx, ny, nz)

            if final_hit is not None:
                _hit_ent, dist_to_ground, nx, ny, nz = final_hit
                # Guard degenerate normal vector
                if abs(nx) < 1e-4 and abs(ny) < 1e-4 and abs(nz) < 1e-4:
                    nx, ny, nz = 0.0, 1.0, 0.0
                w_state.hit_normal[0] = nx
                w_state.hit_normal[1] = ny
                w_state.hit_normal[2] = nz
                w_state.hit_point[0] = self._wheel_attach_world[0] + down_x * dist_to_ground
                w_state.hit_point[1] = self._wheel_attach_world[1] + down_y * dist_to_ground
                w_state.hit_point[2] = self._wheel_attach_world[2] + down_z * dist_to_ground

                # Measure suspension stroke: hub_dist is the distance from anchor to wheel hub center
                hub_dist = dist_to_ground - w_cfg.radius

                # Schmitt trigger contact hysteresis (12mm) eliminates contact flicker on suspension limit
                ground_threshold = max_extended_len + (0.012 if w_state.is_grounded else 0.0)
                if hub_dist <= ground_threshold:
                    w_state.is_grounded = True
                    min_susp_len = max(0.01, w_cfg.suspension_rest_length - w_cfg.max_compression)
                    susp_len = max(min_susp_len, min(max_extended_len, hub_dist))
                    w_state.suspension_length = susp_len
                    w_state.compression = max(0.0, w_cfg.suspension_rest_length - susp_len)

                    # Spring + Damper force (Progressive Bump-Stop + Asymmetric Rebound Damping + Digressive Blow-Off)
                    if not w_state.initialized:
                        w_state.prev_suspension_length = susp_len
                        w_state.initialized = True

                    v_susp = (w_state.prev_suspension_length - susp_len) / dt
                    # Digressive damper blow-off: prevents shock absorber hydro-lock spikes from launching vehicle
                    v_susp_eff = math.copysign(min(3.5, abs(v_susp)), v_susp)

                    bump_threshold = getattr(w_cfg, "bump_stop_threshold", 0.78) * w_cfg.max_compression
                    if w_state.compression > bump_threshold:
                        excess = (w_state.compression - bump_threshold) / max(0.005, w_cfg.max_compression - bump_threshold)
                        bump_mult = 1.0 + (getattr(w_cfg, "bump_stop_stiffness_mult", 3.5) - 1.0) * min(1.0, excess * excess)
                        spring_force = w_cfg.spring_stiffness * bump_mult * w_state.compression
                    else:
                        spring_force = w_cfg.spring_stiffness * w_state.compression

                    rebound_mult = getattr(self.config, "suspension_rebound_damping_factor", 1.40)
                    damper_coeff = w_cfg.spring_damping * (rebound_mult if v_susp < 0.0 else 1.0)
                    damper_force = damper_coeff * v_susp_eff
                    w_state.normal_force = max(0.0, spring_force + damper_force)
                else:
                    w_state.is_grounded = False
                    # Wheel rests at maximum extended droop limit when airborne, never snapping to rest length
                    w_state.suspension_length = max_extended_len
                    w_state.compression = 0.0
                    w_state.normal_force = 0.0
            else:
                w_state.is_grounded = False
                w_state.suspension_length = max_extended_len
                w_state.compression = 0.0
                w_state.normal_force = 0.0

            # Store current visual wheel world position
            w_state.world_pos[0] = self._wheel_attach_world[0] + down_x * w_state.suspension_length
            w_state.world_pos[1] = self._wheel_attach_world[1] + down_y * w_state.suspension_length
            w_state.world_pos[2] = self._wheel_attach_world[2] + down_z * w_state.suspension_length

            if final_hit is not None:
                # Guarantee tire bottom never penetrates below the ground contact surface
                min_center_y = w_state.hit_point[1] + w_cfg.radius
                if w_state.world_pos[1] < min_center_y:
                    w_state.world_pos[1] = min_center_y

            if w_cfg.is_steerable:
                w_state.steer_angle = self.current_steer_angle
            else:
                w_state.steer_angle = 0.0

        # 5. Anti-Roll Bar Stabilization & 2-Axis Dynamic Weight Transfer (Pitch & Roll)
        _gx, _gy, _gz = self.physics.get_gravity()
        world_g = abs(_gy) if abs(_gy) > 0.1 else 9.81
        fz_nominal = (self.config.chassis_mass * world_g) / max(1, num_wheels)
        base_min_fn = fz_nominal * 0.18  # Inside wheels retain at least 18% nominal load reserve

        # Smooth droop fade: normal force floor smoothly tapers to 0 as suspension approaches maximum extension limit
        def _get_wheel_min_fn(w_idx: int) -> float:
            w = self.wheel_states[w_idx]
            w_c = self.config.wheels[w_idx]
            m_droop = getattr(w_c, "max_droop", 0.12)
            if w.suspension_length > w_c.suspension_rest_length and m_droop > 1e-4:
                droop_ratio = min(1.0, (w.suspension_length - w_c.suspension_rest_length) / m_droop)
                return base_min_fn * (1.0 - droop_ratio * droop_ratio)
            return base_min_fn

        min_fn_0 = _get_wheel_min_fn(0)
        min_fn_1 = _get_wheel_min_fn(1)
        min_fn_2 = _get_wheel_min_fn(2)
        min_fn_3 = _get_wheel_min_fn(3)

        if num_wheels >= 4:
            w0 = self.wheel_states[0]
            w1 = self.wheel_states[1]
            w2 = self.wheel_states[2]
            w3 = self.wheel_states[3]

            # A. Pitch Load Transfer (Front Axle <-> Rear Axle, strictly load-conserving)
            f_pitch = pitch_weight_transfer * 0.5
            if f_pitch > 0.0:  # Squat under acceleration: transfer front -> rear
                avail_f = max(0.0, (w0.normal_force - min_fn_0) + (w1.normal_force - min_fn_1))
                actual_pitch = min(f_pitch * 2.0, avail_f)
                half_pitch = actual_pitch * 0.5
                if w0.is_grounded and w1.is_grounded:
                    w0.normal_force -= half_pitch
                    w1.normal_force -= half_pitch
                    w2.normal_force += half_pitch
                    w3.normal_force += half_pitch
            elif f_pitch < 0.0:  # Dive under braking: transfer rear -> front
                neg_pitch = -f_pitch
                avail_r = max(0.0, (w2.normal_force - min_fn_2) + (w3.normal_force - min_fn_3))
                actual_pitch = min(neg_pitch * 2.0, avail_r)
                half_pitch = actual_pitch * 0.5
                if w2.is_grounded and w3.is_grounded:
                    w2.normal_force -= half_pitch
                    w3.normal_force -= half_pitch
                    w0.normal_force += half_pitch
                    w1.normal_force += half_pitch

            # B. Anti-Roll Bar Stabilization (Active only between grounded wheels on same axle)
            # Body roll and lateral load transfer naturally occur via chassis roll angle and spring compression.
            # ARB resists asymmetric axle displacement without injecting artificial forces into ungrounded wheels.
            max_arb = fz_nominal * 0.35
            # Front Axle ARB
            if w0.is_grounded and w1.is_grounded:
                f_diff = w0.compression - w1.compression
                f_arb = max(-max_arb, min(max_arb, f_diff * self.config.anti_roll_stiffness))
                if f_arb > 0.0:
                    t_arb = min(f_arb, max(0.0, w1.normal_force - min_fn_1))
                    w0.normal_force += t_arb
                    w1.normal_force -= t_arb
                elif f_arb < 0.0:
                    t_arb = min(-f_arb, max(0.0, w0.normal_force - min_fn_0))
                    w0.normal_force -= t_arb
                    w1.normal_force += t_arb

            # Rear Axle ARB
            if w2.is_grounded and w3.is_grounded:
                r_diff = w2.compression - w3.compression
                r_arb = max(-max_arb, min(max_arb, r_diff * self.config.anti_roll_stiffness))
                if r_arb > 0.0:
                    t_arb_r = min(r_arb, max(0.0, w3.normal_force - min_fn_3))
                    w2.normal_force += t_arb_r
                    w3.normal_force -= t_arb_r
                elif r_arb < 0.0:
                    t_arb_r = min(-r_arb, max(0.0, w2.normal_force - min_fn_2))
                    w2.normal_force -= t_arb_r
                    w3.normal_force += t_arb_r

        # Guarantee invariant: ungrounded wheels strictly carry zero normal force
        for i in range(num_wheels):
            if not self.wheel_states[i].is_grounded:
                self.wheel_states[i].normal_force = 0.0

        # 6. Apply Forces: Suspension, Traction, Braking, and Lateral Tire Friction (Kamm's Circle)
        effective_surface_grip = self.config.tire_grip * self.surface_friction_mult
        load_sens = getattr(self.config, "tire_load_sensitivity", 0.15)
        brake_bias_f = getattr(self.config, "brake_bias_front", 0.65)
        caster_torque_factor = getattr(self.config, "caster_aligning_torque", 0.08)
        total_grounded_fn = sum(w.normal_force for w in self.wheel_states if w.is_grounded)
        inv_total_fn = (1.0 / total_grounded_fn) if total_grounded_fn > 10.0 else (1.0 / max(1, num_wheels))

        for i in range(num_wheels):
            w_cfg = self.config.wheels[i]
            w_state = self.wheel_states[i]

            if not w_state.is_grounded or w_state.normal_force <= 0.0:
                # Wheel is airborne: free-spin dynamics
                # Driven wheels spin up under throttle; undriven or off-throttle wheels decelerate smoothly
                is_driven = w_cfg.is_driven
                if self.config.drive_type == DriveType.FWD:
                    is_driven = w_cfg.is_steerable
                elif self.config.drive_type == DriveType.RWD:
                    is_driven = not w_cfg.is_steerable
                elif self.config.drive_type == DriveType.AWD:
                    is_driven = True

                if is_driven and abs(self.throttle) > 0.01:
                    target_spin = (self.throttle * self.config.top_speed_mps) / max(0.05, w_cfg.radius)
                    w_state.angular_velocity += (target_spin - w_state.angular_velocity) * min(1.0, dt * 10.0)
                else:
                    w_state.angular_velocity *= max(0.0, 1.0 - dt * 2.5)

                w_state.spin_angle += w_state.angular_velocity * dt
                w_state.slip_ratio = 0.0
                w_state.slip_angle = 0.0
                w_state.lateral_slip = 0.0
                w_state.longitudinal_slip = 0.0
                w_state.skidding = False
                continue

            fn = w_state.normal_force

            # Tire Load Sensitivity: heavier loaded tires produce slightly diminishing returns
            load_ratio = fn / max(100.0, fz_nominal)
            load_factor = max(0.70, min(1.30, 1.0 - load_sens * (load_ratio - 1.0)))
            tire_grip_eff = effective_surface_grip * load_factor
            f_traction_max = fn * tire_grip_eff

            # A. Suspension Reaction Force (acts along surface contact normal, eliminating phantom horizontal rake propulsion)
            fx_susp = w_state.hit_normal[0] * fn
            fy_susp = w_state.hit_normal[1] * fn
            fz_susp = w_state.hit_normal[2] * fn

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

            wheel_mass_share = self.config.chassis_mass * (fn * inv_total_fn if total_grounded_fn > 10.0 else (1.0 / num_wheels))
            stop_clamp = abs(v_forward) * (wheel_mass_share / max(0.005, dt))

            # Resistance budget: aggregate engine braking, rolling resistance, foot brake, handbrake, auto-hold
            f_resist = 0.0

            # Foot Braking with realistic Front/Rear Brake Bias
            if self.brake_input > 0.01 and abs(v_forward) > 0.001:
                bias_ratio = brake_bias_f if w_cfg.is_steerable else (1.0 - brake_bias_f)
                f_resist += self.brake_input * self.config.brake_torque * (bias_ratio * 2.0 / num_wheels) * w_cfg.brake_ratio

            # Handbrake on rear wheels
            if not w_cfg.is_steerable and self.handbrake and abs(v_forward) > 0.001:
                f_resist += self.config.handbrake_torque * (1.0 / num_wheels)

            # When driver is off-throttle:
            if abs(self.throttle) < 0.02 and not self.handbrake:
                if abs(v_forward) > 0.40:
                    # Engine braking above walking pace
                    f_resist += 450.0 * (self.engine_rpm / 3500.0) * (1.0 / num_wheels)
                    # Rolling resistance
                    f_resist += 0.018 * fn
                else:
                    # Stationary Auto-Hold: critically damped stop hold without creeping or rolling
                    f_resist += stop_clamp

            # Total resistance strictly opposes v_forward without ever exceeding stop_clamp
            if f_resist > 0.0 and abs(v_forward) > 0.0001:
                f_long -= math.copysign(min(f_resist, stop_clamp), v_forward)

            # Clamp longitudinal force to total traction circle
            f_long = max(-f_traction_max, min(f_traction_max, f_long))

            # 2. Kamm's Friction Ellipse: Combined Slip Traction Envelope
            # Ensures applying throttle or brake reserves cornering grip so the car doesn't slide off at turns
            f_drive_ratio = min(1.0, abs(f_long) / max(1.0, f_traction_max))
            lat_envelope = math.sqrt(max(0.0, 1.0 - (f_drive_ratio * 0.85) ** 2))
            slip_reserve = getattr(self.config, "combined_slip_reserve", 0.42)
            f_lat_budget = f_traction_max * max(slip_reserve, lat_envelope)

            # Handbrake specifically induces rear-axle sliding breakaway
            if not w_cfg.is_steerable and self.handbrake:
                f_lat_budget = min(f_lat_budget, fn * tire_grip_eff * self.config.drift_friction_factor)

            # 3. Progressive Pacejka Brush Slip Model
            # Broad peak and communicative plateau giving progressive feedback before sliding
            s_norm = abs(math.tan(slip_angle)) / 0.22
            if s_norm < 1.0:
                pacejka_factor = math.sin(s_norm * 1.570796)
            else:
                # Dynamic sliding plateau (~86% of peak grip retains controllable drift feel)
                pacejka_factor = 1.0 - 0.14 * min(1.0, (s_norm - 1.0) / 2.5)

            max_lat_force = f_lat_budget * pacejka_factor
            desired_lateral_force = -v_lateral * (self.config.chassis_mass / num_wheels) / max(0.01, dt * 1.6)
            f_lat = max(-max_lat_force, min(max_lat_force, desired_lateral_force))

            # Small velocity deadzone to eliminate zero-speed oscillation
            if abs(v_lateral) < 0.08 and abs(v_forward) < 0.1:
                f_lat *= (abs(v_lateral) / 0.08)

            # 4. Counter-Steering Caster Aligning Torque from Pneumatic Trail
            # Naturally stabilizes wheels and helps driver counter-steer out of slides
            aligning_torque = -f_lat * caster_torque_factor
            if self.is_drifting and w_cfg.is_steerable:
                drift_counter_mult = getattr(self.config, "drift_counter_steer_assist", 0.30)
                aligning_torque -= math.radians(self.drift_angle_deg) * drift_counter_mult * 160.0

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

        # 7. Active Upright & Dynamic Yaw Stabilizer
        # Rapidly restores chassis roll and pitch back to level after cornering or curb impacts,
        # while mass-scaled yaw damping gives a heavy, planted, grounded feeling without twitchiness
        grounded_count = sum(1 for w in self.wheel_states if w.is_grounded)
        if grounded_count >= 1 and self._up_vec[1] > 0.20:
            # Cross product: local_up x world_up (0, 1, 0) = (-uz, 0, ux)
            tau_x = -self._up_vec[2] * self.config.upright_stiffness
            tau_z = self._up_vec[0] * self.config.upright_stiffness

            # Roll/pitch angular velocity damping (leaving steering responsive while absorbing sway)
            tau_x -= self._chassis_angvel[0] * self.config.upright_damping
            tau_z -= self._chassis_angvel[2] * self.config.upright_damping

            # Mass-scaled dynamic yaw stabilization: keeps heavy cars planted and prevents spinouts
            yaw_damp_factor = getattr(self.config, "dynamic_yaw_damping", 0.50) * (self.config.chassis_mass / 1500.0)
            drift_scale = 0.55 if self.is_drifting else 1.0
            tau_y = -self._chassis_angvel[1] * yaw_damp_factor * 2200.0 * drift_scale

            # Turn-in bite assist: lightweight cars rotate with sharp agility, heavy trucks have deliberate inertia
            turn_in_bite = getattr(self.config, "turn_in_bite_assist", 0.0)
            if abs(self.steering_input) > 0.05 and abs(fwd_speed) > 2.0:
                agility_factor = 1500.0 / max(500.0, self.config.chassis_mass)
                tau_y -= self.steering_input * turn_in_bite * 2400.0 * agility_factor

            self._torque_scratch[0] = tau_x * dt
            self._torque_scratch[1] = tau_y * dt
            self._torque_scratch[2] = tau_z * dt
            self.physics.apply_torque_impulse(self.entity_id, self._torque_scratch)

        # 8. Road Holding Plant & Aerodynamic Downforce (Gives vehicle heavy, grounded asphalt feel)
        if grounded_count >= 2:
            # Baseline road adhesion plant scales smoothly with speed to eliminate resting jitter
            plant_factor = min(1.0, max(0.0, (abs(fwd_speed) - 0.5) / 2.5))
            base_plant = self.config.chassis_mass * 4.2 * plant_factor
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

        # 10. Stationary Park Sleep Damping
        # When all wheels are grounded, vehicle is off-throttle, and moving at sub-walking speeds,
        # eliminate residual micro-drift and Euler discrete spring-damping jitter.
        if grounded_count == num_wheels and abs(self.throttle) < 0.02 and abs(fwd_speed) < 0.35:
            vx = self._chassis_linvel[0]
            vz = self._chassis_linvel[2]
            planar_speed_sq = vx * vx + vz * vz
            if planar_speed_sq < 0.16:  # planar speed < 0.40 m/s
                damp = min(1.0, dt * 15.0)
                self._impulse_scratch[0] = -vx * self.config.chassis_mass * damp
                self._impulse_scratch[1] = 0.0
                self._impulse_scratch[2] = -vz * self.config.chassis_mass * damp
                self.physics.apply_impulse(self.entity_id, self._impulse_scratch)

                # Also damp residual yaw / pitch / roll angular velocity so chassis rests motionless
                self._torque_scratch[0] = -self._chassis_angvel[0] * self.config.chassis_mass * 0.35 * damp
                self._torque_scratch[1] = -self._chassis_angvel[1] * self.config.chassis_mass * 0.35 * damp
                self._torque_scratch[2] = -self._chassis_angvel[2] * self.config.chassis_mass * 0.35 * damp
                self.physics.apply_torque_impulse(self.entity_id, self._torque_scratch)

    def get_wheel_transform(self, wheel_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the world position and orientation quaternion for a visual wheel mesh.

        Returns:
            (pos_xyz, rot_xyzw) as NumPy arrays.
        """
        w_state = self.wheel_states[wheel_idx]
        return w_state.world_pos, self._chassis_rot
