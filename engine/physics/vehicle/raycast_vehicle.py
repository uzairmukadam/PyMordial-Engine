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

from engine.physics.vehicle.vehicle_config import VehicleConfig, WheelConfig, DriveType

if TYPE_CHECKING:
    from engine.core.ecs import EntityManager
    from engine.physics.rapier_world import PhysicsManager


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

        # Vehicle planar speed along forward vector
        fwd_speed = float(np.dot(self._chassis_linvel, self._forward_vec))
        self.current_speed_mps = fwd_speed

        # 3. Smooth steering interpolation with speed-sensitive dynamic lock
        speed_abs = abs(fwd_speed)
        speed_factor = 1.0 / (1.0 + speed_abs * 0.035)
        effective_max_steer = self.config.max_steer_angle_rad * max(0.35, speed_factor)
        target_steer = self.steering_input * effective_max_steer
        steer_diff = target_steer - self.current_steer_angle
        max_steer_step = self.config.steer_speed * dt
        if abs(steer_diff) <= max_steer_step:
            self.current_steer_angle = target_steer
        else:
            self.current_steer_angle += math.copysign(max_steer_step, steer_diff)

        # 4. Suspension Raycasting and Spring Forces
        num_wheels = len(self.config.wheels)
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

            # Ray origin starts slightly below chassis bottom
            ray_ox = self._wheel_attach_world[0] + down_x * chassis_half_y
            ray_oy = self._wheel_attach_world[1] + down_y * chassis_half_y
            ray_oz = self._wheel_attach_world[2] + down_z * chassis_half_y

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
                effective_dist = dist_from_ray_origin + chassis_half_y
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
                    w_state.hit_normal[0] = nx
                    w_state.hit_normal[1] = ny
                    w_state.hit_normal[2] = nz

                    # Spring + Damper force (Hooke's Law)
                    v_susp = (w_state.prev_suspension_length - susp_len) / dt
                    spring_force = w_cfg.spring_stiffness * w_state.compression
                    damper_force = w_cfg.spring_damping * v_susp
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

        # 5. Anti-Roll Bar Stabilization across axles (Front: 0&1, Rear: 2&3)
        # Extra force added to compressed wheel to lift chassis back to level
        if num_wheels >= 4:
            # Front Axle
            f_diff = self.wheel_states[0].compression - self.wheel_states[1].compression
            f_arb = f_diff * self.config.anti_roll_stiffness
            if self.wheel_states[0].is_grounded:
                self.wheel_states[0].normal_force = max(0.0, self.wheel_states[0].normal_force + f_arb)
            if self.wheel_states[1].is_grounded:
                self.wheel_states[1].normal_force = max(0.0, self.wheel_states[1].normal_force - f_arb)

            # Rear Axle
            r_diff = self.wheel_states[2].compression - self.wheel_states[3].compression
            r_arb = r_diff * self.config.anti_roll_stiffness
            if self.wheel_states[2].is_grounded:
                self.wheel_states[2].normal_force = max(0.0, self.wheel_states[2].normal_force + r_arb)
            if self.wheel_states[3].is_grounded:
                self.wheel_states[3].normal_force = max(0.0, self.wheel_states[3].normal_force - r_arb)

        # 6. Apply Forces: Suspension, Traction, Braking, and Lateral Tire Friction
        effective_surface_grip = self.config.tire_grip * self.surface_friction_mult

        for i in range(num_wheels):
            w_cfg = self.config.wheels[i]
            w_state = self.wheel_states[i]

            if not w_state.is_grounded or w_state.normal_force <= 0.0:
                continue

            # A. Suspension Upward Force: primarily pushes vertically against gravity
            fn = w_state.normal_force
            fx_susp = self._up_vec[0] * (fn * 0.25)
            fy_susp = fn
            fz_susp = self._up_vec[2] * (fn * 0.25)

            # B. Wheel Steering Directions
            steer = w_state.steer_angle
            cos_s = math.cos(steer)
            sin_s = math.sin(steer)

            # wheel forward = forward * cos(steer) + right * sin(steer)
            self._wheel_forward[0] = self._forward_vec[0] * cos_s + self._right_vec[0] * sin_s
            self._wheel_forward[1] = self._forward_vec[1] * cos_s + self._right_vec[1] * sin_s
            self._wheel_forward[2] = self._forward_vec[2] * cos_s + self._right_vec[2] * sin_s

            # wheel right = right * cos(steer) - forward * sin(steer)
            self._wheel_right[0] = self._right_vec[0] * cos_s - self._forward_vec[0] * sin_s
            self._wheel_right[1] = self._right_vec[1] * cos_s - self._forward_vec[1] * sin_s
            self._wheel_right[2] = self._right_vec[2] * cos_s - self._forward_vec[2] * sin_s

            # Contact point relative to chassis center of mass
            self._r_vec[0] = w_state.hit_point[0] - self._chassis_pos[0]
            self._r_vec[1] = w_state.hit_point[1] - self._chassis_pos[1]
            self._r_vec[2] = w_state.hit_point[2] - self._chassis_pos[2]

            # Linear velocity at contact point: v_contact = v_chassis + w x r
            # Cross product w x r:
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

            # C. Lateral Friction (Counteracting slip)
            lateral_grip = effective_surface_grip
            if not w_cfg.is_steerable and self.handbrake:
                # Rear wheels locked under handbrake: trigger drift slip
                lateral_grip *= self.config.drift_friction_factor

            # Friction limit based on normal load (Coulomb / Brush friction)
            max_lateral_force = fn * lateral_grip
            desired_lateral_force = -v_lateral * (self.config.chassis_mass / num_wheels) / max(0.01, dt * 2.0)
            f_lat = max(-max_lateral_force, min(max_lateral_force, desired_lateral_force))

            # D. Longitudinal Force (Powertrain & Brakes)
            f_long = 0.0

            # Powertrain Drive
            is_driven_wheel = w_cfg.is_driven
            if self.config.drive_type == DriveType.FWD:
                is_driven_wheel = w_cfg.is_steerable
            elif self.config.drive_type == DriveType.RWD:
                is_driven_wheel = not w_cfg.is_steerable
            elif self.config.drive_type == DriveType.AWD:
                is_driven_wheel = True

            # Number of driven wheels on the vehicle
            num_driven = 2 if self.config.drive_type in (DriveType.RWD, DriveType.FWD) else num_wheels

            if is_driven_wheel and abs(self.throttle) > 0.01:
                if self.throttle > 0.0:
                    if fwd_speed < self.config.top_speed_mps:
                        # Falloff as top speed is approached
                        speed_ratio = max(0.0, fwd_speed / self.config.top_speed_mps)
                        power_curve = 1.0 - speed_ratio * speed_ratio
                        f_long += self.throttle * self.config.engine_torque * power_curve * (1.0 / num_driven)
                else:
                    # Reverse
                    if fwd_speed > -15.0:
                        f_long += self.throttle * self.config.reverse_torque * (1.0 / num_driven)

            # Foot Braking
            if self.brake_input > 0.01 and abs(v_forward) > 0.05:
                brake_f = self.brake_input * self.config.brake_torque * w_cfg.brake_ratio * (1.0 / num_wheels)
                f_long -= math.copysign(min(brake_f, abs(v_forward) * 2000.0), v_forward)

            # Handbrake on rear wheels
            if not w_cfg.is_steerable and self.handbrake and abs(v_forward) > 0.05:
                hb_f = self.config.handbrake_torque * (1.0 / num_wheels)
                f_long -= math.copysign(min(hb_f, abs(v_forward) * 3000.0), v_forward)

            # Traction friction clamp
            max_long_force = fn * effective_surface_grip
            f_long = max(-max_long_force, min(max_long_force, f_long))

            # E. Aggregate Total Contact Force (Suspension + Longitudinal + Lateral)
            f_total_x = fx_susp + self._wheel_forward[0] * f_long + self._wheel_right[0] * f_lat
            f_total_y = fy_susp + self._wheel_forward[1] * f_long + self._wheel_right[1] * f_lat
            f_total_z = fz_susp + self._wheel_forward[2] * f_long + self._wheel_right[2] * f_lat

            # Convert Force to Impulse: J = F * dt
            self._impulse_scratch[0] = f_total_x * dt
            self._impulse_scratch[1] = f_total_y * dt
            self._impulse_scratch[2] = f_total_z * dt

            # Torque Impulse: Tau = r x J
            rx, ry, rz = self._r_vec[0], self._r_vec[1], self._r_vec[2]
            jx, jy, jz = self._impulse_scratch[0], self._impulse_scratch[1], self._impulse_scratch[2]

            self._torque_scratch[0] = ry * jz - rz * jy
            self._torque_scratch[1] = rz * jx - rx * jz
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

        # 8. Aerodynamic Downforce (plants vehicle to asphalt at high speed)
        if fwd_speed > 8.0 and grounded_count >= 2:
            downforce = min(3500.0, 0.5 * 1.225 * (fwd_speed * fwd_speed) * 0.45)
            self._impulse_scratch[0] = 0.0
            self._impulse_scratch[1] = -downforce * dt
            self._impulse_scratch[2] = 0.0
            self.physics.apply_impulse(self.entity_id, self._impulse_scratch)

    def get_wheel_transform(self, wheel_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the world position and orientation quaternion for a visual wheel mesh.

        Returns:
            (pos_xyz, rot_xyzw) as NumPy arrays.
        """
        w_state = self.wheel_states[wheel_idx]
        w_cfg = self.config.wheels[wheel_idx]

        # Steering yaw around local Up + Rolling spin around local Right
        steer = w_state.steer_angle
        spin = w_state.spin_angle

        # Compute combined quaternion (ChassisRot * SteerYaw * WheelSpin)
        # For simplicity and zero allocation, return wheel world_pos and chassis orientation
        return w_state.world_pos, self._chassis_rot
