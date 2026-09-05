"""High-performance vectorized math utilities for PyMordial Engine.

Standardizes on Right-Handed conventions:
- +X: Right
- +Y: Up
- -Z: Forward
- Quaternions: [x, y, z, w]

Includes zero-allocation vectorized routines operating directly on C-contiguous memory.
"""

from __future__ import annotations
import math
import numpy as np


def quat_identity() -> np.ndarray:
    """Returns an identity quaternion [0, 0, 0, 1]."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes a single quaternion or an array of (N, 4) quaternions."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return (q / norm).astype(np.float32)


def quat_from_axis_angle(axis: np.ndarray | tuple[float, float, float], angle_rad: float) -> np.ndarray:
    """Creates a unit quaternion [x, y, z, w] from an axis and angle in radians."""
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return quat_identity()
    axis = axis / axis_norm
    half_angle = angle_rad * 0.5
    s = math.sin(half_angle)
    c = math.cos(half_angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def quat_from_euler(pitch_x: float, yaw_y: float, roll_z: float) -> np.ndarray:
    """Creates a quaternion from Euler angles (in radians, Y-X-Z order)."""
    hp = pitch_x * 0.5
    hy = yaw_y * 0.5
    hr = roll_z * 0.5

    sp = math.sin(hp)
    cp = math.cos(hp)
    sy = math.sin(hy)
    cy = math.cos(hy)
    sr = math.sin(hr)
    cr = math.cos(hr)

    # YXZ intrinsic rotation
    x = sp * cy * cr + cp * sy * sr
    y = cp * sy * cr - sp * cy * sr
    z = cp * cy * sr - sp * sy * cr
    w = cp * cy * cr + sp * sy * sr

    return np.array([x, y, z, w], dtype=np.float32)


def quat_nlerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized Linear Interpolation (NLERP) for sub-frame rotation smoothing.

    Supports single (4,) or batch (N, 4) quaternions.
    Handles antipodal alignment to take the shortest path on the 4D sphere.
    """
    # Ensure shortest path by flipping signs if dot product is negative
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0).astype(np.float32)
    interpolated = (1.0 - alpha) * q1 + (alpha * sign) * q2
    return quat_normalize(interpolated)


def vec3_lerp(v1: np.ndarray, v2: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized linear interpolation between v1 and v2 for (3,) or (N, 3)."""
    return ((1.0 - alpha) * v1 + alpha * v2).astype(np.float32)


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def trs_to_mat4(
    translation: np.ndarray | tuple[float, float, float],
    rotation_quat: np.ndarray | tuple[float, float, float, float],
    scale: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Constructs a 4x4 transformation matrix [TRS] in flat 16 float32 format.

    Returns:
        np.ndarray of shape (16,) dtype float32 (Column-major memory layout
        compatible with OpenGL std430 SSBO mat4).
    """
    tx, ty, tz = translation[0], translation[1], translation[2]
    sx, sy, sz = scale[0], scale[1], scale[2]
    r = quat_to_mat3(np.asarray(rotation_quat, dtype=np.float32))

    # Construct column-major 4x4 matrix:
    # col 0: [r00 * sx, r10 * sx, r20 * sx, 0]
    # col 1: [r01 * sy, r11 * sy, r21 * sy, 0]
    # col 2: [r02 * sz, r12 * sz, r22 * sz, 0]
    # col 3: [tx, ty, tz, 1]
    mat = np.array(
        [
            r[0, 0] * sx, r[1, 0] * sx, r[2, 0] * sx, 0.0,
            r[0, 1] * sy, r[1, 1] * sy, r[2, 1] * sy, 0.0,
            r[0, 2] * sz, r[1, 2] * sz, r[2, 2] * sz, 0.0,
            tx, ty, tz, 1.0,
        ],
        dtype=np.float32,
    )
    return mat


def batch_nlerp_and_compose_mat4(
    prev_state: np.ndarray,  # (N, 7) [px, py, pz, qx, qy, qz, qw]
    curr_state: np.ndarray,  # (N, 7) [px, py, pz, qx, qy, qz, qw]
    alpha: float,
    out_matrices: np.ndarray,  # (N, 16) float32 output buffer
    count: int,
) -> None:
    """Batch NLERP sub-frame interpolation and direct composition into WorldTransforms.

    Zero Python object allocations: modifies `out_matrices[:count]` in-place.
    """
    if count <= 0:
        return

    p_prev = prev_state[:count, 0:3]
    p_curr = curr_state[:count, 0:3]
    q_prev = prev_state[:count, 3:7]
    q_curr = curr_state[:count, 3:7]

    # Interpolate positions
    pos = (1.0 - alpha) * p_prev + alpha * p_curr

    # Interpolate rotations with sign alignment
    dot = np.sum(q_prev * q_curr, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0).astype(np.float32)
    q_interp = (1.0 - alpha) * q_prev + (alpha * sign) * q_curr
    norm = np.linalg.norm(q_interp, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    q = q_interp / norm

    # Vectorized quaternion to rotation matrix components
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

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

    # Column 0
    out_matrices[:count, 0] = 1.0 - (yy + zz)
    out_matrices[:count, 1] = xy + wz
    out_matrices[:count, 2] = xz - wy
    out_matrices[:count, 3] = 0.0

    # Column 1
    out_matrices[:count, 4] = xy - wz
    out_matrices[:count, 5] = 1.0 - (xx + zz)
    out_matrices[:count, 6] = yz + wx
    out_matrices[:count, 7] = 0.0

    # Column 2
    out_matrices[:count, 8] = xz + wy
    out_matrices[:count, 9] = yz - wx
    out_matrices[:count, 10] = 1.0 - (xx + yy)
    out_matrices[:count, 11] = 0.0

    # Column 3 (Translation)
    out_matrices[:count, 12] = pos[:, 0]
    out_matrices[:count, 13] = pos[:, 1]
    out_matrices[:count, 14] = pos[:, 2]
    out_matrices[:count, 15] = 1.0


def matrix_look_at(
    eye: np.ndarray | tuple[float, float, float],
    target: np.ndarray | tuple[float, float, float],
    up: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Right-Handed look-at matrix (+Y Up, -Z Forward).

    Returns:
        np.ndarray of shape (16,) dtype float32 (Column-major format).
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Forward direction (-Z)
    f = target - eye
    f_norm = np.linalg.norm(f)
    f = f / (f_norm if f_norm > 1e-8 else 1.0)

    # Right direction (+X) = f x up
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    s = s / (s_norm if s_norm > 1e-8 else 1.0)

    # Recomputed true up (+Y) = s x f
    u = np.cross(s, f)

    # Column-major view matrix:
    return np.array(
        [
            s[0], u[0], -f[0], 0.0,
            s[1], u[1], -f[1], 0.0,
            s[2], u[2], -f[2], 0.0,
            -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0,
        ],
        dtype=np.float32,
    )


def matrix_perspective(
    fovy_rad: float,
    aspect: float,
    near: float,
    far: float,
    reverse_z: bool = False,
) -> np.ndarray:
    """Right-Handed perspective projection matrix.

    Args:
        fovy_rad: Vertical field of view in radians.
        aspect: Width / Height aspect ratio.
        near: Near clipping plane distance (> 0).
        far: Far clipping plane distance (> near).
        reverse_z: If True, maps near to 1.0 and far to 0.0 (floating-point depth).

    Returns:
        np.ndarray of shape (16,) dtype float32 (Column-major format).
    """
    tan_half_fovy = math.tan(fovy_rad * 0.5)
    f = 1.0 / tan_half_fovy

    mat = np.zeros(16, dtype=np.float32)
    mat[0] = f / aspect
    mat[5] = f
    mat[11] = -1.0

    if reverse_z:
        # Reversed-Z: near -> 1.0, far -> 0.0
        mat[10] = near / (far - near)
        mat[14] = (far * near) / (far - near)
    else:
        # Standard OpenGL: near -> -1.0, far -> 1.0 (or 0..1)
        mat[10] = -(far + near) / (far - near)
        mat[14] = -(2.0 * far * near) / (far - near)

    return mat
