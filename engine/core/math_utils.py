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
    scales: np.ndarray | None = None,  # (N, 3) [sx, sy, sz]
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

    if scales is not None:
        sx = scales[:count, 0]
        sy = scales[:count, 1]
        sz = scales[:count, 2]

        # Column 0
        out_matrices[:count, 0] = (1.0 - (yy + zz)) * sx
        out_matrices[:count, 1] = (xy + wz) * sx
        out_matrices[:count, 2] = (xz - wy) * sx
        out_matrices[:count, 3] = 0.0

        # Column 1
        out_matrices[:count, 4] = (xy - wz) * sy
        out_matrices[:count, 5] = (1.0 - (xx + zz)) * sy
        out_matrices[:count, 6] = (yz + wx) * sy
        out_matrices[:count, 7] = 0.0

        # Column 2
        out_matrices[:count, 8] = (xz + wy) * sz
        out_matrices[:count, 9] = (yz - wx) * sz
        out_matrices[:count, 10] = (1.0 - (xx + yy)) * sz
        out_matrices[:count, 11] = 0.0
    else:
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
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Right-Handed look-at matrix (+Y Up, -Z Forward).

    Computes the column-major 4x4 view matrix using zero-allocation scalar arithmetic.

    Returns:
        np.ndarray of shape (16,) dtype float32 (Column-major format).
    """
    ex, ey, ez = float(eye[0]), float(eye[1]), float(eye[2])
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    ux, uy, uz = float(up[0]), float(up[1]), float(up[2])

    # Forward vector f = target - eye
    fx = tx - ex
    fy = ty - ey
    fz = tz - ez
    f_len = math.sqrt(fx * fx + fy * fy + fz * fz)
    inv_f_len = 1.0 / (f_len if f_len > 1e-8 else 1.0)
    fx *= inv_f_len
    fy *= inv_f_len
    fz *= inv_f_len

    # Right vector s = f x up
    sx = fy * uz - fz * uy
    sy = fz * ux - fx * uz
    sz = fx * uy - fy * ux
    s_len = math.sqrt(sx * sx + sy * sy + sz * sz)
    inv_s_len = 1.0 / (s_len if s_len > 1e-8 else 1.0)
    sx *= inv_s_len
    sy *= inv_s_len
    sz *= inv_s_len

    # Recomputed true up vector u = s x f
    ux_f = sy * fz - sz * fy
    uy_f = sz * fx - sx * fz
    uz_f = sx * fy - sy * fx

    # Dot products for camera translation
    dot_s_e = sx * ex + sy * ey + sz * ez
    dot_u_e = ux_f * ex + uy_f * ey + uz_f * ez
    dot_f_e = fx * ex + fy * ey + fz * ez

    mat = out if out is not None else np.zeros(16, dtype=np.float32)
    mat[0] = sx
    mat[1] = ux_f
    mat[2] = -fx
    mat[3] = 0.0

    mat[4] = sy
    mat[5] = uy_f
    mat[6] = -fy
    mat[7] = 0.0

    mat[8] = sz
    mat[9] = uz_f
    mat[10] = -fz
    mat[11] = 0.0

    mat[12] = -dot_s_e
    mat[13] = -dot_u_e
    mat[14] = dot_f_e
    mat[15] = 1.0

    return mat


def matrix_perspective(
    fovy_rad: float,
    aspect: float,
    near: float,
    far: float,
    reverse_z: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Right-Handed perspective projection matrix.

    Args:
        fovy_rad: Vertical field of view in radians.
        aspect: Width / Height aspect ratio.
        near: Near clipping plane distance (> 0).
        far: Far clipping plane distance (> near).
        reverse_z: If True, maps near to 1.0 and far to 0.0 (floating-point depth).
        out: Optional pre-allocated (16,) float32 array to write into.

    Returns:
        np.ndarray of shape (16,) dtype float32 (Column-major format).
    """
    tan_half_fovy = math.tan(fovy_rad * 0.5)
    f = 1.0 / tan_half_fovy

    mat = out if out is not None else np.zeros(16, dtype=np.float32)
    mat.fill(0.0)
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


def mat4_mul(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Multiplies two column-major 4x4 matrices and returns a column-major 4x4 matrix."""
    a_mat = a.reshape((4, 4), order="F")
    b_mat = b.reshape((4, 4), order="F")
    if out is not None:
        out_mat = out.reshape((4, 4), order="F")
        np.matmul(a_mat, b_mat, out=out_mat)
        return out
    return (a_mat @ b_mat).ravel(order="F").astype(np.float32)


def mat4_inv(m: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Computes the inverse of a column-major 4x4 matrix and returns a column-major 4x4 matrix."""
    m_mat = m.reshape((4, 4), order="F")
    inv_mat = np.linalg.inv(m_mat)
    if out is not None:
        out.reshape((4, 4), order="F")[:] = inv_mat
        return out
    return inv_mat.ravel(order="F").astype(np.float32)
