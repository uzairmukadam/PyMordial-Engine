"""Transform Composition Math for Mapping System.

Provides zero-allocation routines for 3D parent-child spatial composition:
M_world = M_parent * M_child
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.world.mapping.schema import TransformDef


def quat_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiplies two quaternions [x, y, z, w]. Represents rotating by q1, then q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def rotate_vec3_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotates a 3D vector by a unit quaternion [x, y, z, w]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2.0 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + w * t + cross(q.xyz, t)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def combine_transforms(
    parent: TransformDef,
    child: TransformDef,
) -> TransformDef:
    """Combines a parent transform and a child local transform into a world transform.

    M_world = M_parent * M_child
    """
    from engine.world.mapping.schema import TransformDef

    # 1. Scale child local position by parent scale
    scaled_lx = child.position[0] * parent.scale[0]
    scaled_ly = child.position[1] * parent.scale[1]
    scaled_lz = child.position[2] * parent.scale[2]

    # 2. Rotate scaled position by parent rotation
    rotated_pos = rotate_vec3_by_quat((scaled_lx, scaled_ly, scaled_lz), parent.rotation)

    # 3. Translate by parent position
    world_pos = (
        parent.position[0] + rotated_pos[0],
        parent.position[1] + rotated_pos[1],
        parent.position[2] + rotated_pos[2],
    )

    # 4. Multiply rotations: parent rotation * child local rotation
    world_rot = quat_multiply(parent.rotation, child.rotation)

    # 5. Multiply scales
    world_scale = (
        parent.scale[0] * child.scale[0],
        parent.scale[1] * child.scale[1],
        parent.scale[2] * child.scale[2],
    )

    return TransformDef(position=world_pos, rotation=world_rot, scale=world_scale)
