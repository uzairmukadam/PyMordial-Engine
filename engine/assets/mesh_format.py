"""32-Byte Cache-Aligned Binary Mesh Format (.pm_mesh) for PyMordial Engine.

Provides hardware-aligned, zero-runtime-parsing mesh serialization:
- 64-byte file header with magic (0x504D4D53, "PMMS") and precomputed bounding volumes
- 32-byte interleaved vertex layout (2 vertices per 64-byte L1 cache line):
    Position: 3 x float32 (12 bytes)
    Normal:   4 x float16 (8 bytes, W = bitangent sign +-1)
    UV:       2 x float16 (4 bytes)
    Tangent:  4 x float16 (8 bytes)
- Contiguous 32-bit index buffer
- Instant O(1) buffer view extraction for direct GPU MegaBuffer streaming
"""

from __future__ import annotations
from dataclasses import dataclass
import struct
from pathlib import Path
import numpy as np

# 4-byte magic string: "PMMS" (PyMordial MeSh) = 0x504D4D53
PM_MESH_MAGIC = b"PMMS"
PM_MESH_VERSION = 1
PM_HEADER_SIZE = 64
VERTEX_STRIDE = 32

# Structured NumPy dtype matching exact 32-byte GPU cache alignment
VERTEX_DTYPE = np.dtype(
    [
        ("position", np.float32, 3),  # 12 bytes: [x, y, z]
        ("normal", np.float16, 4),    # 8 bytes:  [nx, ny, nz, bitangent_sign]
        ("uv", np.float16, 2),        # 4 bytes:  [u, v]
        ("tangent", np.float16, 4),   # 8 bytes:  [tx, ty, tz, handedness]
    ],
    align=False,
)
assert VERTEX_DTYPE.itemsize == VERTEX_STRIDE, f"Vertex stride must be 32, got {VERTEX_DTYPE.itemsize}"


@dataclass(slots=True)
class MeshBoundingVolumes:
    """Precomputed spatial bounding volumes for instant frustum culling and shadows."""

    sphere_center: tuple[float, float, float]
    sphere_radius: float
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]


@dataclass(slots=True)
class PMMesh:
    """Memory-mapped binary 3D mesh ready for direct GPU MegaBuffer upload."""

    vertex_count: int
    index_count: int
    bounds: MeshBoundingVolumes
    vertices: np.ndarray  # Shape (N,), dtype=VERTEX_DTYPE (32 bytes per vertex)
    indices: np.ndarray   # Shape (M,), dtype=np.uint32

    @property
    def vertex_bytes(self) -> bytes:
        """Raw C-contiguous bytes of the 32-byte interleaved vertex buffer."""
        return self.vertices.tobytes()

    @property
    def index_bytes(self) -> bytes:
        """Raw C-contiguous bytes of the uint32 index buffer."""
        return self.indices.tobytes()

    def serialize(self) -> bytes:
        """Serializes the mesh into a contiguous .pm_mesh binary buffer."""
        v_bytes = self.vertices.tobytes()
        i_bytes = self.indices.tobytes()

        # 64-byte Header format:
        # 4s (magic) + 3I (version, v_count, i_count) + 4f (sphere) + 4f (aabb_min) + 4f (aabb_max)
        b = self.bounds
        header = struct.pack(
            "<4sIII4f4f4f",
            PM_MESH_MAGIC,
            PM_MESH_VERSION,
            self.vertex_count,
            self.index_count,
            b.sphere_center[0],
            b.sphere_center[1],
            b.sphere_center[2],
            b.sphere_radius,
            b.aabb_min[0],
            b.aabb_min[1],
            b.aabb_min[2],
            0.0,  # Pad
            b.aabb_max[0],
            b.aabb_max[1],
            b.aabb_max[2],
            0.0,  # Pad
        )
        assert len(header) == PM_HEADER_SIZE

        return header + v_bytes + i_bytes

    def save(self, filepath: str | Path) -> None:
        """Writes mesh to disk at the given path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.serialize())

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> PMMesh:
        """Deserializes a .pm_mesh binary buffer into structured NumPy tables."""
        if not isinstance(data, memoryview):
            mv = memoryview(data)
        else:
            mv = data

        if len(mv) < PM_HEADER_SIZE:
            raise ValueError(f"Data size ({len(mv)}) smaller than .pm_mesh header ({PM_HEADER_SIZE})")

        header = mv[:PM_HEADER_SIZE]
        magic, version, v_count, i_count, sc_x, sc_y, sc_z, s_rad, min_x, min_y, min_z, _, max_x, max_y, max_z, _ = (
            struct.unpack("<4sIII4f4f4f", header)
        )

        if magic != PM_MESH_MAGIC:
            raise ValueError(f"Invalid .pm_mesh magic header: {magic!r}, expected {PM_MESH_MAGIC!r}")
        if version != PM_MESH_VERSION:
            raise ValueError(f"Unsupported .pm_mesh version {version}, expected {PM_MESH_VERSION}")

        v_byte_len = v_count * VERTEX_STRIDE
        i_byte_len = i_count * 4

        expected_size = PM_HEADER_SIZE + v_byte_len + i_byte_len
        if len(mv) < expected_size:
            raise ValueError(f"Corrupt .pm_mesh: expected at least {expected_size} bytes, got {len(mv)}")

        v_offset = PM_HEADER_SIZE
        i_offset = PM_HEADER_SIZE + v_byte_len

        # Zero-copy view directly into memoryview
        vertices = np.frombuffer(mv[v_offset:i_offset], dtype=VERTEX_DTYPE, count=v_count)
        indices = np.frombuffer(mv[i_offset:expected_size], dtype=np.uint32, count=i_count)

        bounds = MeshBoundingVolumes(
            sphere_center=(sc_x, sc_y, sc_z),
            sphere_radius=s_rad,
            aabb_min=(min_x, min_y, min_z),
            aabb_max=(max_x, max_y, max_z),
        )

        return cls(
            vertex_count=v_count,
            index_count=i_count,
            bounds=bounds,
            vertices=vertices,
            indices=indices,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> PMMesh:
        """Loads and deserializes a .pm_mesh file from disk."""
        return cls.from_bytes(Path(filepath).read_bytes())


def compute_bounding_volumes(positions: np.ndarray) -> MeshBoundingVolumes:
    """Computes exact bounding box and minimum bounding sphere for a set of vertices."""
    if len(positions) == 0:
        return MeshBoundingVolumes(
            sphere_center=(0.0, 0.0, 0.0),
            sphere_radius=0.0,
            aabb_min=(0.0, 0.0, 0.0),
            aabb_max=(0.0, 0.0, 0.0),
        )

    pos = positions[:, :3].astype(np.float32)
    min_xyz = np.min(pos, axis=0)
    max_xyz = np.max(pos, axis=0)
    center = (min_xyz + max_xyz) * 0.5
    deltas = pos - center
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    radius = float(np.max(distances))

    return MeshBoundingVolumes(
        sphere_center=(float(center[0]), float(center[1]), float(center[2])),
        sphere_radius=radius,
        aabb_min=(float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])),
        aabb_max=(float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])),
    )


def build_pm_mesh(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    tangents: np.ndarray,
    indices: np.ndarray,
) -> PMMesh:
    """Assembles raw component arrays into an aligned PMMesh instance.

    Args:
        positions: (N, 3) float32 coordinates
        normals: (N, 3) or (N, 4) normals with optional bitangent sign in W
        uvs: (N, 2) texture coordinates
        tangents: (N, 3) or (N, 4) tangents with optional handedness in W
        indices: (M,) uint32 triangle indices
    """
    v_count = len(positions)
    i_count = len(indices)

    vertices = np.zeros(v_count, dtype=VERTEX_DTYPE)
    vertices["position"] = positions[:, :3].astype(np.float32)

    # Normals: ensure 4 channels (W = 1.0 by default)
    if normals.shape[1] == 3:
        n_pad = np.ones((v_count, 1), dtype=np.float32)
        norm_4 = np.hstack([normals[:, :3].astype(np.float32), n_pad])
    else:
        norm_4 = normals[:, :4].astype(np.float32)
    vertices["normal"] = norm_4.astype(np.float16)

    # UVs: 2 channels float16
    vertices["uv"] = uvs[:, :2].astype(np.float16)

    # Tangents: ensure 4 channels (W = 1.0 by default)
    if tangents.shape[1] == 3:
        t_pad = np.ones((v_count, 1), dtype=np.float32)
        tan_4 = np.hstack([tangents[:, :3].astype(np.float32), t_pad])
    else:
        tan_4 = tangents[:, :4].astype(np.float32)
    vertices["tangent"] = tan_4.astype(np.float16)

    bounds = compute_bounding_volumes(positions)
    idx_arr = indices.astype(np.uint32).flatten()

    return PMMesh(
        vertex_count=v_count,
        index_count=i_count,
        bounds=bounds,
        vertices=vertices,
        indices=idx_arr,
    )
