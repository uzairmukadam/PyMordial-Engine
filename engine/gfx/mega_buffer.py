"""Shared Mega-VBO/IBO Allocator and Built-in Procedural Geometry."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import moderngl


@dataclass(slots=True)
class MeshAllocation:
    first_index: int     # Element offset in IBO
    index_count: int     # Total indices for this mesh
    base_vertex: int     # Vertex offset in Mega-VBO
    vertex_count: int    # Total vertices


class MegaBuffer:
    """Consolidates all scene meshes into a single shared Mega-VBO and IBO."""

    __slots__ = (
        "ctx",
        "vbo",
        "ibo",
        "allocations",
        "vertex_data",
        "index_data",
        "total_vertices",
        "total_indices",
    )

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.allocations: dict[str, MeshAllocation] = {}

        # Accumulator lists before GPU buffer creation
        self.vertex_data: list[np.ndarray] = []
        self.index_data: list[np.ndarray] = []
        self.total_vertices = 0
        self.total_indices = 0

        self.vbo: moderngl.Buffer | None = None
        self.ibo: moderngl.Buffer | None = None

        # Pre-register standard procedural 3D primitives
        self._add_cube_primitive()
        self._add_sphere_primitive()
        self._add_plane_primitive()
        self._add_capsule_primitive()
        self._bake_buffers()

    def _add_mesh(
        self,
        name: str,
        vertices: np.ndarray,  # (V, 12) float32 [pos:3, norm:3, uv:2, tangent:4]
        indices: np.ndarray,   # (I,) uint32
    ) -> MeshAllocation:
        alloc = MeshAllocation(
            first_index=self.total_indices,
            index_count=len(indices),
            base_vertex=self.total_vertices,
            vertex_count=len(vertices),
        )
        self.allocations[name] = alloc
        self.vertex_data.append(vertices.astype(np.float32))
        # Offset mesh-local indices by base_vertex into global Mega-IBO space
        global_indices = (indices + self.total_vertices).astype(np.uint32)
        self.index_data.append(global_indices)

        self.total_vertices += len(vertices)
        self.total_indices += len(indices)
        return alloc

    def _bake_buffers(self) -> None:
        """Concatenates all mesh arrays and allocates single GPU Mega-VBO/IBO."""
        if not self.vertex_data:
            return

        all_verts = np.concatenate(self.vertex_data, axis=0)
        all_indices = np.concatenate(self.index_data, axis=0)

        assert all_verts.flags["C_CONTIGUOUS"]
        assert all_indices.flags["C_CONTIGUOUS"]

        self.vbo = self.ctx.buffer(all_verts.tobytes())
        self.ibo = self.ctx.buffer(all_indices.tobytes())

    def get_vao(self, program: moderngl.Program) -> moderngl.VertexArray:
        """Binds Mega-Buffer into a VertexArray dynamically matching active attributes in program."""
        if self.vbo is None or self.ibo is None:
            raise RuntimeError("MegaBuffer has not been baked.")

        spec = [
            ("in_position", "3f", 12),
            ("in_normal", "3f", 12),
            ("in_uv", "2f", 8),
            ("in_tangent", "4f", 16),
        ]
        fmt_parts = []
        attribs = []
        for name, fmt, byte_size in spec:
            if name in program:
                fmt_parts.append(fmt)
                attribs.append(name)
            else:
                fmt_parts.append(f"{byte_size}x")

        fmt_str = " ".join(fmt_parts)
        return self.ctx.vertex_array(
            program,
            [(self.vbo, fmt_str, *attribs)],
            index_buffer=self.ibo,
            index_element_size=4,
        )

    # ---------------- Primitive Generators ----------------

    def _add_cube_primitive(self) -> None:
        """Generates a standard 1.0m unit box centered at origin."""
        # 24 vertices (4 per face) with normals and UVs
        v_list = [
            # Front face (Z = 0.5)
            -0.5, -0.5,  0.5,   0, 0, 1,   0, 0,   1, 0, 0, 1,
             0.5, -0.5,  0.5,   0, 0, 1,   1, 0,   1, 0, 0, 1,
             0.5,  0.5,  0.5,   0, 0, 1,   1, 1,   1, 0, 0, 1,
            -0.5,  0.5,  0.5,   0, 0, 1,   0, 1,   1, 0, 0, 1,
            # Back face (Z = -0.5)
             0.5, -0.5, -0.5,   0, 0, -1,  0, 0,  -1, 0, 0, 1,
            -0.5, -0.5, -0.5,   0, 0, -1,  1, 0,  -1, 0, 0, 1,
            -0.5,  0.5, -0.5,   0, 0, -1,  1, 1,  -1, 0, 0, 1,
             0.5,  0.5, -0.5,   0, 0, -1,  0, 1,  -1, 0, 0, 1,
            # Top face (Y = 0.5)
            -0.5,  0.5,  0.5,   0, 1, 0,   0, 0,   1, 0, 0, 1,
             0.5,  0.5,  0.5,   0, 1, 0,   1, 0,   1, 0, 0, 1,
             0.5,  0.5, -0.5,   0, 1, 0,   1, 1,   1, 0, 0, 1,
            -0.5,  0.5, -0.5,   0, 1, 0,   0, 1,   1, 0, 0, 1,
            # Bottom face (Y = -0.5)
            -0.5, -0.5, -0.5,   0, -1, 0,  0, 0,   1, 0, 0, 1,
             0.5, -0.5, -0.5,   0, -1, 0,  1, 0,   1, 0, 0, 1,
             0.5, -0.5,  0.5,   0, -1, 0,  1, 1,   1, 0, 0, 1,
            -0.5, -0.5,  0.5,   0, -1, 0,  0, 1,   1, 0, 0, 1,
            # Right face (X = 0.5)
             0.5, -0.5,  0.5,   1, 0, 0,   0, 0,   0, 0, -1, 1,
             0.5, -0.5, -0.5,   1, 0, 0,   1, 0,   0, 0, -1, 1,
             0.5,  0.5, -0.5,   1, 0, 0,   1, 1,   0, 0, -1, 1,
             0.5,  0.5,  0.5,   1, 0, 0,   0, 1,   0, 0, -1, 1,
            # Left face (X = -0.5)
            -0.5, -0.5, -0.5,  -1, 0, 0,   0, 0,   0, 0, 1, 1,
            -0.5, -0.5,  0.5,  -1, 0, 0,   1, 0,   0, 0, 1, 1,
            -0.5,  0.5,  0.5,  -1, 0, 0,   1, 1,   0, 0, 1, 1,
            -0.5,  0.5, -0.5,  -1, 0, 0,   0, 1,   0, 0, 1, 1,
        ]
        verts = np.array(v_list, dtype=np.float32).reshape(-1, 12)

        indices = []
        for face in range(6):
            base = face * 4
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

        self._add_mesh("cube", verts, np.array(indices, dtype=np.uint32))

    def _add_plane_primitive(self) -> None:
        """Generates a 1x1 flat horizontal ground plane quad (+Y normal)."""
        verts = np.array(
            [
                [-0.5, 0.0,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0, 1.0],
                [ 0.5, 0.0,  0.5,  0.0, 1.0, 0.0,  1.0, 0.0,  1.0, 0.0, 0.0, 1.0],
                [ 0.5, 0.0, -0.5,  0.0, 1.0, 0.0,  1.0, 1.0,  1.0, 0.0, 0.0, 1.0],
                [-0.5, 0.0, -0.5,  0.0, 1.0, 0.0,  0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._add_mesh("plane", verts, indices)

    def _add_sphere_primitive(self, sectors: int = 24, stacks: int = 16) -> None:
        """Generates a unit UV sphere with radius 1.0."""
        verts_list = []
        radius = 1.0

        for i in range(stacks + 1):
            stack_angle = math.pi / 2 - i * math.pi / stacks  # pi/2 to -pi/2
            xy = radius * math.cos(stack_angle)
            z = radius * math.sin(stack_angle)

            for j in range(sectors + 1):
                sector_angle = j * 2 * math.pi / sectors  # 0 to 2pi
                x = xy * math.cos(sector_angle)
                y = xy * math.sin(sector_angle)

                # Unit normal
                nx = x / radius
                ny = y / radius
                nz = z / radius

                u = j / sectors
                v = i / stacks

                # Tangent (-sin, cos, 0)
                tx = -math.sin(sector_angle)
                ty = math.cos(sector_angle)
                tz = 0.0

                verts_list.extend([x, z, y, nx, nz, ny, u, v, tx, tz, ty, 1.0])

        verts = np.array(verts_list, dtype=np.float32).reshape(-1, 12)

        indices = []
        for i in range(stacks):
            k1 = i * (sectors + 1)
            k2 = k1 + sectors + 1
            for j in range(sectors):
                if i != 0:
                    indices.extend([k1, k1 + 1, k2])
                if i != (stacks - 1):
                    indices.extend([k1 + 1, k2 + 1, k2])
                k1 += 1
                k2 += 1

        self._add_mesh("sphere", verts, np.array(indices, dtype=np.uint32))

    def _add_capsule_primitive(
        self,
        radius: float = 0.4,
        half_height: float = 0.5,
        sectors: int = 24,
        stacks: int = 8,
    ) -> None:
        """Generates a Y-aligned unit capsule primitive (total height 1.8m, radius 0.4m)."""
        verts_list = []
        indices = []

        total_rings = stacks * 2 + 1
        for i in range(total_rings):
            if i <= stacks:
                # Bottom hemisphere: angle from -pi/2 to 0
                phi = -math.pi * 0.5 + (i / stacks) * (math.pi * 0.5)
                y_offset = -half_height
            else:
                # Top hemisphere: angle from 0 to pi/2
                phi = ((i - stacks) / stacks) * (math.pi * 0.5)
                y_offset = half_height

            ring_radius = radius * math.cos(phi)
            ring_y = y_offset + radius * math.sin(phi)
            ny = math.sin(phi)

            for j in range(sectors + 1):
                theta = j * 2.0 * math.pi / sectors
                nx = math.cos(theta) * math.cos(phi)
                nz = math.sin(theta) * math.cos(phi)

                n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-6:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len

                x = ring_radius * math.cos(theta)
                z = ring_radius * math.sin(theta)
                y = ring_y

                u = j / sectors
                v = i / (stacks * 2)

                tx = -math.sin(theta)
                ty = 0.0
                tz = math.cos(theta)

                verts_list.extend([x, y, z, nx, ny, nz, u, v, tx, ty, tz, 1.0])

        for i in range(total_rings - 1):
            k1 = i * (sectors + 1)
            k2 = k1 + sectors + 1
            for j in range(sectors):
                indices.extend([k1, k1 + 1, k2])
                indices.extend([k1 + 1, k2 + 1, k2])
                k1 += 1
                k2 += 1

        verts = np.array(verts_list, dtype=np.float32).reshape(-1, 12)
        self._add_mesh("capsule", verts, np.array(indices, dtype=np.uint32))

    def destroy(self) -> None:
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
