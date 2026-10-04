"""Shared Mega-VBO/IBO Allocator and Built-in Procedural Geometry."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
import moderngl

if TYPE_CHECKING:
    from engine.assets.mesh_format import PMMesh


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
        "on_bake_callbacks",
    )

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.allocations: dict[str, MeshAllocation] = {}
        self.on_bake_callbacks: list[Callable[[], None]] = []

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
        self._add_cylinder_primitive()
        self._add_cone_primitive()
        self._bake_buffers()

    def _add_mesh(
        self,
        name: str,
        vertices: np.ndarray,  # Either (V,) VERTEX_DTYPE or (V, 12) float32
        indices: np.ndarray,   # (I,) uint32
    ) -> MeshAllocation:
        from engine.assets.mesh_format import VERTEX_DTYPE

        # Normalize to 32-byte cache-aligned VERTEX_DTYPE
        if vertices.dtype != VERTEX_DTYPE:
            if vertices.dtype == np.float32 and vertices.ndim == 2 and vertices.shape[1] == 12:
                v_out = np.zeros(len(vertices), dtype=VERTEX_DTYPE)
                v_out["position"] = vertices[:, 0:3]
                v_out["normal"] = np.hstack(
                    [vertices[:, 3:6], np.ones((len(vertices), 1), dtype=np.float32)]
                ).astype(np.float16)
                v_out["uv"] = vertices[:, 6:8].astype(np.float16)
                v_out["tangent"] = vertices[:, 8:12].astype(np.float16)
                vertices = v_out
            else:
                raise ValueError(f"Unsupported vertex array format: {vertices.dtype}, shape: {vertices.shape}")

        alloc = MeshAllocation(
            first_index=self.total_indices,
            index_count=len(indices),
            base_vertex=self.total_vertices,
            vertex_count=len(vertices),
        )
        self.allocations[name] = alloc
        self.vertex_data.append(vertices)
        # Offset mesh-local indices by base_vertex into global Mega-IBO space
        global_indices = (indices + self.total_vertices).astype(np.uint32)
        self.index_data.append(global_indices)

        self.total_vertices += len(vertices)
        self.total_indices += len(indices)
        return alloc

    def add_pm_mesh(self, name: str, mesh: PMMesh, auto_bake: bool = True) -> MeshAllocation:
        """Registers a cooked 32-byte PMMesh into the Mega-Buffer."""
        if name in self.allocations:
            return self.allocations[name]
        alloc = self._add_mesh(name, mesh.vertices, mesh.indices)
        if auto_bake and self.vbo is not None:
            self.bake()
        return alloc

    def bake(self) -> None:
        """Concatenates all mesh arrays and reallocates GPU Mega-VBO/IBO."""
        self._bake_buffers()
        for cb in self.on_bake_callbacks:
            cb()

    def _bake_buffers(self) -> None:
        """Concatenates all mesh arrays and allocates single GPU Mega-VBO/IBO."""
        if not self.vertex_data:
            return

        all_verts = np.concatenate(self.vertex_data, axis=0)
        all_indices = np.concatenate(self.index_data, axis=0)

        assert all_verts.flags["C_CONTIGUOUS"]
        assert all_indices.flags["C_CONTIGUOUS"]

        if self.vbo is not None:
            self.vbo.release()
        if self.ibo is not None:
            self.ibo.release()

        self.vbo = self.ctx.buffer(all_verts.tobytes())
        self.ibo = self.ctx.buffer(all_indices.tobytes())

    def get_vao(self, program: moderngl.Program, mode: int | None = None) -> moderngl.VertexArray:
        """Binds Mega-Buffer into a VertexArray matching active attributes in program (32-byte layout)."""
        if self.vbo is None or self.ibo is None:
            raise RuntimeError("MegaBuffer has not been baked.")

        # 32-byte layout: Position (12B float32), Normal (8B float16), UV (4B float16), Tangent (8B float16)
        spec = [
            ("in_position", "3f4", 12),
            ("in_normal", "4f2", 8),
            ("in_uv", "2f2", 4),
            ("in_tangent", "4f2", 8),
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
        kwargs = {}
        if mode is not None:
            kwargs["mode"] = mode
        return self.ctx.vertex_array(
            program,
            [(self.vbo, fmt_str, *attribs)],
            index_buffer=self.ibo,
            index_element_size=4,
            **kwargs,
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
        sectors: int = 32,
        stacks: int = 12,
    ) -> None:
        """Generates a Y-aligned unit capsule primitive (total height 1.8m, radius 0.4m).

        The capsule consists of:
        - Bottom hemisphere: stacks intervals from phi = -pi/2 to 0 centered at y = -half_height
        - Middle cylinder: vertical wall from y = -half_height to y = +half_height with radius `radius`
        - Top hemisphere: stacks intervals from phi = 0 to +pi/2 centered at y = +half_height
        """
        verts_list = []
        indices = []

        # Total rings = (stacks + 1 for bottom dome) + (stacks + 1 for top dome)
        # Ring index 'stacks' is at y = -half_height (phi = 0)
        # Ring index 'stacks + 1' is at y = +half_height (phi = 0)
        # The quad between stacks and stacks + 1 forms the perfect cylinder!
        total_rings = (stacks + 1) * 2

        for i in range(total_rings):
            if i <= stacks:
                # Bottom hemisphere: phi from -pi/2 to 0
                phi = -math.pi * 0.5 + (i / stacks) * (math.pi * 0.5)
                y_offset = -half_height
            else:
                # Top hemisphere: phi from 0 to +pi/2
                top_idx = i - (stacks + 1)
                phi = (top_idx / stacks) * (math.pi * 0.5)
                y_offset = half_height

            ring_radius = radius * math.cos(phi)
            ring_y = y_offset + radius * math.sin(phi)
            ny = math.sin(phi)
            cos_phi = math.cos(phi)

            v = i / (total_rings - 1)

            for j in range(sectors + 1):
                theta = j * 2.0 * math.pi / sectors
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)

                nx = cos_theta * cos_phi
                nz = sin_theta * cos_phi

                n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-6:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len

                x = ring_radius * cos_theta
                y = ring_y
                z = ring_radius * sin_theta

                u = j / sectors

                tx = -sin_theta
                ty = 0.0
                tz = cos_theta

                verts_list.extend([x, y, z, nx, ny, nz, u, v, tx, ty, tz, 1.0])

        for i in range(total_rings - 1):
            k1 = i * (sectors + 1)
            k2 = k1 + sectors + 1
            for j in range(sectors):
                indices.extend([k1, k2, k1 + 1])
                indices.extend([k1 + 1, k2, k2 + 1])
                k1 += 1
                k2 += 1

        verts = np.array(verts_list, dtype=np.float32).reshape(-1, 12)
        self._add_mesh("capsule", verts, np.array(indices, dtype=np.uint32))

    def _add_cylinder_primitive(
        self,
        radius: float = 1.0,
        half_width: float = 0.5,
        sectors: int = 32,
    ) -> None:
        """Generates an X-axis aligned unit cylinder primitive (radius 1.0, width 1.0).

        Designed specifically for vehicle wheels and dynamic physics cylinders:
        - Sidewalls at x = -half_width and x = +half_width with flat disc caps.
        - Outer cylindrical barrel with radial tread normals in Y-Z plane.
        """
        verts_list = []
        indices = []

        # 1. Barrel (tread) vertices: 2 rings at -half_width and +half_width
        barrel_start = 0
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / sectors
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            y = radius * cos_t
            z = radius * sin_t
            ny = cos_t
            nz = sin_t
            u = j / sectors

            # Ring 0: -half_width (Left)
            verts_list.extend([-half_width, y, z, 0.0, ny, nz, u, 0.0, 0.0, -sin_t, cos_t, 1.0])
            # Ring 1: +half_width (Right)
            verts_list.extend([half_width, y, z, 0.0, ny, nz, u, 1.0, 0.0, -sin_t, cos_t, 1.0])

        for j in range(sectors):
            k1 = barrel_start + j * 2
            k2 = k1 + 2
            indices.extend([k1, k2, k1 + 1])
            indices.extend([k1 + 1, k2, k2 + 1])

        # 2. Left Cap (-X, normal: -1, 0, 0)
        left_center_idx = len(verts_list) // 12
        verts_list.extend([-half_width, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0])
        left_rim_start = left_center_idx + 1
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / sectors
            y = radius * math.cos(theta)
            z = radius * math.sin(theta)
            u = 0.5 - 0.5 * math.cos(theta)
            v = 0.5 + 0.5 * math.sin(theta)
            verts_list.extend([-half_width, y, z, -1.0, 0.0, 0.0, u, v, 0.0, 0.0, 1.0, 1.0])

        for j in range(sectors):
            indices.extend([left_center_idx, left_rim_start + j + 1, left_rim_start + j])

        # 3. Right Cap (+X, normal: +1, 0, 0)
        right_center_idx = len(verts_list) // 12
        verts_list.extend([half_width, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0])
        right_rim_start = right_center_idx + 1
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / sectors
            y = radius * math.cos(theta)
            z = radius * math.sin(theta)
            u = 0.5 + 0.5 * math.cos(theta)
            v = 0.5 + 0.5 * math.sin(theta)
            verts_list.extend([half_width, y, z, 1.0, 0.0, 0.0, u, v, 0.0, 0.0, 1.0, 1.0])

        for j in range(sectors):
            indices.extend([right_center_idx, right_rim_start + j, right_rim_start + j + 1])

        verts = np.array(verts_list, dtype=np.float32).reshape(-1, 12)
        self._add_mesh("cylinder", verts, np.array(indices, dtype=np.uint32))

    def _add_cone_primitive(
        self,
        radius: float = 0.5,
        height: float = 1.0,
        sectors: int = 24,
    ) -> None:
        """Generates a Y-axis aligned unit cone primitive (base at y=0, apex at y=height)."""
        verts_list = []
        indices = []

        slope = math.sqrt(radius * radius + height * height)
        ny_side = radius / slope
        nr_side = height / slope

        apex_idx = 0
        verts_list.extend([0.0, height, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0])
        side_base_start = 1
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / sectors
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x = radius * cos_t
            z = radius * sin_t
            nx = nr_side * cos_t
            nz = nr_side * sin_t
            u = j / sectors
            verts_list.extend([x, 0.0, z, nx, ny_side, nz, u, 0.0, -sin_t, 0.0, cos_t, 1.0])

        for j in range(sectors):
            indices.extend([apex_idx, side_base_start + j + 1, side_base_start + j])

        bot_center_idx = len(verts_list) // 12
        verts_list.extend([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0])
        bot_rim_start = bot_center_idx + 1
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / sectors
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            u = 0.5 + 0.5 * math.cos(theta)
            v = 0.5 + 0.5 * math.sin(theta)
            verts_list.extend([x, 0.0, z, 0.0, -1.0, 0.0, u, v, 1.0, 0.0, 0.0, 1.0])

        for j in range(sectors):
            indices.extend([bot_center_idx, bot_rim_start + j, bot_rim_start + j + 1])

        verts = np.array(verts_list, dtype=np.float32).reshape(-1, 12)
        self._add_mesh("cone", verts, np.array(indices, dtype=np.uint32))

    def destroy(self) -> None:
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
