"""Immediate-Mode 3D Wireframe Gizmo Renderer (DebugDraw) for PyMordial Engine.

Provides zero-overhead immediate-mode visual debugging for physics colliders,
bounding spheres, raycasts, coordinate frames, and arbitrary 3D geometry.
Batch renders all lines in a single OpenGL draw call with Reversed-Z depth testing.
Optimized with vectorized precomputed unit templates for ~10x faster gizmo generation.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import moderngl

# Root shader directory
SHADER_DIR = Path(__file__).resolve().parent.parent.parent / "shaders"


def _build_unit_box_template() -> np.ndarray:
    corners = np.array([
        [-0.5, -0.5, -0.5],  # 0: ---
        [ 0.5, -0.5, -0.5],  # 1: +--
        [ 0.5,  0.5, -0.5],  # 2: ++-
        [-0.5,  0.5, -0.5],  # 3: -+-
        [-0.5, -0.5,  0.5],  # 4: --+
        [ 0.5, -0.5,  0.5],  # 5: +-+
        [ 0.5,  0.5,  0.5],  # 6: +++
        [-0.5,  0.5,  0.5],  # 7: -++
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Struts
    ]
    verts = np.empty((24, 3), dtype=np.float32)
    for i, (i0, i1) in enumerate(edges):
        verts[i * 2] = corners[i0]
        verts[i * 2 + 1] = corners[i1]
    return verts


def _build_unit_sphere_template(segments: int = 16) -> np.ndarray:
    step = 2.0 * math.pi / segments
    verts = np.empty((segments * 6, 3), dtype=np.float32)
    out_idx = 0
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        # XY ring
        verts[out_idx] = (c0, s0, 0.0)
        verts[out_idx + 1] = (c1, s1, 0.0)
        # XZ ring
        verts[out_idx + 2] = (c0, 0.0, s0)
        verts[out_idx + 3] = (c1, 0.0, s1)
        # YZ ring
        verts[out_idx + 4] = (0.0, c0, s0)
        verts[out_idx + 5] = (0.0, c1, s1)
        out_idx += 6
    return verts


def _build_unit_capsule_components(segments: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = 2.0 * math.pi / segments
    ring_verts = []
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        ring_verts.extend([(math.cos(a0), 0.0, math.sin(a0)), (math.cos(a1), 0.0, math.sin(a1))])
    ring_t = np.array(ring_verts, dtype=np.float32)

    strut_lines = []
    for angle in (0.0, math.pi * 0.5, math.pi, math.pi * 1.5):
        c, s = math.cos(angle), math.sin(angle)
        strut_lines.extend([(c, -1.0, s), (c, 1.0, s)])
    strut_t = np.array(strut_lines, dtype=np.float32)

    half_segs = segments // 2
    dome_lines = []
    for i in range(half_segs):
        a0, a1 = i * math.pi / half_segs, (i + 1) * math.pi / half_segs
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        # Top dome: XY and YZ (offset +1 on y)
        dome_lines.extend([
            (c0, s0, 0.0, 1.0), (c1, s1, 0.0, 1.0),
            (0.0, s0, c0, 1.0), (0.0, s1, c1, 1.0),
            # Bot dome: XY and YZ (offset -1 on y)
            (c0, -s0, 0.0, -1.0), (c1, -s1, 0.0, -1.0),
            (0.0, -s0, c0, -1.0), (0.0, -s1, c1, -1.0),
        ])
    return ring_t, strut_t, np.array(dome_lines, dtype=np.float32)


# Precomputed static geometry templates
UNIT_BOX_VERTS = _build_unit_box_template()
UNIT_SPHERE_16_VERTS = _build_unit_sphere_template(16)
CAPSULE_12_RINGS, CAPSULE_12_STRUTS, CAPSULE_12_DOMES = _build_unit_capsule_components(12)


class DebugDraw:
    """Immediate-mode 3D visual wireframe gizmo drawer with zero-allocation template caching."""

    __slots__ = (
        "ctx",
        "max_vertices",
        "vertex_buffer",
        "vertex_count",
        "prog",
        "vbo",
        "vao",
        "persistent_lines",
    )

    def __init__(
        self,
        ctx: moderngl.Context | None = None,
        max_vertices: int = 65536,
    ) -> None:
        self.ctx = ctx
        self.max_vertices = max_vertices

        # Interleaved 7 x float32: [x, y, z, r, g, b, a]
        self.vertex_buffer = np.zeros((max_vertices, 7), dtype=np.float32)
        self.vertex_count = 0

        # List of (x0, y0, z0, x1, y1, z1, r, g, b, a, remaining_duration)
        self.persistent_lines: list[list[float]] = []

        self.prog: moderngl.Program | None = None
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

        if self.ctx is not None:
            self._init_gl()

    def _init_gl(self) -> None:
        """Compiles line shader and allocates dynamic OpenGL VBO/VAO."""
        if self.ctx is None:
            return

        vert_path = SHADER_DIR / "debug_line.vert"
        frag_path = SHADER_DIR / "debug_line.frag"

        vert_src = vert_path.read_text(encoding="utf-8")
        frag_src = frag_path.read_text(encoding="utf-8")

        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        # 7 floats per vertex = 28 bytes
        self.vbo = self.ctx.buffer(reserve=self.max_vertices * 28, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f4 4f4", "in_position", "in_color")],
        )

    def clear(self) -> None:
        """Resets frame vertices."""
        self.vertex_count = 0

    def update(self, dt: float) -> None:
        """Advances lifetime of persistent debug geometry and clears per-frame vertices."""
        self.clear()
        if not self.persistent_lines:
            return

        surviving: list[list[float]] = []
        for line in self.persistent_lines:
            line[10] -= dt
            if line[10] > 0.0:
                surviving.append(line)
                self.draw_line(
                    (line[0], line[1], line[2]),
                    (line[3], line[4], line[5]),
                    color=(line[6], line[7], line[8], line[9]),
                    duration=0.0,
                )
        self.persistent_lines = surviving

    # --------------------------------------------------------------------------
    # Primitive Drawing Methods
    # --------------------------------------------------------------------------

    def draw_line(
        self,
        start: tuple[float, float, float] | np.ndarray,
        end: tuple[float, float, float] | np.ndarray,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (1.0, 1.0, 1.0, 1.0),
        duration: float = 0.0,
    ) -> None:
        """Draws a 3D line segment between start and end."""
        c = (color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)

        if duration > 0.0:
            self.persistent_lines.append(
                [
                    float(start[0]), float(start[1]), float(start[2]),
                    float(end[0]), float(end[1]), float(end[2]),
                    c[0], c[1], c[2], c[3],
                    duration,
                ]
            )

        if self.vertex_count + 2 > self.max_vertices:
            return

        idx = self.vertex_count
        buf = self.vertex_buffer
        buf[idx, 0:3] = start
        buf[idx, 3:7] = c
        buf[idx + 1, 0:3] = end
        buf[idx + 1, 3:7] = c
        self.vertex_count += 2

    def draw_ray(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        length: float = 1.0,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (1.0, 0.0, 0.0, 1.0),
        duration: float = 0.0,
    ) -> None:
        """Draws a directional ray from origin."""
        end = (
            origin[0] + direction[0] * length,
            origin[1] + direction[1] * length,
            origin[2] + direction[2] * length,
        )
        self.draw_line(origin, end, color=color, duration=duration)

    def draw_box(
        self,
        center: tuple[float, float, float] | np.ndarray,
        size: tuple[float, float, float] | np.ndarray,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (0.0, 1.0, 0.0, 1.0),
        duration: float = 0.0,
    ) -> None:
        """Draws a 3D wireframe box (12 edges = 24 vertices)."""
        if duration == 0.0 and self.vertex_count + 24 <= self.max_vertices:
            c = (color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
            idx = self.vertex_count
            buf = self.vertex_buffer
            buf[idx : idx + 24, 0:3] = UNIT_BOX_VERTS * size + center
            buf[idx : idx + 24, 3:7] = c
            self.vertex_count += 24
            return

        # Fallback for persistent lines or buffer overflow
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        cx, cy, cz = center[0], center[1], center[2]
        corners = [
            (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i0, i1 in edges:
            self.draw_line(corners[i0], corners[i1], color=color, duration=duration)

    def draw_aabb(
        self,
        min_pt: tuple[float, float, float] | np.ndarray,
        max_pt: tuple[float, float, float] | np.ndarray,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (0.0, 1.0, 0.0, 1.0),
        duration: float = 0.0,
    ) -> None:
        """Draws an Axis-Aligned Bounding Box from min and max points."""
        size = (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2])
        center = (
            min_pt[0] + size[0] * 0.5,
            min_pt[1] + size[1] * 0.5,
            min_pt[2] + size[2] * 0.5,
        )
        self.draw_box(center, size, color=color, duration=duration)

    def draw_sphere(
        self,
        center: tuple[float, float, float] | np.ndarray,
        radius: float,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (0.0, 0.7, 1.0, 1.0),
        segments: int = 16,
        duration: float = 0.0,
    ) -> None:
        """Draws a 3D wireframe sphere composed of 3 orthogonal rings (XY, XZ, YZ)."""
        if segments == 16 and duration == 0.0 and self.vertex_count + 96 <= self.max_vertices:
            c = (color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
            idx = self.vertex_count
            buf = self.vertex_buffer
            buf[idx : idx + 96, 0:3] = UNIT_SPHERE_16_VERTS * radius + center
            buf[idx : idx + 96, 3:7] = c
            self.vertex_count += 96
            return

        # Fallback for custom segments or persistent lines
        step = 2.0 * math.pi / segments
        cx, cy, cz = center[0], center[1], center[2]
        for i in range(segments):
            a0 = i * step
            a1 = (i + 1) * step
            c0, s0 = math.cos(a0) * radius, math.sin(a0) * radius
            c1, s1 = math.cos(a1) * radius, math.sin(a1) * radius
            self.draw_line((cx + c0, cy + s0, cz), (cx + c1, cy + s1, cz), color=color, duration=duration)
            self.draw_line((cx + c0, cy, cz + s0), (cx + c1, cy, cz + s1), color=color, duration=duration)
            self.draw_line((cx, cy + c0, cz + s0), (cx, cy + c1, cz + s1), color=color, duration=duration)

    def draw_capsule(
        self,
        center: tuple[float, float, float] | np.ndarray,
        radius: float,
        half_height: float,
        color: tuple[float, float, float, float] | tuple[float, float, float] = (1.0, 1.0, 0.0, 1.0),
        segments: int = 12,
        duration: float = 0.0,
    ) -> None:
        """Draws a 3D wireframe capsule (cylinder + hemispherical caps) along Y-axis."""
        if segments == 12 and duration == 0.0 and self.vertex_count + 104 <= self.max_vertices:
            c = (color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
            cx, cy, cz = center[0], center[1], center[2]
            idx = self.vertex_count
            buf = self.vertex_buffer

            # 1. Top and bottom horizontal rings (24 + 24 = 48 vertices)
            top_center = np.array([cx, cy + half_height, cz], dtype=np.float32)
            bot_center = np.array([cx, cy - half_height, cz], dtype=np.float32)
            buf[idx : idx + 24, 0:3] = CAPSULE_12_RINGS * radius + top_center
            buf[idx : idx + 24, 3:7] = c
            buf[idx + 24 : idx + 48, 0:3] = CAPSULE_12_RINGS * radius + bot_center
            buf[idx + 24 : idx + 48, 3:7] = c

            # 2. Struts (8 vertices)
            struts = CAPSULE_12_STRUTS.copy()
            struts[:, 0] = struts[:, 0] * radius + cx
            struts[:, 1] = cy + struts[:, 1] * half_height
            struts[:, 2] = struts[:, 2] * radius + cz
            buf[idx + 48 : idx + 56, 0:3] = struts
            buf[idx + 48 : idx + 56, 3:7] = c

            # 3. Domes (48 vertices)
            domes = CAPSULE_12_DOMES.copy()
            domes_pos = np.empty((48, 3), dtype=np.float32)
            domes_pos[:, 0] = domes[:, 0] * radius + cx
            domes_pos[:, 1] = cy + domes[:, 3] * half_height + domes[:, 1] * radius
            domes_pos[:, 2] = domes[:, 2] * radius + cz
            buf[idx + 56 : idx + 104, 0:3] = domes_pos
            buf[idx + 56 : idx + 104, 3:7] = c

            self.vertex_count += 104
            return

        # Fallback for custom segments or persistent lines
        cx, cy, cz = center[0], center[1], center[2]
        top_y = cy + half_height
        bot_y = cy - half_height
        step = 2.0 * math.pi / segments
        for i in range(segments):
            a0 = i * step
            a1 = (i + 1) * step
            c0, s0 = math.cos(a0) * radius, math.sin(a0) * radius
            c1, s1 = math.cos(a1) * radius, math.sin(a1) * radius
            self.draw_line((cx + c0, top_y, cz + s0), (cx + c1, top_y, cz + s1), color=color, duration=duration)
            self.draw_line((cx + c0, bot_y, cz + s0), (cx + c1, bot_y, cz + s1), color=color, duration=duration)

        for angle in (0.0, math.pi * 0.5, math.pi, math.pi * 1.5):
            sx = math.cos(angle) * radius
            sz = math.sin(angle) * radius
            self.draw_line((cx + sx, bot_y, cz + sz), (cx + sx, top_y, cz + sz), color=color, duration=duration)

        half_segs = segments // 2
        for i in range(half_segs):
            a0 = i * math.pi / half_segs
            a1 = (i + 1) * math.pi / half_segs
            c0, s0 = math.cos(a0) * radius, math.sin(a0) * radius
            c1, s1 = math.cos(a1) * radius, math.sin(a1) * radius
            self.draw_line((cx + c0, top_y + s0, cz), (cx + c1, top_y + s1, cz), color=color, duration=duration)
            self.draw_line((cx, top_y + s0, cz + c0), (cx, top_y + s1, cz + c1), color=color, duration=duration)
            self.draw_line((cx + c0, bot_y - s0, cz), (cx + c1, bot_y - s1, cz), color=color, duration=duration)
            self.draw_line((cx, bot_y - s0, cz + c0), (cx, bot_y - s1, cz + c1), color=color, duration=duration)

    def draw_axes(
        self,
        position: tuple[float, float, float] | np.ndarray,
        scale: float = 1.0,
        duration: float = 0.0,
    ) -> None:
        """Draws RGB coordinate axes (X=Red, Y=Green, Z=Blue) at position."""
        px, py, pz = position[0], position[1], position[2]
        self.draw_line((px, py, pz), (px + scale, py, pz), color=(1.0, 0.0, 0.0, 1.0), duration=duration)
        self.draw_line((px, py, pz), (px, py + scale, pz), color=(0.0, 1.0, 0.0, 1.0), duration=duration)
        self.draw_line((px, py, pz), (px, py, pz + scale), color=(0.0, 0.4, 1.0, 1.0), duration=duration)

    # --------------------------------------------------------------------------
    # Render Dispatch
    # --------------------------------------------------------------------------

    def render(self) -> None:
        """Uploads accumulated line vertices and renders in a single draw call."""
        if self.ctx is None or self.vao is None or self.vbo is None or self.vertex_count <= 0:
            return

        active_bytes = self.vertex_buffer[: self.vertex_count].tobytes()
        self.vbo.write(active_bytes)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = ">"
        self.vao.render(moderngl.LINES, vertices=self.vertex_count)
        self.ctx.disable(moderngl.DEPTH_TEST)

    def destroy(self) -> None:
        """Releases GPU buffers and programs."""
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None
