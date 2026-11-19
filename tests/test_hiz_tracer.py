"""Validation harness for the shared Hi-Z hierarchical tracer.

Builds a deterministic synthetic scene (a wall, a floor, a thin pillar and open
sky), builds the real RG32F bounds hierarchy on the GPU, then traces a fixed set
of rays and asserts on the reported hit data.
"""

import numpy as np
import pytest

from engine.core.math_utils import mat4_inv, matrix_perspective
from engine.gfx.context import RenderContext
from engine.gfx.passes.hiz_pass import HiZPass
from engine.gfx.shader_utils import load_shader

NEAR = 0.1
FAR = 2000.0
FOVY = np.radians(60.0)
WIDTH, HEIGHT = 64, 64

MISS_NONE = 0
MISS_LEFT_SCREEN = 1
MISS_EXCEEDED_DIST = 2
MISS_EXCEEDED_ITERS = 3
MISS_INVALID_PROJ = 4
MISS_NO_INTERSECTION = 5


def _projection():
    return matrix_perspective(FOVY, WIDTH / HEIGHT, NEAR, FAR, reverse_z=True)


def _raw_depth_for_view_z(view_z, proj):
    clip = proj.reshape((4, 4), order="F") @ np.array([0.0, 0.0, view_z, 1.0])
    return float(clip[2] / clip[3])


def _build_scene():
    """Returns a reversed-Z depth image with a wall, floor, pillar and sky.

    View space: camera at origin looking down -Z.
      * Wall at z = -20 covering the whole frame.
      * Floor plane y = -2 (nearer at the bottom of the screen).
      * Thin pillar at x in [-0.35, 0.35], z = -6.
      * Sky where nothing is hit.
    """
    proj = _projection()
    inv_proj = mat4_inv(proj)
    depths = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            uv = np.array([(x + 0.5) / WIDTH, (y + 0.5) / HEIGHT])
            ndc = uv * 2.0 - 1.0
            # Direction of the pixel ray in view space (z = -1 plane).
            ray = inv_proj.reshape((4, 4), order="F") @ np.array([ndc[0], ndc[1], 1.0, 1.0])
            ray = ray[:3] / ray[3]
            ray = ray / -ray[2]  # normalise so z = -1

            best = None
            # Wall at z = -20
            t_wall = 20.0
            best = t_wall
            # Floor y = -2
            if ray[1] < -1e-6:
                t_floor = -2.0 / ray[1]
                if 0.5 < t_floor < best:
                    best = t_floor
            # Thin pillar: |x| <= 0.35 at z = -6, offset to the right of centre
            # so the screen centre still sees the wall behind it.
            if abs(ray[0] * 6.0 - 1.2) <= 0.35:
                if 6.0 < best:
                    best = 6.0

            depths[y, x] = _raw_depth_for_view_z(-best, proj)

    # Carve an open-sky region in the top-right corner.
    depths[: HEIGHT // 3, 2 * WIDTH // 3 :] = 0.0
    return depths, proj, inv_proj


class TestHiZTracer:
    @pytest.fixture(scope="class")
    def scene(self):
        ctx = RenderContext.create_headless(width=WIDTH, height=HEIGHT)
        depths, proj, inv_proj = _build_scene()

        pass_instance = HiZPass(ctx.ctx, width=WIDTH, height=HEIGHT, reverse_z=True)
        depth_tex = ctx.ctx.depth_texture((WIDTH, HEIGHT))
        depth_tex.filter = (0, 0)
        depth_tex.write(depths.tobytes())

        class _FrameCtx:
            inv_proj_mat = inv_proj

        class _GraphCtx:
            resources = {}
            frame_context = _FrameCtx()

        pass_instance.execute(_GraphCtx(), depth_texture=depth_tex)

        yield ctx, pass_instance, depth_tex, proj, inv_proj

        depth_tex.release()
        pass_instance.destroy()
        ctx.destroy()

    def _trace(self, scene, rays):
        ctx, pass_instance, depth_tex, proj, inv_proj = scene
        count = len(rays)

        out_hit = ctx.ctx.texture((count, 1), 4, dtype="f4")
        out_trace = ctx.ctx.texture((count, 1), 4, dtype="f4")
        out_hit.bind_to_image(0, read=False, write=True)
        out_trace.bind_to_image(1, read=False, write=True)

        prog = ctx.ctx.compute_shader(load_shader("hiz_trace_test.comp"))
        prog["u_Projection"].write(proj.tobytes())
        prog["u_InvProjection"].write(inv_proj.tobytes())
        prog["u_RayCount"].value = count

        origin = np.zeros((64, 3), dtype=np.float32)
        direction = np.zeros((64, 3), dtype=np.float32)
        min_d = np.zeros(64, dtype=np.float32)
        max_d = np.zeros(64, dtype=np.float32)
        thickness = np.zeros(64, dtype=np.float32)
        max_iter = np.zeros(64, dtype=np.int32)
        for i, r in enumerate(rays):
            origin[i] = r["origin"]
            direction[i] = r["direction"]
            min_d[i] = r.get("min_distance", 0.05)
            max_d[i] = r.get("max_distance", 60.0)
            thickness[i] = r.get("thickness", 0.25)
            max_iter[i] = r.get("max_iterations", 128)

        prog["u_Origin"].write(origin.tobytes())
        prog["u_Direction"].write(direction.tobytes())
        prog["u_MinDistance"].write(min_d.tobytes())
        prog["u_MaxDistance"].write(max_d.tobytes())
        prog["u_Thickness"].write(thickness.tobytes())
        prog["u_MaxIterations"].write(max_iter.tobytes())

        pass_instance.hiz_bounds_texture.use(location=0)
        depth_tex.use(location=1)
        prog.run(group_x=count, group_y=1)

        hit = np.frombuffer(out_hit.read(), dtype=np.float32).reshape((count, 4))
        trace = np.frombuffer(out_trace.read(), dtype=np.float32).reshape((count, 4))

        prog.release()
        out_hit.release()
        out_trace.release()
        return hit, trace

    def test_perpendicular_wall_hit(self, scene):
        # Straight ahead from the origin: must hit the wall at z = -20.
        hit, trace = self._trace(scene, [{"origin": (0.0, 0.0, 0.0), "direction": (0.0, 0.0, -1.0)}])
        assert hit[0, 0] == 1.0
        assert hit[0, 1] == pytest.approx(0.5, abs=0.02)
        assert hit[0, 2] == pytest.approx(0.5, abs=0.02)
        assert hit[0, 3] == pytest.approx(20.0, abs=0.5)
        assert trace[0, 3] == MISS_NONE

    def test_floor_hit(self, scene):
        # Downward ray from above the floor plane.
        hit, trace = self._trace(scene, [{"origin": (0.0, 1.0, -4.0), "direction": (0.0, -1.0, -0.2)}])
        assert hit[0, 0] == 1.0
        assert trace[0, 3] == MISS_NONE
        assert hit[0, 3] > 0.0

    def test_empty_sky_miss(self, scene):
        # Aim into the carved sky region (bottom-right in view space).
        hit, trace = self._trace(
            scene,
            [{"origin": (0.0, 0.0, 0.0), "direction": (0.55, -0.55, -1.0), "max_distance": 40.0}],
        )
        assert hit[0, 0] == 0.0
        assert trace[0, 3] in (MISS_LEFT_SCREEN, MISS_EXCEEDED_DIST, MISS_NO_INTERSECTION, MISS_EXCEEDED_ITERS)

    def test_off_screen_escape(self, scene):
        # Ray pointing away from the screen centre exits the frustum.
        hit, trace = self._trace(scene, [{"origin": (0.0, 0.0, -1.0), "direction": (3.0, 0.0, -0.2)}])
        assert hit[0, 0] == 0.0
        assert trace[0, 3] == MISS_LEFT_SCREEN

    def test_grazing_ray(self, scene):
        # Nearly parallel to the wall: must terminate, never loop forever.
        hit, trace = self._trace(
            scene,
            [{"origin": (0.0, 0.0, -1.0), "direction": (1.0, 0.0, -0.02), "max_distance": 30.0}],
        )
        assert trace[0, 1] <= 128
        assert trace[0, 3] in (MISS_NONE, MISS_LEFT_SCREEN, MISS_EXCEEDED_DIST, MISS_EXCEEDED_ITERS)

    def test_thin_object_hit(self, scene):
        # The pillar is only 0.7 m wide at z = -6, offset to the right.
        hit, trace = self._trace(
            scene,
            [{"origin": (0.0, 0.0, 0.0), "direction": (0.2, 0.0, -1.0), "max_distance": 30.0}],
        )
        assert hit[0, 0] == 1.0
        assert hit[0, 3] == pytest.approx(6.0, abs=0.6)

    def test_origin_close_to_surface(self, scene):
        # Start just in front of the wall.
        hit, trace = self._trace(
            scene,
            [{"origin": (0.0, 0.0, -19.5), "direction": (0.0, 0.0, -1.0), "min_distance": 0.05}],
        )
        assert hit[0, 0] == 1.0
        assert hit[0, 3] == pytest.approx(20.0, abs=0.5)

    def test_max_iteration_safeguard(self, scene):
        # A tiny budget must terminate with the iteration miss reason.
        hit, trace = self._trace(
            scene,
            [{"origin": (0.0, 0.0, 0.0), "direction": (0.0, 0.0, -1.0), "max_iterations": 1}],
        )
        assert trace[0, 1] == 1.0
        assert trace[0, 3] == MISS_EXCEEDED_ITERS

    def test_invalid_direction(self, scene):
        hit, trace = self._trace(scene, [{"origin": (0.0, 0.0, 0.0), "direction": (0.0, 0.0, 0.0)}])
        assert hit[0, 0] == 0.0
        assert trace[0, 3] == MISS_INVALID_PROJ

    def test_all_rays_terminate(self, scene):
        rays = []
        for i in range(16):
            angle = i * 0.4
            rays.append(
                {
                    "origin": (0.0, 0.0, 0.0),
                    "direction": (np.cos(angle), np.sin(angle) * 0.5, -1.0),
                    "max_distance": 50.0,
                    "max_iterations": 96,
                }
            )
        hit, trace = self._trace(scene, rays)
        assert np.all(trace[:, 1] <= 96)
        assert np.all(np.isfinite(hit))
        assert np.all(np.isfinite(trace))
