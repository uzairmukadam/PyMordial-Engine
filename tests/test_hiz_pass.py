"""Unit tests for GPU Hierarchical-Z (Hi-Z) Depth Pyramid Pass."""

import math

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.hiz_pass import HiZPass
from engine.gfx.pipeline import RenderPipeline


class TestHiZPass:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_hiz_pass_initialization_and_mip_sizes(self, render_ctx: RenderContext):
        pass_instance = HiZPass(render_ctx.ctx, width=320, height=240, reverse_z=True)
        expected_mips = int(math.floor(math.log2(320))) + 1
        assert pass_instance.num_mips == expected_mips
        assert len(pass_instance.mip_sizes) == expected_mips
        assert pass_instance.mip_sizes[0] == (320, 240)
        assert pass_instance.mip_sizes[1] == (160, 120)
        assert pass_instance.mip_sizes[2] == (80, 60)
        assert pass_instance.hiz_texture is not None
        assert pass_instance.hiz_texture.size == (320, 240)
        assert pass_instance.hiz_texture.components == 1
        pass_instance.destroy()

    def test_hiz_pass_resize(self, render_ctx: RenderContext):
        pass_instance = HiZPass(render_ctx.ctx, width=320, height=240, reverse_z=True)
        pass_instance.resize(640, 480)
        expected_mips = int(math.floor(math.log2(640))) + 1
        assert pass_instance.width == 640
        assert pass_instance.height == 480
        assert pass_instance.num_mips == expected_mips
        assert pass_instance.mip_sizes[0] == (640, 480)
        assert pass_instance.hiz_texture.size == (640, 480)
        pass_instance.destroy()

    def test_conservative_min_reduction(self, render_ctx: RenderContext):
        """Verify conservative min-depth reduction for Reversed-Z."""
        ctx = render_ctx.ctx
        w, h = 4, 4
        pass_instance = HiZPass(ctx, width=w, height=h, reverse_z=True)

        # Create known 4x4 test depth values in Reversed-Z (1.0 = near, 0.0 = far)
        # Block 0 (top-left 2x2): [0.9, 0.8; 0.7, 0.6] -> min = 0.6
        # Block 1 (top-right 2x2): [0.5, 0.5; 0.5, 0.2] -> min = 0.2
        # Block 2 (bottom-left 2x2): [0.95, 0.95; 0.95, 0.95] -> min = 0.95
        # Block 3 (bottom-right 2x2): [0.4, 0.3; 0.2, 0.1] -> min = 0.1
        test_depths = np.array(
            [
                [0.90, 0.80, 0.50, 0.50],
                [0.70, 0.60, 0.50, 0.20],
                [0.95, 0.95, 0.40, 0.30],
                [0.95, 0.95, 0.20, 0.10],
            ],
            dtype=np.float32,
        )

        # Upload to mip 0 of hiz_texture
        pass_instance.hiz_texture.write(test_depths.tobytes(), level=0)

        # Execute downsample compute shader for mip 1 (size 2x2)
        pass_instance.hiz_texture.use(location=0)
        pass_instance._u_down_src_level.value = 0
        pass_instance._u_down_reverse_z.value = 1
        pass_instance.hiz_texture.bind_to_image(0, read=False, write=True, level=1)
        pass_instance.downsample_prog.run(group_x=1, group_y=1)

        # Read back level 1 data (2x2 float32)
        fbo = ctx.framebuffer(color_attachments=[pass_instance.hiz_texture])
        raw_m1 = pass_instance.hiz_texture.read(level=1)
        m1_vals = np.frombuffer(raw_m1, dtype=np.float32).reshape((2, 2))
        fbo.release()

        # In Reversed-Z, conservative reduction must produce the minimum depth
        assert pytest.approx(m1_vals[0, 0], abs=1e-3) == 0.60
        assert pytest.approx(m1_vals[0, 1], abs=1e-3) == 0.20
        assert pytest.approx(m1_vals[1, 0], abs=1e-3) == 0.95
        assert pytest.approx(m1_vals[1, 1], abs=1e-3) == 0.10
        pass_instance.destroy()

    def test_hiz_execute_graph_context(self, render_ctx: RenderContext):
        """Verify execute sets hiz_texture and hiz_mip_count in RenderGraphContext."""
        pass_instance = HiZPass(render_ctx.ctx, width=64, height=64, reverse_z=True)
        dummy_depth = render_ctx.ctx.depth_texture((64, 64))

        class MockGraphContext:
            def __init__(self):
                self.resources = {}

        graph_ctx = MockGraphContext()
        pass_instance.execute(graph_ctx, depth_texture=dummy_depth)

        assert "hiz_texture" in graph_ctx.resources
        assert graph_ctx.resources["hiz_texture"] == pass_instance.hiz_texture
        assert "hiz_mip_count" in graph_ctx.resources
        assert graph_ctx.resources["hiz_mip_count"] == pass_instance.num_mips

        dummy_depth.release()
        pass_instance.destroy()

    def test_pipeline_integration_headless(self, render_ctx: RenderContext):
        """Verify RenderPipeline runs HiZPass seamlessly without errors."""
        pipeline = RenderPipeline(render_ctx)
        assert hasattr(pipeline, "hiz_pass")
        assert pipeline.hiz_pass is not None

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))

        # Render one frame
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.016,
        )

        # Verify output texture exists and pipeline rendered
        assert pipeline.output_texture_id > 0
        assert pipeline.hiz_pass.hiz_texture is not None

    def test_cull_indirect_shader_compilation(self, render_ctx: RenderContext):
        """Verify GPU-driven occlusion culling compute shader (cull_indirect.comp) compiles."""
        from engine.gfx.shader_utils import load_shader

        src = load_shader("cull_indirect.comp")
        prog = render_ctx.ctx.compute_shader(src)
        assert prog is not None
        prog.release()

    def test_hiz_gpu_culling_execution(self, render_ctx: RenderContext):
        """Validates that HiZPass.cull_aabbs_gpu executes and filters occluded geometry."""
        pass_instance = HiZPass(render_ctx.ctx, width=320, height=240, reverse_z=True)

        aabbs = np.zeros((3, 6), dtype=np.float32)
        # Entity 0: Centered at (0, 0, 0), size 1.0 (inside near frustum)
        aabbs[0, 0:3] = [0.0, 0.0, 0.0]
        aabbs[0, 3:6] = [0.5, 0.5, 0.5]

        # Entity 1: Way behind camera at (0, 0, 100)
        aabbs[1, 0:3] = [0.0, 0.0, 100.0]
        aabbs[1, 3:6] = [0.5, 0.5, 0.5]

        # Entity 2: Off to the far right outside frustum at (100, 0, 0)
        aabbs[2, 0:3] = [100.0, 0.0, 0.0]
        aabbs[2, 3:6] = [0.5, 0.5, 0.5]

        vp = np.eye(4, dtype=np.float32)
        mask = pass_instance.cull_aabbs_gpu(
            aabbs=aabbs,
            active_count=3,
            vp_matrix=vp,
            screen_size=(320, 240),
        )

        assert len(mask) == 3
        # Entity 0 should be visible
        assert mask[0]
        # Entity 1 and 2 outside frustum
        assert not mask[1]
        assert not mask[2]

        pass_instance.destroy()

    def test_hiz_culling_conservative_under_camera_motion(self, render_ctx: RenderContext):
        """F2 regression: a partially-visible object must never be culled when the
        camera moves between the frame that built the Hi-Z and the frame that culls.

        The Hi-Z pyramid is built from the previous frame's depth, so the occlusion
        test must run in the previous frame's VP space with the AABB expanded by the
        camera translation. Before the fix, culling used the current VP against the
        stale pyramid and produced false occlusion.
        """
        import math

        from engine.core.math_utils import mat4_mul, matrix_look_at, matrix_perspective

        w, h = 320, 240
        pass_instance = HiZPass(render_ctx.ctx, width=w, height=h, reverse_z=True)

        def make_vp(cam_x: float) -> np.ndarray:
            view = matrix_look_at((cam_x, 0.0, 0.0), (cam_x, 0.0, -1.0), (0.0, 1.0, 0.0))
            proj = matrix_perspective(math.radians(60), w / h, 0.1, 1000.0, reverse_z=True)
            return mat4_mul(proj, view).astype(np.float32)

        def project(vp_flat: np.ndarray, pt) -> tuple[float, float, float]:
            vp = vp_flat.reshape(4, 4, order="F")
            p = vp @ np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32)
            ndc = p[:3] / p[3]
            return ndc[0] * 0.5 + 0.5, ndc[1] * 0.5 + 0.5, ndc[2]

        wall_c, wall_e = (0.0, 0.0, -10.0), (2.0, 3.0, 0.1)
        obj_c, obj_e = (1.8, 0.0, -20.0), (0.5, 0.5, 0.5)

        # Build the previous-frame Hi-Z from a wall rasterised with the previous VP.
        vp_prev = make_vp(0.0)
        raw = np.zeros((h, w), dtype=np.float32)
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                pt = (wall_c[0] + sx * wall_e[0], wall_c[1] + sy * wall_e[1], wall_c[2] + wall_e[2])
                u, v, z = project(vp_prev, pt)
                corners.append((u, v, z))
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        zs = [c[2] for c in corners]
        u0, u1 = int(max(0, min(us) * w)), int(min(w, max(us) * w))
        v0, v1 = int(max(0, min(vs) * h)), int(min(h, max(vs) * h))
        if u1 > u0 and v1 > v0:
            raw[v0:v1, u0:u1] = max(zs)
        pass_instance.hiz_texture.write(raw.tobytes(), level=0)
        pass_instance.hiz_texture.use(location=0)
        for k in range(pass_instance.num_mips - 1):
            pass_instance._u_down_src_level.value = k
            pass_instance._u_down_reverse_z.value = 1
            pass_instance.hiz_texture.bind_to_image(0, read=False, write=True, level=k + 1)
            gx, gy = pass_instance.dispatch_groups[k + 1]
            pass_instance.downsample_prog.run(group_x=gx, group_y=gy)

        # Camera translates right by 3 units; the object now peeks past the wall edge.
        step = 3.0
        vp_cur = make_vp(step)
        aabbs = np.zeros((1, 6), dtype=np.float32)
        aabbs[0, 0:3] = obj_c
        aabbs[0, 3:6] = obj_e

        mask = pass_instance.cull_aabbs_gpu(
            aabbs=aabbs,
            active_count=1,
            vp_matrix=vp_cur,
            screen_size=(w, h),
            prev_vp_matrix=vp_prev,
            camera_delta=(step, 0.0, 0.0),
        )
        assert bool(mask[0]), "partially-visible object was falsely occluded under camera motion"

        pass_instance.destroy()
