"""Tests for the explicit screen-space radiance cascade resource model."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.radiance_cascades_pass import RadianceCascadesPass
from engine.gfx.passes.rc_cascade import (
    RC_MAX_DIRECTIONS,
    RCCascadeConfigError,
    build_cascades,
)
from engine.gfx.pipeline import RenderPipeline


class TestCascadeResources:
    @pytest.fixture
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_three_cascades_with_denser_near_probes(self, render_ctx: RenderContext):
        rc = RadianceCascadesPass(render_ctx.ctx, width=320, height=240)
        try:
            assert len(rc.cascades) == 3
            c0, c1, c2 = rc.cascades
            # Near cascades must have denser spatial probes.
            assert c0.probe_spacing < c1.probe_spacing < c2.probe_spacing
            assert c0.probe_width > c1.probe_width > c2.probe_width
            # Farther cascades must have more directions.
            assert c0.direction_count < c1.direction_count < c2.direction_count
            # Intervals must be contiguous and increasing.
            assert c0.interval_start == 0.0
            assert c0.interval_end == c1.interval_start
            assert c1.interval_end == c2.interval_start
        finally:
            rc.destroy()

    def test_texture_array_layer_counts(self, render_ctx: RenderContext):
        rc = RadianceCascadesPass(render_ctx.ctx, width=320, height=240)
        try:
            for cascade in rc.cascades:
                assert cascade.interval_texture.layers == cascade.direction_count
                assert cascade.merged_texture.layers == cascade.direction_count
                assert cascade.interval_texture.size[:2] == (cascade.probe_width, cascade.probe_height)
                assert cascade.direction_count <= RC_MAX_DIRECTIONS
        finally:
            rc.destroy()

    def test_dispatch_dimensions(self, render_ctx: RenderContext):
        rc = RadianceCascadesPass(render_ctx.ctx, width=320, height=240)
        try:
            for cascade in rc.cascades:
                gx, gy, gz = cascade.dispatch_groups
                assert gx == (cascade.probe_width + 7) // 8
                assert gy == (cascade.probe_height + 7) // 8
                assert gz == cascade.direction_count
        finally:
            rc.destroy()

    def test_repeated_resize_reallocates_cascades(self, render_ctx: RenderContext):
        rc = RadianceCascadesPass(render_ctx.ctx, width=320, height=240)
        try:
            for size in ((640, 480), (160, 120), (320, 240), (321, 241)):
                rc.resize(*size)
                assert rc.width == size[0]
                assert rc.height == size[1]
                for cascade in rc.cascades:
                    assert cascade.interval_texture.size[:2] == (cascade.probe_width, cascade.probe_height)
                    assert cascade.interval_texture.layers == cascade.direction_count
                assert rc._first_frame is True
        finally:
            rc.destroy()

    def test_invalid_configuration_is_rejected(self, render_ctx: RenderContext):
        with pytest.raises(RCCascadeConfigError):
            build_cascades(render_ctx.ctx, 320, 240, ((0.0, 1.0),), spacings=(0,), directions=(8,))
        with pytest.raises(RCCascadeConfigError):
            build_cascades(render_ctx.ctx, 320, 240, ((0.0, 1.0),), spacings=(4,), directions=(0,))
        with pytest.raises(RCCascadeConfigError):
            build_cascades(render_ctx.ctx, 320, 240, ((1.0, 0.5),), spacings=(4,), directions=(8,))
        with pytest.raises(RCCascadeConfigError):
            build_cascades(render_ctx.ctx, 320, 240, ((0.0, 1.0),), spacings=(4, 8), directions=(8,))


class TestCascadeExecution:
    @pytest.fixture
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def _scene(self, ecs: EntityManager) -> None:
        ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))

    def test_cascade_backend_produces_finite_irradiance(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.gi_mode = "RADIANCE_CASCADES"
            pipeline.config.rc_backend = "CASCADES"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            for i in range(4):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))

            data = np.frombuffer(pipeline.rc_pass.resolved_tex.read(), dtype=np.float16).reshape(240, 320, 4)
            assert np.all(np.isfinite(data))
            assert data[..., :3].max() >= 0.0
        finally:
            pipeline.destroy()

    def test_legacy_backend_still_default(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.config.rc_backend == "LEGACY"
            pipeline.config.gi_mode = "RADIANCE_CASCADES"
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.rc_pass._first_frame is False
        finally:
            pipeline.destroy()

    def test_gi_mode_switch_invalidates_history(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            self._scene(ecs)
            pipeline.config.gi_mode = "RADIANCE_CASCADES"
            pipeline.config.rc_backend = "CASCADES"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.rc_pass._first_frame is False

            pipeline.config.gi_mode = "OFF"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.032)
            assert pipeline.rc_pass._first_frame is True
        finally:
            pipeline.destroy()

    def test_resize_matches_fresh_allocation(self):
        """Regression: the deepest cascade's merged texture is never written.

        The first merge must read the deepest cascade's *interval* texture, not
        its (unwritten) merged texture. Reading the unwritten texture happened to
        work on a fresh allocation because the memory was zeroed, but after a
        resize the recycled memory contained garbage and irradiance exploded.
        """

        def build(width, height, resize_to=None):
            ctx = RenderContext.create_headless(width=width, height=height)
            pipeline = RenderPipeline(ctx)
            try:
                pipeline.config.gi_mode = "RADIANCE_CASCADES"
                pipeline.config.rc_backend = "CASCADES"
                ecs = EntityManager(max_entities=10)
                self._scene(ecs)
                for i in range(3):
                    pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
                if resize_to is not None:
                    pipeline.resize(*resize_to)
                    for i in range(3):
                        pipeline.render_frame(
                            ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.2 + i * 0.016
                        )
                rc = pipeline.rc_pass
                data = np.frombuffer(rc.irradiance_tex.read(), dtype=np.float16).reshape(
                    rc.height, rc.width, 4
                )
                return float(data[..., :3].mean())
            finally:
                pipeline.destroy()
                ctx.destroy()

        fresh = build(321, 241)
        resized = build(320, 240, (321, 241))
        assert fresh > 0.0
        # The resized path must land in the same range as a fresh allocation.
        assert abs(resized - fresh) < max(0.01, fresh * 0.5), (
            f"resize produced {resized:.4f} vs fresh {fresh:.4f}"
        )
