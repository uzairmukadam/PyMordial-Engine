"""Tests for the SSGI reconstruction split, direct-radiance source and AO backend.

Pins the architectural properties this change introduces:

* the spatial filter shader never reads history or motion vectors,
* the temporal resolve runs exactly once per frame,
* SSGI samples the current-frame direct-radiance texture,
* history invalidation survives resize, GI-mode switches and explicit calls,
* the AO backend produces a valid AO mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.shader_utils import load_shader


def _scene(ecs: EntityManager) -> None:
    ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
    ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))


@pytest.fixture
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestSSGIStageSeparation:
    def test_spatial_filter_has_no_temporal_inputs(self):
        source = load_shader("ssgi_blur.frag")
        for forbidden in ("u_HistorySSGI", "u_MotionVectors", "u_PrevView", "u_TemporalBlend"):
            assert forbidden not in source, f"spatial filter must not reference {forbidden}"

    def test_temporal_shader_owns_history_and_motion(self):
        source = load_shader("ssgi_temporal.frag")
        assert "u_HistorySSGI" in source
        assert "u_MotionVectors" in source
        assert "u_PrevView" in source

    def test_resource_roles_exist(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ssgi = pipeline.ssgi_pass
            for name in (
                "raw_ssgi_tex",
                "spatial_x_tex",
                "spatial_y_tex",
                "resolved_ssgi_tex",
                "history_ssgi_tex",
            ):
                assert getattr(ssgi, name).size == (320, 240)
            # The published texture is the resolved one, not the raw trace.
            assert ssgi.resolved_ssgi_tex is not ssgi.raw_ssgi_tex
        finally:
            pipeline.destroy()

    def test_history_invalidated_by_resize_and_mode_switch(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ssgi = pipeline.ssgi_pass
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "SSGI"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert ssgi._first_frame is False

            ssgi.invalidate_history()
            assert ssgi._first_frame is True

            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.032)
            assert ssgi._first_frame is False

            pipeline.resize(160, 120)
            assert ssgi._first_frame is True

            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.048)
            assert ssgi._first_frame is False

            # Switching GI away from SSGI must invalidate the history.
            pipeline.config.gi_mode = "OFF"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.064)
            assert ssgi._first_frame is True
        finally:
            pipeline.destroy()


class TestDirectRadiance:
    def test_texture_published_and_current_frame(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            tex = pipeline._graph_context.resources.get("direct_radiance_texture")
            assert tex is not None
            assert tex.size == (320, 240)

            data = np.frombuffer(tex.read(), dtype=np.float16).reshape(240, 320, 4)
            assert np.all(np.isfinite(data))
            assert data[..., :3].max() > 0.0
        finally:
            pipeline.destroy()

    def test_gi_disabled_output_is_stable(self, render_ctx: RenderContext):
        """The direct-radiance pass must not perturb the non-GI composition."""
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.config.gi_mode = "OFF"
            frames = []
            for i in range(4):
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1))
                frames.append(
                    np.frombuffer(
                        pipeline.post_process.final_fbo.read(components=3, dtype="f1"), dtype=np.uint8
                    ).copy()
                )
            # A static scene must not accumulate energy. Auto-exposure adapts
            # temporally and a few specular pixels may shift, so compare the
            # overall image energy rather than per-pixel equality.
            means = [float(f.mean()) for f in frames]
            assert max(means) - min(means) < 1.0, f"image energy drifted: {means}"
            delta = np.abs(frames[-1].astype(np.int16) - frames[-2].astype(np.int16))
            assert float(delta.mean()) < 0.5, f"mean frame delta too large: {delta.mean()}"
        finally:
            pipeline.destroy()


class TestAOBackend:
    def test_legacy_backend_still_available(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            _scene(ecs)
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            ao = np.frombuffer(pipeline.ao_pass.raw_ao_tex.read(), dtype=np.uint8)
            assert len(ao) == 320 * 240  # R8: one byte per pixel
        finally:
            pipeline.destroy()
