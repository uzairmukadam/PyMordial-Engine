"""Tests for Horizon 2 CSM per-cascade culling and sloped bias (H2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset
from engine.gfx.shadow_csm import CascadedShadowMap


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


class TestSlopeBiasConfiguration:
    def test_bias_bounds_are_declared(self) -> None:
        cfg = RenderConfig()
        assert cfg.shadow_bias_slope_scale > 0.0
        assert 0.0 < cfg.shadow_bias_min_scale < cfg.shadow_bias_max_scale

    def test_bias_bounds_are_tunable_per_preset(self) -> None:
        low = get_quality_preset(GraphicsQuality.LOW)
        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        # Both presets must declare the fields so EngineTweaks writes are not shadowed.
        assert hasattr(low, "shadow_bias_slope_scale")
        assert hasattr(cinematic, "shadow_bias_slope_scale")

    def test_shader_declares_bias_uniforms(self) -> None:
        from pathlib import Path

        shader = (Path(__file__).resolve().parent.parent / "shaders" / "deferred_resolve.frag").read_text(
            encoding="utf-8"
        )
        assert "u_ShadowBiasSlopeScale" in shader
        assert "u_ShadowBiasMinScale" in shader
        assert "u_ShadowBiasMaxScale" in shader
        # The slope term must be the documented tan(acos(N.L)) form.
        assert "cos_theta" in shader and "slope" in shader


class TestPerCascadeCulling:
    def test_cull_cascades_returns_one_list_per_cascade(self, render_ctx: RenderContext) -> None:
        csm = CascadedShadowMap(render_ctx.ctx, atlas_size=1024, cascade_count=4, max_distance=100.0)
        try:
            aabbs = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, -50.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            batches = [("cube", 2, 0, False, True)]
            result = csm.cull_cascades(aabbs=aabbs, active_count=2, draw_batches=batches, cascade_count=4)
            assert len(result) == 4
        finally:
            csm.destroy()

    def test_lateral_caster_is_culled_from_cascades(self, render_ctx: RenderContext) -> None:
        """A caster far outside the cascade's lateral bounds must not be submitted.

        Casters behind a cascade are deliberately retained (they can cast into
        it), so this uses a caster displaced far to the side instead.
        """
        cfg = get_quality_preset(GraphicsQuality.HIGH)
        cfg.shadow_distance = 60.0
        pipeline = RenderPipeline(render_ctx, cfg)
        try:
            ecs = EntityManager(max_entities=64)
            # Distinct shapes so each caster forms its own draw batch; culling is
            # evaluated per batch, not per instance.
            ecs.create_entity(position=(0.0, 1.0, -5.0), shape="cube")
            ecs.create_entity(position=(5000.0, 1.0, -5.0), shape="sphere")

            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 0.0),
                camera_target=(0.0, 2.0, -1.0),
                time_elapsed=0.1,
            )

            near_commands = pipeline.cascade_shadow_mdis[0].command_count
            total_commands = pipeline.stats_shadow_commands
            unculled = pipeline.stats_shadow_commands_unculled

            assert near_commands >= 1
            # Culling must remove at least one command across the cascades.
            assert total_commands < unculled
        finally:
            pipeline.destroy()

    def test_shadow_stats_are_populated(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx, get_quality_preset(GraphicsQuality.HIGH))
        try:
            ecs = EntityManager(max_entities=16)
            ecs.create_entity(position=(0.0, 1.0, -5.0), shape="cube")
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 0.0),
                camera_target=(0.0, 2.0, -1.0),
                time_elapsed=0.1,
            )
            assert pipeline.stats_shadow_commands > 0
            assert pipeline.stats_shadow_commands_unculled >= pipeline.stats_shadow_commands
        finally:
            pipeline.destroy()

    def test_caster_scratch_list_is_reused(self, render_ctx: RenderContext) -> None:
        """The shadow caster filter must not allocate a new list each frame."""
        pipeline = RenderPipeline(render_ctx, get_quality_preset(GraphicsQuality.HIGH))
        try:
            ecs = EntityManager(max_entities=16)
            ecs.create_entity(position=(0.0, 1.0, -5.0), shape="cube")
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 0.0),
                camera_target=(0.0, 2.0, -1.0),
                time_elapsed=0.1,
            )
            scratch_id = id(pipeline._caster_batches_scratch)
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 0.0),
                camera_target=(0.0, 2.0, -1.0),
                time_elapsed=0.2,
            )
            assert id(pipeline._caster_batches_scratch) == scratch_id
        finally:
            pipeline.destroy()

    def test_non_casters_are_excluded_from_shadow_commands(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx, get_quality_preset(GraphicsQuality.HIGH))
        try:
            ecs = EntityManager(max_entities=16)
            ecs.create_entity(position=(0.0, 1.0, -5.0), shape="cube")
            batches = [("cube", 1, 0, False, False)]  # cast_shadow=False
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 0.0),
                camera_target=(0.0, 2.0, -1.0),
                time_elapsed=0.1,
                draw_batches=batches,
            )
            assert pipeline.stats_shadow_commands == 0
        finally:
            pipeline.destroy()
