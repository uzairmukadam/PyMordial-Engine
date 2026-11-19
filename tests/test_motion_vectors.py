"""Tests for the P0 per-object motion vectors (Pass 4).

The defect this pass fixes: the G-buffer reprojected with the *current* model
matrix, so a moving object reported zero object velocity and TAA smeared it.

These tests assert the observable contract:

* A static object under a static camera produces exactly zero velocity.
* A moving object produces velocity proportional to its displacement.
* Disabling ``object_motion_vectors_enabled`` restores the legacy (broken)
  camera-only behaviour.
* RT2 is RGBA16F and carries a reactive mask in blue.
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import RenderConfig


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _velocity(pipeline: RenderPipeline) -> np.ndarray:
    raw = pipeline.g_buffer.velocity_texture.read()
    return np.frombuffer(raw, dtype=np.float16).reshape(120, 160, 4).astype(np.float32)


def _render_pair(ctx, dx: float, enabled: bool) -> np.ndarray:
    """Renders a frame, moves the object by ``dx``, renders again, returns velocity."""
    cfg = RenderConfig(
        object_motion_vectors_enabled=enabled,
        aa_mode="OFF",
        taa_enabled=False,
    )
    pipeline = RenderPipeline(ctx, config=cfg)
    try:
        ecs = EntityManager(max_entities=10)
        entity = ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.9, 0.5, 0.2),
        )
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 0.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.0,
            dt=1.0 / 60.0,
        )
        ecs.set_position(entity, (dx, 0.0, 0.0))
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 0.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=1.0 / 60.0,
            dt=1.0 / 60.0,
        )
        return _velocity(pipeline)
    finally:
        pipeline.destroy()


class TestGBufferFormat:
    def test_rt2_is_rgba16f(self, render_ctx):
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.g_buffer.velocity_texture.components == 4
            assert pipeline.g_buffer.velocity_texture.dtype == "f2"
        finally:
            pipeline.destroy()

    def test_reactive_mask_is_zero_for_opaque_geometry(self, render_ctx):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 0.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                dt=1.0 / 60.0,
            )
            velocity = _velocity(pipeline)
            assert float(np.abs(velocity[..., 2]).max()) < 1e-3
        finally:
            pipeline.destroy()


class TestMotionVectors:
    def test_static_object_has_zero_velocity(self, render_ctx):
        # Jitter is disabled: subpixel projection jitter legitimately produces a
        # sub-pixel velocity residual (TAA compensates for it), which would mask
        # the property under test.
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(aa_mode="OFF", taa_enabled=False))
        try:
            ecs = EntityManager(max_entities=10)
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            for frame in range(3):
                pipeline.render_frame(
                    ecs=ecs,
                    camera_pos=(0.0, 0.0, 5.0),
                    camera_target=(0.0, 0.0, 0.0),
                    time_elapsed=frame / 60.0,
                    dt=1.0 / 60.0,
                )
            velocity = _velocity(pipeline)
            assert float(np.abs(velocity[..., :2]).max()) < 1e-4
        finally:
            pipeline.destroy()

    def test_jitter_residual_is_subpixel(self, render_ctx):
        """With TAA jitter active a static object shows only a sub-pixel residual."""
        pipeline = RenderPipeline(render_ctx, config=RenderConfig(aa_mode="TAA", taa_enabled=True))
        try:
            ecs = EntityManager(max_entities=10)
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            for frame in range(4):
                pipeline.render_frame(
                    ecs=ecs,
                    camera_pos=(0.0, 0.0, 5.0),
                    camera_target=(0.0, 0.0, 0.0),
                    time_elapsed=frame / 60.0,
                    dt=1.0 / 60.0,
                )
            velocity = _velocity(pipeline)
            # Halton(2,3) jitter is bounded by half a pixel on each axis.
            assert float(np.abs(velocity[..., :2]).max()) < 1.0 / 160.0
        finally:
            pipeline.destroy()

    def test_moving_object_has_nonzero_velocity(self, render_ctx):
        velocity = _render_pair(render_ctx, 0.5, enabled=True)
        assert float(np.abs(velocity[..., :2]).max()) > 1e-3

    def test_velocity_is_proportional_to_displacement(self, render_ctx):
        small = float(np.abs(_render_pair(render_ctx, 0.25, enabled=True)[..., :2]).max())
        large = float(np.abs(_render_pair(render_ctx, 1.0, enabled=True)[..., :2]).max())
        # 4x the displacement must produce ~4x the velocity.
        ratio = large / max(small, 1e-9)
        assert 3.0 < ratio < 5.0, f"velocity is not proportional to displacement (ratio {ratio:.2f})"

    def test_velocity_direction_matches_motion(self, render_ctx):
        """Moving +X must produce a velocity whose dominant component is X."""
        velocity = _render_pair(render_ctx, 0.5, enabled=True)[..., :2]
        # Look only at pixels that actually moved.
        magnitude = np.abs(velocity).max(axis=-1)
        moved = velocity[magnitude > 1e-3]
        assert moved.size > 0
        assert abs(float(moved[:, 0].mean())) > abs(float(moved[:, 1].mean()))

    def test_disabling_restores_legacy_camera_only_behaviour(self, render_ctx):
        """With the flag off, a moving object reports zero object velocity."""
        velocity = _render_pair(render_ctx, 0.5, enabled=False)
        assert float(np.abs(velocity[..., :2]).max()) < 1e-4

    def test_config_default_is_enabled(self):
        assert RenderConfig().object_motion_vectors_enabled is True


class TestPreviousTransformBuffer:
    def test_previous_transform_ssbo_exists(self, render_ctx):
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.ssbo_prev_transforms is not None
            assert pipeline.ssbo_prev_transforms.size == pipeline.ssbo_transforms.size
        finally:
            pipeline.destroy()

    def test_previous_transforms_are_seeded_on_first_frame(self, render_ctx):
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline._prev_transforms_seeded is False
            ecs = EntityManager(max_entities=10)
            ecs.create_entity(position=(0.0, 0.0, 0.0))
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 0.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                dt=1.0 / 60.0,
            )
            assert pipeline._prev_transforms_seeded is True
        finally:
            pipeline.destroy()
