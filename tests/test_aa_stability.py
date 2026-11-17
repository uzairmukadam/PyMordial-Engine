"""Anti-aliasing stability and exclusivity tests.

Guards two properties that the "SMAA 2x/4x and TAA are wobbly" report was about:

1. A static camera must produce a stable image. Temporal passes that reproject
   history incorrectly (or present the wrong ping-pong buffer) drift frame to
   frame even when nothing moves.
2. Exactly one anti-aliasing path runs per frame. TAA and FXAA/SMAA are
   mutually exclusive by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.debug.engine_tweaks import EngineTweaks
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import RenderConfig

AA_MODES = ("OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X", "TAA")


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _build_scene() -> EntityManager:
    ecs = EntityManager(max_entities=16)
    ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
    ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
    ecs.create_entity(position=(2.0, 0.5, -1.0), scale=(0.6, 0.6, 0.6), color=(0.2, 0.6, 0.9))
    return ecs


def _render_frames(pipeline: RenderPipeline, ecs: EntityManager, count: int) -> list[np.ndarray]:
    """Renders `count` frames from a fixed camera and returns the LDR output."""
    frames: list[np.ndarray] = []
    for index in range(count):
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1 + index * (1.0 / 60.0),
            sun_dir=(0.3, -0.85, -0.4),
        )
        raw = pipeline.post_process.final_fbo.read(components=3, dtype="f1")
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape((120, 160, 3)).astype(np.int16))
    return frames


class TestStaticCameraStability:
    """A still camera must not shimmer."""

    @pytest.mark.parametrize("aa_mode", ["TAA", "SMAA_4X"])
    def test_static_camera_converges(self, render_ctx: RenderContext, aa_mode: str) -> None:
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.aa_mode = aa_mode
        pipeline.config.taa_enabled = aa_mode == "TAA"
        ecs = _build_scene()
        try:
            frames = _render_frames(pipeline, ecs, 8)
            # Compare the last two frames: a converged temporal filter is stable.
            delta = float(np.mean(np.abs(frames[-1] - frames[-2])))
            assert delta < 2.0, (
                f"{aa_mode} is unstable with a static camera: mean frame delta {delta:.3f}. "
                "The temporal history is not converging."
            )
        finally:
            pipeline.destroy()


class TestAAExclusivity:
    """Exactly one anti-aliasing path may run per frame."""

    @pytest.mark.parametrize("aa_mode", AA_MODES)
    def test_mode_renders_without_error(self, render_ctx: RenderContext, aa_mode: str) -> None:
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.aa_mode = aa_mode
        pipeline.config.taa_enabled = aa_mode == "TAA"
        ecs = _build_scene()
        try:
            frames = _render_frames(pipeline, ecs, 2)
            assert frames[-1].shape == (120, 160, 3)
            # A working AA path must not blank the frame.
            assert float(frames[-1].mean()) > 1.0, f"{aa_mode} produced a black frame."
        finally:
            pipeline.destroy()

    def test_taa_and_smaa_are_mutually_exclusive(self, render_ctx: RenderContext) -> None:
        """Enabling TAA must not also run the SMAA pass."""
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.aa_mode = "TAA"
        pipeline.config.taa_enabled = True
        ecs = _build_scene()
        try:
            _render_frames(pipeline, ecs, 2)
            # The pipeline branches on aa_mode, so SMAA must not have advanced.
            assert pipeline.smaa_pass.frame_idx == 0, (
                "SMAA executed while TAA was active; the AA paths are not exclusive."
            )
        finally:
            pipeline.destroy()


class TestTaaTunables:
    """The new TAA feedback knobs must round-trip through EngineTweaks."""

    def test_render_config_defaults(self) -> None:
        cfg = RenderConfig()
        assert cfg.taa_max_feedback == pytest.approx(0.95)
        assert cfg.taa_min_feedback == pytest.approx(0.78)
        assert cfg.taa_variance_gamma == pytest.approx(1.25)
        assert cfg.taa_min_feedback < cfg.taa_max_feedback

    def test_engine_tweaks_roundtrip(self) -> None:
        tweaks = EngineTweaks()
        tweaks.taa_max_feedback = 0.93
        tweaks.taa_min_feedback = 0.70
        tweaks.taa_variance_gamma = 1.6

        restored = EngineTweaks()
        restored.from_dict(tweaks.to_dict())
        assert restored.taa_max_feedback == pytest.approx(0.93)
        assert restored.taa_min_feedback == pytest.approx(0.70)
        assert restored.taa_variance_gamma == pytest.approx(1.6)
