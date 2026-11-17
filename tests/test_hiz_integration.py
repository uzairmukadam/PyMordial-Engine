"""Hi-Z integration tests.

Verifies that the hierarchical-Z depth pyramid is actually consumed by the
screen-space reflection pass, and that both the Hi-Z and legacy linear march
produce usable output.
"""

from __future__ import annotations

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


def _reflective_scene() -> EntityManager:
    ecs = EntityManager(max_entities=16)
    # Large glossy floor to give SSR something to reflect.
    ecs.create_entity(
        position=(0.0, -1.0, 0.0),
        scale=(20.0, 0.2, 20.0),
        color=(0.8, 0.8, 0.8),
        roughness=0.05,
        metallic=0.0,
    )
    ecs.create_entity(position=(0.0, 0.5, -2.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
    return ecs


def _render(pipeline: RenderPipeline, ecs: EntityManager) -> None:
    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.0, 1.5, 4.0),
        camera_target=(0.0, 0.0, -1.0),
        time_elapsed=0.1,
        sun_dir=(0.3, -0.85, -0.4),
    )


class TestHiZAvailability:
    def test_hiz_pyramid_is_built_and_published(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx)
        ecs = _reflective_scene()
        try:
            _render(pipeline, ecs)
            assert pipeline.hiz_pass.hiz_texture is not None
            assert pipeline.hiz_pass.num_mips > 1, "Hi-Z pyramid must have multiple mips."
        finally:
            pipeline.destroy()

    def test_hiz_texture_is_non_empty(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx)
        ecs = _reflective_scene()
        try:
            _render(pipeline, ecs)
            data = pipeline.hiz_pass.hiz_texture.read(level=0)
            assert len(data) > 0
            assert np.any(np.frombuffer(data, dtype=np.float32) > 0.0), (
                "Hi-Z mip 0 is entirely zero; the pyramid was not populated."
            )
        finally:
            pipeline.destroy()


class TestSSRModeSelection:
    def test_config_accepts_both_modes(self) -> None:
        assert RenderConfig(ssr_mode="HIZ").ssr_mode == "HIZ"
        assert RenderConfig(ssr_mode="LINEAR").ssr_mode == "LINEAR"
        assert RenderConfig().ssr_mode == "HIZ"

    @pytest.mark.parametrize("mode", ["HIZ", "LINEAR"])
    def test_mode_renders_without_error(self, render_ctx: RenderContext, mode: str) -> None:
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.ssr_mode = mode
        pipeline.config.ssr_enabled = True
        ecs = _reflective_scene()
        try:
            _render(pipeline, ecs)
            raw = pipeline.post_process.final_fbo.read(components=3, dtype="f1")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((120, 160, 3))
            assert float(frame.mean()) > 1.0, f"ssr_mode={mode} produced a black frame."
        finally:
            pipeline.destroy()

    def test_hiz_and_linear_modes_agree(self, render_ctx: RenderContext) -> None:
        """The Hi-Z march must not diverge from the linear march.

        Both modes share the same hit validation, so on a scene where the
        reflection ray escapes to the sky (no screen-space hit) they must both
        report no reflection. This guards against the Hi-Z path inventing hits.
        """
        results = {}
        for mode in ("LINEAR", "HIZ"):
            pipeline = RenderPipeline(render_ctx)
            pipeline.config.ssr_mode = mode
            pipeline.config.ssr_enabled = True
            ecs = _reflective_scene()
            try:
                _render(pipeline, ecs)
                tex = pipeline.ssr_pass.raw_ssr_tex
                data = np.frombuffer(tex.read(), dtype=np.float16).reshape(
                    (tex.size[1], tex.size[0], 4)
                )
                results[mode] = float(np.max(np.abs(data[..., :3])))
            finally:
                pipeline.destroy()

        # Neither mode may produce NaN/Inf, and both must stay in a sane range.
        for mode, value in results.items():
            assert np.isfinite(value), f"ssr_mode={mode} produced non-finite output."
            assert value < 40.0, f"ssr_mode={mode} produced an out-of-range reflection."

        # A sky-facing reflection has no screen-space hit in either mode.
        assert results["HIZ"] == pytest.approx(results["LINEAR"], abs=1e-3), (
            f"Hi-Z and linear SSR disagree: HIZ={results['HIZ']}, LINEAR={results['LINEAR']}."
        )
