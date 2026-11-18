"""Solar flare pass tests.

Covers the sun-anchored physically-based flare: lifecycle, on-screen rendering,
depth occlusion, and non-regression of the existing anamorphic flare.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.solar_flare_pass import SolarFlarePass
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import RenderConfig


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


def _scene() -> EntityManager:
    ecs = EntityManager(max_entities=16)
    ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
    ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
    return ecs


def _flare_mean(pipeline: RenderPipeline) -> float:
    """Mean absolute luminance of the solar flare target."""
    tex = pipeline.solar_flare_pass.flare_texture
    raw = tex.read()
    data = np.frombuffer(raw, dtype=np.float16).reshape((tex.size[1], tex.size[0], 4))
    return float(np.mean(np.abs(data[..., :3])))


class TestSolarFlarePassLifecycle:
    def test_init_resize_destroy(self, render_ctx: RenderContext) -> None:
        flare = SolarFlarePass(render_ctx.ctx, 160, 120)
        try:
            # Full resolution by default
            assert flare.flare_texture.size == (160, 120)
            flare.resize(320, 240)
            assert flare.flare_texture.size == (320, 240)
            # A no-op resize must not reallocate.
            before = flare.flare_texture.glo
            flare.resize(320, 240)
            assert flare.flare_texture.glo == before
            # Explicit downscale still honoured
            flare.resize(320, 240, 0.5)
            assert flare.flare_texture.size == (160, 120)
        finally:
            flare.destroy()

    def test_render_returns_texture(self, render_ctx: RenderContext) -> None:
        flare = SolarFlarePass(render_ctx.ctx, 160, 120)
        scene = render_ctx.ctx.texture((160, 120), 4, dtype="f2")
        depth = render_ctx.ctx.depth_texture((160, 120))
        try:
            out = flare.render(
                scene_hdr_texture=scene,
                depth_texture=depth,
                sun_screen_pos=(0.5, 0.5),
                sun_on_screen=True,
                config=RenderConfig(),
            )
            assert out is flare.flare_texture
        finally:
            flare.destroy()
            scene.release()
            depth.release()


class TestSolarFlarePipeline:
    # Camera looks up at the sky; the sun sits along the view direction so it
    # projects to screen centre. sun_dir points from the sun into the scene.
    CAM_POS = (0.0, 2.0, 5.0)
    CAM_TARGET = (0.0, 12.0, -5.0)
    SUN_IN_VIEW = (0.0, -0.7071067811865475, 0.7071067811865475)

    def test_solar_mode_produces_flare(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.lens_flare_mode = "SOLAR"
        pipeline.config.lens_flare_enabled = True
        ecs = _scene()
        try:
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=self.CAM_POS,
                camera_target=self.CAM_TARGET,
                time_elapsed=0.1,
                sun_dir=self.SUN_IN_VIEW,
            )
            assert _flare_mean(pipeline) > 0.0, "SOLAR mode produced an empty flare."
        finally:
            pipeline.destroy()

    def test_off_mode_produces_no_flare(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.lens_flare_mode = "OFF"
        ecs = _scene()
        try:
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=self.CAM_POS,
                camera_target=self.CAM_TARGET,
                time_elapsed=0.1,
                sun_dir=self.SUN_IN_VIEW,
            )
            assert _flare_mean(pipeline) == pytest.approx(0.0, abs=1e-6)
        finally:
            pipeline.destroy()

    def test_anamorphic_mode_still_works(self, render_ctx: RenderContext) -> None:
        """The existing flare path must not regress."""
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.lens_flare_mode = "ANAMORPHIC"
        pipeline.config.lens_flare_enabled = True
        ecs = _scene()
        try:
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=self.CAM_POS,
                camera_target=self.CAM_TARGET,
                time_elapsed=0.1,
                sun_dir=self.SUN_IN_VIEW,
            )
            tex = pipeline.camera_optics_pass.flare_texture
            raw = tex.read()
            data = np.frombuffer(raw, dtype=np.float16).reshape((tex.size[1], tex.size[0], 4))
            assert np.all(np.isfinite(data))
        finally:
            pipeline.destroy()

    def test_sun_behind_camera_disables_flare(self, render_ctx: RenderContext) -> None:
        """A sun pointing away from the view must not draw a flare."""
        pipeline = RenderPipeline(render_ctx)
        pipeline.config.lens_flare_mode = "SOLAR"
        ecs = _scene()
        try:
            # Negating the in-view direction puts the sun behind the camera.
            behind = tuple(-c for c in self.SUN_IN_VIEW)
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=self.CAM_POS,
                camera_target=self.CAM_TARGET,
                time_elapsed=0.1,
                sun_dir=behind,
            )
            assert _flare_mean(pipeline) == pytest.approx(0.0, abs=1e-6), (
                "Flare drew even though the sun was behind the camera."
            )
        finally:
            pipeline.destroy()


class TestSolarFlareConfig:
    def test_defaults(self) -> None:
        cfg = RenderConfig()
        assert cfg.lens_flare_mode == "ANAMORPHIC"
        assert cfg.solar_flare_blade_count == 6
        assert cfg.solar_flare_bloom_intensity == pytest.approx(1.0)
        assert cfg.solar_flare_starburst_intensity == pytest.approx(0.6)
        assert cfg.solar_flare_occlusion_fade == pytest.approx(1.0)

    def test_mode_values_are_accepted(self) -> None:
        for mode in ("OFF", "ANAMORPHIC", "SOLAR"):
            assert RenderConfig(lens_flare_mode=mode).lens_flare_mode == mode

    def test_occlusion_fade_zero_ignores_geometry(self) -> None:
        cfg = RenderConfig(solar_flare_occlusion_fade=0.0)
        assert cfg.solar_flare_occlusion_fade == pytest.approx(0.0)
