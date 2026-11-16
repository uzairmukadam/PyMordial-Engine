"""Tests for the Horizon 2 atmosphere module extraction (H2.4)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.app.config import ProjectConfig
from engine.gfx.context import RenderContext
from engine.gfx.environment import EnvironmentProvider, FlatEnvironment, create_default_environment
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset
from engine.modules.atmosphere import (
    ATMOSPHERE_PRESETS,
    AtmosphereConfig,
    AtmosphereEnvironment,
    AtmosphereModule,
    AtmosphereSystem,
)
from engine.modules.atmosphere.module import AtmosphereModule as ModuleDecl


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


class TestModuleDeclaration:
    def test_manifest_is_valid_and_declares_sky_capability(self) -> None:
        from engine.app.module_manifest import ModuleRole, validate_manifest

        manifest = AtmosphereModule.MODULE_MANIFEST
        assert validate_manifest(manifest).is_valid is True
        assert manifest.id == "pymordial.atmosphere"
        assert manifest.role is ModuleRole.RENDER_FEATURE
        assert "render.sky" in manifest.capabilities

    def test_module_class_exposes_manifest(self) -> None:
        assert ModuleDecl.get_module_manifest().id == "pymordial.atmosphere"


class TestEnvironmentProviderContract:
    def test_atmosphere_environment_satisfies_protocol(self, render_ctx: RenderContext) -> None:
        env = AtmosphereEnvironment(render_ctx.ctx)
        try:
            assert isinstance(env, EnvironmentProvider)
        finally:
            env.destroy()

    def test_flat_environment_satisfies_protocol(self, render_ctx: RenderContext) -> None:
        env = FlatEnvironment(render_ctx.ctx)
        try:
            assert isinstance(env, EnvironmentProvider)
            assert env.enabled is False
        finally:
            env.destroy()

    def test_flat_environment_provides_neutral_lighting(self, render_ctx: RenderContext) -> None:
        env = FlatEnvironment(render_ctx.ctx, sun_lux=2.5)
        try:
            light_dir, lux, color, ambient = env.compute_lighting()
            assert lux == 2.5
            assert len(light_dir) == 3
            assert len(color) == 3
            assert ambient >= 0.0
        finally:
            env.destroy()


class TestProviderSelection:
    def test_default_is_atmosphere(self, render_ctx: RenderContext) -> None:
        env = create_default_environment(render_ctx.ctx, ProjectConfig())
        try:
            assert isinstance(env, AtmosphereEnvironment)
            assert env.enabled is True
        finally:
            env.destroy()

    def test_disabled_config_yields_flat_environment(self, render_ctx: RenderContext) -> None:
        env = create_default_environment(render_ctx.ctx, ProjectConfig(atmosphere_enabled=False))
        try:
            assert isinstance(env, FlatEnvironment)
            assert env.enabled is False
        finally:
            env.destroy()

    def test_settings_apply_preset(self, render_ctx: RenderContext) -> None:
        env = create_default_environment(
            render_ctx.ctx,
            ProjectConfig(atmosphere_settings={"preset": "EARTH_SUNSET"}),
        )
        try:
            assert env.config.preset_name == "EARTH_SUNSET"
            assert env.config.time_of_day == ATMOSPHERE_PRESETS["EARTH_SUNSET"]["time_of_day"]
        finally:
            env.destroy()

    def test_settings_apply_direct_overrides(self, render_ctx: RenderContext) -> None:
        env = create_default_environment(
            render_ctx.ctx,
            ProjectConfig(atmosphere_settings={"time_of_day": 6.5, "cloud_enabled": False}),
        )
        try:
            assert env.config.time_of_day == 6.5
            assert env.config.cloud_enabled is False
        finally:
            env.destroy()

    def test_unknown_settings_keys_are_ignored(self, render_ctx: RenderContext) -> None:
        env = create_default_environment(
            render_ctx.ctx,
            ProjectConfig(atmosphere_settings={"not_a_real_setting": 1}),
        )
        try:
            assert not hasattr(env.config, "not_a_real_setting")
        finally:
            env.destroy()


class TestPipelineIntegration:
    def test_pipeline_uses_atmosphere_by_default(self, render_ctx: RenderContext) -> None:
        pipeline = RenderPipeline(render_ctx, get_quality_preset(GraphicsQuality.HIGH))
        try:
            assert isinstance(pipeline.environment, AtmosphereEnvironment)
            assert pipeline.atmosphere is pipeline.environment
        finally:
            pipeline.destroy()

    def test_pipeline_can_disable_atmosphere_without_editing_pipeline(self, render_ctx: RenderContext) -> None:
        cfg = get_quality_preset(GraphicsQuality.HIGH)
        cfg.atmosphere_enabled = False
        pipeline = RenderPipeline(render_ctx, cfg)
        try:
            assert isinstance(pipeline.environment, FlatEnvironment)
        finally:
            pipeline.destroy()

    def test_disabled_atmosphere_still_renders(self, render_ctx: RenderContext) -> None:
        from engine.core.ecs import EntityManager

        cfg = get_quality_preset(GraphicsQuality.HIGH)
        cfg.atmosphere_enabled = False
        pipeline = RenderPipeline(render_ctx, cfg)
        try:
            ecs = EntityManager(max_entities=8)
            ecs.create_entity(position=(0.0, 0.0, 0.0), shape="cube")
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 6.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.1,
            )
            raw = pipeline.post_process.final_fbo.read(components=3, dtype="f1")
            frame = np.frombuffer(raw, dtype=np.uint8)
            assert frame.size > 0
            assert frame.max() > 0
        finally:
            pipeline.destroy()

    def test_atmosphere_and_flat_frames_differ(self, render_ctx: RenderContext) -> None:
        """Disabling the module must visibly change the sky, proving it is not a no-op."""
        from engine.core.ecs import EntityManager

        def render(enabled: bool) -> np.ndarray:
            cfg = get_quality_preset(GraphicsQuality.HIGH)
            cfg.atmosphere_enabled = enabled
            pipeline = RenderPipeline(render_ctx, cfg)
            try:
                ecs = EntityManager(max_entities=8)
                ecs.create_entity(position=(0.0, 0.0, 0.0), shape="cube")
                pipeline.render_frame(
                    ecs=ecs,
                    camera_pos=(0.0, 2.0, 6.0),
                    camera_target=(0.0, 0.0, 0.0),
                    time_elapsed=0.1,
                )
                raw = pipeline.post_process.final_fbo.read(components=3, dtype="f1")
                return np.frombuffer(raw, dtype=np.uint8).copy()
            finally:
                pipeline.destroy()

        with_atmo = render(True)
        without_atmo = render(False)
        assert not np.array_equal(with_atmo, without_atmo)


class TestSystemMath:
    def test_sun_vector_is_normalized(self) -> None:
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=9.0))
        x, y, z = system.compute_sun_vector()
        assert abs((x * x + y * y + z * z) ** 0.5 - 1.0) < 1e-6

    def test_day_speed_advances_time(self) -> None:
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, day_speed=1.0))
        system.update(1.0)
        assert system.config.time_of_day != 12.0

    def test_paused_day_speed_does_not_advance(self) -> None:
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, day_speed=0.0))
        system.update(1.0)
        assert system.config.time_of_day == 12.0

    def test_preset_application(self) -> None:
        system = AtmosphereSystem()
        system.apply_preset("SPACE")
        assert system.config.sky_mode == "SPACE"
        assert system.config.preset_name == "SPACE"


class TestLegacyImportPath:
    def test_old_import_path_still_works(self) -> None:
        from engine.gfx.atmosphere import ATMOSPHERE_PRESETS as legacy_presets
        from engine.gfx.atmosphere import AtmosphereConfig as LegacyConfig
        from engine.gfx.atmosphere import AtmosphereSystem as LegacySystem

        assert LegacyConfig is AtmosphereConfig
        assert LegacySystem is AtmosphereSystem
        assert legacy_presets is ATMOSPHERE_PRESETS
