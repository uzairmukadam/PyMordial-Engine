"""Comprehensive Unit and Integration Tests for Phase 5 High-End Graphics.

Covers RenderGraph architecture, IBL split-sum BRDF LUT, GTAO, SSGI,
Clustered local lights (SSBO 3), SSR, and TAA subpixel jittering.
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.passes.clustered_lights import ClusteredLightingPass
from engine.gfx.passes.ibl import IBLPass
from engine.gfx.passes.taa_pass import _halton
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import (
    GIMode,
    GraphicsQuality,
    RenderConfig,
    get_quality_preset,
)
from engine.gfx.render_graph import RenderGraph, RenderGraphContext, RenderPass


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class DummyPass(RenderPass):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.executed = False

    def execute(self, context: RenderGraphContext) -> None:
        self.executed = True
        context.resources[f"{self.name}_done"] = True


class TestRenderGraphArchitecture:
    def test_render_graph_lifecycle_and_blackboard(self, render_ctx: RenderContext):
        ctx = render_ctx.ctx
        graph = RenderGraph()

        pass_a = DummyPass("PassA")
        pass_b = DummyPass("PassB")
        graph.add_pass(pass_a)
        graph.add_pass(pass_b)

        assert len(graph.passes) == 2
        assert graph.get_pass("PassA") is pass_a

        cfg = RenderConfig()
        fc = FrameContext(ctx)
        graph_ctx = RenderGraphContext(
            ctx=ctx,
            width=320,
            height=240,
            config=cfg,
            frame_context=fc,
        )

        graph.execute(graph_ctx)
        assert pass_a.executed is True
        assert pass_b.executed is True
        assert graph_ctx.resources.get("PassA_done") is True
        assert graph_ctx.resources.get("PassB_done") is True

        # Test disabling a pass
        pass_b.executed = False
        graph.set_pass_enabled("PassB", False)
        graph.execute(graph_ctx)
        assert pass_b.executed is False

        graph.destroy()
        fc.destroy()


class TestIBLAndSplitSumBRDF:
    def test_ibl_brdf_lut_and_env_map(self, render_ctx: RenderContext):
        ctx = render_ctx.ctx
        ibl = IBLPass(ctx)

        assert ibl.lut_tex.size == (256, 256)
        assert ibl.env_tex.size == (512, 256)
        assert ibl.irradiance_tex.size == (128, 64)

        # Read back a portion of BRDF LUT to ensure non-zero values
        lut_data = ibl.lut_tex.read()
        lut_arr = np.frombuffer(lut_data, dtype=np.float16)
        assert np.any(lut_arr > 0.0)

        env_arr = np.frombuffer(ibl.env_tex.read(), dtype=np.float16)
        assert np.max(env_arr) > 1.0  # Environment radiance remains linear HDR.
        rough_env_arr = np.frombuffer(ibl.env_tex.read(level=ibl.max_mip), dtype=np.float16)
        assert np.any(rough_env_arr > 0.0)  # GGX-prefiltered roughness mip.
        irradiance_arr = np.frombuffer(ibl.irradiance_tex.read(), dtype=np.float16)
        assert np.any(irradiance_arr > 0.0)

        ibl.destroy()


class TestGTAOAndBilateralBlur:
    def test_gtao_and_blur_textures(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        # AO defaults to full resolution so its screen-space radius matches the
        # actual viewport (a half-res target silently shrank the effect).
        assert pipeline.ao_pass.raw_ao_tex.size == (320, 240)
        assert pipeline.ao_pass.blur_ao_tex.size == (320, 240)

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
        )

        ao_data = pipeline.ao_pass.raw_ao_tex.read()
        ao_arr = np.frombuffer(ao_data, dtype=np.uint8)
        assert np.any(ao_arr > 0)

        pipeline.destroy()

    def test_ao_resolution_scale_is_honoured(self, render_ctx: RenderContext):
        """A downscaled AO target must still be told the full render size."""
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ao_resolution_scale = 0.5
            pipeline.apply_config(pipeline.config)
            assert pipeline.ao_pass.raw_ao_tex.size == (160, 120)

            pipeline.config.ao_resolution_scale = 1.0
            pipeline.apply_config(pipeline.config)
            assert pipeline.ao_pass.raw_ao_tex.size == (320, 240)
        finally:
            pipeline.destroy()

    def test_screen_space_effects_default_to_full_resolution(self, render_ctx: RenderContext):
        """Every screen-space effect renders at full resolution unless downscaled."""
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.ao_pass.raw_ao_tex.size == (320, 240)
            assert pipeline.ssgi_pass.raw_ssgi_tex.size == (320, 240)
            assert pipeline.rmgi_pass.raw_rmgi_tex.size == (320, 240)
            assert pipeline.ssr_pass.raw_ssr_tex.size == (320, 240)
            assert pipeline.camera_optics_pass.flare_texture.size == (320, 240)
            assert pipeline.solar_flare_pass.flare_texture.size == (320, 240)
            assert pipeline.bloom_pass.downsample_textures[0].size == (320, 240)
        finally:
            pipeline.destroy()

    def test_effect_resolution_scales_are_honoured(self, render_ctx: RenderContext):
        """Each effect honours its own resolution scale through apply_config."""
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssgi_resolution_scale = 0.5
            pipeline.config.rmgi_resolution_scale = 0.5
            pipeline.config.ssr_resolution_scale = 0.5
            pipeline.config.lens_flare_resolution_scale = 0.5
            pipeline.apply_config(pipeline.config)

            assert pipeline.ssgi_pass.raw_ssgi_tex.size == (160, 120)
            assert pipeline.rmgi_pass.raw_rmgi_tex.size == (160, 120)
            assert pipeline.ssr_pass.raw_ssr_tex.size == (160, 120)
            assert pipeline.camera_optics_pass.flare_texture.size == (160, 120)
            assert pipeline.solar_flare_pass.flare_texture.size == (160, 120)

            pipeline.config.ssgi_resolution_scale = 1.0
            pipeline.config.rmgi_resolution_scale = 1.0
            pipeline.config.ssr_resolution_scale = 1.0
            pipeline.config.lens_flare_resolution_scale = 1.0
            pipeline.apply_config(pipeline.config)

            assert pipeline.ssgi_pass.raw_ssgi_tex.size == (320, 240)
            assert pipeline.rmgi_pass.raw_rmgi_tex.size == (320, 240)
            assert pipeline.ssr_pass.raw_ssr_tex.size == (320, 240)
            assert pipeline.camera_optics_pass.flare_texture.size == (320, 240)
            assert pipeline.solar_flare_pass.flare_texture.size == (320, 240)
        finally:
            pipeline.destroy()

    def test_effect_resolution_scales_survive_window_resize(self, render_ctx: RenderContext):
        """A window resize must re-derive every effect target from its own scale."""
        pipeline = RenderPipeline(render_ctx)
        cfg = pipeline.config
        saved = (
            cfg.ssgi_resolution_scale,
            cfg.rmgi_resolution_scale,
            cfg.ssr_resolution_scale,
            cfg.lens_flare_resolution_scale,
        )
        try:
            cfg.ssgi_resolution_scale = 0.5
            cfg.rmgi_resolution_scale = 0.5
            cfg.ssr_resolution_scale = 0.5
            cfg.lens_flare_resolution_scale = 0.5
            pipeline.apply_config(cfg)

            pipeline.resize(640, 480)

            assert pipeline.ssgi_pass.raw_ssgi_tex.size == (320, 240)
            assert pipeline.rmgi_pass.raw_rmgi_tex.size == (320, 240)
            assert pipeline.ssr_pass.raw_ssr_tex.size == (320, 240)
            assert pipeline.camera_optics_pass.flare_texture.size == (320, 240)
            assert pipeline.solar_flare_pass.flare_texture.size == (320, 240)
            assert pipeline.bloom_pass.downsample_textures[0].size == (640, 480)
        finally:
            pipeline.destroy()
            # The context and its RenderConfig are module-scoped; restore both so
            # later tests see the defaults they assert on.
            (
                cfg.ssgi_resolution_scale,
                cfg.rmgi_resolution_scale,
                cfg.ssr_resolution_scale,
                cfg.lens_flare_resolution_scale,
            ) = saved
            render_ctx.resize(320, 240)


class TestSSGIGlobalIllumination:
    def test_ssgi_has_a_temporal_history_target(self, render_ctx: RenderContext):
        """A single-frame SSGI estimate is noisy; history is what converges it."""
        pipeline = RenderPipeline(render_ctx)
        try:
            assert pipeline.ssgi_pass.history_ssgi_tex.size == (320, 240)
            assert pipeline.ssgi_pass.history_ssgi_tex.dtype == "f2"
        finally:
            pipeline.destroy()

    def test_ssgi_temporal_config_defaults(self) -> None:
        cfg = RenderConfig()
        assert cfg.ssgi_temporal_enabled is True
        assert 0.0 < cfg.ssgi_temporal_blend <= 0.95

    def test_ssgi_history_invalidated_on_resize(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager(max_entities=10)
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            pipeline.config.gi_mode = GIMode.SSGI.value
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), camera_target=(0.0, 0.0, 0.0))
            assert pipeline.ssgi_pass._first_frame is False

            pipeline.resize(400, 300)
            assert pipeline.ssgi_pass._first_frame is True, (
                "A resize must invalidate the history: old samples no longer "
                "correspond to the same pixels."
            )
        finally:
            pipeline.destroy()
            # The context is module-scoped; restore the size other tests expect.
            render_ctx.resize(320, 240)

    def test_ssgi_temporal_reduces_flicker(self, render_ctx: RenderContext):
        """Temporal accumulation must make the SSGI signal more stable."""
        import numpy as np

        def sample(temporal: bool) -> float:
            pipeline = RenderPipeline(render_ctx)
            try:
                pipeline.config.gi_mode = GIMode.SSGI.value
                pipeline.config.ssgi_temporal_enabled = temporal
                ecs = EntityManager(max_entities=10)
                ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
                ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
                frames = []
                for i in range(10):
                    pipeline.render_frame(
                        ecs=ecs,
                        camera_pos=(0.0, 2.0, 5.0),
                        camera_target=(0.0, 0.0, 0.0),
                        time_elapsed=0.1 + i / 60.0,
                        sun_dir=(0.5, -0.7, 0.4),
                    )
                    raw = pipeline.ssgi_pass.resolved_ssgi_tex.read()
                    frames.append(
                        np.frombuffer(raw, dtype=np.float16).reshape(240, 320, 4)[..., :3].astype(np.float32)
                    )
                deltas = [float(np.abs(frames[i + 1] - frames[i]).mean()) for i in range(len(frames) - 1)]
                return float(np.mean(deltas[4:]))
            finally:
                pipeline.destroy()

        off = sample(False)
        on = sample(True)
        assert on < off, (
            f"Temporal accumulation did not reduce SSGI flicker (off={off:.5f}, on={on:.5f})."
        )

    def test_ssgi_pass_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        # SSGI defaults to full resolution so its screen-space ray math matches
        # the viewport (a half-res target silently shrank the effect).
        assert pipeline.ssgi_pass.raw_ssgi_tex.size == (320, 240)

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            scale=(1.0, 2.0, 1.0),
            color=(0.95, 0.15, 0.10),
            roughness=0.3,
            metallic=0.1,
        )

        pipeline.config.gi_mode = GIMode.SSGI.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )

        data = pipeline.ssgi_pass.raw_ssgi_tex.read()
        assert len(data) == 320 * 240 * 4 * 2  # RGBA16F

        pipeline.destroy()

    def test_ssgi_with_dynamic_lights(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            scale=(1.0, 2.0, 1.0),
            color=(0.95, 0.15, 0.10),
            roughness=0.3,
            metallic=0.1,
        )

        pipeline.clear_point_lights()
        pipeline.add_point_light((0.5, 1.0, 1.0), radius=5.0, color=(0.2, 0.6, 1.0), intensity=4.0)

        pipeline.config.gi_mode = GIMode.SSGI.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )

        # Verify SSGI texture has data
        ssgi_data = pipeline.ssgi_pass.raw_ssgi_tex.read()
        assert len(ssgi_data) == 320 * 240 * 4 * 2

        # Verify debug buffer modes 8, 10 render without error
        for debug_mode in (8, 10):
            pipeline.config.debug_gbuffer = debug_mode
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.1,
            )

        pipeline.destroy()


class TestClusteredLightingAndSSR:
    def test_clustered_local_lights_ssbo(self, render_ctx: RenderContext):
        ctx = render_ctx.ctx
        lights = ClusteredLightingPass(ctx, max_lights=64)

        l1 = lights.add_light((1.0, 2.0, 3.0), radius=8.0, color=(1.0, 0.5, 0.2), intensity=5.0)
        assert l1.x == 1.0
        assert len(lights.lights) == 1

        fc = FrameContext(ctx)
        cfg = RenderConfig()
        graph_ctx = RenderGraphContext(
            ctx=ctx,
            width=320,
            height=240,
            config=cfg,
            frame_context=fc,
        )

        lights.execute(graph_ctx)
        assert graph_ctx.resources["point_light_count"] == 1

        lights.clear()
        assert len(lights.lights) == 0

        lights.destroy()
        fc.destroy()

    def test_ssr_pass_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, -0.5, 0.0),
            scale=(10.0, 0.5, 10.0),
            color=(0.1, 0.1, 0.1),
            roughness=0.05,
            metallic=0.8,
        )

        pipeline.config.ssr_enabled = True
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 3.0, 6.0),
            camera_target=(0.0, 0.0, 0.0),
        )

        assert pipeline.ssr_pass.raw_ssr_tex.size == (320, 240)
        ssr_data = pipeline.ssr_pass.ssr_tex.read()
        assert len(ssr_data) == pipeline.ssr_pass.width * pipeline.ssr_pass.height * 4 * 2

        pipeline.ssr_pass.resize(640, 480, 1.0)
        assert pipeline.ssr_pass.raw_ssr_tex.size == (640, 480)
        pipeline.ssr_pass.resize(640, 480, 0.5)
        assert pipeline.ssr_pass.raw_ssr_tex.size == (320, 240)

        pipeline.destroy()


class TestTemporalAntiAliasing:
    def test_halton_subpixel_jitter(self):
        j_x = [_halton(i, 2) for i in range(1, 9)]
        j_y = [_halton(i, 3) for i in range(1, 9)]

        assert all(0.0 <= x <= 1.0 for x in j_x)
        assert all(0.0 <= y <= 1.0 for y in j_y)
        assert len(set(j_x)) == 8

    def test_fxaa_and_smaa_modes(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

        modes = ["OFF", "FXAA", "SMAA_1X", "SMAA_2X", "SMAA_4X"]
        for mode in modes:
            pipeline.config.aa_mode = mode
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.016,
            )
            data = pipeline.post_process.final_texture.read()
            assert len(data) == 320 * 240 * 4

        pipeline.destroy()

    def test_smaa_jitter_patterns(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        # SMAA 1x has 0 jitter
        j1 = [pipeline.smaa_pass.get_jitter(1920, 1080, "SMAA_1X", frame_idx=i) for i in range(4)]
        assert all(jx == 0.0 and jy == 0.0 for jx, jy in j1)

        # SMAA 2x has 2 distinct phases
        j2 = [pipeline.smaa_pass.get_jitter(1920, 1080, "SMAA_2X", frame_idx=i) for i in range(4)]
        assert j2[0] != j2[1]
        assert j2[0] == j2[2]

        # SMAA 4x has 4 distinct phases
        j4 = [pipeline.smaa_pass.get_jitter(1920, 1080, "SMAA_4X", frame_idx=i) for i in range(4)]
        assert len(set(j4)) == 4

        pipeline.destroy()


class TestQualityPresetsAndGIMode:
    def test_quality_presets_gi_and_ao_modes(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        assert low.gi_mode == "OFF"
        assert low.ao_mode == "SSAO"
        assert low.ssr_enabled is False
        assert low.aa_mode == "FXAA"

        med = get_quality_preset(GraphicsQuality.MEDIUM)
        assert med.gi_mode == "SSGI"
        assert med.ao_mode == "GTAO"
        assert med.taa_enabled is True
        assert med.aa_mode == "TAA"

        high = get_quality_preset(GraphicsQuality.HIGH)
        assert high.gi_mode == "RMGI"
        assert high.ao_mode == "GTAO"
        assert high.ssr_enabled is True
        assert high.ssr_resolution_scale == 1.0
        assert high.ssr_steps == 64
        assert high.taa_enabled is True
        assert high.aa_mode == "TAA"

        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        assert ultra.gi_mode == "RMGI"
        assert ultra.ao_mode == "GTAO"
        assert ultra.ssr_enabled is True
        assert ultra.taa_enabled is True
        assert ultra.aa_mode == "TAA"

        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert cinematic.gi_mode == "RMGI"
        assert cinematic.ssgi_steps > ultra.ssgi_steps
        assert cinematic.ssr_steps > ultra.ssr_steps
        assert cinematic.ssr_resolution_scale == 1.0
        assert cinematic.taa_enabled is True
        assert cinematic.aa_mode == "TAA"


class TestShadows:
    def test_shadow_parameters_in_presets(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        assert low.shadow_resolution == 1024
        assert low.shadow_distance == 120.0
        assert low.csm_cascades == 2

        med = get_quality_preset(GraphicsQuality.MEDIUM)
        assert med.shadow_resolution == 2048
        assert med.shadow_distance == 250.0
        assert med.csm_cascades == 3

        high = get_quality_preset(GraphicsQuality.HIGH)
        assert high.shadow_bias == 0.0015
        assert high.shadow_resolution == 8192
        assert high.shadow_distance == 500.0
        assert high.csm_cascades == 4

        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        assert ultra.shadow_resolution == 8192
        assert ultra.shadow_distance == 800.0

        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert cinematic.shadow_distance == 1200.0

    def test_presets_author_explicit_cascade_distances(self):
        """Each preset must give every cascade an increasing far distance."""
        for quality in (
            GraphicsQuality.LOW,
            GraphicsQuality.MEDIUM,
            GraphicsQuality.HIGH,
            GraphicsQuality.ULTRA,
            GraphicsQuality.CINEMATIC,
        ):
            preset = get_quality_preset(quality)
            distances = preset.csm_cascade_distances
            assert distances is not None, f"{quality.value} must author cascade distances."
            assert len(distances) == 4
            assert all(distances[i] < distances[i + 1] for i in range(3)), (
                f"{quality.value} cascade distances are not increasing: {distances}"
            )
            # The last cascade must reach the preset's shadow distance.
            assert distances[3] == preset.shadow_distance

    def test_shadow_distance_parameterization_and_dynamic_splits(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        csm = pipeline.csm

        # Default shadow distance
        assert csm.max_distance >= 500.0
        assert len(csm.split_distances) == 4
        assert csm.split_distances[-1] >= 500.0

        # Dynamically increase shadow draw distance
        pipeline.config.shadow_distance = 1200.0
        csm.update_splits(1200.0, cascade_count=4)
        assert csm.max_distance == 1200.0
        assert csm.split_distances[-1] == 1200.0
        assert csm.split_distances[0] < csm.split_distances[1] < csm.split_distances[2] < csm.split_distances[3]

        # Verify preset scaling across presets
        low = get_quality_preset(GraphicsQuality.LOW)
        med = get_quality_preset(GraphicsQuality.MEDIUM)
        high = get_quality_preset(GraphicsQuality.HIGH)
        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert (
            low.shadow_distance
            < med.shadow_distance
            < high.shadow_distance
            < ultra.shadow_distance
            < cinematic.shadow_distance
        )

        pipeline.destroy()

    def test_pipeline_hard_shadow_render_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.8, 0.8, 0.8),
            roughness=0.3,
            metallic=0.1,
        )

        for distance in (120.0, 500.0, 1200.0):
            pipeline.config.shadow_distance = distance
            pipeline.config.shadow_bias = 0.002
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.1,
                sun_dir=(0.5, -0.7, 0.4),
                sun_lux=4.0,
            )
            data = pipeline.post_process.final_texture.read()
            assert len(data) == 320 * 240 * 4

        # Normal bias remains tunable
        pipeline.config.shadow_normal_bias = 0.0025
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.2,
            sun_dir=(0.5, -0.7, 0.4),
            sun_lux=4.0,
        )
        data = pipeline.post_process.final_texture.read()
        assert len(data) == 320 * 240 * 4

        pipeline.destroy()


class TestAmbientOcclusionModes:
    def test_ambient_occlusion_modes_render_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.8, 0.8, 0.8),
            roughness=0.3,
            metallic=0.1,
        )

        assert hasattr(pipeline.ao_pass, "ssao_prog")
        assert hasattr(pipeline.ao_pass, "gtao_prog")
        assert hasattr(pipeline.ao_pass, "blur_prog")

        for mode in ("OFF", "SSAO", "GTAO"):
            pipeline.config.ao_mode = mode
            pipeline.config.ao_intensity = 1.2
            pipeline.config.ao_radius = 0.75
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.1,
                sun_dir=(0.5, -0.7, 0.4),
                sun_lux=4.0,
            )
            final_data = pipeline.post_process.final_texture.read()
            assert len(final_data) > 0
        pipeline.destroy()


class TestFroxelVolumetricFog:
    def test_volumetric_fog_pass_init_and_resize(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        pass_fog = pipeline.volumetric_fog_pass

        assert pass_fog.scatter_ext_vol.width == 160
        assert pass_fog.scatter_ext_vol.height == 90
        assert pass_fog.scatter_ext_vol.depth == 64
        assert pass_fog.integrated_vol.depth == 64

        assert hasattr(pass_fog, "inject_prog")
        assert hasattr(pass_fog, "integrate_prog")
        assert hasattr(pass_fog, "composite_prog")

        pass_fog.resize(640, 480)
        assert pass_fog.width == 640
        assert pass_fog.height == 480

        pipeline.destroy()

    def test_volumetric_fog_render_modes(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.8, 0.4, 0.2))

        # Test normal composite, in-scattering only, and transmittance only
        for debug_mode in (0, 1, 2):
            pipeline.config.volumetric_fog_enabled = True
            pipeline.config.fog_debug_mode = debug_mode
            pipeline.config.fog_density = 0.025
            pipeline.config.fog_height_falloff = 0.10
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 2.0, 5.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.016,
                sun_dir=(0.5, -0.7, 0.4),
                sun_lux=4.0,
            )
            data = pipeline.post_process.final_texture.read()
            assert len(data) == 320 * 240 * 4

        # Test fog disabled bypass
        pipeline.config.volumetric_fog_enabled = False
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.032,
            sun_dir=(0.5, -0.7, 0.4),
            sun_lux=4.0,
        )
        data_off = pipeline.post_process.final_texture.read()
        assert len(data_off) == 320 * 240 * 4

        pipeline.destroy()
