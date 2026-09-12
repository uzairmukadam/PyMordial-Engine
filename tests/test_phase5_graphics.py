"""Comprehensive Unit and Integration Tests for Phase 5 High-End Graphics.

Covers RenderGraph architecture, IBL split-sum BRDF LUT, GTAO, SSGI, LPV 3D volume,
Clustered local lights (SSBO 3), SSR, and TAA subpixel jittering.
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import (
    GraphicsQuality,
    GIMode,
    RenderConfig,
    get_quality_preset,
)
from engine.gfx.render_graph import RenderGraph, RenderPass, RenderGraphContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.passes.ibl import IBLPass
from engine.gfx.passes.clustered_lights import ClusteredLightingPass
from engine.gfx.passes.taa_pass import _halton
from engine.gfx.pipeline import RenderPipeline


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

        # Read back a portion of BRDF LUT to ensure non-zero values
        lut_data = ibl.lut_tex.read()
        lut_arr = np.frombuffer(lut_data, dtype=np.float16)
        assert np.any(lut_arr > 0.0)

        ibl.destroy()


class TestGTAOAndBilateralBlur:
    def test_gtao_and_blur_textures(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        assert pipeline.ao_pass.raw_ao_tex.size == (160, 120)
        assert pipeline.ao_pass.blur_ao_tex.size == (160, 120)

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


class TestSSGIAndLPVGlobalIllumination:
    def test_ssgi_pass_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)

        assert pipeline.ssgi_pass.raw_ssgi_tex.size == (160, 120)

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
        assert len(data) == 160 * 120 * 4 * 2  # RGBA16F

        pipeline.destroy()

    def test_lpv_3d_volume_compute_dispatch(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        assert pipeline.lpv_pass.grid_res == 32

        ecs = EntityManager(max_entities=10)
        ecs.create_entity(position=(0.0, 0.0, 0.0))

        pipeline.config.gi_mode = GIMode.LPV.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )

        assert pipeline.lpv_pass.volume_min is not None
        assert pipeline.lpv_pass.volume_size is not None

        pipeline.destroy()

    def test_gi_with_dynamic_lights(self, render_ctx: RenderContext):
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

        pipeline.config.gi_mode = GIMode.HYBRID.value
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 2.0, 5.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.1,
        )

        # Verify SSGI texture has data and LPV volume is active
        ssgi_data = pipeline.ssgi_pass.raw_ssgi_tex.read()
        assert len(ssgi_data) == 160 * 120 * 4 * 2
        lpv_data = pipeline.lpv_pass._current_src.read()
        assert len(lpv_data) == 32 * 32 * 32 * 4 * 2

        # Verify debug buffer modes 8, 9, 10 render without error
        for debug_mode in (8, 9, 10):
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

        ssr_data = pipeline.ssr_pass.ssr_tex.read()
        assert len(ssr_data) == (320 // 2) * (240 // 2) * 4 * 2

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
        assert low.ao_mode == "OFF"
        assert low.ssr_enabled is False
        assert low.taa_enabled is False
        assert low.aa_mode == "OFF"

        med = get_quality_preset(GraphicsQuality.MEDIUM)
        assert med.gi_mode == "SSGI"
        assert med.ao_mode == "SSAO"
        assert med.taa_enabled is False
        assert med.aa_mode == "OFF"

        high = get_quality_preset(GraphicsQuality.HIGH)
        assert high.gi_mode == "SSGI"
        assert high.ao_mode == "GTAO"
        assert high.ssr_enabled is True
        assert high.taa_enabled is False
        assert high.aa_mode == "OFF"

        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        assert ultra.gi_mode == "HYBRID"
        assert ultra.ao_mode == "GTAO"
        assert ultra.ssr_enabled is True
        assert ultra.taa_enabled is False
        assert ultra.aa_mode == "OFF"

        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert cinematic.gi_mode == "HYBRID"
        assert cinematic.ssgi_steps > ultra.ssgi_steps
        assert cinematic.ssr_steps > ultra.ssr_steps
        assert cinematic.taa_enabled is False
        assert cinematic.aa_mode == "OFF"


class TestShadowsAndSSCS:
    def test_shadow_modes_and_parameters_in_presets(self):
        low = get_quality_preset(GraphicsQuality.LOW)
        assert low.shadow_mode == "HARD"
        assert low.shadow_resolution == 1024
        assert low.shadow_distance == 120.0
        assert low.sscs_enabled is False

        med = get_quality_preset(GraphicsQuality.MEDIUM)
        assert med.shadow_mode == "PCF"
        assert med.shadow_resolution == 2048
        assert med.shadow_distance == 250.0
        assert med.sscs_enabled is False

        high = get_quality_preset(GraphicsQuality.HIGH)
        assert high.shadow_mode == "PCSS"
        assert high.shadow_softness == 1.2
        assert high.shadow_bias == 0.0015
        assert high.shadow_resolution == 4096
        assert high.shadow_distance == 500.0
        assert high.pcf_samples == 24
        assert high.sscs_enabled is False

        ultra = get_quality_preset(GraphicsQuality.ULTRA)
        assert ultra.shadow_mode == "PCSS"
        assert ultra.shadow_resolution == 4096
        assert ultra.shadow_distance == 800.0

        cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
        assert cinematic.shadow_distance == 1200.0

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
        assert low.shadow_distance < med.shadow_distance < high.shadow_distance < ultra.shadow_distance < cinematic.shadow_distance

        pipeline.destroy()

    def test_pipeline_shadow_modes_render_execution(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.8, 0.8, 0.8),
            roughness=0.3,
            metallic=0.1,
        )

        for mode in ("HARD", "PCF", "PCSS"):
            pipeline.config.shadow_mode = mode
            pipeline.config.sscs_enabled = True
            pipeline.config.shadow_softness = 1.5
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

        # Test shadow alignment offset and normal bias controls
        pipeline.config.shadow_normal_bias = 0.0025
        pipeline.config.shadow_offset_x = 0.015
        pipeline.config.shadow_offset_y = -0.010
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
        assert hasattr(pipeline.ao_pass, "hbao_prog")
        assert hasattr(pipeline.ao_pass, "gtao_prog")
        assert hasattr(pipeline.ao_pass, "blur_prog")

        for mode in ("OFF", "SSAO", "HBAO", "GTAO"):
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

