"""Stage 1 tests: shared SSRT shader foundation and AMRGI migration.

These verify the extraction contract without duplicating whole shader sources:

* the shared SSRT GLSL files exist and load through ``shader_utils``;
* AMRGI includes them and uses ``SSRT_TraceLinear`` instead of a local march;
* GI-specific logic (cosine hemisphere generation, direct-radiance sampling,
  range scheduling) remains in the AMRGI shader;
* the shared tracer compiles on a real GL context.
"""

from __future__ import annotations

import pytest

from engine.gfx.context import RenderContext
from engine.gfx.shader_utils import load_shader


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


class TestSSRTFoundationExists:
    def test_ssrt_common_loads(self):
        src = load_shader("shaders/common/ssrt/ssrt_common.glsl")
        assert "SSRT_HashUint" in src
        assert "SSRT_Hash01" in src
        assert "SSRT_BuildBasis" in src
        assert "SSRT_InterleavedGradientNoise" in src

    def test_ssrt_trace_loads(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        assert "struct SSRT_Hit" in src
        assert "SSRT_TraceLinear" in src

    def test_ssrt_common_has_include_guard(self):
        src = load_shader("shaders/common/ssrt/ssrt_common.glsl")
        assert "#ifndef SSRT_COMMON_GLSL" in src
        assert "#endif" in src

    def test_ssrt_trace_has_include_guard(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        assert "#ifndef SSRT_TRACE_GLSL" in src
        assert "#endif" in src


class TestAMRGIUsesSharedTracer:
    def test_amrgi_includes_ssrt_files(self):
        src = load_shader("amrgi_trace.comp")
        # The expanded source must contain the shared definitions.
        assert "SSRT_TraceLinear" in src
        assert "SSRT_Hash01" in src
        assert "SSRT_BuildBasis" in src

    def test_local_marchrange_removed(self):
        src = load_shader("amrgi_trace.comp")
        # The local traversal function definition must be gone (only the shared
        # SSRT_TraceLinear remains). A bare "MarchRange" identifier would mean a
        # leftover local definition or call.
        assert "Hit MarchRange(" not in src
        assert "MarchRange(origin" not in src

    def test_amrgi_calls_shared_tracer(self):
        src = load_shader("amrgi_trace.comp")
        assert "SSRT_TraceLinear(origin, rdir" in src


class TestGISpecificLogicRemainsInAMRGI:
    def test_cosine_hemisphere_stays_effect_specific(self):
        src = load_shader("amrgi_trace.comp")
        assert "vec3 CosineHemisphereRay(vec3 n, float u1, float u2)" in src
        # It must NOT be in the generic common layer.
        common = load_shader("shaders/common/ssrt/ssrt_common.glsl")
        assert "CosineHemisphereRay" not in common

    def test_direct_radiance_sampling_stays_in_amrgi(self):
        src = load_shader("amrgi_trace.comp")
        assert "texture(u_DirectRadiance, hit.uv)" in src

    def test_range_scheduling_stays_in_amrgi(self):
        src = load_shader("amrgi_trace.comp")
        assert "wantMid" in src and "wantFar" in src
        assert "u_MidRate" in src and "u_FarRate" in src


class TestSharedTracerCompiles:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_amrgi_trace_compiles(self, render_ctx: RenderContext):
        prog = render_ctx.ctx.compute_shader(load_shader("amrgi_trace.comp"))
        try:
            assert prog is not None
        finally:
            prog.release()


class TestGenericHitContract:
    """Stage 2: the shared hit result must be effect-agnostic."""

    def test_hit_has_no_range_class(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        assert "rangeClass" not in src

    def test_hit_generic_fields(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        for field in ("bool  hit;", "vec2  uv;", "float distance;",
                      "float confidence;", "int   stepsTaken;", "uint  flags;"):
            assert field in src

    def test_generic_flags_defined(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        for flag in ("SSRT_HIT_VALID", "SSRT_MISS_BACKGROUND", "SSRT_MISS_SCREEN_EDGE",
                     "SSRT_MISS_MAX_DISTANCE", "SSRT_REJECT_SELF", "SSRT_REJECT_BACKFACE",
                     "SSRT_REJECT_PLANE", "SSRT_REJECT_CONVEX", "SSRT_REJECT_ORIENTATION"):
            assert flag in src

    def test_trace_config_policies(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        assert "struct SSRT_TraceConfig" in src
        for flag in ("SSRT_TRACE_REJECT_SELF_ID", "SSRT_TRACE_REJECT_BACKFACE",
                     "SSRT_TRACE_REJECT_PLANE", "SSRT_TRACE_REJECT_CONVEX",
                     "SSRT_TRACE_REJECT_ORIENTATION", "SSRT_TRACE_REFINE_BOUNDED"):
            assert flag in src

    def test_steps_tracked(self):
        src = load_shader("shaders/common/ssrt/ssrt_trace.glsl")
        assert "h.stepsTaken = s + 1;" in src


class TestAMRGIRangeClassInCaller:
    def test_range_class_assigned_in_amrgi(self):
        src = load_shader("amrgi_trace.comp")
        # Range classification moved back to the AMRGI caller.
        assert "hitRange = 1" in src
        assert "hitRange = 2" in src
        assert "hitRange = 3" in src
        assert "rangeClass = hitRange;" in src

    def test_amrgi_config_reproduces_stage1_thickness(self):
        src = load_shader("amrgi_trace.comp")
        assert "traceCfg.thicknessStepScale  = 1.35;" in src
        assert "traceCfg.thicknessStepCap    = 1.5;" in src
        assert "traceCfg.thicknessDepthScale = 0.03;" in src
        assert "SSRT_TRACE_REJECT_SELF_ID | SSRT_TRACE_REJECT_BACKFACE" in src


class TestSSRUsesSharedTracer:
    def test_ssr_includes_shared_tracer(self):
        src = load_shader("ssr.frag")
        # load_shader expands includes, so the shared tracer body is inlined.
        assert "SSRT_TraceLinear(" in src
        assert "struct SSRT_TraceConfig" in src

    def test_ssr_local_march_removed(self):
        src = load_shader("ssr.frag")
        # The independent ray-march loop must be gone.
        assert "for (int s = 1; s <= steps; ++s)" not in src
        assert "adaptive_thickness" not in src
        assert "GetViewPos" not in src

    def test_ssr_retains_reflection_specific_behavior(self):
        src = load_shader("ssr.frag")
        assert "vec3 R = reflect(-view_dir, view_normal);" in src
        assert "texture(u_SceneColor, hit_uv).rgb" in src
        assert "u_SSR_MaxRoughness" in src
        assert "rough_fade" in src and "grazing_fade" in src

    def test_ssr_preserves_thickness_and_refinement(self):
        src = load_shader("ssr.frag")
        assert "traceCfg.thicknessStepScale   = 0.20;" in src
        assert "traceCfg.thicknessDepthScale  = 0.004;" in src
        assert "traceCfg.thicknessMin         = 0.03;" in src
        assert "traceCfg.thicknessMax         = 0.35;" in src
        assert "traceCfg.refineSteps          = 6;" in src
        assert "SSRT_TRACE_REFINE_BOUNDED" in src

    def test_ssr_orientation_policy_preserved(self):
        src = load_shader("ssr.frag")
        assert "traceCfg.planeThreshold       = 0.02;" in src
        assert "traceCfg.convexNormalDot      = 0.65;" in src
        assert "traceCfg.convexNearDist       = 2.5;" in src
        assert "traceCfg.convexPlaneDist      = 1.5;" in src
        assert "traceCfg.orientationThreshold = -0.05;" in src

    def test_ssr_binds_surface_id(self):
        src = load_shader("ssr.frag")
        assert "u_GBufferSurfaceID" in src


class TestSSRCompiles:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_ssr_frag_compiles(self, render_ctx: RenderContext):
        vert = load_shader("fullscreen_quad.vert")
        prog = render_ctx.ctx.program(vertex_shader=vert, fragment_shader=load_shader("ssr.frag"))
        try:
            assert prog is not None
        finally:
            prog.release()


class TestStage3SharedVisibility:
    """Stage 3: shared local-diffuse visibility feeds SSRT AO without a second trace."""

    def test_amrgi_trace_accumulates_visibility(self):
        src = load_shader("amrgi_trace.comp")
        # Visibility is accumulated inside the diffuse trace from the same hit.
        assert "u_OutputVisibility" in src
        assert "visSum" in src and "visCount" in src
        assert "u_AORadius" in src
        # Only near-range hits contribute to AO.
        assert "hitRange == 1" in src

    def test_ssrt_ao_has_no_traversal(self):
        src = load_shader("ssrt_visibility_reconstruct.frag")
        # The reconstruct shader must NOT trace rays.
        assert "SSRT_TraceLinear" not in src
        assert "u_Visibility" in src

    def test_ao_pass_has_no_second_trace(self):
        import inspect
        from engine.gfx.passes import ao_pass as ao_mod
        src = inspect.getsource(ao_mod)
        assert "SSRT_TraceLinear" not in src
        assert "ssrt_diffuse_visibility_raw" in src
        assert "SSRT" in src

    def test_ssrt_ao_mode_accepted(self):
        from engine.gfx.quality_presets import RenderConfig
        cfg = RenderConfig()
        cfg.ao_mode = "SSRT"
        assert cfg.ao_mode == "SSRT"
        assert hasattr(cfg, "ssrt_ao_radius")

    def test_visibility_reconstruct_compiles(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        try:
            vert = load_shader("fullscreen_quad.vert")
            prog = ctx.ctx.program(
                vertex_shader=vert, fragment_shader=load_shader("ssrt_visibility_reconstruct.frag")
            )
            assert prog is not None
            prog.release()
        finally:
            ctx.destroy()


class TestStage31IndependentSSRTAO:
    """Stage 3.1: SSRT AO runs independently of AMRGI via a shared diffuse trace.

    The diffuse SSRT workload is a single trace with two independent output
    requests (radiance for AMRGI, visibility for SSRT AO). These tests verify the
    request plumbing, the removal of the AMRGI-only GTAO fallback, and that AO
    intensity reaches the SSRT path exactly once.
    """

    def test_trace_has_output_request_flags(self):
        src = load_shader("amrgi_trace.comp")
        assert "u_RadianceRequested" in src
        assert "u_VisibilityRequested" in src
        # Radiance accumulation is gated by the request flag.
        assert "wantRadiance" in src
        assert "wantVisibility" in src

    def test_pass_sets_request_flags(self):
        import inspect
        from engine.gfx.passes import amrgi_pass as amrgi_mod
        src = inspect.getsource(amrgi_mod)
        assert "u_RadianceRequested" in src
        assert "u_VisibilityRequested" in src
        assert 'radiance_requested = gi_mode == "AMRGI"' in src
        assert 'visibility_requested = ao_mode == "SSRT"' in src

    def test_visibility_only_does_not_advance_history(self):
        import inspect
        from engine.gfx.passes import amrgi_pass as amrgi_mod
        src = inspect.getsource(amrgi_mod)
        # Reconstruction (and thus history ping-pong) is gated on radiance.
        assert "if radiance_requested:" in src
        # Visibility-only path keeps history invalid.
        assert "Visibility-only" in src

    def test_pipeline_runs_trace_for_ssrt_ao_without_amrgi(self):
        import inspect
        from engine.gfx import pipeline as pipeline_mod
        src = inspect.getsource(pipeline_mod)
        assert 'self.config.gi_mode != "AMRGI"' in src
        assert '"SSRT"' in src

    def test_ao_pass_uses_ssrt_intensity(self):
        import inspect
        from engine.gfx.passes import ao_pass as ao_mod
        src = inspect.getsource(ao_mod)
        assert "_u_ssrt_intensity" in src
        # The AMRGI-disabled fallback must be gone; only a genuine
        # missing-resource safety fallback may remain.
        assert "AMRGI is not enabled" not in src
        assert "SSRT visibility unavailable; using GTAO" in src

    def test_reconstruct_applies_intensity_once(self):
        src = load_shader("ssrt_visibility_reconstruct.frag")
        assert "u_Intensity" in src
        assert "1.0 - u_Intensity * (1.0 - visibility)" in src

    def test_ssrt_ao_uses_visibility_not_gtao(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline

        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager()
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
            pipeline.config.gi_mode = "OFF"
            pipeline.config.ao_mode = "SSRT"
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.ao_pass.active_backend == "SSRT"
            assert pipeline.ao_pass.fallback_reason is None
            # Visibility-only trace must not advance AMRGI temporal history.
            assert pipeline.amrgi_pass._first_frame is True
        finally:
            pipeline.destroy()

    def test_rmgi_and_ssgi_with_ssrt_ao(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline

        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager()
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
            for gi in ("RMGI", "SSGI"):
                pipeline.config.gi_mode = gi
                pipeline.config.ao_mode = "SSRT"
                pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
                assert pipeline.ao_pass.active_backend == "SSRT", gi
                assert pipeline.ao_pass.fallback_reason is None, gi
        finally:
            pipeline.destroy()

    def test_amrgi_plus_ssrt_shares_one_trace(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline

        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager()
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))
            pipeline.config.gi_mode = "AMRGI"
            pipeline.config.ao_mode = "SSRT"
            pipeline.config.amrgi_ray_count = 4
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            # One shared trace: the configured ray count is unchanged by AO.
            assert pipeline.amrgi_pass._u_trace["u_RayCount"].value == 4
            assert pipeline.amrgi_pass._u_trace["u_RadianceRequested"].value == 1
            assert pipeline.amrgi_pass._u_trace["u_VisibilityRequested"].value == 1
            assert pipeline.ao_pass.active_backend == "SSRT"
            # AMRGI history advances normally in the fused mode.
            assert pipeline.amrgi_pass._first_frame is False
        finally:
            pipeline.destroy()

    def test_ssrt_ao_does_not_increase_ray_count(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline

        pipeline = RenderPipeline(render_ctx)
        try:
            ecs = EntityManager()
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
            pipeline.config.gi_mode = "OFF"
            pipeline.config.ao_mode = "SSRT"
            pipeline.config.amrgi_ray_count = 2
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016)
            assert pipeline.amrgi_pass._u_trace["u_RayCount"].value == 2
            assert pipeline.amrgi_pass._u_trace["u_RadianceRequested"].value == 0
            assert pipeline.amrgi_pass._u_trace["u_VisibilityRequested"].value == 1
        finally:
            pipeline.destroy()

    def test_resolution_scale_changes_raw_target_dims(self, render_ctx: RenderContext):
        from engine.gfx.passes.amrgi_pass import AMRGIPass

        p = AMRGIPass(render_ctx.ctx, width=320, height=240, resolution_scale=0.5)
        try:
            assert p.raw_radiance_tex.size == (160, 120)
            assert p.visibility_tex.size == (160, 120)
            p.resize(320, 240, resolution_scale=1.0)
            assert p.raw_radiance_tex.size == (320, 240)
            assert p.visibility_tex.size == (320, 240)
        finally:
            p.destroy()


class TestSSRTemporalStabilization:
    """Stage 6: SSR temporal reprojection, history validation, and mode selection."""

    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def test_temporal_shader_compiles(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        try:
            vert = load_shader("fullscreen_quad.vert")
            prog = ctx.ctx.program(vertex_shader=vert, fragment_shader=load_shader("ssr_temporal.frag"))
            assert prog is not None
            prog.release()
        finally:
            ctx.destroy()

    def test_temporal_shader_uses_velocity_and_validation(self):
        src = load_shader("ssr_temporal.frag")
        assert "u_VelocityTexture" in src
        assert "hist_uv = v_UV - velocity" in src
        assert "u_HistoryValid" in src
        assert "u_GBufferSurfaceID" in src
        # Output contract preserved: RGB radiance + A confidence.
        assert "out_SSR = vec4" in src

    def test_pass_has_history_pingpong(self, render_ctx: RenderContext):
        from engine.gfx.passes.ssr_pass import SSRPass
        p = SSRPass(render_ctx.ctx, width=320, height=240)
        try:
            assert p.hist_a is not p.hist_b
            assert p._read_hist is not p._write_hist
            assert p._history_valid is False
        finally:
            p.destroy()

    def test_first_frame_does_not_use_history(self, render_ctx: RenderContext):
        from engine.gfx.passes.ssr_pass import SSRPass
        p = SSRPass(render_ctx.ctx, width=320, height=240)
        try:
            assert p._history_valid is False
        finally:
            p.destroy()

    def test_invalidate_history(self, render_ctx: RenderContext):
        from engine.gfx.passes.ssr_pass import SSRPass
        p = SSRPass(render_ctx.ctx, width=320, height=240)
        try:
            p._history_valid = True
            p.invalidate_history()
            assert p._history_valid is False
        finally:
            p.destroy()

    def test_resize_recreates_and_invalidates_history(self, render_ctx: RenderContext):
        from engine.gfx.passes.ssr_pass import SSRPass
        p = SSRPass(render_ctx.ctx, width=320, height=240)
        try:
            p._history_valid = True
            old_a = p.hist_a
            p.resize(400, 300)
            assert p._history_valid is False
            assert p.hist_a is not old_a
            # History matches the SSR (scaled) resolution, not the window size.
            assert p.hist_a.size == (p.width, p.height)
        finally:
            p.destroy()

    def test_config_fields_exist(self):
        from engine.gfx.quality_presets import RenderConfig
        cfg = RenderConfig()
        assert hasattr(cfg, "ssr_temporal_enabled")
        assert hasattr(cfg, "ssr_filter_mode")
        assert hasattr(cfg, "ssr_temporal_history_weight")

    def test_ssr_disabled_skips_temporal(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = False
            pipeline.apply_config(pipeline.config)
            ecs = EntityManager(max_entities=5)
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.5, 0.5, 0.5))
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            # SSR disabled: history must never become valid.
            assert pipeline.ssr_pass._history_valid is False
        finally:
            pipeline.destroy()

    def test_temporal_runs_when_enabled(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = True
            pipeline.config.ssr_temporal_enabled = True
            pipeline.config.ssr_filter_mode = 1
            pipeline.apply_config(pipeline.config)
            ecs = EntityManager(max_entities=5)
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.5, 0.5, 0.5))
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            assert pipeline.ssr_pass._history_valid is True
        finally:
            pipeline.destroy()

    def test_legacy_spatial_only_mode(self, render_ctx: RenderContext):
        from engine.core.ecs import EntityManager
        from engine.gfx.pipeline import RenderPipeline
        pipeline = RenderPipeline(render_ctx)
        try:
            pipeline.config.ssr_enabled = True
            pipeline.config.ssr_temporal_enabled = False
            pipeline.apply_config(pipeline.config)
            ecs = EntityManager(max_entities=5)
            ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.5, 0.5, 0.5))
            pipeline.render_frame(ecs=ecs, camera_pos=(0.0, 3.0, 6.0), time_elapsed=0.1)
            # Legacy path: no temporal accumulation.
            assert pipeline.ssr_pass._history_valid is False
        finally:
            pipeline.destroy()
