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
