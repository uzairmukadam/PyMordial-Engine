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
