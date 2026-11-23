"""Tests for the debug-menu plumbing of the new tracing backends.

Pins the properties that make the F2 panel usable:

* every backend toggle exists on EngineTweaks and defaults to LEGACY,
* apply_to_pipeline forwards them (and hiz_debug_mip) to RenderConfig,
* the values survive a to_dict/from_dict round trip,
* the debug menu's backend combo helper resolves names safely.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.debug.debug_menu import _AO_BACKENDS, _SSGI_BACKENDS, _SSR_BACKENDS, _backend_index
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline


class TestBackendToggles:
    def test_defaults_are_legacy(self):
        et = EngineTweaks()
        assert et.ssgi_backend == "LEGACY"
        assert et.ssr_backend == "LEGACY"
        assert et.ao_backend == "LEGACY"
        assert et.hiz_debug_mip == 0

    def test_apply_to_pipeline_forwards_backends(self):
        ctx = RenderContext.create_headless(width=160, height=120)
        pipeline = RenderPipeline(ctx)
        try:
            et = EngineTweaks()
            et.ssgi_backend = "HIZ_BOUNDS"
            et.ssr_backend = "HIZ_BOUNDS"
            et.ao_backend = "HIZ_BOUNDS"
            et.hiz_debug_mip = 2
            et.ssgi_debug_view = 6
            et.ao_debug_view = 3
            et.ssgi_surface_reject = True
            et.apply_to_pipeline(pipeline, force=True)

            cfg = pipeline.config
            assert cfg.ssgi_backend == "HIZ_BOUNDS"
            assert cfg.ssr_backend == "HIZ_BOUNDS"
            assert cfg.ao_backend == "HIZ_BOUNDS"
            # Regression: hiz_debug_mip was declared but never forwarded.
            assert cfg.hiz_debug_mip == 2
            assert cfg.ssgi_debug_view == 6
            assert cfg.ao_debug_view == 3
            assert cfg.ssgi_surface_reject is True
        finally:
            pipeline.destroy()
            ctx.destroy()

    def test_round_trip_persists_backends(self):
        et = EngineTweaks()
        et.ssgi_backend = "HIZ_BOUNDS"
        et.ssr_backend = "HIZ_BOUNDS"
        et.ao_backend = "HIZ_BOUNDS"
        et.hiz_debug_mip = 3
        et.ssgi_debug_view = 5
        et.ssgi_surface_reject = True

        restored = EngineTweaks()
        restored.from_dict(et.to_dict())
        assert restored.ssgi_backend == "HIZ_BOUNDS"
        assert restored.ssr_backend == "HIZ_BOUNDS"
        assert restored.ao_backend == "HIZ_BOUNDS"
        assert restored.hiz_debug_mip == 3
        assert restored.ssgi_debug_view == 5
        assert restored.ssgi_surface_reject is True

    def test_backend_index_helper(self):
        assert _backend_index("LEGACY", _SSGI_BACKENDS) == 0
        assert _backend_index("HIZ_BOUNDS", _SSGI_BACKENDS) == 1
        # Unknown or lowercase values fall back to LEGACY rather than raising.
        assert _backend_index("nonsense", _AO_BACKENDS) == 0
        assert _backend_index("hiz_bounds", _SSR_BACKENDS) == 1

    def test_bounds_debug_modes_are_known(self):
        # The mip slider is shown for these modes, so they must exist.
        for mode in (
            GBufferDebugMode.HIZ_BOUNDS_NEAREST,
            GBufferDebugMode.HIZ_BOUNDS_FARTHEST,
            GBufferDebugMode.HIZ_BOUNDS_SPAN,
            GBufferDebugMode.HIZ_BOUNDS_MASK,
            GBufferDebugMode.HIZ_BOUNDS_INVALID,
            GBufferDebugMode.HIZ_BOUNDS_RAW,
        ):
            assert int(mode) >= 17


class TestBackendRendering:
    @pytest.fixture
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=320, height=240)
        yield ctx
        ctx.destroy()

    def _scene(self, ecs: EntityManager) -> None:
        ecs.create_entity(position=(0.0, -1.0, 0.0), scale=(10.0, 0.2, 10.0), color=(0.7, 0.7, 0.7))
        ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.9, 0.2, 0.2))

    def test_all_backends_render_finite_output(self, render_ctx: RenderContext):
        pipeline = RenderPipeline(render_ctx)
        try:
            et = EngineTweaks()
            et.ssgi_backend = "HIZ_BOUNDS"
            et.ssr_backend = "HIZ_BOUNDS"
            et.ao_backend = "HIZ_BOUNDS"
            et.apply_to_pipeline(pipeline, force=True)

            ecs = EntityManager(max_entities=10)
            self._scene(ecs)

            for gi_mode in ("SSGI", "RMGI"):
                et.gi_mode = gi_mode
                et.apply_to_pipeline(pipeline, force=True)
                for i in range(3):
                    pipeline.render_frame(
                        ecs=ecs, camera_pos=(0.0, 2.0, 5.0), time_elapsed=0.016 * (i + 1)
                    )
                img = np.frombuffer(
                    pipeline.post_process.final_fbo.read(components=3, dtype="f1"), dtype=np.uint8
                )
                assert np.all(np.isfinite(img))
                assert img.mean() > 0.0
        finally:
            pipeline.destroy()
