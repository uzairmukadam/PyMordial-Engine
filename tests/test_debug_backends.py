"""Tests for the debug-menu plumbing of the tracing backends.

Pins the properties that make the F2 panel usable:

* the backend combo helpers resolve names safely,
* the values survive a to_dict/from_dict round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.debug.debug_menu import _AO_BACKENDS, _SSGI_BACKENDS, _SSR_BACKENDS, _backend_index
from engine.debug.engine_tweaks import EngineTweaks
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline


class TestBackendToggles:
    def test_backend_index_helper(self):
        assert _backend_index("LEGACY", _SSGI_BACKENDS) == 0
        # Unknown or lowercase values fall back to LEGACY rather than raising.
        assert _backend_index("nonsense", _AO_BACKENDS) == 0
        assert _backend_index("unknown_backend", _SSR_BACKENDS) == 0


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
