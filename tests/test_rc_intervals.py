"""Synthetic validation of radiance-cascade interval composition.

Exercises ``shaders/common/rc_composite.glsl`` through the real merge shader with
known synthetic intervals, independent of any traced scene data.
"""

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.shader_utils import load_shader

NEAR_SIZE = (4, 4)
FAR_SIZE = (2, 2)
DIRECTIONS = 2


class TestIntervalComposition:
    @pytest.fixture(scope="class")
    def render_ctx(self):
        ctx = RenderContext.create_headless(width=64, height=64)
        yield ctx
        ctx.destroy()

    def _merge(self, ctx, near_value, far_value):
        """Runs the merge shader with uniform synthetic intervals."""
        near = ctx.ctx.texture_array((NEAR_SIZE[0], NEAR_SIZE[1], DIRECTIONS), 4, dtype="f2")
        far = ctx.ctx.texture_array((FAR_SIZE[0], FAR_SIZE[1], DIRECTIONS), 4, dtype="f2")
        out = ctx.ctx.texture_array((NEAR_SIZE[0], NEAR_SIZE[1], DIRECTIONS), 4, dtype="f2")

        near_data = np.tile(np.asarray(near_value, dtype=np.float16), (NEAR_SIZE[0] * NEAR_SIZE[1] * DIRECTIONS, 1))
        far_data = np.tile(np.asarray(far_value, dtype=np.float16), (FAR_SIZE[0] * FAR_SIZE[1] * DIRECTIONS, 1))
        near.write(near_data.tobytes())
        far.write(far_data.tobytes())

        out.bind_to_image(0, read=False, write=True, level=0)
        near.use(location=1)
        far.use(location=2)

        prog = ctx.ctx.compute_shader(load_shader("rc_merge_cascades.comp"))
        prog["u_NearProbeSize"].value = NEAR_SIZE
        prog["u_FarProbeSize"].value = FAR_SIZE
        prog["u_DirectionCount"].value = DIRECTIONS
        prog["u_FarDirectionCount"].value = DIRECTIONS
        prog["u_RCMergeMaxCascade"].value = 1
        prog.run(group_x=1, group_y=1, group_z=DIRECTIONS)

        result = np.frombuffer(out.read(), dtype=np.float16).reshape(
            (NEAR_SIZE[1], NEAR_SIZE[0], DIRECTIONS, 4)
        )
        prog.release()
        near.release()
        far.release()
        out.release()
        return result

    def test_opaque_near_blocks_far(self, render_ctx: RenderContext):
        # Near interval fully opaque (a = 0): far radiance must contribute nothing.
        result = self._merge(render_ctx, (0.5, 0.5, 0.5, 0.0), (9.0, 9.0, 9.0, 1.0))
        assert np.allclose(result[..., :3], 0.5, atol=1e-3)
        assert np.allclose(result[..., 3], 0.0, atol=1e-3)

    def test_transparent_near_admits_far(self, render_ctx: RenderContext):
        # Near interval fully transparent (a = 1): full far radiance contributes.
        result = self._merge(render_ctx, (0.5, 0.5, 0.5, 1.0), (9.0, 9.0, 9.0, 1.0))
        assert np.allclose(result[..., :3], 9.5, atol=1e-2)
        assert np.allclose(result[..., 3], 1.0, atol=1e-3)

    def test_partial_transmittance(self, render_ctx: RenderContext):
        # merged.rgb = 1.0 + 0.5 * 4.0 = 3.0 ; merged.a = 0.5 * 0.8 = 0.4
        result = self._merge(render_ctx, (1.0, 1.0, 1.0, 0.5), (4.0, 4.0, 4.0, 0.8))
        assert np.allclose(result[..., :3], 3.0, atol=1e-2)
        assert np.allclose(result[..., 3], 0.4, atol=1e-3)

    def test_zero_radiance(self, render_ctx: RenderContext):
        result = self._merge(render_ctx, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        assert np.allclose(result[..., :3], 0.0, atol=1e-6)
        # Both intervals are fully transparent, so transmittance stays 1.
        assert np.allclose(result[..., 3], 1.0, atol=1e-3)

    def test_maximum_valid_radiance_is_clamped(self, render_ctx: RenderContext):
        # 1e4 + 1e4 must clamp to the documented 1e4 ceiling.
        result = self._merge(render_ctx, (1e4, 1e4, 1e4, 1.0), (1e4, 1e4, 1e4, 1.0))
        assert np.all(result[..., :3] <= 1e4 + 1.0)

    def test_nan_input_is_sanitised(self, render_ctx: RenderContext):
        result = self._merge(render_ctx, (np.nan, np.nan, np.nan, np.nan), (1.0, 1.0, 1.0, 1.0))
        assert np.all(np.isfinite(result))
        # A NaN interval is replaced by a fully transparent, zero-radiance one.
        assert np.allclose(result[..., :3], 1.0, atol=1e-2)

    def test_direction_layers_are_independent(self, render_ctx: RenderContext):
        result = self._merge(render_ctx, (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
        # Every layer must carry the same composed result.
        assert np.allclose(result[..., 0, :3], result[..., 1, :3], atol=1e-3)
