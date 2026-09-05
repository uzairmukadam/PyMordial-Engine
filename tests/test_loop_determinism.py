"""Unit tests for fixed-accumulator loop determinism and sub-frame interpolation."""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.core.loop import EngineLoop
from engine.core.math_utils import quat_from_euler, quat_identity


class TestLoopDeterminism:
    def test_fixed_tick_accumulation_exact(self):
        ecs = EntityManager(max_entities=100)
        loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)

        # Feed 60 frames of exactly 1/60s
        for _ in range(60):
            sub_ticks = loop.step_frame(1.0 / 60.0)
            assert sub_ticks == 1

        assert loop.fixed_tick_count == 60
        assert loop.total_frames == 60
        assert abs(loop.accumulator) < 1e-5

    def test_variable_framerates_yield_consistent_fixed_ticks(self):
        """Under jittery/fluctuating frametimes, 1 second must still yield 60 ticks."""
        ecs = EntityManager(max_entities=100)
        loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)

        # Alternate between fast frames (8ms) and slow frames (25ms)
        frame_pattern = [0.008, 0.025, 0.012, 0.030, 0.005, 0.020]
        total_time = 0.0

        i = 0
        while total_time < 1.0:
            dt = frame_pattern[i % len(frame_pattern)]
            loop.step_frame(dt)
            total_time += dt
            i += 1

        # Expected ticks is floor(total_time * 60)
        expected_ticks = int(total_time * 60.0)
        assert loop.fixed_tick_count == expected_ticks

    def test_accumulator_clamps_at_max(self):
        """A huge frame spike (e.g. 1.0s lag spike) should clamp to max_accumulator (0.20s)."""
        ecs = EntityManager(max_entities=100)
        loop = EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0, max_accumulator=0.20)

        sub_ticks = loop.step_frame(1.0)  # 1.0s lag spike
        # 0.20 / (1/60) = 12 ticks
        assert sub_ticks == 12
        assert loop.fixed_tick_count == 12

    def test_subframe_nlerp_interpolation(self):
        ecs = EntityManager(max_entities=100)
        ent = ecs.create_entity(position=(0.0, 0.0, 0.0))
        dense_idx = ecs.pool.get_dense_index(ent)

        # Simulate physics state: previous position (0, 0, 0), current position (10, 20, 30)
        ecs.rigid_body_state[0, dense_idx, 0:3] = [0.0, 0.0, 0.0]
        ecs.rigid_body_state[1, dense_idx, 0:3] = [10.0, 20.0, 30.0]

        # Interpolate at alpha = 0.5 (halfway between ticks)
        ecs.interpolate_render_transforms(alpha=0.5)

        # Check translation components in WorldTransforms (columns 12, 13, 14)
        mat = ecs.world_transforms[dense_idx]
        np.testing.assert_allclose(mat[12:15], [5.0, 10.0, 15.0])

        # Interpolate at alpha = 0.25
        ecs.interpolate_render_transforms(alpha=0.25)
        np.testing.assert_allclose(mat[12:15], [2.5, 5.0, 7.5])
