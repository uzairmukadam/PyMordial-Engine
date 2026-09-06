"""Unit tests for PyMordial Engine Multi-Domain Time & Clock System."""

from __future__ import annotations
import time
import pytest
from engine.time import TimeSystem, Stopwatch, Cooldown, FPSLimiter


def test_time_domains_and_dilation():
    ts = TimeSystem(fixed_dt=1.0 / 60.0, max_delta_time=0.20)

    # 1. Normal time scale (1.0)
    real_dt, game_dt = ts.tick(0.016)
    assert abs(real_dt - 0.016) < 1e-5
    assert abs(game_dt - 0.016) < 1e-5

    # 2. Slow motion / Time dilation (0.5)
    ts.set_time_scale(0.5)
    real_dt, game_dt = ts.tick(0.020)
    assert abs(real_dt - 0.020) < 1e-5
    assert abs(game_dt - 0.010) < 1e-5
    assert abs(ts.real_time - 0.036) < 1e-5
    assert abs(ts.game_time - 0.026) < 1e-5

    # 3. Paused state
    ts.pause()
    assert ts.is_paused
    real_dt, game_dt = ts.tick(0.030)
    assert abs(real_dt - 0.030) < 1e-5
    assert game_dt == 0.0  # Game time paused
    assert abs(ts.game_time - 0.026) < 1e-5

    # 4. Resume
    ts.resume()
    assert not ts.is_paused


def test_time_accumulator_clamping():
    ts = TimeSystem(max_delta_time=0.20)
    # Huge spike (e.g. dragging window or breakpoint)
    real_dt, game_dt = ts.tick(1.50)
    assert real_dt == 0.20
    assert game_dt == 0.20


def test_stopwatch_and_cooldown():
    sw = Stopwatch()
    sw.start()
    time.sleep(0.01)
    elapsed = sw.stop()
    assert elapsed >= 0.009

    cd = Cooldown(duration_seconds=0.10)
    curr = 1.0
    assert cd.is_ready(curr)
    assert cd.trigger(curr)
    assert not cd.is_ready(curr + 0.05)
    assert cd.is_ready(curr + 0.11)
