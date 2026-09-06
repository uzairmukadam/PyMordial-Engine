"""Unit tests for PyMordial Engine 3D Spatial Audio Subsystem."""

from __future__ import annotations
import numpy as np
from engine.audio import (
    AudioMixer,
    AudioListener,
    calculate_spatial_pan_and_attenuation,
    get_audio_engine,
)


def test_audio_bus_effective_volume():
    mixer = AudioMixer()
    # Master = 1.0, SFX = 1.0 -> effective SFX = 1.0
    sfx = mixer.get_bus("sfx")
    assert sfx is not None
    assert sfx.get_effective_volume() == 1.0

    # Reduce Master to 0.5 -> effective SFX = 0.5
    mixer.master_bus.set_volume(0.5)
    assert sfx.get_effective_volume() == 0.5

    # Reduce SFX to 0.5 -> effective SFX = 0.25
    sfx.set_volume(0.5)
    assert sfx.get_effective_volume() == 0.25

    # Mute master -> effective SFX = 0.0
    mixer.master_bus.set_muted(True)
    assert sfx.get_effective_volume() == 0.0


def test_3d_spatial_pan_and_attenuation():
    listener = AudioListener()
    listener.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    listener.forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    listener.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 1. Source directly ahead at 1.0m (within min distance)
    src_ahead = (0.0, 0.0, -1.0)
    l_vol, r_vol, atten = calculate_spatial_pan_and_attenuation(
        listener=listener,
        source_position=src_ahead,
        min_distance=1.0,
        max_distance=20.0,
    )
    assert atten == 1.0
    assert abs(l_vol - r_vol) < 0.05  # Centered stereo balance

    # 2. Source directly to the right
    src_right = (5.0, 0.0, 0.0)
    l_vol_r, r_vol_r, atten_r = calculate_spatial_pan_and_attenuation(
        listener=listener,
        source_position=src_right,
        min_distance=1.0,
        max_distance=20.0,
    )
    assert r_vol_r > l_vol_r  # Right channel significantly louder

    # 3. Source directly to the left
    src_left = (-5.0, 0.0, 0.0)
    l_vol_l, r_vol_l, atten_l = calculate_spatial_pan_and_attenuation(
        listener=listener,
        source_position=src_left,
        min_distance=1.0,
        max_distance=20.0,
    )
    assert l_vol_l > r_vol_l  # Left channel significantly louder


def test_procedural_synthesizer_and_audio_engine():
    audio = get_audio_engine()
    assert audio is not None

    if audio.is_initialized:
        voice = audio.play_sound("click", volume=0.5)
        assert voice is not None or audio.is_headless
