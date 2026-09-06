"""Sound Cue Models and Procedural Waveform Synthesizer for PyMordial Engine."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pygame

from engine.audio.spatial import AttenuationModel
from engine.logging import log_info, log_warn, LogChannel


@dataclass(slots=True)
class SoundCue:
    """Configured audio asset with playback properties."""
    name: str
    sound: pygame.mixer.Sound | None = None
    bus_name: str = "SFX"
    base_volume: float = 1.0
    pitch_randomness: float = 0.0
    is_3d: bool = True
    min_distance: float = 1.0
    max_distance: float = 50.0
    attenuation_model: AttenuationModel = AttenuationModel.INVERSE_DISTANCE
    priority: int = 0


class ProceduralSoundSynthesizer:
    """Generates pure in-memory 16-bit PCM sound effects using vectorized NumPy arrays."""

    SAMPLE_RATE: int = 44100

    @classmethod
    def create_tone(
        cls,
        frequency: float = 440.0,
        duration: float = 0.2,
        volume: float = 0.5,
    ) -> pygame.mixer.Sound:
        """Generates a pure sine tone with fade-out envelope."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        envelope = np.linspace(1.0, 0.0, num_samples)
        wave = np.sin(2.0 * np.pi * frequency * t) * envelope * volume

        samples = (wave * 32767).astype(np.int16)
        # Convert to 2-channel stereo
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_footstep(cls, duration: float = 0.12, volume: float = 0.6) -> pygame.mixer.Sound:
        """Generates an organic thud impact footstep waveform."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Exponential pitch drop from 180Hz down to 40Hz
        freq = 180.0 * np.exp(-t * 25.0) + 40.0
        phase = 2.0 * np.pi * np.cumsum(freq) / cls.SAMPLE_RATE
        # Noise burst layered in
        noise = (np.random.rand(num_samples) * 2.0 - 1.0) * 0.25
        envelope = np.exp(-t * 30.0)

        wave = (np.sin(phase) * 0.75 + noise) * envelope * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_jump(cls, duration: float = 0.22, volume: float = 0.5) -> pygame.mixer.Sound:
        """Generates an energetic upward pitch sweep for character jump."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Upward frequency sweep 150Hz -> 500Hz
        freq = 150.0 + (500.0 - 150.0) * (t / duration) ** 1.5
        phase = 2.0 * np.pi * np.cumsum(freq) / cls.SAMPLE_RATE
        envelope = np.sin(np.pi * (t / duration))

        wave = np.sin(phase) * envelope * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_ui_click(cls, duration: float = 0.03, volume: float = 0.4) -> pygame.mixer.Sound:
        """Generates a crisp UI click / toggle sound."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        wave = np.sin(2.0 * np.pi * 1200.0 * t) * np.exp(-t * 150.0) * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_ambient_wind(cls, duration: float = 2.0, volume: float = 0.3) -> pygame.mixer.Sound:
        """Generates a looping low-pass filtered wind atmospheric rumble."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        # White noise
        raw = np.random.rand(num_samples) * 2.0 - 1.0
        # Simple moving average low-pass filter
        kernel_size = 40
        kernel = np.ones(kernel_size) / kernel_size
        filtered = np.convolve(raw, kernel, mode="same")
        # Seamless loop cross-fade at ends
        fade = min(int(cls.SAMPLE_RATE * 0.1), num_samples // 4)
        filtered[:fade] *= np.linspace(0.0, 1.0, fade)
        filtered[-fade:] *= np.linspace(1.0, 0.0, fade)

        wave = filtered * volume * 2.5
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())
