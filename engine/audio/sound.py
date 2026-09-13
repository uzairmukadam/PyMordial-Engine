"""Sound Cue Models and Procedural Waveform Synthesizer for PyMordial Engine."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pygame

from engine.audio.spatial import AttenuationModel


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

    @classmethod
    def create_wood_break(cls, duration: float = 0.28, volume: float = 0.7) -> pygame.mixer.Sound:
        """Generates a shattering wood fracture sound with sharp cracks and splinter rattle."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Initial sharp snap
        snap = np.sin(2.0 * np.pi * 320.0 * np.exp(-t * 40.0) * t) * np.exp(-t * 50.0)
        # Low woody resonance
        thud = np.sin(2.0 * np.pi * 95.0 * t) * np.exp(-t * 22.0)
        # Noise burst for splinter shatter
        noise = (np.random.rand(num_samples) * 2.0 - 1.0) * np.exp(-t * 18.0) * 0.5
        # Secondary crackle spikes
        crackle = np.zeros(num_samples)
        for offset in (0.04, 0.08, 0.13, 0.19):
            idx = int(offset * cls.SAMPLE_RATE)
            span = min(int(0.02 * cls.SAMPLE_RATE), num_samples - idx)
            if span > 0:
                crackle[idx : idx + span] += (np.random.rand(span) * 2.0 - 1.0) * 0.4

        wave = (snap * 0.4 + thud * 0.35 + noise * 0.25 + crackle * 0.3) * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_throw_whoosh(cls, duration: float = 0.18, volume: float = 0.5) -> pygame.mixer.Sound:
        """Generates an aerodynamic whoosh sound for launching thrown objects."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Bell-shaped envelope peaking in the middle
        envelope = np.sin(np.pi * (t / duration)) ** 2
        # Swept noise filter
        noise = np.random.rand(num_samples) * 2.0 - 1.0
        # Pitch sweep tone
        freq = 160.0 + 340.0 * np.sin(np.pi * (t / duration))
        phase = 2.0 * np.pi * np.cumsum(freq) / cls.SAMPLE_RATE
        wave = (np.sin(phase) * 0.35 + noise * 0.65) * envelope * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_impact(
        cls,
        duration: float = 0.15,
        volume: float = 0.6,
        frequency: float = 140.0,
    ) -> pygame.mixer.Sound:
        """Generates a solid physical impact thud with rapid pitch decay."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        freq = frequency * np.exp(-t * 30.0) + 40.0
        phase = 2.0 * np.pi * np.cumsum(freq) / cls.SAMPLE_RATE
        noise = (np.random.rand(num_samples) * 2.0 - 1.0) * 0.3
        envelope = np.exp(-t * 25.0)

        wave = (np.sin(phase) * 0.7 + noise) * envelope * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

    @classmethod
    def create_water_splash(cls, duration: float = 0.32, volume: float = 0.65) -> pygame.mixer.Sound:
        """Generates a fluid water splash and bubbling ripple sound."""
        num_samples = int(cls.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Noise splash burst
        noise = np.random.rand(num_samples) * 2.0 - 1.0
        splash_env = np.exp(-t * 14.0)
        # Bubble resonance frequencies
        b1 = np.sin(2.0 * np.pi * 380.0 * (1.0 + 0.4 * np.exp(-t * 10.0)) * t) * np.exp(-t * 18.0)
        b2 = np.sin(2.0 * np.pi * 540.0 * (1.0 + 0.5 * np.exp(-t * 12.0)) * t) * np.exp(-t * 22.0)

        wave = (noise * 0.5 * splash_env + (b1 + b2) * 0.3) * volume
        samples = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))
        return pygame.mixer.Sound(buffer=stereo.tobytes())

