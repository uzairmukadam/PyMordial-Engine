"""Master Audio Engine for PyMordial Engine."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pygame

from engine.audio.bus import AudioBus, AudioMixer
from engine.audio.sound import SoundCue, ProceduralSoundSynthesizer
from engine.audio.spatial import AudioListener
from engine.audio.voice_pool import VoicePool, ActiveVoice
from engine.events import subscribe_event, PlaySoundCueEvent
from engine.logging import log_info, log_warn, log_debug, LogChannel


class AudioEngine:
    """Universal 3D Audio Engine with hierarchical buses and procedural synthesis."""

    __slots__ = (
        "is_initialized",
        "is_headless",
        "mixer",
        "listener",
        "voice_pool",
        "_cues",
    )

    def __init__(self, headless: bool = False, max_channels: int = 32) -> None:
        self.is_initialized: bool = False
        self.is_headless = headless
        self.mixer = AudioMixer()
        self.listener = AudioListener()
        self._cues: dict[str, SoundCue] = {}
        self.voice_pool: VoicePool | None = None

        self._init_hardware(max_channels=max_channels)

        # Auto-subscribe to engine audio events
        subscribe_event(PlaySoundCueEvent, self._on_play_sound_event)

    def _init_hardware(self, max_channels: int = 32) -> None:
        """Initializes SDL2 Audio mixer hardware."""
        try:
            if not pygame.mixer.get_init():
                # 44.1 kHz, 16-bit signed, 2 channels (stereo), 512 buffer size
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.voice_pool = VoicePool(self.mixer, max_channels=max_channels)
            self.is_initialized = True
            log_info(LogChannel.AUDIO, f"Audio hardware initialized: 44.1kHz stereo, {max_channels} channels")
            self._register_default_cues()
        except Exception as e:
            self.is_initialized = False
            log_warn(LogChannel.AUDIO, f"Audio device unavailable (running silent): {e}")

    def _register_default_cues(self) -> None:
        """Synthesizes and registers standard built-in sound effects."""
        try:
            footstep_snd = ProceduralSoundSynthesizer.create_footstep()
            self.register_cue(SoundCue("footstep", footstep_snd, bus_name="SFX", is_3d=True, min_distance=1.0, max_distance=30.0))

            jump_snd = ProceduralSoundSynthesizer.create_jump()
            self.register_cue(SoundCue("jump", jump_snd, bus_name="SFX", is_3d=True, min_distance=2.0, max_distance=40.0))

            click_snd = ProceduralSoundSynthesizer.create_ui_click()
            self.register_cue(SoundCue("click", click_snd, bus_name="UI", is_3d=False))

            wind_snd = ProceduralSoundSynthesizer.create_ambient_wind()
            self.register_cue(SoundCue("ambient_wind", wind_snd, bus_name="Ambient", is_3d=False))
        except Exception as e:
            log_warn(LogChannel.AUDIO, f"Failed to register procedural cues: {e}")

    def register_cue(self, cue: SoundCue) -> None:
        """Registers a named sound cue."""
        self._cues[cue.name.lower()] = cue

    def get_cue(self, name: str) -> SoundCue | None:
        return self._cues.get(name.lower(), None)

    def load_sound(self, name: str, filepath: str | Path, bus_name: str = "SFX", is_3d: bool = True) -> SoundCue | None:
        """Loads a WAV/OGG audio file into a registered SoundCue."""
        if not self.is_initialized:
            return None
        path = Path(filepath)
        if not path.exists():
            log_warn(LogChannel.AUDIO, f"Sound file not found: {path}")
            return None

        try:
            snd = pygame.mixer.Sound(str(path))
            cue = SoundCue(name=name, sound=snd, bus_name=bus_name, is_3d=is_3d)
            self.register_cue(cue)
            log_info(LogChannel.AUDIO, f"Loaded sound '{name}' from {path}")
            return cue
        except Exception as e:
            log_warn(LogChannel.AUDIO, f"Failed to load sound {path}: {e}")
            return None

    def play_sound(
        self,
        cue_or_name: str | SoundCue,
        position: tuple[float, float, float] | np.ndarray | None = None,
        volume: float = 1.0,
        bus: str | None = None,
        loop: bool = False,
        priority: int = 0,
    ) -> ActiveVoice | None:
        """Plays a one-shot or looping sound with 3D spatialization."""
        if not self.is_initialized or self.voice_pool is None:
            return None

        cue = self.get_cue(cue_or_name) if isinstance(cue_or_name, str) else cue_or_name
        if cue is None or cue.sound is None:
            return None

        voice = self.voice_pool.allocate_voice(priority=max(priority, cue.priority))
        if voice is None:
            return None

        bus_target = self.mixer.get_bus(bus if bus is not None else cue.bus_name)
        voice.cue = cue
        voice.bus = bus_target
        voice.base_volume = max(0.0, min(1.0, volume * cue.base_volume))
        voice.priority = priority
        voice.is_looping = loop

        if position is not None and cue.is_3d:
            voice.position = np.array(position, dtype=np.float32)
            voice.is_3d = True
        else:
            voice.position = None
            voice.is_3d = False

        loops = -1 if loop else 0
        voice.channel.play(cue.sound, loops=loops)

        # Set initial volume
        self.voice_pool.update_spatial_voices(self.listener)
        return voice

    def _on_play_sound_event(self, event: PlaySoundCueEvent) -> None:
        """Dispatches sound cue requests arriving through the EventBus."""
        self.play_sound(
            cue_or_name=event.cue_name,
            position=event.position,
            volume=event.volume,
            bus=event.bus,
        )

    def update(
        self,
        dt: float,
        camera_position: tuple[float, float, float] | np.ndarray | None = None,
        camera_forward: tuple[float, float, float] | np.ndarray | None = None,
    ) -> None:
        """Per-frame update refreshing spatial listener and active voices."""
        if not self.is_initialized or self.voice_pool is None:
            return

        if camera_position is not None and camera_forward is not None:
            self.listener.sync_with_camera(camera_position, camera_forward)

        self.voice_pool.update_spatial_voices(self.listener)

    def stop_all(self) -> None:
        """Silences all active channels."""
        if self.voice_pool is not None:
            self.voice_pool.stop_all()


# Global default audio engine instance
_GLOBAL_AUDIO_ENGINE: AudioEngine = AudioEngine()


def get_audio_engine() -> AudioEngine:
    return _GLOBAL_AUDIO_ENGINE
