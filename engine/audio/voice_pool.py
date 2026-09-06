"""Channel Allocation and Voice Pooling Manager for PyMordial Engine."""

from __future__ import annotations
import numpy as np
import pygame

from engine.audio.bus import AudioBus, AudioMixer
from engine.audio.sound import SoundCue
from engine.audio.spatial import (
    AudioListener,
    calculate_spatial_pan_and_attenuation,
)


class ActiveVoice:
    """Tracks an actively playing audio channel and its 3D spatial properties."""

    __slots__ = (
        "channel_id",
        "channel",
        "cue",
        "bus",
        "position",
        "is_3d",
        "priority",
        "base_volume",
        "is_looping",
    )

    def __init__(self, channel_id: int, channel: pygame.mixer.Channel) -> None:
        self.channel_id = channel_id
        self.channel = channel
        self.cue: SoundCue | None = None
        self.bus: AudioBus | None = None
        self.position: np.ndarray | None = None
        self.is_3d: bool = False
        self.priority: int = 0
        self.base_volume: float = 1.0
        self.is_looping: bool = False

    @property
    def is_busy(self) -> bool:
        return self.channel.get_busy()

    def stop(self) -> None:
        self.channel.stop()
        self.cue = None
        self.bus = None
        self.position = None
        self.is_3d = False


class VoicePool:
    """Allocates, steals, and dynamically updates 3D spatial channels."""

    __slots__ = ("_voices", "max_channels", "mixer")

    def __init__(self, mixer: AudioMixer, max_channels: int = 32) -> None:
        self.mixer = mixer
        self.max_channels = max_channels
        self._voices: list[ActiveVoice] = []

        # Ensure pygame.mixer has enough hardware channels
        try:
            pygame.mixer.set_num_channels(max_channels)
            for i in range(max_channels):
                ch = pygame.mixer.Channel(i)
                self._voices.append(ActiveVoice(i, ch))
        except Exception:
            pass

    def allocate_voice(self, priority: int = 0) -> ActiveVoice | None:
        """Finds an idle channel or steals the lowest-priority busy voice."""
        # 1. Look for free channel
        for v in self._voices:
            if not v.is_busy:
                return v

        # 2. Voice stealing: steal lowest priority voice if new priority is >=
        lowest_voice: ActiveVoice | None = None
        lowest_priority = priority

        for v in self._voices:
            if v.priority < lowest_priority:
                lowest_priority = v.priority
                lowest_voice = v

        if lowest_voice is not None:
            lowest_voice.stop()
            return lowest_voice

        return None

    def update_spatial_voices(self, listener: AudioListener) -> None:
        """Updates left/right volume balance on all active 3D voices."""
        for v in self._voices:
            if not v.is_busy:
                continue

            bus_vol = v.bus.get_effective_volume() if v.bus is not None else 1.0
            if bus_vol <= 0.0:
                v.channel.set_volume(0.0, 0.0)
                continue

            if v.is_3d and v.position is not None and v.cue is not None:
                l_vol, r_vol, _ = calculate_spatial_pan_and_attenuation(
                    listener=listener,
                    source_position=v.position,
                    min_distance=v.cue.min_distance,
                    max_distance=v.cue.max_distance,
                    attenuation_model=v.cue.attenuation_model,
                )
                final_l = l_vol * v.base_volume * bus_vol
                final_r = r_vol * v.base_volume * bus_vol
                v.channel.set_volume(final_l, final_r)
            else:
                eff = v.base_volume * bus_vol
                v.channel.set_volume(eff, eff)

    def stop_all(self) -> None:
        """Immediately halts playback on all channels."""
        for v in self._voices:
            v.stop()
