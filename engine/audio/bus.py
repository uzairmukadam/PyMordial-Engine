"""Hierarchical Audio Bus Graph and Mixer for PyMordial Engine."""

from __future__ import annotations
from typing import Optional
from engine.logging import log_info, LogChannel


class AudioBus:
    """Represents a mix node in the hierarchical audio routing tree."""

    __slots__ = (
        "name",
        "volume",
        "muted",
        "ducking_factor",
        "parent",
        "children",
    )

    def __init__(self, name: str, parent: AudioBus | None = None, default_volume: float = 1.0) -> None:
        self.name = name
        self.parent = parent
        self.volume = max(0.0, min(1.0, default_volume))
        self.muted = False
        self.ducking_factor = 1.0
        self.children: list[AudioBus] = []

        if parent is not None:
            parent.children.append(self)

    def get_effective_volume(self) -> float:
        """Recursively computes final multiplier through parent buses."""
        if self.muted:
            return 0.0

        eff = self.volume * self.ducking_factor
        if self.parent is not None:
            eff *= self.parent.get_effective_volume()

        return max(0.0, min(1.0, eff))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def __repr__(self) -> str:
        return f"<AudioBus '{self.name}' vol={self.volume:.2f} eff={self.get_effective_volume():.2f}>"


class AudioMixer:
    """Manages the standard AAA audio bus hierarchy."""

    __slots__ = (
        "_buses",
        "master_bus",
        "music_bus",
        "sfx_bus",
        "voice_bus",
        "ui_bus",
        "ambient_bus",
    )

    def __init__(self) -> None:
        self._buses: dict[str, AudioBus] = {}

        # Construct standard tree: Master -> Music, SFX, Voice, UI, Ambient
        self.master_bus = AudioBus("Master", parent=None, default_volume=1.0)
        self._buses["master"] = self.master_bus

        self.music_bus = AudioBus("Music", parent=self.master_bus, default_volume=0.8)
        self._buses["music"] = self.music_bus

        self.sfx_bus = AudioBus("SFX", parent=self.master_bus, default_volume=1.0)
        self._buses["sfx"] = self.sfx_bus

        self.voice_bus = AudioBus("Voice", parent=self.master_bus, default_volume=1.0)
        self._buses["voice"] = self.voice_bus

        self.ui_bus = AudioBus("UI", parent=self.master_bus, default_volume=1.0)
        self._buses["ui"] = self.ui_bus

        self.ambient_bus = AudioBus("Ambient", parent=self.master_bus, default_volume=0.7)
        self._buses["ambient"] = self.ambient_bus

    def get_bus(self, name: str) -> AudioBus | None:
        """Finds an audio bus by name (case-insensitive)."""
        return self._buses.get(name.lower(), None)

    def set_volume(self, bus_name: str, volume: float) -> bool:
        """Sets the volume of a specific bus."""
        bus = self.get_bus(bus_name)
        if bus is not None:
            bus.set_volume(volume)
            return True
        return False

    def set_muted(self, bus_name: str, muted: bool) -> bool:
        """Mutes or unmutes a specific bus."""
        bus = self.get_bus(bus_name)
        if bus is not None:
            bus.set_muted(muted)
            return True
        return False
