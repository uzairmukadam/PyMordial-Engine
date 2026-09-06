"""Dual-Motor Gamepad Rumble and Haptics Controller for PyMordial Engine."""

from __future__ import annotations
import pygame
from engine.logging import log_warn, LogChannel


class HapticsManager:
    """Controls controller vibration motors with frequency and duration controls."""

    __slots__ = ("_enabled",)

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val
        if not val:
            self.stop_all()

    def set_rumble(
        self,
        low_frequency: float,
        high_frequency: float,
        duration_seconds: float,
        device_index: int = 0,
    ) -> bool:
        """Triggers dual-motor haptic rumble on a connected controller.

        Args:
            low_frequency: Low frequency motor intensity (0.0 to 1.0)
            high_frequency: High frequency motor intensity (0.0 to 1.0)
            duration_seconds: Playback duration in seconds
            device_index: Controller index (default 0)
        """
        if not self._enabled:
            return False

        duration_ms = int(max(0.0, duration_seconds) * 1000)
        low_f = max(0.0, min(1.0, low_frequency))
        high_f = max(0.0, min(1.0, high_frequency))

        # Check SDL2 Controller API first
        try:
            if pygame.joystick.get_count() > device_index:
                js = pygame.joystick.Joystick(device_index)
                if hasattr(js, "rumble"):
                    return js.rumble(low_f, high_f, duration_ms)
        except Exception as e:
            log_warn(LogChannel.INPUT, f"Haptics rumble error: {e}")

        return False

    def stop_all(self) -> None:
        """Silences all active vibration motors."""
        try:
            for i in range(pygame.joystick.get_count()):
                js = pygame.joystick.Joystick(i)
                if hasattr(js, "stop_rumble"):
                    js.stop_rumble()
        except Exception:
            pass
