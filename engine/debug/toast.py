"""On-screen toast notification queue for PyMordial Engine debug messages."""

from __future__ import annotations
import time


class DebugToast:
    """Manages temporary on-screen notification messages that smoothly expire."""

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        # List of (message_text, expire_timestamp, rgb_color)
        self._queue: list[tuple[str, float, tuple[int, int, int]]] = []

    def show(
        self,
        message: str,
        duration: float = 3.0,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Pushes a new toast notification with a specified duration in seconds."""
        expire_at = time.perf_counter() + duration
        self._queue.append((message, expire_at, color))

    def get_active(self) -> list[tuple[str, tuple[int, int, int]]]:
        """Returns currently active messages and evicts expired ones."""
        now = time.perf_counter()
        self._queue = [item for item in self._queue if item[1] > now]
        return [(item[0], item[2]) for item in self._queue]

    def clear(self) -> None:
        """Clears all pending toasts."""
        self._queue.clear()
