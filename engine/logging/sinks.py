"""Log Sinks for PyMordial Engine Structured Logging."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
import sys
import threading
from typing import Sequence


class LogLevel(IntEnum):
    """Severity levels for engine logging."""
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5


@dataclass(slots=True, frozen=True)
class LogEntry:
    """Represents a single immutable log event."""
    timestamp: float
    datetime_str: str
    level: LogLevel
    channel: str
    message: str


class LogSink(ABC):
    """Abstract base class for log output destinations."""

    @abstractmethod
    def emit(self, entry: LogEntry) -> None:
        """Emits a single log entry."""
        pass

    def flush(self) -> None:
        """Flushes buffered records, if applicable."""
        pass

    def close(self) -> None:
        """Closes the sink and releases resources."""
        pass


class ConsoleSink(LogSink):
    """Writes formatted colored logs directly to stdout/stderr."""

    # ANSI color codes
    RESET = "\033[0m"
    LEVEL_COLORS = {
        LogLevel.TRACE: "\033[90m",       # Gray
        LogLevel.DEBUG: "\033[36m",       # Cyan
        LogLevel.INFO: "\033[32m",        # Green
        LogLevel.WARN: "\033[33m",        # Yellow
        LogLevel.ERROR: "\033[31m",       # Red
        LogLevel.FATAL: "\033[41;97m",    # White on Red
    }
    CHANNEL_COLOR = "\033[35m"            # Magenta

    __slots__ = ("min_level", "use_colors")

    def __init__(self, min_level: LogLevel = LogLevel.DEBUG, use_colors: bool = True) -> None:
        self.min_level = min_level
        self.use_colors = use_colors and sys.stdout.isatty()

    def emit(self, entry: LogEntry) -> None:
        if entry.level < self.min_level:
            return

        lvl_name = entry.level.name.ljust(5)
        if self.use_colors:
            lvl_color = self.LEVEL_COLORS.get(entry.level, self.RESET)
            formatted = (
                f"{entry.datetime_str} "
                f"{lvl_color}[{lvl_name}]{self.RESET} "
                f"{self.CHANNEL_COLOR}[{entry.channel}]{self.RESET} "
                f"{entry.message}"
            )
        else:
            formatted = f"{entry.datetime_str} [{lvl_name}] [{entry.channel}] {entry.message}"

        stream = sys.stderr if entry.level >= LogLevel.ERROR else sys.stdout
        print(formatted, file=stream)

    def flush(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()


class FileSink(LogSink):
    """Writes log records to a persistent disk file with error resilience."""

    __slots__ = ("filepath", "min_level", "_file", "_lock")

    def __init__(
        self,
        filepath: str | Path = "logs/pymordial.log",
        min_level: LogLevel = LogLevel.DEBUG,
        mode: str = "a",
    ) -> None:
        self.filepath = Path(filepath)
        self.min_level = min_level
        self._lock = threading.Lock()

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, mode, encoding="utf-8", buffering=1)

    def emit(self, entry: LogEntry) -> None:
        if entry.level < self.min_level or self._file is None:
            return

        lvl_name = entry.level.name.ljust(5)
        line = f"{entry.datetime_str} [{lvl_name}] [{entry.channel}] {entry.message}\n"
        with self._lock:
            try:
                self._file.write(line)
            except Exception:
                pass

    def flush(self) -> None:
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None


class RingBufferSink(LogSink):
    """Fixed-capacity in-memory ring buffer for in-game debug consoles and UI."""

    __slots__ = ("_buffer", "_lock", "capacity", "min_level")

    def __init__(self, capacity: int = 1000, min_level: LogLevel = LogLevel.TRACE) -> None:
        self.capacity = capacity
        self.min_level = min_level
        self._buffer: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, entry: LogEntry) -> None:
        if entry.level < self.min_level:
            return
        with self._lock:
            self._buffer.append(entry)

    def get_entries(
        self,
        min_level: LogLevel | None = None,
        channel: str | None = None,
        count: int | None = None,
    ) -> list[LogEntry]:
        """Queries buffered logs with optional level and channel filtering."""
        with self._lock:
            entries = list(self._buffer)

        if min_level is not None:
            entries = [e for e in entries if e.level >= min_level]
        if channel is not None:
            entries = [e for e in entries if e.channel.lower() == channel.lower()]
        if count is not None and count > 0:
            entries = entries[-count:]

        return entries

    def clear(self) -> None:
        """Clears all records in the buffer."""
        with self._lock:
            self._buffer.clear()
