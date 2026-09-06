"""Category-Based Structured Logger for PyMordial Engine."""

from __future__ import annotations
from datetime import datetime
import time
from engine.logging.sinks import LogLevel, LogEntry, LogSink, ConsoleSink, FileSink, RingBufferSink


class LogChannel:
    """Pre-defined standard log channels."""
    CORE = "Core"
    WINDOW = "Window"
    GFX = "Gfx"
    PHYSICS = "Physics"
    AUDIO = "Audio"
    INPUT = "Input"
    ASSET = "Asset"
    TIME = "Time"
    CONFIG = "Config"
    EVENTS = "Events"
    CAMERA = "Camera"


class Logger:
    """Central structured logger coordinating sinks with zero-overhead release mode."""

    __slots__ = ("_sinks", "_ring_buffer", "_is_release_mode", "_min_level")

    def __init__(
        self,
        console: bool = True,
        file_path: str | None = None,
        ring_buffer: bool = True,
        min_level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        self._sinks: list[LogSink] = []
        self._is_release_mode: bool = False
        self._min_level: LogLevel = min_level
        self._ring_buffer: RingBufferSink | None = None

        if console:
            self._sinks.append(ConsoleSink(min_level=min_level))
        if file_path:
            self._sinks.append(FileSink(filepath=file_path, min_level=min_level))
        if ring_buffer:
            self._ring_buffer = RingBufferSink(capacity=1000, min_level=LogLevel.TRACE)
            self._sinks.append(self._ring_buffer)

    @property
    def is_release_mode(self) -> bool:
        """Returns True if the logger is running in stripped release mode."""
        return self._is_release_mode

    def set_release_mode(self, enabled: bool) -> None:
        """Enables stripped release mode.

        In release mode, TRACE, DEBUG, and INFO calls are early-out no-ops.
        Only WARN, ERROR, and FATAL are processed to capture crash logs.
        """
        self._is_release_mode = enabled
        if enabled:
            self._min_level = LogLevel.WARN
        else:
            self._min_level = LogLevel.DEBUG

    def add_sink(self, sink: LogSink) -> None:
        """Attaches a new sink to receive log entries."""
        if sink not in self._sinks:
            self._sinks.append(sink)

    def remove_sink(self, sink: LogSink) -> None:
        """Detaches a sink and closes it."""
        if sink in self._sinks:
            self._sinks.remove(sink)
            sink.close()

    def get_ring_buffer(self) -> RingBufferSink | None:
        """Returns the in-memory ring buffer sink, if registered."""
        return self._ring_buffer

    def log(self, level: LogLevel, channel: str, message: str) -> None:
        """Dispatches a log record to all attached sinks."""
        # Fast exit in release mode or if below minimum level
        if self._is_release_mode and level < LogLevel.WARN:
            return
        if level < self._min_level:
            return

        now_ts = time.time()
        now_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]
        entry = LogEntry(
            timestamp=now_ts,
            datetime_str=now_str,
            level=level,
            channel=channel,
            message=str(message),
        )

        for sink in self._sinks:
            sink.emit(entry)

    def trace(self, channel: str, message: str) -> None:
        if not self._is_release_mode:
            self.log(LogLevel.TRACE, channel, message)

    def debug(self, channel: str, message: str) -> None:
        if not self._is_release_mode:
            self.log(LogLevel.DEBUG, channel, message)

    def info(self, channel: str, message: str) -> None:
        if not self._is_release_mode:
            self.log(LogLevel.INFO, channel, message)

    def warn(self, channel: str, message: str) -> None:
        self.log(LogLevel.WARN, channel, message)

    def error(self, channel: str, message: str) -> None:
        self.log(LogLevel.ERROR, channel, message)

    def fatal(self, channel: str, message: str) -> None:
        self.log(LogLevel.FATAL, channel, message)

    def flush(self) -> None:
        """Flushes all underlying sinks."""
        for sink in self._sinks:
            sink.flush()

    def shutdown(self) -> None:
        """Flushes and closes all sinks."""
        for sink in self._sinks:
            sink.flush()
            sink.close()
        self._sinks.clear()


# Global default logger instance
_GLOBAL_LOGGER: Logger = Logger(console=True, file_path="logs/pymordial.log", ring_buffer=True)


def get_logger() -> Logger:
    """Retrieves the global default logger instance."""
    return _GLOBAL_LOGGER


def log_trace(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.trace(channel, message)


def log_debug(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.debug(channel, message)


def log_info(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.info(channel, message)


def log_warn(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.warn(channel, message)


def log_error(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.error(channel, message)


def log_fatal(channel: str, message: str) -> None:
    _GLOBAL_LOGGER.fatal(channel, message)
