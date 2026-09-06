"""Unit tests for PyMordial Engine Structured Logging System."""

from __future__ import annotations
from pathlib import Path
from engine.logging import (
    Logger,
    LogLevel,
    LogChannel,
)


def test_logging_levels_and_formatting(tmp_path: Path):
    log_file = tmp_path / "test.log"
    logger = Logger(console=False, file_path=str(log_file), ring_buffer=True, min_level=LogLevel.TRACE)

    logger.trace(LogChannel.CORE, "Trace message")
    logger.debug(LogChannel.GFX, "Debug message")
    logger.info(LogChannel.WINDOW, "Info message")
    logger.warn(LogChannel.PHYSICS, "Warn message")
    logger.error(LogChannel.AUDIO, "Error message")
    logger.fatal(LogChannel.INPUT, "Fatal message")

    ring = logger.get_ring_buffer()
    assert ring is not None
    entries = ring.get_entries()
    assert len(entries) == 6

    # Test filtering by channel
    gfx_entries = ring.get_entries(channel="Gfx")
    assert len(gfx_entries) == 1
    assert gfx_entries[0].message == "Debug message"

    # Test filtering by minimum level
    warn_entries = ring.get_entries(min_level=LogLevel.WARN)
    assert len(warn_entries) == 3

    # Test file output
    logger.flush()
    logger.shutdown()
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Trace message" in content
    assert "Fatal message" in content


def test_logging_release_mode_stripping(tmp_path: Path):
    log_file = tmp_path / "release.log"
    logger = Logger(console=False, file_path=str(log_file), ring_buffer=True)
    logger.set_release_mode(True)
    assert logger.is_release_mode

    # In release mode: trace, debug, info should be discarded immediately
    logger.trace(LogChannel.CORE, "Should be stripped")
    logger.debug(LogChannel.CORE, "Should be stripped")
    logger.info(LogChannel.CORE, "Should be stripped")

    # Warn, error, fatal should pass through for crash logs
    logger.warn(LogChannel.CORE, "Keep warning")
    logger.error(LogChannel.CORE, "Keep error")
    logger.fatal(LogChannel.CORE, "Keep fatal")

    ring = logger.get_ring_buffer()
    entries = ring.get_entries()
    assert len(entries) == 3
    assert entries[0].level == LogLevel.WARN
    assert entries[1].level == LogLevel.ERROR
    assert entries[2].level == LogLevel.FATAL

    logger.shutdown()
