"""Unit tests for PyMordial Engine Windowing Subsystem."""

from __future__ import annotations
from engine.window import Window, WindowConfig, WindowMode, VSyncMode
from engine.events import subscribe_event, WindowResizeEvent


def test_window_initialization_headless():
    cfg = WindowConfig(width=1024, height=768, hidden=True)
    win = Window(cfg)
    assert win.width == 1024
    assert win.height == 768
    assert win.is_headless
    assert win.mode == WindowMode.WINDOWED
    assert win.vsync == VSyncMode.ON


def test_window_runtime_resolution_change():
    cfg = WindowConfig(width=800, height=600, hidden=True)
    win = Window(cfg)

    received_events: list[tuple[int, int]] = []
    subscribe_event(WindowResizeEvent, lambda e: received_events.append((e.width, e.height)))

    win.set_resolution(1280, 720)
    assert win.width == 1280
    assert win.height == 720
    assert (1280, 720) in received_events


def test_window_supported_resolutions():
    cfg = WindowConfig(width=800, height=600, hidden=True)
    win = Window(cfg)
    res_list = win.get_supported_resolutions()
    assert len(res_list) > 0
    assert (1280, 720) in res_list
