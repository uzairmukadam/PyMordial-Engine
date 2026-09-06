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
    assert win.vsync == VSyncMode.OFF
    win.set_vsync(VSyncMode.ON)
    assert win.vsync == VSyncMode.ON
    win.set_vsync(VSyncMode.OFF)
    assert win.vsync == VSyncMode.OFF


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


def test_window_resizable_flag():
    cfg_default = WindowConfig(width=800, height=600, hidden=True)
    assert not cfg_default.resizable
    win = Window(cfg_default)
    assert not win.config.resizable

    cfg_resizable = WindowConfig(width=800, height=600, hidden=True, resizable=True)
    assert cfg_resizable.resizable
    win2 = Window(cfg_resizable)
    assert win2.config.resizable


def test_window_set_mode_transitions():
    cfg = WindowConfig(width=800, height=600, hidden=True)
    win = Window(cfg)
    assert win.mode == WindowMode.WINDOWED

    win.set_mode(WindowMode.BORDERLESS_FULLSCREEN)
    assert win.mode == WindowMode.BORDERLESS_FULLSCREEN
    assert win.width == win._desktop_width
    assert win.height == win._desktop_height

    win.set_mode(WindowMode.WINDOWED)
    assert win.mode == WindowMode.WINDOWED
    assert win.width == 800
    assert win.height == 600


