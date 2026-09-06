"""Unit tests for PyMordial Engine Centralized Event Bus."""

from __future__ import annotations
import pytest
from engine.events import (
    Event,
    EventBus,
    WindowResizeEvent,
    InputActionEvent,
    CVarChangedEvent,
)


def test_event_bus_priority_dispatch():
    bus = EventBus()
    order: list[str] = []

    def low_priority(e: WindowResizeEvent):
        order.append("low")

    def high_priority(e: WindowResizeEvent):
        order.append("high")

    def mid_priority(e: WindowResizeEvent):
        order.append("mid")

    bus.subscribe(WindowResizeEvent, low_priority, priority=0)
    bus.subscribe(WindowResizeEvent, high_priority, priority=100)
    bus.subscribe(WindowResizeEvent, mid_priority, priority=50)

    bus.publish(WindowResizeEvent(1920, 1080))
    assert order == ["high", "mid", "low"]


def test_event_bus_stop_propagation():
    bus = EventBus()
    order: list[str] = []

    def blocker(e: WindowResizeEvent):
        order.append("blocker")
        e.stop_propagation()

    def blocked(e: WindowResizeEvent):
        order.append("blocked")

    bus.subscribe(WindowResizeEvent, blocker, priority=10)
    bus.subscribe(WindowResizeEvent, blocked, priority=0)

    bus.publish(WindowResizeEvent(1280, 720))
    assert order == ["blocker"]


def test_event_bus_queued_dispatch():
    bus = EventBus()
    received: list[str] = []

    bus.subscribe(InputActionEvent, lambda e: received.append(e.action))

    bus.queue(InputActionEvent(action="jump", is_pressed=True))
    bus.queue(InputActionEvent(action="sprint", is_pressed=True))

    # Queue should not dispatch immediately
    assert len(received) == 0

    # Flush processes in FIFO order
    count = bus.flush()
    assert count == 2
    assert received == ["jump", "sprint"]


def test_event_bus_unsubscribe():
    bus = EventBus()
    count = [0]

    def handler(e: WindowResizeEvent):
        count[0] += 1

    bus.subscribe(WindowResizeEvent, handler)
    bus.publish(WindowResizeEvent(800, 600))
    assert count[0] == 1

    assert bus.unsubscribe(WindowResizeEvent, handler)
    bus.publish(WindowResizeEvent(1024, 768))
    assert count[0] == 1
