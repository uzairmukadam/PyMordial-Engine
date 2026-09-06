"""High-Performance Centralized Event Bus for PyMordial Engine."""

from __future__ import annotations
from collections import deque
import threading
from typing import Any, Callable, TypeVar
from engine.events.event import Event
from engine.logging import log_error, log_trace, LogChannel

T = TypeVar("T", bound=Event)
EventHandler = Callable[[Any], None]


class EventBus:
    """Type-safe publish-subscribe message bus supporting priority listeners and queues."""

    __slots__ = ("_subscribers", "_queue", "_lock")

    def __init__(self) -> None:
        # Map event_type -> list of (priority, handler)
        self._subscribers: dict[type[Event], list[tuple[int, EventHandler]]] = {}
        self._queue: deque[Event] = deque()
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[T], handler: Callable[[T], None], priority: int = 0) -> None:
        """Registers a listener for a specific event type.
        
        Higher priority numbers execute earlier.
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            # Insert keeping sorted descending by priority
            listeners = self._subscribers[event_type]
            listeners.append((priority, handler))
            listeners.sort(key=lambda item: item[0], reverse=True)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> bool:
        """Removes a registered listener."""
        with self._lock:
            if event_type not in self._subscribers:
                return False
            listeners = self._subscribers[event_type]
            initial_len = len(listeners)
            self._subscribers[event_type] = [item for item in listeners if item[1] != handler]
            return len(self._subscribers[event_type]) < initial_len

    def publish(self, event: Event) -> None:
        """Immediately dispatches an event synchronously to all registered listeners."""
        event_cls = type(event)
        with self._lock:
            listeners = list(self._subscribers.get(event_cls, []))

        for _, handler in listeners:
            if event.handled:
                break
            try:
                handler(event)
            except Exception as e:
                log_error(LogChannel.EVENTS, f"Exception in event handler {handler.__name__}: {e}")

    def queue(self, event: Event) -> None:
        """Enqueues an event for deferred phase-aligned dispatch."""
        with self._lock:
            self._queue.append(event)

    def flush(self) -> int:
        """Dispatches all enqueued events in FIFO order. Returns count dispatched."""
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()

        for ev in batch:
            self.publish(ev)

        return len(batch)

    def clear(self) -> None:
        """Clears all subscribers and queued events."""
        with self._lock:
            self._subscribers.clear()
            self._queue.clear()


# Global default EventBus
_GLOBAL_EVENT_BUS: EventBus = EventBus()


def get_event_bus() -> EventBus:
    """Retrieves the global default EventBus instance."""
    return _GLOBAL_EVENT_BUS


def subscribe_event(event_type: type[T], handler: Callable[[T], None], priority: int = 0) -> None:
    _GLOBAL_EVENT_BUS.subscribe(event_type, handler, priority)


def unsubscribe_event(event_type: type[T], handler: Callable[[T], None]) -> bool:
    return _GLOBAL_EVENT_BUS.unsubscribe(event_type, handler)


def publish_event(event: Event) -> None:
    _GLOBAL_EVENT_BUS.publish(event)


def queue_event(event: Event) -> None:
    _GLOBAL_EVENT_BUS.queue(event)


def flush_events() -> int:
    return _GLOBAL_EVENT_BUS.flush()
