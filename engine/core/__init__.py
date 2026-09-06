"""PyMordial Core Engine Systems.

Contains memory tables, entity pool allocator, deterministic tick loop,
semantic input system, and vectorized math routines.
"""

from engine.core.entity_pool import EntityPool
from engine.core.ecs import EntityManager, TransformProxy
from engine.core.loop import EngineLoop
from engine.input import InputManager

__all__ = [
    "EntityPool",
    "EntityManager",
    "TransformProxy",
    "EngineLoop",
    "InputManager",
]
