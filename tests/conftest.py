"""Pytest configuration and shared fixtures for PyMordial Engine tests."""

import os
import pytest

# Ensure audio driver does not block in test environments
os.environ["SDL_AUDIODRIVER"] = "dummy"

from engine.core.ecs import EntityManager
from engine.core.entity_pool import EntityPool
from engine.core.loop import EngineLoop


@pytest.fixture
def entity_pool():
    """Provides a fresh EntityPool with 1,000 capacity for fast testing."""
    return EntityPool(max_entities=1_000)


@pytest.fixture
def ecs():
    """Provides a fresh EntityManager with 1,000 capacity."""
    return EntityManager(max_entities=1_000)


@pytest.fixture
def engine_loop(ecs):
    """Provides a fresh EngineLoop instance."""
    return EngineLoop(ecs=ecs, fixed_dt=1.0 / 60.0)
