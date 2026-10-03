"""Pytest root configuration ensuring zero GPU resource leaks and audio driver safety."""

import os
import gc

# Ensure audio driver does not block or require real hardware in test environments
os.environ["SDL_AUDIODRIVER"] = "dummy"


def pytest_runtest_teardown(item, nextitem):
    """Universal teardown ensuring immediate garbage collection of WGL/GL contexts and buffers."""
    gc.collect()
