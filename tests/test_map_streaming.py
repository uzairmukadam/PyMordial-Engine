"""Unit Tests for Spatial Map Streaming Subsystem."""

from __future__ import annotations
import pytest

from engine.app.project_app import ProjectApp
from engine.app.config import ProjectConfig
from projects.shotgun_escape_the_heat.world.map_streamer import MapStreamer


@pytest.fixture
def headless_app():
    """Creates a minimal headless app instance for streaming testing."""
    config = ProjectConfig(
        title="TestMapStreamer",
        headless=True,
        max_frames=1,
        enable_debug=False,
    )
    app = ProjectApp(config)
    yield app
    app.shutdown()


def test_map_streamer_spatial_indexing() -> None:
    """Verifies that static elements are placed into correct spatial chunks."""
    streamer = MapStreamer(chunk_size=200.0, load_radius=400.0, unload_radius=500.0)

    # Element 1: at (50, 0, 50) -> chunk (0, 0)
    elem1 = streamer.add_element(
        "road",
        position=(50.0, 0.25, 50.0),
        scale=(14.0, 0.5, 50.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        color=(0.1, 0.1, 0.1),
    )
    # Element 2: at (650, 0, 650) -> chunk (3, 3)
    elem2 = streamer.add_element(
        "building",
        position=(650.0, 0.5, 650.0),
        scale=(30.0, 1.0, 30.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        color=(0.3, 0.3, 0.3),
    )

    assert (0, 0) in streamer.chunks
    assert (3, 3) in streamer.chunks
    assert elem1 in streamer.chunks[(0, 0)].elements
    assert elem2 in streamer.chunks[(3, 3)].elements


def test_map_streamer_dynamic_activation_and_hysteresis(headless_app: ProjectApp) -> None:
    """Verifies that chunks stream in within load_radius and stream out beyond unload_radius."""
    app = headless_app
    streamer = MapStreamer(chunk_size=200.0, load_radius=350.0, unload_radius=450.0)

    # Origin chunk element
    near_elem = streamer.add_element(
        "road",
        position=(0.0, 0.25, 0.0),
        scale=(14.0, 0.5, 100.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        color=(0.1, 0.1, 0.1),
    )
    # Far chunk element (700m away)
    far_elem = streamer.add_element(
        "building",
        position=(700.0, 0.5, 0.0),
        scale=(20.0, 1.0, 20.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        color=(0.3, 0.3, 0.3),
    )

    # Build at origin (0, 0, 0)
    streamer.build(app, initial_pos=(0.0, 0.0, 0.0))

    # Near chunk should be active, far chunk should be inactive
    near_chunk = streamer.chunks[(0, 0)]
    far_chunk = streamer.chunks[(3, 0)]

    assert near_chunk.is_active is True
    assert near_elem.is_body_active is True
    assert far_chunk.is_active is False
    assert far_elem.is_body_active is False

    # Player drives towards the far chunk: position (600, 0, 0)
    # Distance to far chunk center is now ~100m (< load_radius 350m)
    # Distance to near chunk center is now 500m (> unload_radius 450m)
    streamer.update_streaming(600.0, 0.0, force=True)

    assert far_chunk.is_active is True
    assert far_elem.is_body_active is True
    assert near_chunk.is_active is False
    assert near_elem.is_body_active is False

    # Clean teardown
    streamer.teardown(app)
    assert far_elem.is_body_active is False
    assert near_elem.is_body_active is False
