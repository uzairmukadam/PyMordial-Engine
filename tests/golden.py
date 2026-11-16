"""Deterministic headless frame capture used as the Horizon 2 visual regression gate.

The helper builds a fixed scene (ground plane + a few primitives + a fixed camera),
renders a fixed number of frames, and reads back the post-processed LDR framebuffer.
Golden images live in ``tests/baseline/`` and are compared with a mean-absolute-error
tolerance so that driver-level float noise does not cause spurious failures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.world.default_world import DefaultWorldBuilder

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"

# Fixed capture resolution: small enough for fast CI, large enough to expose
# sky/atmosphere regressions (gradient banding, LUT misbinding, sun disc loss).
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 180
CAPTURE_FRAMES = 3
CAPTURE_DT = 1.0 / 60.0

# Mean absolute error (0..255 scale) tolerated between a capture and its golden image.
DEFAULT_TOLERANCE = 2.0


def build_capture_app(**config_overrides) -> ProjectApp:
    """Creates a deterministic headless app with a fixed world and camera."""
    cfg = ProjectConfig(
        title="GoldenCapture",
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
        headless=True,
        enable_debug=False,
        quality_preset="high",
        particles_enabled=False,
    )
    for key, value in config_overrides.items():
        setattr(cfg, key, value)

    app = ProjectApp(cfg)
    app.set_world_builder(DefaultWorldBuilder(size=40.0, uv_tiles=8.0))

    # A few primitives give the frame depth, shadowing, and silhouette coverage.
    # `shape=` is passed explicitly so they land in `explicit_mesh_draw_batches`,
    # which the host appends alongside the world builder's static batches.
    ecs = app.ecs
    ecs.create_entity(position=(0.0, 1.0, 0.0), scale=(1.0, 1.0, 1.0), shape="cube", color=(0.85, 0.35, 0.25))
    ecs.create_entity(position=(3.0, 0.75, -2.0), scale=(0.75, 0.75, 0.75), shape="sphere", color=(0.25, 0.55, 0.85))
    ecs.create_entity(position=(-3.0, 0.5, -4.0), scale=(0.5, 0.5, 0.5), shape="capsule", color=(0.35, 0.80, 0.40))

    from engine.camera.camera import VirtualCamera

    camera = VirtualCamera(position=(6.0, 4.0, 9.0), target=(0.0, 1.0, 0.0), fov=60.0)
    app.camera_manager.register_camera("capture", camera, make_active=True)
    return app


def capture_frame(app: ProjectApp) -> np.ndarray:
    """Renders the fixed number of frames and returns the LDR framebuffer as (H, W, 3) uint8."""
    for _ in range(CAPTURE_FRAMES):
        app.step_frame(CAPTURE_DT)

    final_fbo = app.pipeline.post_process.final_fbo
    raw = final_fbo.read(components=3, dtype="f1")
    img = np.frombuffer(raw, dtype=np.uint8).reshape((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3))
    # OpenGL reads bottom-up; flip so the array matches image orientation.
    return np.ascontiguousarray(img[::-1])


def capture_scene(**config_overrides) -> np.ndarray:
    """Builds, renders, and tears down a capture app, returning the frame."""
    app = build_capture_app(**config_overrides)
    try:
        return capture_frame(app)
    finally:
        app.shutdown()


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-channel difference on the 0..255 scale."""
    if a.shape != b.shape:
        raise AssertionError(f"Frame shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def golden_path(name: str) -> Path:
    """Returns the golden image path for a named capture."""
    return BASELINE_DIR / f"{name}.npy"


def save_golden(name: str, frame: np.ndarray) -> Path:
    """Persists a captured frame as a golden reference."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(name)
    np.save(path, frame)
    return path


def load_golden(name: str) -> np.ndarray:
    """Loads a golden reference frame."""
    path = golden_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing golden frame '{name}' at {path}. "
            f"Regenerate with: python -m tests.golden_capture --write"
        )
    return np.load(path)
