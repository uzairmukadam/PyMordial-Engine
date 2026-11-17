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

# Named capture variants. Each entry is the config override set applied on top of
# the deterministic base scene, so a regression in one subsystem is isolated to
# its own golden instead of being averaged into a single frame.
#
# Keys are split by target: ``project`` overrides land on ``ProjectConfig`` before
# the app is built, ``render`` overrides land on the pipeline's ``RenderConfig``
# afterwards. Render settings are not ProjectConfig fields, so they must be
# applied post-construction.
CAPTURE_VARIANTS: dict[str, dict[str, dict[str, object]]] = {
    "atmosphere_on": {},
    "atmosphere_off": {"project": {"atmosphere_enabled": False}},
    "shadows_pcss": {"render": {"shadow_mode": "PCSS", "shadow_softness": 1.2}},
    "shadows_pcf": {"render": {"shadow_mode": "PCF", "shadow_softness": 1.0}},
    "aa_taa": {"render": {"aa_mode": "TAA", "taa_enabled": True}},
    "aa_smaa4x": {"render": {"aa_mode": "SMAA_4X"}},
    "fog_volumetric": {"render": {"volumetric_fog_enabled": True, "fog_density": 0.012}},
    "motion_blur": {"render": {"motion_blur_enabled": True, "motion_blur_intensity": 1.5}},
    "flare_solar": {"render": {"lens_flare_enabled": True}},
}


def build_capture_app(**config_overrides) -> ProjectApp:
    """Creates a deterministic headless app with a fixed world and camera.

    ``config_overrides`` are ``ProjectConfig`` fields applied before construction.
    """
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


def capture_variant(name: str) -> np.ndarray:
    """Captures a named variant from ``CAPTURE_VARIANTS``.

    ``project`` overrides are applied to ``ProjectConfig`` before the app is
    built; ``render`` overrides are applied to the pipeline's ``RenderConfig``
    afterwards, because render settings are not ``ProjectConfig`` fields.
    """
    if name not in CAPTURE_VARIANTS:
        raise KeyError(f"Unknown capture variant '{name}'. Known: {sorted(CAPTURE_VARIANTS)}")

    spec = CAPTURE_VARIANTS[name]
    project_overrides = dict(spec.get("project", {}))
    render_overrides = dict(spec.get("render", {}))

    app = build_capture_app(**project_overrides)
    try:
        for key, value in render_overrides.items():
            if not hasattr(app.pipeline.config, key):
                raise AttributeError(
                    f"Capture variant '{name}' sets unknown RenderConfig field '{key}'."
                )
            setattr(app.pipeline.config, key, value)
        app.pipeline.apply_config(app.pipeline.config)
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
