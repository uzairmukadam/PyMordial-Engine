"""Reproducible graphical benchmark for PyMordial Engine.

Runs a deterministic scene through the real deferred render pipeline and records
CPU frame time, asynchronous total-GPU frame time, and per-pass GPU timings.

This is a development tool. It does not optimize or alter any graphical feature;
it only measures the existing renderer through its existing interfaces
(``RenderPipeline.gpu_profiler``, ``SystemMonitor`` GPU queries, ``RenderConfig``).

Usage::

    python scripts/benchmark_graphics.py --config BASELINE_LOW --view VIEW_TOP
    python scripts/benchmark_graphics.py --config HIGH --view VIEW_CLOSE --frames 240
    python scripts/benchmark_graphics.py --config BASELINE_LOW --view VIEW_TOP \
        --set gi_mode=OFF --set ssr_enabled=false --output results/top.json

See ``docs/guides/BENCHMARKING.md`` for the full reference.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from engine.app.config import ProjectConfig
from engine.app.project_app import ProjectApp
from engine.camera import FreeFlyCamera
from engine.debug.monitor import SystemMonitor
from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset

SCHEMA_VERSION = 1
TOOL_NAME = "pymordial-graphics-benchmark"

# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------

# Named benchmark configurations. Each is derived from an existing quality
# preset, then adjusted with explicit, documented overrides. BASELINE_LOW keeps
# the renderer able to produce a normal frame while disabling optional effects.
CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "BASELINE_LOW": {
        "base": "low",
        "overrides": {
            "gi_mode": "OFF",
            "ssr_enabled": False,
            "volumetric_fog_enabled": False,
            "bloom_enabled": False,
            "lens_flare_enabled": False,
            "dof_enabled": False,
            "motion_blur_enabled": False,
            "tess_enabled": False,
            "ssdm_enabled": False,
            "pom_enabled": False,
            "auto_exposure_enabled": False,
        },
    },
    "LOW": {"base": "low", "overrides": {}},
    "MEDIUM": {"base": "medium", "overrides": {}},
    "HIGH": {"base": "high", "overrides": {}},
    "ULTRA": {"base": "ultra", "overrides": {}},
    "CINEMATIC": {"base": "cinematic", "overrides": {}},
}

# Fields that may be overridden from the CLI via --set key=value.
_OVERRIDABLE_FIELDS = set(RenderConfig.__dataclass_fields__.keys())


def _coerce(value: str) -> Any:
    """Coerces a CLI string into bool/int/float/str for a RenderConfig field."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_config(name: str, overrides: dict[str, Any] | None = None) -> RenderConfig:
    """Builds a RenderConfig for a named benchmark configuration.

    Args:
        name: A key of ``CONFIG_PRESETS`` (case-insensitive).
        overrides: Additional field overrides applied last.

    Returns:
        An independent RenderConfig instance.
    """
    key = name.upper()
    if key not in CONFIG_PRESETS:
        raise ValueError(f"Unknown config {name!r}. Expected one of {sorted(CONFIG_PRESETS)}.")
    spec = CONFIG_PRESETS[key]
    cfg = get_quality_preset(GraphicsQuality(spec["base"]))
    for field, value in spec["overrides"].items():
        setattr(cfg, field, value)
    if overrides:
        for field, value in overrides.items():
            if field not in _OVERRIDABLE_FIELDS:
                raise ValueError(f"Unknown RenderConfig field {field!r}.")
            setattr(cfg, field, value)
    return cfg


def config_to_dict(cfg: RenderConfig) -> dict[str, Any]:
    """Serializes a RenderConfig to a JSON-safe dict."""
    out: dict[str, Any] = {}
    for field in RenderConfig.__dataclass_fields__:
        value = getattr(cfg, field)
        if isinstance(value, tuple):
            value = list(value)
        out[field] = value
    return out


# ---------------------------------------------------------------------------
# Deterministic scene
# ---------------------------------------------------------------------------

SCENE_ID = "benchmark_showcase_v1"


def build_scene(app: ProjectApp) -> dict[str, Any]:
    """Builds the deterministic benchmark scene and returns its metadata.

    The scene is intentionally small (~32 entities) and exercises normal
    renderer behaviour: a ground plane, curved/rough/reflective primitives at
    near/mid/far depths, shadow casters, and AO/contact regions. Polygon count
    is not a benchmark variable.
    """
    ecs = app.ecs
    counts: dict[str, int] = {}

    def add(shape: str, pos, dims, color, rough, metal, static=True) -> None:
        ecs.create_entity(
            shape=shape,
            position=pos,
            dimensions=dims,
            color=color,
            roughness=rough,
            metallic=metal,
            is_static=static,
        )
        counts[shape] = counts.get(shape, 0) + 1

    # Ground plane (large flat cube) — AO/contact + shadow receiver.
    add("cube", (0.0, -0.5, 0.0), (60.0, 1.0, 60.0), (0.16, 0.16, 0.18), 0.85, 0.0)

    # A ring of objects at varying depths and materials.
    ring = [
        # (radius, angle_deg, shape, height, color, roughness, metallic)
        (6.0, 0.0, "sphere", 1.2, (0.85, 0.30, 0.25), 0.15, 0.0),
        (6.0, 60.0, "cylinder", 1.6, (0.25, 0.55, 0.85), 0.35, 0.0),
        (6.0, 120.0, "cone", 1.8, (0.30, 0.80, 0.40), 0.55, 0.0),
        (6.0, 180.0, "sphere", 1.0, (0.90, 0.85, 0.30), 0.05, 1.0),
        (6.0, 240.0, "cube", 1.4, (0.70, 0.70, 0.75), 0.20, 1.0),
        (6.0, 300.0, "capsule", 1.5, (0.55, 0.35, 0.75), 0.45, 0.0),
        (12.0, 30.0, "sphere", 2.0, (0.20, 0.75, 0.80), 0.10, 0.0),
        (12.0, 150.0, "cylinder", 2.4, (0.80, 0.45, 0.20), 0.30, 0.0),
        (12.0, 270.0, "cube", 2.2, (0.35, 0.35, 0.40), 0.15, 1.0),
        (20.0, 90.0, "sphere", 3.0, (0.60, 0.20, 0.55), 0.25, 0.0),
        (20.0, 210.0, "cone", 3.2, (0.25, 0.65, 0.35), 0.60, 0.0),
        (20.0, 330.0, "cylinder", 2.8, (0.85, 0.80, 0.70), 0.40, 0.0),
    ]
    for radius, angle, shape, height, color, rough, metal in ring:
        rad = math.radians(angle)
        x = math.cos(rad) * radius
        z = math.sin(rad) * radius
        dims = (height, height * 1.4, height) if shape in ("cylinder", "cone", "capsule") else (height, height, height)
        add(shape, (x, height * 0.5, z), dims, color, rough, metal)

    # A cluster of small contact objects near the origin for AO detail.
    for i in range(8):
        ang = math.radians(i * 45.0)
        add(
            "sphere",
            (math.cos(ang) * 2.2, 0.35, math.sin(ang) * 2.2),
            (0.7, 0.7, 0.7),
            (0.5 + 0.05 * i, 0.4, 0.6 - 0.05 * i),
            0.3,
            0.0,
        )

    # A tall bright emissive-ish pillar for light-relevant regions.
    add("cylinder", (0.0, 4.0, -14.0), (8.0, 1.2, 8.0), (1.0, 0.95, 0.85), 0.2, 0.0)

    total = sum(counts.values())
    return {
        "id": SCENE_ID,
        "entity_count": total,
        "shape_counts": counts,
        "description": "Ground plane + 12-object depth ring + 8 contact spheres + 1 pillar",
    }


# ---------------------------------------------------------------------------
# Deterministic camera workloads
# ---------------------------------------------------------------------------

VIEWS = ("VIEW_TOP", "VIEW_CLOSE", "VIEW_REFLECTIVE", "VIEW_DEPTH", "VIEW_MOTION")


def camera_pose(view: str, frame_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Returns a deterministic (position, target) for a view at a frame index.

    Static views ignore ``frame_index`` so every run uses identical camera state.
    ``VIEW_MOTION`` uses a fixed, frame-indexed orbit so the sequence is identical
    across runs.
    """
    key = view.upper()
    if key == "VIEW_TOP":
        return (0.0, 34.0, 34.0), (0.0, 0.0, 0.0)
    if key == "VIEW_CLOSE":
        return (0.0, 2.2, 7.5), (0.0, 1.2, 0.0)
    if key == "VIEW_REFLECTIVE":
        return (5.5, 1.6, 5.5), (0.0, 1.0, 0.0)
    if key == "VIEW_DEPTH":
        return (0.0, 6.0, 46.0), (0.0, 2.0, -10.0)
    if key == "VIEW_MOTION":
        # Deterministic orbit: one full revolution over 240 frames.
        t = (frame_index % 240) / 240.0
        ang = t * 2.0 * math.pi
        radius = 16.0
        pos = (math.cos(ang) * radius, 8.0, math.sin(ang) * radius)
        return pos, (0.0, 1.5, 0.0)
    raise ValueError(f"Unknown view {view!r}. Expected one of {VIEWS}.")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(values: list[float]) -> dict[str, float] | None:
    """Returns min/median/mean/p95/max for a timing series, or None if empty."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "samples": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        width, height = int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid resolution {text!r}; expected WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return width, height


def run_benchmark(
    config_name: str,
    view: str,
    resolution: tuple[int, int],
    warmup: int,
    frames: int,
    overrides: dict[str, Any] | None = None,
    windowed: bool = False,
) -> dict[str, Any]:
    """Runs one benchmark and returns a JSON-safe result dict."""
    cfg = build_config(config_name, overrides)
    width, height = resolution

    project_cfg = ProjectConfig(
        title="PyMordial Graphics Benchmark",
        width=width,
        height=height,
        quality_preset=cfg_quality_name(config_name),
        enable_debug=False,
        headless=not windowed,
        particles_enabled=False,
        max_entities=4096,
        vsync=False,
    )
    app = ProjectApp(project_cfg)
    # Apply the exact benchmark RenderConfig (preset + overrides) to the pipeline.
    app.pipeline.apply_config(cfg)

    scene_meta = build_scene(app)

    camera = FreeFlyCamera(position=(0.0, 34.0, 34.0), yaw_deg=0.0, pitch_deg=0.0, fov=60.0)
    app.camera_manager.register_camera("bench", camera, make_active=True)

    # Asynchronous total-GPU frame timer (existing SystemMonitor query ring).
    monitor = SystemMonitor()
    dt = 1.0 / 60.0

    def step(frame_index: int) -> None:
        pos, target = camera_pose(view, frame_index)
        camera.position[:] = pos
        camera.target[:] = target
        camera.mark_dirty()
        monitor.begin_gpu_frame()
        try:
            app.step_frame(dt)
        finally:
            monitor.end_gpu_frame()

    try:
        for i in range(warmup):
            step(i)

        cpu_ms: list[float] = []
        gpu_ms: list[float] = []
        pass_series: dict[str, list[float]] = {}
        profiler = app.pipeline.gpu_profiler

        for i in range(frames):
            t0 = time.perf_counter()
            step(warmup + i)
            cpu_ms.append((time.perf_counter() - t0) * 1000.0)

            if monitor.gpu_frame_time_ms is not None:
                gpu_ms.append(float(monitor.gpu_frame_time_ms))
            for name, value in profiler.results().items():
                if value > 0.0:
                    pass_series.setdefault(name, []).append(float(value))

        stats = app.pipeline.feature_state()
        draw_commands = int(app.pipeline.mdi.command_count + app.pipeline.tess_mdi.command_count)
        triangles = 0
        for batcher in (app.pipeline.mdi, app.pipeline.tess_mdi):
            for cmd in batcher.command_buffer[: batcher.command_count]:
                triangles += int(cmd["count"]) * int(cmd["instance_count"]) // 3
        visible = int(app.pipeline.stats_visible_entities)
        ctx_info = dict(app.render_ctx.ctx.info)
    finally:
        monitor.destroy()
        app.shutdown()

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": {"width": width, "height": height},
        "scene": scene_meta,
        "view": view.upper(),
        "warmup_frames": warmup,
        "measured_frames": frames,
        "config_name": config_name.upper(),
        "config": config_to_dict(cfg),
        "active_feature_state": stats,
        "device": {
            "gl_renderer": ctx_info.get("GL_RENDERER"),
            "gl_version": ctx_info.get("GL_VERSION"),
            "gl_vendor": ctx_info.get("GL_VENDOR"),
        },
        "cpu": {
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
        },
        "scene_stats": {
            "visible_entities": visible,
            "draw_commands": draw_commands,
            "triangles_estimate": triangles,
        },
        "timing": {
            "cpu_frame_ms": compute_stats(cpu_ms),
            "gpu_frame_ms": compute_stats(gpu_ms) if gpu_ms else None,
            "gpu_frame_ms_available": bool(gpu_ms),
        },
        "passes": {name: compute_stats(series) for name, series in sorted(pass_series.items())},
    }
    return result


def cfg_quality_name(config_name: str) -> str:
    """Maps a benchmark config name to the ProjectConfig quality preset string."""
    key = config_name.upper()
    if key in CONFIG_PRESETS:
        return CONFIG_PRESETS[key]["base"]
    return "low"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_human(result: dict[str, Any]) -> str:
    """Renders a compact human-readable summary of a benchmark result."""
    lines: list[str] = []
    res = result["resolution"]
    lines.append(f"PyMordial Graphics Benchmark - {result['config_name']} / {result['view']}")
    lines.append(f"  Resolution : {res['width']}x{res['height']}")
    lines.append(f"  Scene      : {result['scene']['id']} ({result['scene']['entity_count']} entities)")
    lines.append(f"  Frames     : {result['warmup_frames']} warmup + {result['measured_frames']} measured")
    dev = result["device"]
    lines.append(f"  Device     : {dev.get('gl_renderer')} | {dev.get('gl_version')}")

    cpu = result["timing"]["cpu_frame_ms"]
    if cpu:
        lines.append(
            f"  CPU frame  : median {cpu['median']:.3f} ms  (min {cpu['min']:.3f}, "
            f"p95 {cpu['p95']:.3f}, max {cpu['max']:.3f})"
        )
    gpu = result["timing"]["gpu_frame_ms"]
    if gpu:
        lines.append(
            f"  GPU frame  : median {gpu['median']:.3f} ms  (min {gpu['min']:.3f}, "
            f"p95 {gpu['p95']:.3f}, max {gpu['max']:.3f})"
        )
    else:
        lines.append("  GPU frame  : unavailable (async query not ready)")

    ss = result["scene_stats"]
    lines.append(
        f"  Scene stats: visible {ss['visible_entities']}, draw cmds {ss['draw_commands']}, "
        f"~{ss['triangles_estimate']} tris"
    )

    if result["passes"]:
        lines.append("  GPU passes (median ms):")
        for name, st in result["passes"].items():
            if st:
                lines.append(f"    {name:<20} {st['median']:.3f}")
    return "\n".join(lines)


def default_output_path(config_name: str, view: str) -> Path:
    """Returns the default JSON output path under build/benchmarks/."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("build") / "benchmarks" / f"{config_name.lower()}_{view.lower()}_{stamp}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible PyMordial graphical benchmark (deterministic scene + camera).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="BASELINE_LOW", choices=sorted(CONFIG_PRESETS),
                        help="Named benchmark configuration.")
    parser.add_argument("--view", default="VIEW_TOP", choices=list(VIEWS),
                        help="Deterministic camera workload.")
    parser.add_argument("--resolution", type=_parse_resolution, default=(1280, 720),
                        help="Render resolution as WIDTHxHEIGHT.")
    parser.add_argument("--warmup", type=int, default=60, help="Warmup frames (excluded from stats).")
    parser.add_argument("--frames", type=int, default=240, help="Measured frames.")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (default: build/benchmarks/<config>_<view>_<time>.json).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a RenderConfig field (repeatable).")
    parser.add_argument("--windowed", action="store_true",
                        help="Render to a visible window instead of headless.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable output.")
    return parser


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parses repeated ``KEY=VALUE`` CLI overrides into a dict."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        out[key.strip()] = _coerce(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.frames <= 0:
        parser.error("--frames must be > 0")

    try:
        overrides = parse_overrides(args.set)
        result = run_benchmark(
            config_name=args.config,
            view=args.view,
            resolution=args.resolution,
            warmup=args.warmup,
            frames=args.frames,
            overrides=overrides,
            windowed=args.windowed,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    output = args.output or default_output_path(args.config, args.view)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        print(format_human(result))
        print(f"\nJSON written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
