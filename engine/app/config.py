"""Project and Engine Application Configuration.

Defines the declarative configuration for any PyMordial project, specifying display,
pipeline quality, physics stepping frequency, debug flags, and asset paths.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    """Strongly-typed declarative configuration for a PyMordial project."""

    title: str = "PyMordial Project"
    width: int = 1280
    height: int = 720
    fullscreen: bool = False
    vsync: bool = True
    quality_preset: str = "ultra"  # "low", "medium", "high", "ultra", "cinematic"
    fixed_hz: float = 60.0
    gravity: tuple[float, float, float] = (0.0, -20.0, 0.0)
    enable_debug: bool = True
    headless: bool = False
    max_frames: int | None = None
    asset_dir: Path | str | None = None
    screenshot: Path | str | None = None
    custom_settings: dict[str, object] = field(default_factory=dict)
