"""Project and Configuration Validation for PyMordial Engine.

Performs early validation of project directory layouts and ProjectConfig instances
before allocating heavy graphics contexts or physics worlds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from engine.gfx.quality_presets import GraphicsQuality

if TYPE_CHECKING:
    from engine.app.config import ProjectConfig


class ProjectValidationError(Exception):
    """Raised when a project directory structure or configuration is invalid."""
    pass


@dataclass
class ValidationReport:
    """Contains diagnostic messages from a project validation check."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.is_valid or self.errors:
            msg = "Project validation failed with errors:\n" + "\n".join(f"  - {e}" for e in self.errors)
            raise ProjectValidationError(msg)


def validate_project_config(config: ProjectConfig) -> ValidationReport:
    """Validates runtime configuration parameters before subsystem initialization."""
    report = ValidationReport()

    # 1. Display dimensions
    if config.width <= 0 or config.height <= 0:
        report.errors.append(f"Invalid window resolution: ({config.width}x{config.height}). Dimensions must be positive.")

    # 2. Window title
    if not str(getattr(config, "title", "")).strip():
        report.errors.append("Invalid window title: must be a non-empty string.")

    # 3. Physics fixed tick rate
    if config.fixed_hz <= 0.0 or config.fixed_hz > 1000.0:
        report.errors.append(f"Invalid fixed tick rate: {config.fixed_hz} Hz. Expected 1.0 <= fixed_hz <= 1000.0.")

    # 4. Graphics quality preset
    valid_presets = {q.value for q in GraphicsQuality}
    if str(config.quality_preset).lower() not in valid_presets:
        report.errors.append(
            f"Invalid quality preset: '{config.quality_preset}'. Expected one of: {sorted(valid_presets)}."
        )

    # 5. Gravity vector
    gravity = getattr(config, "gravity", None)
    try:
        gravity_len = len(gravity)  # type: ignore[arg-type]
    except TypeError:
        gravity_len = -1
    if gravity_len != 3:
        report.errors.append(f"Invalid gravity vector: {gravity!r}. Must be a 3-tuple (gx, gy, gz).")
    else:
        for val in gravity:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                report.errors.append(f"Gravity components must be numeric, got: {gravity!r}.")
                break

    # 6. Entity capacity
    if getattr(config, "max_entities", 1) <= 0:
        report.errors.append(f"Invalid max_entities: {config.max_entities}. Must be > 0.")

    # 7. Headless / fullscreen are mutually exclusive presentation modes
    if getattr(config, "headless", False) and getattr(config, "fullscreen", False):
        report.errors.append("Invalid presentation mode: 'headless' and 'fullscreen' cannot both be enabled.")

    # 8. Frame budget (None means run until quit)
    max_frames = getattr(config, "max_frames", None)
    if max_frames is not None and max_frames <= 0:
        report.errors.append(f"Invalid max_frames: {max_frames}. Must be > 0 or None for unlimited.")

    # 9. Asset directory (if specified)
    if config.asset_dir:
        p = Path(config.asset_dir)
        if not p.exists():
            report.warnings.append(f"Configured asset_dir does not exist: {p}")

    report.is_valid = len(report.errors) == 0
    return report


def resolve_project_path(project_identifier: str | Path, engine_root: Path | None = None) -> Path:
    """Resolves a project identifier to an absolute directory path.

    Accepts:
    - Simple name (e.g. 'shotgun_beat_the_speed') looked up under <engine_root>/projects/
    - Relative path (e.g. 'projects/shotgun_beat_the_speed')
    - Absolute filesystem path

    Returns the normalized candidate path even when it does not exist, so callers
    can surface a clean "directory does not exist" diagnostic.
    """
    if engine_root is None:
        engine_root = Path(__file__).resolve().parent.parent.parent

    path = Path(project_identifier)
    if path.is_absolute():
        return path.resolve()

    candidate = engine_root / path
    if candidate.is_dir():
        return candidate.resolve()

    candidate_in_projects = engine_root / "projects" / path
    if candidate_in_projects.is_dir():
        return candidate_in_projects.resolve()

    # Prefer the projects/ location for bare names so error messages point at the
    # canonical project root rather than the engine root.
    if len(path.parts) == 1:
        return candidate_in_projects.resolve()
    return candidate.resolve()


def validate_project_directory(project_path: Path | str) -> ValidationReport:
    """Validates the standard directory structure and entry points of a game project."""
    report = ValidationReport()
    p = Path(project_path).resolve()

    if not p.exists():
        report.errors.append(f"Project directory does not exist: {p}")
        report.is_valid = False
        return report

    if not p.is_dir():
        report.errors.append(f"Project path is not a directory: {p}")
        report.is_valid = False
        return report

    # 1. Entry point (main.py)
    main_py = p / "main.py"
    if not main_py.exists():
        report.errors.append(f"Missing project entry point: {main_py}")
    else:
        # Check that main.py can be parsed
        try:
            with open(main_py, "r", encoding="utf-8") as f:
                compile(f.read(), str(main_py), "exec")
        except Exception as e:
            report.errors.append(f"Syntax error in project entry point ({main_py}): {e}")

    # 2. Configuration file (config.py)
    config_py = p / "config.py"
    if not config_py.exists():
        report.warnings.append(f"Recommended project configuration file not found: {config_py}")

    # 3. Recommended structural directories
    for sub in ("modules", "world", "assets"):
        sub_dir = p / sub
        if not sub_dir.exists():
            report.warnings.append(f"Recommended directory '{sub}/' not found in {p.name}")

    report.is_valid = len(report.errors) == 0
    return report
