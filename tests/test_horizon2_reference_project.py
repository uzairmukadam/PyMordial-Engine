"""End-to-end tests for the Horizon 2 reference project.

Proves that a project can be discovered, validated, and launched without editing
engine source, and that the atmosphere engine module can be toggled purely from
project configuration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "projects" / "reference_game"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not PROJECT_DIR.is_dir(), reason="reference project not present")
class TestReferenceProject:
    def test_project_validates(self) -> None:
        result = _run("-m", "engine.tools.project", "validate", "--project", "reference_game")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VALID" in result.stdout

    def test_project_runs_headless(self) -> None:
        result = _run(
            "-m",
            "engine.tools.project",
            "run",
            "--project",
            "reference_game",
            "--headless",
            "--frames",
            "3",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_project_runs_with_atmosphere_disabled(self) -> None:
        result = _run(str(PROJECT_DIR / "main.py"), "--headless", "--frames", "3", "--no-atmosphere")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_config_declares_engine_module_settings(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from projects.reference_game.config import get_project_config

            cfg = get_project_config()
            assert cfg.atmosphere_enabled is True
            assert cfg.atmosphere_settings["preset"] == "EARTH_DAY"
        finally:
            sys.path.remove(str(REPO_ROOT))

    def test_gameplay_module_declares_manifest(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from engine.app.module_manifest import ModuleRole, validate_manifest
            from projects.reference_game.modules.spinner import SpinnerModule

            manifest = SpinnerModule.MODULE_MANIFEST
            assert validate_manifest(manifest).is_valid is True
            assert manifest.role is ModuleRole.GAMEPLAY
            assert manifest.id == "reference_game.spinner"
        finally:
            sys.path.remove(str(REPO_ROOT))
