"""Automated regression tests for PyMordial Engine reference examples."""

import subprocess
import sys
from pathlib import Path
import pytest


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize(
    "example_script",
    [
        "01_modern_ui_and_menu.py",
        "02_player_physics_and_cubes.py",
        "03_full_graphics_and_display_menu.py",
    ],
)
def test_example_headless_execution(example_script: str) -> None:
    """Verifies that each reference example boots, executes, and cleanly shuts down in headless mode."""
    script_path = EXAMPLES_DIR / example_script
    assert script_path.exists(), f"Example script not found: {script_path}"

    cmd = [sys.executable, str(script_path), "--headless", "--frames", "30"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"Example {example_script} failed with return code {proc.returncode}!\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
