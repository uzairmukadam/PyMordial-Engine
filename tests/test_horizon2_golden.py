"""Horizon 2 visual regression gate.

Captures a deterministic headless frame and compares it against the golden
reference recorded before the module/render-graph refactor. Any change to the
default sky, lighting, or resolve path must keep this within tolerance.

The capture runs in a **subprocess**. Creating and destroying several OpenGL
contexts inside one process measurably perturbs later frames on this driver
(observed MAE up to 3.2 after unrelated pipeline tests), so an in-process
capture would report false regressions depending on test ordering. A fresh
process makes the gate hermetic and order-independent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_capture(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tests.golden_capture", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def capture_result() -> subprocess.CompletedProcess[str]:
    return _run_capture("--check")


def test_default_frame_matches_golden(capture_result: subprocess.CompletedProcess[str]) -> None:
    """The default pipeline output must stay visually identical to the baseline."""
    assert capture_result.returncode == 0, (
        "Golden frame check failed:\n"
        f"{capture_result.stdout}\n{capture_result.stderr}\n"
        "Regenerate with 'python -m tests.golden_capture --write' only if the change is intentional."
    )
    assert "atmosphere_on" in capture_result.stdout


def _result_lines(output: str) -> list[str]:
    """Extracts the tool's own result lines, ignoring timestamped engine logs."""
    return [line for line in output.splitlines() if line.startswith(("[+]", "[!]"))]


def test_capture_is_deterministic() -> None:
    """Two independent capture processes must agree within driver noise."""
    first = _run_capture("--check")
    second = _run_capture("--check")
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert _result_lines(first.stdout) == _result_lines(second.stdout)
