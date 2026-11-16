"""Regenerates and verifies the Horizon 2 golden reference frames.

Usage:
    python -m tests.golden_capture --write    # record baselines
    python -m tests.golden_capture --check    # verify against baselines
"""

from __future__ import annotations

import argparse
import sys

from tests.golden import (
    DEFAULT_TOLERANCE,
    capture_scene,
    golden_path,
    load_golden,
    mean_absolute_error,
    save_golden,
)

CAPTURES = ("atmosphere_on", "atmosphere_off")


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden reference frame tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Write golden frames to tests/baseline/")
    group.add_argument("--check", action="store_true", help="Verify captures against golden frames")
    args = parser.parse_args()

    if args.write:
        for name in CAPTURES:
            frame = capture_scene()
            path = save_golden(name, frame)
            print(f"[+] Wrote {name}: {path} ({frame.shape})")
        return 0

    failures = 0
    for name in CAPTURES:
        if not golden_path(name).exists():
            print(f"[!] Missing golden frame '{name}' at {golden_path(name)}")
            failures += 1
            continue
        frame = capture_scene()
        error = mean_absolute_error(frame, load_golden(name))
        if error <= DEFAULT_TOLERANCE:
            print(f"[+] {name}: MAE={error:.3f} (tolerance {DEFAULT_TOLERANCE})")
        else:
            print(f"[!] {name}: MAE={error:.3f} exceeds tolerance {DEFAULT_TOLERANCE}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

