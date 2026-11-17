"""Regenerates and verifies the Horizon 2 golden reference frames.

Usage:
    python -m tests.golden_capture --write    # record baselines
    python -m tests.golden_capture --check    # verify against baselines
"""

from __future__ import annotations

import argparse
import sys

from tests.golden import (
    CAPTURE_VARIANTS,
    DEFAULT_TOLERANCE,
    capture_variant,
    golden_path,
    load_golden,
    mean_absolute_error,
    save_golden,
)

CAPTURES = tuple(CAPTURE_VARIANTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden reference frame tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Write golden frames to tests/baseline/")
    group.add_argument("--check", action="store_true", help="Verify captures against golden frames")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="Restrict to one or more named variants (repeatable).",
    )
    args = parser.parse_args()

    selected = tuple(args.only) if args.only else CAPTURES
    unknown = [name for name in selected if name not in CAPTURE_VARIANTS]
    if unknown:
        print(f"[!] Unknown variant(s): {', '.join(unknown)}")
        print(f"    Known: {', '.join(CAPTURES)}")
        return 2

    if args.write:
        for name in selected:
            frame = capture_variant(name)
            path = save_golden(name, frame)
            print(f"[+] Wrote {name}: {path} ({frame.shape})")
        return 0

    failures = 0
    for name in selected:
        if not golden_path(name).exists():
            print(f"[!] Missing golden frame '{name}' at {golden_path(name)}")
            failures += 1
            continue
        frame = capture_variant(name)
        error = mean_absolute_error(frame, load_golden(name))
        if error <= DEFAULT_TOLERANCE:
            print(f"[+] {name}: MAE={error:.3f} (tolerance {DEFAULT_TOLERANCE})")
        else:
            print(f"[!] {name}: MAE={error:.3f} exceeds tolerance {DEFAULT_TOLERANCE}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
