"""Automated Build Script for PyMordial Native C++ Accelerators.

Compiles performance-critical game modules (e.g. Pedestrian crowd simulation)
into high-speed shared libraries (.dll on Windows, .so on Linux).
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_compiler() -> tuple[str, str]:
    """Finds available C++ compiler (MinGW g++ or MSVC cl)."""
    # 1. Check MinGW MSYS2 / WinLibs g++
    candidates = [
        r"C:\msys64\mingw64\bin\g++.exe",
        r"C:\msys64\ucrt64\bin\g++.exe",
        shutil.which("g++"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return "gcc", str(c)

    # 2. Check MSVC
    cl_path = shutil.which("cl")
    if cl_path:
        return "msvc", cl_path

    return "none", ""


def build_pedestrian_accel() -> bool:
    """Compiles projects/shotgun_escape_the_heat/native/pedestrian_accel.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "projects" / "shotgun_escape_the_heat" / "native"
    src_file = src_dir / "pedestrian_accel.cpp"
    out_dll = src_dir / ("pedestrian_accel.dll" if sys.platform == "win32" else "libpedestrian_accel.so")

    if not src_file.is_file():
        print(f"[build_native] Source file not found: {src_file}")
        return False

    compiler_type, compiler_bin = find_compiler()
    if compiler_type == "none":
        print("[build_native] No C++ compiler found. Native acceleration will be skipped.")
        return False

    print(f"[build_native] Found {compiler_type} compiler: {compiler_bin}")
    print(f"[build_native] Compiling {src_file.name} -> {out_dll.name} ...")

    if compiler_type == "gcc":
        cmd = [
            compiler_bin,
            "-O3",
            "-shared",
            "-mavx2",
            "-ffast-math",
            "-fPIC",
            "-std=c++20",
            str(src_file),
            "-o",
            str(out_dll),
        ]
    else:  # msvc
        cmd = [
            compiler_bin,
            "/O2",
            "/LD",
            "/arch:AVX2",
            "/std:c++20",
            str(src_file),
            f"/Fe:{out_dll}",
        ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[build_native] Successfully compiled: {out_dll} ({out_dll.stat().st_size} bytes)")
        return True
    except subprocess.CalledProcessError as err:
        print(f"[build_native] Compilation failed:\n{err.stderr}")
        return False


def build_vehicle_native() -> bool:
    """Compiles projects/shotgun_escape_the_heat/native/vehicle_native.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "projects" / "shotgun_escape_the_heat" / "native"
    src_file = src_dir / "vehicle_native.cpp"
    out_dll = src_dir / ("vehicle_native.dll" if sys.platform == "win32" else "libvehicle_native.so")

    if not src_file.is_file():
        print(f"[build_native] Source file not found: {src_file}")
        return False

    compiler_type, compiler_bin = find_compiler()
    if compiler_type == "none":
        print("[build_native] No C++ compiler found. Native acceleration will be skipped.")
        return False

    print(f"[build_native] Compiling {src_file.name} -> {out_dll.name} ...")

    if compiler_type == "gcc":
        cmd = [
            compiler_bin,
            "-O3",
            "-shared",
            "-mavx2",
            "-ffast-math",
            "-fPIC",
            "-std=c++20",
            str(src_file),
            "-o",
            str(out_dll),
        ]
    else:  # msvc
        cmd = [
            compiler_bin,
            "/O2",
            "/LD",
            "/arch:AVX2",
            "/std:c++20",
            str(src_file),
            f"/Fe:{out_dll}",
        ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[build_native] Successfully compiled: {out_dll} ({out_dll.stat().st_size} bytes)")
        return True
    except subprocess.CalledProcessError as err:
        print(f"[build_native] Compilation failed:\n{err.stderr}")
        return False


if __name__ == "__main__":
    ok_ped = build_pedestrian_accel()
    ok_veh = build_vehicle_native()
    sys.exit(0 if (ok_ped and ok_veh) else 1)
