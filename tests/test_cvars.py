"""Unit tests for PyMordial Engine CVar & Configuration System."""

from __future__ import annotations
from pathlib import Path
import pytest
from engine.config import CVar, CVarFlags, CVarRegistry


def test_cvar_typing_and_clamping():
    # Float with bounds
    cvar_fov = CVar("fov", 75.0, min_val=30.0, max_val=120.0)
    assert cvar_fov.value == 75.0

    # Valid set
    assert cvar_fov.set(90.0)
    assert cvar_fov.value == 90.0

    # Underflow clamp
    cvar_fov.set(10.0)
    assert cvar_fov.value == 30.0

    # Overflow clamp
    cvar_fov.set(150.0)
    assert cvar_fov.value == 120.0

    # String parsing
    cvar_fov.set_from_string("85.5")
    assert cvar_fov.value == 85.5

    # Bool CVar
    cvar_vsync = CVar("vsync", True)
    cvar_vsync.set("false")
    assert cvar_vsync.value is False
    cvar_vsync.set("1")
    assert cvar_vsync.value is True


def test_cvar_callbacks():
    cvar = CVar("speed", 10.0)
    history: list[tuple[float, float]] = []

    def on_change(old_v: float, new_v: float) -> None:
        history.append((old_v, new_v))

    cvar.add_callback(on_change)
    cvar.set(15.0)
    cvar.set(20.0)
    cvar.set(20.0)  # No change, callback should not fire

    assert len(history) == 2
    assert history[0] == (10.0, 15.0)
    assert history[1] == (15.0, 20.0)


def test_cvar_registry_and_toml_persistence(tmp_path: Path):
    reg = CVarRegistry(register_defaults=True)
    assert reg.get("r_vsync") is not None
    assert reg.get_value("r_fov") == 75.0

    # Modify some values
    reg.set_value("r_fov", 95.0)
    reg.set_value("r_vsync", 0)

    # Save to TOML
    cfg_file = tmp_path / "settings.toml"
    assert reg.save_to_file(cfg_file)
    assert cfg_file.exists()

    # Load into a fresh registry
    fresh_reg = CVarRegistry(register_defaults=True)
    assert fresh_reg.get_value("r_fov") == 75.0
    assert fresh_reg.load_from_file(cfg_file)
    assert fresh_reg.get_value("r_fov") == 95.0
    assert fresh_reg.get_value("r_vsync") == 0
