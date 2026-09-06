"""Global Console Variable (CVar) Registry & TOML Configuration Engine."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from engine.config.cvar import CVar, CVarFlags
from engine.logging import log_info, log_warn, LogChannel


class CVarRegistry:
    """Central repository of engine and game configuration variables."""

    __slots__ = ("_cvars",)

    def __init__(self, register_defaults: bool = True) -> None:
        self._cvars: dict[str, CVar[Any]] = {}
        if register_defaults:
            self._register_default_cvars()

    def register(self, cvar: CVar[Any]) -> CVar[Any]:
        """Registers a CVar in the repository."""
        key = cvar.name.lower()
        if key in self._cvars:
            log_warn(LogChannel.CONFIG, f"Overwriting registered CVar: {key}")
        self._cvars[key] = cvar
        return cvar

    def get(self, name: str) -> CVar[Any] | None:
        """Retrieves a registered CVar by name."""
        return self._cvars.get(name.lower(), None)

    def get_value(self, name: str, fallback: Any = None) -> Any:
        """Retrieves the current value of a CVar, or a fallback if unregistered."""
        cvar = self.get(name)
        return cvar.value if cvar is not None else fallback

    def set_value(self, name: str, value: Any) -> bool:
        """Assigns a new value to a named CVar."""
        cvar = self.get(name)
        if cvar is None:
            log_warn(LogChannel.CONFIG, f"Attempted to set unregistered CVar: '{name}'")
            return False
        return cvar.set(value)

    def all_cvars(self) -> dict[str, CVar[Any]]:
        """Returns all registered CVars."""
        return self._cvars.copy()

    def _register_default_cvars(self) -> None:
        """Initializes core AAA engine variables."""
        # Graphics & Display
        self.register(CVar("r_vsync", 1, "VSync mode: 0=Off, 1=On (FIFO), -1=Adaptive", min_val=-1, max_val=1))
        self.register(CVar("r_fullscreen", 0, "Display mode: 0=Windowed, 1=Borderless, 2=Exclusive", min_val=0, max_val=2))
        self.register(CVar("r_width", 1280, "Window width in pixels", min_val=640, max_val=7680))
        self.register(CVar("r_height", 720, "Window height in pixels", min_val=480, max_val=4320))
        self.register(CVar("r_fov", 75.0, "Vertical field of view in degrees", min_val=30.0, max_val=120.0))
        self.register(CVar("r_csm_cascades", 4, "Cascaded shadow map cascade count", min_val=1, max_val=4))
        self.register(CVar("r_sscs_enabled", True, "Screen-space contact shadows toggle"))
        self.register(CVar("r_pcf_samples", 16, "Poisson shadow filtering samples (4, 8, 16)", min_val=4, max_val=16))
        self.register(CVar("r_tonemap", "aces", "Active tonemapping operator ('aces', 'agx', 'reinhard')"))
        self.register(CVar("r_bloom", True, "HDR bloom pyramid pass toggle"))

        # Time & Simulation
        self.register(CVar("t_timescale", 1.0, "Global time dilation factor (0.1 = slow-mo, 1.0 = normal)", min_val=0.0, max_val=10.0))
        self.register(CVar("t_max_dt", 0.20, "Maximum frame time accumulator clamp in seconds", min_val=0.05, max_val=1.0))
        self.register(CVar("t_max_fps", 0, "Frame rate limiter target (0 = unlimited)", min_val=0, max_val=360))

        # Audio
        self.register(CVar("snd_master_volume", 1.0, "Global audio master volume", min_val=0.0, max_val=1.0))
        self.register(CVar("snd_sfx_volume", 1.0, "Sound effects volume", min_val=0.0, max_val=1.0))
        self.register(CVar("snd_music_volume", 0.8, "Music bus volume", min_val=0.0, max_val=1.0))
        self.register(CVar("snd_voice_volume", 1.0, "Dialogue and voice volume", min_val=0.0, max_val=1.0))
        self.register(CVar("snd_ui_volume", 1.0, "UI sound effects volume", min_val=0.0, max_val=1.0))
        self.register(CVar("snd_spatial_enabled", True, "3D spatial audio listener calculations toggle"))

        # Input
        self.register(CVar("in_mouse_sensitivity", 0.25, "Hardware mouse orbit look sensitivity", min_val=0.01, max_val=5.0))
        self.register(CVar("in_invert_y", False, "Invert camera vertical pitch input"))
        self.register(CVar("in_stick_deadzone", 0.15, "Gamepad analog stick circular deadzone", min_val=0.0, max_val=0.9))

    def save_to_file(self, filepath: str | Path = "config/engine_settings.toml") -> bool:
        """Serializes all ARCHIVE CVars to a readable TOML configuration file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# PyMordial Engine Configuration",
            "# Auto-generated settings file",
            "",
            "[graphics]",
        ]
        # Group variables into sections
        gfx_vars = [c for c in self._cvars.values() if c.name.startswith("r_") and (CVarFlags.ARCHIVE in c.flags)]
        time_vars = [c for c in self._cvars.values() if c.name.startswith("t_") and (CVarFlags.ARCHIVE in c.flags)]
        snd_vars = [c for c in self._cvars.values() if c.name.startswith("snd_") and (CVarFlags.ARCHIVE in c.flags)]
        in_vars = [c for c in self._cvars.values() if c.name.startswith("in_") and (CVarFlags.ARCHIVE in c.flags)]
        other_vars = [
            c for c in self._cvars.values()
            if not any(c.name.startswith(p) for p in ("r_", "t_", "snd_", "in_"))
            and (CVarFlags.ARCHIVE in c.flags)
        ]

        def format_val(val: Any) -> str:
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, str):
                return f'"{val}"'
            return str(val)

        for c in gfx_vars:
            lines.append(f"{c.name} = {format_val(c.value)}  # {c.description}")

        lines.extend(["", "[time]"])
        for c in time_vars:
            lines.append(f"{c.name} = {format_val(c.value)}  # {c.description}")

        lines.extend(["", "[audio]"])
        for c in snd_vars:
            lines.append(f"{c.name} = {format_val(c.value)}  # {c.description}")

        lines.extend(["", "[input]"])
        for c in in_vars:
            lines.append(f"{c.name} = {format_val(c.value)}  # {c.description}")

        if other_vars:
            lines.extend(["", "[general]"])
            for c in other_vars:
                lines.append(f"{c.name} = {format_val(c.value)}  # {c.description}")

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log_info(LogChannel.CONFIG, f"Saved engine settings to {path}")
            return True
        except Exception as e:
            log_warn(LogChannel.CONFIG, f"Failed to save settings: {e}")
            return False

    def load_from_file(self, filepath: str | Path = "config/engine_settings.toml") -> bool:
        """Parses and applies CVar values from a TOML configuration file."""
        path = Path(filepath)
        if not path.exists():
            return False

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            log_warn(LogChannel.CONFIG, f"Failed to read settings file {path}: {e}")
            return False

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue

            # Strip inline comment
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if "=" not in line:
                continue

            key_part, val_part = line.split("=", 1)
            name = key_part.strip().lower()
            val_str = val_part.strip().strip('"').strip("'")

            cvar = self.get(name)
            if cvar is not None:
                cvar.set_from_string(val_str)

        log_info(LogChannel.CONFIG, f"Loaded settings from {path}")
        return True


# Global default registry
_GLOBAL_CVAR_REGISTRY: CVarRegistry = CVarRegistry()


def get_cvar_registry() -> CVarRegistry:
    """Retrieves the global default CVar registry."""
    return _GLOBAL_CVAR_REGISTRY


def get_cvar(name: str) -> CVar[Any] | None:
    return _GLOBAL_CVAR_REGISTRY.get(name)


def get_cvar_value(name: str, fallback: Any = None) -> Any:
    return _GLOBAL_CVAR_REGISTRY.get_value(name, fallback)


def set_cvar_value(name: str, val: Any) -> bool:
    return _GLOBAL_CVAR_REGISTRY.set_value(name, val)
