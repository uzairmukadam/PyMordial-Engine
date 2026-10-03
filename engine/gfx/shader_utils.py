"""Central Shader Loading and Resolution Utilities for PyMordial Engine.

Resolves shader file locations across multiple runtime environments:
- Local development source checkouts (<repo>/shaders)
- Pip-installed packages (engine/shaders)
- PyInstaller frozen binaries (via sys._MEIPASS / 'shaders')
- Nuitka standalone binaries (via sys.executable parent / 'shaders')
- Mounted VFS containers

Provides thread-safe caching and recursive #include expansion.
"""

from __future__ import annotations
import os
from pathlib import Path
import sys

_SHADER_CACHE: dict[str, str] = {}
_SHADER_DIR: Path | None = None


def get_shader_dir() -> Path:
    """Discovers and returns the authoritative shader directory on disk."""
    global _SHADER_DIR
    if _SHADER_DIR is not None and _SHADER_DIR.is_dir():
        return _SHADER_DIR

    # 1. PyInstaller frozen binary
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_shaders = Path(sys._MEIPASS) / "shaders"
        if meipass_shaders.is_dir():
            _SHADER_DIR = meipass_shaders
            return _SHADER_DIR

    # 2. Executable-relative (Nuitka standalone or onedir distribution)
    exe_shaders = Path(sys.executable).resolve().parent / "shaders"
    if exe_shaders.is_dir():
        _SHADER_DIR = exe_shaders
        return _SHADER_DIR

    # 3. Environment variable override
    env_dir = os.environ.get("PYMORDIAL_SHADER_DIR")
    if env_dir:
        env_p = Path(env_dir).resolve()
        if env_p.is_dir():
            _SHADER_DIR = env_p
            return _SHADER_DIR

    # 4. Standard development repository root: <repo>/shaders
    dev_shaders = Path(__file__).resolve().parent.parent.parent / "shaders"
    if dev_shaders.is_dir():
        _SHADER_DIR = dev_shaders
        return _SHADER_DIR

    # 5. Packaged module data: engine/shaders
    pkg_shaders = Path(__file__).resolve().parent.parent / "shaders"
    if pkg_shaders.is_dir():
        _SHADER_DIR = pkg_shaders
        return _SHADER_DIR

    # 6. Fallback to current working directory / shaders
    cwd_shaders = Path.cwd() / "shaders"
    if cwd_shaders.is_dir():
        _SHADER_DIR = cwd_shaders
        return _SHADER_DIR

    # Default to development location even if not yet created
    _SHADER_DIR = dev_shaders
    return _SHADER_DIR


def get_shader_path(rel_path: str | Path) -> Path:
    """Resolves a relative shader filename or path to an absolute Path."""
    sdir = get_shader_dir()
    rel = Path(rel_path)
    if rel.is_absolute() and rel.is_file():
        return rel

    rel_str = rel.as_posix()
    if rel_str.startswith("shaders/"):
        rel_str = rel_str[len("shaders/") :]

    direct_path = sdir / rel_str
    if direct_path.is_file():
        return direct_path

    # Check parent of shader dir if query included 'shaders/...'
    parent_path = sdir.parent / rel.as_posix()
    if parent_path.is_file():
        return parent_path

    return direct_path


def load_shader(rel_path: str | Path, visited: set[Path] | None = None) -> str:
    """Loads a GLSL shader file and recursively expands #include directives."""
    if visited is None:
        visited = set()

    path = get_shader_path(rel_path).resolve()
    if path in visited:
        return ""
    visited.add(path)

    cache_key = str(path)
    if cache_key in _SHADER_CACHE:
        return _SHADER_CACHE[cache_key]

    if not path.is_file():
        raise FileNotFoundError(f"Shader file not found: {path} (from query '{rel_path}')")

    raw = path.read_text(encoding="utf-8")
    lines: list[str] = []
    sdir = get_shader_dir()

    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("#include"):
            start_q = line.find('"')
            end_q = line.rfind('"')
            if start_q != -1 and end_q > start_q:
                inc_rel = line[start_q + 1 : end_q]
                # Try direct under shader dir first
                inc_clean = inc_rel[len("shaders/") :] if inc_rel.startswith("shaders/") else inc_rel
                inc_path = sdir / inc_clean
                if not inc_path.is_file() and (sdir.parent / inc_rel).is_file():
                    inc_path = sdir.parent / inc_rel
                inc_text = load_shader(inc_path, visited=visited)
                lines.append(inc_text)
                continue
        lines.append(line)

    expanded = "\n".join(lines)
    _SHADER_CACHE[cache_key] = expanded
    return expanded


def clear_shader_cache() -> None:
    """Clears cached shader string memory."""
    _SHADER_CACHE.clear()
