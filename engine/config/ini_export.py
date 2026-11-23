"""INI export and import for engine and project configuration.

The engine's debug tweaks are a flat dictionary of primitives. This module turns
that dictionary into a human-editable ``.ini`` file and back again, so a project
can offer "save my settings" without inventing its own serialization.

Design notes:

* **Sections are derived from key prefixes**, not stored in the data. A flat
  dictionary is the source of truth; the section layout is presentation only.
  That keeps ``EngineTweaks.to_dict()`` unchanged and means a new tweak lands in
  a sensible section automatically.
* **Values round-trip through text.** Booleans become ``true``/``false``, tuples
  become comma-separated lists, and everything else is written with ``repr``-free
  formatting so the file stays readable and diffable.
* **Reading is permissive.** Unknown keys are ignored rather than raising, so a
  settings file written by an older build still loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Ordered (section, prefixes) table. The first matching prefix wins, so more
# specific prefixes must come before broader ones.
SECTION_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shadows", ("shadow_", "csm_", "pcf_", "pcss_")),
    ("lighting", ("ao_", "gi_", "ssgi_", "rmgi_", "ssr_", "ibl_", "spot_", "point_")),
    ("antialiasing", ("aa_", "taa_", "fxaa_", "smaa_")),
    ("optics", ("dof_", "motion_blur_", "lens_flare_", "solar_flare_", "chromatic_", "vignette_", "film_grain_")),
    ("postprocess", ("bloom_", "tonemap_", "exposure")),
    ("fog", ("fog_", "volumetric_")),
    ("microgeometry", ("pom_", "ssdm_", "tess_", "disp_")),
    ("particles", ("particle_", "particles_")),
    (
        "atmosphere",
        (
            "sun_",
            "atmo_",
            "sky_",
            "cloud_",
            "rayleigh_",
            "mie_",
            "turbidity",
            "star_",
            "nebula_",
            "moon_",
            "space_",
            "night_",
            "ground_",
            "planet_",
            "ozone_",
            "h_r",
            "h_m",
            "time_of_day",
            "day_speed",
            "azimuth_",
            "latitude_",
            "declination_",
        ),
    ),
)

DEFAULT_SECTION = "general"


def section_for_key(key: str) -> str:
    """Returns the INI section a configuration key belongs to."""
    lowered = key.lower()
    for section, prefixes in SECTION_PREFIXES:
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return section
    return DEFAULT_SECTION


def format_value(value: Any) -> str:
    """Formats a Python value as an INI scalar or comma-separated list."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return ", ".join(format_value(item) for item in value)
    if isinstance(value, float):
        # ``repr`` keeps full precision without the trailing noise of ``str`` on
        # values like 0.1 + 0.2, and never emits an exponent for normal ranges.
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    text = str(value)
    # Quote strings that would otherwise be read back as a different type. The
    # obvious case is a mode name like "OFF", which collides with the boolean
    # spelling; numeric-looking strings collide the same way.
    if _would_reparse_as_non_string(text):
        return f'"{text}"'
    return text


def _would_reparse_as_non_string(text: str) -> bool:
    """True when ``parse_value`` would not return this text unchanged."""
    if not text:
        return False
    if text.lower() in ("true", "false", "yes", "no", "on", "off"):
        return True
    if "," in text:
        return True
    try:
        int(text)
        return True
    except ValueError:
        pass
    try:
        float(text)
        return True
    except ValueError:
        pass
    return False


def parse_value(text: str) -> Any:
    """Parses an INI scalar back into a Python value.

    Lists are returned as tuples so they match the tuple-typed fields on
    ``RenderConfig`` and ``EngineTweaks``. Quoted values are always returned as
    strings, which is how a mode name like ``"OFF"`` survives a round trip.
    """
    raw = text.strip()
    if not raw:
        return ""
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if "," in raw:
        return tuple(parse_value(part) for part in raw.split(","))
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def write_ini(path: str | Path, data: dict[str, Any], *, header: str | None = None) -> bool:
    """Writes a flat configuration dictionary to an INI file.

    Keys are grouped into sections by :func:`section_for_key` and sorted within
    each section so the file is stable across saves. Returns ``False`` on any
    filesystem or formatting error rather than raising, because this is called
    from a UI button.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list[tuple[str, Any]]] = {}
        for key, value in data.items():
            grouped.setdefault(section_for_key(key), []).append((key, value))

        lines: list[str] = []
        if header:
            for line in header.splitlines():
                lines.append(f"; {line}" if line.strip() else ";")
            lines.append("")

        # Emit the default section first, then the rest alphabetically, so the
        # file reads top-down from general settings to subsystem detail.
        ordered_sections = sorted(grouped, key=lambda name: (name != DEFAULT_SECTION, name))
        for section in ordered_sections:
            lines.append(f"[{section}]")
            for key, value in sorted(grouped[section], key=lambda item: item[0]):
                lines.append(f"{key} = {format_value(value)}")
            lines.append("")

        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_ini(path: str | Path) -> dict[str, Any]:
    """Reads an INI file into a flat dictionary of parsed values.

    Section headers are ignored: the returned dictionary is flat, matching the
    shape ``EngineTweaks.from_dict`` expects. Missing files return an empty
    dictionary.
    """
    target = Path(path)
    if not target.is_file():
        return {}

    try:
        content = target.read_text(encoding="utf-8")
    except OSError:
        return {}

    result: dict[str, Any] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith((";", "#", "[")):
            continue
        # Strip trailing inline comments, but only when preceded by whitespace so
        # a value that legitimately contains ';' is not truncated.
        if " ;" in line:
            line = line.split(" ;", 1)[0].rstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            result[key] = parse_value(value)
    return result
