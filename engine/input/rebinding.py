"""Input Rebinding Persistence and Serialization for PyMordial Engine."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from engine.logging import log_info, log_warn, LogChannel


def export_bindings_to_dict(actions_dict: dict[str, Any]) -> dict[str, Any]:
    """Extracts raw key, mouse, and pad mappings from action bindings into a serializable dict."""
    data: dict[str, Any] = {}
    for name, action in actions_dict.items():
        data[name] = {
            "keys": [int(k) for k in getattr(action, "keys", [])],
            "mouse_buttons": [int(m) for m in getattr(action, "mouse_buttons", [])],
            "pad_buttons": [int(p) for p in getattr(action, "pad_buttons", [])],
        }
    return data


def save_bindings_to_file(actions_dict: dict[str, Any], filepath: str | Path) -> bool:
    """Serializes action bindings to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = export_bindings_to_dict(actions_dict)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_info(LogChannel.INPUT, f"Saved input bindings to {path}")
        return True
    except Exception as e:
        log_warn(LogChannel.INPUT, f"Failed to save input bindings: {e}")
        return False


def load_bindings_from_file(filepath: str | Path) -> dict[str, Any] | None:
    """Loads action bindings from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as e:
        log_warn(LogChannel.INPUT, f"Failed to read input bindings from {path}: {e}")
        return None
