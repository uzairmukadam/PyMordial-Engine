"""Prefab Manager and Template Instantiator.

Loads, caches, and instances declarative prefabs with property and transform overrides.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from engine.world.mapping.schema import EntityDef


class PrefabManager:
    """Manages reusable compound entity templates (prefabs)."""

    __slots__ = ("_cache", "_search_paths")

    def __init__(self, search_paths: list[str | Path] | None = None) -> None:
        self._cache: dict[str, EntityDef] = {}
        self._search_paths: list[Path] = [Path(p) for p in (search_paths or [Path("assets/prefabs"), Path("prefabs")])]

    def register_template(self, name: str, template: EntityDef) -> None:
        """Registers an in-memory prefab template."""
        self._cache[name] = template

    def get_template(self, identifier: str) -> EntityDef:
        """Retrieves a cached prefab template or loads it from disk."""
        if identifier in self._cache:
            return self._cache[identifier]

        # Attempt to find file
        candidate_path = Path(identifier)
        if not candidate_path.is_file():
            for base_dir in self._search_paths:
                p = base_dir / identifier
                if p.is_file():
                    candidate_path = p
                    break
                if not identifier.endswith(".json"):
                    p_json = base_dir / f"{identifier}.json"
                    if p_json.is_file():
                        candidate_path = p_json
                        break

        if not candidate_path.is_file():
            raise FileNotFoundError(f"Prefab template not found: '{identifier}'")

        with candidate_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Prefab JSON must be an object: {candidate_path}")

        # If wrapped under 'root' or 'entity', extract root
        root_data = data.get("root") or data.get("entity") or data
        template = EntityDef.from_dict(root_data)
        self._cache[identifier] = template
        return template

    def instantiate(
        self,
        identifier: str,
        instance_id: str | None = None,
        property_overrides: dict[str, Any] | None = None,
    ) -> EntityDef:
        """Creates a fresh, isolated EntityDef from a template with applied overrides."""
        template = self.get_template(identifier)

        # Deep copy template definition so mutations don't alter cached master
        instance = copy.deepcopy(template)

        if instance_id is not None:
            instance.id = instance_id

        if property_overrides:
            instance.properties.update(property_overrides)
            # Propagate matching overrides to children if needed
            for child in instance.children:
                child_override_key = f"{child.id}."
                for k, v in property_overrides.items():
                    if k.startswith(child_override_key):
                        child_prop = k[len(child_override_key):]
                        child.properties[child_prop] = v

        return instance
