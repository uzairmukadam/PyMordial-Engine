"""Module Metadata Schema for PyMordial Engine.

Defines the declarative identity, role, versioning, and dependency metadata for
pluggable modules. This module is intentionally side-effect free: it never
imports or executes module code, so manifests can be validated before any
runtime service (window, GPU context, physics world) is allocated.

Metadata is expressed as typed Python dataclasses rather than a separate
manifest file format. A module declares its metadata by exposing a
``MODULE_MANIFEST`` attribute (or a ``get_module_manifest()`` callable) from its
package ``__init__``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from engine.app.project_validator import ValidationReport

# Current module API generation. Modules declaring a different major version are
# rejected; the minor component is informational for now.
MODULE_API_VERSION = "1.0"

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_API_PATTERN = re.compile(r"^\d+(\.\d+)?$")


class ModuleRole(str, Enum):
    """The extension contract a module implements."""

    GAMEPLAY = "gameplay"
    RENDER_FEATURE = "render_feature"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ModuleDependency:
    """A declared dependency on another module."""

    module_id: str
    version: str = "*"
    optional: bool = False

    def __str__(self) -> str:
        suffix = " (optional)" if self.optional else ""
        return f"{self.module_id} {self.version}{suffix}"


@dataclass(frozen=True, slots=True)
class ModuleManifest:
    """Declarative metadata describing a single module."""

    id: str
    version: str = "1.0.0"
    api_version: str = MODULE_API_VERSION
    role: ModuleRole = ModuleRole.GAMEPLAY
    entry: str = ""
    dependencies: tuple[ModuleDependency, ...] = ()
    capabilities: tuple[str, ...] = ()
    config_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def required_dependencies(self) -> tuple[ModuleDependency, ...]:
        return tuple(d for d in self.dependencies if not d.optional)

    @property
    def optional_dependencies(self) -> tuple[ModuleDependency, ...]:
        return tuple(d for d in self.dependencies if d.optional)

    def dependency_ids(self) -> tuple[str, ...]:
        return tuple(d.module_id for d in self.dependencies)


def _coerce_role(value: Any) -> ModuleRole | None:
    if isinstance(value, ModuleRole):
        return value
    if isinstance(value, str):
        try:
            return ModuleRole(value.strip().lower())
        except ValueError:
            return None
    return None


def _coerce_dependency(value: Any) -> ModuleDependency | None:
    """Accepts a ModuleDependency, a bare id string, or a mapping."""
    if isinstance(value, ModuleDependency):
        return value
    if isinstance(value, str):
        return ModuleDependency(module_id=value.strip())
    if isinstance(value, Mapping):
        module_id = value.get("id", value.get("module_id"))
        if not isinstance(module_id, str):
            return None
        version = value.get("version", "*")
        optional = value.get("optional", False)
        return ModuleDependency(
            module_id=module_id.strip(),
            version=str(version),
            optional=bool(optional),
        )
    return None


def validate_manifest(manifest: ModuleManifest) -> ValidationReport:
    """Validates a manifest's identity, versioning, role, and dependency declarations."""
    report = ValidationReport()

    if not isinstance(manifest, ModuleManifest):
        report.errors.append(f"Expected a ModuleManifest instance, got {type(manifest).__name__}.")
        report.is_valid = False
        return report

    # 1. Identity
    if not manifest.id:
        report.errors.append("Module manifest is missing a required 'id'.")
    elif not _ID_PATTERN.match(manifest.id):
        report.errors.append(
            f"Invalid module id '{manifest.id}': expected lowercase dotted segments "
            f"(e.g. 'pymordial.atmosphere')."
        )

    # 2. Versions
    if not _VERSION_PATTERN.match(manifest.version):
        report.errors.append(
            f"Invalid module version '{manifest.version}' for '{manifest.id}': expected MAJOR.MINOR.PATCH."
        )
    if not _API_PATTERN.match(manifest.api_version):
        report.errors.append(
            f"Invalid api_version '{manifest.api_version}' for '{manifest.id}': expected MAJOR or MAJOR.MINOR."
        )
    else:
        declared_major = manifest.api_version.split(".")[0]
        supported_major = MODULE_API_VERSION.split(".")[0]
        if declared_major != supported_major:
            report.errors.append(
                f"Unsupported api_version '{manifest.api_version}' for '{manifest.id}': "
                f"this engine provides module API {MODULE_API_VERSION}."
            )

    # 3. Role
    if not isinstance(manifest.role, ModuleRole):
        report.errors.append(
            f"Invalid role {manifest.role!r} for '{manifest.id}': expected one of "
            f"{[r.value for r in ModuleRole]}."
        )

    # 4. Entry point
    if not manifest.entry:
        report.errors.append(f"Module '{manifest.id}' is missing a required 'entry' (e.g. 'pkg.module:ClassName').")
    elif ":" not in manifest.entry:
        report.errors.append(
            f"Invalid entry '{manifest.entry}' for '{manifest.id}': expected 'module.path:ClassName'."
        )
    else:
        module_path, _, class_name = manifest.entry.partition(":")
        if not module_path or not class_name:
            report.errors.append(
                f"Invalid entry '{manifest.entry}' for '{manifest.id}': expected 'module.path:ClassName'."
            )

    # 5. Dependencies
    seen: set[str] = set()
    for dep in manifest.dependencies:
        if not isinstance(dep, ModuleDependency):
            report.errors.append(f"Module '{manifest.id}' declares a malformed dependency: {dep!r}.")
            continue
        if not _ID_PATTERN.match(dep.module_id):
            report.errors.append(
                f"Module '{manifest.id}' declares an invalid dependency id '{dep.module_id}'."
            )
        if dep.module_id == manifest.id:
            report.errors.append(f"Module '{manifest.id}' cannot depend on itself.")
        if dep.module_id in seen:
            report.errors.append(f"Module '{manifest.id}' declares duplicate dependency '{dep.module_id}'.")
        seen.add(dep.module_id)

    # 6. Capabilities
    for capability in manifest.capabilities:
        if not isinstance(capability, str) or not capability.strip():
            report.errors.append(f"Module '{manifest.id}' declares an invalid capability {capability!r}.")

    report.is_valid = len(report.errors) == 0
    return report


def build_manifest(
    module_id: str,
    *,
    version: str = "1.0.0",
    api_version: str = MODULE_API_VERSION,
    role: ModuleRole | str = ModuleRole.GAMEPLAY,
    entry: str = "",
    dependencies: Iterable[Any] = (),
    capabilities: Iterable[str] = (),
    config_schema: Mapping[str, Any] | None = None,
    description: str = "",
) -> ModuleManifest:
    """Builds a manifest from loosely-typed values, normalizing roles and dependencies.

    Raises:
        ValueError: if the role or a dependency cannot be interpreted.
    """
    resolved_role = _coerce_role(role)
    if resolved_role is None:
        raise ValueError(f"Unknown module role {role!r}; expected one of {[r.value for r in ModuleRole]}.")

    resolved_deps: list[ModuleDependency] = []
    for raw in dependencies:
        dep = _coerce_dependency(raw)
        if dep is None:
            raise ValueError(f"Malformed dependency declaration: {raw!r}.")
        resolved_deps.append(dep)

    return ModuleManifest(
        id=module_id,
        version=version,
        api_version=api_version,
        role=resolved_role,
        entry=entry,
        dependencies=tuple(resolved_deps),
        capabilities=tuple(capabilities),
        config_schema=dict(config_schema or {}),
        description=description,
    )


def manifest_from_object(obj: Any) -> ModuleManifest | None:
    """Extracts a manifest from a module object or class, if it declares one.

    Accepts a ``MODULE_MANIFEST`` attribute or a ``get_module_manifest()`` callable.
    Returns ``None`` when the object declares no metadata (legacy module).
    """
    getter = getattr(obj, "get_module_manifest", None)
    if callable(getter):
        manifest = getter()
        if isinstance(manifest, ModuleManifest):
            return manifest
        if isinstance(manifest, Mapping):
            return _manifest_from_mapping(manifest)

    declared = getattr(obj, "MODULE_MANIFEST", None)
    if isinstance(declared, ModuleManifest):
        return declared
    if isinstance(declared, Mapping):
        return _manifest_from_mapping(declared)
    return None


def _manifest_from_mapping(data: Mapping[str, Any]) -> ModuleManifest:
    """Builds a manifest from a mapping, raising ValueError on malformed input."""
    module_id = data.get("id")
    if not isinstance(module_id, str):
        raise ValueError("Module manifest mapping requires a string 'id'.")
    return build_manifest(
        module_id,
        version=str(data.get("version", "1.0.0")),
        api_version=str(data.get("api_version", MODULE_API_VERSION)),
        role=data.get("role", ModuleRole.GAMEPLAY),
        entry=str(data.get("entry", "")),
        dependencies=data.get("dependencies", ()),
        capabilities=data.get("capabilities", ()),
        config_schema=data.get("config_schema"),
        description=str(data.get("description", "")),
    )


__all__ = [
    "MODULE_API_VERSION",
    "ModuleDependency",
    "ModuleManifest",
    "ModuleRole",
    "build_manifest",
    "manifest_from_object",
    "validate_manifest",
]
