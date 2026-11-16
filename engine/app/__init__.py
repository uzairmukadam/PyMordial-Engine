"""PyMordial Project Application Framework.

Provides the foundational ProjectApp host, ProjectConfig settings, and ProjectModule interface.
"""

from engine.app.config import ProjectConfig
from engine.app.module import ProjectModule
from engine.app.module_contracts import GameplayModule, RenderFeatureModule, ToolModule
from engine.app.module_manifest import (
    MODULE_API_VERSION,
    ModuleDependency,
    ModuleManifest,
    ModuleRole,
    build_manifest,
    manifest_from_object,
    validate_manifest,
)
from engine.app.module_registry import (
    ModuleAttachError,
    ModuleHost,
    ModuleRecord,
    ModuleRegistry,
    ModuleResolutionError,
)
from engine.app.project_app import ProjectApp

__all__ = [
    "ProjectApp",
    "ProjectConfig",
    "ProjectModule",
    "GameplayModule",
    "RenderFeatureModule",
    "ToolModule",
    "MODULE_API_VERSION",
    "ModuleDependency",
    "ModuleManifest",
    "ModuleRole",
    "build_manifest",
    "manifest_from_object",
    "validate_manifest",
    "ModuleAttachError",
    "ModuleHost",
    "ModuleRecord",
    "ModuleRegistry",
    "ModuleResolutionError",
]
