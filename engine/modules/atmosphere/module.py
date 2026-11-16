"""Atmosphere & Sky Module Declaration.

Declares the module's identity, role, dependencies, and capabilities so the
module host can resolve and validate it like any other module.
"""

from __future__ import annotations

from engine.app.module_manifest import ModuleRole, build_manifest


class AtmosphereModule:
    """Engine feature module providing sky, atmosphere, and cloud rendering."""

    MODULE_MANIFEST = build_manifest(
        "pymordial.atmosphere",
        version="1.0.0",
        role=ModuleRole.RENDER_FEATURE,
        entry="engine.modules.atmosphere:AtmosphereEnvironment",
        capabilities=("render.sky", "lighting.environment", "render.clouds"),
        description="Physically based sky, atmospheric scattering, and cloud deck.",
    )

    @staticmethod
    def get_module_manifest():
        return AtmosphereModule.MODULE_MANIFEST


__all__ = ["AtmosphereModule"]
