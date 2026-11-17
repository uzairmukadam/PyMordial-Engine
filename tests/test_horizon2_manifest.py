"""Tests for the Horizon 2 module metadata schema (H2.1)."""

from __future__ import annotations

import pytest

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


class TestManifestValidation:
    def test_valid_manifest_passes(self) -> None:
        manifest = build_manifest(
            "pymordial.atmosphere",
            version="1.2.0",
            role=ModuleRole.RENDER_FEATURE,
            entry="engine.modules.atmosphere:AtmosphereModule",
            dependencies=["pymordial.renderer"],
            capabilities=["render.sky", "lighting.environment"],
        )
        report = validate_manifest(manifest)
        assert report.is_valid is True
        assert report.errors == []

    def test_missing_id_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="", entry="a.b:C"))
        assert report.is_valid is False
        assert any("'id'" in e for e in report.errors)

    def test_malformed_id_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="Bad-Id", entry="a.b:C"))
        assert report.is_valid is False
        assert any("Invalid module id" in e for e in report.errors)

    def test_malformed_version_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", version="1.0", entry="a.b:C"))
        assert report.is_valid is False
        assert any("Invalid module version" in e for e in report.errors)

    def test_unsupported_api_major_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", api_version="99.0", entry="a.b:C"))
        assert report.is_valid is False
        assert any("Unsupported api_version" in e for e in report.errors)

    def test_current_api_version_is_accepted(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", api_version=MODULE_API_VERSION, entry="a.b:C"))
        assert report.is_valid is True

    def test_missing_entry_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", entry=""))
        assert report.is_valid is False
        assert any("'entry'" in e for e in report.errors)

    def test_entry_without_class_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", entry="engine.modules.atmosphere"))
        assert report.is_valid is False
        assert any("expected 'module.path:ClassName'" in e for e in report.errors)

    def test_self_dependency_is_rejected(self) -> None:
        manifest = ModuleManifest(
            id="good.id",
            entry="a.b:C",
            dependencies=(ModuleDependency("good.id"),),
        )
        report = validate_manifest(manifest)
        assert report.is_valid is False
        assert any("cannot depend on itself" in e for e in report.errors)

    def test_duplicate_dependency_is_rejected(self) -> None:
        manifest = ModuleManifest(
            id="good.id",
            entry="a.b:C",
            dependencies=(ModuleDependency("other.mod"), ModuleDependency("other.mod")),
        )
        report = validate_manifest(manifest)
        assert report.is_valid is False
        assert any("duplicate dependency" in e for e in report.errors)

    def test_invalid_role_is_rejected(self) -> None:
        report = validate_manifest(ModuleManifest(id="good.id", entry="a.b:C", role="nonsense"))  # type: ignore[arg-type]
        assert report.is_valid is False
        assert any("Invalid role" in e for e in report.errors)

    def test_non_manifest_input_is_rejected(self) -> None:
        report = validate_manifest("not a manifest")  # type: ignore[arg-type]
        assert report.is_valid is False
        assert any("Expected a ModuleManifest" in e for e in report.errors)


class TestManifestConstruction:
    def test_build_manifest_normalizes_role_string(self) -> None:
        manifest = build_manifest("a.b", role="render_feature", entry="a.b:C")
        assert manifest.role is ModuleRole.RENDER_FEATURE

    def test_build_manifest_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="Unknown module role"):
            build_manifest("a.b", role="wizard", entry="a.b:C")

    def test_build_manifest_accepts_mapping_dependencies(self) -> None:
        manifest = build_manifest(
            "a.b",
            entry="a.b:C",
            dependencies=[{"id": "dep.one", "version": "^1.0", "optional": True}],
        )
        assert len(manifest.dependencies) == 1
        dep = manifest.dependencies[0]
        assert dep.module_id == "dep.one"
        assert dep.version == "^1.0"
        assert dep.optional is True

    def test_required_and_optional_split(self) -> None:
        manifest = build_manifest(
            "a.b",
            entry="a.b:C",
            dependencies=["req.one", {"id": "opt.one", "optional": True}],
        )
        assert [d.module_id for d in manifest.required_dependencies] == ["req.one"]
        assert [d.module_id for d in manifest.optional_dependencies] == ["opt.one"]

    def test_build_manifest_rejects_malformed_dependency(self) -> None:
        with pytest.raises(ValueError, match="Malformed dependency"):
            build_manifest("a.b", entry="a.b:C", dependencies=[42])


class TestManifestDiscovery:
    def test_manifest_from_class_attribute(self) -> None:
        class WithManifest:
            MODULE_MANIFEST = ModuleManifest(id="a.b", entry="a.b:C")

        assert manifest_from_object(WithManifest).id == "a.b"

    def test_manifest_from_getter(self) -> None:
        class WithGetter:
            @staticmethod
            def get_module_manifest() -> ModuleManifest:
                return ModuleManifest(id="c.d", entry="c.d:E")

        assert manifest_from_object(WithGetter).id == "c.d"

    def test_manifest_from_mapping_attribute(self) -> None:
        class WithMapping:
            MODULE_MANIFEST = {"id": "e.f", "entry": "e.f:G", "role": "tool"}

        manifest = manifest_from_object(WithMapping)
        assert manifest.id == "e.f"
        assert manifest.role is ModuleRole.TOOL

    def test_legacy_module_without_manifest_returns_none(self) -> None:
        class Legacy:
            pass

        assert manifest_from_object(Legacy) is None


class TestRoleContracts:
    def test_gameplay_module_is_a_project_module(self) -> None:
        from engine.app.module import ProjectModule

        assert issubclass(GameplayModule, ProjectModule)
        assert GameplayModule.role is ModuleRole.GAMEPLAY

    def test_render_feature_defaults_to_no_passes(self) -> None:
        feature = RenderFeatureModule()
        assert feature.on_register(None) == []  # type: ignore[arg-type]
        feature.on_unregister()

    def test_tool_module_lifecycle_is_optional(self) -> None:
        tool = ToolModule()
        tool.on_attach(None)  # type: ignore[arg-type]
        tool.on_ui(None)  # type: ignore[arg-type]
        tool.on_detach(None)  # type: ignore[arg-type]
