"""Tests for the Horizon 2 module registry, resolver, and host (H2.2)."""

from __future__ import annotations

import pytest

from engine.app.module_manifest import ModuleDependency, ModuleManifest, ModuleRole, build_manifest
from engine.app.module_registry import (
    ModuleAttachError,
    ModuleHost,
    ModuleRegistry,
    ModuleResolutionError,
)
from engine.events.event import Event


class RecordingModule:
    """Minimal module that records lifecycle calls into a shared log."""

    def __init__(self, name: str, log: list[str], fail_on_attach: bool = False) -> None:
        self.name = name
        self.log = log
        self.fail_on_attach = fail_on_attach

    def on_attach(self, app) -> None:
        if self.fail_on_attach:
            raise RuntimeError(f"{self.name} exploded")
        self.log.append(f"attach:{self.name}")

    def on_detach(self, app) -> None:
        self.log.append(f"detach:{self.name}")


def _manifest(module_id: str, deps: tuple[str, ...] = (), optional: tuple[str, ...] = ()) -> ModuleManifest:
    dependencies = tuple(ModuleDependency(d) for d in deps) + tuple(
        ModuleDependency(d, optional=True) for d in optional
    )
    return build_manifest(module_id, entry=f"{module_id}:Mod", dependencies=dependencies)


class TestRegistryValidation:
    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ModuleRegistry()
        registry.register("a.b", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("a.b", lambda: None)

    def test_missing_required_dependency_is_reported(self) -> None:
        registry = ModuleRegistry()
        registry.register("a.b", lambda: None, _manifest("a.b", deps=("missing.mod",)))
        report = registry.validate()
        assert report.is_valid is False
        assert any("requires 'missing.mod'" in e for e in report.errors)

    def test_missing_optional_dependency_is_a_warning(self) -> None:
        registry = ModuleRegistry()
        registry.register("a.b", lambda: None, _manifest("a.b", optional=("absent.mod",)))
        report = registry.validate()
        assert report.is_valid is True
        assert any("optional dependency 'absent.mod'" in w for w in report.warnings)

    def test_manifest_id_mismatch_is_reported(self) -> None:
        registry = ModuleRegistry()
        registry.register("registered.id", lambda: None, _manifest("declared.id"))
        report = registry.validate()
        assert report.is_valid is False
        assert any("declares manifest id" in e for e in report.errors)

    def test_invalid_manifest_is_reported(self) -> None:
        registry = ModuleRegistry()
        registry.register("bad.id", lambda: None, ModuleManifest(id="bad.id", entry=""))
        report = registry.validate()
        assert report.is_valid is False
        assert any("'entry'" in e for e in report.errors)


class TestResolver:
    def test_dependencies_come_before_dependents(self) -> None:
        registry = ModuleRegistry()
        registry.register("core", lambda: None, _manifest("core"))
        registry.register("mid", lambda: None, _manifest("mid", deps=("core",)))
        registry.register("top", lambda: None, _manifest("top", deps=("mid",)))

        order = [r.module_id for r in registry.resolve()]
        assert order.index("core") < order.index("mid") < order.index("top")

    def test_resolution_is_deterministic(self) -> None:
        def build() -> list[str]:
            registry = ModuleRegistry()
            for name in ("zeta", "alpha", "mike"):
                registry.register(name, lambda: None, _manifest(name))
            return [r.module_id for r in registry.resolve()]

        assert build() == build() == ["alpha", "mike", "zeta"]

    def test_independent_modules_are_sorted_by_id(self) -> None:
        registry = ModuleRegistry()
        registry.register("b.mod", lambda: None, _manifest("b.mod"))
        registry.register("a.mod", lambda: None, _manifest("a.mod"))
        assert [r.module_id for r in registry.resolve()] == ["a.mod", "b.mod"]

    def test_transitive_dependency_chain(self) -> None:
        registry = ModuleRegistry()
        registry.register("a", lambda: None, _manifest("a", deps=("b",)))
        registry.register("b", lambda: None, _manifest("b", deps=("c",)))
        registry.register("c", lambda: None, _manifest("c"))
        assert [r.module_id for r in registry.resolve()] == ["c", "b", "a"]

    def test_cycle_is_detected_with_path(self) -> None:
        registry = ModuleRegistry()
        registry.register("a", lambda: None, _manifest("a", deps=("b",)))
        registry.register("b", lambda: None, _manifest("b", deps=("a",)))
        with pytest.raises(ModuleResolutionError, match="Dependency cycle detected"):
            registry.resolve()

    def test_self_cycle_is_detected(self) -> None:
        registry = ModuleRegistry()
        registry.register("a", lambda: None, _manifest("a", deps=("a",)))
        with pytest.raises(ModuleResolutionError):
            registry.resolve()

    def test_missing_dependency_aborts_before_attach(self) -> None:
        registry = ModuleRegistry()
        registry.register("a", lambda: None, _manifest("a", deps=("ghost",)))
        with pytest.raises(ModuleResolutionError, match="requires 'ghost'"):
            registry.resolve()

    def test_legacy_module_without_manifest_resolves(self) -> None:
        registry = ModuleRegistry()
        registry.register("legacy", lambda: None)
        assert [r.module_id for r in registry.resolve()] == ["legacy"]


class TestHostLifecycle:
    def test_attach_order_follows_dependencies(self) -> None:
        log: list[str] = []
        registry = ModuleRegistry()
        registry.register("core", lambda: RecordingModule("core", log), _manifest("core"))
        registry.register("top", lambda: RecordingModule("top", log), _manifest("top", deps=("core",)))

        host = ModuleHost(registry)
        host.attach_all(app=None)
        assert log == ["attach:core", "attach:top"]

    def test_detach_is_reverse_order(self) -> None:
        log: list[str] = []
        registry = ModuleRegistry()
        registry.register("core", lambda: RecordingModule("core", log), _manifest("core"))
        registry.register("top", lambda: RecordingModule("top", log), _manifest("top", deps=("core",)))

        host = ModuleHost(registry)
        host.attach_all(app=None)
        host.detach_all(app=None)
        assert log == ["attach:core", "attach:top", "detach:top", "detach:core"]

    def test_attach_failure_rolls_back_attached_modules(self) -> None:
        log: list[str] = []
        registry = ModuleRegistry()
        registry.register("core", lambda: RecordingModule("core", log), _manifest("core"))
        registry.register(
            "boom",
            lambda: RecordingModule("boom", log, fail_on_attach=True),
            _manifest("boom", deps=("core",)),
        )

        host = ModuleHost(registry)
        with pytest.raises(ModuleAttachError) as excinfo:
            host.attach_all(app=None)

        assert excinfo.value.module_id == "boom"
        assert host.failed_module_id == "boom"
        # core attached, then rolled back; boom never recorded an attach.
        assert log == ["attach:core", "detach:core"]
        assert host.attached == ()

    def test_attach_failure_removes_host_subscriptions(self) -> None:
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        registry = ModuleRegistry()
        registry.register("boom", lambda: RecordingModule("boom", [], fail_on_attach=True), _manifest("boom"))

        host = ModuleHost(registry)
        host.subscribe(Event, handler)
        with pytest.raises(ModuleAttachError):
            host.attach_all(app=None)

        # The subscription must be gone: publishing must not reach the handler.
        from engine.events import publish_event

        publish_event(Event())
        assert received == []

    def test_repeated_attach_detach_is_idempotent(self) -> None:
        log: list[str] = []
        registry = ModuleRegistry()
        registry.register("core", lambda: RecordingModule("core", log), _manifest("core"))

        host = ModuleHost(registry)
        host.attach_all(app=None)
        host.detach_all(app=None)
        host.detach_all(app=None)
        assert log == ["attach:core", "detach:core"]
        assert host.attached == ()

    def test_failing_detach_does_not_block_others(self) -> None:
        log: list[str] = []

        class BadDetach(RecordingModule):
            def on_detach(self, app) -> None:
                raise RuntimeError("detach failed")

        registry = ModuleRegistry()
        registry.register("bad", lambda: BadDetach("bad", log), _manifest("bad"))
        registry.register("good", lambda: RecordingModule("good", log), _manifest("good"))

        host = ModuleHost(registry)
        host.attach_all(app=None)
        host.detach_all(app=None)
        assert "detach:good" in log
        assert host.attached == ()

    def test_render_feature_uses_register_hook(self) -> None:
        registered: list[object] = []

        class FakeApp:
            def register_render_feature(self, feature) -> None:
                registered.append(feature)

        feature = object()
        registry = ModuleRegistry()
        registry.register(
            "render.feature",
            lambda: feature,
            build_manifest("render.feature", entry="x:Y", role=ModuleRole.RENDER_FEATURE),
        )

        host = ModuleHost(registry)
        host.attach_all(FakeApp())
        assert registered == [feature]

    def test_render_feature_without_host_support_fails_cleanly(self) -> None:
        registry = ModuleRegistry()
        registry.register(
            "render.feature",
            lambda: object(),
            build_manifest("render.feature", entry="x:Y", role=ModuleRole.RENDER_FEATURE),
        )
        host = ModuleHost(registry)
        with pytest.raises(ModuleAttachError, match="register_render_feature"):
            host.attach_all(app=None)

    def test_register_instance_discovers_manifest(self) -> None:
        class WithManifest:
            MODULE_MANIFEST = build_manifest("auto.id", entry="a:B")

        registry = ModuleRegistry()
        record = registry.register_instance(WithManifest())
        assert record.module_id == "auto.id"
        assert record.manifest is not None
