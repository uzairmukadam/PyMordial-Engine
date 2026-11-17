"""Module Registry, Dependency Resolver, and Lifecycle Host for PyMordial Engine.

Provides deterministic module ordering, cycle/missing-dependency diagnostics, and
a host that attaches modules in dependency order and rolls back cleanly when any
module fails to initialize.

Design notes:
- Resolution is deterministic: ties are broken by module id, so the same inputs
  always produce the same initialization order.
- The host never leaves partially-attached modules behind. If attach fails, every
  already-attached module is detached in reverse order and every event
  subscription made through the host is removed.
- Legacy modules without a manifest are still supported; they are treated as
  gameplay modules with no dependencies and ordered after declared modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from engine.app.module_manifest import (
    ModuleDependency,
    ModuleManifest,
    ModuleRole,
    manifest_from_object,
    validate_manifest,
)
from engine.app.project_validator import ValidationReport
from engine.events import subscribe_event, unsubscribe_event


class ModuleResolutionError(Exception):
    """Raised when the module dependency graph cannot be resolved."""

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class ModuleAttachError(Exception):
    """Raised when a module fails during attach; the host has already rolled back."""

    def __init__(self, module_id: str, cause: BaseException) -> None:
        super().__init__(f"Module '{module_id}' failed to attach: {cause}")
        self.module_id = module_id
        self.cause = cause


@dataclass(slots=True)
class ModuleRecord:
    """A registered module: its manifest (if any) and the factory that builds it."""

    module_id: str
    factory: Callable[[], Any]
    manifest: ModuleManifest | None = None
    instance: Any = None

    @property
    def role(self) -> ModuleRole:
        return self.manifest.role if self.manifest is not None else ModuleRole.GAMEPLAY

    @property
    def dependencies(self) -> tuple[ModuleDependency, ...]:
        return self.manifest.dependencies if self.manifest is not None else ()


class ModuleRegistry:
    """Collects module declarations and resolves them into a deterministic order."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[str, ModuleRecord] = {}

    def register(
        self,
        module_id: str,
        factory: Callable[[], Any],
        manifest: ModuleManifest | None = None,
    ) -> ModuleRecord:
        """Registers a module factory under a unique id.

        Raises:
            ValueError: if the id is already registered.
        """
        if module_id in self._records:
            raise ValueError(f"Module '{module_id}' is already registered.")
        record = ModuleRecord(module_id=module_id, factory=factory, manifest=manifest)
        self._records[module_id] = record
        return record

    def register_instance(self, instance: Any, module_id: str | None = None) -> ModuleRecord:
        """Registers an already-constructed module, discovering its manifest if present."""
        manifest = manifest_from_object(instance)
        resolved_id = module_id or (manifest.id if manifest is not None else type(instance).__name__)
        return self.register(resolved_id, lambda: instance, manifest)

    def get(self, module_id: str) -> ModuleRecord | None:
        return self._records.get(module_id)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, module_id: object) -> bool:
        return module_id in self._records

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def validate(self) -> ValidationReport:
        """Validates every declared manifest and cross-checks dependency availability."""
        report = ValidationReport()

        for module_id, record in sorted(self._records.items()):
            if record.manifest is None:
                continue
            manifest_report = validate_manifest(record.manifest)
            report.errors.extend(manifest_report.errors)
            if record.manifest.id != module_id:
                report.errors.append(f"Module registered as '{module_id}' declares manifest id '{record.manifest.id}'.")

        for module_id, record in sorted(self._records.items()):
            for dep in record.dependencies:
                if dep.module_id in self._records:
                    continue
                if dep.optional:
                    report.warnings.append(
                        f"Module '{module_id}' declares optional dependency '{dep.module_id}' which is not registered."
                    )
                else:
                    report.errors.append(f"Module '{module_id}' requires '{dep.module_id}' which is not registered.")

        report.is_valid = len(report.errors) == 0
        return report

    def resolve(self) -> list[ModuleRecord]:
        """Returns modules in deterministic dependency order.

        Raises:
            ModuleResolutionError: on invalid manifests, missing required
                dependencies, or dependency cycles.
        """
        report = self.validate()
        if not report.is_valid:
            raise ModuleResolutionError(
                "Module graph validation failed:\n" + "\n".join(f"  - {e}" for e in report.errors),
                report,
            )

        ordered: list[ModuleRecord] = []
        # 0 = unvisited, 1 = in progress (on stack), 2 = emitted
        state: dict[str, int] = {module_id: 0 for module_id in self._records}
        stack: list[str] = []

        def visit(module_id: str) -> None:
            current = state[module_id]
            if current == 2:
                return
            if current == 1:
                cycle_start = stack.index(module_id)
                cycle = stack[cycle_start:] + [module_id]
                raise ModuleResolutionError(
                    "Dependency cycle detected: " + " -> ".join(cycle),
                    report,
                )
            state[module_id] = 1
            stack.append(module_id)
            record = self._records[module_id]
            # Sorted dependency ids keep traversal deterministic.
            for dep in sorted(record.dependencies, key=lambda d: d.module_id):
                if dep.module_id in self._records:
                    visit(dep.module_id)
            stack.pop()
            state[module_id] = 2
            ordered.append(record)

        for module_id in sorted(self._records):
            visit(module_id)

        return ordered


class ModuleHost:
    """Attaches modules in dependency order and guarantees clean rollback."""

    __slots__ = ("registry", "_attached", "_subscriptions", "_failed")

    def __init__(self, registry: ModuleRegistry | None = None) -> None:
        self.registry = registry if registry is not None else ModuleRegistry()
        self._attached: list[ModuleRecord] = []
        self._subscriptions: list[tuple[type, Callable[[Any], None]]] = []
        self._failed: str | None = None

    @property
    def attached(self) -> tuple[ModuleRecord, ...]:
        return tuple(self._attached)

    @property
    def failed_module_id(self) -> str | None:
        return self._failed

    def subscribe(self, event_type: type, handler: Callable[[Any], None], priority: int = 0) -> None:
        """Subscribes on behalf of a module so the host can guarantee cleanup."""
        subscribe_event(event_type, handler, priority=priority)
        self._subscriptions.append((event_type, handler))

    def attach_all(self, app: Any) -> list[Any]:
        """Instantiates and attaches every registered module in dependency order.

        On failure, all previously attached modules are detached in reverse order
        and all host-owned subscriptions are removed before raising.

        Returns:
            The attached module instances, in attach order.
        """
        ordered = self.registry.resolve()
        instances: list[Any] = []

        for record in ordered:
            try:
                instance = record.factory()
                record.instance = instance
                self._attach_one(record, instance, app)
            except Exception as exc:
                self._failed = record.module_id
                self._rollback(app)
                raise ModuleAttachError(record.module_id, exc) from exc
            self._attached.append(record)
            instances.append(instance)

        return instances

    def _attach_one(self, record: ModuleRecord, instance: Any, app: Any) -> None:
        """Dispatches the role-appropriate attach hook."""
        if record.role is ModuleRole.RENDER_FEATURE:
            # Render features register with the renderer, not the app lifecycle.
            register = getattr(app, "register_render_feature", None)
            if register is None:
                raise RuntimeError(
                    f"Render feature '{record.module_id}' requires a host exposing register_render_feature()."
                )
            register(instance)
            return

        hook = getattr(instance, "on_attach", None)
        if callable(hook):
            hook(app)

    def detach_all(self, app: Any) -> None:
        """Detaches every attached module in reverse order and clears subscriptions."""
        for record in reversed(self._attached):
            instance = record.instance
            if instance is None:
                continue
            try:
                if record.role is ModuleRole.RENDER_FEATURE:
                    unregister = getattr(app, "unregister_render_feature", None)
                    if callable(unregister):
                        unregister(instance)
                else:
                    hook = getattr(instance, "on_detach", None)
                    if callable(hook):
                        hook(app)
            except Exception:
                # A failing detach must not prevent the remaining modules from cleaning up.
                pass
        self._attached.clear()
        self._clear_subscriptions()

    def _rollback(self, app: Any) -> None:
        """Detaches already-attached modules and drops host-owned subscriptions."""
        for record in reversed(self._attached):
            instance = record.instance
            if instance is None:
                continue
            try:
                if record.role is ModuleRole.RENDER_FEATURE:
                    unregister = getattr(app, "unregister_render_feature", None)
                    if callable(unregister):
                        unregister(instance)
                else:
                    hook = getattr(instance, "on_detach", None)
                    if callable(hook):
                        hook(app)
            except Exception:
                pass
        self._attached.clear()
        self._clear_subscriptions()

    def _clear_subscriptions(self) -> None:
        for event_type, handler in self._subscriptions:
            try:
                unsubscribe_event(event_type, handler)
            except Exception:
                pass
        self._subscriptions.clear()

    def forget(self) -> None:
        """Drops host bookkeeping without detaching.

        Used when the owning application has already detached the modules itself
        (for example during shutdown), so detach hooks never run twice.
        """
        self._attached.clear()
        self._clear_subscriptions()


__all__ = [
    "ModuleAttachError",
    "ModuleHost",
    "ModuleRecord",
    "ModuleRegistry",
    "ModuleResolutionError",
]
