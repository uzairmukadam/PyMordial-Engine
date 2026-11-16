"""Render Feature Registration for PyMordial Engine.

A ``RenderFeature`` is the module-facing contract for contributing passes to the
renderer. Features declare their passes and the resources those passes consume
and produce; the pipeline validates the resulting graph before executing it.

Features are registered through ``RenderPipeline.register_feature`` and never
modify the pipeline's hardcoded pass order directly.
"""

from __future__ import annotations

from typing import Iterable

from engine.gfx.render_graph import RenderGraph, RenderGraphReport, RenderPass
from engine.gfx.render_services import RenderServices


class RenderFeature:
    """Base class for a renderer extension contributed by a module.

    Subclasses implement ``create_passes`` and return the passes they own. The
    pipeline inserts them into the render graph and validates ordering and
    resource declarations before the first frame executes.
    """

    name: str = "RenderFeature"
    #: Pass names this feature's passes must run after.
    after: tuple[str, ...] = ()
    #: Pass names this feature's passes must run before.
    before: tuple[str, ...] = ()
    #: GPU capabilities required by this feature.
    requires: tuple[str, ...] = ()

    def create_passes(self, services: RenderServices) -> Iterable[RenderPass]:
        """Builds and returns the passes owned by this feature."""
        return ()

    def destroy(self) -> None:
        """Releases resources owned by this feature."""
        pass


class RenderFeatureRegistry:
    """Collects render features and applies their ordering constraints."""

    __slots__ = ("_features",)

    def __init__(self) -> None:
        self._features: list[RenderFeature] = []

    def __len__(self) -> int:
        return len(self._features)

    @property
    def features(self) -> tuple[RenderFeature, ...]:
        return tuple(self._features)

    def register(self, feature: RenderFeature) -> RenderFeature:
        """Registers a feature, rejecting duplicate names."""
        if any(f.name == feature.name for f in self._features):
            raise ValueError(f"Render feature '{feature.name}' is already registered.")
        self._features.append(feature)
        return feature

    def unregister(self, feature: RenderFeature) -> bool:
        if feature in self._features:
            self._features.remove(feature)
            return True
        return False

    def apply_to(self, graph: RenderGraph, services: RenderServices) -> RenderGraphReport:
        """Adds every feature's passes to the graph and validates the result.

        Ordering constraints declared by a feature are applied to each of its
        passes. Missing capabilities are reported as errors rather than raised,
        so the caller can decide whether to disable the feature or abort.
        """
        report = RenderGraphReport()

        for feature in self._features:
            for capability in feature.requires:
                if not services.has_capability(capability):
                    report.errors.append(
                        f"Render feature '{feature.name}' requires the '{capability}' GPU capability."
                    )
            if report.errors:
                continue

            for render_pass in feature.create_passes(services):
                if graph.get_pass(render_pass.name) is not None:
                    report.errors.append(
                        f"Render feature '{feature.name}' produced duplicate pass '{render_pass.name}'."
                    )
                    continue
                # Feature-level ordering applies to every pass it owns.
                render_pass.after = tuple(render_pass.after) + tuple(feature.after)
                render_pass.before = tuple(render_pass.before) + tuple(feature.before)
                graph.add_pass(render_pass)

        if not report.errors:
            graph_report = graph.validate()
            report.errors.extend(graph_report.errors)
            report.warnings.extend(graph_report.warnings)

        report.is_valid = len(report.errors) == 0
        return report

    def destroy_all(self) -> None:
        for feature in reversed(self._features):
            try:
                feature.destroy()
            except Exception:
                pass
        self._features.clear()


__all__ = ["RenderFeature", "RenderFeatureRegistry"]
