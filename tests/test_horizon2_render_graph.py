"""Tests for the Horizon 2 render graph extension points (H2.3)."""

from __future__ import annotations

import pytest

from engine.gfx.render_feature import RenderFeature, RenderFeatureRegistry
from engine.gfx.render_graph import (
    RenderGraph,
    RenderGraphContext,
    RenderGraphError,
    RenderPass,
)
from engine.gfx.render_services import (
    RESERVED_TEXTURE_UNITS,
    RenderCapabilityError,
    RenderServices,
)


class DummyPass(RenderPass):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.executed = False

    def execute(self, context: RenderGraphContext) -> None:
        self.executed = True


class TestGraphValidation:
    def test_undeclared_passes_keep_insertion_order(self) -> None:
        graph = RenderGraph()
        for name in ("A", "B", "C"):
            graph.add_pass(DummyPass(name))
        assert [p.name for p in graph.compile()] == ["A", "B", "C"]

    def test_after_constraint_reorders(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A"))
        graph.add_pass(DummyPass("B", after=("A",)))
        graph.add_pass(DummyPass("C", after=("B",)))
        assert [p.name for p in graph.compile()] == ["A", "B", "C"]

    def test_before_constraint_reorders(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("Late"))
        graph.add_pass(DummyPass("Early", before=("Late",)))
        assert [p.name for p in graph.compile()] == ["Early", "Late"]

    def test_unknown_ordering_reference_is_rejected(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A", after=("Ghost",)))
        report = graph.validate()
        assert report.is_valid is False
        assert any("unknown pass 'Ghost'" in e for e in report.errors)

    def test_duplicate_output_is_rejected(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A", outputs=("shared",)))
        graph.add_pass(DummyPass("B", outputs=("shared",)))
        report = graph.validate()
        assert report.is_valid is False
        assert any("produced by both" in e for e in report.errors)

    def test_missing_input_is_rejected(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A", inputs=("never_produced",)))
        report = graph.validate()
        assert report.is_valid is False
        assert any("requires resource 'never_produced'" in e for e in report.errors)

    def test_external_input_satisfies_requirement(self) -> None:
        graph = RenderGraph()
        graph.declare_external_input("g_buffer")
        graph.add_pass(DummyPass("A", inputs=("g_buffer",)))
        assert graph.validate().is_valid is True

    def test_produced_input_satisfies_later_pass(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("Producer", outputs=("tex",)))
        graph.add_pass(DummyPass("Consumer", inputs=("tex",)))
        assert graph.validate().is_valid is True

    def test_cycle_is_detected(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A", after=("B",)))
        graph.add_pass(DummyPass("B", after=("A",)))
        report = graph.validate()
        assert report.is_valid is False
        assert any("cycle" in e.lower() for e in report.errors)

    def test_compile_raises_on_invalid_graph(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A", inputs=("missing",)))
        with pytest.raises(RenderGraphError, match="validation failed"):
            graph.compile()

    def test_remove_pass_invalidates_compiled_order(self) -> None:
        graph = RenderGraph()
        graph.add_pass(DummyPass("A"))
        graph.add_pass(DummyPass("B"))
        assert len(graph.compile()) == 2
        assert graph.remove_pass("B") is True
        assert [p.name for p in graph.compile()] == ["A"]

    def test_execute_skips_disabled_passes(self) -> None:
        graph = RenderGraph()
        a = graph.add_pass(DummyPass("A"))
        b = graph.add_pass(DummyPass("B"))
        graph.set_pass_enabled("B", False)
        graph.execute(RenderGraphContext(ctx=None, width=1, height=1, config=None, frame_context=None))  # type: ignore[arg-type]
        assert a.executed is True
        assert b.executed is False


class TestRenderServices:
    def _services(self, extensions=(), version_code=450) -> RenderServices:
        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.extensions = extensions
        ctx.version_code = version_code
        return RenderServices(ctx=ctx, config=None, width=1, height=1)  # type: ignore[arg-type]

    def test_capability_detection(self) -> None:
        services = self._services(extensions=("GL_ARB_multi_draw_indirect",), version_code=450)
        assert services.has_capability("compute") is True
        assert services.has_capability("tessellation") is True
        assert services.has_capability("multi_draw_indirect") is True
        assert services.has_capability("shader_draw_parameters") is False
        assert services.has_capability("unknown_capability") is False

    def test_require_capability_raises(self) -> None:
        services = self._services(extensions=(), version_code=330)
        with pytest.raises(RenderCapabilityError, match="compute"):
            services.require_capability("compute", "TestFeature")

    def test_reserved_texture_unit_cannot_be_claimed(self) -> None:
        services = self._services()
        reserved = next(iter(RESERVED_TEXTURE_UNITS))
        with pytest.raises(RenderCapabilityError, match="reserved for"):
            services.claim_texture_unit(reserved)

    def test_feature_texture_units_are_allocated_and_released(self) -> None:
        services = self._services()
        first = services.claim_next_texture_unit("A")
        second = services.claim_next_texture_unit("B")
        assert first != second
        with pytest.raises(RenderCapabilityError, match="already claimed"):
            services.claim_texture_unit(first, "C")
        services.release_texture_unit(first)
        assert services.claim_next_texture_unit("C") == first


class TestRenderFeatureRegistry:
    def _services(self) -> RenderServices:
        class FakeCtx:
            extensions = ("GL_ARB_compute_shader",)
            version_code = 450

        return RenderServices(ctx=FakeCtx(), config=None, width=1, height=1)  # type: ignore[arg-type]

    def test_feature_passes_are_added_to_graph(self) -> None:
        class Feature(RenderFeature):
            name = "TestFeature"

            def create_passes(self, services):
                return [DummyPass("FeaturePass", outputs=("feature_tex",))]

        graph = RenderGraph()
        registry = RenderFeatureRegistry()
        registry.register(Feature())
        report = registry.apply_to(graph, self._services())
        assert report.is_valid is True
        assert graph.get_pass("FeaturePass") is not None

    def test_feature_ordering_constraints_are_applied(self) -> None:
        class Feature(RenderFeature):
            name = "OrderedFeature"
            after = ("DeferredResolve",)

            def create_passes(self, services):
                return [DummyPass("OrderedPass")]

        graph = RenderGraph()
        graph.add_pass(DummyPass("DeferredResolve"))
        registry = RenderFeatureRegistry()
        registry.register(Feature())
        assert registry.apply_to(graph, self._services()).is_valid is True
        order = [p.name for p in graph.compile()]
        assert order.index("DeferredResolve") < order.index("OrderedPass")

    def test_missing_capability_is_reported(self) -> None:
        class Feature(RenderFeature):
            name = "NeedsTess"
            requires = ("tessellation",)

            def create_passes(self, services):
                return [DummyPass("TessPass")]

        class WeakCtx:
            extensions = ()
            version_code = 330

        services = RenderServices(ctx=WeakCtx(), config=None, width=1, height=1)  # type: ignore[arg-type]
        graph = RenderGraph()
        registry = RenderFeatureRegistry()
        registry.register(Feature())
        report = registry.apply_to(graph, services)
        assert report.is_valid is False
        assert any("tessellation" in e for e in report.errors)

    def test_duplicate_feature_name_is_rejected(self) -> None:
        registry = RenderFeatureRegistry()
        registry.register(RenderFeature())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(RenderFeature())

    def test_duplicate_pass_name_is_reported(self) -> None:
        class Feature(RenderFeature):
            name = "Dup"

            def create_passes(self, services):
                return [DummyPass("Same")]

        graph = RenderGraph()
        graph.add_pass(DummyPass("Same"))
        registry = RenderFeatureRegistry()
        registry.register(Feature())
        report = registry.apply_to(graph, self._services())
        assert report.is_valid is False
        assert any("duplicate pass 'Same'" in e for e in report.errors)
