"""Modular Render Graph and Pass Architecture for PyMordial Engine.

Provides an extensible, node-based rendering graph inspired by modern AAA
engines (Unreal Engine RDG / Frostbite FrameGraph). Decouples passes,
manages transient resource blackboards, and enables dynamic pass toggling.

Passes may declare the resources they consume and produce, plus ordering
constraints. ``RenderGraph.validate()`` rejects missing inputs, duplicate
outputs, unknown ordering references, and cycles *before* any GPU work runs.
Passes that declare nothing keep plain insertion order, so the existing
hardcoded pipeline can be described and validated without being rewritten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import moderngl

from engine.core.ecs import EntityManager
from engine.debug.debug_draw import DebugDraw
from engine.gfx.frame_context import FrameContext
from engine.gfx.quality_presets import RenderConfig


@dataclass
class RenderGraphContext:
    """Frame context and resource blackboard passed to all render passes."""

    ctx: moderngl.Context
    width: int
    height: int
    config: RenderConfig
    frame_context: FrameContext
    ecs: EntityManager | None = None
    camera_pos: tuple[float, float, float] = (0.0, 5.0, 10.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    time_elapsed: float = 0.0
    delta_time: float = 0.016
    sun_dir: tuple[float, float, float] = (0.35, -0.85, 0.40)
    sun_lux: float = 4.0
    fovy_deg: float = 60.0
    draw_batches: list[Any] | None = None
    debug_draw: DebugDraw | None = None
    # Shared blackboard for inter-pass texture/buffer resource exchange
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderGraphReport:
    """Diagnostics produced by render graph validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.is_valid or self.errors:
            raise RenderGraphError("Render graph validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors))


class RenderGraphError(Exception):
    """Raised when a render graph cannot be validated or scheduled."""


class RenderPass(ABC):
    """Abstract base class for all Render Graph passes.

    Args:
        name: Unique pass name.
        enabled: Whether the pass runs.
        inputs: Resource names this pass reads from the blackboard.
        outputs: Resource names this pass writes to the blackboard.
        after: Pass names that must execute before this one.
        before: Pass names that must execute after this one.
        capabilities: GPU capabilities this pass requires (e.g. ``"compute"``).
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        after: Iterable[str] = (),
        before: Iterable[str] = (),
        capabilities: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.inputs: tuple[str, ...] = tuple(inputs)
        self.outputs: tuple[str, ...] = tuple(outputs)
        self.after: tuple[str, ...] = tuple(after)
        self.before: tuple[str, ...] = tuple(before)
        self.capabilities: tuple[str, ...] = tuple(capabilities)

    def prepare(self, context: RenderGraphContext) -> None:
        """Called before pass execution to bind or prepare prerequisites."""
        pass

    @abstractmethod
    def execute(self, context: RenderGraphContext) -> None:
        """Executes GPU draw calls, compute dispatches, or state changes."""
        pass

    def resize(self, width: int, height: int) -> None:
        """Handles framebuffer and viewport dimension changes."""
        pass

    def destroy(self) -> None:
        """Releases all GPU textures, programs, and framebuffers allocated by this pass."""
        pass


class RenderGraph:
    """Coordinates execution order, dependency blackboard, and pass lifecycles."""

    def __init__(self) -> None:
        self.passes: list[RenderPass] = []
        self._pass_map: dict[str, RenderPass] = {}
        self.execution_times: dict[str, float] = {}
        # Resources supplied by the pipeline itself rather than by a pass.
        self.external_inputs: set[str] = set()
        self._compiled: list[RenderPass] | None = None

    def add_pass(self, render_pass: RenderPass) -> RenderPass:
        """Appends a new pass to the graph."""
        if render_pass.name in self._pass_map:
            raise ValueError(f"Render pass with name '{render_pass.name}' already registered.")
        self.passes.append(render_pass)
        self._pass_map[render_pass.name] = render_pass
        self._compiled = None
        return render_pass

    def remove_pass(self, name: str) -> bool:
        """Removes a registered pass by name."""
        render_pass = self._pass_map.pop(name, None)
        if render_pass is None:
            return False
        self.passes.remove(render_pass)
        self._compiled = None
        return True

    def get_pass(self, name: str) -> RenderPass | None:
        """Returns registered render pass by name, or None."""
        return self._pass_map.get(name)

    def set_pass_enabled(self, name: str, enabled: bool) -> None:
        """Enables or disables a specific pass by name."""
        p = self._pass_map.get(name)
        if p is not None:
            p.enabled = enabled

    def declare_external_input(self, *names: str) -> None:
        """Declares resources produced outside the graph (by the pipeline)."""
        self.external_inputs.update(names)

    def validate(self) -> RenderGraphReport:
        """Checks ordering references, resource availability, and cycles."""
        report = RenderGraphReport()

        # 1. Ordering references must exist.
        for p in self.passes:
            for ref in p.after + p.before:
                if ref not in self._pass_map:
                    report.errors.append(
                        f"Pass '{p.name}' references unknown pass '{ref}' in its ordering constraints."
                    )

        # 2. Duplicate outputs.
        producers: dict[str, str] = {}
        for p in self.passes:
            for out in p.outputs:
                if out in producers:
                    report.errors.append(f"Resource '{out}' is produced by both '{producers[out]}' and '{p.name}'.")
                else:
                    producers[out] = p.name

        # 3. Cycles (checked on the ordering graph).
        try:
            self._topological_order()
        except RenderGraphError as exc:
            report.errors.append(str(exc))

        # 4. Inputs must be produced by an earlier pass or declared external.
        if not report.errors:
            available = set(self.external_inputs)
            for p in self._compiled or []:
                for name in p.inputs:
                    if name not in available:
                        report.errors.append(
                            f"Pass '{p.name}' requires resource '{name}' which no earlier pass "
                            f"produces and which is not declared as an external input."
                        )
                available.update(p.outputs)

        report.is_valid = len(report.errors) == 0
        return report

    def compile(self) -> list[RenderPass]:
        """Validates the graph and returns the deterministic execution order."""
        report = self.validate()
        report.raise_for_errors()
        return list(self._compiled or [])

    def _topological_order(self) -> list[RenderPass]:
        """Kahn-style topological sort with insertion order as the tie-break."""
        if self._compiled is not None:
            return self._compiled

        index = {p.name: i for i, p in enumerate(self.passes)}
        successors: dict[str, set[str]] = {p.name: set() for p in self.passes}
        indegree: dict[str, int] = {p.name: 0 for p in self.passes}

        def add_edge(before_name: str, after_name: str) -> None:
            if after_name not in successors[before_name]:
                successors[before_name].add(after_name)
                indegree[after_name] += 1

        for p in self.passes:
            for ref in p.after:
                if ref in self._pass_map:
                    add_edge(ref, p.name)
            for ref in p.before:
                if ref in self._pass_map:
                    add_edge(p.name, ref)

        ready = sorted((n for n, d in indegree.items() if d == 0), key=lambda n: index[n])
        ordered: list[RenderPass] = []
        while ready:
            name = ready.pop(0)
            ordered.append(self._pass_map[name])
            for succ in sorted(successors[name], key=lambda n: index[n]):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
            ready.sort(key=lambda n: index[n])

        if len(ordered) != len(self.passes):
            remaining = [n for n, d in indegree.items() if d > 0]
            raise RenderGraphError("Render graph contains a cycle involving: " + ", ".join(sorted(remaining)))

        self._compiled = ordered
        return ordered

    def execute(self, context: RenderGraphContext) -> None:
        """Executes all enabled passes in validated order."""
        for p in self._compiled or self._topological_order():
            if p.enabled:
                p.prepare(context)
                p.execute(context)

    def resize(self, width: int, height: int) -> None:
        """Notifies all registered passes of a window or resolution change."""
        for p in self.passes:
            p.resize(width, height)

    def destroy(self) -> None:
        """Releases all registered passes in reverse order."""
        for p in reversed(self.passes):
            p.destroy()
        self.passes.clear()
        self._pass_map.clear()
        self._compiled = None


__all__ = [
    "RenderGraph",
    "RenderGraphContext",
    "RenderGraphError",
    "RenderGraphReport",
    "RenderPass",
]
