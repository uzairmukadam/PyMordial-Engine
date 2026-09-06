"""Modular Render Graph and Pass Architecture for PyMordial Engine.

Provides an extensible, node-based rendering graph inspired by modern AAA
engines (Unreal Engine RDG / Frostbite FrameGraph). Decouples passes,
manages transient resource blackboards, and enables dynamic pass toggling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import moderngl

from engine.core.ecs import EntityManager
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.frame_context import FrameContext
from engine.debug.debug_draw import DebugDraw


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


class RenderPass(ABC):
    """Abstract base class for all Render Graph passes."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

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

    def add_pass(self, render_pass: RenderPass) -> RenderPass:
        """Appends a new pass to the graph."""
        if render_pass.name in self._pass_map:
            raise ValueError(f"Render pass with name '{render_pass.name}' already registered.")
        self.passes.append(render_pass)
        self._pass_map[render_pass.name] = render_pass
        return render_pass

    def get_pass(self, name: str) -> RenderPass | None:
        """Returns registered render pass by name, or None."""
        return self._pass_map.get(name)

    def set_pass_enabled(self, name: str, enabled: bool) -> None:
        """Enables or disables a specific pass by name."""
        p = self._pass_map.get(name)
        if p is not None:
            p.enabled = enabled

    def execute(self, context: RenderGraphContext) -> None:
        """Executes all enabled passes in registered sequential order."""
        for p in self.passes:
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
