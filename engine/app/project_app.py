"""PyMordial Project Application Host.

Orchestrates the lightweight engine core: Window, RenderContext, ModernGL Deferred Pipeline,
Rapier3D physics, Audio, ECS, Input, Cameras, and self-contained ProjectModules.
"""

from __future__ import annotations
import sys
from typing import TYPE_CHECKING, TypeVar
import pygame

from engine.app.config import ProjectConfig
from engine.app.module import ProjectModule
from engine.audio import AudioEngine, get_audio_engine
from engine.camera.camera_manager import CameraManager
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset
from engine.core.ecs import EntityManager
from engine.core.loop import EngineLoop
from engine.gfx.pipeline import RenderPipeline
from engine.input import InputManager
from engine.physics.rapier_world import PhysicsManager
from engine.gfx.context import RenderContext
from engine.window import VSyncMode

if TYPE_CHECKING:
    from engine.gfx.mega_buffer import MeshAllocation
    from engine.world.base import BaseWorldBuilder

T = TypeVar("T", bound=ProjectModule)


class ProjectApp:
    """The central application host managing the engine lifecycle for a game project."""

    __slots__ = (
        "config",
        "render_ctx",
        "pipeline",
        "ecs",
        "physics",
        "audio",
        "input_manager",
        "camera_manager",
        "engine_loop",
        "world_builder",
        "is_running",
        "_modules",
        "_static_draw_batches",
        "_clock",
        "_frame_counter",
    )

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config if config is not None else ProjectConfig()

        # 1. Window & ModernGL Render Context
        preset_enum = GraphicsQuality(self.config.quality_preset.lower())
        quality_cfg = get_quality_preset(preset_enum)

        self.render_ctx = RenderContext(
            width=self.config.width,
            height=self.config.height,
            title=self.config.title,
            hidden=self.config.headless,
            config=quality_cfg,
        )
        self.render_ctx.window.set_vsync(VSyncMode.ON if self.config.vsync else VSyncMode.OFF)

        # 2. ModernGL Deferred Pipeline
        self.pipeline = RenderPipeline(self.render_ctx, quality_cfg)
        self.pipeline.apply_config(self.pipeline.config)

        # 3. ECS & Native Rapier3D Physics
        self.ecs = EntityManager(max_entities=1000)
        gx, gy, gz = self.config.gravity
        self.physics = PhysicsManager(gravity_x=gx, gravity_y=gy, gravity_z=gz)

        # 4. Input & Deterministic Engine Loop
        self.input_manager = InputManager()
        if not self.config.headless:
            self.input_manager.set_mouse_grab(True)
            self.input_manager._was_mouse_grabbed = True

        fixed_dt = 1.0 / max(1.0, float(self.config.fixed_hz))
        self.engine_loop = EngineLoop(ecs=self.ecs, input_manager=self.input_manager, fixed_dt=fixed_dt)

        # 5. Audio & Cameras
        self.audio: AudioEngine = get_audio_engine()
        self.camera_manager = CameraManager()

        # 6. Modules & World Generation
        self.world_builder: BaseWorldBuilder | None = None
        self._modules: list[ProjectModule] = []
        self._static_draw_batches: list[tuple[MeshAllocation, int, int, bool]] = []
        self._clock = pygame.time.Clock()
        self._frame_counter = 0
        self.is_running = False

        # Hook engine loop callbacks
        self._bind_loop_callbacks()

    # ---------------- Module Management ----------------

    def add_module(self, module: ProjectModule) -> ProjectModule:
        """Attaches a project module and triggers its on_attach lifecycle hook."""
        self._modules.append(module)
        module.on_attach(self)
        return module

    def get_module(self, module_type: type[T]) -> T | None:
        """Finds and returns the first attached module matching the given class."""
        for m in self._modules:
            if isinstance(m, module_type):
                return m
        return None

    def has_module(self, module_type: type[T]) -> bool:
        """Returns True if an attached module matches the given class."""
        return self.get_module(module_type) is not None

    def remove_module(self, module: ProjectModule) -> None:
        """Detaches a module and triggers its on_detach lifecycle hook."""
        if module in self._modules:
            module.on_detach(self)
            self._modules.remove(module)

    # ---------------- World & Batch Management ----------------

    def set_world_builder(self, builder: BaseWorldBuilder) -> None:
        """Assigns and executes a world builder to generate map geometry and colliders."""
        if self.world_builder is not None:
            self.world_builder.teardown_world(self)

        self.world_builder = builder
        self.world_builder.build_world(self)

    def register_draw_batch(
        self,
        alloc: MeshAllocation,
        instance_count: int,
        first_instance: int,
        is_animated: bool = False,
    ) -> None:
        """Registers a static or persistent MDI draw batch."""
        self._static_draw_batches.append((alloc, instance_count, first_instance, is_animated))

    # ---------------- Loop Callbacks ----------------

    def _bind_loop_callbacks(self) -> None:
        def on_fixed_step(dt: float) -> None:
            # 1. Update active modules on fixed simulation tick
            for m in self._modules:
                if m.enabled:
                    m.on_fixed_update(self, dt)

            # 2. Step native Rapier physics simulation
            self.physics.step_simulation(dt)

            # 3. Synchronize physics state to contiguous ECS memory
            self.physics.sync_to_ecs(self.ecs)

        def on_variable_step(dt: float, alpha: float) -> None:
            # Update active modules on variable render frame
            for m in self._modules:
                if m.enabled:
                    m.on_update(self, dt)

        def on_render_step(alpha: float) -> None:
            # 1. Determine active camera
            cam = self.camera_manager.active_camera
            if cam is not None:
                cam_pos = cam.position
                cam_target = cam.target
                fovy_deg = cam.fov
            else:
                cam_pos = (0.0, 5.0, 10.0)
                cam_target = (0.0, 0.0, 0.0)
                fovy_deg = 60.0

            # 2. Gather draw batches (static + any module-provided dynamic batches)
            batches = list(self._static_draw_batches)
            for m in self._modules:
                if m.enabled and hasattr(m, "get_draw_batches"):
                    extra_batches = m.get_draw_batches(self)
                    if extra_batches:
                        batches.extend(extra_batches)

            # 3. Sun & lighting parameters
            sun_dir = getattr(self.pipeline.atmosphere, "sun_direction", (0.35, -0.85, 0.40))
            sun_lux = getattr(self.pipeline.config, "sun_intensity", 4.0)

            # 4. Submit frame to deferred render pipeline
            self.pipeline.render_frame(
                ecs=self.ecs,
                camera_pos=cam_pos,
                camera_target=cam_target,
                time_elapsed=self.engine_loop.elapsed_time,
                sun_dir=sun_dir,
                sun_lux=sun_lux,
                fovy_deg=fovy_deg,
                draw_batches=batches,
                debug_draw=self.pipeline.debug,
                dt=1.0 / 60.0,
            )

        def on_ui_step() -> None:
            for m in self._modules:
                if m.enabled:
                    m.on_ui(self)

            if not self.config.headless:
                self.render_ctx.window.swap_buffers()

        self.engine_loop.on_fixed_update = on_fixed_step
        self.engine_loop.on_update = on_variable_step
        self.engine_loop.on_render = on_render_step
        self.engine_loop.on_ui = on_ui_step

    # ---------------- Execution ----------------

    def step_frame(self, dt: float) -> int:
        """Executes a single deterministic frame step (useful for headless test suites)."""
        return self.engine_loop.step_frame(dt)

    def run(self) -> None:
        """Runs the main real-time game loop until window close or quit requested."""
        self.is_running = True
        self._frame_counter = 0

        while self.is_running:
            raw_ms = self._clock.tick(0)
            dt = min(raw_ms * 0.001, 0.1)

            # 1. Input event polling & dispatch
            events = self.input_manager.poll_events()
            for ev in events:
                if ev.type == pygame.QUIT:
                    self.is_running = False
                    break

                # Dispatch event to modules
                consumed = False
                for m in self._modules:
                    if m.enabled and m.on_event(self, ev):
                        consumed = True
                        break
                if consumed:
                    continue

            if self.input_manager.should_quit:
                self.is_running = False
                break

            # 2. Step simulation, interpolation, and rendering
            self.step_frame(dt)

            self._frame_counter += 1
            if self.config.max_frames is not None and self._frame_counter >= self.config.max_frames:
                self.is_running = False
                break

        self.shutdown()

    def shutdown(self) -> None:
        """Cleans up all modules, world geometry, physics bodies, and window context."""
        for m in reversed(self._modules):
            m.on_detach(self)
        self._modules.clear()

        if self.world_builder is not None:
            self.world_builder.teardown_world(self)
            self.world_builder = None

        if self.config.screenshot:
            try:
                self.render_ctx.save_screenshot(str(self.config.screenshot))
            except Exception as e:
                print(f"[WARN] Failed to save screenshot: {e}", file=sys.stderr)

        self.render_ctx.destroy()
