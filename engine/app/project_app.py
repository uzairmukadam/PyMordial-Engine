"""PyMordial Project Application Host.

Orchestrates the lightweight engine core: Window, RenderContext, ModernGL Deferred Pipeline,
Rapier3D physics, Audio, ECS, Input, Cameras, and self-contained ProjectModules.
"""

from __future__ import annotations
import sys
from typing import TYPE_CHECKING, TypeVar, Any
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
from engine.ui import UIManager
from engine.core.state import GameStateManager

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
        "ui",
        "state_manager",
        "engine_loop",
        "world_builder",
        "is_running",
        "system_monitor",
        "engine_tweaks",
        "game_tweaks",
        "debug_toast",
        "debug_menu",
        "_modules",
        "_static_draw_batches",
        "_active_draw_batches",
        "_clock",
        "_frame_counter",
    )

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config if config is not None else ProjectConfig()

        # 1. Window & ModernGL Render Context
        preset_enum = GraphicsQuality(self.config.quality_preset.lower())
        quality_cfg = get_quality_preset(preset_enum)
        if hasattr(self.config, "water_enabled") and not self.config.water_enabled:
            quality_cfg.water_enabled = False
        if hasattr(self.config, "particles_enabled") and not self.config.particles_enabled:
            quality_cfg.particles_enabled = False
            quality_cfg.particle_mode = "OFF"

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
        self.render_ctx.pipeline = self.pipeline


        # 3. ECS & Native Rapier3D Physics
        max_ent = getattr(self.config, "max_entities", 50_000)
        self.ecs = EntityManager(max_entities=max_ent)
        self.ecs.set_material_registry(self.pipeline.material_registry)
        gx, gy, gz = self.config.gravity
        self.physics = PhysicsManager(gravity_x=gx, gravity_y=gy, gravity_z=gz)

        # 4. Input & Deterministic Engine Loop
        self.input_manager = InputManager()
        if not self.config.headless:
            self.input_manager.set_mouse_grab(False)
            self.input_manager._was_mouse_grabbed = False

        fixed_dt = 1.0 / max(1.0, float(self.config.fixed_hz))
        self.engine_loop = EngineLoop(ecs=self.ecs, input_manager=self.input_manager, fixed_dt=fixed_dt)

        # 5. Audio & Cameras & Native UI & State Management
        self.audio: AudioEngine = get_audio_engine()
        self.camera_manager = CameraManager()
        self.ui: UIManager = UIManager(self.render_ctx.ctx, self.config.width, self.config.height)
        self.ui.attach_app(self)
        self.state_manager: GameStateManager = GameStateManager(app=self)

        # 6. Central Sacred Debug Subsystem (F1, F2, F3)
        self.system_monitor = None
        self.engine_tweaks = None
        self.game_tweaks = None
        self.debug_toast = None
        self.debug_menu = None

        if self.config.enable_debug and not self.config.headless:
            try:
                from engine.debug import (
                    SystemMonitor,
                    EngineTweaks,
                    GameTweaks,
                    DebugToast,
                    DebugMenu,
                )

                self.system_monitor = SystemMonitor()
                self.engine_tweaks = EngineTweaks()
                self.game_tweaks = GameTweaks()
                self.debug_toast = DebugToast()
                self.debug_menu = DebugMenu(
                    ctx=self.render_ctx.ctx,
                    monitor=self.system_monitor,
                    engine_tweaks=self.engine_tweaks,
                    game_tweaks=self.game_tweaks,
                    toast=self.debug_toast,
                    input_mgr=self.input_manager,
                    screen_width=self.config.width,
                    screen_height=self.config.height,
                    app=self,
                )
                if self.engine_tweaks is not None:
                    self.engine_tweaks.sync_from_pipeline(self.pipeline)
            except Exception as e:
                import logging
                logging.getLogger("PyMordial").warning("Failed to initialize DebugMenu: %s", e)

        # 7. Modules & World Generation
        self.world_builder: BaseWorldBuilder | None = None
        self._modules: list[ProjectModule] = []
        self._static_draw_batches: list[tuple[MeshAllocation, int, int, bool, bool]] = []
        self._active_draw_batches: list[tuple[MeshAllocation, int, int, bool, bool]] = []
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
            self._static_draw_batches.clear()

        self.world_builder = builder
        self.world_builder.build_world(self)

    @property
    def waypoint_graph(self) -> Any:
        """Retrieves the navigation waypoint graph from the active world builder, if present."""
        return getattr(self.world_builder, "waypoint_graph", None)

    def register_draw_batch(
        self,
        alloc: MeshAllocation,
        instance_count: int,
        first_instance: int,
        is_animated: bool = False,
        cast_shadow: bool = True,
    ) -> None:
        """Registers a static or persistent MDI draw batch, coalescing contiguous runs."""
        if instance_count <= 0:
            return
        if self._static_draw_batches:
            prev = self._static_draw_batches[-1]
            prev_cast = prev[4] if len(prev) > 4 else True
            if (
                prev[0] == alloc
                and prev[3] == is_animated
                and prev_cast == cast_shadow
                and (prev[2] + prev[1] == first_instance)
            ):
                self._static_draw_batches[-1] = (alloc, prev[1] + instance_count, prev[2], is_animated, cast_shadow)
                return
        self._static_draw_batches.append((alloc, instance_count, first_instance, is_animated, cast_shadow))

    # ---------------- Loop Callbacks ----------------

    def _bind_loop_callbacks(self) -> None:
        def on_fixed_step(dt: float) -> None:
            c_state = self.state_manager.current_state
            if c_state is not None:
                if not c_state.allow_fixed_update:
                    return
                c_state.on_fixed_update(self, dt)

            # 1. Update active modules on fixed simulation tick
            for m in self._modules:
                if m.enabled:
                    m.on_fixed_update(self, dt)

            # 2. Step native Rapier physics simulation
            self.physics.step_simulation(dt)

            # 3. Synchronize physics state to contiguous ECS memory
            self.physics.sync_to_ecs(self.ecs)

        def on_variable_step(dt: float, alpha: float) -> None:
            c_state = self.state_manager.current_state
            if c_state is not None:
                c_state.on_update(self, dt)
                if not c_state.allow_variable_update:
                    if self.ui is not None:
                        self.ui.update(dt)
                    return

            # Update active modules on variable render frame
            for m in self._modules:
                if m.enabled:
                    m.on_update(self, dt)

            self.camera_manager.update(dt)

            # Synchronize 3D audio listener with active camera
            cam = self.camera_manager.active_camera
            if cam is not None and self.audio is not None:
                self.audio.update(dt, cam.position, cam.forward)

            if self.ui is not None:
                self.ui.update(dt)

            if self.system_monitor is not None:
                self.system_monitor.update(dt)
                self.system_monitor.record_fps(1.0 / max(1e-4, dt))

        def on_render_step(alpha: float) -> None:
            c_state = self.state_manager.current_state
            if c_state is not None and not c_state.allow_render:
                return

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

            # 2. Gather draw batches (reusing pre-allocated list to prevent per-frame allocations)
            self._active_draw_batches.clear()
            self._active_draw_batches.extend(self._static_draw_batches)
            for m in self._modules:
                if m.enabled and hasattr(m, "get_draw_batches"):
                    extra_batches = m.get_draw_batches(self)
                    if extra_batches:
                        self._active_draw_batches.extend(extra_batches)

            # Fallback: if entities exist in ECS but no custom draw batches were registered, render them as cubes
            if not self._active_draw_batches and self.ecs.active_count > 0:
                alloc_cube = self.pipeline.mega_buffer.allocations.get("cube")
                if alloc_cube is not None:
                    self._active_draw_batches.append((alloc_cube, self.ecs.active_count, 0, False, True))

            # 3. Synchronize live graphical tweaks to render pipeline
            if self.engine_tweaks is not None:
                self.engine_tweaks.apply_to_pipeline(self.pipeline)

            # 4. Sun & lighting parameters
            sun_dir = getattr(self.pipeline.atmosphere, "sun_direction", (0.35, -0.85, 0.40))
            sun_lux = getattr(self.pipeline.config, "sun_intensity", 4.0)

            # 5. Submit frame to deferred render pipeline
            self.pipeline.render_frame(
                ecs=self.ecs,
                camera_pos=cam_pos,
                camera_target=cam_target,
                time_elapsed=self.engine_loop.elapsed_time,
                sun_dir=sun_dir,
                sun_lux=sun_lux,
                fovy_deg=fovy_deg,
                draw_batches=self._active_draw_batches,
                debug_draw=self.pipeline.debug,
                dt=1.0 / 60.0,
            )

        def on_ui_step() -> None:
            if hasattr(self.pipeline, "post_process") and self.pipeline.post_process and self.pipeline.post_process.final_fbo:
                self.pipeline.post_process.final_fbo.use()

            if self.ui is not None:
                self.ui.render()

            for m in self._modules:
                if m.enabled:
                    m.on_ui(self)

            if not self.config.headless:
                # Present any UI drawn to the final FBO onto the window backbuffer
                if (
                    hasattr(self.pipeline, "post_process")
                    and self.pipeline.post_process is not None
                    and self.pipeline.post_process.final_fbo is not None
                ):
                    self.render_ctx.ctx.copy_framebuffer(
                        self.render_ctx.ctx.screen,
                        self.pipeline.post_process.final_fbo,
                    )

                # Render Dear ImGui Sacred Debug Overlays on top of the screen
                if self.debug_menu is not None:
                    self.debug_menu.render()

                self.render_ctx.window.swap_buffers()


        self.engine_loop.on_fixed_update = on_fixed_step
        self.engine_loop.on_update = on_variable_step
        self.engine_loop.on_render = on_render_step
        self.engine_loop.on_ui = on_ui_step

    # ---------------- Execution ----------------

    def step_frame(self, dt: float) -> int:
        """Executes a single deterministic frame step (useful for headless test suites)."""
        return self.engine_loop.step_frame(dt)

    def stop(self) -> None:
        """Signals the engine main loop to stop and commence orderly shutdown."""
        self.is_running = False

    def run(self) -> None:
        """Runs the main real-time game loop until window close or quit requested."""
        self.is_running = True
        self._frame_counter = 0

        try:
            while self.is_running:
                raw_ms = self._clock.tick(0)
                dt = min(raw_ms * 0.001, 0.1)

                # Maintain mouse grab policy if active state requests it and debug menu is closed
                c_state = self.state_manager.current_state
                if (
                    c_state is not None
                    and c_state.grab_mouse
                    and not self.config.headless
                    and (self.debug_menu is None or not self.debug_menu.visible)
                ):
                    if not pygame.event.get_grab():
                        try:
                            pygame.event.set_grab(True)
                            pygame.mouse.set_visible(False)
                            pygame.mouse.get_rel()
                        except Exception:
                            pass

                # 1. Input event polling & dispatch
                events = self.input_manager.poll_events()
                for ev in events:
                    if ev.type == pygame.QUIT:
                        self.is_running = False
                        break

                    # Sacred debug hotkeys (F1, F2, F3) are handled first if debug subsystem is active
                    if self.debug_menu is not None:
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_F1:
                                style = self.debug_menu.cycle_f1()
                                if style == 2 and self.input_manager.is_mouse_grabbed:
                                    self.input_manager.set_mouse_grab(False)
                                    self.input_manager._was_mouse_grabbed = False
                                if self.debug_toast:
                                    msg = "Debug: Performance HUD [F1]" if style == 1 else ("Debug: Profiler [F1]" if style == 2 else "Debug HUD Closed")
                                    self.debug_toast.show(msg, duration=2.0, color=(56, 189, 248))
                                continue
                            elif ev.key == pygame.K_F2:
                                is_open = self.debug_menu.toggle_graphics()
                                if is_open and self.input_manager.is_mouse_grabbed:
                                    self.input_manager.set_mouse_grab(False)
                                    self.input_manager._was_mouse_grabbed = False
                                if self.debug_toast:
                                    msg = "Debug: Graphics Options [F2]" if is_open else "Graphics Options Closed"
                                    self.debug_toast.show(msg, duration=1.5, color=(147, 197, 253))
                                continue
                            elif ev.key == pygame.K_F3:
                                is_open = self.debug_menu.toggle_game_tweaks()
                                if is_open and self.input_manager.is_mouse_grabbed:
                                    self.input_manager.set_mouse_grab(False)
                                    self.input_manager._was_mouse_grabbed = False
                                if self.debug_toast:
                                    msg = "Debug: Game Developer Tweaks [F3]" if is_open else "Game Tweaks Closed"
                                    self.debug_toast.show(msg, duration=1.5, color=(147, 197, 253))
                                continue

                        # Forward events to ImGui if debug menu is currently open and visible
                        if self.debug_menu.visible and self.debug_menu.process_event(ev):
                            continue

                    # UI subsystem handles events first
                    if self.ui is not None and self.ui.handle_event(ev):
                        continue

                    # Game state manager handles events second
                    if self.state_manager.handle_event(ev):
                        continue

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
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Cleans up all modules, world geometry, physics bodies, and window context."""
        while self.state_manager.stack_depth > 0:
            try:
                self.state_manager.pop_state()
            except Exception:
                break

        for m in reversed(self._modules):
            try:
                m.on_detach(self)
            except Exception:
                pass
        self._modules.clear()

        if self.world_builder is not None:
            self.world_builder.teardown_world(self)
            self.world_builder = None

        if self.config.screenshot:
            try:
                self.render_ctx.save_screenshot(str(self.config.screenshot))
            except Exception as e:
                print(f"[WARN] Failed to save screenshot: {e}", file=sys.stderr)

        if self.ui is not None:
            try:
                self.ui.destroy()
            except Exception:
                pass
            self.ui = None

        if hasattr(self, "audio") and self.audio is not None:
            try:
                self.audio.stop_all()
            except Exception:
                pass

        if hasattr(self, "pipeline") and self.pipeline is not None:
            try:
                self.pipeline.destroy()
            except Exception:
                pass
            self.pipeline = None

        self.render_ctx.destroy()
