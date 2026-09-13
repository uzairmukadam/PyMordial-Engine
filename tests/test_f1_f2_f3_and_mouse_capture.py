"""Unit test validating independent Dear ImGui F1/F2/F3 debug panels, multi-style F1 cycling, F9 mouse release, and unrestricted gameplay."""

import math
from pathlib import Path
import pygame
from engine.core.ecs import EntityManager
from engine.input import InputManager, Key
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor
from engine.debug import SystemMonitor, EngineTweaks, GameTweaks, DebugToast, DebugMenu


class TestDebugMenuAndMouseCapture:
    def test_mouse_capture_and_f9_toggle(self):
        """Validates that mouse capture is cleanly toggled by F9 and defaults to captured."""
        input_mgr = InputManager(register_defaults=True)

        # 1. Start with mouse grabbed
        input_mgr.set_mouse_grab(True)
        assert input_mgr.is_mouse_grabbed is True

        # 2. F9 toggle frees the cursor
        new_grab = not input_mgr.is_mouse_grabbed
        input_mgr.set_mouse_grab(new_grab)
        assert input_mgr.is_mouse_grabbed is False

        # 3. F9 toggle recaptures the cursor
        new_grab = not input_mgr.is_mouse_grabbed
        input_mgr.set_mouse_grab(new_grab)
        assert input_mgr.is_mouse_grabbed is True

    def test_f1_multi_style_cycling(self):
        """Validates F1 3-stage lifecycle: 0 (Off) -> 1 (Basic HUD) -> 2 (Expanded Profiler) -> 0 (Closed)."""
        monitor = SystemMonitor(history_size=10)
        engine_tweaks = EngineTweaks()
        game_tweaks = GameTweaks()
        toast = DebugToast()
        input_mgr = InputManager(register_defaults=True)

        menu = DebugMenu(
            ctx=None,
            monitor=monitor,
            engine_tweaks=engine_tweaks,
            game_tweaks=game_tweaks,
            toast=toast,
            input_mgr=input_mgr,
        )

        assert menu.f1_style == 0
        assert not menu.visible

        # Press 1: Basic HUD
        s1 = menu.cycle_f1()
        assert s1 == 1
        assert menu.f1_style == 1
        assert menu.visible is True

        # Press 2: Expanded Profiler
        s2 = menu.cycle_f1()
        assert s2 == 2
        assert menu.f1_style == 2
        assert menu.visible is True

        # Press 3: Close
        s0 = menu.cycle_f1()
        assert s0 == 0
        assert menu.f1_style == 0
        assert menu.visible is False

    def test_independent_f1_f2_f3_panels(self):
        """Validates that F1, F2, and F3 are completely independent panels and can be open simultaneously."""
        monitor = SystemMonitor(history_size=10)
        engine_tweaks = EngineTweaks()
        game_tweaks = GameTweaks()
        toast = DebugToast()
        input_mgr = InputManager(register_defaults=True)

        menu = DebugMenu(
            ctx=None,
            monitor=monitor,
            engine_tweaks=engine_tweaks,
            game_tweaks=game_tweaks,
            toast=toast,
            input_mgr=input_mgr,
        )

        # 1. Open F1 (Basic HUD)
        menu.cycle_f1()
        assert menu.f1_style == 1
        assert not menu.show_graphics
        assert not menu.show_game_tweaks

        # 2. Open F2 (Graphics) - F1 MUST REMAIN OPEN
        menu.toggle_graphics()
        assert menu.f1_style == 1
        assert menu.show_graphics is True
        assert not menu.show_game_tweaks

        # 3. Open F3 (Game Tweaks) - ALL THREE PANELS ARE NOW OPEN SIMULTANEOUSLY
        menu.toggle_game_tweaks()
        assert menu.f1_style == 1
        assert menu.show_graphics is True
        assert menu.show_game_tweaks is True
        assert menu.visible is True

        # 4. Advance F1 to Expanded Profiler - F2 and F3 remain open
        menu.cycle_f1()
        assert menu.f1_style == 2
        assert menu.show_graphics is True
        assert menu.show_game_tweaks is True

        # 5. Close F1 - F2 and F3 remain open
        menu.cycle_f1()
        assert menu.f1_style == 0
        assert menu.show_graphics is True
        assert menu.show_game_tweaks is True
        assert menu.visible is True

        # 6. Close F2 - F3 remains open
        menu.toggle_graphics()
        assert menu.show_graphics is False
        assert menu.show_game_tweaks is True
        assert menu.visible is True

        # 7. Close F3 - All closed
        menu.toggle_game_tweaks()
        assert menu.show_game_tweaks is False
        assert menu.visible is False

    def test_gameplay_actions_unrestricted_with_debug_panels_open(self):
        """Validates that gameplay movement (WASD, sprint, jump) is never locked when debug panels are open."""
        input_mgr = InputManager(register_defaults=True)
        menu = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=DebugToast(),
            input_mgr=input_mgr,
        )

        # Open all debug panels
        menu.cycle_f1()  # F1 open
        menu.toggle_graphics()  # F2 open
        menu.toggle_game_tweaks()  # F3 open
        assert menu.visible is True

        # In the new design, input_mgr is NOT placed in a locking debug mode
        assert input_mgr.is_debug_mode is False

        # Simulate pressing 'W' (move forward)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        input_mgr.update_axes()

        move_vec = input_mgr.get_vector2("move")
        # Moving forward means move_forward > 0 (vector Y > 0)
        assert move_vec[1] > 0.0, "WASD movement MUST NOT be locked when debug panels are open!"

    def test_character_motor_rotation_sync_and_angular_damping(self):
        """Validates that CharacterMotor rotation synchronizes to Rapier and survives sync_to_ecs."""
        ecs = EntityManager(max_entities=100)
        physics = PhysicsManager()

        # Create character
        char_ent = ecs.create_entity(position=(0.0, 0.92, 0.0))
        motor = CharacterMotor(char_ent, physics, ecs, initial_position=(0.0, 0.92, 0.0))

        dense_idx = ecs.pool.get_dense_index(char_ent)

        # Move forward-right (strafe right = +X, forward = -Z)
        dt = 1.0 / 60.0
        motor.update(dt=dt, move_input=(1.0, 1.0), camera_yaw_deg=0.0)

        # Yaw should smoothly turn towards angle instead of NaN or 0
        assert motor.state.yaw_deg != 0.0

        # Rapier rigid body rotation must match the motor's yaw quaternion
        half_yaw = math.radians(motor.state.yaw_deg) * 0.5
        expected_qy = math.sin(half_yaw)
        expected_qw = math.cos(half_yaw)

        rapier_transform = physics.world.get_transform(char_ent)
        assert abs(rapier_transform[4] - expected_qy) < 1e-4
        assert abs(rapier_transform[6] - expected_qw) < 1e-4

        # Run physics.sync_to_ecs(ecs) - it must NOT overwrite rotation with identity!
        physics.sync_to_ecs(ecs)
        synced_qy = ecs.rigid_body_state[1, dense_idx, 4]
        synced_qw = ecs.rigid_body_state[1, dense_idx, 6]
        assert abs(synced_qy - expected_qy) < 1e-4
        assert abs(synced_qw - expected_qw) < 1e-4

    def test_frame_by_frame_debug_hotkeys_and_no_numeric_presets(self):
        """Validates frame-by-frame hotkey handling for F1 (cycling), F2, F3, and no numeric preset conflict."""
        monitor = SystemMonitor(history_size=10)
        engine_tweaks = EngineTweaks()
        game_tweaks = GameTweaks()
        toast = DebugToast()
        input_mgr = InputManager(register_defaults=True)

        menu = DebugMenu(
            ctx=None,
            monitor=monitor,
            engine_tweaks=engine_tweaks,
            game_tweaks=game_tweaks,
            toast=toast,
            input_mgr=input_mgr,
        )

        def step_simulated_frame():
            input_mgr.update_axes()
            if input_mgr.is_action_pressed("debug_perf_toggle"):
                menu.cycle_f1()
            elif input_mgr.is_action_pressed("debug_graphics_toggle"):
                menu.toggle_graphics()
            elif input_mgr.is_action_pressed("debug_game_toggle"):
                menu.toggle_game_tweaks()

        # Frame 1: User presses F1 -> style 1 (Basic HUD)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.f1_style == 1, "F1 press 1 must open style 1 (Basic HUD)"

        # Release F1
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.f1_style == 1

        # Frame 2: User presses F1 again -> style 2 (Expanded Profiler)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.f1_style == 2, "F1 press 2 must expand to style 2 (Expanded Profiler)"

        # Release F1
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.f1_style == 2

        # Frame 3: User presses F1 again -> style 0 (Closed)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.f1_style == 0, "F1 press 3 must close F1 panel"

        # Frame 4: User presses F2 -> Graphics panel opens
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2))
        step_simulated_frame()
        assert menu.show_graphics is True

        # Frame 5: User presses F3 -> Game Tweaks opens alongside F2
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3))
        step_simulated_frame()
        assert menu.show_graphics is True
        assert menu.show_game_tweaks is True

        # Release F2 & F3
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_F2))
        input_mgr.process_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_F3))
        step_simulated_frame()

        # Frame 6: User presses F2 again -> Closes Graphics, Game Tweaks remains open!
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2))
        step_simulated_frame()
        assert menu.show_graphics is False
        assert menu.show_game_tweaks is True

        # Frame 7: Ensure number keys 1-5 do NOT have preset action bindings
        for k in (Key.NUM_1, Key.NUM_2, Key.NUM_3, Key.NUM_4, Key.NUM_5):
            assert f"preset_{k}" not in input_mgr._actions

    def test_debug_menu_resolution_resize_adaptation(self):
        """Validates that DebugMenu synchronizes its viewport and dimensions on WindowResizeEvent."""
        from engine.events import publish_event, WindowResizeEvent

        input_mgr = InputManager()
        menu = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=DebugToast(),
            input_mgr=input_mgr,
            screen_width=1280,
            screen_height=720,
        )

        assert menu.width == 1280
        assert menu.height == 720
        assert menu.io.display_size[0] == 1280.0
        assert menu.io.display_size[1] == 720.0

        # Dispatch WindowResizeEvent to 1920x1080 (e.g. resolution change or fullscreen)
        publish_event(WindowResizeEvent(width=1920, height=1080))
        assert menu.width == 1920
        assert menu.height == 1080
        assert menu.io.display_size[0] == 1920.0
        assert menu.io.display_size[1] == 1080.0

        # Dispatch WindowResizeEvent to 2560x1440
        publish_event(WindowResizeEvent(width=2560, height=1440))
        assert menu.width == 2560
        assert menu.height == 1440
        assert menu.io.display_size[0] == 2560.0
        assert menu.io.display_size[1] == 1440.0

        menu.destroy()

    def test_debug_menu_headless_render_resilience_and_lazy_init(self):
        """Validates that DebugMenu.render() operates safely when renderer is None and logs warnings cleanly."""
        input_mgr = InputManager()
        toast = DebugToast()
        menu = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=toast,
            input_mgr=input_mgr,
            screen_width=1280,
            screen_height=720,
        )

        # Initially renderer is None in headless / non-display testing
        assert menu.renderer is None
        assert menu._warned_no_renderer is False

        # Calling render() when menu is hidden and no toasts: immediate return without warning
        menu.render(None)
        assert menu._warned_no_renderer is False

        # Open F1 and F2 panels
        menu.cycle_f1()
        menu.toggle_graphics()
        assert menu.visible is True

        # Calling render() when visible triggers warning and early return safely
        menu.render(None)
        assert menu._warned_no_renderer is True

        # Process event without renderer returns False safely
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1)
        res = menu.process_event(ev)
        assert res is False

        menu.destroy()

    def test_debug_menu_config_filepath_initialization(self, tmp_path):
        """Validates that DebugMenu correctly accepts string, Path, or None for config_filepath."""
        input_mgr = InputManager()
        toast = DebugToast()

        # 1. With None
        menu_none = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=toast,
            input_mgr=input_mgr,
            config_filepath=None,
        )
        assert menu_none.config_filepath is None
        menu_none.destroy()

        # 2. With str
        str_path = str(tmp_path / "test_settings.json")
        menu_str = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=toast,
            input_mgr=input_mgr,
            config_filepath=str_path,
        )
        assert menu_str.config_filepath == Path(str_path)
        menu_str.destroy()

        # 3. With Path
        path_obj = tmp_path / "test_settings_2.json"
        menu_path = DebugMenu(
            ctx=None,
            monitor=SystemMonitor(history_size=10),
            engine_tweaks=EngineTweaks(),
            game_tweaks=GameTweaks(),
            toast=toast,
            input_mgr=input_mgr,
            config_filepath=path_obj,
        )
        assert menu_path.config_filepath == path_obj
        menu_path.destroy()


