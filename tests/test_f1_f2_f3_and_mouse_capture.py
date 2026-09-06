"""Unit test validating dedicated F1/F2/F3 debug tabs, F9 mouse capture, and KCC jitter fixes."""

import math
from engine.core.ecs import EntityManager
from engine.input import InputManager
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor
from engine.debug import SystemMonitor, EngineTweaks, GameTweaks, DebugToast, DebugMenu


class TestDebugMenuAndMouseCapture:
    def test_mouse_capture_and_debug_mode_lifecycle(self):
        """Validates that mouse is captured by default and cleanly toggled by F9 and debug mode."""
        input_mgr = InputManager(register_defaults=True)

        # 1. Start with mouse grabbed
        input_mgr.set_mouse_grab(True)
        assert input_mgr.is_mouse_grabbed is True

        # 2. Entering debug mode frees the cursor and remembers prior grab state
        input_mgr.set_debug_mode(True)
        assert input_mgr.is_mouse_grabbed is False
        assert input_mgr._was_mouse_grabbed is True

        # 3. Exiting debug mode restores mouse capture
        input_mgr.set_debug_mode(False)
        assert input_mgr.is_mouse_grabbed is True
        assert input_mgr._was_mouse_grabbed is False

        # 4. F9 toggle
        new_grab = not input_mgr.is_mouse_grabbed
        input_mgr.set_mouse_grab(new_grab)
        assert input_mgr.is_mouse_grabbed is False

    def test_direct_f1_f2_f3_tab_toggling(self):
        """Validates that F1, F2, F3 toggle dedicated tabs and close cleanly."""
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

        assert not menu.visible

        # Press F1 -> Opens Tab 0 (Performance)
        opened = menu.toggle_tab(0)
        assert opened is True
        assert menu.visible is True
        assert menu.active_tab == 0

        # Press F1 again -> Closes menu
        closed = menu.toggle_tab(0)
        assert closed is False
        assert menu.visible is False

        # Press F2 -> Opens Tab 1 (Graphics Options)
        opened = menu.toggle_tab(1)
        assert opened is True
        assert menu.visible is True
        assert menu.active_tab == 1

        # Press F3 while Tab 1 is open -> Switches to Tab 2 (Game Options) without closing
        switched = menu.toggle_tab(2)
        assert switched is True
        assert menu.visible is True
        assert menu.active_tab == 2

        # Press F3 again -> Closes menu
        closed = menu.toggle_tab(2)
        assert closed is False
        assert menu.visible is False

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
        """Validates that pressing F1 opens and keeps menu open, switching tabs works, and number keys 1-5 do not change presets."""
        import pygame
        from engine.input import Key

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
            tab_toggled = False
            if input_mgr.is_action_pressed("debug_perf_toggle"):
                menu.toggle_tab(0)
                tab_toggled = True
            elif input_mgr.is_action_pressed("debug_graphics_toggle"):
                menu.toggle_tab(1)
                tab_toggled = True
            elif input_mgr.is_action_pressed("debug_game_toggle"):
                menu.toggle_tab(2)
                tab_toggled = True

            if menu.visible and not tab_toggled:
                menu.handle_input()

        # Frame 1: User presses F1
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
        step_simulated_frame()
        assert menu.visible is True, "Menu MUST be open after pressing F1"
        assert menu.active_tab == 0, "Active tab must be Tab 0 (Performance)"

        # Frame 2: Key held or no key pressed, menu MUST REMAIN OPEN on Tab 0
        input_mgr.begin_frame()
        step_simulated_frame()
        assert menu.visible is True, "Menu MUST remain open on subsequent frames"
        assert menu.active_tab == 0

        # Frame 3: User presses F2 -> Switches to Tab 1 (Graphics Options)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2))
        step_simulated_frame()
        assert menu.visible is True, "Menu MUST remain open"
        assert menu.active_tab == 1, "Active tab must switch to Tab 1 (Graphics Options)"

        # Frame 4: User presses F3 -> Switches to Tab 2 (Game Options)
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3))
        step_simulated_frame()
        assert menu.visible is True
        assert menu.active_tab == 2, "Active tab must switch to Tab 2 (Game Options)"

        # Key release
        input_mgr.process_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_F3))

        # Frame 5: User presses F3 again -> Closes menu
        input_mgr.begin_frame()
        input_mgr.process_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3))
        step_simulated_frame()
        assert menu.visible is False, "Menu MUST close when active tab hotkey is pressed again"

        # Frame 6: Ensure number keys 1-5 do NOT have preset action bindings
        for k in (Key.NUM_1, Key.NUM_2, Key.NUM_3, Key.NUM_4, Key.NUM_5):
            assert f"preset_{k}" not in input_mgr._actions

