"""Unit tests for the dedicated 3-Tiered Debug System."""

import time
from engine.debug import (
    SystemMonitor,
    EngineTweaks,
    GBufferDebugMode,
    GameTweaks,
    DebugToast,
    DebugDraw,
)


class TestDebugSystem:
    """Comprehensive test suite for the PyMordial 3-Tiered Debug Subsystem."""

    def test_tier1_system_monitor_and_profiler(self):
        """Validates CPU stage microsecond profiling and rolling metrics."""
        monitor = SystemMonitor(history_size=30)

        # Profile a microsecond scope
        with monitor.scope("test_stage"):
            time.sleep(0.002)  # 2 ms

        assert "test_stage" in monitor.stage_times_us
        assert monitor.stage_times_us["test_stage"] >= 1500.0  # At least 1.5 ms in microseconds
        assert monitor.stage_averages_us["test_stage"] > 0.0

        # Simulate several frames
        for _ in range(15):
            monitor.begin_frame()

        assert monitor.fps > 0.0
        assert monitor.avg_fps > 0.0
        assert monitor.one_percent_low_fps > 0.0
        assert monitor.avg_frame_time_ms > 0.0

        summary = monitor.get_resource_summary()
        assert "fps" in summary
        assert "frame_time_ms" in summary

    def test_tier2_engine_graphics_tweaks(self):
        """Validates runtime graphics presets, tonemappers, and G-Buffer modes."""
        tweaks = EngineTweaks()
        notified_presets = []
        tweaks.on_quality_changed(lambda p: notified_presets.append(p))

        # Cycle preset
        initial = tweaks.quality_preset
        next_p = tweaks.cycle_quality_preset()
        assert next_p != initial
        assert len(notified_presets) == 1
        assert notified_presets[0] == next_p

        # Cycle tonemapper
        t0 = tweaks.tonemap_mode
        t1 = tweaks.cycle_tonemapper()
        assert t1 != t0
        assert t1 in ("ACES", "AgX", "Reinhard")

        # Cycle G-Buffer visualizer modes
        g0 = tweaks.gbuffer_debug
        g1 = tweaks.cycle_gbuffer_debug()
        assert g1 != g0
        assert isinstance(g1, GBufferDebugMode)

        # Shadow & Contact Shadow properties
        assert tweaks.shadow_resolution in (1024, 2048, 4096)
        assert tweaks.shadow_mode in ("HARD", "PCF", "PCSS")
        assert tweaks.shadow_softness > 0.0
        assert tweaks.shadow_bias > 0.0
        assert tweaks.sscs_enabled is False
        assert tweaks.sscs_thickness > 0.0
        assert tweaks.sscs_ray_distance > 0.0
        assert tweaks.sscs_max_distance > 0.0

        tweaks.shadow_mode = "HARD"
        tweaks.shadow_resolution = 1024
        assert tweaks.shadow_mode == "HARD"
        assert tweaks.shadow_resolution == 1024

    def test_tier3_game_developer_tweaks(self):
        """Validates declarative registration of dev sliders, booleans, triggers, and watches."""
        gt = GameTweaks()

        # 1. Boolean Tweak
        b_item = gt.add_bool(category="Player", name="God Mode", default=False)
        assert b_item.value is False
        b_item.set_value(True)
        assert gt.get_value("Player", "God Mode") is True

        # 2. Float Numeric Slider with bounds
        f_item = gt.add_float(category="World", name="Time Scale", default=1.0, min_val=0.1, max_val=5.0, step=0.1)
        f_item.set_value(10.0)  # Beyond max
        assert gt.get_value("World", "Time Scale") == 5.0  # Clamped to max

        # 3. Action Trigger
        action_ran = []
        gt.add_action(category="Spawning", name="Spawn Boss", callback=lambda: action_ran.append(True))
        gt.trigger_action("Spawning", "Spawn Boss")
        assert len(action_ran) == 1

        # 4. Live Watch Expression
        test_val = [42]
        gt.add_watch(category="Diagnostics", name="Live Val", getter=lambda: test_val[0])
        assert gt.get_value("Diagnostics", "Live Val") == 42
        test_val[0] = 99
        assert gt.get_value("Diagnostics", "Live Val") == 99

        # Categories query
        cats = gt.get_categories()
        assert "Player" in cats
        assert "World" in cats
        assert "Spawning" in cats
        assert "Diagnostics" in cats

    def test_debug_toast_queue(self):
        """Validates on-screen toast notification lifecycle and expiration."""
        toast = DebugToast()
        toast.show("Saved Snapshot", duration=0.1, color=(50, 255, 50))
        toast.show("Warning Message", duration=2.0, color=(255, 50, 50))

        active = toast.get_active()
        assert len(active) == 2
        assert active[0][0] == "Saved Snapshot"

        # Wait for first toast to expire
        time.sleep(0.15)
        active_after = toast.get_active()
        assert len(active_after) == 1
        assert active_after[0][0] == "Warning Message"

        toast.clear()
        assert len(toast.get_active()) == 0

    def test_debug_draw_geometry_generation(self):
        """Validates 3D immediate-mode wireframe generation and persistent decay."""
        drawer = DebugDraw(ctx=None, max_vertices=1024)

        # 1. Draw Line (2 vertices)
        drawer.draw_line((0, 0, 0), (1, 1, 1), color=(1, 0, 0, 1))
        assert drawer.vertex_count == 2

        # 2. Draw Ray (2 vertices)
        drawer.draw_ray((0, 0, 0), (0, 1, 0), length=5.0)
        assert drawer.vertex_count == 4

        # 3. Draw Box (12 edges * 2 = 24 vertices)
        drawer.draw_box((0, 0, 0), (2, 2, 2))
        assert drawer.vertex_count == 28

        # 4. Draw Sphere (3 rings * 16 segments * 2 = 96 vertices)
        drawer.draw_sphere((0, 0, 0), radius=1.0, segments=16)
        assert drawer.vertex_count == 28 + 96

        # 5. Draw Capsule
        prev = drawer.vertex_count
        drawer.draw_capsule((0, 0, 0), radius=0.5, half_height=1.0, segments=12)
        assert drawer.vertex_count > prev

        # 6. Draw Axes (3 axes * 2 = 6 vertices)
        prev_axes = drawer.vertex_count
        drawer.draw_axes((0, 0, 0), scale=2.0)
        assert drawer.vertex_count == prev_axes + 6

        # Test Clear
        drawer.clear()
        assert drawer.vertex_count == 0

        # Test Persistent Line Duration Decay
        drawer.draw_line((0, 0, 0), (1, 0, 0), duration=0.2)
        assert len(drawer.persistent_lines) == 1
        drawer.update(dt=0.1)  # 0.1s remains
        assert len(drawer.persistent_lines) == 1
        assert drawer.vertex_count == 2  # Re-emitted for frame

        drawer.update(dt=0.15)  # Expired
        assert len(drawer.persistent_lines) == 0

    def test_wireframe_and_physics_colliders_toggling(self):
        """Validates wireframe and physics collider toggles on EngineTweaks and RenderConfig."""
        from engine.gfx.quality_presets import RenderConfig

        config = RenderConfig()
        assert config.wireframe is False

        tweaks = EngineTweaks()
        assert tweaks.show_wireframe is False
        assert tweaks.show_physics_colliders is False

        # Toggle wireframe
        w1 = tweaks.toggle_wireframe()
        assert w1 is True
        assert tweaks.show_wireframe is True

        w2 = tweaks.toggle_wireframe()
        assert w2 is False
        assert tweaks.show_wireframe is False

        # Toggle physics colliders
        p1 = tweaks.toggle_physics_colliders()
        assert p1 is True
        assert tweaks.show_physics_colliders is True

        p2 = tweaks.toggle_physics_colliders()
        assert p2 is False
        assert tweaks.show_physics_colliders is False

        # Validate that DebugDraw per-frame vertices reset when toggled off
        drawer = DebugDraw(ctx=None)
        drawer.update(0.016)
        tweaks.show_physics_colliders = True
        if tweaks.show_physics_colliders:
            drawer.draw_box((0, 0, 0), (1, 1, 1))
        assert drawer.vertex_count == 24

        # Toggle OFF
        tweaks.toggle_physics_colliders()
        assert tweaks.show_physics_colliders is False
        drawer.update(0.016)
        if tweaks.show_physics_colliders:
            drawer.draw_box((0, 0, 0), (1, 1, 1))
        assert drawer.vertex_count == 0, "Drawer should have 0 vertices when colliders are toggled OFF"

    def test_engine_tweaks_apply_to_pipeline(self):
        """Validates that EngineTweaks.apply_to_pipeline synchronizes all post-processing and optics settings."""
        from engine.gfx.quality_presets import RenderConfig
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        pipeline.config = RenderConfig()
        pipeline.post_process = MagicMock()
        pipeline.csm = MagicMock()
        pipeline.csm.atlas_size = 4096
        pipeline.csm.max_distance = 500.0
        pipeline.csm.cascade_count = 4
        pipeline.water_pass = MagicMock()
        pipeline.particle_pass = MagicMock()
        pipeline.atmosphere = MagicMock()
        pipeline.atmosphere.config = MagicMock()

        tweaks = EngineTweaks()
        tweaks.chromatic_aberration_enabled = False
        tweaks.chromatic_aberration_intensity = 0.018
        tweaks.vignette_enabled = True
        tweaks.vignette_intensity = 0.85
        tweaks.film_grain_enabled = True
        tweaks.film_grain_intensity = 0.12
        tweaks.bloom_enabled = False
        tweaks.bloom_intensity = 0.075
        tweaks.dof_enabled = False
        tweaks.motion_blur_enabled = True
        tweaks.motion_blur_samples = 24
        tweaks.motion_blur_intensity = 2.0
        tweaks.lens_flare_enabled = False
        tweaks.aa_mode = "SMAA_1X"
        tweaks.tonemap_mode = "AgX"
        tweaks.exposure = 1.65

        tweaks.apply_to_pipeline(pipeline)

        assert pipeline.config.chromatic_aberration_enabled is False
        assert pipeline.config.chromatic_aberration_intensity == 0.018
        assert pipeline.config.vignette_enabled is True
        assert pipeline.config.vignette_intensity == 0.85
        assert pipeline.config.film_grain_enabled is True
        assert pipeline.config.film_grain_intensity == 0.12
        assert pipeline.config.bloom_enabled is False
        assert pipeline.config.bloom_intensity == 0.075
        assert pipeline.config.dof_enabled is False
        assert pipeline.config.motion_blur_enabled is True
        assert pipeline.config.motion_blur_samples == 24
        assert pipeline.config.motion_blur_intensity == 2.0
        assert pipeline.config.lens_flare_enabled is False
        assert pipeline.config.aa_mode == "SMAA_1X"
        assert pipeline.config.tonemap_mode == "AgX"
        assert pipeline.config.exposure == 1.65
        assert pipeline.post_process.config is pipeline.config

    def test_engine_tweaks_serialization(self, tmp_path):
        """Validates that EngineTweaks can serialize to/from dict and persist to JSON file."""
        from engine.gfx.quality_presets import GraphicsQuality

        tweaks = EngineTweaks()
        tweaks.quality_preset = GraphicsQuality.CINEMATIC
        tweaks.exposure = 1.85
        tweaks.tonemap_mode = "AgX"
        tweaks.vignette_intensity = 0.95
        tweaks.chromatic_aberration_intensity = 0.015
        tweaks.film_grain_intensity = 0.10
        tweaks.bloom_intensity = 0.09
        tweaks.aa_mode = "SMAA_2X"

        # Serialize to dict
        data = tweaks.to_dict()
        assert isinstance(data, dict)
        assert data["quality_preset"] == "cinematic"
        assert data["exposure"] == 1.85
        assert data["tonemap_mode"] == "AgX"
        assert data["vignette_intensity"] == 0.95
        assert data["aa_mode"] == "SMAA_2X"

        # Persist to file
        cfg_file = tmp_path / "test_graphics.json"
        assert tweaks.save_to_file(cfg_file) is True
        assert cfg_file.is_file()

        # Restore into a fresh EngineTweaks instance
        restored = EngineTweaks()
        assert restored.load_from_file(cfg_file) is True
        assert restored.quality_preset == GraphicsQuality.CINEMATIC
        assert restored.exposure == 1.85
        assert restored.tonemap_mode == "AgX"
        assert restored.vignette_intensity == 0.95
        assert restored.chromatic_aberration_intensity == 0.015
        assert restored.film_grain_intensity == 0.10
        assert restored.bloom_intensity == 0.09
        assert restored.aa_mode == "SMAA_2X"
