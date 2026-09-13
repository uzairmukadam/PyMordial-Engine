"""Validation Playground: Player on an Empty Plane with POM Mud (mud_cracked_dry_03).

A focused, clean sandbox featuring:
- Playable Kinematic Character Controller (KCC) with 3rd-person follow camera & free-fly camera
- High-fidelity Parallax Occlusion Mapping (POM) on a 100m x 100m mud_cracked_dry_03 ground plane
- POM self-shadowing and depth-refinement raymarching
- Bruneton Spherical Atmospheric Scattering with dynamic day-night cycle
- 4-Cascade Directional CSM Shadows
- Full Dear ImGui F1 (Perf), F2 (Graphics), F3 (Game Tweaks), F9 (Mouse Grab), and F10 (Camera Toggle)
"""

from __future__ import annotations
import argparse
import math
import random
import sys
from pathlib import Path
import pygame
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ruff: noqa: E402
from engine.core.ecs import EntityManager
from engine.core.loop import EngineLoop
from engine.input import (
    InputManager,
    Key,
    MouseButton,
    GamepadButton,
)
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig
from engine.physics.props import InteractivePropManager
from engine.physics.interaction import PhysicsGrabber
from engine.core.character_camera import CharacterCamera, CharacterCameraConfig
from engine.camera import CameraManager, FreeFlyCamera
from engine.assets.mesh_format import PMMesh, build_pm_mesh
from engine.gfx import (
    RenderContext,
    RenderPipeline,
    GraphicsQuality,
    get_quality_preset,
    LoadingScreen,
)
from engine.gfx.texture_atlas import (
    DisplacementMode,
    encode_mat_flags,
    decode_material_folder,
    TextureArrayAtlas,
)
from engine.gfx.mega_buffer import MeshAllocation
from engine.debug import (
    SystemMonitor,
    EngineTweaks,
    GBufferDebugMode,
    GameTweaks,
    DebugToast,
    DebugMenu,
)
from engine.audio import get_audio_engine
from engine.window import VSyncMode


def make_tiled_plane_pm_mesh(size: float = 100.0, uv_tiles: float = 50.0) -> PMMesh:
    """Builds a high-precision 32-byte cache-aligned PMMesh quad with tiled UVs and tangents for POM."""
    half_s = size * 0.5
    pos = np.array(
        [
            [-half_s, 0.0,  half_s],
            [ half_s, 0.0,  half_s],
            [ half_s, 0.0, -half_s],
            [-half_s, 0.0, -half_s],
        ],
        dtype=np.float32,
    )
    nor = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    uvs = np.array(
        [
            [0.0,      0.0],
            [uv_tiles, 0.0],
            [uv_tiles, uv_tiles],
            [0.0,      uv_tiles],
        ],
        dtype=np.float32,
    )
    tan = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return build_pm_mesh(pos, nor, uvs, tan, idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Engine: Player on POM Mud Plane")
    parser.add_argument("--headless", action="store_true", help="Run without window display")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames")
    parser.add_argument("--screenshot", type=str, default="", help="Save screenshot to path")
    parser.add_argument(
        "--preset",
        type=str,
        default="ultra",
        choices=["custom", "low", "medium", "high", "ultra", "cinematic"],
        help="Initial graphics preset (default: ultra)",
    )
    parser.add_argument("--vsync", action="store_true", default=False, help="Enable VSync")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument(
        "--atmo-preset",
        type=str,
        default="earth_day",
        choices=["", "earth_day", "earth_sunset", "earth_night", "alien_cyan_purple", "alien_crimson_mars"],
        help="Physical atmosphere preset (default: earth_day)",
    )
    parser.add_argument("--time-of-day", type=float, default=None, help="Initial time of day (0.0 to 24.0)")
    parser.add_argument("--day-speed", type=float, default=None, help="Day-night orbital speed multiplier")
    parser.add_argument("--pom-scale", type=float, default=1.0, help="POM height scale multiplier (default: 1.0)")
    parser.add_argument("--camera-distance", type=float, default=5.0, help="Initial camera follow distance (default: 5.0)")
    parser.add_argument("--camera-pitch", type=float, default=18.0, help="Initial camera pitch angle in degrees (default: 18.0)")
    parser.add_argument("--camera-yaw", type=float, default=25.0, help="Initial camera yaw angle in degrees (default: 25.0)")
    parser.add_argument("--plane-tiles", type=float, default=50.0, help="Plane UV repetitions (default: 50.0)")
    parser.add_argument(
        "--debug-gbuffer",
        type=str,
        default="",
        choices=["", "disabled", "albedo", "normals", "material", "depth", "shadow_atlas", "shadow_mask", "ao", "ssgi", "lpv", "gi_total", "hiz"],
        help="Initial G-Buffer debug inspection view",
    )
    parser.add_argument("--hiz-mip", type=int, default=0, help="Hi-Z mip level to visualize (0 to 10)")
    parser.add_argument("--wireframe", action="store_true", default=False, help="Enable polygon wireframe rendering")
    parser.add_argument("--show-colliders", action="store_true", default=False, help="Enable 3D physics collider wireframes")
    parser.add_argument("--show-sun-ray", action="store_true", default=False, help="Enable 3D sun ray visualization")
    parser.add_argument("--exposure", type=float, default=None, help="Camera exposure multiplier")
    parser.add_argument("--tonemap", type=str, default=None, choices=["ACES", "AgX", "Reinhard"], help="Tonemapping operator")
    parser.add_argument("--particles", action=argparse.BooleanOptionalAction, default=True, help="Enable volumetric particle system")
    parser.add_argument("--particle-count", type=int, default=16384, help="Particle count (1024 to 65536)")
    parser.add_argument("--particle-mode", type=str, default="DUST_MOTES", choices=["DUST_MOTES", "EMBERS", "FIREFLIES", "OFF"], help="Particle VFX mode")
    parser.add_argument("--dof", action=argparse.BooleanOptionalAction, default=True, help="Enable Bokeh Depth of Field")
    parser.add_argument("--focus-dist", type=float, default=5.0, help="DoF focus distance in meters")
    parser.add_argument("--fstop", type=float, default=2.8, help="DoF aperture f-stop")
    parser.add_argument("--motion-blur", action=argparse.BooleanOptionalAction, default=True, help="Enable Velocity Motion Blur")
    parser.add_argument("--lens-flare", action=argparse.BooleanOptionalAction, default=True, help="Enable Anamorphic Lens Flare")
    parser.add_argument("--chromatic-aberration", action=argparse.BooleanOptionalAction, default=True, help="Enable Chromatic Aberration")
    parser.add_argument("--vignette", action=argparse.BooleanOptionalAction, default=True, help="Enable Physical Lens Vignette")
    parser.add_argument("--film-grain", action=argparse.BooleanOptionalAction, default=True, help="Enable Filmic 35mm Grain")
    parser.add_argument("--water", action=argparse.BooleanOptionalAction, default=True, help="Enable Dynamic Water & Screen-Space Refraction")
    parser.add_argument("--water-height", type=float, default=0.0, help="Water surface height in meters")
    parser.add_argument("--water-waves", type=float, default=0.15, help="Water wave amplitude in meters")
    parser.add_argument("--water-speed", type=float, default=1.0, help="Water wave speed multiplier")
    parser.add_argument("--water-clarity", type=float, default=4.0, help="Water optical depth clarity scale")
    args = parser.parse_args()

    width, height = args.width, args.height

    # 1. Initialize Window & ModernGL Render Context
    initial_preset_enum = GraphicsQuality(args.preset.lower())
    render_ctx = RenderContext(
        width=width,
        height=height,
        title="PyMordial Engine — POM Mud Plane Validation",
        hidden=args.headless,
        config=get_quality_preset(initial_preset_enum),
    )
    render_ctx.window.set_vsync(VSyncMode.ON if args.vsync else VSyncMode.OFF)

    loading_screen = LoadingScreen(
        ctx=render_ctx.ctx,
        width=width,
        height=height,
        is_headless=args.headless,
    )
    loading_screen.set_progress(0.15, status="Initializing ModernGL 4.5 Deferred Pipeline...")
    loading_screen.update(0.016)
    loading_screen.render(render_ctx=render_ctx)

    pipeline = RenderPipeline(render_ctx, get_quality_preset(initial_preset_enum))
    pipeline.apply_config(pipeline.config)

    # 2. Physics & ECS Architecture
    ecs = EntityManager(max_entities=500)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-20.0, gravity_z=0.0)
    input_mgr = InputManager()
    if not args.headless:
        input_mgr.set_mouse_grab(True)
        input_mgr._was_mouse_grabbed = True
    loop = EngineLoop(ecs=ecs, input_manager=input_mgr, fixed_dt=1.0 / 60.0)

    # 3. Load 4K PBR Texture Arrays for mud_cracked_dry_03 (Fast-path binary streaming from .pm_tex)
    loading_screen.set_progress(0.40, status="Loading 4K PBR Mud Material Arrays...")
    loading_screen.update(0.016)
    loading_screen.render(render_ctx=render_ctx)

    raw_mud_dir = ROOT_DIR / "assets" / "textures" / "mud_cracked_dry_03"
    cooked_mud_dir = ROOT_DIR / "build" / "cooked_assets" / "textures" / "mud_cracked_dry_03"
    target_res = 4096
    if pipeline.texture_atlas.width != target_res or pipeline.texture_atlas.height != target_res:
        pipeline.texture_atlas.destroy()
        pipeline.texture_atlas = TextureArrayAtlas(pipeline.ctx, width=target_res, height=target_res, max_layers=32)

    decoded = decode_material_folder(
        folder=raw_mud_dir,
        width=target_res,
        height=target_res,
        name="mud_cracked_dry_03",
        cooked_folder=cooked_mud_dir if cooked_mud_dir.is_dir() else None,
    )
    l_mud = pipeline.texture_atlas.upload_decoded_layer(decoded, rebuild_mipmaps=True)
    print(f"Loaded 'mud_cracked_dry_03' into atlas layer {l_mud} with displacement depth {pipeline.texture_atlas.material_depths[l_mud]:.3f}m")

    # 4. Register Tiled Ground Mesh in MegaBuffer
    plane_size = 100.0
    plane_tiles = float(args.plane_tiles)  # 1 repetition per 2.0 meters by default (50.0)
    tiled_plane_pm = make_tiled_plane_pm_mesh(size=plane_size, uv_tiles=plane_tiles)
    alloc_plane = pipeline.mega_buffer.add_pm_mesh("pom_plane", tiled_plane_pm)
    alloc_capsule = pipeline.mega_buffer.allocations["capsule"]

    # Rebind pipeline VAOs with the newly baked plane mesh
    pipeline.csm_vao = pipeline.mega_buffer.get_vao(pipeline.csm_prog)
    pipeline.gbuffer_vao = pipeline.mega_buffer.get_vao(pipeline.gbuffer_prog)
    pipeline.gbuffer_tess_vao = pipeline.mega_buffer.get_vao(pipeline.gbuffer_tess_prog, mode=pipeline.ctx.PATCHES)
    pipeline.csm_tess_vao = pipeline.mega_buffer.get_vao(pipeline.csm_tess_prog, mode=pipeline.ctx.PATCHES)

    # 5. Spawn Empty POM Ground Plane
    plane_id = ecs.create_entity(
        position=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        color=(0.65, 0.55, 0.42),
        roughness=0.85,
        metallic=0.02,
    )
    physics.create_body(plane_id, body_type="fixed", position=(0.0, -0.5, 0.0))
    physics.attach_box_collider(plane_id, half_x=plane_size * 0.5, half_y=0.5, half_z=plane_size * 0.5, friction=0.80, restitution=0.05)

    d_plane = ecs.pool.get_dense_index(plane_id)
    ecs.material_data[d_plane, 6] = float(l_mud)
    ecs.material_data[d_plane, 7] = encode_mat_flags(has_texture=(l_mud > 0), disp_mode=DisplacementMode.POM)

    # 6. Spawn Playable Character
    char_start_pos = (0.0, 0.95, 0.0)
    char_id = ecs.create_entity(
        position=char_start_pos,
        scale=(1.0, 1.0, 1.0),
        color=(0.15, 0.85, 0.95),
        roughness=0.20,
        metallic=0.85,
    )
    d_char = ecs.pool.get_dense_index(char_id)

    motor = CharacterMotor(
        entity_id=char_id,
        physics=physics,
        ecs=ecs,
        config=CharacterMotorConfig(
            walk_speed=6.0,
            run_speed=11.5,
            jump_force=8.5,
            gravity=-20.0,
            step_height=0.30,
        ),
        initial_position=char_start_pos,
    )

    # 7. Virtual Camera Stack (3rd-Person Follow + Free-Fly)
    audio_engine = get_audio_engine()
    camera_mgr = CameraManager()

    def on_character_footstep(pos: tuple[float, float, float], in_water: bool) -> None:
        cue_name = "footstep_water" if in_water else "footstep"
        vol = 0.70 if is_sprinting else 0.45
        audio_engine.play_sound(cue_name, position=pos, volume=vol)

    motor.on_footstep = on_character_footstep

    # 7.5 Interactive Props Subsystem & Opt-in Physics Grabber
    prop_mgr = InteractivePropManager(physics=physics, ecs=ecs, audio=audio_engine)
    grabber = PhysicsGrabber(physics=physics, prop_manager=prop_mgr, audio=audio_engine)

    cam = CharacterCamera(
        config=CharacterCameraConfig(
            distance=float(args.camera_distance),
            target_offset_y=1.20,
            camera_radius=0.25,
            yaw_sensitivity=0.25,
            pitch_sensitivity=0.25,
        ),
        initial_yaw_deg=float(args.camera_yaw),
        initial_pitch_deg=float(args.camera_pitch),
    )
    camera_mgr.register_camera("follow", cam, make_active=True)

    free_cam = FreeFlyCamera(position=(0.0, 3.5, 6.0), fov=70.0)
    camera_mgr.register_camera("free", free_cam)

    # 8. Standardized 3-Tier Debug System
    monitor = SystemMonitor(history_size=120)
    engine_tweaks = EngineTweaks()
    engine_tweaks.quality_preset = initial_preset_enum
    engine_tweaks.vsync_enabled = args.vsync
    engine_tweaks.pom_enabled = True
    engine_tweaks.pom_height_scale = float(args.pom_scale)
    engine_tweaks.pom_self_shadow = True
    engine_tweaks.disp_near_radius = 25.0
    engine_tweaks.disp_mid_radius = 60.0

    if args.atmo_preset:
        engine_tweaks.apply_atmo_preset(args.atmo_preset)
    if args.time_of_day is not None:
        engine_tweaks.time_of_day = float(args.time_of_day)
    if args.day_speed is not None:
        engine_tweaks.day_speed = float(args.day_speed)
    if args.debug_gbuffer:
        engine_tweaks.gbuffer_debug = GBufferDebugMode[args.debug_gbuffer.upper()]
    if getattr(args, "hiz_mip", None) is not None:
        engine_tweaks.hiz_debug_mip = int(args.hiz_mip)
    if args.wireframe:
        engine_tweaks.show_wireframe = True
    if args.show_colliders:
        engine_tweaks.show_physics_colliders = True
    if args.show_sun_ray:
        engine_tweaks.show_sun_ray = True
    if args.exposure is not None:
        engine_tweaks.exposure = float(args.exposure)
    if args.tonemap:
        engine_tweaks.tonemap_mode = args.tonemap
    engine_tweaks.particles_enabled = args.particles
    engine_tweaks.particle_count = args.particle_count
    engine_tweaks.particle_mode = args.particle_mode
    engine_tweaks.dof_enabled = args.dof
    engine_tweaks.dof_focus_distance = float(args.focus_dist)
    engine_tweaks.dof_aperture = float(args.fstop)
    engine_tweaks.motion_blur_enabled = args.motion_blur
    engine_tweaks.lens_flare_enabled = args.lens_flare
    engine_tweaks.chromatic_aberration_enabled = args.chromatic_aberration
    engine_tweaks.vignette_enabled = args.vignette
    engine_tweaks.film_grain_enabled = args.film_grain
    if args.water is not None:
        engine_tweaks.water_enabled = args.water
    if args.water_height is not None:
        engine_tweaks.water_height = float(args.water_height)
    if args.water_waves is not None:
        engine_tweaks.water_wave_amplitude = float(args.water_waves)
    if args.water_speed is not None:
        engine_tweaks.water_wave_speed = float(args.water_speed)
    if args.water_clarity is not None:
        engine_tweaks.water_clarity = float(args.water_clarity)

    game_tweaks = GameTweaks()
    toast = DebugToast()

    debug_menu = DebugMenu(
        ctx=render_ctx.ctx,
        monitor=monitor,
        engine_tweaks=engine_tweaks,
        game_tweaks=game_tweaks,
        toast=toast,
        input_mgr=input_mgr,
        screen_width=width,
        screen_height=height,
    )

    # Input Bindings
    input_mgr.bind_action("jump", keys=[Key.SPACE], gamepad_buttons=[GamepadButton.A])
    input_mgr.bind_action("sprint", keys=[Key.LSHIFT, Key.RSHIFT], gamepad_buttons=[GamepadButton.LEFT_STICK])
    input_mgr.bind_action("reset_player", keys=[Key.R], gamepad_buttons=[GamepadButton.Y])
    input_mgr.bind_action("reset_camera", keys=[Key.Z], gamepad_buttons=[GamepadButton.RIGHT_STICK])
    input_mgr.bind_action("toggle_camera", keys=[Key.F10], is_debug=True)
    input_mgr.bind_action("debug_perf", keys=[Key.F1, Key.GRAVE], is_debug=True)
    input_mgr.bind_action("debug_graphics", keys=[Key.F2], is_debug=True)
    input_mgr.bind_action("debug_game", keys=[Key.F3], is_debug=True)
    input_mgr.bind_action("debug_mouse_capture", keys=[Key.F9], is_debug=True)
    input_mgr.bind_action("zoom_in", keys=[Key.Q])
    input_mgr.bind_action("zoom_out", keys=[Key.E])
    input_mgr.bind_action("quit", keys=[Key.ESCAPE], gamepad_buttons=[GamepadButton.START])

    def on_quality_changed(new_preset: GraphicsQuality) -> None:
        if new_preset == GraphicsQuality.CUSTOM:
            return
        cfg = get_quality_preset(new_preset)
        cfg.wireframe = engine_tweaks.show_wireframe
        pipeline.apply_config(cfg)
        engine_tweaks.shadow_resolution = cfg.shadow_resolution
        engine_tweaks.shadow_mode = cfg.shadow_mode
        engine_tweaks.shadow_softness = cfg.shadow_softness
        engine_tweaks.shadow_bias = cfg.shadow_bias
        engine_tweaks.shadow_distance = cfg.shadow_distance
        engine_tweaks.csm_cascades = cfg.csm_cascades
        engine_tweaks.sscs_enabled = cfg.sscs_enabled
        engine_tweaks.sscs_steps = cfg.sscs_steps
        engine_tweaks.sscs_thickness = cfg.sscs_thickness
        engine_tweaks.sscs_ray_distance = cfg.sscs_ray_distance
        engine_tweaks.sscs_max_distance = cfg.sscs_max_distance
        engine_tweaks.ao_mode = cfg.ao_mode
        engine_tweaks.gi_mode = cfg.gi_mode
        engine_tweaks.ibl_enabled = cfg.ibl_enabled
        engine_tweaks.ssr_enabled = cfg.ssr_enabled
        engine_tweaks.aa_mode = cfg.aa_mode
        engine_tweaks.taa_enabled = cfg.taa_enabled
        engine_tweaks.point_lights_enabled = cfg.clustered_lights_enabled
        engine_tweaks.volumetric_fog_enabled = getattr(cfg, "volumetric_fog_enabled", True)
        engine_tweaks.fog_resolution = getattr(cfg, "fog_resolution", "HIGH")
        engine_tweaks.fog_point_lights = getattr(cfg, "fog_point_lights", True)
        engine_tweaks.fog_density = getattr(cfg, "fog_density", 0.005)
        engine_tweaks.fog_height_falloff = getattr(cfg, "fog_height_falloff", 0.10)
        engine_tweaks.fog_anisotropy = getattr(cfg, "fog_anisotropy", 0.65)
        engine_tweaks.fog_distance = getattr(cfg, "fog_distance", 400.0)
        engine_tweaks.fog_ambient = getattr(cfg, "fog_ambient", 0.35)
        engine_tweaks.fog_debug_mode = getattr(cfg, "fog_debug_mode", 0)
        engine_tweaks.particles_enabled = getattr(cfg, "particles_enabled", True)
        engine_tweaks.particle_count = getattr(cfg, "particle_count", 16384)
        engine_tweaks.particle_mode = getattr(cfg, "particle_mode", "DUST_MOTES")
        engine_tweaks.particle_size = getattr(cfg, "particle_size", 0.04)
        engine_tweaks.particle_turbulence = getattr(cfg, "particle_turbulence", 1.0)
        engine_tweaks.dof_enabled = getattr(cfg, "dof_enabled", True)
        engine_tweaks.dof_bokeh_shape = getattr(cfg, "dof_bokeh_shape", "CIRCULAR")
        engine_tweaks.motion_blur_enabled = getattr(cfg, "motion_blur_enabled", True)
        engine_tweaks.motion_blur_samples = getattr(cfg, "motion_blur_samples", 12)
        engine_tweaks.lens_flare_enabled = getattr(cfg, "lens_flare_enabled", True)
        engine_tweaks.chromatic_aberration_enabled = getattr(cfg, "chromatic_aberration_enabled", True)
        engine_tweaks.chromatic_aberration_intensity = getattr(cfg, "chromatic_aberration_intensity", 0.005)
        engine_tweaks.vignette_enabled = getattr(cfg, "vignette_enabled", True)
        engine_tweaks.film_grain_enabled = getattr(cfg, "film_grain_enabled", True)
        engine_tweaks.water_enabled = getattr(cfg, "water_enabled", True)
        engine_tweaks.water_wave_amplitude = getattr(cfg, "water_wave_amplitude", 0.15)
        engine_tweaks.water_refraction_enabled = getattr(cfg, "water_refraction_enabled", True)
        engine_tweaks.water_foam_enabled = getattr(cfg, "water_foam_enabled", True)
        toast.show(f"Quality Preset: {new_preset.value.upper()}", duration=2.5, color=(56, 189, 248))

    engine_tweaks.on_quality_changed(on_quality_changed)
    engine_tweaks.on_vsync_changed(lambda enabled: render_ctx.window.set_vsync(VSyncMode.ON if enabled else VSyncMode.OFF))

    # Game Tweaks
    game_tweaks.add_action("Character", "Reset to Origin", lambda: motor.teleport(char_start_pos))
    game_tweaks.add_action("Camera", "Toggle Follow/Free (F10)", lambda: camera_mgr.switch_to("free" if camera_mgr.active_camera_name == "follow" else "follow"))
    game_tweaks.add_action("Render", "Toggle Wireframe", lambda: setattr(engine_tweaks, "show_wireframe", not engine_tweaks.show_wireframe))
    game_tweaks.add_action("Render", "Toggle Colliders", lambda: setattr(engine_tweaks, "show_physics_colliders", not engine_tweaks.show_physics_colliders))
    game_tweaks.add_action("Render", "Toggle Sun Ray", lambda: setattr(engine_tweaks, "show_sun_ray", not engine_tweaks.show_sun_ray))
    game_tweaks.add_watch("POM", "Height Scale", lambda: f"{engine_tweaks.pom_height_scale:.2f}x")
    game_tweaks.add_watch("Time", "Hour", lambda: f"{engine_tweaks.time_of_day:.2f} h")
    game_tweaks.add_watch("Sun", "Elevation", lambda: f"{engine_tweaks.sun_elevation_deg:.1f}°")
    game_tweaks.add_watch("Sun", "Azimuth", lambda: f"{engine_tweaks.sun_angle_deg:.1f}°")
    game_tweaks.add_watch("G-Buffer", "Mode", lambda: engine_tweaks.gbuffer_debug.name)
    game_tweaks.add_watch("Wireframe", "Active", lambda: engine_tweaks.show_wireframe)
    game_tweaks.add_watch("VFX", "Particle Mode", lambda: engine_tweaks.particle_mode)
    game_tweaks.add_watch("VFX", "Particles Active", lambda: f"{engine_tweaks.particle_count:,}" if engine_tweaks.particles_enabled else "OFF")
    game_tweaks.add_watch("Optics", "DoF", lambda: f"{engine_tweaks.dof_focus_distance:.1f}m (f/{engine_tweaks.dof_aperture:.1f})" if engine_tweaks.dof_enabled else "OFF")
    game_tweaks.add_watch("Optics", "Motion Blur", lambda: f"{engine_tweaks.motion_blur_intensity:.1f}x" if engine_tweaks.motion_blur_enabled else "OFF")
    game_tweaks.add_watch("Optics", "Lens Flare", lambda: "ON" if engine_tweaks.lens_flare_enabled else "OFF")
    game_tweaks.add_watch("Water", "Active", lambda: "ON" if engine_tweaks.water_enabled else "OFF")
    game_tweaks.add_watch("Water", "Height", lambda: f"{engine_tweaks.water_height:.2f}m")
    game_tweaks.add_watch("Water", "Wave Amp", lambda: f"{engine_tweaks.water_wave_amplitude:.2f}m")

    # Physics & Interactive Props Game Tweaks
    game_tweaks.add_action("Physics & Props", "Spawn Crate (Ahead)", lambda: (
        prop_mgr.spawn_crate(
            position=(
                camera_pos[0] + scratch_fwd_vec[0] * 3.5,
                max(1.0, camera_pos[1] + scratch_fwd_vec[1] * 3.5),
                camera_pos[2] + scratch_fwd_vec[2] * 3.5,
            ),
            size=0.9,
            health=40.0,
        ),
        toast.show("Spawned Dynamic Crate", duration=1.5, color=(74, 222, 128)),
    ))
    game_tweaks.add_action("Physics & Props", "Spawn Pyramid (3x3)", lambda: (
        prop_mgr.spawn_pyramid(
            base_center=(
                char_world_pos[0] - math.sin(math.radians(cam.yaw_deg)) * 4.5,
                0.45,
                char_world_pos[2] - math.cos(math.radians(cam.yaw_deg)) * 4.5,
            ),
            rows=3,
            box_size=0.8,
        ),
        toast.show("Spawned Crate Pyramid", duration=1.5, color=(74, 222, 128)),
    ))
    game_tweaks.add_action("Physics & Props", "Spawn 4 Spheres", lambda: (
        [
            prop_mgr.spawn_sphere(
                position=(
                    char_world_pos[0] + random.uniform(-2.0, 2.0),
                    char_world_pos[1] + 2.5 + i * 0.8,
                    char_world_pos[2] + random.uniform(2.5, 4.5),
                ),
                radius=random.uniform(0.35, 0.55),
                color=(random.uniform(0.3, 0.95), random.uniform(0.3, 0.95), random.uniform(0.3, 0.95)),
            )
            for i in range(4)
        ],
        toast.show("Spawned 4 Tumbling Spheres", duration=1.5, color=(74, 222, 128)),
    ))
    game_tweaks.add_action("Physics & Props", "Clear All Props", lambda: (
        grabber.release_held(),
        prop_mgr.clear_all_props(),
        toast.show("Cleared Dynamic Props", duration=1.5, color=(248, 113, 113)),
    ))
    game_tweaks.add_float(
        "Physics & Props",
        "Gravity Y",
        default=-20.0,
        min_val=-35.0,
        max_val=0.0,
        step=1.0,
        on_changed=lambda val: (physics.set_gravity(0.0, val, 0.0), setattr(motor.config, "gravity", val)),
    )
    game_tweaks.add_float(
        "Physics & Props",
        "Throw Speed",
        default=11.0,
        min_val=3.0,
        max_val=35.0,
        step=1.0,
        on_changed=lambda val: setattr(grabber, "throw_force", val),
    )
    game_tweaks.add_float(
        "Physics & Props",
        "Grab Reach",
        default=3.8,
        min_val=1.5,
        max_val=8.0,
        step=0.2,
        on_changed=lambda val: setattr(grabber, "max_reach", val),
    )
    game_tweaks.add_watch("Physics & Props", "Props Active", lambda: f"{prop_mgr.active_props_count}")
    game_tweaks.add_watch("Physics & Props", "Holding Prop", lambda: f"Entity {grabber.held_entity_id}" if grabber.is_holding else "None")
    game_tweaks.add_watch("Physics & Props", "Last Event", lambda: prop_mgr.last_impact_info)

    toast.show("POM Mud Plane Ready — WASD Move, Shift Sprint, Space Jump, E Grab/Drop, L-Click Throw/Hit", duration=4.5, color=(56, 189, 248))

    # Finalize Loading Screen & Fade Transition
    loading_screen.set_progress(1.0, status="Ready. Launching POM Mud Plane...")
    for _ in range(3):
        loading_screen.update(0.016)
        loading_screen.render(render_ctx=render_ctx)
    loading_screen.fade_out(duration=0.25, render_ctx=render_ctx)
    loading_screen.destroy()

    # Zero-allocation MDI batch: [Plane, Character Capsule]
    draw_batches: list[tuple[MeshAllocation, int, int, bool]] = [
        (alloc_plane, 1, d_plane, False),
        (alloc_capsule, 1, d_char, False),
    ]

    alloc_cube = pipeline.mega_buffer.allocations["cube"]
    alloc_sphere = pipeline.mega_buffer.allocations["sphere"]

    # Initial demonstration dynamic props
    prop_mgr.spawn_pyramid(base_center=(0.0, 0.45, 4.0), rows=3, box_size=0.8)
    prop_mgr.spawn_sphere(position=(-3.0, 1.5, 3.0), radius=0.45, color=(0.95, 0.35, 0.25))
    prop_mgr.spawn_sphere(position=(3.0, 1.5, 3.0), radius=0.45, color=(0.25, 0.85, 0.45))
    prop_mgr.spawn_barrel(position=(-2.5, 1.0, 5.0), color=(0.35, 0.55, 0.85))
    prop_mgr.spawn_barrel(position=(2.5, 1.0, 5.0), color=(0.85, 0.65, 0.25))

    init_sun = pipeline.atmosphere.compute_sun_vector()
    scratch_sun_dir = [float(init_sun[0]), float(init_sun[1]), float(init_sun[2])]
    scratch_sun_tuple = (scratch_sun_dir[0], scratch_sun_dir[1], scratch_sun_dir[2])
    scratch_fwd_vec = [0.0, 0.0, 0.0]
    scratch_cam_pos = [0.0, 0.0, 0.0]

    engine_tweaks.sun_elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, -init_sun[1]))))
    engine_tweaks.sun_angle_deg = math.degrees(math.atan2(init_sun[0], init_sun[2]))

    prev_tod = engine_tweaks.time_of_day
    prev_sun_el = engine_tweaks.sun_elevation_deg
    prev_sun_az = engine_tweaks.sun_angle_deg

    # Simulation State
    move_fwd = 0.0
    move_strafe = 0.0
    is_sprinting = False
    jump_requested = False
    char_world_pos = (char_start_pos[0], char_start_pos[1], char_start_pos[2])
    camera_pos = (0.0, 3.5, 6.0)
    camera_target = (0.0, 1.0, 0.0)

    def fixed_update(dt: float) -> None:
        nonlocal jump_requested
        with monitor.scope("kcc_motor"):
            motor.update(
                dt=dt,
                move_input=(move_fwd, move_strafe),
                is_running=is_sprinting,
                jump_requested=jump_requested,
                camera_yaw_deg=cam.yaw_deg,
            )
            jump_requested = False

        # Character to dynamic props impulse transfer (kicking & shoving)
        try:
            t = physics.world.get_transform(char_id)
            c_pos = (t[0], t[1], t[2])
        except Exception:
            c_pos = (0.0, 1.0, 0.0)

        prop_mgr.push_props_near_character(
            character_pos=c_pos,
            character_velocity=motor.state.velocity,
            push_radius=None,
            push_force=6.0 if is_sprinting else 3.5,
        )

        with monitor.scope("physics_step"):
            physics.step_simulation(dt)

        with monitor.scope("ecs_sync"):
            physics.sync_to_ecs(ecs)

        # Update prop lifecycles and held prop spring tracking
        prop_mgr.update(dt)
        grabber.update(dt, camera_pos=scratch_cam_pos, camera_forward=scratch_fwd_vec)

    loop.on_fixed_update = fixed_update

    clock = pygame.time.Clock()
    running = True
    frame_idx = 0

    # Main Simulation & Rendering Loop
    while running:
        raw_dt_ms = clock.tick(0)
        dt = min(raw_dt_ms * 0.001, 0.1)

        monitor.begin_frame()

        # Event Processing
        if not args.headless:
            events = input_mgr.poll_events()
            if not input_mgr.is_mouse_grabbed and debug_menu.visible:
                for ev in events:
                    debug_menu.process_event(ev)
        else:
            input_mgr.begin_frame()
            input_mgr.update_axes()

        if input_mgr.should_quit:
            running = False
            break
        if input_mgr.is_action_pressed("quit"):
            if debug_menu.visible:
                debug_menu.toggle()
                toast.show("Debug Menu Closed", duration=1.5, color=(147, 197, 253))
            else:
                running = False
                break

        # Mouse Grab / Release (F9)
        if input_mgr.is_action_pressed("debug_mouse_capture"):
            new_grab = not input_mgr.is_mouse_grabbed
            input_mgr.set_mouse_grab(new_grab)
            input_mgr._was_mouse_grabbed = new_grab
            msg = "Mouse Captured (Gameplay Control)" if new_grab else "Mouse Released (Debug Panel Free)"
            toast.show(msg, duration=1.5, color=(147, 197, 253))

        # Panel Toggles
        if input_mgr.is_action_pressed("debug_perf"):
            debug_menu.cycle_f1()
            if debug_menu.f1_style > 0 and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
        elif input_mgr.is_action_pressed("debug_graphics"):
            is_open = debug_menu.toggle_graphics()
            if is_open and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
        elif input_mgr.is_action_pressed("debug_game"):
            is_open = debug_menu.toggle_game_tweaks()
            if is_open and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
        elif input_mgr.is_action_pressed("toggle_camera"):
            next_cam = "free" if camera_mgr.active_camera_name == "follow" else "follow"
            camera_mgr.blend_to(next_cam, 0.5)
            toast.show(f"Active Camera: {next_cam.upper()}", duration=1.5, color=(74, 222, 128))

        # Character & Camera Controls
        if not args.headless:
            move_vec = input_mgr.get_vector2("move")
            move_strafe = move_vec[0]
            move_fwd = move_vec[1]
            is_sprinting = input_mgr.is_action_down("sprint")

            if input_mgr.is_action_pressed("jump"):
                jump_requested = True
                if motor.state.is_grounded:
                    audio_engine.play_sound("jump", position=char_start_pos, volume=0.8)

            # Pick-and-Carry (E interact)
            if input_mgr.is_action_pressed("interact"):
                if grabber.is_holding:
                    grabber.release_held()
                    toast.show("Dropped Prop", duration=1.2, color=(147, 197, 253))
                else:
                    grabbed = grabber.try_grab(origin=camera_pos, direction=scratch_fwd_vec, ignore_entity_id=char_id)
                    if grabbed:
                        toast.show("Grabbed Prop (E Drop, L-Click Throw)", duration=2.0, color=(74, 222, 128))
                    else:
                        char_eye = (char_world_pos[0], char_world_pos[1] + 0.6, char_world_pos[2])
                        yaw_r = math.radians(cam.yaw_deg)
                        cam_fwd_h = (-math.sin(yaw_r), 0.0, -math.cos(yaw_r))
                        if grabber.try_grab(origin=char_eye, direction=cam_fwd_h, ignore_entity_id=char_id):
                            toast.show("Grabbed Prop (E Drop, L-Click Throw)", duration=2.0, color=(74, 222, 128))

            # Throw or punch/hit (Left Click)
            if input_mgr.is_action_pressed("primary_action"):
                if grabber.is_holding:
                    grabber.throw_held(direction=scratch_fwd_vec)
                    toast.show("Prop Thrown!", duration=1.5, color=(251, 146, 60))
                else:
                    # Raycast forward to damage/shatter destructibles or apply hit impulse
                    punch_orig = camera_pos
                    hit = physics.raycast(punch_orig, scratch_fwd_vec, max_distance=4.5)
                    if hit is not None and hit[0] == char_id:
                        advance = hit[1] + 0.5
                        punch_orig = (
                            camera_pos[0] + scratch_fwd_vec[0] * advance,
                            camera_pos[1] + scratch_fwd_vec[1] * advance,
                            camera_pos[2] + scratch_fwd_vec[2] * advance,
                        )
                        hit = physics.raycast(punch_orig, scratch_fwd_vec, max_distance=3.5)

                    if hit is not None and prop_mgr.is_prop(hit[0]):
                        hit_id, hit_dist, _ = hit
                        hit_pt = (
                            punch_orig[0] + scratch_fwd_vec[0] * hit_dist,
                            punch_orig[1] + scratch_fwd_vec[1] * hit_dist,
                            punch_orig[2] + scratch_fwd_vec[2] * hit_dist,
                        )
                        destroyed = prop_mgr.damage_prop(hit_id, damage=40.0, hit_point=hit_pt, hit_direction=scratch_fwd_vec)
                        if destroyed:
                            toast.show("Crate Shattered!", duration=1.5, color=(248, 113, 113))
                        else:
                            physics.apply_impulse(
                                hit_id,
                                (scratch_fwd_vec[0] * 8.0, scratch_fwd_vec[1] * 2.0, scratch_fwd_vec[2] * 8.0)
                            )
                            toast.show("Hit Prop!", duration=1.0, color=(251, 191, 36))

            if input_mgr.is_action_pressed("reset_player"):
                motor.teleport(char_start_pos)
            if input_mgr.is_action_pressed("reset_camera"):
                cam.yaw_deg = 0.0
                cam.pitch_deg = 16.0

            if input_mgr.is_action_down("zoom_in"):
                cam.handle_zoom(12.0 * dt)
            if input_mgr.is_action_down("zoom_out"):
                cam.handle_zoom(-12.0 * dt)

            # Camera Mouse Look / Orbit
            if input_mgr.is_mouse_grabbed or input_mgr.is_mouse_down(MouseButton.RIGHT):
                dx, dy = input_mgr.mouse_delta
                if dx != 0 or dy != 0:
                    if camera_mgr.active_camera_name == "follow":
                        cam.handle_mouse_orbit(dx, dy)
                    else:
                        free_cam.handle_mouse_look(dx, dy)

            if input_mgr.mouse_wheel != 0.0:
                cam.handle_zoom(input_mgr.mouse_wheel)
        else:
            move_fwd = 0.0
            move_strafe = 0.0
            is_sprinting = False
            jump_requested = False

        # Step deterministic simulation (Fixed 60 Hz tick + Sub-Frame Render Interpolation)
        loop.step_frame(dt)

        # Synchronize EngineTweaks with Pipeline Config & CSM
        if engine_tweaks.shadow_resolution != pipeline.csm.atlas_size:
            pipeline.csm.resize_atlas(engine_tweaks.shadow_resolution)
        if (
            engine_tweaks.shadow_distance != pipeline.csm.max_distance
            or engine_tweaks.csm_cascades != pipeline.csm.cascade_count
        ):
            pipeline.csm.update_splits(engine_tweaks.shadow_distance, engine_tweaks.csm_cascades)
        pipeline.config.shadow_resolution = engine_tweaks.shadow_resolution
        pipeline.config.shadow_distance = engine_tweaks.shadow_distance
        pipeline.config.csm_cascades = engine_tweaks.csm_cascades
        pipeline.config.shadow_mode = engine_tweaks.shadow_mode
        pipeline.config.shadow_softness = engine_tweaks.shadow_softness
        pipeline.config.shadow_bias = engine_tweaks.shadow_bias
        pipeline.config.shadow_normal_bias = engine_tweaks.shadow_normal_bias
        pipeline.config.shadow_offset_x = engine_tweaks.shadow_offset_x
        pipeline.config.shadow_offset_y = engine_tweaks.shadow_offset_y
        pipeline.config.sscs_enabled = engine_tweaks.sscs_enabled
        pipeline.config.sscs_steps = engine_tweaks.sscs_steps
        pipeline.config.sscs_thickness = engine_tweaks.sscs_thickness
        pipeline.config.sscs_ray_distance = engine_tweaks.sscs_ray_distance
        pipeline.config.sscs_max_distance = engine_tweaks.sscs_max_distance
        pipeline.config.wireframe = engine_tweaks.show_wireframe
        pipeline.config.tonemap_mode = engine_tweaks.tonemap_mode
        pipeline.config.debug_gbuffer = int(engine_tweaks.gbuffer_debug)
        pipeline.config.hiz_debug_mip = int(getattr(engine_tweaks, "hiz_debug_mip", 0))
        pipeline.config.exposure = engine_tweaks.exposure
        pipeline.config.sun_intensity = engine_tweaks.sun_lux
        pipeline.config.ao_mode = engine_tweaks.ao_mode
        pipeline.config.ao_intensity = engine_tweaks.ao_intensity
        pipeline.config.ao_radius = engine_tweaks.ao_radius
        pipeline.config.gi_mode = engine_tweaks.gi_mode
        pipeline.config.ssgi_steps = engine_tweaks.ssgi_steps
        pipeline.config.ssgi_rays = engine_tweaks.ssgi_rays
        pipeline.config.ssgi_thickness = engine_tweaks.ssgi_thickness
        pipeline.config.ssgi_ray_distance = engine_tweaks.ssgi_ray_distance
        pipeline.config.ssgi_intensity = engine_tweaks.ssgi_intensity
        pipeline.config.lpv_intensity = engine_tweaks.lpv_intensity
        pipeline.config.ibl_enabled = engine_tweaks.ibl_enabled
        pipeline.config.ssr_enabled = engine_tweaks.ssr_enabled
        pipeline.config.ssr_steps = engine_tweaks.ssr_steps
        pipeline.config.ssr_thickness = engine_tweaks.ssr_thickness
        pipeline.config.aa_mode = getattr(engine_tweaks, "aa_mode", "OFF")
        pipeline.config.taa_enabled = False
        pipeline.config.clustered_lights_enabled = engine_tweaks.point_lights_enabled

        # Volumetric Fog
        pipeline.config.volumetric_fog_enabled = engine_tweaks.volumetric_fog_enabled
        pipeline.config.fog_density = engine_tweaks.fog_density
        pipeline.config.fog_height_falloff = engine_tweaks.fog_height_falloff
        pipeline.config.fog_anisotropy = engine_tweaks.fog_anisotropy
        pipeline.config.fog_distance = engine_tweaks.fog_distance
        pipeline.config.fog_ambient = engine_tweaks.fog_ambient
        pipeline.config.fog_debug_mode = engine_tweaks.fog_debug_mode

        # Volumetric Particles (Phase 2)
        pipeline.config.particles_enabled = engine_tweaks.particles_enabled
        pipeline.config.particle_count = engine_tweaks.particle_count
        pipeline.config.particle_mode = engine_tweaks.particle_mode
        pipeline.config.particle_size = engine_tweaks.particle_size
        pipeline.config.particle_turbulence = engine_tweaks.particle_turbulence
        if pipeline.particle_pass is not None:
            pipeline.particle_pass.enabled = engine_tweaks.particles_enabled
            pipeline.particle_pass.set_mode(engine_tweaks.particle_mode)
            pipeline.particle_pass.set_active_count(engine_tweaks.particle_count)
            pipeline.particle_pass.turbulence_strength = engine_tweaks.particle_turbulence
            pipeline.particle_pass.base_size_multiplier = engine_tweaks.particle_size
            pipeline.particle_pass.sun_scatter_intensity = engine_tweaks.particle_brightness

        # Cinematic Camera Optics & Lens Effects (Phase 3)
        pipeline.config.dof_enabled = engine_tweaks.dof_enabled
        pipeline.config.dof_focus_distance = engine_tweaks.dof_focus_distance
        pipeline.config.dof_focal_length = engine_tweaks.dof_focal_length
        pipeline.config.dof_aperture = engine_tweaks.dof_aperture
        pipeline.config.dof_bokeh_shape = engine_tweaks.dof_bokeh_shape
        pipeline.config.dof_anamorphic_ratio = engine_tweaks.dof_anamorphic_ratio
        pipeline.config.dof_max_coc = engine_tweaks.dof_max_coc
        pipeline.config.motion_blur_enabled = engine_tweaks.motion_blur_enabled
        pipeline.config.motion_blur_samples = engine_tweaks.motion_blur_samples
        pipeline.config.motion_blur_intensity = engine_tweaks.motion_blur_intensity
        pipeline.config.motion_blur_max_radius = engine_tweaks.motion_blur_max_radius
        pipeline.config.lens_flare_enabled = engine_tweaks.lens_flare_enabled
        pipeline.config.lens_flare_threshold = engine_tweaks.lens_flare_threshold
        pipeline.config.lens_flare_streak_intensity = engine_tweaks.lens_flare_streak_intensity
        pipeline.config.lens_flare_streak_width = engine_tweaks.lens_flare_streak_width
        pipeline.config.lens_flare_ghost_intensity = engine_tweaks.lens_flare_ghost_intensity
        pipeline.config.lens_flare_halo_intensity = engine_tweaks.lens_flare_halo_intensity
        pipeline.config.chromatic_aberration_enabled = engine_tweaks.chromatic_aberration_enabled
        pipeline.config.chromatic_aberration_intensity = engine_tweaks.chromatic_aberration_intensity
        pipeline.config.vignette_enabled = engine_tweaks.vignette_enabled
        pipeline.config.vignette_intensity = engine_tweaks.vignette_intensity
        pipeline.config.vignette_roundness = engine_tweaks.vignette_roundness
        pipeline.config.vignette_smoothness = engine_tweaks.vignette_smoothness
        pipeline.config.film_grain_enabled = engine_tweaks.film_grain_enabled
        pipeline.config.film_grain_intensity = engine_tweaks.film_grain_intensity

        # Dynamic Water & Screen-Space Refraction (Phase 4)
        pipeline.config.water_enabled = engine_tweaks.water_enabled
        pipeline.config.water_height = engine_tweaks.water_height
        pipeline.config.water_wave_amplitude = engine_tweaks.water_wave_amplitude
        pipeline.config.water_wave_speed = engine_tweaks.water_wave_speed
        pipeline.config.water_wave_steepness = engine_tweaks.water_wave_steepness
        pipeline.config.water_refraction_enabled = engine_tweaks.water_refraction_enabled
        pipeline.config.water_refraction_strength = engine_tweaks.water_refraction_strength
        pipeline.config.water_foam_enabled = engine_tweaks.water_foam_enabled
        pipeline.config.water_foam_threshold = engine_tweaks.water_foam_threshold
        pipeline.config.water_foam_scale = engine_tweaks.water_foam_scale
        pipeline.config.water_foam_intensity = engine_tweaks.water_foam_intensity
        pipeline.config.water_clarity = engine_tweaks.water_clarity
        pipeline.config.water_roughness = engine_tweaks.water_roughness
        pipeline.config.water_color_shallow = engine_tweaks.water_color_shallow
        pipeline.config.water_color_deep = engine_tweaks.water_color_deep
        if pipeline.water_pass is not None:
            pipeline.water_pass.enabled = engine_tweaks.water_enabled

        # Micro-Geometry (POM, SSDM, Hardware Tessellation)
        pipeline.config.pom_enabled = engine_tweaks.pom_enabled
        pipeline.config.pom_height_scale = engine_tweaks.pom_height_scale
        pipeline.config.pom_self_shadow = engine_tweaks.pom_self_shadow
        pipeline.config.disp_near_radius = getattr(engine_tweaks, "disp_near_radius", 25.0)
        pipeline.config.disp_mid_radius = getattr(engine_tweaks, "disp_mid_radius", 60.0)

        # Physical Atmosphere & Dynamic Day-Night Cycle
        pipeline.atmosphere.config.day_speed = engine_tweaks.day_speed
        if engine_tweaks.day_speed > 0.0:
            engine_tweaks.time_of_day = pipeline.atmosphere.config.time_of_day
        else:
            pipeline.atmosphere.config.time_of_day = engine_tweaks.time_of_day

        pipeline.atmosphere.config.rayleigh_beta = (
            engine_tweaks.rayleigh_r * 1e-6,
            engine_tweaks.rayleigh_g * 1e-6,
            engine_tweaks.rayleigh_b * 1e-6,
        )
        pipeline.atmosphere.config.mie_beta = engine_tweaks.mie_coeff * 1e-6
        pipeline.atmosphere.config.turbidity = engine_tweaks.turbidity
        pipeline.atmosphere.config.star_intensity = engine_tweaks.star_intensity

        # Bidirectional Sun Direction: Atmosphere Simulation vs Manual Sliders
        tod_changed = abs(engine_tweaks.time_of_day - prev_tod) > 1e-4
        slider_changed = (
            abs(engine_tweaks.sun_elevation_deg - prev_sun_el) > 1e-3
            or abs(engine_tweaks.sun_angle_deg - prev_sun_az) > 1e-3
        )

        if engine_tweaks.day_speed > 0.0 or tod_changed:
            sun_v = pipeline.atmosphere.compute_sun_vector()
            scratch_sun_dir[0] = sun_v[0]
            scratch_sun_dir[1] = sun_v[1]
            scratch_sun_dir[2] = sun_v[2]
            scratch_sun_tuple = (sun_v[0], sun_v[1], sun_v[2])
            engine_tweaks.sun_elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, -sun_v[1]))))
            engine_tweaks.sun_angle_deg = math.degrees(math.atan2(sun_v[0], sun_v[2]))
            prev_tod = engine_tweaks.time_of_day
            prev_sun_el = engine_tweaks.sun_elevation_deg
            prev_sun_az = engine_tweaks.sun_angle_deg
        elif slider_changed:
            rad_sun_az = math.radians(engine_tweaks.sun_angle_deg)
            rad_sun_el = math.radians(engine_tweaks.sun_elevation_deg)
            scratch_sun_dir[0] = math.cos(rad_sun_el) * math.sin(rad_sun_az)
            scratch_sun_dir[1] = -math.sin(rad_sun_el)
            scratch_sun_dir[2] = math.cos(rad_sun_el) * math.cos(rad_sun_az)
            scratch_sun_tuple = (scratch_sun_dir[0], scratch_sun_dir[1], scratch_sun_dir[2])
            prev_sun_el = engine_tweaks.sun_elevation_deg
            prev_sun_az = engine_tweaks.sun_angle_deg

        # Update Character Position & Camera Tracking
        char_idx = ecs.pool.get_dense_index(char_id)
        if char_idx >= 0:
            char_world_pos = (
                float(ecs.world_transforms[char_idx, 12]),
                float(ecs.world_transforms[char_idx, 13]),
                float(ecs.world_transforms[char_idx, 14]),
            )
        else:
            char_world_pos = (0.0, 0.0, 0.0)

        with monitor.scope("camera_update"):
            cam.update_follow(dt=dt, character_pos=char_world_pos, physics=physics)

        if camera_mgr.active_camera_name == "free":
            free_cam.move(
                dt=dt,
                forward_axis=move_fwd,
                right_axis=move_strafe,
                up_axis=1.0 if input_mgr.is_action_down("jump") else (-1.0 if input_mgr.is_action_down("sprint") else 0.0),
                boost=input_mgr.is_action_down("sprint"),
            )

        camera_mgr.update(dt)
        active_cam = camera_mgr.active_camera
        camera_pos = (float(active_cam.position[0]), float(active_cam.position[1]), float(active_cam.position[2]))
        camera_target = (float(active_cam.target[0]), float(active_cam.target[1]), float(active_cam.target[2]))

        # Update 3D Audio spatial listener
        scratch_cam_pos[0] = camera_pos[0]
        scratch_cam_pos[1] = camera_pos[1]
        scratch_cam_pos[2] = camera_pos[2]
        scratch_fwd_vec[0] = camera_target[0] - camera_pos[0]
        scratch_fwd_vec[1] = camera_target[1] - camera_pos[1]
        scratch_fwd_vec[2] = camera_target[2] - camera_pos[2]
        audio_engine.update(dt, camera_position=camera_pos, camera_forward=scratch_fwd_vec)

        # 3D Immediate-Mode Wireframe Debug Drawing
        pipeline.debug.update(dt)
        if engine_tweaks.show_physics_colliders:
            # Ground platform box wireframe
            pipeline.debug.draw_box(
                center=(0.0, -0.5, 0.0),
                size=(plane_size, 1.0, plane_size),
                color=(0.25, 0.65, 0.85, 0.35),
            )
            # Player capsule wireframe
            pipeline.debug.draw_capsule(
                center=char_world_pos,
                radius=0.4,
                half_height=0.5,
                color=(0.15, 0.85, 0.95, 0.9),
            )
            # Ground probe ray
            probe_color = (0.2, 0.9, 0.3, 1.0) if motor.state.is_grounded else (0.9, 0.2, 0.2, 1.0)
            pipeline.debug.draw_ray(
                origin=char_world_pos,
                direction=(0.0, -1.0, 0.0),
                length=1.15,
                color=probe_color,
            )

        if getattr(engine_tweaks, "show_bounds", False):
            # Ground plane AABB bounds
            pipeline.debug.draw_box(
                center=(0.0, 0.0, 0.0),
                size=(plane_size, 0.1, plane_size),
                color=(0.95, 0.85, 0.25, 0.5),
            )
            # Character bounds
            pipeline.debug.draw_box(
                center=char_world_pos,
                size=(0.8, 1.8, 0.8),
                color=(0.95, 0.35, 0.85, 0.5),
            )

        if engine_tweaks.show_sun_ray:
            pipeline.debug.draw_ray(
                origin=camera_target,
                direction=(-scratch_sun_dir[0], -scratch_sun_dir[1], -scratch_sun_dir[2]),
                length=10.0,
                color=(1.0, 0.9, 0.2, 1.0),
            )

        # Collect static batches + all dynamic props and fragments
        frame_draw_batches = list(draw_batches)
        frame_draw_batches.extend(
            prop_mgr.get_draw_batches(
                alloc_cube=alloc_cube,
                alloc_sphere=alloc_sphere,
            )
        )

        # Render Frame via MDI Deferred Pipeline
        with monitor.scope("render_pipeline"):
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=camera_pos,
                camera_target=camera_target,
                time_elapsed=loop.elapsed_time,
                sun_dir=scratch_sun_tuple,
                sun_lux=pipeline.config.sun_intensity,
                draw_batches=frame_draw_batches,
                debug_draw=pipeline.debug,
                dt=dt,
            )

        # Render Debug Menu
        if debug_menu.visible or (debug_menu.toast is not None and len(debug_menu.toast.get_active()) > 0):
            pipeline.post_process.final_fbo.use()
            debug_menu.render(pipeline.post_process.final_fbo)
            if not args.headless:
                render_ctx.ctx.copy_framebuffer(render_ctx.ctx.screen, pipeline.post_process.final_fbo)

        if not args.headless:
            render_ctx.swap_buffers()

        frame_idx += 1

        # Automated Screenshot Capture & Exit
        if args.screenshot and (frame_idx >= args.frames or args.frames <= 0):
            out_p = Path(args.screenshot).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            raw_pixels = pipeline.post_process.final_fbo.read(components=4, dtype="f1")
            img = Image.frombytes("RGBA", (render_ctx.width, render_ctx.height), raw_pixels)
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img.save(str(out_p))
            print(f"Validation screenshot saved to: {out_p}")
            break

        if args.frames > 0 and frame_idx >= args.frames:
            break

    # Clean Engine Shutdown
    debug_menu.destroy()
    pipeline.destroy()
    render_ctx.destroy()
    audio_engine.shutdown()
    pygame.quit()
    print("POM Mud Plane validation completed successfully.")


if __name__ == "__main__":
    main()
