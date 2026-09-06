"""Unified Master Validation Demo for PyMordial Engine.

Comprehensive, unified interactive showcase testing all engine subsystems in a single playground scene:
- Phase 1: High-Performance C-RAM, SIMD Rapier3D physics, sub-frame NLERP, instant PIE snapshot/restore
- Phase 2: ModernGL 4.5 Core Profile 3D Deferred PBR, Cook-Torrance GGX, Reversed-Z 32F depth,
           4-Cascade CSM + SSCS contact shadows, ACES/AgX/Reinhard tonemapping, runtime quality presets,
           and G-Buffer debug visualizer
- Phase 3: Kinematic Character Controller (KCC) with omnidirectional movement, sprint, jump,
           auto-stepping (<= 0.30m), slope sliding (> 45 deg), collision-aware 3rd-person follow camera,
           and live wireframe capsule & ground probe gizmos
- Phase 4: Offline asset cooker (.pm_mesh, .pm_tex), .pak archive packaging, zero-copy memory-mapped VFS
           streaming, and MegaBuffer MDI batch ingestion
- AAA Input: Dedicated dual-device input engine (Keyboard/Mouse + Gamepad) with normalized diagonal clamping
- 3-Tier Debug: SystemMonitor microsecond profiler, EngineTweaks graphics pipeline, GameTweaks registry,
                and glassmorphic 3-tab DebugMenu
- Immediate-Mode Wireframe Gizmos: Character capsule, ground probe, sun ray, monolith bounding spheres,
                                   and stepping course obstacles
- CLI Flags: --headless, --frames N, --screenshot PATH, --preset PRESET, --width W, --height H
"""

from __future__ import annotations
import argparse
import math
import random
import sys
from pathlib import Path
import pygame
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.core.ecs import EntityManager  # noqa: E402
from engine.core.loop import EngineLoop  # noqa: E402
from engine.input import (  # noqa: E402
    InputManager,
    Key,
    MouseButton,
    GamepadButton,
)
from engine.physics.rapier_world import PhysicsManager  # noqa: E402
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig  # noqa: E402
from engine.core.character_camera import CharacterCamera, CharacterCameraConfig  # noqa: E402
from engine.assets.cooker import cook_mesh, cook_texture, pack_directory  # noqa: E402
from engine.gfx import (  # noqa: E402
    RenderContext,
    RenderPipeline,
    GraphicsQuality,
    GIMode,
    get_quality_preset,
)
from engine.gfx.mega_buffer import MeshAllocation  # noqa: E402
from engine.debug import (  # noqa: E402
    SystemMonitor,
    EngineTweaks,
    GameTweaks,
    DebugToast,
    DebugMenu,
    GBufferDebugMode,
)
from engine.camera import CameraManager, FreeFlyCamera  # noqa: E402
from engine.audio import get_audio_engine  # noqa: E402
from engine.window import WindowMode, VSyncMode  # noqa: E402

VALIDATION_LANTERNS = (
    (-8.0, 1.8, 0.0, 12.0, (1.0, 0.25, 0.15), 4.0),
    (8.0, 1.8, 0.0, 12.0, (0.15, 0.75, 1.0), 4.0),
    (0.0, 2.2, -8.0, 14.0, (0.20, 1.0, 0.35), 4.0),
    (0.0, 2.2, 8.0, 14.0, (1.0, 0.85, 0.20), 4.5),
    (10.0, 2.5, 10.0, 12.0, (0.85, 0.20, 0.95), 4.0),
    (-10.0, 2.5, -10.0, 12.0, (1.0, 0.50, 0.10), 4.0),
)


def make_quat_rot_x(angle_rad: float) -> tuple[float, float, float, float]:
    """Computes unit quaternion for rotation around local X axis."""
    half = angle_rad * 0.5
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def generate_monolith_obj(filepath: Path) -> None:
    """Generates an intricate, beveled 3D monolith pillar model with solid architectural plinth."""
    obj_lines = [
        "# PyMordial Master Validation: Architectural Beveled Monolith Pillar",
        # Plinth bottom Y=0.0
        "v -1.0 0.0 -1.0",  # 1
        "v  1.0 0.0 -1.0",  # 2
        "v  1.0 0.0  1.0",  # 3
        "v -1.0 0.0  1.0",  # 4
        # Plinth top Y=0.25 (vertical sides)
        "v -1.0 0.25 -1.0", # 5
        "v  1.0 0.25 -1.0", # 6
        "v  1.0 0.25  1.0", # 7
        "v -1.0 0.25  1.0", # 8
        # Shaft bottom Y=0.45 (beveled inward transition)
        "v -0.8 0.45 -0.8", # 9
        "v  0.8 0.45 -0.8", # 10
        "v  0.8 0.45  0.8", # 11
        "v -0.8 0.45  0.8", # 12
        # Shaft top Y=3.6
        "v -0.6 3.6 -0.6",  # 13
        "v  0.6 3.6 -0.6",  # 14
        "v  0.6 3.6  0.6",  # 15
        "v -0.6 3.6  0.6",  # 16
        # Pyramidion apex vertex (Y = 4.2)
        "v  0.0 4.2  0.0",  # 17
        # Texture coordinates
        "vt 0.0 0.0",
        "vt 1.0 0.0",
        "vt 1.0 1.0",
        "vt 0.0 1.0",
        "vt 0.5 1.0",
        # Normals
        "vn  0.0 -1.0  0.0",   # 1 Bottom
        "vn  0.0  1.0  0.0",   # 2 Top
        "vn  0.0  0.0 -1.0",   # 3 North
        "vn  1.0  0.0  0.0",   # 4 East
        "vn  0.0  0.0  1.0",   # 5 South
        "vn -1.0  0.0  0.0",   # 6 West
        "vn  0.0  0.707 -0.707", # 7 Bevel North
        "vn  0.707 0.707 0.0",   # 8 Bevel East
        "vn  0.0  0.707  0.707", # 9 Bevel South
        "vn -0.707 0.707 0.0",   # 10 Bevel West
        # Faces: Bottom
        "f 1/1/1 2/2/1 3/3/1",
        "f 1/1/1 3/3/1 4/4/1",
        # Plinth vertical sides (Y=0.0 to Y=0.25)
        "f 1/1/3 6/3/3 2/2/3",
        "f 1/1/3 5/4/3 6/3/3",
        "f 2/1/4 7/3/4 3/2/4",
        "f 2/1/4 6/4/4 7/3/4",
        "f 3/1/5 8/3/5 4/2/5",
        "f 3/1/5 7/4/5 8/3/5",
        "f 4/1/6 5/3/6 1/2/6",
        "f 4/1/6 8/4/6 5/3/6",
        # Bevel transition (Y=0.25 to Y=0.45)
        "f 5/1/7 10/3/7 6/2/7",
        "f 5/1/7 9/4/7 10/3/7",
        "f 6/1/8 11/3/8 7/2/8",
        "f 6/1/8 10/4/8 11/3/8",
        "f 7/1/9 12/3/9 8/2/9",
        "f 7/1/9 11/4/9 12/3/9",
        "f 8/1/10 9/3/10 5/2/10",
        "f 8/1/10 12/4/10 9/3/10",
        # Main Shaft faces (Y=0.45 to Y=3.6)
        "f 9/1/3 14/3/3 10/2/3",
        "f 9/1/3 13/4/3 14/3/3",
        "f 10/1/4 15/3/4 11/2/4",
        "f 10/1/4 14/4/4 15/3/4",
        "f 11/1/5 16/3/5 12/2/5",
        "f 11/1/5 15/4/5 16/3/5",
        "f 12/1/6 13/3/6 9/2/6",
        "f 12/1/6 16/4/6 13/3/6",
        # Pyramidion Cap
        "f 13/1/7 17/5/7 14/2/7",
        "f 14/1/8 17/5/8 15/2/8",
        "f 15/1/9 17/5/9 16/2/9",
        "f 16/1/10 17/5/10 13/2/10",
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(obj_lines), encoding="utf-8")


def generate_pattern_texture(filepath: Path) -> None:
    """Generates a procedural high-res texture with rich accents for offline baking."""
    w, h = 128, 128
    img = Image.new("RGBA", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            cx = (x % 32) < 16
            cy = (y % 32) < 16
            is_check = cx ^ cy
            r = 190 if is_check else 130
            g = 150 if is_check else 95
            b = 85 if is_check else 55
            pixels[x, y] = (r, g, b, 255)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(filepath)


def bake_asset_package(build_dir: Path) -> Path:
    """Bakes raw source assets into 32-byte .pm_mesh, .pm_tex, and packages into a .pak archive."""
    src_dir = build_dir / "raw_assets"
    cooked_dir = build_dir / "cooked_assets"
    pak_path = build_dir / "validation_assets.pak"

    raw_obj = src_dir / "monolith.obj"
    raw_tex = src_dir / "monolith_albedo.png"

    generate_monolith_obj(raw_obj)
    generate_pattern_texture(raw_tex)

    cooked_mesh_path = cooked_dir / "models" / "monolith.pm_mesh"
    cooked_tex_path = cooked_dir / "textures" / "monolith_albedo.pm_tex"

    # 1. Cook raw .obj -> 32-byte cache-aligned .pm_mesh
    cook_mesh(raw_obj, cooked_mesh_path)
    # 2. Cook raw .png -> mipmapped .pm_tex
    cook_texture(raw_tex, cooked_tex_path)

    # 3. Pack cooked directory into contiguous .pak archive container
    pack_directory(cooked_dir, pak_path, compress=False)
    return pak_path


def spawn_static_box(
    ecs: EntityManager,
    physics: PhysicsManager,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    roughness: float = 0.5,
    metallic: float = 0.1,
) -> int:
    """Spawns a static box where visual scale exactly matches physical half-extents."""
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    ent = ecs.create_entity(
        position=position,
        rotation=rotation,
        scale=size,
        color=color,
        roughness=roughness,
        metallic=metallic,
    )
    physics.create_body(ent, body_type="fixed", position=position, rotation=rotation)
    physics.attach_box_collider(ent, half_x=hx, half_y=hy, half_z=hz)
    return ent


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Engine Master Validation")
    parser.add_argument("--headless", action="store_true", help="Run without window display")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames")
    parser.add_argument("--screenshot", type=str, default="", help="Save screenshot to path")
    parser.add_argument("--preset", type=str, default="custom", choices=["custom", "low", "medium", "high", "ultra", "cinematic"], help="Initial quality preset (default: custom)")
    parser.add_argument("--vsync", action="store_true", default=False, help="Enable VSync (default: False/uncapped)")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--wireframe", action="store_true", help="Start with mesh wireframe enabled")
    parser.add_argument("--no-physics-colliders", action="store_true", default=True, help="Disable physics colliders wireframe (default: True/disabled)")
    parser.add_argument("--show-physics-colliders", action="store_true", default=False, help="Enable physics colliders wireframe")
    parser.add_argument(
        "--debug-menu",
        dest="debug_menu",
        action="store_true",
        default=True,
        help="Start with debug menu open (default: True)",
    )
    parser.add_argument(
        "--no-debug-menu",
        dest="debug_menu",
        action="store_false",
        help="Start with debug menu closed",
    )
    parser.add_argument("--debug-tab", type=int, default=0, help="Initial debug menu tab (0, 1, 2)")
    parser.add_argument("--shadow-mode", type=str, default="pcss", choices=["hard", "pcf", "pcss"], help="Shadow filtering mode (default: pcss)")
    parser.add_argument("--shadow-res", type=int, default=2048, choices=[1024, 2048, 4096], help="Shadow map atlas resolution (default: 2048)")
    parser.add_argument("--gbuffer-debug", type=int, default=0, help="G-Buffer / Shadow / GI debug mode (0=Off, 1=Albedo, 2=Normals, 3=Material, 4=Depth, 5=Atlas, 6=ShadowMask, 7=AO, 8=SSGI, 9=LPV, 10=GI_Total)")
    parser.add_argument("--ao-mode", type=str, default="", choices=["", "off", "ssao", "hbao", "gtao"], help="AO mode override (off, ssao, hbao, gtao)")
    parser.add_argument("--gi-mode", type=str, default="", choices=["", "off", "ssgi", "lpv", "hybrid"], help="GI mode override (off, ssgi, lpv, hybrid)")
    parser.add_argument("--aa-mode", type=str, default="", choices=["", "off", "fxaa", "smaa_1x", "smaa_2x", "smaa_4x", "taa"], help="AA mode override (off, fxaa, smaa_1x, smaa_2x, smaa_4x, taa)")
    parser.add_argument("--all-effects", action="store_true", default=False, help="Enable secondary graphics effects (GTAO, SSGI, SSR, IBL, TAA, point lights). Default is False (isolated shadows).")
    args = parser.parse_args()

    width, height = args.width, args.height

    # Bake validation asset package (.pak)
    build_dir = ROOT_DIR / "build" / "validation_assets"
    pak_path = bake_asset_package(build_dir)

    # 1. Initialize Display & Engine Core
    initial_preset_enum = GraphicsQuality(args.preset.lower())
    render_ctx = RenderContext(
        width=width,
        height=height,
        title="PyMordial Engine — Master Validation Playground",
        hidden=args.headless,
        config=get_quality_preset(initial_preset_enum),
    )
    if args.vsync:
        render_ctx.window.set_vsync(VSyncMode.ON)
    else:
        render_ctx.window.set_vsync(VSyncMode.OFF)

    pipeline = RenderPipeline(render_ctx, get_quality_preset(initial_preset_enum))
    pipeline.apply_config(pipeline.config)

    ecs = EntityManager(max_entities=2000)
    physics = PhysicsManager(gravity_x=0.0, gravity_y=-20.0, gravity_z=0.0)
    input_mgr = InputManager()
    if not args.headless:
        input_mgr.set_mouse_grab(True)
        input_mgr._was_mouse_grabbed = True
    loop = EngineLoop(ecs=ecs, input_manager=input_mgr, fixed_dt=1.0 / 60.0)

    # 2. Standardized 3-Tier Debug System
    monitor = SystemMonitor(history_size=120)
    engine_tweaks = EngineTweaks()
    engine_tweaks.quality_preset = initial_preset_enum
    engine_tweaks.vsync_enabled = args.vsync
    engine_tweaks.shadow_mode = args.shadow_mode.upper()
    engine_tweaks.shadow_resolution = args.shadow_res
    engine_tweaks.gbuffer_debug = GBufferDebugMode(args.gbuffer_debug)
    if args.wireframe:
        engine_tweaks.show_wireframe = True
    # Default to False (disabled) unless explicitly requested via --show-physics-colliders
    engine_tweaks.show_physics_colliders = args.show_physics_colliders
    if not args.all_effects:
        engine_tweaks.ao_mode = "OFF"
        engine_tweaks.gi_mode = GIMode.OFF
        engine_tweaks.ibl_enabled = False
        engine_tweaks.ssr_enabled = False
        engine_tweaks.aa_mode = "OFF"
        engine_tweaks.taa_enabled = False
        engine_tweaks.point_lights_enabled = False
    if args.ao_mode:
        engine_tweaks.ao_mode = args.ao_mode.upper()
    if args.gi_mode:
        engine_tweaks.gi_mode = args.gi_mode.upper()
    if args.aa_mode:
        engine_tweaks.aa_mode = args.aa_mode.upper()
        engine_tweaks.taa_enabled = (args.aa_mode.upper() == "TAA")
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
    if args.debug_menu:
        debug_menu.show_graphics = True
        debug_menu.f1_style = 1
        if not args.headless:
            input_mgr.set_mouse_grab(False)
            input_mgr._was_mouse_grabbed = False

    toast.show("Press [F1] or [F2] for Graphics Options | [F9] Mouse Grab", duration=6.0, color=(56, 189, 248))

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
        toast.show(f"Quality Preset: {new_preset.value.upper()}", duration=2.5, color=(56, 189, 248))

    engine_tweaks.on_quality_changed(on_quality_changed)
    engine_tweaks.on_vsync_changed(lambda enabled: render_ctx.window.set_vsync(VSyncMode.ON if enabled else VSyncMode.OFF))

    # 3. Mount Cooked .pak Archive via Virtual File System (VFS)
    pipeline.resources.mount_pak(pak_path)

    # 4. Load Cooked 32-Byte .pm_mesh Directly from .pak into GPU MegaBuffer
    alloc_monolith = pipeline.load_cooked_mesh("monolith", "models/monolith.pm_mesh")
    cooked_mesh = pipeline.resources.load_mesh("models/monolith.pm_mesh")

    # 5. Construct Unified Playground Scene Environment
    cube_ids: list[int] = []
    step_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    # A. Central Arena Ground Platform (60m x 1m x 60m, top surface at y = 0.0)
    ground_id = spawn_static_box(
        ecs,
        physics,
        position=(0.0, -0.5, 0.0),
        size=(60.0, 1.0, 60.0),
        color=(0.22, 0.25, 0.28),
        roughness=0.65,
        metallic=0.10,
    )
    cube_ids.append(ground_id)
    step_boxes.append(((0.0, -0.5, 0.0), (60.0, 1.0, 60.0)))

    # Reflective Marble Plaza Platform (Showcase for SSR, IBL, and SSGI Diffuse Bounce)
    plaza_id = spawn_static_box(
        ecs,
        physics,
        position=(0.0, 0.025, 0.0),
        size=(22.0, 0.05, 22.0),
        color=(0.12, 0.14, 0.18),
        roughness=0.08,
        metallic=0.75,
    )
    cube_ids.append(plaza_id)
    step_boxes.append(((0.0, 0.025, 0.0), (22.0, 0.05, 22.0)))

    # B. Stepping Course & Curbs (West side, testing auto-stepping <= 0.30m)
    # Step 1: 0.15m height (easy auto-step)
    s1_pos, s1_size = (-7.0, 0.075, -6.0), (3.5, 0.15, 1.6)
    cube_ids.append(spawn_static_box(ecs, physics, s1_pos, s1_size, color=(0.60, 0.55, 0.50), roughness=0.4, metallic=0.0))
    step_boxes.append((s1_pos, s1_size))

    # Step 2: 0.30m total height (auto-step limit)
    s2_pos, s2_size = (-7.0, 0.150, -8.0), (3.5, 0.30, 1.6)
    cube_ids.append(spawn_static_box(ecs, physics, s2_pos, s2_size, color=(0.65, 0.60, 0.55), roughness=0.4, metallic=0.0))
    step_boxes.append((s2_pos, s2_size))

    # Step 3: 0.45m total height (requires jump)
    s3_pos, s3_size = (-7.0, 0.225, -10.0), (3.5, 0.45, 1.6)
    cube_ids.append(spawn_static_box(ecs, physics, s3_pos, s3_size, color=(0.70, 0.65, 0.60), roughness=0.4, metallic=0.0))
    step_boxes.append((s3_pos, s3_size))

    # Tall Barrier (1.2m high - jump obstacle)
    s4_pos, s4_size = (-7.0, 0.600, -12.5), (3.5, 1.20, 1.2)
    cube_ids.append(spawn_static_box(ecs, physics, s4_pos, s4_size, color=(0.78, 0.42, 0.20), roughness=0.3, metallic=0.2))
    step_boxes.append((s4_pos, s4_size))

    # C. Slope Limit Testing Ramps (East side)
    # Ramp A: 20° gentle incline (Walkable green ramp, <= 45°)
    q_ramp_a = make_quat_rot_x(math.radians(-20.0))
    ramp_a = spawn_static_box(
        ecs,
        physics,
        position=(7.0, 0.85, -6.0),
        size=(3.5, 0.2, 6.0),
        rotation=q_ramp_a,
        color=(0.22, 0.68, 0.38),
        roughness=0.4,
        metallic=0.1,
    )
    cube_ids.append(ramp_a)

    # Ramp B: 52° steep incline (Exceeds 45° slope limit - Blocked / Sliding red ramp)
    q_ramp_b = make_quat_rot_x(math.radians(-52.0))
    ramp_b = spawn_static_box(
        ecs,
        physics,
        position=(12.0, 1.80, -6.0),
        size=(3.5, 0.2, 6.0),
        rotation=q_ramp_b,
        color=(0.78, 0.22, 0.25),
        roughness=0.3,
        metallic=0.2,
    )
    cube_ids.append(ramp_b)

    # D. Obstacle Columns & Boundary Walls
    cube_ids.append(spawn_static_box(ecs, physics, (-4.0, 2.0, 7.0), (1.2, 4.0, 1.2), color=(0.40, 0.45, 0.55), roughness=0.2, metallic=0.7))
    cube_ids.append(spawn_static_box(ecs, physics, (4.0, 2.0, 7.0), (1.2, 4.0, 1.2), color=(0.40, 0.45, 0.55), roughness=0.2, metallic=0.7))
    cube_ids.append(spawn_static_box(ecs, physics, (0.0, 2.0, 11.0), (12.0, 4.0, 0.6), color=(0.32, 0.36, 0.44), roughness=0.5, metallic=0.1))

    # E. Cooked Monolith Obelisks (North plaza, top of marble floor at Y = 0.05)
    monolith_ids: list[int] = []
    monolith_positions = [
        (-4.0, 0.05, -1.0),
        (4.0, 0.05, -1.0),
        (-4.0, 0.05, -7.0),
        (4.0, 0.05, -7.0),
    ]
    for pos in monolith_positions:
        ent = ecs.create_entity(
            position=pos,
            scale=(1.0, 1.0, 1.0),
            color=(0.88, 0.74, 0.48),
            roughness=0.32,
            metallic=0.18,
        )
        physics.create_body(ent, body_type="fixed", position=(pos[0], pos[1] + 2.1, pos[2]))
        physics.attach_box_collider(ent, half_x=0.8, half_y=2.1, half_z=0.8)
        monolith_ids.append(ent)

    # F. Playable KCC Capsule Character (Rests on marble floor at Y = 0.05 + 0.5 + 0.4 = 0.95)
    char_start_pos = (0.0, 0.95, 2.5)
    char_id = ecs.create_entity(
        position=char_start_pos,
        scale=(1.0, 1.0, 1.0),
        color=(0.15, 0.85, 0.95),
        roughness=0.20,
        metallic=0.85,
    )
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

    # G. Dynamic PBR Physics Spheres Cluster (Rich Metallic & Roughness Materials)
    PBR_PALETTE = [
        ((0.95, 0.75, 0.25), 0.15, 0.90),  # Polished Gold
        ((0.90, 0.92, 0.95), 0.20, 0.90),  # Chrome
        ((0.85, 0.20, 0.15), 0.35, 0.05),  # Ruby
        ((0.15, 0.55, 0.85), 0.20, 0.10),  # Cobalt Gloss
        ((0.25, 0.85, 0.45), 0.40, 0.00),  # Emerald Matte
        ((0.75, 0.30, 0.85), 0.30, 0.50),  # Metallic Violet
        ((0.90, 0.50, 0.20), 0.40, 0.40),  # Copper
    ]

    sphere_ids: list[int] = []

    def spawn_pbr_sphere(px: float | None = None, py: float | None = None, pz: float | None = None) -> int:
        x = px if px is not None else random.uniform(-2.5, 2.5)
        y = py if py is not None else random.uniform(3.5, 8.0)
        z = pz if pz is not None else random.uniform(3.0, 7.0)
        color, rough, metal = random.choice(PBR_PALETTE)
        radius = random.uniform(0.35, 0.55)

        sph = ecs.create_entity(
            position=(x, y, z),
            scale=(radius, radius, radius),
            color=color,
            roughness=rough,
            metallic=metal,
        )
        physics.create_body(sph, body_type="dynamic", position=(x, y, z))
        physics.attach_sphere_collider(sph, radius=radius)
        sphere_ids.append(sph)
        return sph

    # Seed initial cluster of 16 dynamic physics spheres
    for _ in range(16):
        spawn_pbr_sphere()

    # Pre-simulation instant PIE memory snapshot
    saved_snapshot = ecs.snapshot_memory()

    # H. 3rd-Person Collision-Aware Follow Camera
    cam = CharacterCamera(
        config=CharacterCameraConfig(
            distance=7.5,
            target_offset_y=1.35,
            camera_radius=0.25,
            yaw_sensitivity=0.25,
            pitch_sensitivity=0.25,
        ),
        initial_yaw_deg=45.0,
        initial_pitch_deg=22.0,
    )

    # Audio Engine & Virtual Camera Stack
    audio_engine = get_audio_engine()
    camera_mgr = CameraManager()
    camera_mgr.register_camera("follow", cam, make_active=True)
    free_cam = FreeFlyCamera(position=(0.0, 5.0, 10.0), fov=75.0)
    camera_mgr.register_camera("free", free_cam)

    # 6. Configure Unified Input Bindings (AAA Standard)
    # Character actions
    input_mgr.bind_action("jump", keys=[Key.SPACE], gamepad_buttons=[GamepadButton.A])
    input_mgr.bind_action("sprint", keys=[Key.LSHIFT, Key.RSHIFT], gamepad_buttons=[GamepadButton.LEFT_STICK, GamepadButton.LEFT_BUMPER])
    input_mgr.bind_action("reset_player", keys=[Key.R], gamepad_buttons=[GamepadButton.Y])
    input_mgr.bind_action("reset_camera", keys=[Key.Z], gamepad_buttons=[GamepadButton.RIGHT_STICK])

    # Simulation actions (X = Spawn, B = Clear)
    input_mgr.bind_action("spawn_spheres", keys=[Key.B], gamepad_buttons=[GamepadButton.X])
    input_mgr.bind_action("clear_spheres", keys=[Key.C], gamepad_buttons=[GamepadButton.B])
    input_mgr.bind_action("pie_save", keys=[Key.P])
    input_mgr.bind_action("pie_restore", keys=[Key.U])

    # Graphics & Pipeline actions
    input_mgr.bind_action("cycle_tonemap", keys=[Key.T], gamepad_buttons=[GamepadButton.DPAD_DOWN])
    input_mgr.bind_action("cycle_gbuffer", keys=[Key.G], gamepad_buttons=[GamepadButton.DPAD_LEFT])
    input_mgr.bind_action("toggle_sun_ray", keys=[Key.L], gamepad_buttons=[GamepadButton.DPAD_UP])
    # Dedicated Debug System hotkeys: F1 (Perf), F2 (Graphics), F3 (Game), F9 (Mouse Capture), F10 (Camera), F11 (Fullscreen)
    input_mgr.bind_action("debug_perf", keys=[Key.F1, Key.GRAVE], gamepad_buttons=[GamepadButton.BACK], is_debug=True)
    input_mgr.bind_action("debug_graphics", keys=[Key.F2], is_debug=True)
    input_mgr.bind_action("debug_game", keys=[Key.F3], is_debug=True)
    input_mgr.bind_action("debug_mouse_capture", keys=[Key.F9], is_debug=True)
    input_mgr.bind_action("toggle_camera", keys=[Key.F10], is_debug=True)
    input_mgr.bind_action("toggle_fullscreen", keys=[Key.F11], is_debug=True)
    input_mgr.bind_action("toggle_wireframe", keys=[Key.F7], is_debug=True)
    input_mgr.bind_action("toggle_physics_wireframe", keys=[Key.F8], is_debug=True)
    input_mgr.bind_action("quit", keys=[Key.ESCAPE], gamepad_buttons=[GamepadButton.START])

    # Camera zoom
    input_mgr.bind_action("zoom_in", keys=[Key.Q])
    input_mgr.bind_action("zoom_out", keys=[Key.E])

    # 7. Tier 3 Game-Specific Tweaks Registration
    def trigger_reset_player() -> None:
        motor.teleport(char_start_pos)
        toast.show("Player Teleported to Spawn", duration=2.0, color=(56, 189, 248))

    def trigger_spawn_batch() -> None:
        batch_size = int(sphere_batch_tweak.value)
        for _ in range(batch_size):
            spawn_pbr_sphere()
        toast.show(f"Spawned {batch_size} PBR Spheres (Total: {ecs.active_count})", duration=2.0, color=(74, 222, 128))

    def trigger_clear_spheres() -> None:
        count = len(sphere_ids)
        for sph in sphere_ids:
            physics.remove_body(sph)
            ecs.destroy_entity(sph)
        sphere_ids.clear()
        toast.show(f"Cleared {count} Spheres (Entities: {ecs.active_count})", duration=2.0, color=(248, 113, 113))

    def trigger_pie_save() -> None:
        nonlocal saved_snapshot
        saved_snapshot = ecs.snapshot_memory()
        toast.show(f"PIE Snapshot Saved ({ecs.active_count} entities in C-RAM)", duration=2.5, color=(56, 189, 248))

    def trigger_pie_restore() -> None:
        nonlocal saved_snapshot
        for sph in sphere_ids:
            physics.remove_body(sph)
            ecs.destroy_entity(sph)
        sphere_ids.clear()

        ecs.restore_snapshot(saved_snapshot)
        # Re-attach dynamic bodies for spheres present in snapshot
        for i in range(ecs.active_count):
            ent_id = ecs.pool.get_entity_id(i)
            if ent_id not in cube_ids and ent_id not in monolith_ids and ent_id != char_id:
                pos = ecs.rigid_body_state[1, i, 0:3]
                rot = ecs.rigid_body_state[1, i, 3:7]
                physics.create_body(ent_id, body_type="dynamic", position=pos, rotation=rot)
                physics.attach_sphere_collider(ent_id, radius=float(ecs.scales[i, 0]))
                sphere_ids.append(ent_id)

        toast.show(f"PIE Restored via memcpy ({ecs.active_count} entities)", duration=2.5, color=(250, 204, 21))

    # Tweaks & Sliders
    walk_spd_tweak = game_tweaks.add_float("Character", "Walk Speed", default=6.0, min_val=2.0, max_val=15.0, step=0.5)
    run_spd_tweak = game_tweaks.add_float("Character", "Sprint Speed", default=11.5, min_val=5.0, max_val=25.0, step=1.0)
    jump_force_tweak = game_tweaks.add_float("Character", "Jump Force", default=8.5, min_val=4.0, max_val=20.0, step=0.5)
    invert_pitch_tweak = game_tweaks.add_bool("Controls", "Invert Look Y", default=False)
    cam_sens_tweak = game_tweaks.add_float("Controls", "Camera Sensitivity", default=1.0, min_val=0.2, max_val=3.0, step=0.1)
    sphere_batch_tweak = game_tweaks.add_int("Physics", "Batch Spawn Count", default=8, min_val=1, max_val=32)

    # Rendering Tweaks
    wireframe_tweak = game_tweaks.add_bool(
        "Rendering", "Mesh Wireframe", default=False,
        on_changed=lambda val: setattr(engine_tweaks, "show_wireframe", val),
    )
    physics_wire_tweak = game_tweaks.add_bool(
        "Rendering", "Physics Gizmos", default=False,
        on_changed=lambda val: setattr(engine_tweaks, "show_physics_colliders", val),
    )
    game_tweaks.add_float("Shadows", "Align X", default=0.0, min_val=-0.05, max_val=0.05, step=0.001, on_changed=lambda v: setattr(engine_tweaks, "shadow_offset_x", v))
    game_tweaks.add_float("Shadows", "Align Y", default=0.0, min_val=-0.05, max_val=0.05, step=0.001, on_changed=lambda v: setattr(engine_tweaks, "shadow_offset_y", v))
    game_tweaks.add_float("Shadows", "Normal Bias", default=0.0010, min_val=0.0, max_val=0.0050, step=0.0001, on_changed=lambda v: setattr(engine_tweaks, "shadow_normal_bias", v))

    def trigger_reset_camera() -> None:
        cam.yaw_deg = 45.0
        cam.pitch_deg = 22.0
        cam.current_distance = cam.config.distance
        toast.show("Camera View Centered", duration=1.5, color=(147, 197, 253))

    def trigger_toggle_camera() -> None:
        if camera_mgr.active_camera_name == "follow":
            camera_mgr.blend_to("free", duration_seconds=0.75)
            toast.show("Camera: 6-DOF Free Fly (F10)", duration=2.0, color=(147, 197, 253))
        else:
            camera_mgr.blend_to("follow", duration_seconds=0.75)
            toast.show("Camera: 3rd-Person Follow (F10)", duration=2.0, color=(56, 189, 248))

    def trigger_set_resolution(w: int, h: int) -> None:
        render_ctx.window.set_resolution(w, h)
        toast.show(f"Resolution: {w}x{h}", duration=2.0, color=(74, 222, 128))

    def trigger_toggle_fullscreen() -> None:
        render_ctx.window.toggle_fullscreen()
        mode_str = "BORDERLESS FULLSCREEN" if render_ctx.window.mode == WindowMode.BORDERLESS_FULLSCREEN else "WINDOWED"
        toast.show(f"Display Mode: {mode_str} (F11)", duration=2.0, color=(74, 222, 128))

    def trigger_toggle_vsync() -> None:
        new_vsync = VSyncMode.OFF if render_ctx.window.vsync == VSyncMode.ON else VSyncMode.ON
        render_ctx.window.set_vsync(new_vsync)
        engine_tweaks.vsync_enabled = (new_vsync == VSyncMode.ON)
        toast.show(f"VSync: {new_vsync.name}", duration=2.0, color=(74, 222, 128))

    # Audio Tweaks
    game_tweaks.add_float("Audio", "Master Volume", default=1.0, min_val=0.0, max_val=1.0, step=0.05, on_changed=lambda v: audio_engine.mixer.set_volume("master", v))
    game_tweaks.add_float("Audio", "SFX Volume", default=1.0, min_val=0.0, max_val=1.0, step=0.05, on_changed=lambda v: audio_engine.mixer.set_volume("sfx", v))
    game_tweaks.add_action("Audio", "Play Test Click", lambda: audio_engine.play_sound("click", volume=0.8))

    # Actions
    game_tweaks.add_action("Character", "Reset to Spawn", trigger_reset_player)
    game_tweaks.add_action("Camera", "Center Camera View", trigger_reset_camera)
    game_tweaks.add_action("Camera", "Toggle Mode (Follow/Free)", trigger_toggle_camera)
    game_tweaks.add_action("Display", "Toggle Fullscreen (F11)", trigger_toggle_fullscreen)
    game_tweaks.add_action("Display", "Toggle VSync", trigger_toggle_vsync)
    game_tweaks.add_action("Display", "Set 1280x720 (HD)", lambda: trigger_set_resolution(1280, 720))
    game_tweaks.add_action("Display", "Set 1600x900 (HD+)", lambda: trigger_set_resolution(1600, 900))
    game_tweaks.add_action("Display", "Set 1920x1080 (FHD)", lambda: trigger_set_resolution(1920, 1080))
    game_tweaks.add_action("Display", "Set 2560x1440 (QHD)", lambda: trigger_set_resolution(2560, 1440))
    game_tweaks.add_action("Physics", "Spawn Spheres", trigger_spawn_batch)
    game_tweaks.add_action("Physics", "Clear Spheres", trigger_clear_spheres)
    game_tweaks.add_action("PIE", "Save Snapshot", trigger_pie_save)
    game_tweaks.add_action("PIE", "Restore Snapshot", trigger_pie_restore)

    # Watches
    game_tweaks.add_watch("Display", "Resolution", lambda: f"{render_ctx.window.width}x{render_ctx.window.height}")
    game_tweaks.add_watch("Display", "Mode", lambda: render_ctx.window.mode.name)
    game_tweaks.add_watch("Display", "VSync", lambda: render_ctx.window.vsync.name)
    game_tweaks.add_watch("Entities", "Active Count", lambda: ecs.active_count)
    game_tweaks.add_watch("KCC", "Grounded", lambda: motor.state.is_grounded)
    game_tweaks.add_watch("KCC", "Speed", lambda: f"{motor.state.horizontal_speed:.1f} m/s")
    game_tweaks.add_watch("Camera", "Distance", lambda: f"{cam.current_distance:.1f} m")

    toast.show("PyMordial Engine Ready — F1 (Perf), F2 (Graphics), F3 (Game), F9 (Mouse Capture)", duration=4.5, color=(56, 189, 248))

    # Pre-cache MeshAllocation objects for zero-allocation MDI batch rendering
    alloc_cube = pipeline.mega_buffer.allocations["cube"]
    alloc_capsule = pipeline.mega_buffer.allocations["capsule"]
    alloc_sphere = pipeline.mega_buffer.allocations["sphere"]

    num_cubes = len(cube_ids)
    num_monoliths = len(monolith_ids)

    # Zero-allocation MDI batch list: cubes, monoliths, character capsule, dynamic spheres
    draw_batches: list[tuple[MeshAllocation, int, int]] = [
        (alloc_cube, num_cubes, 0),
        (alloc_monolith, num_monoliths, num_cubes),
        (alloc_capsule, 1, num_cubes + num_monoliths),
        (alloc_sphere, len(sphere_ids), num_cubes + num_monoliths + 1),
    ]

    # Simulation State

    # Simulation State
    move_fwd = 0.0
    move_strafe = 0.0
    is_sprinting = False
    jump_requested = False

    def fixed_update(dt: float) -> None:
        nonlocal jump_requested
        # Sync live tweaks to motor config
        motor.config.walk_speed = walk_spd_tweak.value
        motor.config.run_speed = run_spd_tweak.value
        motor.config.jump_force = jump_force_tweak.value

        with monitor.scope("kcc_motor"):
            motor.update(
                dt=dt,
                move_input=(move_fwd, move_strafe),
                is_running=is_sprinting,
                jump_requested=jump_requested,
                camera_yaw_deg=cam.yaw_deg,
            )
            jump_requested = False

        with monitor.scope("physics_step"):
            physics.step_simulation(dt)

        with monitor.scope("ecs_sync"):
            physics.sync_to_ecs(ecs)

    loop.on_fixed_update = fixed_update

    # Main Engine Loop
    clock = pygame.time.Clock()
    running = True
    frame_idx = 0
    footstep_timer = 0.0

    # Zero-allocation reusable per-frame vectors
    scratch_fwd_vec = [0.0, 0.0, 0.0]
    scratch_sun_dir = [0.0, 0.0, 0.0]

    while running:
        raw_dt_ms = clock.tick(0)
        dt = min(raw_dt_ms * 0.001, 0.1)

        monitor.begin_frame()

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

        # Mouse capture toggle (F9)
        if (
            input_mgr.is_action_pressed("debug_mouse_capture")
            or input_mgr.is_action_pressed("debug_mouse_capture_toggle")
        ):
            new_grab = not input_mgr.is_mouse_grabbed
            input_mgr.set_mouse_grab(new_grab)
            input_mgr._was_mouse_grabbed = new_grab
            state_str = "CAPTURED" if new_grab else "FREE"
            toast.show(f"Mouse Capture: {state_str} (F9)", duration=2.0, color=(56, 189, 248))

        # Camera & Window Hotkeys (F10 = Camera Toggle, F11 = Fullscreen Toggle)
        if input_mgr.is_action_pressed("toggle_camera"):
            trigger_toggle_camera()
        if input_mgr.is_action_pressed("toggle_fullscreen"):
            trigger_toggle_fullscreen()

        # Direct Debug Panel Hotkeys (F1 = Multi-Style Perf + Graphics, F2 = Graphics Options, F3 = Game Tweaks)
        if input_mgr.is_action_pressed("debug_perf") or input_mgr.is_action_pressed("debug_perf_toggle") or input_mgr.is_action_pressed("debug_menu_toggle"):
            style = debug_menu.cycle_f1()
            # If opening F1, also open Graphics Options and free the mouse
            if style > 0 and not debug_menu.show_graphics:
                debug_menu.show_graphics = True
            if debug_menu.show_graphics and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
            if style == 1:
                toast.show("Debug: Performance HUD & Graphics Options [F1/F2] (Mouse Free)", duration=2.0, color=(56, 189, 248))
            elif style == 2:
                toast.show("Debug: Performance Profiler [Expanded] (F1)", duration=1.5, color=(147, 197, 253))
            else:
                toast.show("Debug: Performance Monitor Closed", duration=1.5, color=(147, 197, 253))
        elif input_mgr.is_action_pressed("debug_graphics") or input_mgr.is_action_pressed("debug_graphics_toggle"):
            is_open = debug_menu.toggle_graphics()
            if is_open and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
            msg = "Debug: Graphics Options [F2] (Mouse Free)" if is_open else "Graphics Options Closed"
            toast.show(msg, duration=1.5, color=(147, 197, 253))
        elif input_mgr.is_action_pressed("debug_game") or input_mgr.is_action_pressed("debug_game_toggle"):
            is_open = debug_menu.toggle_game_tweaks()
            if is_open and input_mgr.is_mouse_grabbed:
                input_mgr.set_mouse_grab(False)
                input_mgr._was_mouse_grabbed = False
            msg = "Debug: Game Developer Tweaks [F3] (Mouse Free)" if is_open else "Game Tweaks Closed"
            toast.show(msg, duration=1.5, color=(147, 197, 253))

        # Secondary wireframe hotkeys (F7 / F8)
        if input_mgr.is_action_pressed("toggle_wireframe"):
            engine_tweaks.toggle_wireframe()
            wireframe_tweak.value = engine_tweaks.show_wireframe
            state_str = "ON" if engine_tweaks.show_wireframe else "OFF"
            toast.show(f"Mesh Wireframe: {state_str}", duration=2.0, color=(56, 189, 248))

        if input_mgr.is_action_pressed("toggle_physics_wireframe"):
            engine_tweaks.toggle_physics_colliders()
            physics_wire_tweak.value = engine_tweaks.show_physics_colliders
            state_str = "ON" if engine_tweaks.show_physics_colliders else "OFF"
            toast.show(f"Physics Gizmos: {state_str}", duration=2.0, color=(74, 222, 128))

        # Gameplay & Character Input (Never locked by debug menu)
        if not args.headless:
            move_vec = input_mgr.get_vector2("move")
            move_strafe = move_vec[0]
            move_fwd = move_vec[1]
            is_sprinting = input_mgr.is_action_down("sprint")

            if input_mgr.is_action_pressed("jump"):
                jump_requested = True
                if motor.state.is_grounded:
                    audio_engine.play_sound("jump", position=char_start_pos, volume=0.8)
            if input_mgr.is_action_pressed("reset_player"):
                trigger_reset_player()
            if input_mgr.is_action_pressed("reset_camera"):
                trigger_reset_camera()
            if input_mgr.is_action_pressed("spawn_spheres"):
                trigger_spawn_batch()
            if input_mgr.is_action_pressed("clear_spheres"):
                trigger_clear_spheres()
            if input_mgr.is_action_pressed("pie_save"):
                trigger_pie_save()
            if input_mgr.is_action_pressed("pie_restore"):
                trigger_pie_restore()

            if input_mgr.is_action_pressed("cycle_tonemap"):
                m = engine_tweaks.cycle_tonemapper()
                toast.show(f"Tonemapper: {m}", duration=2.0, color=(192, 132, 252))

            if input_mgr.is_action_pressed("cycle_gbuffer"):
                g = engine_tweaks.cycle_gbuffer_debug()
                toast.show(f"G-Buffer Mode: {g.name}", duration=2.0, color=(251, 146, 60))

            if input_mgr.is_action_pressed("toggle_sun_ray"):
                engine_tweaks.show_sun_ray = not engine_tweaks.show_sun_ray
                toast.show(f"Sun Ray Gizmo: {engine_tweaks.show_sun_ray}", duration=2.0, color=(250, 204, 21))

            if input_mgr.is_action_pressed("cycle_preset"):
                p = engine_tweaks.cycle_quality_preset()
                toast.show(f"Preset: {p.value.upper()}", duration=2.0, color=(56, 189, 248))

            # Camera Look & Orbit (Active only when mouse is grabbed / captured)
            if input_mgr.is_mouse_grabbed:
                look_vec = input_mgr.get_vector2("look")
                if abs(look_vec[0]) > 0.01 or abs(look_vec[1]) > 0.01:
                    sens = cam_sens_tweak.value
                    cam.yaw_deg -= look_vec[0] * 120.0 * sens * dt
                    pitch_mult = 1.0 if invert_pitch_tweak.value else -1.0
                    cam.pitch_deg = max(
                        cam.config.min_pitch_deg,
                        min(cam.config.max_pitch_deg, cam.pitch_deg + pitch_mult * look_vec[1] * 90.0 * sens * dt),
                    )

                dx, dy = input_mgr.mouse_delta
                if dx != 0 or dy != 0:
                    cam.handle_mouse_orbit(dx, dy)

                if input_mgr.mouse_wheel != 0.0:
                    cam.handle_zoom(input_mgr.mouse_wheel)

                if input_mgr.is_action_down("zoom_in"):
                    cam.handle_zoom(12.0 * dt)
                if input_mgr.is_action_down("zoom_out"):
                    cam.handle_zoom(-12.0 * dt)
            else:
                # Automated script behavior for headless validation & testing
                if frame_idx < 30:
                    move_fwd = 1.0
                    move_strafe = 0.0
                    is_sprinting = False
                elif frame_idx < 60:
                    move_fwd = 1.0
                    move_strafe = 0.0
                    is_sprinting = True
                elif frame_idx == 60:
                    jump_requested = True
                else:
                    move_fwd = 0.0
                    move_strafe = 0.0

        # Step deterministic simulation (60 Hz fixed tick + sub-frame NLERP)
        loop.step_frame(dt)

        # Synchronize EngineTweaks to pipeline & dev tweaks
        if engine_tweaks.shadow_resolution != pipeline.csm.atlas_size:
            pipeline.csm.resize_atlas(engine_tweaks.shadow_resolution)
        pipeline.config.shadow_resolution = engine_tweaks.shadow_resolution
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
        pipeline.config.aa_mode = getattr(engine_tweaks, "aa_mode", "TAA" if engine_tweaks.taa_enabled else "OFF")
        pipeline.config.taa_enabled = (pipeline.config.aa_mode == "TAA")
        pipeline.config.taa_feedback = getattr(engine_tweaks, "taa_feedback", 0.92)
        pipeline.config.taa_sharpness = getattr(engine_tweaks, "taa_sharpness", 0.35)
        pipeline.config.clustered_lights_enabled = engine_tweaks.point_lights_enabled
        wireframe_tweak.value = engine_tweaks.show_wireframe
        physics_wire_tweak.value = engine_tweaks.show_physics_colliders

        # Synchronize Camera & Character (Sub-Frame Interpolated for Jitter-Free Tracking)
        char_idx = ecs.pool.get_dense_index(char_id)
        if char_idx >= 0:
            char_world_pos = (
                float(ecs.world_transforms[char_idx, 12]),
                float(ecs.world_transforms[char_idx, 13]),
                float(ecs.world_transforms[char_idx, 14]),
            )
        else:
            char_world_pos = (0.0, 0.0, 0.0)

        # Procedural footstep audio updates
        if motor.state.is_grounded and motor.state.horizontal_speed > 1.5:
            step_interval = 0.35 if is_sprinting else 0.55
            footstep_timer += dt
            if footstep_timer >= step_interval:
                footstep_timer = 0.0
                audio_engine.play_sound("footstep", position=char_world_pos, volume=0.6)
        else:
            footstep_timer = 0.3

        # Update Camera Manager and Active Camera
        camera_mgr.update(dt)
        if camera_mgr.active_camera_name == "free":
            free_cam.move(
                dt=dt,
                forward_axis=move_fwd,
                right_axis=move_strafe,
                up_axis=1.0 if input_mgr.is_action_down("jump") else (-1.0 if input_mgr.is_action_down("sprint") else 0.0),
                boost=input_mgr.is_action_down("sprint"),
            )
            if input_mgr.is_mouse_grabbed or input_mgr.is_mouse_down(MouseButton.RIGHT):
                dx, dy = input_mgr.mouse_delta
                if dx != 0 or dy != 0:
                    free_cam.handle_mouse_look(dx, dy)
            camera_pos = (float(free_cam.position[0]), float(free_cam.position[1]), float(free_cam.position[2]))
            camera_target = (float(free_cam.target[0]), float(free_cam.target[1]), float(free_cam.target[2]))
        else:
            with monitor.scope("camera_update"):
                camera_pos, camera_target = cam.update(
                    dt=dt,
                    character_pos=char_world_pos,
                    physics=physics,
                )

        # Update 3D Audio spatial listener
        scratch_fwd_vec[0] = camera_target[0] - camera_pos[0]
        scratch_fwd_vec[1] = camera_target[1] - camera_pos[1]
        scratch_fwd_vec[2] = camera_target[2] - camera_pos[2]
        audio_engine.update(dt, camera_position=camera_pos, camera_forward=scratch_fwd_vec)

        # Compute Sun vector from Azimuth & Elevation
        rad_sun_az = math.radians(engine_tweaks.sun_angle_deg)
        rad_sun_el = math.radians(engine_tweaks.sun_elevation_deg)
        sun_x = math.cos(rad_sun_el) * math.sin(rad_sun_az)
        sun_y = -math.sin(rad_sun_el)
        sun_z = math.cos(rad_sun_el) * math.cos(rad_sun_az)
        scratch_sun_dir[0] = sun_x
        scratch_sun_dir[1] = sun_y
        scratch_sun_dir[2] = sun_z

        # 3D Immediate-Mode Wireframe Debug Drawing
        pipeline.debug.update(dt)
        if engine_tweaks.show_physics_colliders:
            # Ground platform box wireframe
            pipeline.debug.draw_box(
                center=(0.0, -0.5, 0.0),
                size=(60.0, 1.0, 60.0),
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
            # Monolith bounding spheres from cooked mesh header bounds
            for mpos in monolith_positions:
                center_y = mpos[1] + cooked_mesh.bounds.sphere_center[1]
                pipeline.debug.draw_sphere(
                    center=(mpos[0], center_y, mpos[2]),
                    radius=cooked_mesh.bounds.sphere_radius,
                    color=(0.95, 0.70, 0.20, 0.6),
                )
            # Stepping course wireframe boxes
            for b_pos, b_size in step_boxes:
                pipeline.debug.draw_box(
                    center=b_pos,
                    size=b_size,
                    color=(0.35, 0.75, 0.45, 0.5),
                )
            # Dynamic spheres colliders (Interpolated Sub-Frame Coordinates)
            for sph in sphere_ids:
                s_idx = ecs.pool.get_dense_index(sph)
                if s_idx >= 0:
                    s_pos = (
                        float(ecs.world_transforms[s_idx, 12]),
                        float(ecs.world_transforms[s_idx, 13]),
                        float(ecs.world_transforms[s_idx, 14]),
                    )
                    s_r = float(ecs.scales[s_idx, 0])
                    pipeline.debug.draw_sphere(
                        center=s_pos,
                        radius=s_r,
                        color=(0.95, 0.45, 0.25, 0.7),
                    )

        if engine_tweaks.show_sun_ray:
            pipeline.debug.draw_ray(
                origin=camera_target,
                direction=(-sun_x, -sun_y, -sun_z),
                length=10.0,
                color=(1.0, 0.9, 0.2, 1.0),
            )

        # Dynamic Sphere Batch Count Update
        num_spheres = len(sphere_ids)
        if draw_batches[3][1] != num_spheres:
            draw_batches[3] = (alloc_sphere, num_spheres, num_cubes + num_monoliths + 1)

        # Update Clustered Dynamic Local Lights (SSBO 3)
        pipeline.clear_point_lights()
        if engine_tweaks.point_lights_enabled:
            t = loop.elapsed_time
            for i, (bx, by, bz, r, col, intensity) in enumerate(VALIDATION_LANTERNS):
                angle = t * 0.7 + i * 1.047
                px = bx + math.cos(angle) * 3.0
                py = by + math.sin(t * 1.5 + i) * 0.6
                pz = bz + math.sin(angle) * 3.0
                pipeline.add_point_light((px, py, pz), radius=r, color=col, intensity=intensity)
                if engine_tweaks.show_physics_colliders:
                    pipeline.debug.draw_sphere((px, py, pz), radius=0.20, color=(col[0], col[1], col[2], 0.9))

        # Render ModernGL 4.5 Core 3D frame via MDI with Tier 1 profiling
        with monitor.scope("render_pipeline"):
            pipeline.render_frame(
                ecs=ecs,
                camera_pos=camera_pos,
                camera_target=camera_target,
                time_elapsed=loop.elapsed_time,
                sun_dir=scratch_sun_dir,
                sun_lux=pipeline.config.sun_intensity,
                draw_batches=draw_batches,
                debug_draw=pipeline.debug,
            )



        # Render 3-Tier Glassmorphic Debug Menu if visible
        if debug_menu.visible or (debug_menu.toast is not None and len(debug_menu.toast.get_active()) > 0):
            pipeline.post_process.final_fbo.use()
            debug_menu.render(pipeline.post_process.final_fbo)
            if not args.headless:
                render_ctx.ctx.copy_framebuffer(render_ctx.ctx.screen, pipeline.post_process.final_fbo)

        if not args.headless:
            render_ctx.swap_buffers()

        frame_idx += 1
        if args.frames > 0 and frame_idx >= args.frames:
            break

    # Automated Screenshot Capture
    if args.screenshot:
        out_path = Path(args.screenshot).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data = pipeline.post_process.final_fbo.read(components=4, dtype="f1")
        img = Image.frombytes("RGBA", (render_ctx.width, render_ctx.height), raw_data)
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img.save(out_path)
        print(f"Validation screenshot saved to: {out_path}")

    # Clean engine shutdown
    debug_menu.destroy()
    pipeline.destroy()
    render_ctx.destroy()
    audio_engine.shutdown()
    pygame.quit()
    print("PyMordial Engine Master Validation completed successfully.")


if __name__ == "__main__":
    main()
