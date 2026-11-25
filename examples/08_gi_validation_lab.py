"""GI Validation Lab — deterministic scene for comparing GI implementations.

A compact, reproducible laboratory for manually and programmatically comparing
GI Off / SSGI / RMGI / AMRGI (Low..Cinematic). It exercises neutral diffuse
bounce, RGB color bleeding, local-light bounce, shadow-aware bounce, emissive
bounce, near/mid/far coverage, mixed interior/exterior, thin-geometry leakage,
temporal accumulation, moving-object disocclusion, moving-light response, and
screen-space limitations.

This is a VALIDATION example, not a showcase. It does not modify or tune any GI
implementation; it only configures the existing renderer through its public
`RenderConfig` interface.

Run::

    python examples/08_gi_validation_lab.py
    python examples/08_gi_validation_lab.py --headless --frames 30

Controls (see on-screen help):
    1-8   camera presets (Reference, RGB, Thin, Shadow/Emissive, Doorway,
          Range, Motion, Screen-Space)
    G     cycle GI mode (OFF -> SSGI -> RMGI -> AMRGI)
    [ / ] cycle AMRGI quality (Low/Medium/High/Ultra/Cinematic)
    E     equalized comparison state (GI intensity 1.0, fixed exposure)
    L     cycle light configuration (Sun only / +Point / +Spot / +Emissive / All)
    M     toggle moving object
    N     toggle moving light
    R     reset motion/time
    O     toggle AO
    H     toggle help overlay
"""

from __future__ import annotations

import math

import pygame
from _common import FlyCameraControls, add_box, create_app, create_parser

from engine.app.module import ProjectModule
from engine.camera import FreeFlyCamera
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset
from engine.ui import UILabel, UIPanel, UIScreen

# ---------------------------------------------------------------------------
# Scene constants (deterministic)
# ---------------------------------------------------------------------------

ROOM_W = 12.0
ROOM_L = 12.0
ROOM_H = 4.0
WALL = 0.3
DOOR_W = 4.0
DOOR_H = 3.2

# Courtyard extends north (negative Z) beyond the doorway.
COURT_L = 20.0
COURT_W = 16.0

NEUTRAL = (0.62, 0.62, 0.64)
NEUTRAL_DARK = (0.42, 0.42, 0.45)
RED = (0.85, 0.12, 0.10)
GREEN = (0.10, 0.78, 0.18)
BLUE = (0.10, 0.22, 0.88)
WARM = (1.0, 0.55, 0.18)

AMRGI_QUALITIES = (
    GraphicsQuality.LOW,
    GraphicsQuality.MEDIUM,
    GraphicsQuality.HIGH,
    GraphicsQuality.ULTRA,
    GraphicsQuality.CINEMATIC,
)
AMRGI_QUALITY_LABELS = ("Low", "Medium", "High", "Ultra", "Cinematic")

# Camera presets: (name, position, target, fov)
CAMERA_PRESETS = (
    ("Reference Room", (0.0, 1.7, 4.6), (0.0, 1.2, -1.0), 60.0),
    ("RGB Bounce", (-3.0, 1.5, 2.6), (-3.0, 1.3, -4.6), 62.0),
    ("Thin Geometry", (4.4, 1.6, 3.6), (4.4, 1.0, -1.5), 60.0),
    ("Shadow / Emissive", (0.0, 1.7, 4.8), (0.0, 1.0, -3.0), 60.0),
    ("Doorway (Mixed)", (0.0, 1.7, 3.0), (0.0, 1.6, -14.0), 62.0),
    ("Range / Long Depth", (0.0, 2.2, 5.0), (0.0, 1.4, -22.0), 58.0),
    ("Motion / Disocclusion", (3.2, 1.7, 3.6), (1.0, 0.9, -1.0), 60.0),
    ("Screen-Space Limit", (-5.2, 1.7, 2.4), (-1.0, 0.9, -2.0), 60.0),
)

# Deterministic motion parameters.
MOTION_PERIOD = 6.0
MOTION_A = (1.2, 0.9, -1.0)
MOTION_B = (3.6, 0.9, -1.0)
LIGHT_PERIOD = 8.0
LIGHT_CENTER = (2.4, 2.2, -1.0)
LIGHT_RADIUS = 2.2


def _wall(app, position, dimensions, color=NEUTRAL, roughness=0.85) -> int:
    return add_box(app, position, dimensions, color, roughness, 0.0)


def _build_room(app) -> None:
    """Central enclosed validation room with a large north doorway."""
    half_w = ROOM_W * 0.5
    half_l = ROOM_L * 0.5

    # Floor + ceiling.
    _wall(app, (0.0, -0.15, 0.0), (ROOM_W, WALL, ROOM_L), NEUTRAL_DARK, 0.9)
    _wall(app, (0.0, ROOM_H + WALL * 0.5, 0.0), (ROOM_W, WALL, ROOM_L), NEUTRAL, 0.9)

    # Side walls.
    _wall(app, (-half_w, ROOM_H * 0.5, 0.0), (WALL, ROOM_H, ROOM_L))
    _wall(app, (half_w, ROOM_H * 0.5, 0.0), (WALL, ROOM_H, ROOM_L))

    # South wall (behind camera start).
    _wall(app, (0.0, ROOM_H * 0.5, half_l), (ROOM_W, ROOM_H, WALL))

    # North wall with a large doorway into the courtyard.
    side = (ROOM_W - DOOR_W) * 0.5
    for sign in (-1.0, 1.0):
        x = sign * (DOOR_W * 0.5 + side * 0.5)
        _wall(app, (x, ROOM_H * 0.5, -half_l), (side, ROOM_H, WALL))
    # Lintel above the doorway.
    _wall(app, (0.0, DOOR_H + (ROOM_H - DOOR_H) * 0.5, -half_l), (DOOR_W, ROOM_H - DOOR_H, WALL))


def _build_courtyard(app) -> None:
    """Open courtyard north of the doorway with near/mid/far targets."""
    z0 = -ROOM_L * 0.5 - COURT_L * 0.5
    # Courtyard floor + far wall + side walls (open sky above).
    _wall(app, (0.0, -0.15, z0), (COURT_W, WALL, COURT_L), NEUTRAL_DARK, 0.9)
    _wall(app, (0.0, ROOM_H * 0.5, z0 - COURT_L * 0.5), (COURT_W, ROOM_H, WALL))
    _wall(app, (-COURT_W * 0.5, ROOM_H * 0.5, z0), (WALL, ROOM_H, COURT_L))
    _wall(app, (COURT_W * 0.5, ROOM_H * 0.5, z0), (WALL, ROOM_H, COURT_L))

    # Near (red), mid (green), far (blue) bounce targets on neutral pedestals.
    for color, z, size in ((RED, -8.0, 1.0), (GREEN, -16.0, 1.4), (BLUE, -26.0, 1.8)):
        add_box(app, (0.0, size * 0.5, z), (size, size, size), color, 0.9, 0.0)
        add_box(app, (0.0, 0.05, z), (size * 2.4, 0.1, size * 2.4), NEUTRAL, 0.9, 0.0)


def _build_neutral_reference(app) -> None:
    """Central diffuse primitives + metallic/diffuse control pair."""
    add_box(app, (-1.6, 0.6, -1.0), (1.2, 1.2, 1.2), NEUTRAL, 0.8, 0.0)
    app.ecs.create_entity(shape="sphere", position=(0.0, 0.7, -1.0), dimensions=(1.4, 1.4, 1.4),
                          color=NEUTRAL, roughness=0.8, metallic=0.0, is_static=True)
    app.ecs.create_entity(shape="cylinder", position=(1.6, 0.7, -1.0), dimensions=(0.9, 1.4, 0.9),
                          color=NEUTRAL, roughness=0.8, metallic=0.0, is_static=True)
    # Metallic vs diffuse control spheres.
    app.ecs.create_entity(shape="sphere", position=(-3.0, 0.5, 1.2), dimensions=(1.0, 1.0, 1.0),
                          color=(0.9, 0.9, 0.92), roughness=0.15, metallic=1.0, is_static=True)
    app.ecs.create_entity(shape="sphere", position=(-1.6, 0.5, 1.2), dimensions=(1.0, 1.0, 1.0),
                          color=(0.9, 0.9, 0.92), roughness=0.15, metallic=0.0, is_static=True)


def _build_rgb_zone(app) -> None:
    """Three saturated matte panels with a neutral receiver in front."""
    for color, x in ((RED, -4.6), (GREEN, -3.0), (BLUE, -1.4)):
        add_box(app, (x, 1.4, -4.6), (1.2, 2.4, 0.2), color, 0.95, 0.0)
    # Neutral receiver ~1 m in front of the panels.
    add_box(app, (-3.0, 0.9, -3.2), (4.4, 1.8, 0.2), NEUTRAL, 0.85, 0.0)


def _build_thin_zone(app) -> None:
    """Thin slabs of decreasing thickness + a narrow post and overhang."""
    x0 = 4.4
    for i, thickness in enumerate((0.20, 0.10, 0.05, 0.02)):
        z = -1.0 - i * 0.9
        add_box(app, (x0, 1.2, z), (thickness, 2.4, 1.6), NEUTRAL, 0.8, 0.0)
    # Colored source behind the thin divider, neutral receiver in front.
    add_box(app, (x0, 1.2, -5.2), (1.6, 2.4, 0.2), RED, 0.9, 0.0)
    add_box(app, (x0, 1.2, 0.4), (1.6, 2.4, 0.2), NEUTRAL, 0.85, 0.0)
    # Narrow post + thin horizontal overhang.
    add_box(app, (x0 + 1.4, 1.2, -2.0), (0.12, 2.4, 0.12), NEUTRAL, 0.8, 0.0)
    add_box(app, (x0 + 1.4, 2.3, -2.0), (1.2, 0.06, 1.2), NEUTRAL, 0.8, 0.0)


def _build_shadow_emissive_zone(app) -> None:
    """Blocker splitting a colored source + an emissive panel with receiver."""
    # Colored floor patch under a blocker.
    add_box(app, (0.0, 0.02, -3.4), (3.0, 0.04, 2.4), RED, 0.95, 0.0)
    # Blocker casting a shadow across the colored patch.
    add_box(app, (0.0, 1.6, -3.4), (0.4, 1.2, 2.4), NEUTRAL_DARK, 0.85, 0.0)
    # Neutral receiver beside the patch.
    add_box(app, (2.4, 0.9, -3.4), (0.2, 1.8, 2.4), NEUTRAL, 0.85, 0.0)
    # Emissive panel (warm) with a neutral receiver in front.
    app.ecs.create_entity(
        shape="cube", position=(-2.6, 1.4, -4.4), dimensions=(1.6, 2.0, 0.2),
        color=WARM, roughness=0.9, metallic=0.0, emissive_intensity=3.0, is_static=True,
    )
    add_box(app, (-2.6, 0.9, -3.0), (1.8, 1.8, 0.2), NEUTRAL, 0.85, 0.0)


def _build_motion_zone(app) -> int:
    """Deterministic moving cube that occludes/reveals a colored source."""
    add_box(app, (2.4, 1.0, -1.0), (1.4, 2.0, 0.2), GREEN, 0.9, 0.0)  # colored source
    moving = app.ecs.create_entity(
        shape="cube", position=MOTION_A, dimensions=(0.9, 0.9, 0.9),
        color=NEUTRAL, roughness=0.8, metallic=0.0, is_static=False,
    )
    return moving


def build_scene(app) -> dict:
    """Builds the deterministic validation scene and returns its metadata."""
    _build_room(app)
    _build_courtyard(app)
    _build_neutral_reference(app)
    _build_rgb_zone(app)
    _build_thin_zone(app)
    _build_shadow_emissive_zone(app)
    moving_id = _build_motion_zone(app)

    # Controlled lighting: one stable directional sun; local lights added by
    # the overlay only when their test is active.
    app.pipeline.lights_pass.clear()

    camera = FreeFlyCamera(position=CAMERA_PRESETS[0][1], yaw_deg=0.0, pitch_deg=0.0, move_speed=5.0)
    app.camera_manager.register_camera("gi_lab_flycam", camera, make_active=True)
    app.add_module(FlyCameraControls(camera))

    overlay = GIValidationOverlay(camera, moving_id)
    app.add_module(overlay)

    return {
        "moving_id": moving_id,
        "overlay": overlay,
        "camera": camera,
    }


class GIValidationOverlay(ProjectModule):
    """GI comparison controls, camera presets, and deterministic motion."""

    name = "GIValidationOverlay"

    def __init__(self, camera: FreeFlyCamera, moving_id: int) -> None:
        self.camera = camera
        self.moving_id = moving_id
        self.gi_mode = "OFF"
        self.amrgi_quality_idx = 1  # Medium
        self.light_config = 0  # 0 sun only, 1 +point, 2 +spot, 3 +emissive, 4 all
        self.motion_enabled = True
        self.moving_light_enabled = False
        self.ao_enabled = False
        self.show_help = True
        self._time = 0.0
        self._point_light = None
        self._spot_light = None
        self._moving_light = None
        self._emissive_id = None
        self.help_label: UILabel | None = None
        self.status_label: UILabel | None = None

    # -- lifecycle -----------------------------------------------------

    def on_attach(self, app) -> None:
        for action, key in (
            ("gi_lab_gi", pygame.K_g),
            ("gi_lab_quality", pygame.K_RIGHTBRACKET),
            ("gi_lab_quality_prev", pygame.K_LEFTBRACKET),
            ("gi_lab_equalize", pygame.K_e),
            ("gi_lab_lights", pygame.K_l),
            ("gi_lab_motion", pygame.K_m),
            ("gi_lab_moving_light", pygame.K_n),
            ("gi_lab_reset", pygame.K_r),
            ("gi_lab_ao", pygame.K_o),
            ("gi_lab_help", pygame.K_h),
        ):
            app.input_manager.bind_action(action, keys=[key])

        screen = UIScreen(name="GIValidationHUD", is_modal=False)
        panel = screen.add_child(UIPanel(x=16.0, y=16.0, w=560.0, h=210.0))
        panel.add_child(UILabel("PYMORDIAL | GI VALIDATION LAB", x=14.0, y=10.0, w=530.0, h=26.0, font_size=17))
        self.status_label = panel.add_child(UILabel("", x=14.0, y=40.0, w=530.0, h=24.0, font_size=14))
        self.help_label = panel.add_child(UILabel(
            "1-8 camera | G GI mode | [ ] AMRGI quality | E equalize | L lights\n"
            "M motion | N moving light | R reset | O AO | H help",
            x=14.0, y=70.0, w=530.0, h=120.0, font_size=13,
        ))
        app.ui.set_screen(screen)

        self._apply_analytical_config(app)
        self._apply_gi_mode(app)
        self._apply_lights(app)

    def on_detach(self, app) -> None:
        pass

    # -- configuration -------------------------------------------------

    def _apply_analytical_config(self, app) -> None:
        """Fixed, deterministic analytical render state (no auto exposure)."""
        cfg = app.pipeline.config
        cfg.auto_exposure_enabled = False
        cfg.exposure = 1.0
        cfg.exposure_compensation = 0.0
        cfg.ssr_enabled = False
        cfg.bloom_enabled = False
        cfg.dof_enabled = False
        cfg.motion_blur_enabled = False
        cfg.volumetric_fog_enabled = False
        cfg.lens_flare_enabled = False
        cfg.chromatic_aberration_enabled = False
        cfg.vignette_enabled = False
        cfg.film_grain_enabled = False
        cfg.ao_mode = "OFF"
        cfg.aa_mode = "FXAA"
        cfg.taa_enabled = False
        cfg.sun_intensity = 3.0
        cfg.ambient_factor = 0.03
        app.pipeline.apply_config(cfg)

    def _apply_gi_mode(self, app) -> None:
        cfg = app.pipeline.config
        cfg.gi_mode = self.gi_mode
        if self.gi_mode == "AMRGI":
            preset = get_quality_preset(AMRGI_QUALITIES[self.amrgi_quality_idx])
            for field in (
                "amrgi_resolution_scale", "amrgi_ray_count", "amrgi_near_steps",
                "amrgi_mid_steps", "amrgi_far_steps", "amrgi_near_refine_steps",
                "amrgi_mid_rate", "amrgi_far_rate",
            ):
                setattr(cfg, field, getattr(preset, field))
        # Equalized comparison: GI intensity is always 1.0 for every mode.
        cfg.amrgi_intensity = 1.0
        cfg.ssgi_intensity = 1.0
        cfg.rmgi_intensity = 1.0
        app.pipeline.apply_config(cfg)

    def _apply_lights(self, app) -> None:
        app.pipeline.clear_point_lights()
        app.pipeline.clear_spot_lights()
        self._point_light = None
        self._spot_light = None
        self._moving_light = None
        if self.light_config in (1, 4):
            self._point_light = app.pipeline.add_point_light(
                (-3.0, 2.6, -1.0), radius=6.0, color=(1.0, 0.85, 0.6), intensity=6.0
            )
        if self.light_config in (2, 4):
            self._spot_light = app.pipeline.add_spot_light(
                position=(3.0, 3.6, 1.0), direction=(0.0, -1.0, -0.4), radius=10.0,
                inner_cone_angle=math.radians(16.0), outer_cone_angle=math.radians(28.0),
                color=(0.7, 0.85, 1.0), intensity=10.0, cast_shadow=True,
            )
        if self.moving_light_enabled:
            self._moving_light = app.pipeline.add_point_light(
                LIGHT_CENTER, radius=5.0, color=(1.0, 0.4, 0.2), intensity=6.0
            )

    # -- update --------------------------------------------------------

    def on_update(self, app, dt: float) -> None:
        if self.motion_enabled or self.moving_light_enabled:
            self._time += dt
        self._update_motion(app)
        self._update_moving_light(app)
        self._update_status(app)

    def _update_motion(self, app) -> None:
        if not self.motion_enabled:
            return
        t = (self._time % MOTION_PERIOD) / MOTION_PERIOD
        s = 0.5 - 0.5 * math.cos(t * 2.0 * math.pi)  # smooth 0..1..0
        pos = (
            MOTION_A[0] + (MOTION_B[0] - MOTION_A[0]) * s,
            MOTION_A[1],
            MOTION_A[2] + (MOTION_B[2] - MOTION_A[2]) * s,
        )
        app.ecs.set_position(self.moving_id, pos)

    def _update_moving_light(self, app) -> None:
        if not self.moving_light_enabled or self._moving_light is None:
            return
        t = (self._time % LIGHT_PERIOD) / LIGHT_PERIOD
        ang = t * 2.0 * math.pi
        pos = (
            LIGHT_CENTER[0] + math.cos(ang) * LIGHT_RADIUS,
            LIGHT_CENTER[1],
            LIGHT_CENTER[2] + math.sin(ang) * LIGHT_RADIUS,
        )
        self._moving_light.position[:] = pos

    def _update_status(self, app) -> None:
        if self.status_label is None:
            return
        quality = AMRGI_QUALITY_LABELS[self.amrgi_quality_idx] if self.gi_mode == "AMRGI" else "-"
        lights = ("Sun", "+Point", "+Spot", "+Emissive", "All")[self.light_config]
        self.status_label.text = (
            f"GI: {self.gi_mode} ({quality}) | Lights: {lights} | "
            f"Motion: {'on' if self.motion_enabled else 'off'} | "
            f"MovingLight: {'on' if self.moving_light_enabled else 'off'} | "
            f"AO: {'on' if self.ao_enabled else 'off'}"
        )
        if self.help_label is not None:
            self.help_label.visible = self.show_help

    # -- events --------------------------------------------------------

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False
        key = event.key
        if pygame.K_1 <= key <= pygame.K_8:
            self._set_camera_preset(app, key - pygame.K_1)
            return True
        if key == pygame.K_g:
            modes = ("OFF", "SSGI", "RMGI", "AMRGI")
            self.gi_mode = modes[(modes.index(self.gi_mode) + 1) % len(modes)]
            self._apply_gi_mode(app)
            return True
        if key == pygame.K_RIGHTBRACKET:
            self.amrgi_quality_idx = (self.amrgi_quality_idx + 1) % len(AMRGI_QUALITIES)
            if self.gi_mode == "AMRGI":
                self._apply_gi_mode(app)
            return True
        if key == pygame.K_LEFTBRACKET:
            self.amrgi_quality_idx = (self.amrgi_quality_idx - 1) % len(AMRGI_QUALITIES)
            if self.gi_mode == "AMRGI":
                self._apply_gi_mode(app)
            return True
        if key == pygame.K_e:
            self.equalize(app)
            return True
        if key == pygame.K_l:
            self.light_config = (self.light_config + 1) % 5
            self._apply_lights(app)
            return True
        if key == pygame.K_m:
            self.motion_enabled = not self.motion_enabled
            return True
        if key == pygame.K_n:
            self.moving_light_enabled = not self.moving_light_enabled
            self._apply_lights(app)
            return True
        if key == pygame.K_r:
            self.reset_motion(app)
            return True
        if key == pygame.K_o:
            self.ao_enabled = not self.ao_enabled
            app.pipeline.config.ao_mode = "GTAO" if self.ao_enabled else "OFF"
            app.pipeline.apply_config(app.pipeline.config)
            return True
        if key == pygame.K_h:
            self.show_help = not self.show_help
            return True
        return False

    # -- public helpers (also used by tests) ---------------------------

    def equalize(self, app) -> None:
        """Sets the standard equalized comparison state (intensity 1.0)."""
        cfg = app.pipeline.config
        cfg.exposure = 1.0
        cfg.exposure_compensation = 0.0
        cfg.auto_exposure_enabled = False
        cfg.amrgi_intensity = 1.0
        cfg.ssgi_intensity = 1.0
        cfg.rmgi_intensity = 1.0
        app.pipeline.apply_config(cfg)

    def reset_motion(self, app) -> None:
        """Resets deterministic time and returns the moving object to its start."""
        self._time = 0.0
        app.ecs.set_position(self.moving_id, MOTION_A)
        if self._moving_light is not None:
            self._moving_light.position[:] = (
                LIGHT_CENTER[0] + LIGHT_RADIUS, LIGHT_CENTER[1], LIGHT_CENTER[2]
            )

    def _set_camera_preset(self, app, index: int) -> None:
        if not (0 <= index < len(CAMERA_PRESETS)):
            return
        _, pos, target, fov = CAMERA_PRESETS[index]
        cam = self.camera
        cam.position[:] = pos
        cam.target[:] = target
        cam.fov = fov
        cam.mark_dirty()
        # Camera cut: invalidate GI temporal history so stale samples are dropped.
        for pass_name in ("ssgi_pass", "rmgi_pass", "amrgi_pass"):
            gi_pass = getattr(app.pipeline, pass_name, None)
            if gi_pass is not None and hasattr(gi_pass, "invalidate_history"):
                gi_pass.invalidate_history()


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "GI Validation Lab").parse_args(argv)
    app = create_app(args, "PyMordial | GI Validation Lab")
    build_scene(app)
    app.run()


if __name__ == "__main__":
    main()
