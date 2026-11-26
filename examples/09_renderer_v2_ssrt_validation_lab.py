"""Renderer V2 SSRT Validation Lab — one static hall with six test areas.

A deterministic, manually-driven validation scene for the shared Screen-Space
Ray Tracing (SSRT) subsystem introduced in Renderer V2 Stage 2. It does not
modify or tune any renderer algorithm; it only builds geometry, materials and
lights through the existing public APIs so the current behaviour of AMRGI, AO,
SSR and the shared tracer can be inspected by eye.

The hall is divided into six visually distinct areas:

    NORTH
    +---------------------------+
    |  GI   |   AO   |   SSR     |
    |-------+--------+-----------|
    | THIN  |TEMPORAL|PERFORMANCE|
    +---------------------------+
    SOUTH

Each area is built by its own ``build_*_area`` helper and is reachable by free
flight. Camera presets jump to a useful view of each area.

Run::

    python examples/09_renderer_v2_ssrt_validation_lab.py
    python examples/09_renderer_v2_ssrt_validation_lab.py --headless --frames 30

Controls:
    1-9   camera presets (see CAMERA_PRESETS)
    H     toggle the help overlay
    Home  reset the camera to the starting pose
    WASD/QE + mouse   free flight (see examples/_common.py)

Rendering options (GI mode, AMRGI quality, AO mode, SSR, debug views) are
provided by the engine's existing debug/settings system (F1/F2/F3) and are not
duplicated here.
"""

from __future__ import annotations

import math

import pygame
from _common import FlyCameraControls, add_box, create_app, create_parser

from engine.app.module import ProjectModule
from engine.camera import FreeFlyCamera
from engine.ui import UILabel, UIPanel, UIScreen

# ---------------------------------------------------------------------------
# Materials (simple parameter sets; no textures)
# ---------------------------------------------------------------------------

# Each entry is a keyword set for ``ecs.create_entity``. Purpose is noted so the
# visual tests stay understandable.
MATERIALS: dict[str, dict] = {
    # Neutral matte — the default diffuse receiver used across every area.
    "neutral": {"color": (0.62, 0.62, 0.64), "roughness": 0.85, "metallic": 0.0},
    # Bright neutral matte — high-albedo receiver for bounce/contrast tests.
    "neutral_bright": {"color": (0.88, 0.88, 0.90), "roughness": 0.80, "metallic": 0.0},
    # Dark matte — low-albedo receiver for contrast and AO visibility.
    "dark": {"color": (0.10, 0.10, 0.12), "roughness": 0.90, "metallic": 0.0},
    # Saturated matte walls — drive visible color bleeding.
    "red": {"color": (0.85, 0.12, 0.10), "roughness": 0.92, "metallic": 0.0},
    "green": {"color": (0.10, 0.78, 0.18), "roughness": 0.92, "metallic": 0.0},
    "blue": {"color": (0.10, 0.22, 0.88), "roughness": 0.92, "metallic": 0.0},
    # Smooth metal — mirror-like SSR surface.
    "metal_smooth": {"color": (0.92, 0.92, 0.95), "roughness": 0.05, "metallic": 1.0},
    # Medium-rough metal — SSR with visible roughness response.
    "metal_medium": {"color": (0.85, 0.80, 0.70), "roughness": 0.30, "metallic": 1.0},
    # Rough metal — near the SSR roughness cutoff.
    "metal_rough": {"color": (0.70, 0.70, 0.72), "roughness": 0.60, "metallic": 1.0},
    # Emissive — feeds AMRGI emissive bounce where supported.
    "emissive": {"color": (1.0, 0.85, 0.55), "roughness": 0.90, "metallic": 0.0, "emissive_intensity": 4.0},
}


def create_materials() -> dict[str, dict]:
    """Returns the shared material parameter table used by every area."""
    return MATERIALS


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def _prim(app, shape: str, position, dimensions, mat: dict, rotation=None) -> int:
    """Creates one static primitive from a material parameter set."""
    kwargs = dict(
        shape=shape,
        position=position,
        dimensions=dimensions,
        color=mat["color"],
        roughness=mat["roughness"],
        metallic=mat["metallic"],
        is_static=True,
    )
    if "emissive_intensity" in mat:
        kwargs["emissive_intensity"] = mat["emissive_intensity"]
    if rotation is not None:
        kwargs["rotation"] = rotation
    return app.ecs.create_entity(**kwargs)


def _box(app, position, dimensions, mat: dict, rotation=None) -> int:
    return _prim(app, "cube", position, dimensions, mat, rotation)


# ---------------------------------------------------------------------------
# Hall shell
# ---------------------------------------------------------------------------

HALL_HALF = 32.0
WALL_H = 6.0
WALL_T = 0.4


def _build_hall(app) -> None:
    """Floor, perimeter walls and partial dividers that separate the six areas."""
    # Floor.
    _box(app, (0.0, -0.15, 0.0), (HALL_HALF * 2.0, 0.3, HALL_HALF * 2.0), MATERIALS["dark"])
    # Perimeter walls.
    _box(app, (0.0, WALL_H * 0.5, -HALL_HALF), (HALL_HALF * 2.0, WALL_H, WALL_T), MATERIALS["neutral"])
    _box(app, (0.0, WALL_H * 0.5, HALL_HALF), (HALL_HALF * 2.0, WALL_H, WALL_T), MATERIALS["neutral"])
    _box(app, (-HALL_HALF, WALL_H * 0.5, 0.0), (WALL_T, WALL_H, HALL_HALF * 2.0), MATERIALS["neutral"])
    _box(app, (HALL_HALF, WALL_H * 0.5, 0.0), (WALL_T, WALL_H, HALL_HALF * 2.0), MATERIALS["neutral"])

    # Column dividers between the three north/south columns (leave gaps to fly through).
    for x in (-9.5, 9.5):
        _box(app, (x, WALL_H * 0.5, -16.0), (WALL_T, WALL_H, 14.0), MATERIALS["neutral"])
        _box(app, (x, WALL_H * 0.5, 14.0), (WALL_T, WALL_H, 14.0), MATERIALS["neutral"])
    # Row divider between the north and south rows (partial, with a central gap).
    for x in (-22.0, 22.0):
        _box(app, (x, WALL_H * 0.5, -1.0), (18.0, WALL_H, WALL_T), MATERIALS["neutral"])


# ---------------------------------------------------------------------------
# Area 1 — GI / color bleed
# ---------------------------------------------------------------------------


def build_gi_area(app) -> None:
    """Semi-enclosed chamber: red/green walls, neutral receivers, shelf, emissive.

    Validates: colored indirect bounce, dark-corner fill, GI under a shelf,
    near/mid/far response and screen-space loss when contributors leave view.
    """
    cx, cz = -19.0, -16.0
    # Neutral floor patch (slightly raised so it reads as its own zone).
    _box(app, (cx, 0.02, cz), (15.0, 0.04, 15.0), MATERIALS["neutral"])

    # Back (north) wall + ceiling half.
    _box(app, (cx, 2.5, cz - 7.5), (15.0, 5.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx, 5.0, cz - 3.75), (15.0, 0.3, 7.5), MATERIALS["neutral"])
    # Red wall (west) and green wall (east) — the color-bleed sources.
    _box(app, (cx - 7.5, 2.5, cz), (0.3, 5.0, 15.0), MATERIALS["red"])
    _box(app, (cx + 7.5, 2.5, cz), (0.3, 5.0, 15.0), MATERIALS["green"])
    # South side: two stubs leaving a central doorway for light and camera.
    _box(app, (cx - 5.0, 2.5, cz + 7.5), (5.0, 5.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx + 5.0, 2.5, cz + 7.5), (5.0, 5.0, 0.3), MATERIALS["neutral"])

    # Central neutral receivers competing for red/green bounce.
    _box(app, (cx - 1.6, 0.6, cz), (1.2, 1.2, 1.2), MATERIALS["neutral_bright"])
    _prim(app, "sphere", (cx, 0.7, cz), (1.4, 1.4, 1.4), MATERIALS["neutral_bright"])
    _prim(app, "cylinder", (cx + 1.6, 0.7, cz), (0.9, 1.4, 0.9), MATERIALS["neutral_bright"])

    # Recessed corner alcove (mostly indirect light).
    _box(app, (cx - 6.0, 1.0, cz - 5.5), (2.0, 2.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx - 6.0, 1.0, cz - 4.0), (0.3, 2.0, 3.0), MATERIALS["neutral"])

    # Raised shelf with an object underneath (GI under-platform test).
    _box(app, (cx + 4.5, 2.2, cz - 2.0), (3.0, 0.2, 3.0), MATERIALS["neutral"])
    _box(app, (cx + 4.5, 0.4, cz - 2.0), (0.8, 0.8, 0.8), MATERIALS["neutral_bright"])

    # Emissive panel on the back wall with a neutral receiver in front.
    _prim(app, "cube", (cx - 3.0, 1.6, cz - 7.2), (1.6, 2.0, 0.2), MATERIALS["emissive"])
    _box(app, (cx - 3.0, 0.9, cz - 5.6), (1.8, 1.8, 0.2), MATERIALS["neutral"])


# ---------------------------------------------------------------------------
# Area 2 — AO / contact
# ---------------------------------------------------------------------------


def build_ao_area(app) -> None:
    """Neutral contact configurations at several gap scales.

    Validates: AO over-darkening, contact detection, gap-scale response, and
    stability of local visibility. Compare AO OFF / SSAO / GTAO here.
    """
    cx, cz = 0.0, -16.0
    _box(app, (cx, 0.02, cz), (15.0, 0.04, 15.0), MATERIALS["neutral"])

    # Cube resting directly on the floor.
    _box(app, (cx - 4.5, 0.5, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    # Cube slightly above the floor (small air gap).
    _box(app, (cx - 2.5, 0.55, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    # Two cubes with a tiny gap between them.
    _box(app, (cx - 0.55, 0.5, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    _box(app, (cx + 0.55, 0.5, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    # Two objects with a medium gap.
    _box(app, (cx + 3.0, 0.5, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    _box(app, (cx + 5.0, 0.5, cz - 3.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])

    # Stacked cubes.
    _box(app, (cx - 4.5, 0.5, cz + 1.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])
    _box(app, (cx - 4.5, 1.5, cz + 1.0), (1.0, 1.0, 1.0), MATERIALS["neutral_bright"])

    # Pillar next to a wall (wall/floor + pillar/wall corners).
    _box(app, (cx + 6.5, 2.5, cz + 1.0), (0.3, 5.0, 6.0), MATERIALS["neutral"])
    _prim(app, "cylinder", (cx + 5.4, 1.5, cz + 1.0), (0.6, 3.0, 0.6), MATERIALS["neutral_bright"])

    # Object underneath a low shelf.
    _box(app, (cx - 1.0, 1.6, cz + 4.5), (3.0, 0.2, 3.0), MATERIALS["neutral"])
    _box(app, (cx - 1.0, 0.4, cz + 4.5), (0.8, 0.8, 0.8), MATERIALS["neutral_bright"])

    # Shallow recess and deeper recess in a back wall.
    _box(app, (cx + 2.0, 2.5, cz - 7.5), (10.0, 5.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx + 2.0, 0.6, cz - 7.0), (1.6, 1.2, 0.6), MATERIALS["dark"])   # shallow
    _box(app, (cx + 5.0, 0.9, cz - 6.6), (1.2, 1.8, 1.2), MATERIALS["dark"])   # deeper

    # Isolated control object (no nearby contact).
    _prim(app, "sphere", (cx + 4.0, 0.7, cz + 5.5), (1.4, 1.4, 1.4), MATERIALS["neutral_bright"])


# ---------------------------------------------------------------------------
# Area 3 — SSR / specular
# ---------------------------------------------------------------------------


def build_ssr_area(app) -> None:
    """Reflection gallery: smooth/rough metals, curved surfaces, colored content.

    Validates: reflection placement, roughness response, curved-surface
    reflections, self-intersection, screen-edge disappearance and grazing angles.
    """
    cx, cz = 19.0, -16.0
    # Large smooth reflective floor section.
    _box(app, (cx, 0.03, cz), (14.0, 0.06, 14.0), MATERIALS["metal_smooth"])

    # Vertical mirror-like panel on the back wall.
    _box(app, (cx, 2.0, cz - 7.2), (6.0, 4.0, 0.2), MATERIALS["metal_smooth"])

    # Smooth / medium / rough metallic objects.
    _prim(app, "sphere", (cx - 3.5, 0.9, cz - 2.0), (1.8, 1.8, 1.8), MATERIALS["metal_smooth"])
    _prim(app, "cylinder", (cx, 0.9, cz - 2.0), (1.2, 1.8, 1.2), MATERIALS["metal_medium"])
    _box(app, (cx + 3.5, 0.9, cz - 2.0), (1.6, 1.6, 1.6), MATERIALS["metal_rough"])

    # Brightly colored objects for the reflective surfaces to reflect.
    _prim(app, "sphere", (cx - 2.0, 0.6, cz + 3.0), (1.2, 1.2, 1.2), MATERIALS["red"])
    _prim(app, "sphere", (cx + 2.0, 0.6, cz + 3.0), (1.2, 1.2, 1.2), MATERIALS["green"])
    _box(app, (cx, 0.6, cz + 4.5), (1.2, 1.2, 1.2), MATERIALS["blue"])

    # Thin pillar / rail visible in reflections.
    _box(app, (cx + 5.5, 2.0, cz + 1.0), (0.12, 4.0, 0.12), MATERIALS["neutral_bright"])
    _box(app, (cx + 5.5, 3.6, cz + 1.0), (0.12, 0.12, 4.0), MATERIALS["neutral_bright"])

    # Emissive object to reflect.
    _prim(app, "cube", (cx - 5.5, 1.2, cz + 1.0), (1.0, 1.6, 1.0), MATERIALS["emissive"])


# ---------------------------------------------------------------------------
# Area 4 — Thin geometry / depth stress
# ---------------------------------------------------------------------------


def build_thin_area(app) -> None:
    """Thin walls, rails, parallel surfaces and depth discontinuities.

    Validates: thickness-related misses, GI/reflection leaking, self-hit and
    backface rejection, refinement failures and depth-edge artifacts.
    """
    cx, cz = -19.0, 14.0
    _box(app, (cx, 0.02, cz), (15.0, 0.04, 15.0), MATERIALS["neutral"])

    # Walls of increasing thickness (very thin -> medium -> thick reference).
    for i, thickness in enumerate((0.02, 0.10, 0.40)):
        _box(app, (cx - 4.5 + i * 2.5, 1.5, cz - 4.0), (thickness, 3.0, 4.0), MATERIALS["neutral_bright"])

    # Thin horizontal platform.
    _box(app, (cx + 4.0, 1.4, cz - 4.0), (4.0, 0.05, 3.0), MATERIALS["neutral_bright"])

    # Railing-like repeated narrow vertical bars.
    for i in range(6):
        _box(app, (cx - 5.0 + i * 0.6, 1.2, cz + 1.0), (0.06, 2.4, 0.06), MATERIALS["neutral_bright"])
    _box(app, (cx - 3.5, 2.3, cz + 1.0), (3.6, 0.06, 0.06), MATERIALS["neutral_bright"])

    # Closely spaced parallel surfaces.
    for i in range(3):
        _box(app, (cx + 3.0, 1.2, cz + 1.0 + i * 0.25), (3.0, 2.4, 0.04), MATERIALS["neutral_bright"])

    # Diagonal thin surface (45 degrees about Y).
    q45 = (0.0, math.sin(math.pi / 8.0), 0.0, math.cos(math.pi / 8.0))
    _box(app, (cx + 1.0, 1.2, cz + 5.0), (4.0, 2.4, 0.05), MATERIALS["neutral_bright"], rotation=q45)

    # Small foreground object with a distant wall behind it (depth discontinuity).
    _box(app, (cx - 1.0, 0.4, cz + 5.5), (0.6, 0.8, 0.6), MATERIALS["red"])
    _box(app, (cx - 1.0, 2.5, cz + 7.4), (10.0, 5.0, 0.3), MATERIALS["neutral"])


# ---------------------------------------------------------------------------
# Area 5 — Temporal / motion inspection
# ---------------------------------------------------------------------------


def build_temporal_area(app) -> None:
    """Static geometry that reveals temporal behaviour as the camera moves.

    Validates: GI pulsing/lag, ghosting, disocclusion through a doorway,
    reflection instability and AO crawling. All motion is manual camera motion.
    """
    cx, cz = 0.0, 14.0
    _box(app, (cx, 0.02, cz), (15.0, 0.04, 15.0), MATERIALS["neutral"])

    # Foreground pillars framing the view.
    for x in (-5.0, 5.0):
        _prim(app, "cylinder", (cx + x, 2.0, cz + 5.0), (0.8, 4.0, 0.8), MATERIALS["neutral_bright"])

    # Doorway in a wall: crossing it creates strong disocclusion.
    _box(app, (cx - 4.0, 2.5, cz - 6.0), (7.0, 5.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx + 4.0, 2.5, cz - 6.0), (7.0, 5.0, 0.3), MATERIALS["neutral"])
    _box(app, (cx, 4.2, cz - 6.0), (1.0, 1.6, 0.3), MATERIALS["neutral"])

    # High-contrast bright/dark surfaces behind the doorway.
    _box(app, (cx - 2.0, 1.5, cz - 7.4), (3.0, 3.0, 0.2), MATERIALS["neutral_bright"])
    _box(app, (cx + 2.0, 1.5, cz - 7.4), (3.0, 3.0, 0.2), MATERIALS["dark"])

    # Colored GI surface + emissive object + reflective surface.
    _box(app, (cx - 5.5, 1.2, cz - 2.0), (0.3, 2.4, 3.0), MATERIALS["red"])
    _prim(app, "cube", (cx + 5.5, 1.2, cz - 2.0), (1.0, 1.6, 1.0), MATERIALS["emissive"])
    _box(app, (cx + 2.5, 0.03, cz + 1.0), (3.0, 0.06, 3.0), MATERIALS["metal_smooth"])

    # Thin object + repeating small geometry (makes ghosting visible).
    _box(app, (cx - 2.5, 1.2, cz + 1.0), (0.05, 2.4, 2.0), MATERIALS["neutral_bright"])
    for i in range(5):
        _box(app, (cx - 1.0 + i * 0.5, 0.3, cz + 3.5), (0.3, 0.6, 0.3), MATERIALS["neutral_bright"])


# ---------------------------------------------------------------------------
# Area 6 — Performance stress
# ---------------------------------------------------------------------------


def build_performance_area(app) -> None:
    """A denser field of repeated primitives than any other area.

    Validates: G-buffer cost, instance/MDI behaviour and SSRT cost at complex
    depth. Kept moderate so the default scene stays interactive.
    """
    cx, cz = 19.0, 14.0
    _box(app, (cx, 0.02, cz), (15.0, 0.04, 15.0), MATERIALS["neutral"])

    # Repeated object field (8x8) with cycling shapes and materials.
    shapes = ("cube", "sphere", "cylinder", "cone")
    mats = (MATERIALS["neutral_bright"], MATERIALS["metal_medium"], MATERIALS["red"], MATERIALS["green"])
    spacing = 1.5
    half = (8 - 1) * spacing * 0.5
    for row in range(8):
        for col in range(8):
            idx = row * 8 + col
            shape = shapes[idx % len(shapes)]
            mat = mats[idx % len(mats)]
            pos = (cx + col * spacing - half, 0.6, cz + row * spacing - half)
            dims = (0.9, 1.2, 0.9) if shape in ("cylinder", "cone") else (0.9, 0.9, 0.9)
            _prim(app, shape, pos, dims, mat)

    # A row of repeated pillars along the back.
    for i in range(6):
        _prim(app, "cylinder", (cx - 6.0 + i * 2.4, 2.0, cz - 7.0), (0.5, 4.0, 0.5), MATERIALS["neutral"])


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------


def create_lights(app) -> None:
    """Controlled lighting: sun (from config) plus a few local lights.

    The directional sun comes from the atmosphere/config; here we add a small
    number of local lights so each area has useful direct and indirect content.
    """
    app.pipeline.lights_pass.clear()

    # GI area: bright light entering the doorway + a colored bounce light.
    app.pipeline.add_point_light((-19.0, 3.2, -8.5), radius=9.0, color=(1.0, 0.95, 0.85), intensity=7.0)
    app.pipeline.add_point_light((-22.0, 1.6, -16.0), radius=6.0, color=(1.0, 0.35, 0.25), intensity=4.0)

    # AO area: soft neutral fill so contact shadows read clearly.
    app.pipeline.add_point_light((0.0, 4.0, -16.0), radius=12.0, color=(0.9, 0.92, 1.0), intensity=4.0)

    # SSR area: a spot to give the reflective surfaces visible content.
    app.pipeline.add_spot_light(
        position=(19.0, 5.0, -8.0), direction=(0.0, -1.0, -0.3), radius=14.0,
        inner_cone_angle=math.radians(20.0), outer_cone_angle=math.radians(34.0),
        color=(1.0, 0.95, 0.9), intensity=9.0, cast_shadow=True,
    )

    # Temporal area: a colored local light for GI/reflection content.
    app.pipeline.add_point_light((0.0, 3.0, 10.0), radius=8.0, color=(0.4, 0.7, 1.0), intensity=5.0)


# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

# (name, position, target, fov)
CAMERA_PRESETS: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], float], ...] = (
    ("OVERVIEW", (0.0, 26.0, 34.0), (0.0, 0.0, 0.0), 60.0),
    ("GI", (-19.0, 3.0, -6.0), (-19.0, 1.5, -20.0), 62.0),
    ("GI_CLOSE", (-19.0, 1.6, -12.0), (-22.0, 1.2, -20.0), 65.0),
    ("AO", (0.0, 2.6, -6.0), (0.0, 0.6, -18.0), 60.0),
    ("SSR", (19.0, 2.2, -6.0), (19.0, 0.8, -20.0), 62.0),
    ("SSR_EDGE", (24.0, 1.6, -10.0), (14.0, 1.0, -20.0), 65.0),
    ("THIN", (-19.0, 2.4, 22.0), (-19.0, 1.2, 8.0), 60.0),
    ("TEMPORAL", (0.0, 2.2, 24.0), (0.0, 1.4, 6.0), 62.0),
    ("PERFORMANCE", (19.0, 6.0, 26.0), (19.0, 0.8, 10.0), 60.0),
)


def create_camera_presets() -> tuple:
    """Returns the deterministic camera preset table."""
    return CAMERA_PRESETS


# ---------------------------------------------------------------------------
# Overlay (camera presets + help)
# ---------------------------------------------------------------------------


class SSRTValidationOverlay(ProjectModule):
    """Camera-preset switching and a small help/status HUD.

    Rendering options are intentionally left to the engine debug system.
    """

    name = "SSRTValidationOverlay"

    def __init__(self, camera: FreeFlyCamera) -> None:
        self.camera = camera
        self.preset_index = 0
        self.show_help = True
        self.status_label: UILabel | None = None
        self.help_label: UILabel | None = None

    def on_attach(self, app) -> None:
        app.input_manager.bind_action("ssrt_lab_help", keys=[pygame.K_h])

        screen = UIScreen(name="SSRTLabHUD", is_modal=False)
        panel = screen.add_child(UIPanel(x=16.0, y=16.0, w=600.0, h=150.0))
        panel.add_child(UILabel("PYMORDIAL | RENDERER V2 SSRT VALIDATION LAB", x=14.0, y=10.0, w=570.0, h=26.0, font_size=17))
        self.status_label = panel.add_child(UILabel("", x=14.0, y=40.0, w=570.0, h=24.0, font_size=14))
        self.help_label = panel.add_child(UILabel(
            "1-9 camera presets | H help | Home reset | WASD/QE fly\n"
            "GI / AO / SSR / debug views: engine debug system (F1/F2/F3)",
            x=14.0, y=70.0, w=570.0, h=60.0, font_size=13,
        ))
        app.ui.set_screen(screen)

        self._apply_preset(app, 0)

    def on_update(self, app, dt: float) -> None:
        if self.status_label is not None:
            name = CAMERA_PRESETS[self.preset_index][0]
            cfg = app.pipeline.config
            self.status_label.text = (
                f"Preset: {name} | GI: {cfg.gi_mode} | AO: {cfg.ao_mode} | "
                f"SSR: {'ON' if cfg.ssr_enabled else 'OFF'}"
            )
        if self.help_label is not None:
            self.help_label.visible = self.show_help

    def on_event(self, app, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False
        if pygame.K_1 <= event.key <= pygame.K_9:
            self._apply_preset(app, event.key - pygame.K_1)
            return True
        if event.key == pygame.K_h:
            self.show_help = not self.show_help
            return True
        return False

    def _apply_preset(self, app, index: int) -> None:
        if not (0 <= index < len(CAMERA_PRESETS)):
            return
        self.preset_index = index
        _, pos, target, fov = CAMERA_PRESETS[index]
        cam = self.camera
        cam.position[:] = pos
        cam.target[:] = target
        cam.fov = fov
        cam.mark_dirty()
        # Camera cut: drop stale GI temporal history so the new view converges.
        for pass_name in ("ssgi_pass", "rmgi_pass", "amrgi_pass"):
            gi_pass = getattr(app.pipeline, pass_name, None)
            if gi_pass is not None and hasattr(gi_pass, "invalidate_history"):
                gi_pass.invalidate_history()


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def build_scene(app) -> dict:
    """Builds the full validation hall and returns metadata for tooling."""
    _build_hall(app)
    build_gi_area(app)
    build_ao_area(app)
    build_ssr_area(app)
    build_thin_area(app)
    build_temporal_area(app)
    build_performance_area(app)
    create_lights(app)

    camera = FreeFlyCamera(position=CAMERA_PRESETS[0][1], yaw_deg=0.0, pitch_deg=0.0, move_speed=8.0)
    app.camera_manager.register_camera("ssrt_lab_flycam", camera, make_active=True)
    app.add_module(FlyCameraControls(camera))

    overlay = SSRTValidationOverlay(camera)
    app.add_module(overlay)

    return {
        "camera": camera,
        "overlay": overlay,
        "zones": ("GI", "AO", "SSR", "THIN", "TEMPORAL", "PERFORMANCE"),
        "presets": tuple(p[0] for p in CAMERA_PRESETS),
    }


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Renderer V2 SSRT Validation Lab").parse_args(argv)
    app = create_app(args, "PyMordial | Renderer V2 SSRT Validation Lab")
    build_scene(app)
    app.run()


if __name__ == "__main__":
    main()
