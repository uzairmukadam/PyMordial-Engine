"""Explore five compact connected water chambers with skylight and colored GI."""

from __future__ import annotations

import math

from _common import FlyCameraControls, add_box, create_app, create_parser
from engine.camera import FreeFlyCamera
from engine.core import WaterProperties


ROOM_WIDTH = 8.5
ROOM_LENGTH = 8.0
ROOM_HEIGHT = 3.8
WALL = 0.35
DOOR_WIDTH = 2.5


def _end_wall(app, center_z: float, color: tuple[float, float, float], *, is_lintel: bool = False) -> None:
    half_side = (ROOM_WIDTH - DOOR_WIDTH) * 0.5
    for sign in (-1.0, 1.0):
        x = sign * (DOOR_WIDTH * 0.5 + half_side * 0.5)
        add_box(
            app,
            (x, ROOM_HEIGHT * 0.5, center_z),
            (half_side, ROOM_HEIGHT, WALL),
            color,
            0.78,
        )
    add_box(
        app,
        (0.0, ROOM_HEIGHT - 0.55, center_z),
        (DOOR_WIDTH, 1.1, WALL),
        color,
        0.78,
    )
    trim = (0.78, 0.57, 0.25) if is_lintel else (0.34, 0.39, 0.4)
    for x in (-DOOR_WIDTH * 0.5 - 0.09, DOOR_WIDTH * 0.5 + 0.09):
        add_box(app, (x, 1.45, center_z), (0.18, 2.9, WALL + 0.04), trim, 0.42, 0.35)


def _room(
    app,
    index: int,
    center_z: float,
    name_color: tuple[float, float, float],
    *,
    skylight: bool = False,
    water: bool = False,
) -> None:
    half_w = ROOM_WIDTH * 0.5
    half_l = ROOM_LENGTH * 0.5
    front_z = center_z + half_l
    back_z = center_z - half_l

    add_box(app, (0.0, -0.22, center_z), (ROOM_WIDTH, 0.44, ROOM_LENGTH), (0.23, 0.24, 0.23), 0.82)
    add_box(app, (-half_w, ROOM_HEIGHT * 0.5, center_z), (WALL, ROOM_HEIGHT, ROOM_LENGTH), (0.33, 0.31, 0.27), 0.76)
    add_box(app, (half_w, ROOM_HEIGHT * 0.5, center_z), (WALL, ROOM_HEIGHT, ROOM_LENGTH), (0.29, 0.3, 0.29), 0.74)
    _end_wall(app, front_z, (0.32, 0.31, 0.27), is_lintel=index == 0)
    _end_wall(app, back_z, (0.32, 0.31, 0.27), is_lintel=index == 4)

    if skylight:
        opening = 2.8
        panel_w = (ROOM_WIDTH - opening) * 0.5
        for sign in (-1.0, 1.0):
            x = sign * (opening * 0.5 + panel_w * 0.5)
            add_box(app, (x, ROOM_HEIGHT + WALL * 0.5, center_z), (panel_w, WALL, ROOM_LENGTH), (0.17, 0.2, 0.22), 0.8)
        end_depth = (ROOM_LENGTH - opening) * 0.5
        for sign in (-1.0, 1.0):
            z = center_z + sign * (opening * 0.5 + end_depth * 0.5)
            add_box(app, (0.0, ROOM_HEIGHT + WALL * 0.5, z), (opening, WALL, end_depth), (0.17, 0.2, 0.22), 0.8)
    else:
        add_box(app, (0.0, ROOM_HEIGHT + WALL * 0.5, center_z), (ROOM_WIDTH, WALL, ROOM_LENGTH), (0.17, 0.2, 0.22), 0.82)

    # Colored inset wall panels make the indirect-light response easy to see.
    panel_z = center_z - 1.7
    add_box(app, (half_w - WALL - 0.04, 1.65, panel_z), (0.1, 2.1, 2.7), name_color, 0.74)
    for x in (-3.05, 3.05):
        add_box(app, (x, 1.25, center_z + 0.4), (0.24, 2.5, 0.24), (0.54, 0.48, 0.36), 0.6)

    if water:
        app.ecs.create_entity(
            shape="water",
            position=(0.45 if index % 2 else -0.45, 0.025, center_z + 0.15),
            dimensions=(5.2, 0.0, 4.6),
            water=WaterProperties(
                color_shallow=(0.08, 0.42, 0.48) if index % 2 else (0.12, 0.34, 0.48),
                wave_amplitude=0.10 + index % 3 * 0.05,
                foam_enabled=index % 2 == 0,
            ),
        )
        for x in (-3.1, 3.1):
            add_box(app, (x, 0.42, center_z + 0.15), (1.3, 0.45, 4.8), (0.47, 0.42, 0.32), 0.68)


def build_scene(app) -> None:
    config = app.pipeline.config
    config.gi_mode = "RADIANCE_CASCADES"
    config.ao_mode = "GTAO"
    config.ssr_enabled = True
    config.ssr_resolution_scale = 1.0
    config.bloom_enabled = True
    config.volumetric_fog_enabled = False
    config.sun_intensity = 5.0
    config.ambient_factor = 0.025
    app.pipeline.apply_config(config)
    app.pipeline.lights_pass.clear()

    # Each doorway lines up with the next, creating a readable five-room tour.
    room_specs = (
        (0.0, (0.08, 0.47, 0.43), False, True),
        (-ROOM_LENGTH, (0.72, 0.18, 0.08), True, False),
        (-ROOM_LENGTH * 2.0, (0.76, 0.56, 0.24), False, False),
        (-ROOM_LENGTH * 3.0, (0.08, 0.25, 0.78), False, True),
        (-ROOM_LENGTH * 4.0, (0.56, 0.22, 0.78), False, True),
    )

    # room_specs = (
    #         (0.0, (0.08, 0.47, 0.43), False, True),
    #         (-ROOM_LENGTH, (0.76, 0.56, 0.24), True, False),
    #         (-ROOM_LENGTH * 2.0, (0.72, 0.18, 0.08), False, False),
    #         (-ROOM_LENGTH * 3.0, (0.08, 0.25, 0.78), False, True),
    #         (-ROOM_LENGTH * 4.0, (0.56, 0.22, 0.78), False, True),
    # )

    for index, (center_z, color, skylight, water) in enumerate(room_specs):
        _room(app, index, center_z, color, skylight=skylight, water=water)

    # Sun shaft and contrasting practical lights distinguish the connected zones.
    app.pipeline.add_spot_light(
        position=(0.0, 10.0, -ROOM_LENGTH),
        direction=(0.0, -1.0, 0.0),
        radius=13.0,
        inner_cone_angle=math.radians(13.0),
        outer_cone_angle=math.radians(25.0),
        color=(1.0, 0.8, 0.48),
        intensity=24.0,
        cast_shadow=True,
    )
    for position, color, intensity in (
        ((-2.8, 2.5, 0.0), (0.1, 1.0, 0.68), 3.2),
        ((2.8, 2.4, -ROOM_LENGTH * 2.0), (1.0, 0.12, 0.035), 5.0),
        ((-2.8, 2.3, -ROOM_LENGTH * 3.0), (0.08, 0.24, 1.0), 4.5),
        ((2.8, 2.3, -ROOM_LENGTH * 4.0), (0.56, 0.12, 1.0), 4.0),
    ):
        app.pipeline.add_point_light(position, radius=6.5, color=color, intensity=intensity)

    camera = FreeFlyCamera(position=(0.0, 1.7, 2.8), yaw_deg=-90.0, pitch_deg=-2.0, move_speed=4.5)
    app.camera_manager.register_camera("lighting_flycam", camera, make_active=True)
    app.add_module(FlyCameraControls(camera))


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Water and lighting lab").parse_args(argv)
    app = create_app(args, "PyMordial | Water & Lighting Lab")
    build_scene(app)
    app.run()


if __name__ == "__main__":
    main()
