"""Explore a detailed cliffside ruin platformer built from level data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from _common import CapsulePlayerControls, add_box, create_app, create_parser
from engine.core import WaterProperties
from engine.core.math_utils import quat_from_euler


LEVEL_PATH = Path(__file__).with_name("last_aqueduct_level.json")


def build_level(app, level_path: Path = LEVEL_PATH) -> dict[str, Any]:
    with level_path.open(encoding="utf-8") as level_file:
        level = json.load(level_file)

    config = app.pipeline.config
    config.gi_mode = "RADIANCE_CASCADES"
    config.ao_mode = "GTAO"
    config.ssr_enabled = True
    config.volumetric_fog_enabled = True
    config.fog_density = 0.0035
    config.fog_distance = 180.0
    config.bloom_enabled = True
    app.pipeline.apply_config(config)
    app.pipeline.atmosphere.config.time_of_day = 17.0
    app.pipeline.atmosphere.config.day_speed = 0.0

    for surface_group in ("platforms", "bridges", "ramps"):
        for surface in level.get(surface_group, []):
            rotation = quat_from_euler(
                *(math.radians(float(angle)) for angle in surface.get("rotation_deg", (0.0, 0.0, 0.0)))
            )
            add_box(
                app,
                tuple(surface["position"]),
                tuple(surface["size"]),
                tuple(surface["color"]),
                float(surface.get("roughness", 0.88)),
                float(surface.get("metallic", 0.0)),
                physics=True,
                rotation=tuple(float(component) for component in rotation),
            )

    for prop in level["props"]:
        rotation = quat_from_euler(
            *(math.radians(float(angle)) for angle in prop.get("rotation_deg", (0.0, 0.0, 0.0)))
        )
        entity = app.ecs.create_entity(
            shape=prop["shape"],
            position=tuple(prop["position"]),
            rotation=rotation,
            dimensions=tuple(prop["size"]),
            color=tuple(prop["color"]),
            roughness=float(prop.get("roughness", 0.72)),
            metallic=float(prop.get("metallic", 0.0)),
            emissive_intensity=float(prop.get("emissive", 0.0)),
            is_static=True,
        )
        if prop.get("solid", prop["shape"] in {"cube", "cylinder"}):
            app.physics.create_body(
                entity,
                body_type="fixed",
                position=tuple(prop["position"]),
                rotation=tuple(float(component) for component in rotation),
            )
            half_x, half_y, half_z = (float(component) * 0.5 for component in prop["size"])
            app.physics.attach_box_collider(entity, half_x, half_y, half_z, friction=0.75)

    for pool in level.get("water", []):
        app.ecs.create_entity(
            shape="water",
            position=tuple(pool["position"]),
            dimensions=tuple(pool["size"]),
            is_static=True,
            water=WaterProperties(
                wave_amplitude=float(pool.get("wave_amplitude", 0.08)),
                wave_speed=float(pool.get("wave_speed", 0.65)),
                color_shallow=tuple(pool.get("color_shallow", (0.08, 0.32, 0.31))),
                color_deep=tuple(pool.get("color_deep", (0.015, 0.10, 0.15))),
                clarity=float(pool.get("clarity", 5.0)),
                foam_enabled=bool(pool.get("foam_enabled", False)),
            ),
        )

    app.pipeline.lights_pass.clear()
    for light in level["lights"]:
        app.pipeline.add_point_light(
            tuple(light["position"]),
            radius=float(light["radius"]),
            color=tuple(light["color"]),
            intensity=float(light["intensity"]),
        )

    app.add_module(
        CapsulePlayerControls(
            spawn=tuple(level["spawn"]),
            camera_yaw=float(level.get("camera_yaw", 0.0)),
            fall_reset_y=float(level.get("fall_reset_y", -18.0)),
        )
    )
    return level


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Cliffside aqueduct platformer").parse_args(argv)
    app = create_app(args, "PyMordial | The Last Aqueduct")
    build_level(app)
    app.run()


if __name__ == "__main__":
    main()
