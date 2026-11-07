"""Walk a detailed multi-section platformer assembled from JSON data."""

from __future__ import annotations

import json
from pathlib import Path

from _common import CapsulePlayerControls, add_box, create_app, create_parser


LEVEL_PATH = Path(__file__).with_name("platformer_level.json")


def build_level(app, level_path: Path = LEVEL_PATH) -> dict:
    with level_path.open(encoding="utf-8") as level_file:
        level = json.load(level_file)

    app.pipeline.config.gi_mode = "RADIANCE_CASCADES"
    app.pipeline.config.bloom_enabled = True
    app.pipeline.config.volumetric_fog_enabled = False
    app.pipeline.apply_config(app.pipeline.config)

    for platform in level["platforms"]:
        add_box(
            app,
            tuple(platform["position"]),
            tuple(platform["size"]),
            tuple(platform["color"]),
            0.78,
            physics=True,
        )

    for decoration in level["decorations"]:
        app.ecs.create_entity(
            shape=decoration["shape"],
            position=decoration["position"],
            dimensions=decoration["size"],
            color=decoration["color"],
            roughness=0.34,
            metallic=0.12,
        )

    app.pipeline.lights_pass.clear()
    for light in level["lights"]:
        app.pipeline.add_point_light(
            tuple(light["position"]),
            radius=light["radius"],
            color=tuple(light["color"]),
            intensity=light["intensity"],
        )

    app.add_module(
        CapsulePlayerControls(
            spawn=tuple(level["spawn"]),
            camera_yaw=0.0,
            fall_reset_y=float(level.get("fall_reset_y", -10.0)),
        )
    )
    return level


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Data-driven platformer").parse_args(argv)
    app = create_app(args, "PyMordial | The Sunken Observatory")
    build_level(app)
    app.run()


if __name__ == "__main__":
    main()
