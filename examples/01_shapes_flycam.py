"""Explore all built-in ECS shapes and live material settings with a flycam."""

from __future__ import annotations

from _common import FlyCameraControls, add_box, create_app, create_parser
from engine.camera import FreeFlyCamera
from engine.core import WaterProperties


def build_scene(app) -> None:
    add_box(app, (0.0, -0.25, 0.0), (24.0, 0.5, 16.0), (0.11, 0.14, 0.19), 0.9)
    samples = (
        ("cube", (-8.0, 1.0, 0.0), (1.7, 2.0, 1.7), (0.92, 0.25, 0.16), 0.38, 0.0),
        ("sphere", (-5.0, 1.0, 0.0), (2.0, 2.0, 2.0), (0.12, 0.58, 0.94), 0.12, 0.25),
        ("plane", (-2.0, 0.1, 0.0), (2.4, 0.0, 2.4), (0.17, 0.73, 0.55), 0.7, 0.0),
        ("capsule", (1.0, 1.2, 0.0), (1.2, 2.4, 1.2), (0.76, 0.34, 0.9), 0.28, 0.1),
        ("cylinder", (4.0, 1.1, 0.0), (1.8, 2.2, 1.8), (0.93, 0.68, 0.16), 0.42, 0.3),
        ("cone", (7.5, 1.0, 0.0), (2.0, 2.0, 2.0), (0.93, 0.43, 0.13), 0.52, 0.0),
    )
    for shape, position, dimensions, color, roughness, metallic in samples:
        app.ecs.create_entity(
            shape=shape,
            position=position,
            dimensions=dimensions,
            color=color,
            roughness=roughness,
            metallic=metallic,
        )

    app.ecs.create_entity(
        shape="water",
        position=(0.0, 0.02, 5.0),
        dimensions=(6.0, 0.0, 3.0),
        water=WaterProperties(wave_amplitude=0.08, foam_enabled=False),
    )

    for index in range(5):
        x = -6.0 + index * 3.0
        app.ecs.create_entity(
            shape="cube",
            position=(x, 0.6, -4.5),
            dimensions=(1.1, 1.1, 1.1),
            color=(0.2 + index * 0.13, 0.45, 0.85 - index * 0.1),
            roughness=0.15 + index * 0.17,
            metallic=index * 0.2,
        )

    camera = FreeFlyCamera(position=(12.0, 7.0, 14.0), yaw_deg=-135.0, pitch_deg=-18.0, move_speed=8.0)
    app.camera_manager.register_camera("flycam", camera, make_active=True)
    app.add_module(FlyCameraControls(camera))


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Built-in shapes and flycam").parse_args(argv)
    app = create_app(args, "PyMordial | Shapes Flycam")
    build_scene(app)
    app.run()


if __name__ == "__main__":
    main()
