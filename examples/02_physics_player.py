"""Test fixed and dynamic rigid bodies and walk among them as a capsule."""

from __future__ import annotations

from _common import CapsulePlayerControls, add_box, create_app, create_parser


def build_scene(app) -> None:
    add_box(
        app,
        (0.0, -0.25, 0.0),
        (34.0, 0.5, 24.0),
        (0.16, 0.19, 0.23),
        0.9,
        physics=True,
    )

    obstacles = (
        ((-8.0, 1.0, -3.0), (2.0, 2.0, 2.0), (0.8, 0.25, 0.16)),
        ((-5.0, 0.65, -3.0), (2.0, 1.3, 2.0), (0.84, 0.58, 0.18)),
        ((-2.0, 1.25, -3.0), (2.0, 2.5, 2.0), (0.18, 0.62, 0.34)),
        ((4.0, 2.0, -3.0), (2.0, 4.0, 2.0), (0.18, 0.5, 0.86)),
        ((8.0, 0.7, -3.0), (3.0, 1.4, 3.0), (0.72, 0.28, 0.74)),
    )
    for position, dimensions, color in obstacles:
        add_box(app, position, dimensions, color, 0.38, physics=True)

    dynamic_items = (
        ("cube", (-4.0, 3.0, 2.0), (1.2, 1.2, 1.2), (0.92, 0.32, 0.15)),
        ("sphere", (-1.5, 4.0, 2.0), (1.4, 1.4, 1.4), (0.16, 0.65, 0.95)),
        ("cylinder", (1.5, 3.0, 2.0), (1.2, 1.8, 1.2), (0.92, 0.72, 0.18)),
        ("capsule", (4.5, 3.5, 2.0), (0.9, 1.8, 0.9), (0.28, 0.8, 0.45)),
    )
    for shape, position, dimensions, color in dynamic_items:
        entity = app.ecs.create_entity(
            shape=shape,
            position=position,
            dimensions=dimensions,
            color=color,
            roughness=0.28,
        )
        app.physics.create_body(entity, body_type="dynamic", position=position)
        if shape == "sphere":
            app.physics.attach_sphere_collider(entity, dimensions[0] * 0.5, density=1.0, restitution=0.25)
        elif shape == "capsule":
            app.physics.attach_capsule_collider(entity, 0.45, 0.45, density=1.0)
        else:
            app.physics.attach_box_collider(
                entity,
                dimensions[0] * 0.5,
                dimensions[1] * 0.5,
                dimensions[2] * 0.5,
                density=1.0,
                restitution=0.15,
            )

    app.add_module(CapsulePlayerControls(spawn=(0.0, 1.0, 8.0), camera_yaw=0.0))


def main(argv: list[str] | None = None) -> None:
    args = create_parser(__doc__ or "Physics and player").parse_args(argv)
    app = create_app(args, "PyMordial | Physics Player")
    build_scene(app)
    app.run()


if __name__ == "__main__":
    main()
