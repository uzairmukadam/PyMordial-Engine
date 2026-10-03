"""PyMordial Engine Example 01: Minimal 3D Application.

Demonstrates the core engine bootstrap, deferred PBR rendering, camera controls,
and a simple rotating entity using the zero-allocation ProjectModule lifecycle.

Usage:
    python examples/01_minimal_app.py
    python examples/01_minimal_app.py --headless --frames 120
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import ProjectApp, ProjectConfig, ProjectModule, FreeCamera


class RotatingEntityModule(ProjectModule):
    """Simple gameplay module rotating a 3D entity every frame."""

    name = "RotatingEntity"

    def __init__(self) -> None:
        self.entity_id = -1
        self.angle = 0.0

    def on_attach(self, app: ProjectApp) -> None:
        # Create a single 3D box entity in the ECS
        self.entity_id = app.ecs.create_entity(
            position=(0.0, 1.0, 0.0),
            color=(0.2, 0.8, 1.0, 1.0),
            roughness=0.3,
            metallic=0.7,
        )

        # Attach FreeCamera for user navigation
        cam = FreeCamera(
            position=(0.0, 3.0, 6.0),
            yaw_deg=-90.0,
            pitch_deg=-20.0,
            fov=60.0,
        )
        app.camera_manager.register_camera("main", cam)
        app.camera_manager.set_active_camera("main")

    def on_update(self, app: ProjectApp, dt: float) -> None:
        self.angle += dt * 1.5
        cy = math.cos(self.angle * 0.5)
        sy = math.sin(self.angle * 0.5)
        # Update rotation quaternion in continuous ECS memory
        app.ecs.set_rotation(self.entity_id, (0.0, sy, 0.0, cy))


def main() -> None:
    parser = argparse.ArgumentParser(description="PyMordial Minimal App Example")
    parser.add_argument("--headless", action="store_true", help="Run without opening a window")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to run before exiting")
    args = parser.parse_args()

    config = ProjectConfig(
        title="PyMordial - Minimal 3D Example",
        width=1280,
        height=720,
        headless=args.headless,
        max_frames=args.frames,
        enable_debug=True,
    )

    app = ProjectApp(config=config)
    app.add_module(RotatingEntityModule())

    try:
        if args.headless and args.frames:
            for _ in range(args.frames):
                app.step_frame(1.0 / 60.0)
        else:
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
