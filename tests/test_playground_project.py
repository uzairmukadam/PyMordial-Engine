"""End-to-end tests for the Playground reference project.

Playground is the canonical project-workflow reference: declarative config, a
project-owned world builder, manifest-declared modules loaded through the
registry, and a state-machine driven pause flow. These tests prove that path
works without editing engine source.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "projects" / "playground"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not PROJECT_DIR.is_dir(), reason="playground project not present")
class TestPlaygroundProject:
    def test_project_validates(self) -> None:
        result = _run("-m", "engine.tools.project", "validate", "--project", "playground")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VALID" in result.stdout

    def test_project_runs_headless(self) -> None:
        result = _run(
            "-m", "engine.tools.project", "run",
            "--project", "playground", "--headless", "--frames", "3",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_project_runs_with_atmosphere_disabled(self) -> None:
        result = _run(str(PROJECT_DIR / "main.py"), "--headless", "--frames", "3", "--no-atmosphere")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_config_declares_engine_module_settings(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from projects.playground.config import get_project_config

            cfg = get_project_config()
            assert cfg.atmosphere_enabled is True
            assert cfg.atmosphere_settings["preset"] == "EARTH_DAY"
            assert cfg.custom_settings["spawn_point"] == (0.0, 1.2, 10.0)
        finally:
            sys.path.remove(str(REPO_ROOT))

    def test_modules_declare_valid_manifests(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from engine.app.module_manifest import ModuleRole, validate_manifest
            from projects.playground.modules.hud import HudModule
            from projects.playground.modules.menu import MenuModule
            from projects.playground.modules.player import PlayerModule
            from projects.playground.modules.tweaks import TweaksModule

            expected = {
                PlayerModule: ("playground.player", ModuleRole.GAMEPLAY),
                HudModule: ("playground.hud", ModuleRole.GAMEPLAY),
                MenuModule: ("playground.menu", ModuleRole.GAMEPLAY),
                TweaksModule: ("playground.tweaks", ModuleRole.TOOL),
            }
            for module_type, (module_id, role) in expected.items():
                manifest = module_type.MODULE_MANIFEST
                assert validate_manifest(manifest).is_valid is True
                assert manifest.id == module_id
                assert manifest.role is role
        finally:
            sys.path.remove(str(REPO_ROOT))

    def test_registry_resolves_dependencies_before_dependents(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from projects.playground.config import get_project_config
            from projects.playground.main import build_registry

            registry = build_registry(get_project_config())
            assert registry.validate().is_valid is True

            order = [record.module_id for record in registry.resolve()]
            assert order.index("playground.player") < order.index("playground.hud")
            assert order.index("playground.player") < order.index("playground.menu")
            assert order.index("playground.player") < order.index("playground.tweaks")
        finally:
            sys.path.remove(str(REPO_ROOT))


@pytest.mark.skipif(not PROJECT_DIR.is_dir(), reason="playground project not present")
class TestPlaygroundRuntime:
    """In-process checks of the world builder and Playing/Paused state flow."""

    @pytest.fixture()
    def app(self):
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from engine.app.config import ProjectConfig
            from engine.app.project_app import ProjectApp
            from projects.playground.main import build_registry
            from projects.playground.world.arena import PlaygroundArenaBuilder

            cfg = ProjectConfig(
                title="PlaygroundTest",
                width=320,
                height=240,
                headless=True,
                enable_debug=False,
            )
            application = ProjectApp(cfg)
            application.set_world_builder(PlaygroundArenaBuilder())
            application.load_modules(build_registry(cfg))
            yield application
            application.shutdown()
        finally:
            sys.path.remove(str(REPO_ROOT))

    def test_world_builder_creates_geometry_and_colliders(self, app) -> None:
        assert app.ecs.active_count >= 1
        assert app.physics.world is not None

    def test_player_settles_on_ground(self, app) -> None:
        from projects.playground.modules.player import PlayerModule

        player = app.get_module(PlayerModule)
        assert player is not None
        for _ in range(20):
            app.step_frame(1.0 / 60.0)
        assert player.is_grounded is True
        assert player.motor is not None
        assert player.motor.position[1] > 0.5

    def test_pause_halts_simulation_and_resume_restores_hud(self, app) -> None:
        from projects.playground.modules.menu import MenuModule
        from projects.playground.modules.player import PlayerModule

        player = app.get_module(PlayerModule)
        menu = app.get_module(MenuModule)
        assert player is not None and menu is not None

        assert app.state_manager.current_state_name == "Playing"
        assert app.ui.active_screen is not None
        assert app.ui.active_screen.name == "PlaygroundHUD"

        menu.pause(app)
        assert app.state_manager.current_state_name == "Paused"
        assert app.ui.active_screen is not None
        assert app.ui.active_screen.name == "PauseMenu"

        dense_idx = app.ecs.pool.get_dense_index(player.player_id)
        before = tuple(float(v) for v in app.ecs.rigid_body_state[1, dense_idx, 0:3])
        for _ in range(10):
            app.step_frame(1.0 / 60.0)
        after = tuple(float(v) for v in app.ecs.rigid_body_state[1, dense_idx, 0:3])
        assert before == after, "fixed simulation must not advance while paused"

        menu.resume(app)
        assert app.state_manager.current_state_name == "Playing"
        assert app.ui.active_screen is not None
        assert app.ui.active_screen.name == "PlaygroundHUD"

    def test_respawn_returns_player_to_spawn(self, app) -> None:
        from projects.playground.modules.player import PlayerModule

        player = app.get_module(PlayerModule)
        assert player is not None
        for _ in range(20):
            app.step_frame(1.0 / 60.0)

        player.respawn()
        dense_idx = app.ecs.pool.get_dense_index(player.player_id)
        position = tuple(round(float(v), 3) for v in app.ecs.rigid_body_state[1, dense_idx, 0:3])
        assert position == player.spawn