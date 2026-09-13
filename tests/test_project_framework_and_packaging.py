"""Unit and Integration tests for Project Framework, World Builders, Modules, and Dual-Target Packager."""

from pathlib import Path
import numpy as np

from engine.app.config import ProjectConfig
from engine.app.module import ProjectModule
from engine.app.project_app import ProjectApp
from engine.world.default_world import DefaultWorldBuilder, make_tiled_plane_pm_mesh
from engine.tools.packager import PackagingConfig, ProjectPackager


class DummyModule(ProjectModule):
    """Test module tracking lifecycle callback execution."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.enabled = True
        self.attached = False
        self.detached = False
        self.fixed_count = 0
        self.update_count = 0
        self.ui_count = 0

    def on_attach(self, app: ProjectApp) -> None:
        self.attached = True

    def on_fixed_update(self, app: ProjectApp, dt: float) -> None:
        self.fixed_count += 1

    def on_update(self, app: ProjectApp, dt: float) -> None:
        self.update_count += 1

    def on_ui(self, app: ProjectApp) -> None:
        self.ui_count += 1

    def on_detach(self, app: ProjectApp) -> None:
        self.detached = True


def test_project_config_defaults():
    """Validates default project configuration values."""
    cfg = ProjectConfig(title="TestGame")
    assert cfg.title == "TestGame"
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.headless is False
    assert cfg.fixed_hz == 60.0
    assert cfg.quality_preset == "ultra"


def test_default_world_builder_plane_mesh():
    """Validates generation of tiled PBR mesh for default world."""
    mesh = make_tiled_plane_pm_mesh(size=10.0, uv_tiles=4.0)
    assert mesh.vertex_count == 4
    assert mesh.index_count == 6
    assert len(mesh.vertices) == 4
    assert mesh.indices.dtype == np.uint32


def test_project_app_headless_lifecycle():
    """Runs a complete headless test of ProjectApp with world builder and dummy module."""
    cfg = ProjectConfig(
        title="TestProjectApp",
        width=320,
        height=240,
        headless=True,
    )
    app = ProjectApp(cfg)
    world = DefaultWorldBuilder(size=20.0)
    app.set_world_builder(world)

    mod = DummyModule()
    app.add_module(mod)
    assert mod.attached is True
    assert app.physics.world is not None
    assert app.ecs.active_count >= 1  # Ground entity created

    # Step 5 simulation frames
    for _ in range(5):
        app.step_frame(1.0 / 60.0)

    assert mod.fixed_count >= 1
    assert mod.update_count >= 1

    # Teardown
    app.shutdown()
    assert mod.detached is True


def test_packager_module_detection_and_tree_shaking(tmp_path: Path):
    """Verifies that packager detects modules, strips debug modules in release, and retains them in debug."""
    project_dir = tmp_path / "test_game"
    project_dir.mkdir()
    modules_dir = project_dir / "modules"
    modules_dir.mkdir()

    # Create dummy modules
    (modules_dir / "gameplay").mkdir()
    (modules_dir / "weapons").mkdir()
    (modules_dir / "debug_overlay").mkdir()

    # Release mode: should strip debug_overlay
    cfg_release = PackagingConfig(project_dir=project_dir, release=True, dry_run=True)
    packager_release = ProjectPackager(cfg_release)
    used, stripped = packager_release.detect_modules()
    assert "gameplay" in used
    assert "weapons" in used
    assert "debug_overlay" not in used
    assert "debug_overlay" in stripped

    # Debug mode: should retain debug_overlay
    cfg_debug = PackagingConfig(project_dir=project_dir, release=False, dry_run=True)
    packager_debug = ProjectPackager(cfg_debug)
    used_dbg, stripped_dbg = packager_debug.detect_modules()
    assert "debug_overlay" in used_dbg
    assert len(stripped_dbg) == 0


def test_packager_commands_pyinstaller_and_nuitka(tmp_path: Path):
    """Verifies that PyInstaller and Nuitka generate valid command arguments."""
    project_dir = tmp_path / "sample_game"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    # PyInstaller
    cfg_py = PackagingConfig(project_dir=project_dir, target="pyinstaller", release=True, dry_run=True)
    res_py = ProjectPackager(cfg_py).build(dry_run=True)
    assert res_py.success is True
    cmd_str_py = " ".join(res_py.command)
    assert "PyInstaller" in cmd_str_py
    assert "--onedir" in cmd_str_py
    assert "--windowed" in cmd_str_py
    assert "--hidden-import=moderngl" in cmd_str_py
    assert "--hidden-import=rapier2d" in cmd_str_py

    # Nuitka
    cfg_nk = PackagingConfig(project_dir=project_dir, target="nuitka", release=True, dry_run=True)
    res_nk = ProjectPackager(cfg_nk).build(dry_run=True)
    assert res_nk.success is True
    cmd_str_nk = " ".join(res_nk.command)
    assert "nuitka" in cmd_str_nk
    assert "--standalone" in cmd_str_nk
    assert "--windows-disable-console" in cmd_str_nk
    assert "--include-package=moderngl" in cmd_str_nk
