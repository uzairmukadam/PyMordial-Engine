"""Dual-Target Release Packager for PyMordial Engine Projects.

Supports two compilation/distribution targets:
1. 'pyinstaller': Fast, accessible bundling with zero C++ compiler requirements (ideal for beginners).
2. 'nuitka': Native Ahead-of-Time (AOT) C++ compilation producing optimized binaries (ideal for advanced users).

Both targets perform:
- Debug stripping: omits debug overlays, profiling hooks, and asserts when --release is enabled.
- Module tree-shaking: copies and stages only active/used modules.
- Asset packing: packs shaders and project assets into zero-copy game.pak containers.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

from engine.assets.vfs import PakWriter


@dataclass
class PackagingConfig:
    """Configuration options for bundling a PyMordial project."""
    project_dir: Path
    target: str = "pyinstaller"           # 'pyinstaller' or 'nuitka'
    release: bool = True                  # When True, strips debug modules and enables full optimizations
    output_dir: Path = Path("dist")
    clean: bool = True
    compress_pak: bool = False
    windowed: bool = True                 # Hide console window on Windows release builds
    dry_run: bool = False
    extra_hidden_imports: list[str] = field(default_factory=list)


@dataclass
class StagingResult:
    """Artifacts created during the project staging step."""
    staging_dir: Path
    entry_point: Path
    pak_path: Path
    used_modules: list[str]
    stripped_modules: list[str]
    total_assets_packed: int


@dataclass
class BuildResult:
    """Outcome of the packaging build process."""
    success: bool
    staging: StagingResult
    command: list[str]
    target: str
    output_dir: Path
    error_message: str | None = None


class ProjectPackager:
    """Packages a PyMordial Project into a distributable standalone release."""

    def __init__(self, config: PackagingConfig) -> None:
        self.config = config
        self.project_dir = Path(config.project_dir).resolve()
        self.project_name = self.project_dir.name
        self.engine_dir = Path(__file__).resolve().parent.parent.parent

    def detect_modules(self) -> tuple[list[str], list[str]]:
        """Identifies modules present in the project and filters debug modules for release builds."""
        modules_dir = self.project_dir / "modules"
        if not modules_dir.exists():
            return [], []

        all_modules = [d.name for d in modules_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]

        used: list[str] = []
        stripped: list[str] = []

        debug_module_names = {"debug_overlay", "profiler_overlay", "dev_cheats"}

        for mod in all_modules:
            if self.config.release and mod in debug_module_names:
                stripped.append(mod)
            else:
                used.append(mod)

        return used, stripped

    def stage(self, staging_dir: Path | None = None) -> StagingResult:
        """Prepares a clean staging directory with code, selected modules, and game.pak."""
        if staging_dir is None:
            staging_dir = self.project_dir / ".staging"

        staging_dir = staging_dir.resolve()
        if self.config.clean and staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        used_modules, stripped_modules = self.detect_modules()

        # 1. Copy engine source
        staged_engine = staging_dir / "engine"
        if staged_engine.exists():
            shutil.rmtree(staged_engine)
        shutil.copytree(
            self.engine_dir / "engine",
            staged_engine,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git*"),
        )

        # 2. Copy project core scripts (config.py, world/, etc.)
        staged_project = staging_dir / "project"
        staged_project.mkdir(parents=True, exist_ok=True)

        for item in self.project_dir.iterdir():
            if item.name in ("modules", "assets", ".staging", "__pycache__", ".git"):
                continue
            dest = staged_project / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, dest)

        # 3. Copy only used modules (tree-shaking)
        staged_modules = staged_project / "modules"
        staged_modules.mkdir(parents=True, exist_ok=True)
        (staged_modules / "__init__.py").touch()

        project_modules_dir = self.project_dir / "modules"
        for mod in used_modules:
            mod_src = project_modules_dir / mod
            if mod_src.exists():
                shutil.copytree(
                    mod_src,
                    staged_modules / mod,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )

        # 4. Pack assets and shaders into game.pak
        pak_writer = PakWriter()
        total_assets = 0

        # Bundle core engine shaders
        shaders_dir = self.engine_dir / "shaders"
        if shaders_dir.exists():
            for root, _, files in os.walk(shaders_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = f"shaders/{full_p.relative_to(shaders_dir).as_posix()}"
                    pak_writer.add_file(rel_p, full_p.read_bytes(), compress=self.config.compress_pak)
                    total_assets += 1

        # Bundle project assets if present
        proj_assets_dir = self.project_dir / "assets"
        if proj_assets_dir.exists():
            for root, _, files in os.walk(proj_assets_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = f"assets/{full_p.relative_to(proj_assets_dir).as_posix()}"
                    pak_writer.add_file(rel_p, full_p.read_bytes(), compress=self.config.compress_pak)
                    total_assets += 1

        # Also copy shaders directory directly to staging for runtime source compilation if needed
        staged_shaders = staging_dir / "shaders"
        if shaders_dir.exists():
            shutil.copytree(
                shaders_dir,
                staged_shaders,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        pak_dest = staging_dir / "game.pak"
        pak_writer.write(pak_dest)

        # 5. Generate standalone entry point runner
        entry_point = staging_dir / "run_game.py"
        entry_content = f'''"""Release runner for {self.project_name} generated by PyMordial Packager."""
import os
import sys
from pathlib import Path

# Set up local paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure release flag if specified
RELEASE_MODE = {self.config.release}

from project.main import main

if __name__ == "__main__":
    main()
'''
        entry_point.write_text(entry_content, encoding="utf-8")

        return StagingResult(
            staging_dir=staging_dir,
            entry_point=entry_point,
            pak_path=pak_dest,
            used_modules=used_modules,
            stripped_modules=stripped_modules,
            total_assets_packed=total_assets,
        )

    def get_pyinstaller_command(self, staging: StagingResult, dist_dir: Path) -> list[str]:
        """Constructs the PyInstaller command arguments."""
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--onedir",
            f"--name={self.project_name}",
            f"--distpath={dist_dir}",
            f"--workpath={staging.staging_dir / 'build'}",
            f"--specpath={staging.staging_dir}",
        ]

        if self.config.windowed and self.config.release:
            cmd.append("--windowed")
        else:
            cmd.append("--console")

        # Add data files (game.pak and shaders)
        sep = ";" if sys.platform == "win32" else ":"
        cmd.extend([
            f"--add-data={staging.pak_path}{sep}.",
            f"--add-data={staging.staging_dir / 'shaders'}{sep}shaders",
        ])

        # Essential hidden imports for OpenGL, Rapier, and PyGame
        hidden_imports = [
            "moderngl",
            "OpenGL",
            "OpenGL.GL",
            "pygame",
            "pygame._sdl2",
            "rapier2d",
            "numpy",
            "zstandard",
            "engine",
            "project",
        ]
        hidden_imports.extend(self.config.extra_hidden_imports)

        for hi in hidden_imports:
            cmd.append(f"--hidden-import={hi}")

        cmd.append(str(staging.entry_point))
        return cmd

    def get_nuitka_command(self, staging: StagingResult, dist_dir: Path) -> list[str]:
        """Constructs the Nuitka native C++ compilation command arguments."""
        cmd = [
            sys.executable,
            "-m",
            "nuitka",
            "--standalone",
            f"--output-dir={dist_dir}",
            f"--output-filename={self.project_name}",
            "--enable-plugin=numpy",
            "--follow-imports",
        ]

        if self.config.windowed and self.config.release:
            cmd.append("--windows-disable-console")

        # Include shaders and pak archive
        cmd.extend([
            f"--include-data-files={staging.pak_path}=game.pak",
            f"--include-data-dir={staging.staging_dir / 'shaders'}=shaders",
        ])

        # Hidden packages
        for pkg in ("moderngl", "pygame", "rapier2d", "zstandard"):
            cmd.append(f"--include-package={pkg}")

        cmd.append(str(staging.entry_point))
        return cmd

    def build(self, dry_run: bool = False) -> BuildResult:
        """Executes the full packaging pipeline."""
        dist_dir = self.config.output_dir.resolve()
        dist_dir.mkdir(parents=True, exist_ok=True)

        staging = self.stage()

        if self.config.target.lower() == "nuitka":
            cmd = self.get_nuitka_command(staging, dist_dir)
        else:
            cmd = self.get_pyinstaller_command(staging, dist_dir)

        if dry_run or self.config.dry_run:
            return BuildResult(
                success=True,
                staging=staging,
                command=cmd,
                target=self.config.target,
                output_dir=dist_dir,
            )

        # Check if the chosen packager is installed
        packager_mod = "PyInstaller" if self.config.target.lower() == "pyinstaller" else "nuitka"
        if importlib.util.find_spec(packager_mod) is None:
            return BuildResult(
                success=False,
                staging=staging,
                command=cmd,
                target=self.config.target,
                output_dir=dist_dir,
                error_message=(
                    f"Required packaging package '{packager_mod}' is not installed in the current environment.\n"
                    f"To install run: pip install {packager_mod.lower()}"
                ),
            )

        try:
            subprocess.run(cmd, check=True, cwd=str(staging.staging_dir))
            return BuildResult(
                success=True,
                staging=staging,
                command=cmd,
                target=self.config.target,
                output_dir=dist_dir,
            )
        except subprocess.CalledProcessError as e:
            return BuildResult(
                success=False,
                staging=staging,
                command=cmd,
                target=self.config.target,
                output_dir=dist_dir,
                error_message=f"Packaging build failed with returncode {e.returncode}",
            )


def main() -> None:
    """CLI entry point for the PyMordial project packager."""
    parser = argparse.ArgumentParser(
        description="PyMordial Engine Dual-Target Release Packager (PyInstaller / Nuitka)"
    )
    parser.add_argument(
        "-p", "--project",
        type=str,
        required=True,
        help="Path or name of the project to package (e.g. 'shotgun_escape_the_heat' or 'projects/shotgun_escape_the_heat')",
    )
    parser.add_argument(
        "-t", "--target",
        choices=["pyinstaller", "nuitka"],
        default="pyinstaller",
        help="Target compiler/bundler (default: 'pyinstaller' for zero C++ tool dependencies)",
    )
    parser.add_argument(
        "--release",
        dest="release",
        action="store_true",
        default=True,
        help="Build release binary without debug overlays and profiling overhead (default: True)",
    )
    parser.add_argument(
        "--debug",
        dest="release",
        action="store_false",
        help="Build debug binary including debug overlays and console output",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("dist"),
        help="Destination directory for distribution build (default: dist/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files and output build command without invoking external compilers",
    )
    parser.add_argument(
        "--compress-pak",
        action="store_true",
        help="Compress assets using Zstandard inside game.pak",
    )

    args = parser.parse_args()

    project_path = Path(args.project)
    if not project_path.exists():
        # Check inside projects/
        cand = Path("projects") / args.project
        if cand.exists():
            project_path = cand
        else:
            print(f"[ERROR] Project '{args.project}' not found.", file=sys.stderr)
            sys.exit(1)

    cfg = PackagingConfig(
        project_dir=project_path,
        target=args.target,
        release=args.release,
        output_dir=args.output,
        compress_pak=args.compress_pak,
        dry_run=args.dry_run,
    )

    packager = ProjectPackager(cfg)
    result = packager.build(dry_run=args.dry_run)

    print("==================================================")
    print(f" PyMordial Packager: {cfg.project_dir.name}")
    print(f" Target: {result.target.upper()} | Mode: {'RELEASE' if cfg.release else 'DEBUG'}")
    print(f" Used Modules: {len(result.staging.used_modules)} {result.staging.used_modules}")
    print(f" Stripped Modules: {len(result.staging.stripped_modules)} {result.staging.stripped_modules}")
    print(f" Packed Assets: {result.staging.total_assets_packed} items -> {result.staging.pak_path.name}")
    print(f" Planned Command: {' '.join(result.command)}")
    print("==================================================")

    if not result.success:
        print(f"[ERROR] {result.error_message}", file=sys.stderr)
        sys.exit(1)
    else:
        print("[SUCCESS] Project packaging completed successfully.")


if __name__ == "__main__":
    main()
