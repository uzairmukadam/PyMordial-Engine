"""PyMordial Engine Project CLI Tool.

Provides commands to create, validate, test, and run PyMordial game projects.

Usage:
    python -m engine.tools.project validate --project <name_or_path>
    python -m engine.tools.project run --project <name_or_path> [--headless] [--frames N]
    python -m engine.tools.project create --project <name> [--title "My Game"]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from engine.app.project_validator import (
    resolve_project_path,
    validate_project_config,
    validate_project_directory,
)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validates the structure and configuration of a game project."""
    engine_root = Path(__file__).resolve().parent.parent.parent
    project_dir = resolve_project_path(args.project, engine_root)

    print(f"[*] Validating project at: {project_dir}")
    report = validate_project_directory(project_dir)

    for warning in report.warnings:
        print(f"  [WARN] {warning}")

    if not report.is_valid or report.errors:
        print("\n[!] Project validation FAILED:")
        for err in report.errors:
            print(f"  - {err}")
        return 1

    print("[+] Directory structure: VALID")

    # Try importing config.py if it exists
    config_py = project_dir / "config.py"
    if config_py.exists():
        added_paths = [str(engine_root), str(project_dir.parent)]
        for entry in added_paths:
            sys.path.insert(0, entry)
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"{project_dir.name}_config", config_py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                getter = getattr(mod, "get_project_config", None)
                if callable(getter):
                    cfg = getter()
                    cfg_report = validate_project_config(cfg)
                    for err in cfg_report.errors:
                        print(f"  [ERROR] Config: {err}")
                    if not cfg_report.is_valid:
                        return 1
                    print("[+] Configuration shape: VALID")
        except Exception as e:
            print(f"  [ERROR] Failed to load config.py: {e}")
            return 1
        finally:
            for entry in added_paths:
                try:
                    sys.path.remove(entry)
                except ValueError:
                    pass

    print("[SUCCESS] Project passed all validation checks.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Runs a game project entry point."""
    engine_root = Path(__file__).resolve().parent.parent.parent
    project_dir = resolve_project_path(args.project, engine_root)

    report = validate_project_directory(project_dir)
    if not report.is_valid:
        print("[!] Cannot run invalid project:")
        for err in report.errors:
            print(f"  - {err}")
        return 1

    main_py = project_dir / "main.py"
    cmd = [sys.executable, str(main_py)]
    if args.headless:
        cmd.append("--headless")
    if args.frames:
        cmd.extend(["--frames", str(args.frames)])

    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{engine_root}{os.pathsep}{pythonpath}" if pythonpath else str(engine_root)

    print(f"[*] Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(engine_root), env=env)
    return result.returncode


def cmd_create(args: argparse.Namespace) -> int:
    """Scaffolds a new game project with standard directory structure."""
    engine_root = Path(__file__).resolve().parent.parent.parent
    project_name = args.project.strip().replace(" ", "_").lower()
    projects_root = engine_root / "projects"
    project_dir = projects_root / project_name

    if project_dir.exists():
        print(f"[!] Target directory already exists: {project_dir}")
        return 1

    project_title = args.title if args.title else project_name.replace("_", " ").title()
    template_dir = projects_root / "starter_template"

    if template_dir.exists():
        shutil.copytree(template_dir, project_dir)
        # Update config title
        cfg_file = project_dir / "config.py"
        if cfg_file.exists():
            content = cfg_file.read_text(encoding="utf-8")
            content = content.replace("Starter Game", project_title)
            cfg_file.write_text(content, encoding="utf-8")
    else:
        # Create minimal directories
        (project_dir / "modules").mkdir(parents=True, exist_ok=True)
        (project_dir / "world").mkdir(parents=True, exist_ok=True)
        (project_dir / "assets").mkdir(parents=True, exist_ok=True)

        (project_dir / "__init__.py").write_text("", encoding="utf-8")
        (project_dir / "modules" / "__init__.py").write_text("", encoding="utf-8")
        (project_dir / "world" / "__init__.py").write_text("", encoding="utf-8")

        # config.py
        (project_dir / "config.py").write_text(
            f'''"""Project configuration for {project_title}."""

from engine.app.config import ProjectConfig


def get_project_config() -> ProjectConfig:
    return ProjectConfig(
        title="{project_title}",
        width=1280,
        height=720,
        quality_preset="high",
        fixed_hz=60.0,
        gravity=(0.0, -9.81, 0.0),
        enable_debug=True,
    )
''',
            encoding="utf-8",
        )

        # main.py
        (project_dir / "main.py").write_text(
            f'''"""Entry point for {project_title}."""

import argparse
from engine.app.project_app import ProjectApp
from engine.world.default_world import DefaultWorldBuilder

try:
    from projects.{project_name}.config import get_project_config
except ImportError:
    from config import get_project_config


def main() -> None:
    parser = argparse.ArgumentParser(description="{project_title}")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    cfg = get_project_config()
    if args.headless:
        cfg.headless = True
    if args.frames:
        cfg.max_frames = args.frames

    app = ProjectApp(cfg)
    app.set_world_builder(DefaultWorldBuilder(size=50.0))
    app.run()


if __name__ == "__main__":
    main()
''',
            encoding="utf-8",
        )

    print(f"[+] Successfully created new game project at: {project_dir}")
    print(f"[*] To run: python -m engine.tools.project run --project {project_name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m engine.tools.project", description="PyMordial Project Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    v_parser = subparsers.add_parser("validate", help="Validate a project directory")
    v_parser.add_argument("--project", "-p", required=True, help="Project name under projects/ or directory path")

    # run
    r_parser = subparsers.add_parser("run", help="Run a project")
    r_parser.add_argument("--project", "-p", required=True, help="Project name under projects/ or directory path")
    r_parser.add_argument("--headless", action="store_true", help="Run in hidden headless mode")
    r_parser.add_argument("--frames", type=int, default=None, help="Exit after N frames")

    # create
    c_parser = subparsers.add_parser("create", help="Create a new game project")
    c_parser.add_argument("--project", "-p", required=True, help="Name for new project folder")
    c_parser.add_argument("--title", "-t", default=None, help="Display title for the game")

    args = parser.parse_args()
    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "create":
        sys.exit(cmd_create(args))


if __name__ == "__main__":
    main()
