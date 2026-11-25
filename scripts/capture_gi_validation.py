"""Controlled screenshot-capture tool for the GI Validation Lab example.

Runs ``examples/08_gi_validation_lab.py`` under a fixed, equalized analytical
state and captures comparable screenshots for GI Off / SSGI / RMGI / AMRGI
(Medium/High/Cinematic), plus GI debug buffers and AMRGI diagnostics where the
engine exposes them.

This tool does NOT modify any GI implementation, scene geometry, or renderer
algorithm. It only drives the existing example and renderer through their public
interfaces and records evidence.

Usage::

    python scripts/capture_gi_validation.py --help
    python scripts/capture_gi_validation.py --resolution 640x360 \
        --cameras reference --methods OFF,SSGI,RMGI,AMRGI \
        --amrgi-quality MEDIUM --output build/gi_validation_captures_smoke
    python scripts/capture_gi_validation.py            # full 1920x1080 matrix

Output (gitignored under build/):
    build/gi_validation_captures/
        capture_manifest.json
        summary.md
        camera_01_reference/final/*.png
        camera_01_reference/gi_buffers/*.png
        camera_01_reference/amrgi_diagnostics/*.png
        contact_sheets/*.png   (if Pillow is available)
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SCHEMA_VERSION = 1
TOOL_NAME = "pymordial-gi-validation-capture"
TOOL_VERSION = "1.0"
FIXED_DT = 1.0 / 60.0

# Camera preset keys (index -> slug) matching examples/08_gi_validation_lab.py.
CAMERA_KEYS = (
    "reference", "rgb", "thin", "shadow", "doorway", "range", "motion", "screen_space",
)

# GI debug view enum values (verified from engine/debug/engine_tweaks.py).
DEBUG_VIEWS = {
    "SSGI": 8,       # GBufferDebugMode.SSGI
    "RMGI": 13,      # GBufferDebugMode.RMGI
    "AMRGI": 24,     # GBufferDebugMode.AMRGI (resolved total)
    "GI_TOTAL": 10,  # GBufferDebugMode.GI_TOTAL (consolidated indirect diffuse)
    "AMRGI_RAW": 25,        # raw trace radiance (pre-reconstruction)
    "AMRGI_CONFIDENCE": 26,  # confidence (grayscale)
    "AMRGI_HIT_RANGE": 27,   # hit range class (R=near, G=mid, B=far)
    "AMRGI_HIT_DISTANCE": 28,  # normalized hit distance
    "AMRGI_VALIDITY": 29,    # validity (G=hit, B=miss, R=background)
}

# AMRGI diagnostic views that are now implemented (Phase 2 of AMRGI-3).
AMRGI_DIAGNOSTIC_VIEWS = (
    "AMRGI_RAW", "AMRGI_CONFIDENCE", "AMRGI_HIT_RANGE",
    "AMRGI_HIT_DISTANCE", "AMRGI_VALIDITY",
)

# AMRGI diagnostic views that would be desirable but are NOT implemented.
UNSUPPORTED_AMRGI_DIAGNOSTICS = (
    "temporal_radiance", "scheduling_mask",
)

LIGHT_CONFIGS = {"SUN": 0, "+POINT": 1, "+SPOT": 2, "+EMISSIVE": 3, "ALL": 4}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        width, height = int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid resolution {text!r}; expected WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return width, height


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled screenshot capture for the GI Validation Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resolution", type=_parse_resolution, default=(1920, 1080))
    parser.add_argument("--output", type=Path, default=Path("build/gi_validation_captures"))
    parser.add_argument("--cameras", type=_csv, default=list(CAMERA_KEYS),
                        help="Comma-separated camera keys (reference,rgb,thin,shadow,doorway,range,motion,screen_space).")
    parser.add_argument("--methods", type=_csv, default=["OFF", "SSGI", "RMGI", "AMRGI"],
                        help="Comma-separated GI methods (OFF,SSGI,RMGI,AMRGI).")
    parser.add_argument("--amrgi-quality", type=_csv, default=["MEDIUM", "HIGH", "CINEMATIC"],
                        help="Comma-separated AMRGI qualities (LOW,MEDIUM,HIGH,ULTRA,CINEMATIC).")
    parser.add_argument("--light-config", default="SUN",
                        help="Default light configuration (SUN,+POINT,+SPOT,+EMISSIVE,ALL).")
    parser.add_argument("--ao", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--warmup", type=int, default=60, help="Frames after app init before any capture.")
    parser.add_argument("--settle", type=int, default=60, help="Frames after each camera/GI switch.")
    parser.add_argument("--capture", type=_csv, default=["final", "gi"],
                        help="What to capture: final,gi,amrgi_diag.")
    parser.add_argument("--motion-state", choices=("paused", "running"), default="paused")
    parser.add_argument("--motion-sequence", action="store_true",
                        help="Capture a deterministic motion sequence instead of the static matrix.")
    parser.add_argument("--motion-frames", type=_csv, default=["0", "15", "30", "60", "90", "120"])
    parser.add_argument("--motion-mode", choices=("object", "light", "both"), default="object")
    parser.add_argument("--include-ui", action="store_true", help="Include UI overlay in captures.")
    parser.add_argument("--contact-sheets", action="store_true", help="Generate contact sheets (needs Pillow).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--quiet", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Example loading
# ---------------------------------------------------------------------------


def load_example():
    """Dynamically imports the numbered example module."""
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    return importlib.import_module("08_gi_validation_lab")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Capture session
# ---------------------------------------------------------------------------


@dataclass
class CaptureRecord:
    camera_key: str
    camera_index: int
    camera_name: str
    camera_position: list[float]
    camera_target: list[float]
    camera_fov: float
    method: str
    amrgi_quality: str | None
    light_config: str
    debug_view: str
    filename: str
    success: bool
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class CaptureSession:
    """Owns one ProjectApp and drives deterministic captures."""

    def __init__(self, args: argparse.Namespace, lab) -> None:
        self.args = args
        self.lab = lab
        self.width, self.height = args.resolution
        self.records: list[CaptureRecord] = []
        self.app = None
        self.meta = None
        self.overlay = None
        self.camera = None
        self._frame = 0

    # -- lifecycle -----------------------------------------------------

    def open(self) -> None:
        from engine.app.config import ProjectConfig
        from engine.app.project_app import ProjectApp

        cfg = ProjectConfig(
            title="PyMordial | GI Validation Capture",
            width=self.width,
            height=self.height,
            quality_preset="high",
            enable_debug=False,
            headless=True,
            particles_enabled=False,
            max_entities=4096,
            vsync=False,
        )
        self.app = ProjectApp(cfg)
        self.meta = self.lab.build_scene(self.app)
        self.overlay = self.meta["overlay"]
        self.camera = self.meta["camera"]
        # Hide the HUD unless explicitly requested.
        if not self.args.include_ui:
            self._hide_ui()
        # Apply the analytical equalized state.
        self.overlay._apply_analytical_config(self.app)
        self.overlay.equalize(self.app)
        self.overlay.motion_enabled = self.args.motion_state == "running"
        self.overlay.moving_light_enabled = False
        self.overlay.light_config = LIGHT_CONFIGS.get(self.args.light_config.upper(), 0)
        self.overlay._apply_lights(self.app)
        self._apply_ao()
        # Initial warmup.
        self._step(self.args.warmup)

    def close(self) -> None:
        if self.app is not None:
            try:
                self.app.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self.app = None

    def _hide_ui(self) -> None:
        try:
            self.app.ui.set_screen(None)
        except Exception:  # noqa: BLE001
            pass

    def _apply_ao(self) -> None:
        cfg = self.app.pipeline.config
        cfg.ao_mode = "GTAO" if self.args.ao == "ON" else "OFF"
        self.app.pipeline.apply_config(cfg)

    # -- stepping ------------------------------------------------------

    def _step(self, frames: int) -> None:
        for _ in range(max(0, frames)):
            self.app.step_frame(FIXED_DT)
            self._frame += 1

    def _invalidate_gi_history(self) -> None:
        for name in ("ssgi_pass", "rmgi_pass", "amrgi_pass"):
            gi_pass = getattr(self.app.pipeline, name, None)
            if gi_pass is not None and hasattr(gi_pass, "invalidate_history"):
                gi_pass.invalidate_history()

    # -- state application --------------------------------------------

    def set_camera(self, index: int) -> None:
        self.overlay._set_camera_preset(self.app, index)
        self._invalidate_gi_history()

    def set_method(self, method: str, amrgi_quality: str | None) -> None:
        self.overlay.gi_mode = method
        if method == "AMRGI" and amrgi_quality is not None:
            labels = [lbl.upper() for lbl in self.lab.AMRGI_QUALITY_LABELS]
            self.overlay.amrgi_quality_idx = labels.index(amrgi_quality.upper())
        self.overlay._apply_gi_mode(self.app)
        self.overlay.equalize(self.app)
        self._invalidate_gi_history()

    def set_debug_view(self, view: str) -> None:
        cfg = self.app.pipeline.config
        cfg.debug_gbuffer = DEBUG_VIEWS.get(view, 0)
        self.app.pipeline.apply_config(cfg)

    # -- capture -------------------------------------------------------

    def _read_final(self) -> np.ndarray:
        data = self.app.pipeline.post_process.final_fbo.read(components=3, dtype="f1")
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(arr)  # moderngl reads bottom-up

    def _save(self, arr: np.ndarray, path: Path) -> bool:
        try:
            from PIL import Image
        except ImportError:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)
        return True

    def capture(self, camera_key: str, camera_index: int, method: str,
                amrgi_quality: str | None, debug_view: str, subdir: str,
                filename: str) -> None:
        _, name, pos, target, fov = self._camera_info(camera_index)
        path = self.args.output / subdir / filename
        ok = False
        note = ""
        try:
            arr = self._read_final()
            ok = self._save(arr, path)
            if not ok:
                note = "Pillow unavailable; image not written"
        except Exception as exc:  # noqa: BLE001
            note = f"capture failed: {exc}"
        self.records.append(CaptureRecord(
            camera_key=camera_key, camera_index=camera_index, camera_name=name,
            camera_position=list(pos), camera_target=list(target), camera_fov=fov,
            method=method, amrgi_quality=amrgi_quality,
            light_config=self.args.light_config.upper(), debug_view=debug_view,
            filename=str(path.relative_to(self.args.output)) if ok else "",
            success=ok, note=note,
        ))

    def _camera_info(self, index: int):
        name, pos, target, fov = self.lab.CAMERA_PRESETS[index]
        return index, name, pos, target, fov

    # -- live config snapshot -----------------------------------------

    def live_config(self) -> dict[str, Any]:
        cfg = self.app.pipeline.config
        return {
            "gi_mode": cfg.gi_mode,
            "ssgi_intensity": cfg.ssgi_intensity,
            "rmgi_intensity": cfg.rmgi_intensity,
            "amrgi_intensity": cfg.amrgi_intensity,
            "amrgi_resolution_scale": cfg.amrgi_resolution_scale,
            "amrgi_ray_count": cfg.amrgi_ray_count,
            "amrgi_near_steps": cfg.amrgi_near_steps,
            "amrgi_mid_steps": cfg.amrgi_mid_steps,
            "amrgi_far_steps": cfg.amrgi_far_steps,
            "amrgi_near_refine_steps": cfg.amrgi_near_refine_steps,
            "amrgi_mid_rate": cfg.amrgi_mid_rate,
            "amrgi_far_rate": cfg.amrgi_far_rate,
            "exposure": cfg.exposure,
            "exposure_compensation": cfg.exposure_compensation,
            "auto_exposure_enabled": cfg.auto_exposure_enabled,
            "ao_mode": cfg.ao_mode,
            "aa_mode": cfg.aa_mode,
            "shadow_filter_mode": cfg.shadow_filter_mode,
            "csm_cascades": cfg.csm_cascades,
            "shadow_resolution": cfg.shadow_resolution,
            "ssr_enabled": cfg.ssr_enabled,
            "bloom_enabled": cfg.bloom_enabled,
            "dof_enabled": cfg.dof_enabled,
            "volumetric_fog_enabled": cfg.volumetric_fog_enabled,
            "debug_gbuffer": cfg.debug_gbuffer,
        }

    def device_info(self) -> dict[str, Any]:
        try:
            info = dict(self.app.render_ctx.ctx.info)
        except Exception:  # noqa: BLE001
            info = {}
        return {
            "gl_renderer": info.get("GL_RENDERER"),
            "gl_version": info.get("GL_VERSION"),
            "gl_vendor": info.get("GL_VENDOR"),
        }


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_").lower()


def camera_dir(index: int, key: str) -> str:
    return f"camera_{index + 1:02d}_{slug(key)}"


def method_slug(method: str, quality: str | None) -> str:
    if method == "AMRGI" and quality:
        return f"amrgi_{quality.lower()}"
    return method.lower()


# ---------------------------------------------------------------------------
# Capture plan
# ---------------------------------------------------------------------------


def _method_variants(args) -> list[tuple[str, str | None]]:
    variants: list[tuple[str, str | None]] = []
    for method in args.methods:
        m = method.upper()
        if m == "AMRGI":
            for q in args.amrgi_quality:
                variants.append(("AMRGI", q.upper()))
        elif m in ("OFF", "SSGI", "RMGI"):
            variants.append((m, None))
    return variants


def run_static_matrix(session: CaptureSession, args) -> None:
    variants = _method_variants(args)
    want_final = "final" in args.capture
    want_gi = "gi" in args.capture
    want_diag = "amrgi_diag" in args.capture
    diag_cameras = {"doorway", "range", "motion"}
    # GI buffers are bounded to the six static diagnostic cameras (spec §21).
    gi_buffer_cameras = {"reference", "rgb", "thin", "shadow", "doorway", "range"}

    for cam_key in args.cameras:
        if cam_key not in CAMERA_KEYS:
            continue
        cam_index = CAMERA_KEYS.index(cam_key)
        cdir = camera_dir(cam_index, cam_key)
        session.set_camera(cam_index)
        session._step(args.settle)

        for method, quality in variants:
            session.set_method(method, quality)
            session._step(args.settle)
            ms = method_slug(method, quality)

            if want_final:
                session.set_debug_view("OFF")
                session._step(2)
                session.capture(cam_key, cam_index, method, quality, "OFF",
                                f"{cdir}/final", f"{cdir.split('/')[-1]}__{ms}__final.png")

            if want_gi and method in ("SSGI", "RMGI", "AMRGI") and cam_key in gi_buffer_cameras:
                session.set_debug_view(method)
                session._step(2)
                session.capture(cam_key, cam_index, method, quality, method,
                                f"{cdir}/gi_buffers", f"{cdir.split('/')[-1]}__{ms}__gi_buffer.png")
                session.set_debug_view("OFF")

            if want_diag and method == "AMRGI" and cam_key in diag_cameras:
                # Capture every implemented AMRGI diagnostic view.
                for view in ("AMRGI",) + AMRGI_DIAGNOSTIC_VIEWS:
                    session.set_debug_view(view)
                    session._step(2)
                    session.capture(cam_key, cam_index, method, quality, view,
                                    f"{cdir}/amrgi_diagnostics",
                                    f"{cdir.split('/')[-1]}__{ms}__{view.lower()}.png")
                session.set_debug_view("OFF")

    session.set_debug_view("OFF")


def run_motion_sequence(session: CaptureSession, args) -> None:
    frames = sorted({int(f) for f in args.motion_frames})
    variants = _method_variants(args)
    cam_key = "motion"
    cam_index = CAMERA_KEYS.index(cam_key)
    cdir = camera_dir(cam_index, cam_key)
    session.set_camera(cam_index)
    session.overlay.motion_enabled = args.motion_mode in ("object", "both")
    session.overlay.moving_light_enabled = args.motion_mode in ("light", "both")
    session.overlay._apply_lights(session.app)

    for method, quality in variants:
        session.set_method(method, quality)
        session.overlay.reset_motion(session.app)
        session._invalidate_gi_history()
        session._step(args.warmup)
        last = 0
        for target in frames:
            session._step(target - last)
            last = target
            ms = method_slug(method, quality)
            session.set_debug_view("OFF")
            session.capture(cam_key, cam_index, method, quality, "OFF",
                            f"motion/{args.motion_mode}/{cdir}",
                            f"{cdir.split('/')[-1]}__{ms}__f{target:03d}.png")


# ---------------------------------------------------------------------------
# Manifest / summary / contact sheets
# ---------------------------------------------------------------------------


def write_manifest(session: CaptureSession, args, started: str) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "timestamp_utc": started,
        "engine_commit": _git_commit(),
        "resolution": {"width": session.width, "height": session.height},
        "device": session.device_info(),
        "controlled_state": {
            "exposure": 1.0,
            "exposure_compensation": 0.0,
            "auto_exposure_enabled": False,
            "gi_intensity": 1.0,
            "ao": args.ao,
            "aa": "FXAA",
            "light_config": args.light_config.upper(),
            "motion_state": args.motion_state,
            "include_ui": args.include_ui,
        },
        "timing": {
            "fixed_dt": FIXED_DT,
            "warmup_frames": args.warmup,
            "settle_frames": args.settle,
        },
        "live_config": session.live_config(),
        "unsupported_amrgi_diagnostics": list(UNSUPPORTED_AMRGI_DIAGNOSTICS),
        "captures": [
            {
                "camera_key": r.camera_key,
                "camera_index": r.camera_index,
                "camera_name": r.camera_name,
                "camera_position": r.camera_position,
                "camera_target": r.camera_target,
                "camera_fov": r.camera_fov,
                "method": r.method,
                "amrgi_quality": r.amrgi_quality,
                "light_config": r.light_config,
                "debug_view": r.debug_view,
                "filename": r.filename,
                "success": r.success,
                "note": r.note,
            }
            for r in session.records
        ],
    }
    (args.output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_summary(session: CaptureSession, args, started: str) -> None:
    ok = sum(1 for r in session.records if r.success)
    failed = [r for r in session.records if not r.success]
    lines = [
        "# GI Validation Capture Summary",
        "",
        f"- Tool: `{TOOL_NAME}` v{TOOL_VERSION}",
        f"- Timestamp (UTC): {started}",
        f"- Engine commit: `{_git_commit()}`",
        f"- Resolution: {session.width}x{session.height}",
        f"- Warmup: {args.warmup} frames | Settle: {args.settle} frames | dt: {FIXED_DT:.6f}s",
        "",
        "## Controlled comparison state",
        "- Exposure 1.0, compensation 0.0, auto-exposure OFF",
        "- GI intensity 1.0 for SSGI / RMGI / AMRGI",
        f"- AO: {args.ao} | AA: FXAA | Light config: {args.light_config.upper()}",
        f"- Motion: {args.motion_state} | UI overlay: {'included' if args.include_ui else 'excluded'}",
        "",
        "## Cameras",
        ", ".join(args.cameras),
        "",
        "## Methods",
        ", ".join(args.methods),
        "",
        "## AMRGI qualities",
        ", ".join(args.amrgi_quality),
        "",
        "## Inventory",
        f"- Captures written: {ok}",
        f"- Captures failed: {len(failed)}",
        "",
        "## Unsupported AMRGI diagnostics",
        "The engine exposes only the AMRGI total debug view (mode 24). The following",
        "diagnostic views do not exist and were NOT fabricated:",
        ", ".join(UNSUPPORTED_AMRGI_DIAGNOSTICS),
        "",
    ]
    if failed:
        lines.append("## Failures")
        for r in failed:
            lines.append(f"- {r.camera_key}/{r.method}/{r.amrgi_quality}: {r.note}")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_contact_sheets(session: CaptureSession, args) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    sheets_dir = args.output / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # Final methods by camera.
    methods = [("OFF", None), ("SSGI", None), ("RMGI", None),
               ("AMRGI", "MEDIUM"), ("AMRGI", "HIGH"), ("AMRGI", "CINEMATIC")]
    rows = []
    for cam_key in args.cameras:
        if cam_key not in CAMERA_KEYS:
            continue
        idx = CAMERA_KEYS.index(cam_key)
        cdir = camera_dir(idx, cam_key)
        row = []
        for method, q in methods:
            ms = method_slug(method, q)
            p = args.output / cdir / "final" / f"{cdir}__{ms}__final.png"
            row.append(p if p.exists() else None)
        if any(row):
            rows.append((cam_key, row))
    if rows:
        _compose_sheet(sheets_dir / "final_methods_by_camera.png", rows,
                       [f"{m}{'' if q is None else ' ' + q}" for m, q in methods], Image, ImageDraw)


def _compose_sheet(path: Path, rows, headers, Image, ImageDraw) -> None:
    thumb_w, thumb_h = 320, 180
    label_h = 18
    cols = len(headers)
    sheet = Image.new("RGB", (cols * thumb_w, len(rows) * (thumb_h + label_h) + label_h), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for c, h in enumerate(headers):
        draw.text((c * thumb_w + 4, 2), h, fill=(230, 230, 230))
    for r, (name, row) in enumerate(rows):
        y = label_h + r * (thumb_h + label_h)
        draw.text((4, y), name, fill=(200, 200, 200))
        for c, p in enumerate(row):
            if p is None:
                continue
            try:
                img = Image.open(p).convert("RGB").resize((thumb_w, thumb_h))
                sheet.paste(img, (c * thumb_w, y + label_h))
            except Exception:  # noqa: BLE001
                continue
    sheet.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error(f"Output directory {args.output} is not empty; pass --overwrite to replace.")
    args.output.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    lab = load_example()
    session = CaptureSession(args, lab)
    try:
        session.open()
        if args.motion_sequence:
            run_motion_sequence(session, args)
        else:
            run_static_matrix(session, args)
        write_manifest(session, args, started)
        write_summary(session, args, started)
        if args.contact_sheets:
            write_contact_sheets(session, args)
    finally:
        session.close()

    ok = sum(1 for r in session.records if r.success)
    if not args.quiet:
        print(f"Captured {ok}/{len(session.records)} images -> {args.output}")
        print(f"Manifest: {args.output / 'capture_manifest.json'}")
        print(f"Summary : {args.output / 'summary.md'}")
    return 0 if ok == len(session.records) else 1


if __name__ == "__main__":
    sys.exit(main())
