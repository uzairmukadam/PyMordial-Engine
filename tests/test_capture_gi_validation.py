"""Focused tests for the GI validation capture tool (scripts/capture_gi_validation.py).

Covers CLI parsing, selection helpers, naming, deterministic config generation,
manifest structure, and unsupported-debug handling. The full screenshot matrix is
NOT a unit test; one short headless smoke capture is included.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def cap():
    sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module("capture_gi_validation")
    yield module
    sys.path.remove(str(SCRIPTS_DIR))
    sys.modules.pop("capture_gi_validation", None)


class TestCliParsing:
    def test_defaults(self, cap):
        args = cap.build_parser().parse_args([])
        assert args.resolution == (1920, 1080)
        assert args.methods == ["OFF", "SSGI", "RMGI", "AMRGI"]
        assert args.amrgi_quality == ["MEDIUM", "HIGH", "CINEMATIC"]
        assert args.ao == "OFF"
        assert args.motion_state == "paused"

    def test_resolution_parser(self, cap):
        assert cap._parse_resolution("640x360") == (640, 360)

    def test_resolution_parser_invalid(self, cap):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            cap._parse_resolution("bad")

    def test_csv_parser(self, cap):
        assert cap._csv("a, b ,c") == ["a", "b", "c"]

    def test_camera_selection(self, cap):
        args = cap.build_parser().parse_args(["--cameras", "reference,doorway"])
        assert args.cameras == ["reference", "doorway"]

    def test_method_selection(self, cap):
        args = cap.build_parser().parse_args(["--methods", "OFF,AMRGI"])
        assert args.methods == ["OFF", "AMRGI"]

    def test_amrgi_quality_selection(self, cap):
        args = cap.build_parser().parse_args(["--amrgi-quality", "LOW,ULTRA"])
        assert args.amrgi_quality == ["LOW", "ULTRA"]


class TestNaming:
    def test_slug(self, cap):
        assert cap.slug("Reference Room") == "reference_room"
        assert cap.slug("+Point") == "point"

    def test_camera_dir(self, cap):
        assert cap.camera_dir(0, "reference") == "camera_01_reference"
        assert cap.camera_dir(7, "screen_space") == "camera_08_screen_space"

    def test_method_slug(self, cap):
        assert cap.method_slug("OFF", None) == "off"
        assert cap.method_slug("SSGI", None) == "ssgi"
        assert cap.method_slug("AMRGI", "HIGH") == "amrgi_high"


class TestMethodVariants:
    def test_amrgi_expands_to_qualities(self, cap):
        args = cap.build_parser().parse_args(
            ["--methods", "OFF,AMRGI", "--amrgi-quality", "MEDIUM,HIGH"]
        )
        variants = cap._method_variants(args)
        assert ("OFF", None) in variants
        assert ("AMRGI", "MEDIUM") in variants
        assert ("AMRGI", "HIGH") in variants
        assert len(variants) == 3

    def test_legacy_methods_single(self, cap):
        args = cap.build_parser().parse_args(["--methods", "SSGI,RMGI"])
        variants = cap._method_variants(args)
        assert variants == [("SSGI", None), ("RMGI", None)]


class TestDebugViews:
    def test_debug_view_mapping(self, cap):
        # Verified enum values from engine/debug/engine_tweaks.py.
        assert cap.DEBUG_VIEWS["SSGI"] == 8
        assert cap.DEBUG_VIEWS["RMGI"] == 13
        assert cap.DEBUG_VIEWS["AMRGI"] == 24
        assert cap.DEBUG_VIEWS["GI_TOTAL"] == 10

    def test_unsupported_diagnostics_recorded(self, cap):
        # Views that remain unimplemented must be declared unsupported.
        assert "temporal_radiance" in cap.UNSUPPORTED_AMRGI_DIAGNOSTICS
        assert "scheduling_mask" in cap.UNSUPPORTED_AMRGI_DIAGNOSTICS

    def test_implemented_diagnostics(self, cap):
        # These diagnostic views are now implemented (AMRGI-3 Phase 2).
        for view in ("AMRGI_RAW", "AMRGI_CONFIDENCE", "AMRGI_HIT_RANGE",
                     "AMRGI_HIT_DISTANCE", "AMRGI_VALIDITY"):
            assert view in cap.DEBUG_VIEWS
            assert view in cap.AMRGI_DIAGNOSTIC_VIEWS


class TestLightConfigs:
    def test_light_config_map(self, cap):
        assert cap.LIGHT_CONFIGS["SUN"] == 0
        assert cap.LIGHT_CONFIGS["ALL"] == 4


class TestSmokeCapture:
    """One short headless capture to prove the pipeline writes valid PNGs."""

    def test_reference_smoke(self, cap, tmp_path):
        out = tmp_path / "smoke"
        rc = cap.main([
            "--resolution", "160x120",
            "--cameras", "reference",
            "--methods", "OFF,AMRGI",
            "--amrgi-quality", "MEDIUM",
            "--warmup", "4", "--settle", "4",
            "--output", str(out),
            "--overwrite", "--quiet",
        ])
        assert rc == 0
        manifest = json.loads((out / "capture_manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == cap.SCHEMA_VERSION
        assert manifest["resolution"] == {"width": 160, "height": 120}
        assert manifest["controlled_state"]["gi_intensity"] == 1.0
        assert manifest["controlled_state"]["auto_exposure_enabled"] is False
        # OFF final + AMRGI final + AMRGI gi_buffer = 3 captures.
        assert len(manifest["captures"]) == 3
        assert all(c["success"] for c in manifest["captures"])
        # Verify a PNG exists and has the right dimensions.
        from PIL import Image
        pngs = list(out.rglob("*.png"))
        assert pngs
        img = Image.open(pngs[0])
        assert img.size == (160, 120)
        assert (out / "summary.md").exists()
