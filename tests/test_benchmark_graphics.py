"""Focused tests for the reproducible graphics benchmark tool.

These validate the benchmark's deterministic building blocks (config, scene,
camera, statistics, JSON schema, CLI parsing). The full performance run is not a
unit test.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from engine.gfx.quality_presets import RenderConfig

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module("benchmark_graphics")
    yield module
    sys.path.remove(str(SCRIPTS_DIR))
    sys.modules.pop("benchmark_graphics", None)


class TestConfigModel:
    def test_baseline_low_disables_optional_effects(self, bench):
        cfg = bench.build_config("BASELINE_LOW")
        assert isinstance(cfg, RenderConfig)
        assert cfg.gi_mode == "OFF"
        assert cfg.ssr_enabled is False
        assert cfg.volumetric_fog_enabled is False
        assert cfg.bloom_enabled is False
        assert cfg.tess_enabled is False

    def test_configs_are_independent_copies(self, bench):
        a = bench.build_config("HIGH")
        b = bench.build_config("HIGH")
        a.gi_mode = "OFF"
        assert b.gi_mode != "OFF"

    def test_unknown_config_raises(self, bench):
        with pytest.raises(ValueError):
            bench.build_config("NOPE")

    def test_overrides_applied(self, bench):
        cfg = bench.build_config("BASELINE_LOW", {"gi_mode": "SSGI", "ssr_enabled": True})
        assert cfg.gi_mode == "SSGI"
        assert cfg.ssr_enabled is True

    def test_unknown_override_field_raises(self, bench):
        with pytest.raises(ValueError):
            bench.build_config("BASELINE_LOW", {"not_a_field": 1})

    def test_config_to_dict_is_json_safe(self, bench):
        d = bench.config_to_dict(bench.build_config("HIGH"))
        json.dumps(d)  # must not raise
        assert "gi_mode" in d


class TestCameraWorkloads:
    def test_static_views_ignore_frame_index(self, bench):
        for view in ("VIEW_TOP", "VIEW_CLOSE", "VIEW_REFLECTIVE", "VIEW_DEPTH"):
            assert bench.camera_pose(view, 0) == bench.camera_pose(view, 999)

    def test_motion_view_is_deterministic(self, bench):
        assert bench.camera_pose("VIEW_MOTION", 17) == bench.camera_pose("VIEW_MOTION", 17)
        assert bench.camera_pose("VIEW_MOTION", 0) != bench.camera_pose("VIEW_MOTION", 60)

    def test_unknown_view_raises(self, bench):
        with pytest.raises(ValueError):
            bench.camera_pose("VIEW_NOPE", 0)


class TestStatistics:
    def test_compute_stats_basic(self, bench):
        st = bench.compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert st["min"] == 1.0
        assert st["max"] == 5.0
        assert st["median"] == 3.0
        assert st["samples"] == 5
        assert st["min"] <= st["p95"] <= st["max"]

    def test_compute_stats_empty(self, bench):
        assert bench.compute_stats([]) is None


class TestCliParsing:
    def test_parse_overrides(self, bench):
        out = bench.parse_overrides(["gi_mode=SSGI", "ssr_enabled=true", "ssr_steps=32"])
        assert out == {"gi_mode": "SSGI", "ssr_enabled": True, "ssr_steps": 32}

    def test_parse_overrides_invalid(self, bench):
        with pytest.raises(ValueError):
            bench.parse_overrides(["no_equals_sign"])

    def test_resolution_parser(self, bench):
        assert bench._parse_resolution("1920x1080") == (1920, 1080)

    def test_resolution_parser_invalid(self, bench):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            bench._parse_resolution("bad")

    def test_main_rejects_bad_frames(self, bench):
        with pytest.raises(SystemExit):
            bench.main(["--frames", "0"])

    def test_main_rejects_bad_warmup(self, bench):
        with pytest.raises(SystemExit):
            bench.main(["--warmup", "-1"])


class TestSceneDeterminism:
    def test_scene_metadata_is_stable(self, bench):
        # Scene construction requires a live app; validate the declared metadata
        # contract instead of building a GL context in a unit test.
        assert bench.SCENE_ID == "benchmark_showcase_v1"
        assert "VIEW_TOP" in bench.VIEWS
        assert "BASELINE_LOW" in bench.CONFIG_PRESETS
