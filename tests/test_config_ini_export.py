"""INI settings export tests.

The export path is opt-in per project: ``ProjectConfig.settings_export_path``
is ``None`` unless a project declares one, and the debug menu only offers the
save action when it is set. These tests pin the round-trip fidelity, the section
layout, and the opt-in detection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.app.config import ProjectConfig
from engine.config.ini_export import (
    DEFAULT_SECTION,
    format_value,
    parse_value,
    read_ini,
    section_for_key,
    write_ini,
)
from engine.debug.engine_tweaks import EngineTweaks


class TestValueRoundTrip:
    """Every value type the tweaks dictionary contains must survive a save/load."""

    @pytest.mark.parametrize(
        "value",
        [True, False, 0, 42, -7, 0.0, 1.5, -0.25, 1e-6, "PCSS", "OFF", "EARTH_DAY"],
    )
    def test_scalar_round_trip(self, value: object) -> None:
        assert parse_value(format_value(value)) == value

    def test_tuple_round_trip(self) -> None:
        value = (20.0, 60.0, 160.0, 500.0)
        assert parse_value(format_value(value)) == value

    def test_int_tuple_round_trip(self) -> None:
        value = (1, 2, 3, 4)
        assert parse_value(format_value(value)) == value

    def test_boolean_like_string_is_quoted(self) -> None:
        """A mode name like "OFF" must not be read back as a boolean."""
        assert format_value("OFF") == '"OFF"'
        assert parse_value(format_value("OFF")) == "OFF"

    def test_numeric_string_is_quoted(self) -> None:
        assert parse_value(format_value("123")) == "123"

    def test_float_precision_is_preserved(self) -> None:
        value = 0.1234567890123
        assert parse_value(format_value(value)) == pytest.approx(value, rel=1e-12)


class TestSectionLayout:
    """Sections are derived from key prefixes, not stored in the data."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("shadow_bias", "shadows"),
            ("csm_cascades", "shadows"),
            ("pcf_samples", "shadows"),
            ("pcss_light_size", "shadows"),
            ("ao_mode", "lighting"),
            ("ssgi_temporal_blend", "lighting"),
            ("aa_mode", "antialiasing"),
            ("taa_feedback", "antialiasing"),
            ("dof_enabled", "optics"),
            ("bloom_intensity", "postprocess"),
            ("fog_density", "fog"),
            ("pom_enabled", "microgeometry"),
            ("sun_lux", "atmosphere"),
            ("cloud_coverage", "atmosphere"),
            ("exposure", "postprocess"),
        ],
    )
    def test_key_maps_to_section(self, key: str, expected: str) -> None:
        assert section_for_key(key) == expected

    def test_unknown_key_falls_back_to_general(self) -> None:
        assert section_for_key("something_unmapped") == DEFAULT_SECTION


class TestFileRoundTrip:
    """A full EngineTweaks dictionary must survive write -> read -> from_dict."""

    def test_full_config_round_trip(self, tmp_path: Path) -> None:
        original = EngineTweaks()
        data = original.to_dict()

        target = tmp_path / "settings.ini"
        assert write_ini(target, data, header="test") is True
        assert target.is_file()

        restored = EngineTweaks()
        restored.from_dict(read_ini(target))

        mismatches: list[tuple[str, object, object]] = []
        for key, expected in data.items():
            if not hasattr(restored, key):
                continue
            actual = getattr(restored, key)
            if isinstance(expected, (list, tuple)):
                if tuple(actual) != tuple(expected):
                    mismatches.append((key, expected, actual))
            elif isinstance(expected, float):
                if abs(float(actual) - float(expected)) > 1e-9:
                    mismatches.append((key, expected, actual))
            elif actual != expected:
                mismatches.append((key, expected, actual))

        assert not mismatches, f"Round-trip mismatches: {mismatches[:5]}"

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "settings.ini"
        assert write_ini(target, {"shadow_bias": 0.001}) is True
        assert target.is_file()

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_ini(tmp_path / "does_not_exist.ini") == {}

    def test_unknown_keys_are_ignored_on_load(self, tmp_path: Path) -> None:
        """A settings file from an older build must still load."""
        target = tmp_path / "settings.ini"
        target.write_text("[shadows]\nshadow_bias = 0.002\nlegacy_key = 99\n", encoding="utf-8")

        tweaks = EngineTweaks()
        tweaks.from_dict(read_ini(target))
        assert tweaks.shadow_bias == pytest.approx(0.002)

    def test_inline_comments_are_stripped(self, tmp_path: Path) -> None:
        target = tmp_path / "settings.ini"
        target.write_text("[shadows]\nshadow_bias = 0.002 ; tuned by hand\n", encoding="utf-8")
        assert read_ini(target)["shadow_bias"] == pytest.approx(0.002)

    def test_sections_are_ignored_on_read(self, tmp_path: Path) -> None:
        """The returned dictionary is flat, matching from_dict's expectation."""
        target = tmp_path / "settings.ini"
        target.write_text("[a]\nshadow_bias = 0.001\n[b]\nao_mode = GTAO\n", encoding="utf-8")
        data = read_ini(target)
        assert data["shadow_bias"] == pytest.approx(0.001)
        assert data["ao_mode"] == "GTAO"


class TestProjectOptIn:
    """Export is opt-in: only projects that declare a path get the save action."""

    def test_default_project_has_no_export_path(self) -> None:
        assert ProjectConfig().settings_export_path is None

    def test_project_can_declare_an_export_path(self) -> None:
        cfg = ProjectConfig(settings_export_path="projects/demo/settings.ini")
        assert cfg.settings_export_path == "projects/demo/settings.ini"

    def test_playground_declares_an_export_path(self) -> None:
        from projects.playground.config import SETTINGS_PATH, get_project_config

        cfg = get_project_config()
        assert cfg.settings_export_path is not None
        assert Path(cfg.settings_export_path) == SETTINGS_PATH
        assert SETTINGS_PATH.name == "settings.ini"
        assert SETTINGS_PATH.parent.name == "playground"

    def test_other_projects_do_not_declare_an_export_path(self) -> None:
        """Only the playground opts in for now."""
        from projects.reference_game.config import get_project_config as reference_config

        assert reference_config().settings_export_path is None
