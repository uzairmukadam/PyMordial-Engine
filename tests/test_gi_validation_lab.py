"""Focused tests for the GI Validation Lab example (08_gi_validation_lab.py).

Validates deterministic scene construction, camera presets, GI mode selection,
equalized comparison, and deterministic motion — without brittle pixel checks.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def lab():
    sys.path.insert(0, str(EXAMPLES_DIR))
    module = importlib.import_module("08_gi_validation_lab")
    yield module
    sys.path.remove(str(EXAMPLES_DIR))
    sys.modules.pop("08_gi_validation_lab", None)


class _FakePipeline:
    def __init__(self) -> None:
        from engine.gfx.quality_presets import get_quality_preset, GraphicsQuality
        self.config = get_quality_preset(GraphicsQuality.HIGH)
        self._point = []
        self._spot = []
        self.applied = 0

    def apply_config(self, cfg) -> None:
        self.applied += 1

    def clear_point_lights(self) -> None:
        self._point.clear()

    def clear_spot_lights(self) -> None:
        self._spot.clear()

    def add_point_light(self, position, radius=10.0, color=(1, 1, 1), intensity=2.0):
        light = SimpleNamespace(position=np.array(position, dtype=np.float32))
        self._point.append(light)
        return light

    def add_spot_light(self, **kwargs):
        light = SimpleNamespace(position=np.array(kwargs["position"], dtype=np.float32))
        self._spot.append(light)
        return light


class _FakeECS:
    def __init__(self) -> None:
        self.positions = {}

    def set_position(self, entity_id, pos) -> None:
        self.positions[entity_id] = tuple(float(v) for v in pos)


def _fake_app(lab):
    from engine.camera import FreeFlyCamera
    cam = FreeFlyCamera(position=(0, 0, 0), yaw_deg=0.0, pitch_deg=0.0)
    return SimpleNamespace(
        pipeline=_FakePipeline(),
        ecs=_FakeECS(),
        camera_manager=SimpleNamespace(register_camera=lambda *a, **k: None),
        add_module=lambda m: None,
        input_manager=SimpleNamespace(bind_action=lambda *a, **k: None),
        ui=SimpleNamespace(set_screen=lambda s: None),
        _cam=cam,
    )


class TestSceneDefinition:
    def test_camera_presets_deterministic(self, lab):
        assert len(lab.CAMERA_PRESETS) == 8
        # Each preset is (name, position, target, fov) with finite values.
        for name, pos, target, fov in lab.CAMERA_PRESETS:
            assert isinstance(name, str) and name
            assert len(pos) == 3 and len(target) == 3
            assert all(np.isfinite(v) for v in (*pos, *target, fov))

    def test_amrgi_quality_labels(self, lab):
        assert lab.AMRGI_QUALITY_LABELS == ("Low", "Medium", "High", "Ultra", "Cinematic")
        assert len(lab.AMRGI_QUALITIES) == 5

    def test_motion_endpoints_deterministic(self, lab):
        assert lab.MOTION_A != lab.MOTION_B
        assert lab.MOTION_PERIOD > 0.0
        assert lab.LIGHT_PERIOD > 0.0


class TestOverlayControls:
    def test_gi_mode_cycle(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        seen = [ov.gi_mode]
        for _ in range(4):
            ov.on_event(app, SimpleNamespace(type=768, key=ord("g")))
            seen.append(ov.gi_mode)
        assert seen == ["OFF", "SSGI", "RMGI", "AMRGI", "OFF"]

    def test_equalize_sets_intensity_one(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        app.pipeline.config.amrgi_intensity = 2.5
        app.pipeline.config.ssgi_intensity = 3.0
        ov.equalize(app)
        assert app.pipeline.config.amrgi_intensity == 1.0
        assert app.pipeline.config.ssgi_intensity == 1.0
        assert app.pipeline.config.rmgi_intensity == 1.0
        assert app.pipeline.config.auto_exposure_enabled is False

    def test_amrgi_quality_cycle(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        ov.gi_mode = "AMRGI"
        start = ov.amrgi_quality_idx
        ov.on_event(app, SimpleNamespace(type=768, key=ord("]")))
        assert ov.amrgi_quality_idx == (start + 1) % 5

    def test_motion_reset(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=7)
        ov.on_attach(app)
        ov._time = 3.3
        ov.reset_motion(app)
        assert ov._time == 0.0
        assert app.ecs.positions[7] == lab.MOTION_A

    def test_motion_deterministic(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        ov._time = 0.0
        ov._update_motion(app)
        p0 = app.ecs.positions[1]
        ov._time = lab.MOTION_PERIOD  # full period -> back to start
        ov._update_motion(app)
        p1 = app.ecs.positions[1]
        assert p0 == pytest.approx(p1, abs=1e-5)

    def test_camera_preset_switch(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        ov._set_camera_preset(app, 4)
        _, pos, target, fov = lab.CAMERA_PRESETS[4]
        assert tuple(app._cam.position) == pytest.approx(pos)
        assert tuple(app._cam.target) == pytest.approx(target)
        assert app._cam.fov == fov

    def test_light_config_cycle(self, lab):
        app = _fake_app(lab)
        ov = lab.GIValidationOverlay(app._cam, moving_id=1)
        ov.on_attach(app)
        assert ov.light_config == 0
        ov.on_event(app, SimpleNamespace(type=768, key=ord("l")))
        assert ov.light_config == 1
        assert len(app.pipeline._point) == 1
