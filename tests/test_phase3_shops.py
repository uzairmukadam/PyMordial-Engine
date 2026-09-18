"""Unit & Integration tests for Phase 3: Map Tuning Shops, Live Repairs, Bribes & Toast Notifications."""

import tempfile
from pathlib import Path
import pytest

from projects.shotgun_escape_the_heat.modules.career_manager import CareerManagerModule
from projects.shotgun_escape_the_heat.modules.station_manager import StationManagerModule, ToastNotification
from projects.shotgun_escape_the_heat.modules.heat_system import HeatSystemModule
from projects.shotgun_escape_the_heat.modules.vehicle_controller.vehicle_presets import (
    CarClass,
    CLASS_SPECS,
    VEHICLE_CATALOG,
    build_tuned_vehicle_config,
)


def test_chop_shop_upgrades_and_recalc():
    """Verify purchasing Stage 1-3 upgrades updates money, stages, and recalculated configs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_save.json"
        career = CareerManagerModule(save_file_path=save_path, initial_money=25_000)

        # Active starter sedan: initial stage 0 for all
        car = career.active_car
        assert car.engine_stage == 0
        assert car.armor_stage == 0

        # Initial config
        cfg0 = career.get_active_vehicle_config()
        spec = CLASS_SPECS[CarClass.STARTER_SEDAN]

        # Upgrade Engine to Stage 1 ($1,200)
        ok, msg = career.upgrade_spec("engine")
        assert ok is True
        assert car.engine_stage == 1
        assert career.money == 25_000 - 1_200

        cfg1 = career.get_active_vehicle_config()
        assert cfg1.engine_torque > cfg0.engine_torque
        assert pytest.approx(cfg1.engine_torque, 1.0) == cfg0.engine_torque * spec.engine_multipliers[1]

        # Upgrade Armor to Stage 1 ($800)
        ok, msg = career.upgrade_spec("armor")
        assert ok is True
        assert car.armor_stage == 1
        assert spec.armor_damage_factors[1] == 0.80

        # Cannot exceed Stage 3
        career.upgrade_spec("engine")  # Stage 2 ($3,000)
        career.upgrade_spec("engine")  # Stage 3 ($7,500)
        assert car.engine_stage == 3

        ok, msg = career.upgrade_spec("engine")
        assert ok is False
        assert "already at maximum" in msg


def test_chop_shop_and_garage_repairs():
    """Verify repairing damaged car restores mint condition and correctly calculates fees."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_save.json"
        career = CareerManagerModule(save_file_path=save_path, initial_money=2000)

        # Inflict 50% damage: repair cost is 50 * $12 = $600
        career.set_damage(50.0)
        assert career.active_car.damage == 50.0

        ok, msg = career.repair_car()
        assert ok is True
        assert career.active_car.damage == 0.0
        assert career.money == 2000 - 600

        # Repairing when already at 0%
        ok, msg = career.repair_car()
        assert ok is False
        assert "already in pristine condition" in msg


def test_respray_and_bribe_system():
    """Verify vehicle respray and dirty official heat bribe mechanics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_save.json"
        career = CareerManagerModule(save_file_path=save_path, initial_money=5000)
        heat_sys = HeatSystemModule()

        # Respray: costs $350
        new_color = (0.85, 0.12, 0.08)  # Crimson Red
        ok, msg = career.respray_car(new_color)
        assert ok is True
        assert career.active_car.color == new_color
        assert career.money == 5000 - 350

        # Verify respray dynamically notifies VehicleControllerModule if attached
        class MockVC:
            def __init__(self):
                self.color = None
            def set_paint_color(self, app, color):
                self.color = color

        class MockApp:
            def __init__(self, vc):
                self._vc = vc
            def get_module(self, mod_cls):
                return self._vc

        mock_vc = MockVC()
        career._app = MockApp(mock_vc)
        career.respray_car((0.2, 0.7, 0.9))
        assert mock_vc.color == (0.2, 0.7, 0.9)

        # Add heat: 40 points (Tier 2 Heavy Pursuit)
        career.add_heat(40.0)
        heat_sys.heat_level = career.active_car.heat
        assert heat_sys.cop_tier == 2

        # Bribe payoff: 40 * $25 = $1,000
        ok, msg = career.pay_bribe()
        assert ok is True
        assert career.active_car.heat == 0.0
        assert career.money == 5000 - (350 * 2) - 1000

        heat_sys.heat_level = career.active_car.heat
        assert heat_sys.cop_tier == 0
        assert heat_sys.status_label == "CLEAN"


def test_dealership_purchase_and_fleet_switch():
    """Verify purchasing a vehicle adds it to fleet, deducts funds, and allows switching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_save.json"
        career = CareerManagerModule(save_file_path=save_path, initial_money=15_000)

        # Cannot afford Exotic ($45,000)
        ok, msg = career.purchase_car("exotic_coupe")
        assert ok is False
        assert "Insufficient funds" in msg

        # Purchase Classic Muscle ($8,500)
        ok, msg = career.purchase_car("classic_muscle")
        assert ok is True
        assert "classic_muscle" in career.owned_cars
        assert career.money == 15_000 - 8_500

        # Switch to Classic Muscle with app sync verification
        class MockVC:
            def __init__(self):
                self.reconfigured = False
            def apply_vehicle_config(self, app, reset_pose=False):
                self.reconfigured = True

        class MockApp:
            def __init__(self, vc):
                self._vc = vc
            def get_module(self, mod_cls):
                return self._vc

        mock_vc = MockVC()
        career._app = MockApp(mock_vc)
        ok, msg = career.select_active_car("classic_muscle")
        assert ok is True
        assert mock_vc.reconfigured is True
        assert career.active_car_id == "classic_muscle"
        assert career.active_car_definition.car_class == CarClass.CLASSIC_MUSCLE

        # Disk persistence check
        career2 = CareerManagerModule(save_file_path=save_path)
        assert career2.active_car_id == "classic_muscle"
        assert "classic_muscle" in career2.owned_cars
        assert career2.money == 6_500


def test_toast_notification_queue():
    """Verify toast notification queueing and TTL decay."""
    station = StationManagerModule()
    assert len(station.toasts) == 0

    station.add_toast("Test Upgrade Installed", category="UPGRADE", ttl=2.0)
    assert len(station.toasts) == 1
    assert station.toasts[0].message == "Test Upgrade Installed"
    assert station.toasts[0].category == "UPGRADE"

    # Advance 1.0 second
    station.on_fixed_update(None, 1.0)
    assert len(station.toasts) == 1
    assert pytest.approx(station.toasts[0].ttl, 0.01) == 1.0

    # Advance 1.5 seconds (expired)
    station.on_fixed_update(None, 1.5)
    assert len(station.toasts) == 0
