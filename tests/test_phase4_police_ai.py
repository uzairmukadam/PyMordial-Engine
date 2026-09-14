"""Unit and Integration Tests for Phase 4: Police AI Pursuit & Tactical Roadblocks."""

import math
import pytest
from engine.app.project_app import ProjectApp
from projects.shotgun_escape_the_heat.main import create_game
from projects.shotgun_escape_the_heat.modules.heat_system import HeatSystemModule
from projects.shotgun_escape_the_heat.modules.police_ai import PoliceSystemModule
from projects.shotgun_escape_the_heat.modules.police_ai.police_vehicle import PoliceVehicle
from projects.shotgun_escape_the_heat.modules.police_ai.roadblock import TacticalRoadblock
from projects.shotgun_escape_the_heat.modules.vehicle_controller import VehicleControllerModule


@pytest.fixture(scope="module")
def test_app():
    """Creates a headless game application for deterministic testing."""
    app = create_game(headless=True, max_frames=5)
    for _ in range(3):
        app.step_frame(0.02)
    yield app
    app.shutdown()


def test_police_vehicle_spawning(test_app: ProjectApp):
    """Verifies that police cruisers (Sedan and SUV) spawn with chassis, cabin, lightbar, and wheels."""
    sedan = PoliceVehicle(test_app, vehicle_type="sedan", position=(10.0, 1.0, 10.0), heading_deg=0.0)
    assert sedan.chassis_id >= 0
    assert sedan.cabin_id >= 0
    assert sedan.strobe_bar_id >= 0
    assert sedan.beacon_red_id >= 0
    assert sedan.beacon_blue_id >= 0
    assert len(sedan.wheel_entity_ids) == 4
    assert sedan.vehicle is not None

    suv = PoliceVehicle(test_app, vehicle_type="suv", position=(20.0, 1.0, 20.0), heading_deg=45.0)
    assert suv.config.chassis_mass == 2250.0  # Heavy Interceptor SUV mass
    assert suv.vehicle_type == "suv"

    # Cleanup
    sedan.destroy(test_app)
    suv.destroy(test_app)


def test_police_pursuit_steering_and_pit(test_app: ProjectApp):
    """Verifies pursuit steering calculations and PIT maneuver trigger conditions."""
    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(0.0, 1.0, 0.0), heading_deg=0.0)

    # 1. Player directly ahead (-Z direction in PyMordial coordinates)
    cop.update_ai(dt=0.02, player_pos=(0.0, 1.0, -30.0), player_vel=(0.0, 0.0, -10.0))
    # Steer angle should be near 0 (straight ahead) and throttle at maximum
    assert abs(cop.vehicle.steering_input) < 0.2
    assert cop.vehicle.throttle == 1.0

    # 2. Player to the right (+X direction, angle > 0)
    cop.update_ai(dt=0.02, player_pos=(20.0, 1.0, -10.0), player_vel=(0.0, 0.0, 0.0))
    assert cop.vehicle.steering_input > 0.3  # Steers right

    # 3. PIT Maneuver: alongside player at high speed
    cop.vehicle.current_speed_mps = 16.0
    cop.update_ai(dt=0.02, player_pos=(3.0, 1.0, -1.0), player_vel=(0.0, 0.0, -16.0))
    assert cop.pit_timer > 0.0
    assert abs(cop.pit_direction) > 0.8  # Aggressive turn-in to player

    cop.destroy(test_app)


def test_tactical_roadblock_spawning(test_app: ProjectApp):
    """Verifies that tactical roadblocks spawn cruisers, barricades, and fixed physics bodies."""
    rb = TacticalRoadblock(test_app, center_pos=(0.0, 0.0, -60.0), heading_deg=0.0, span_width=14.0)
    assert len(rb.entity_ids) >= 5
    assert len(rb.physics_body_ids) >= 3

    # Ensure colliders were attached in physics simulation
    for bid in rb.physics_body_ids:
        assert bid in test_app.physics._tracked_entities

    rb.destroy(test_app)
    for bid in rb.physics_body_ids:
        assert bid not in test_app.physics._tracked_entities


def test_police_system_tier_escalation(test_app: ProjectApp):
    """Verifies dynamic cop spawning, alert tier escalation, and roadblock deployment."""
    heat_sys = test_app.get_module(HeatSystemModule)
    police_sys = test_app.get_module(PoliceSystemModule)
    vc = test_app.get_module(VehicleControllerModule)

    assert heat_sys is not None
    assert police_sys is not None
    assert vc is not None

    # Step at Tier 0 (Clean)
    heat_sys.heat_level = 0.0
    for cop in police_sys.cops:
        cop.destroy(test_app)
    police_sys.cops.clear()
    for rb in police_sys.roadblocks:
        rb.destroy(test_app)
    police_sys.roadblocks.clear()
    police_sys._spawn_cooldown = 0.0
    for _ in range(3):
        test_app.step_frame(0.02)
    assert len(police_sys.cops) == 0
    assert len(police_sys.roadblocks) == 0

    # Escalate to Tier 1 (Patrol Alert: 25 Heat)
    heat_sys.heat_level = 25.0
    police_sys._spawn_cooldown = 0.0
    for _ in range(5):
        test_app.step_frame(0.02)
    assert len(police_sys.cops) >= 1
    assert any(c.vehicle_type == "sedan" for c in police_sys.cops)

    # Escalate to Tier 2 (Heavy Pursuit: 55 Heat)
    heat_sys.heat_level = 55.0
    police_sys._spawn_cooldown = 0.0
    for _ in range(10):
        test_app.step_frame(0.02)
    assert len(police_sys.cops) >= 2
    # At Tier 2, heavy SUVs are added to the pursuit
    assert any(c.vehicle_type == "suv" for c in police_sys.cops)

    # Escalate to Tier 3 (Full APB: 85 Heat) and simulate movement
    heat_sys.heat_level = 85.0
    police_sys._spawn_cooldown = 0.0
    police_sys._roadblock_cooldown = 0.0
    test_app.physics.set_linvel(vc.chassis_id, (0.0, 0.0, -18.0))
    for _ in range(5):
        test_app.step_frame(0.02)
    police_sys._spawn_cooldown = 0.0
    test_app.physics.set_linvel(vc.chassis_id, (0.0, 0.0, -18.0))
    for _ in range(5):
        test_app.step_frame(0.02)
    assert len(police_sys.cops) >= 3
    assert len(police_sys.roadblocks) >= 1

    # Cool down back to Tier 0 (Clean Bribe/Payoff)
    heat_sys.heat_level = 0.0
    for _ in range(5):
        test_app.step_frame(0.02)
    assert len(police_sys.cops) == 0
    assert len(police_sys.roadblocks) == 0


def test_police_draw_batches_zero_allocation(test_app: ProjectApp):
    """Verifies that police cruisers generate valid MDI draw batches without exceptions."""
    police_sys = test_app.get_module(PoliceSystemModule)
    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(0.0, 1.0, 10.0), heading_deg=0.0)
    police_sys.cops.append(cop)

    batches = police_sys.get_draw_batches(test_app)
    # Chassis, Cabin, Pushbar, Strobe Bar, Red Beacon, Blue Beacon, 4 Wheels = 10 batches per cruiser
    assert len(batches) >= 10

    # Verify second call reuses batch container
    batches_again = police_sys.get_draw_batches(test_app)
    assert batches is batches_again

    cop.destroy(test_app)
    police_sys.cops.clear()


def test_police_vehicle_damage_and_takedown(test_app: ProjectApp):
    """Verifies cop damage tracking, smoke emission, totaling, and takedown bounty awards."""
    from projects.shotgun_escape_the_heat.modules.career_manager import CareerManagerModule
    from projects.shotgun_escape_the_heat.modules.station_manager import StationManagerModule

    police_sys = test_app.get_module(PoliceSystemModule)
    career = test_app.get_module(CareerManagerModule)
    sm = test_app.get_module(StationManagerModule)

    start_money = career.money if career else 0

    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(0.0, 1.0, 10.0), heading_deg=0.0)
    police_sys.cops.append(cop)

    heat_sys = test_app.get_module(HeatSystemModule)
    if heat_sys:
        heat_sys.heat_level = 30.0

    assert cop.damage == 0.0
    assert not cop.is_wrecked

    # Inflict 60% damage -> smoke activates
    cop.add_damage(60.0)
    assert cop.damage == 60.0
    cop.update_visuals(test_app, 0.2)
    # Radiator smoke puffs should be active
    smoke_batches = cop.vfx_mgr.get_draw_batches(test_app)
    assert len(smoke_batches) > 0

    # Inflict another 45% damage -> totaled
    cop.add_damage(45.0)
    assert cop.damage >= 100.0
    assert cop.damage_ctrl.is_totaled

    # Step simulation
    test_app.step_frame(0.02)
    assert cop.is_wrecked
    assert cop.vehicle.throttle == 0.0
    assert cop.vehicle.brake_input == 1.0
    assert cop.bounty_awarded

    # Check career money credited with +$450 sedan bounty
    if career:
        assert career.money >= start_money + 450

    # Check toast notification
    if sm:
        assert any("TAKEDOWN" in t.category for t in sm.toasts)

    police_sys.cops.clear()
    cop.destroy(test_app)


def test_police_whisker_obstacle_avoidance(test_app: ProjectApp):
    """Verifies that cop cruisers detect obstacles using whiskers and steer away or reverse."""
    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(0.0, 1.0, 0.0), heading_deg=0.0)

    # Spawn an obstacle directly ahead and close (2m ahead)
    # In PyMordial, -Z is forward. Front of car is at z=-2.5m. Place box at z=-4.0m
    box_id = test_app.ecs.create_entity(
        position=(0.0, 1.0, -4.0),
        scale=(4.0, 2.0, 2.0),
    )
    test_app.physics.create_body(box_id, body_type="fixed", position=(0.0, 1.0, -4.0))
    test_app.physics.attach_box_collider(box_id, half_x=2.0, half_y=1.0, half_z=1.0)

    # Update AI with cop stopped facing the wall
    cop.vehicle.current_speed_mps = 0.5
    cop.update_ai(dt=0.02, player_pos=(0.0, 1.0, -20.0), player_vel=(0.0, 0.0, 0.0))

    # Should detect close obstacle dead-ahead and trigger immediate reverse!
    assert cop.reverse_timer > 0.0
    assert cop.vehicle.throttle < 0.0  # In reverse throttle

    test_app.physics.remove_body(box_id)
    test_app.ecs.destroy_entity(box_id)
    cop.destroy(test_app)


def test_police_lead_interception(test_app: ProjectApp):
    """Verifies that cop AI leads the player's position based on player velocity vector."""
    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(0.0, 1.0, 0.0), heading_deg=0.0)

    # Player is at (0, 1, -20) moving fast to the right (+X at 25 m/s)
    # Without lead, cop would aim straight at (0, 1, -20) with steer ~ 0.
    # With lead, cop predicts player will be at (+X) and steers hard right (> 0.25).
    cop.update_ai(dt=0.02, player_pos=(0.0, 1.0, -20.0), player_vel=(25.0, 0.0, 0.0))
    assert cop.vehicle.steering_input > 0.25

    cop.destroy(test_app)


def test_player_busted_when_staying_still(test_app: ProjectApp):
    """Verifies that player is busted when remaining still near an active police cruiser."""
    from projects.shotgun_escape_the_heat.modules.career_manager import CareerManagerModule
    from projects.shotgun_escape_the_heat.modules.station_manager import StationManagerModule

    police_sys = test_app.get_module(PoliceSystemModule)
    career = test_app.get_module(CareerManagerModule)
    heat_sys = test_app.get_module(HeatSystemModule)
    sm = test_app.get_module(StationManagerModule)
    vc = test_app.get_module(VehicleControllerModule)

    # Setup: heat active, cop pinned close
    heat_sys.heat_level = 35.0
    career.money = 2000
    start_money = career.money

    p_pos = vc.vehicle.position
    cop = PoliceVehicle(test_app, vehicle_type="sedan", position=(p_pos[0] + 3.0, p_pos[1], p_pos[2]), heading_deg=0.0)
    police_sys.cops.append(cop)

    # Ensure player vehicle is stationary
    test_app.physics.set_linvel(vc.chassis_id, (0.0, 0.0, 0.0))

    # Step simulation frames (dt=0.05 * 45 = 2.25s > 1.8s bust threshold)
    for _ in range(45):
        test_app.step_frame(0.05)

    # Verify arrest state
    assert career.money < start_money  # Fine was deducted
    assert heat_sys.heat_level == 0.0  # Heat cleared
    assert len(police_sys.cops) == 0   # Cruisers despawned
    if sm:
        assert any("BUSTED" in t.category for t in sm.toasts)


def test_heat_increments_on_damage_not_time(test_app: ProjectApp):
    """Verifies that running away does not increase heat over time, but physical damage does."""
    heat_sys = test_app.get_module(HeatSystemModule)
    vc = test_app.get_module(VehicleControllerModule)

    heat_sys.heat_level = 20.0
    heat_sys.cop_count_in_radius = 2
    heat_sys.in_pursuit = True

    # 1. Step simulation without any collision damage: heat must NOT increase over time
    initial_heat = heat_sys.heat_level
    for _ in range(10):
        test_app.step_frame(0.02)
    assert heat_sys.heat_level <= initial_heat

    # 2. Simulate collision impact damage: heat should increase proportionally
    heat_sys.add_heat(15.0)
    assert heat_sys.heat_level >= initial_heat + 14.9
