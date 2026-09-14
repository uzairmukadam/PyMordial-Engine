"""Unit and Integration Tests for Phase 5: PSX Driver Pedestrian System."""

import math
import pytest
from engine.app.project_app import ProjectApp
from projects.shotgun_escape_the_heat.main import create_game
from projects.shotgun_escape_the_heat.modules.pedestrians import PedestrianModule
from projects.shotgun_escape_the_heat.modules.pedestrians.pedestrian import Pedestrian


@pytest.fixture(scope="module")
def test_app():
    """Creates a headless game application for deterministic pedestrian testing."""
    app = create_game(headless=True, max_frames=5)
    for _ in range(3):
        app.step_frame(0.02)
    yield app
    app.shutdown()


def test_pedestrian_spawning(test_app: ProjectApp):
    """Verifies that pedestrian creates torso capsule and head sphere entities with proper styling."""
    waypoints = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 10.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=1.5, start_waypoint_idx=0)

    assert ped.torso_id >= 0
    assert ped.head_id >= 0
    assert ped.state == Pedestrian.STATE_WALKING
    assert ped.position[0] == 0.0
    assert ped.current_wp_idx == 0

    ped.destroy(test_app)


def test_pedestrian_patrol_waypoints(test_app: ProjectApp):
    """Verifies that pedestrian moves along waypoints and advances stride phase."""
    waypoints = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=2.0, start_waypoint_idx=0)

    initial_x = ped.position[0]
    initial_phase = ped.stride_phase

    # Step simulation with car parked far away
    car_pos = (0.0, 0.0, 50.0)
    car_vel = (0.0, 0.0, 0.0)
    for _ in range(10):
        ped.fixed_update(0.05, car_pos, car_vel)

    assert ped.position[0] > initial_x
    assert ped.stride_phase != initial_phase

    ped.destroy(test_app)


def test_pedestrian_dodge_trigger(test_app: ProjectApp):
    """Verifies that an oncoming car heading towards a pedestrian triggers an evasive dive."""
    waypoints = [(0.0, 0.0, -10.0), (10.0, 0.0, -10.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=1.2)

    # Car positioned at (0, 0, 0) speeding directly along -Z towards pedestrian at (0, 0, -10)
    car_pos = (0.0, 0.0, 0.0)
    car_vel = (0.0, 0.0, -16.0)

    ped.fixed_update(0.02, car_pos, car_vel)

    # Threat confirmed: pedestrian must enter DODGING state
    assert ped.state == Pedestrian.STATE_DODGING
    assert ped.dodge_timer == 0.02
    # Evasive dive velocity must be perpendicular to vehicle trajectory (along X axis)
    assert abs(ped.dodge_vel[0]) > 4.0
    assert abs(ped.dodge_vel[2]) < 0.1

    ped.destroy(test_app)


def test_pedestrian_ignore_non_threats(test_app: ProjectApp):
    """Verifies that pedestrian does not dodge when vehicle is distant, slow, or moving away."""
    waypoints = [(0.0, 0.0, -10.0), (10.0, 0.0, -10.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=1.2)

    # 1. Distant vehicle (> 25m)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, 30.0), car_vel=(0.0, 0.0, -15.0))
    assert ped.state == Pedestrian.STATE_WALKING

    # 2. Creeping / stopped vehicle (< 3.8 m/s)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, -4.0), car_vel=(0.0, 0.0, -2.0))
    assert ped.state == Pedestrian.STATE_WALKING

    # 3. Vehicle moving away (+Z)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, -5.0), car_vel=(0.0, 0.0, 15.0))
    assert ped.state == Pedestrian.STATE_WALKING

    ped.destroy(test_app)


def test_pedestrian_recovery_state_transition(test_app: ProjectApp):
    """Verifies transition from DODGE_JUMP -> RECOVER -> RUN -> WALKING once car passes."""
    waypoints = [(0.0, 0.0, -10.0), (10.0, 0.0, -10.0)]
    ped = Pedestrian(test_app, waypoints=waypoints)

    # Trigger dodge
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, 0.0), car_vel=(0.0, 0.0, -16.0))
    assert ped.state == Pedestrian.STATE_DODGE_JUMP

    # Step through dodge duration (0.65s, car far so no roll)
    for _ in range(35):
        ped.fixed_update(0.02, car_pos=(0.0, 0.0, 0.0), car_vel=(0.0, 0.0, -16.0))

    assert ped.state == Pedestrian.STATE_RECOVER

    # Step through recovery (0.85s = ~43 steps) -> transitions to RUNNING
    for _ in range(50):
        ped.fixed_update(0.02, car_pos=(0.0, 0.0, -35.0), car_vel=(0.0, 0.0, -16.0))

    assert ped.state == Pedestrian.STATE_RUNNING

    # Step through sprint run duration (2.2s = 110 steps) -> transitions to WALKING
    for _ in range(120):
        ped.fixed_update(0.02, car_pos=(0.0, 0.0, -35.0), car_vel=(0.0, 0.0, -16.0))

    assert ped.state == Pedestrian.STATE_WALKING

    ped.destroy(test_app)


def test_pedestrian_full_seven_stage_state_machine(test_app: ProjectApp):
    """Verifies the complete 7-stage PSX Driver sequence:
    Walking -> Run -> Dodge Jump -> Roll (car still close) -> Recover -> Run -> Walking.
    """
    waypoints = [(0.0, 0.0, -18.0), (10.0, 0.0, -18.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=1.35)

    # 1. WALKING (initial state)
    assert ped.state == Pedestrian.STATE_WALKING

    # 2. RUN (early warning horizon: car approaching at 18m away, speed 15 m/s)
    # Car at (0, 0, 0), ped at (0, 0, -18), along_path = 18m (between 13.5m and 22m)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, 0.0), car_vel=(0.0, 0.0, -15.0))
    assert ped.state == Pedestrian.STATE_RUNNING

    # 3. DODGE JUMP (immediate threat horizon: car reaches 10m away)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, -8.0), car_vel=(0.0, 0.0, -15.0))
    assert ped.state == Pedestrian.STATE_DODGE_JUMP
    assert ped.dodge_timer > 0.0

    # Step through dodge jump duration (0.65s = ~33 steps).
    # Maintain car position close to pedestrian (car at 4.0m distance)
    for _ in range(35):
        car_near = (float(ped.position[0]), 0.0, float(ped.position[2] + 4.0))
        ped.fixed_update(0.02, car_pos=car_near, car_vel=(0.0, 0.0, -5.0))

    # 4. ROLL (car is still close <= 5.8m on landing)
    assert ped.state == Pedestrian.STATE_ROLL
    assert ped.roll_timer > 0.0

    # Step through roll duration (0.52s = 26 steps)
    for _ in range(30):
        car_near = (float(ped.position[0]), 0.0, float(ped.position[2] + 4.0))
        ped.fixed_update(0.02, car_pos=car_near, car_vel=(0.0, 0.0, -5.0))

    # 5. RECOVER (shocked crouch)
    assert ped.state == Pedestrian.STATE_RECOVER

    # Step through recovery while car speeds far away (dist > 5.5m)
    car_distant = (0.0, 0.0, 100.0)
    for _ in range(45):  # 0.9s > 0.85s recovery duration
        ped.fixed_update(0.02, car_pos=car_distant, car_vel=(0.0, 0.0, 15.0))

    # 6. RUN (post-recovery sprint away from roadway)
    assert ped.state == Pedestrian.STATE_RUNNING

    # Step through sprint run duration (2.2s)
    for _ in range(115):  # 2.3s > 2.2s
        ped.fixed_update(0.02, car_pos=car_distant, car_vel=(0.0, 0.0, 15.0))

    # 7. WALKING (settles back to normal sidewalk patrol)
    assert ped.state == Pedestrian.STATE_WALKING

    ped.destroy(test_app)


def test_pedestrian_inviolability_cannot_be_hit(test_app: ProjectApp):
    """Verifies that vehicle can never collide with or clip through pedestrians (PSX Driver inviolability)."""
    waypoints = [(0.0, 0.0, -10.0), (10.0, 0.0, -10.0)]
    ped = Pedestrian(test_app, waypoints=waypoints, walk_speed=1.2)

    # 1. High-speed approach (45 m/s = 100 mph directly at pedestrian)
    ped.fixed_update(0.02, car_pos=(0.0, 0.0, 0.0), car_vel=(0.0, 0.0, -45.0))
    assert ped.state == Pedestrian.STATE_DODGE_JUMP
    # Evasive jump speed must be high enough to clear vehicle width
    assert abs(ped.dodge_vel[0]) >= 7.8

    # 2. Proximity panic when car is idling or creeping within 4m
    ped2 = Pedestrian(test_app, waypoints=[(5.0, 0.0, 5.0), (15.0, 0.0, 5.0)])
    ped2.fixed_update(0.02, car_pos=(5.0, 0.0, 7.5), car_vel=(0.0, 0.0, -0.5))
    assert ped2.state == Pedestrian.STATE_DODGE_JUMP
    ped2.destroy(test_app)

    # 3. Guaranteed clearance hull prevents chassis intersection even at (0, 0, 0) overlap
    ped3 = Pedestrian(test_app, waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
    # Position car exactly at pedestrian's location
    car_center = (0.0, 0.0, 0.0)
    ped3.fixed_update(0.02, car_pos=car_center, car_vel=(0.0, 0.0, 0.0))
    # Pedestrian must be pushed laterally outside the vehicle hull (> 1.45m)
    assert abs(ped3.position[0] - car_center[0]) >= 1.45
    assert ped3.state in (Pedestrian.STATE_DODGE_JUMP, Pedestrian.STATE_ROLL)
    ped3.destroy(test_app)

    ped.destroy(test_app)




def test_pedestrian_module_and_draw_batches(test_app: ProjectApp):
    """Verifies PedestrianModule ambient spawning and zero-allocation MDI batch generation."""
    ped_mod = test_app.get_module(PedestrianModule)
    assert ped_mod is not None
    assert len(ped_mod.pedestrians) >= 10

    # Test MDI draw batch generation
    batches = ped_mod.get_draw_batches(test_app)
    # Each pedestrian has 1 torso + 1 head = 2 batches
    assert len(batches) == len(ped_mod.pedestrians) * 2

    # Verify zero-allocation batch container reuse
    batches_again = ped_mod.get_draw_batches(test_app)
    assert batches is batches_again
