"""Opt-in Decoupled Physics Pick-and-Carry and Throw System.

Completely decoupled from the character locomotion controller. Games that require
Half-Life / Portal style physics grabbing simply instantiate `PhysicsGrabber`.
Games that do not need it never import or instantiate it, incurring zero overhead.
"""

from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from engine.audio.audio_engine import AudioEngine
    from engine.physics.props import InteractivePropManager
    from engine.physics.rapier_world import PhysicsManager


class PhysicsGrabber:
    """Standalone physics interaction manipulator for grabbing, carrying, and throwing props."""

    __slots__ = (
        "physics",
        "prop_manager",
        "audio",
        "max_reach",
        "hold_distance",
        "spring_stiffness",
        "throw_force",
        "held_entity_id",
        "_scratch_target",
    )

    def __init__(
        self,
        physics: PhysicsManager,
        prop_manager: InteractivePropManager,
        audio: AudioEngine | None = None,
        max_reach: float = 3.8,
        hold_distance: float = 2.2,
        spring_stiffness: float = 24.0,
        throw_force: float = 11.0,
    ) -> None:
        self.physics = physics
        self.prop_manager = prop_manager
        self.audio = audio
        self.max_reach = float(max_reach)
        self.hold_distance = float(hold_distance)
        self.spring_stiffness = float(spring_stiffness)
        self.throw_force = float(throw_force)
        self.held_entity_id: int | None = None
        self._scratch_target = np.zeros(3, dtype=np.float32)

    @property
    def is_holding(self) -> bool:
        """True if currently holding an active physics prop."""
        return self.held_entity_id is not None

    def try_grab(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        ignore_entity_id: int | None = None,
    ) -> bool:
        """Casts a ray to grab a dynamic prop within reach."""
        if self.held_entity_id is not None:
            return False

        dir_vec = np.array(direction, dtype=np.float32)
        dir_len = float(np.linalg.norm(dir_vec))
        if dir_len < 1e-4:
            return False
        dir_vec /= dir_len

        orig_x, orig_y, orig_z = float(origin[0]), float(origin[1]), float(origin[2])
        hit = self.physics.raycast((orig_x, orig_y, orig_z), dir_vec, max_distance=self.max_reach)

        # If ray hits an ignored entity (e.g. player capsule in 3rd-person), advance past it
        if hit is not None and ignore_entity_id is not None and hit[0] == ignore_entity_id:
            advance = hit[1] + 0.6
            orig_x += float(dir_vec[0]) * advance
            orig_y += float(dir_vec[1]) * advance
            orig_z += float(dir_vec[2]) * advance
            rem_reach = max(1.0, self.max_reach - advance)
            hit = self.physics.raycast((orig_x, orig_y, orig_z), dir_vec, max_distance=rem_reach)

        if hit is None:
            return False

        hit_ent_id, _dist, _norm = hit
        if not self.prop_manager.is_prop(hit_ent_id):
            return False

        prop = self.prop_manager.get_prop(hit_ent_id)
        if prop is None:
            return False

        self.held_entity_id = hit_ent_id
        prop.is_grabbed = True

        # Dampen current momentum
        self.physics.set_linvel(hit_ent_id, (0.0, 0.0, 0.0))
        self.physics.set_angvel(hit_ent_id, (0.0, 0.0, 0.0))

        if self.audio is not None:
            self.audio.play_sound("click", volume=0.4)

        return True

    def release_held(
        self,
        impulse: tuple[float, float, float] | None = None,
    ) -> int | None:
        """Gently drops or releases the currently held prop."""
        if self.held_entity_id is None:
            return None

        ent_id = self.held_entity_id
        prop = self.prop_manager.get_prop(ent_id)
        if prop is not None:
            prop.is_grabbed = False

        if impulse is not None:
            self.physics.apply_impulse(ent_id, impulse)

        self.held_entity_id = None
        return ent_id

    def throw_held(
        self,
        direction: tuple[float, float, float] | np.ndarray,
        speed: float | None = None,
    ) -> int | None:
        """Launches the held prop forward with launch velocity and spin torque."""
        if self.held_entity_id is None:
            return None

        ent_id = self.held_entity_id
        launch_speed = speed if speed is not None else self.throw_force

        dir_vec = np.array(direction, dtype=np.float32)
        dir_len = float(np.linalg.norm(dir_vec))
        if dir_len > 1e-4:
            dir_vec /= dir_len
        else:
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Apply launch velocity
        vx = float(dir_vec[0] * launch_speed)
        vy = float(dir_vec[1] * launch_speed + 0.6)  # Subtle upward arc
        vz = float(dir_vec[2] * launch_speed)
        self.physics.set_linvel(ent_id, (vx, vy, vz))

        # Add natural spin torque
        tx = random.uniform(-1.5, 1.5)
        ty = random.uniform(-1.5, 1.5)
        tz = random.uniform(-1.5, 1.5)
        self.physics.apply_torque_impulse(ent_id, (tx, ty, tz))

        prop = self.prop_manager.get_prop(ent_id)
        if prop is not None:
            prop.is_grabbed = False

        # Play aerodynamic whoosh sound
        if self.audio is not None:
            try:
                t = self.physics.world.get_transform(ent_id)
                self.audio.play_sound("throw", position=(t[0], t[1], t[2]), volume=0.8)
            except Exception:
                self.audio.play_sound("throw", volume=0.8)

        self.held_entity_id = None
        return ent_id

    def toggle_grab(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        ignore_entity_id: int | None = None,
    ) -> bool:
        """Convenience method: releases held prop if holding, or grabs prop if not."""
        if self.held_entity_id is not None:
            self.release_held()
            return False
        return self.try_grab(origin, direction, ignore_entity_id=ignore_entity_id)

    def update(
        self,
        dt: float,
        camera_pos: tuple[float, float, float] | np.ndarray,
        camera_forward: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Tracks the held prop to target carry position using a spring force."""
        if self.held_entity_id is None or dt <= 1e-6:
            return

        # Ensure prop is still valid in manager and physics
        if not self.prop_manager.is_prop(self.held_entity_id):
            self.held_entity_id = None
            return

        try:
            curr_trans = self.physics.world.get_transform(self.held_entity_id)
            curr_px, curr_py, curr_pz = curr_trans[0], curr_trans[1], curr_trans[2]
        except Exception:
            self.held_entity_id = None
            return

        # Compute ideal target position ahead of camera
        fwd_x, fwd_y, fwd_z = float(camera_forward[0]), float(camera_forward[1]), float(camera_forward[2])
        fwd_len = math.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z)
        if fwd_len > 1e-4:
            inv = 1.0 / fwd_len
            fwd_x *= inv
            fwd_y *= inv
            fwd_z *= inv

        target_x = float(camera_pos[0]) + fwd_x * self.hold_distance
        target_y = max(0.4, float(camera_pos[1]) + fwd_y * self.hold_distance)
        target_z = float(camera_pos[2]) + fwd_z * self.hold_distance

        # Critically damped spring velocity calculation
        diff_x = target_x - curr_px
        diff_y = target_y - curr_py
        diff_z = target_z - curr_pz

        k = self.spring_stiffness
        vel_x = diff_x * k
        vel_y = diff_y * k
        vel_z = diff_z * k

        # Clamp max carry velocity
        max_v = 24.0
        v_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z
        if v_sq > max_v * max_v:
            scale = max_v / math.sqrt(v_sq)
            vel_x *= scale
            vel_y *= scale
            vel_z *= scale

        self.physics.set_linvel(self.held_entity_id, (vel_x, vel_y, vel_z))
        # Keep angular velocity damped while carried
        self.physics.set_angvel(self.held_entity_id, (0.0, 0.0, 0.0))
