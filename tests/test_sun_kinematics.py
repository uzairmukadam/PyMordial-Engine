"""Solar kinematics continuity tests.

Guards against the azimuth discontinuity that made the sun (and therefore the
shadow direction) snap by 90 degrees at solar noon. The sun must trace a
continuous arc across the sky: no adjacent pair of samples may differ by more
than a small angular step, and the azimuth must advance monotonically through
the morning and afternoon halves of the day.
"""

from __future__ import annotations

import math

import pytest

from engine.modules.atmosphere.config import AtmosphereConfig
from engine.modules.atmosphere.system import AtmosphereSystem

# One sample per minute over a full 24 hour cycle.
SAMPLE_STEP_HOURS = 1.0 / 60.0
SAMPLE_COUNT = int(24.0 / SAMPLE_STEP_HOURS)

# A continuous arc moves well under this per minute. The historical bug produced
# a 90 degree jump, so any threshold below that catches it while tolerating
# legitimate near-horizon acceleration.
MAX_STEP_DEGREES = 2.0


def _sun_sky_vector(system: AtmosphereSystem) -> tuple[float, float, float]:
    """Returns the unit vector pointing *towards* the sun (negated light dir)."""
    lx, ly, lz = system.compute_sun_vector()
    return (-lx, -ly, -lz)


def _angle_between(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _azimuth_deg(v: tuple[float, float, float]) -> float:
    """Compass azimuth of a sky vector: 0 = +Z, 90 = +X, measured clockwise."""
    return math.degrees(math.atan2(v[0], v[2])) % 360.0


class TestSunContinuity:
    """The sun must move continuously, never teleport."""

    @pytest.mark.parametrize("latitude", [0.0, 35.0, 60.0])
    def test_no_azimuth_discontinuity_over_full_day(self, latitude: float) -> None:
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=0.0, latitude_deg=latitude, azimuth_deg=0.0))
        previous = _sun_sky_vector(system)
        worst_step = 0.0
        worst_time = 0.0

        for index in range(1, SAMPLE_COUNT + 1):
            system.config.time_of_day = (index * SAMPLE_STEP_HOURS) % 24.0
            current = _sun_sky_vector(system)
            step = _angle_between(previous, current)
            if step > worst_step:
                worst_step = step
                worst_time = system.config.time_of_day
            previous = current

        assert worst_step <= MAX_STEP_DEGREES, (
            f"Sun jumped {worst_step:.2f} deg in one minute at t={worst_time:.2f}h "
            f"(latitude {latitude}). The solar arc must be continuous."
        )

    def test_azimuth_advances_monotonically_through_morning(self) -> None:
        """From sunrise to noon the sun must sweep steadily in one direction."""
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=6.0, latitude_deg=35.0, azimuth_deg=0.0))
        azimuths: list[float] = []
        for index in range(0, 361):
            system.config.time_of_day = 6.0 + index * SAMPLE_STEP_HOURS
            azimuths.append(_azimuth_deg(_sun_sky_vector(system)))

        # Unwrap so a 359 -> 1 degree crossing is not read as a reversal.
        unwrapped = [azimuths[0]]
        for value in azimuths[1:]:
            delta = value - unwrapped[-1]
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
            unwrapped.append(unwrapped[-1] + delta)

        total_sweep = unwrapped[-1] - unwrapped[0]
        assert abs(total_sweep) > 30.0, (
            f"Sun azimuth barely moved across six hours ({total_sweep:.2f} deg). "
            "The azimuth derivation is not tracking the hour angle."
        )

        # Every step must share the sign of the total sweep (no reversals).
        direction = 1.0 if total_sweep > 0.0 else -1.0
        for index in range(1, len(unwrapped)):
            step = unwrapped[index] - unwrapped[index - 1]
            assert step * direction >= -1e-6, f"Azimuth reversed at sample {index} (step {step:.4f} deg)."

    def test_elevation_peaks_near_solar_noon(self) -> None:
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, latitude_deg=35.0, azimuth_deg=0.0))
        noon_elevation = _sun_sky_vector(system)[1]
        for hour in (6.0, 9.0, 15.0, 18.0):
            system.config.time_of_day = hour
            assert _sun_sky_vector(system)[1] < noon_elevation, (
                f"Sun elevation at {hour:.0f}h should be below its noon peak."
            )

    def test_sun_vector_is_unit_length(self) -> None:
        system = AtmosphereSystem(AtmosphereConfig(latitude_deg=35.0))
        for hour in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0):
            system.config.time_of_day = hour
            x, y, z = system.compute_sun_vector()
            length = math.sqrt(x * x + y * y + z * z)
            assert length == pytest.approx(1.0, abs=1e-6), f"Sun vector at {hour:.0f}h has length {length:.6f}."

    def test_noon_sun_is_overhead_at_equator(self) -> None:
        """At the equator on the equinox the noon sun is essentially at zenith."""
        system = AtmosphereSystem(AtmosphereConfig(time_of_day=12.0, latitude_deg=0.0, azimuth_deg=0.0))
        sky = _sun_sky_vector(system)
        assert sky[1] == pytest.approx(1.0, abs=1e-3), f"Equatorial noon sun should be overhead, got y={sky[1]:.4f}."
