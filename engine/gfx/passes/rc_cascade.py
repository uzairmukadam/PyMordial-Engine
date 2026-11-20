"""Explicit screen-space Radiance Cascade resource model.

Each cascade owns a probe grid and a set of directional interval textures. The
data model is deliberately separate from the tracing/merging/reconstruction
code so the resource layout can be inspected and validated on its own.

Texture contract (per cascade, RGBA16F texture array):

    XY      probe grid coordinate
    layer   direction index
    RGB     radiance accumulated inside the cascade's distance interval
    A       remaining transmittance through that interval (1 = fully open)

Direction encoding is shared by every cascade: all cascades sample the same
Fibonacci sphere of ``RC_MAX_DIRECTIONS`` directions and use a prefix of it.
A cascade with ``direction_count`` directions therefore uses global direction
indices ``0 .. direction_count-1``, which makes near-to-far direction mapping an
exact index match (a near cascade always has fewer directions than a far one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl

# Global direction budget. Every cascade uses a prefix of this set.
RC_MAX_DIRECTIONS = 256

# Probe spacing in screen pixels. Near cascades are denser.
RC_CASCADE_SPACING = (4, 8, 16)
# Direction counts. The near cascade needs enough angular resolution to resolve
# contact occlusion: a 16-direction set smears crevices and object bases because
# a whole hemisphere quadrant collapses into a single sample. 64 directions
# keeps the near field crisp. The mid cascade shares that count (its interval is
# already covered by the near cascade's angular detail), and the far cascade
# refines to 256 directions for distant detail. Counts must stay in the nested
# 4x hierarchy (16 -> 64 -> 256) so the merge can map a parent direction to its
# four children.
RC_CASCADE_DIRECTIONS = (64, 64, 256)


class RCCascadeConfigError(ValueError):
    """Raised when a cascade configuration cannot be allocated."""


@dataclass
class RCCascade:
    """One screen-space radiance cascade.

    Attributes:
        index: Cascade index (0 = nearest).
        probe_spacing: Probe spacing in screen pixels.
        probe_width: Probe grid width.
        probe_height: Probe grid height.
        direction_count: Number of directional layers.
        interval_start: Near end of this cascade's distance interval (metres).
        interval_end: Far end of this cascade's distance interval (metres).
        interval_texture: RGBA16F texture array of traced intervals.
        merged_texture: RGBA16F texture array holding the merged result.
    """

    index: int
    probe_spacing: int
    probe_width: int
    probe_height: int
    direction_count: int
    interval_start: float
    interval_end: float
    interval_texture: moderngl.TextureArray
    merged_texture: moderngl.TextureArray | None = None

    @property
    def dispatch_groups(self) -> tuple[int, int, int]:
        """Compute dispatch dimensions for one invocation per probe/direction."""
        return (
            (self.probe_width + 7) // 8,
            (self.probe_height + 7) // 8,
            self.direction_count,
        )

    def describe(self) -> dict[str, object]:
        """Returns cascade metadata for debugging and tests."""
        return {
            "index": self.index,
            "probe_spacing": self.probe_spacing,
            "probe_grid": (self.probe_width, self.probe_height),
            "direction_count": self.direction_count,
            "interval": (self.interval_start, self.interval_end),
            "dispatch": self.dispatch_groups,
            "layers": self.interval_texture.layers,
        }

    def release(self) -> None:
        """Releases every GPU resource owned by this cascade."""
        if self.merged_texture is not None:
            self.merged_texture.release()
            self.merged_texture = None
        self.interval_texture.release()


def _validate(index: int, spacing: int, directions: int, start: float, end: float) -> None:
    if spacing < 1:
        raise RCCascadeConfigError(f"cascade {index}: probe spacing must be >= 1 (got {spacing})")
    if directions < 1 or directions > RC_MAX_DIRECTIONS:
        raise RCCascadeConfigError(
            f"cascade {index}: direction count must be in [1, {RC_MAX_DIRECTIONS}] (got {directions})"
        )
    if not math.isfinite(start) or not math.isfinite(end):
        raise RCCascadeConfigError(f"cascade {index}: interval bounds must be finite")
    if start < 0.0:
        raise RCCascadeConfigError(f"cascade {index}: interval start must be >= 0 (got {start})")
    if end <= start:
        raise RCCascadeConfigError(f"cascade {index}: interval end must exceed start (got {start}..{end})")


def build_cascades(
    ctx: moderngl.Context,
    width: int,
    height: int,
    intervals: tuple[tuple[float, float], ...],
    spacings: tuple[int, ...] = RC_CASCADE_SPACING,
    directions: tuple[int, ...] = RC_CASCADE_DIRECTIONS,
) -> list[RCCascade]:
    """Allocates one :class:`RCCascade` per configured interval.

    Args:
        ctx: ModernGL context.
        width: Render target width in pixels.
        height: Render target height in pixels.
        intervals: ``(start, end)`` distance interval per cascade, near first.
        spacings: Probe spacing in pixels per cascade.
        directions: Direction layer count per cascade.

    Raises:
        RCCascadeConfigError: if any dimension or interval is impossible.
    """
    if width < 1 or height < 1:
        raise RCCascadeConfigError(f"invalid render target size {width}x{height}")
    if len(intervals) != len(spacings) or len(intervals) != len(directions):
        raise RCCascadeConfigError(
            "cascade intervals, spacings and direction counts must have equal length "
            f"(got {len(intervals)}, {len(spacings)}, {len(directions)})"
        )

    cascades: list[RCCascade] = []
    for index, ((start, end), spacing, direction_count) in enumerate(zip(intervals, spacings, directions)):
        _validate(index, spacing, direction_count, start, end)

        probe_width = max(1, (width + spacing - 1) // spacing)
        probe_height = max(1, (height + spacing - 1) // spacing)

        interval_tex = ctx.texture_array((probe_width, probe_height, direction_count), 4, dtype="f2")
        interval_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        interval_tex.repeat_x = False
        interval_tex.repeat_y = False

        merged_tex = ctx.texture_array((probe_width, probe_height, direction_count), 4, dtype="f2")
        merged_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        merged_tex.repeat_x = False
        merged_tex.repeat_y = False

        cascades.append(
            RCCascade(
                index=index,
                probe_spacing=spacing,
                probe_width=probe_width,
                probe_height=probe_height,
                direction_count=direction_count,
                interval_start=float(start),
                interval_end=float(end),
                interval_texture=interval_tex,
                merged_texture=merged_tex,
            )
        )
    return cascades


__all__ = [
    "RC_MAX_DIRECTIONS",
    "RC_CASCADE_SPACING",
    "RC_CASCADE_DIRECTIONS",
    "RCCascade",
    "RCCascadeConfigError",
    "build_cascades",
]
