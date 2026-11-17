"""Shadow filter physical-correctness tests.

The PCSS penumbra must be derived from the sun's real angular radius, not an
arbitrary constant. An oversized spread turns every caster into a soft blob and
makes distant shadows detach from their receivers.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
RESOLVE_SHADER = SHADER_DIR / "deferred_resolve.frag"

# Physical solar angular radius: 0.266 degrees.
SUN_ANGULAR_RADIUS_RAD = math.radians(0.266)

# The penumbra spread constant must stay within a small factor of the physical
# value. The historical bug used 0.045 rad, roughly 10x too wide.
MAX_SPREAD_RATIO = 2.0


def _read_resolve_shader() -> str:
    return RESOLVE_SHADER.read_text(encoding="utf-8")


class TestPenumbraPhysics:
    """Penumbra width must track the real solar angular size."""

    def test_shader_declares_sun_angular_radius_uniform(self) -> None:
        source = _read_resolve_shader()
        assert "u_SunAngularRadius" in source, (
            "deferred_resolve.frag must expose a u_SunAngularRadius uniform so the "
            "penumbra is driven by the physical solar size."
        )

    def test_penumbra_spread_matches_physical_sun_size(self) -> None:
        source = _read_resolve_shader()
        match = re.search(r"float\s+sun_spread\s*=\s*([^;]+);", source)
        assert match is not None, "Could not locate the sun_spread definition in the PCSS path."

        expression = match.group(1)
        # The spread must be derived from the uniform, not a bare literal.
        assert "u_SunAngularRadius" in expression, (
            f"sun_spread is '{expression.strip()}', which ignores the physical solar "
            "angular radius. Drive it from u_SunAngularRadius."
        )

        # Any literal multiplier applied to the uniform must stay near 1.0.
        literals = [
            float(value)
            for value in re.findall(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])", expression)
        ]
        for literal in literals:
            assert literal <= MAX_SPREAD_RATIO, (
                f"sun_spread multiplies the solar radius by {literal}, which is far "
                f"wider than the physical {SUN_ANGULAR_RADIUS_RAD:.5f} rad."
            )

    def test_blocker_search_radius_is_bounded_in_world_space(self) -> None:
        """A fixed atlas-UV clamp makes distant cascades collapse into blobs."""
        source = _read_resolve_shader()
        match = re.search(r"float\s+search_radius\s*=\s*clamp\((.*?)\);", source, re.DOTALL)
        assert match is not None, "Could not locate the PCSS blocker search radius."

        body = match.group(1)
        assert "world_texel_size" in body or "inv_rad" in body, (
            "The blocker search radius must be expressed in world units so it scales "
            "with cascade texel density instead of clamping to a fixed atlas fraction."
        )

    def test_filter_sample_count_scales_with_radius(self) -> None:
        """A fixed tap count over a large radius under-samples and bands."""
        source = _read_resolve_shader()
        assert "filter_samples" in source
        match = re.search(r"int\s+filter_samples\s*=\s*([^;]+);", source)
        assert match is not None, "Could not locate the PCSS filter sample count."

        expression = match.group(1)
        assert "filter_radius" in expression or "radius" in expression, (
            f"filter_samples is '{expression.strip()}'. It must scale with the filter "
            "radius so taps-per-texel stays high enough to avoid banding."
        )

    def test_cascade_texel_size_is_not_doubled(self) -> None:
        """One cascade texel must be 1/full_size in atlas UV, not 2/full_size.

        The cascade occupies ``atlas_scale`` of the atlas, so a single texel is
        ``atlas_scale / (full_size * atlas_scale) == 1 / full_size``. Using
        ``2.0 / full_size`` doubled every filter radius, which is what made PCSS
        look blurry and banded.
        """
        source = _read_resolve_shader()
        match = re.search(r"vec2\s+cascade_texel_size\s*=\s*([^;]+);", source)
        assert match is not None, "Could not locate cascade_texel_size."

        expression = match.group(1)
        assert "2.0 / full_size" not in expression, (
            f"cascade_texel_size is '{expression.strip()}', which is 2x too large. "
            "Use 1.0 / full_size."
        )
        assert "1.0 / full_size" in expression, (
            f"cascade_texel_size should be 1.0 / full_size, got '{expression.strip()}'."
        )

    def test_vogel_disk_is_rotated_per_pixel(self) -> None:
        """A fixed spiral orientation produces visible concentric banding."""
        source = _read_resolve_shader()
        calls = re.findall(r"VogelDiskSample\(([^;]*?)\)", source)
        assert calls, "No VogelDiskSample calls found."

        rotated = [
            call
            for call in calls
            if "InterleavedGradientNoise" in call or "IGN" in call or "phi" in call
        ]
        assert rotated, (
            "Every VogelDiskSample call passes a constant phase, so the sampling "
            "pattern is identical per pixel and bands. Rotate it with per-pixel noise."
        )


class TestCascadeSplitQuality:
    """Adjacent cascades must not differ wildly in texel density."""

    def test_split_texel_density_ratio_is_bounded(self) -> None:
        from engine.gfx.shadow_csm import CascadedShadowMap

        csm = CascadedShadowMap.__new__(CascadedShadowMap)
        csm.cascade_count = 4
        csm.max_distance = 500.0
        csm.split_lambda = 0.85
        csm.update_splits(500.0, cascade_count=4, lambda_val=0.85)

        splits = csm.split_distances
        # World-space span of each cascade slice.
        spans: list[float] = []
        previous = 0.1
        for split in splits:
            spans.append(split - previous)
            previous = split

        # Texel density is inversely proportional to slice span for a fixed map
        # size, so a large span ratio between neighbours is a visible quality step.
        ratios = [
            spans[index + 1] / max(spans[index], 1e-6)
            for index in range(len(spans) - 1)
        ]
        worst = max(ratios)
        assert worst <= 2.5, (
            f"Adjacent cascade span ratio reaches {worst:.2f}x, so texel density drops "
            f"in visible steps. Spans: {[round(s, 1) for s in spans]}. "
            "Equalize the geometric spacing."
        )
