"""Tests for the P0 energy-conserving BRDF (Pass 3).

Covers:

* The BRDF LUT now carries the directional albedo in its blue channel.
* ``GGXEnergyCompensation`` restores the energy single-scatter GGX drops, and
  the restored lobe integrates to ~1.0 across the roughness range.
* Specular anti-aliasing widens roughness in proportion to normal variance.
* The resolve honours ``energy_compensation_enabled``.
"""

import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.passes.ibl import IBLPass
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.quality_presets import RenderConfig

# ---------------------------------------------------------------------------
#  CPU mirrors of shaders/common/pbr_brdf.glsl
# ---------------------------------------------------------------------------


def ggx_average_albedo(roughness: float) -> float:
    r = min(max(roughness, 0.0), 1.0)
    return 1.0 - 0.28 * r * r


def ggx_energy_compensation(f0: float, roughness: float, e: float) -> float:
    e_avg = ggx_average_albedo(roughness)
    e = min(max(e, 0.0), 1.0)
    denom = max(1.0 - e_avg * (1.0 - e), 1e-3)
    f_ms = f0 * (e_avg / denom)
    scale = 1.0 + f_ms * (1.0 / max(e, 1e-3) - 1.0)
    return min(max(scale, 1.0), 4.0)


def specular_aa(roughness: float, variance: float, strength: float) -> float:
    v = max(variance, 0.0) * max(strength, 0.0)
    return min(max(np.sqrt(roughness * roughness + v), 0.0), 1.0)


class TestEnergyCompensationMath:
    def test_compensation_is_never_below_one(self):
        """The correction may only add energy, never remove it."""
        for roughness in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            for e in (0.5, 0.7, 0.9, 0.99):
                assert ggx_energy_compensation(1.0, roughness, e) >= 1.0

    def test_compensation_grows_with_roughness(self):
        """Rough surfaces lose more energy, so they need a larger correction."""
        smooth = ggx_energy_compensation(1.0, 0.1, 0.98)
        rough = ggx_energy_compensation(1.0, 1.0, 0.72)
        assert rough > smooth

    def test_compensation_is_bounded(self):
        """A degenerate albedo must not produce an unbounded multiplier."""
        for e in (1e-6, 0.01, 0.1):
            value = ggx_energy_compensation(1.0, 1.0, e)
            assert np.isfinite(value)
            assert value <= 4.0

    def test_restores_energy_to_unity(self):
        """E * compensation must land near 1.0 for a white metal."""
        for roughness, e in ((0.1, 0.98), (0.5, 0.88), (0.9, 0.74), (1.0, 0.72)):
            restored = e * ggx_energy_compensation(1.0, roughness, e)
            assert 0.95 <= restored <= 1.10, (roughness, e, restored)

    def test_dielectric_gets_a_smaller_correction(self):
        """F0 = 0.04 means far less energy is at stake than for a metal."""
        metal = ggx_energy_compensation(1.0, 0.9, 0.74)
        dielectric = ggx_energy_compensation(0.04, 0.9, 0.74)
        assert dielectric < metal


class TestSpecularAA:
    def test_zero_variance_is_a_no_op(self):
        assert abs(specular_aa(0.3, 0.0, 1.0) - 0.3) < 1e-6

    def test_variance_widens_roughness(self):
        assert specular_aa(0.3, 0.05, 1.0) > 0.3

    def test_result_is_clamped(self):
        assert specular_aa(0.9, 100.0, 1.0) <= 1.0

    def test_strength_scales_the_widening(self):
        weak = specular_aa(0.3, 0.05, 0.25)
        strong = specular_aa(0.3, 0.05, 1.0)
        assert strong > weak


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=160, height=120)
    yield ctx
    ctx.destroy()


class TestBRDFLUT:
    def test_lut_is_rgba16f(self, render_ctx):
        ibl = IBLPass(render_ctx.ctx)
        try:
            assert ibl.lut_tex.size == (256, 256)
            assert ibl.lut_tex.components == 4
        finally:
            ibl.destroy()

    def test_blue_channel_carries_directional_albedo(self, render_ctx):
        """E must be in (0, 1] and fall as roughness rises."""
        ibl = IBLPass(render_ctx.ctx)
        try:
            data = np.frombuffer(ibl.lut_tex.read(), dtype=np.float16).reshape(256, 256, 4)
            energy = data[..., 2].astype(np.float32)
            assert energy.min() > 0.0
            assert energy.max() <= 1.001

            # v = roughness. Compare a smooth row against a rough row.
            smooth = float(energy[10, :].mean())
            rough = float(energy[250, :].mean())
            assert rough < smooth, (smooth, rough)
        finally:
            ibl.destroy()


def _render_rough_metal(ctx, energy_comp: bool) -> np.ndarray:
    cfg = RenderConfig(
        energy_compensation_enabled=energy_comp,
        specular_aa_enabled=False,
        bloom_enabled=False,
        lens_flare_enabled=False,
        chromatic_aberration_enabled=False,
        vignette_enabled=False,
        film_grain_enabled=False,
    )
    pipeline = RenderPipeline(ctx, config=cfg)
    try:
        ecs = EntityManager(max_entities=10)
        ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.9, 0.9, 0.9),
            roughness=0.95,
            metallic=1.0,
        )
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0),
            time_elapsed=0.5,
            sun_dir=(0.5, -0.5, 0.5),
            sun_lux=4.0,
        )
        return np.frombuffer(pipeline.post_process.hdr_texture.read(), dtype=np.float16).reshape(120, 160, 4)
    finally:
        pipeline.destroy()


class TestResolveIntegration:
    def test_energy_compensation_brightens_rough_metal(self, render_ctx):
        without = _render_rough_metal(render_ctx, energy_comp=False)
        with_comp = _render_rough_metal(render_ctx, energy_comp=True)
        assert with_comp[..., :3].mean() > without[..., :3].mean(), (
            "energy compensation must add energy to a rough metal"
        )

    def test_energy_compensation_is_not_explosive(self, render_ctx):
        """The correction must stay physically plausible, not blow out."""
        with_comp = _render_rough_metal(render_ctx, energy_comp=True)
        assert np.isfinite(with_comp).all()
        assert with_comp[..., :3].max() < 50.0

    def test_config_defaults(self):
        cfg = RenderConfig()
        assert cfg.energy_compensation_enabled is True
        assert cfg.specular_aa_enabled is True
        assert cfg.specular_aa_strength > 0.0
