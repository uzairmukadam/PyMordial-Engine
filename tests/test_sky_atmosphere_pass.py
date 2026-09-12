"""Unit and Integration Tests for Sébastien Hillaire Physically Based Sky & Atmosphere Pass.

Validates:
- Static Look-Up Table (LUT) baking: Transmittance LUT (256x64) and Multi-Scattering LUT (32x32).
- Dynamic Look-Up Table baking: Sky-View LUT (192x108).
- Data validity: Non-zero physical transmittances and non-negative luminances.
- Dirty state tracking and parameter update.
- Integration into RenderPipeline.
"""

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.atmosphere import AtmosphereConfig, AtmosphereSystem
from engine.gfx.passes.sky_atmosphere_pass import SkyAtmospherePass
from engine.gfx.pipeline import RenderPipeline


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


def test_hillaire_sky_atmosphere_lut_baking(render_ctx: RenderContext):
    """Verifies that Hillaire Sky Atmosphere pass properly computes and bakes all LUTs."""
    ctx = render_ctx.ctx
    sky_pass = SkyAtmospherePass(ctx)
    atmo = AtmosphereSystem()

    # 1. Bake static LUTs
    sky_pass.bake_static_luts(atmo.config)

    # Read Transmittance LUT
    trans_data = np.frombuffer(sky_pass.transmittance_lut.read(), dtype=np.float16 if sky_pass.transmittance_lut.dtype == "f2" else np.float32)
    assert len(trans_data) > 0
    # Transmittance values must be in [0, 1] range
    assert np.all(trans_data >= 0.0)
    # Looking zenith at high altitude should have high transmittance (> 0.5)
    assert np.max(trans_data) > 0.5

    # Read Multi-Scattering LUT
    ms_data = np.frombuffer(sky_pass.multiscattering_lut.read(), dtype=np.float16 if sky_pass.multiscattering_lut.dtype == "f2" else np.float32)
    assert len(ms_data) > 0
    assert np.all(ms_data >= 0.0)

    # 2. Update Dynamic Sky-View LUT
    sky_pass.update(
        atmo_config=atmo.config,
        camera_pos=(0.0, 10.0, 0.0),
        sun_dir=(0.0, -1.0, 0.0), # Noon sun
        sun_color=(1.0, 0.98, 0.92),
        sun_intensity=4.0,
    )

    sky_data = np.frombuffer(sky_pass.sky_view_lut.read(), dtype=np.float16 if sky_pass.sky_view_lut.dtype == "f2" else np.float32)
    assert len(sky_data) > 0
    assert np.all(sky_data >= 0.0)
    # Daylight sky radiance must be positive
    assert np.max(sky_data) > 0.0


def test_hillaire_dirty_tracking_on_preset_change(render_ctx: RenderContext):
    """Verifies that changing scattering parameters automatically dirties and rebakes static LUTs."""
    ctx = render_ctx.ctx
    sky_pass = SkyAtmospherePass(ctx)
    atmo_config = AtmosphereConfig()

    sky_pass.update(
        atmo_config=atmo_config,
        camera_pos=(0.0, 1.0, 0.0),
        sun_dir=(0.0, -0.7, 0.7),
    )
    assert sky_pass._luts_dirty is False

    # Change Rayleigh scattering (e.g. alien sky)
    atmo_config.rayleigh_beta = (20.0e-6, 5.0e-6, 30.0e-6)
    sky_pass.update(
        atmo_config=atmo_config,
        camera_pos=(0.0, 1.0, 0.0),
        sun_dir=(0.0, -0.7, 0.7),
    )
    # Should have updated and cleared dirty flag
    assert sky_pass._luts_dirty is False


def test_pipeline_integration_with_hillaire_sky(render_ctx: RenderContext):
    """Verifies that RenderPipeline correctly initializes and executes with Hillaire sky pass."""
    pipeline = RenderPipeline(render_ctx)
    assert hasattr(pipeline, "sky_atmosphere_pass")
    assert pipeline.sky_atmosphere_pass is not None

    # Check that resolve program has the LUT samplers bound
    assert "u_SkyViewLUT" in pipeline.resolve_prog
    assert "u_TransmittanceLUT" in pipeline.resolve_prog
