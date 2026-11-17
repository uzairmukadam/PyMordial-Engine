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

from engine.core.ecs import EntityManager
from engine.gfx.atmosphere import AtmosphereConfig, AtmosphereSystem
from engine.gfx.context import RenderContext
from engine.gfx.passes.sky_atmosphere_pass import SkyAtmospherePass
from engine.gfx.passes.sky_cloud_pass import SkyCloudPass
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
    trans_data = np.frombuffer(
        sky_pass.transmittance_lut.read(), dtype=np.float16 if sky_pass.transmittance_lut.dtype == "f2" else np.float32
    )
    assert len(trans_data) > 0
    # Transmittance values must be in [0, 1] range
    assert np.all(trans_data >= 0.0)
    # Looking zenith at high altitude should have high transmittance (> 0.5)
    assert np.max(trans_data) > 0.5

    # Read Multi-Scattering LUT
    ms_data = np.frombuffer(
        sky_pass.multiscattering_lut.read(),
        dtype=np.float16 if sky_pass.multiscattering_lut.dtype == "f2" else np.float32,
    )
    assert len(ms_data) > 0
    assert np.all(ms_data >= 0.0)

    # 2. Update Dynamic Sky-View LUT
    sky_pass.update(
        atmo_config=atmo.config,
        camera_pos=(0.0, 10.0, 0.0),
        sun_dir=(0.0, -1.0, 0.0),  # Noon sun
        sun_color=(1.0, 0.98, 0.92),
        sun_intensity=4.0,
    )

    sky_data = np.frombuffer(
        sky_pass.sky_view_lut.read(), dtype=np.float16 if sky_pass.sky_view_lut.dtype == "f2" else np.float32
    )
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
    old_signature = sky_pass._prev_params["sig"]
    atmo_config.ozone_width = 22000.0
    sky_pass.update(
        atmo_config=atmo_config,
        camera_pos=(0.0, 1.0, 0.0),
        sun_dir=(0.0, -0.7, 0.7),
    )
    assert sky_pass._prev_params["sig"] != old_signature


def test_procedural_cloud_pass_composites_only_sky_pixels(render_ctx: RenderContext):
    ctx = render_ctx.ctx
    cloud_pass = SkyCloudPass(ctx, noise_size=64)
    depth_texture = ctx.texture((320, 240), 1, dtype="f4")
    depth = np.zeros((240, 320), dtype=np.float32)
    depth[:120, :] = 1.0
    depth_texture.write(depth.tobytes())
    hdr_texture = ctx.texture((320, 240), 4, dtype="f2")
    hdr_fbo = ctx.framebuffer(color_attachments=[hdr_texture])
    try:
        hdr_fbo.use()
        ctx.viewport = (0, 0, 320, 240)
        hdr_fbo.clear(0.0, 0.0, 0.0, 1.0)
        cloud_pass.render(
            hdr_fbo=hdr_fbo,
            depth_texture=depth_texture,
            inverse_projection=np.identity(4, dtype=np.float32).flatten(),
            inverse_view=np.identity(4, dtype=np.float32).flatten(),
            camera_position=(0.0, 0.0, 0.0),
            sun_direction=(0.0, -1.0, 0.0),
            sun_color=(1.0, 0.98, 0.92),
            moon_color=(0.70, 0.82, 1.0),
            moon_intensity=0.4,
            time_elapsed=0.5,
            cloud_config=AtmosphereConfig(cloud_coverage=1.0, cloud_opacity=1.0),
        )
        pixels = np.frombuffer(hdr_texture.read(), dtype=np.float16).reshape(240, 320, 4)
        assert np.any(pixels[depth == 0.0, :3] > 0.0)
        assert np.all(pixels[depth > 0.0, :3] == 0.0)

        noise = np.frombuffer(cloud_pass.noise_texture.read(), dtype=np.float16).reshape(
            cloud_pass.noise_size, cloud_pass.noise_size, 4
        )
        assert all(float(np.std(noise[:, :, channel])) > 0.03 for channel in range(4))
        assert float(np.mean(np.abs(noise[0, :, 0] - noise[-1, :, 0]))) < 0.10
    finally:
        hdr_fbo.release()
        hdr_texture.release()
        depth_texture.release()
        cloud_pass.destroy()


def test_pipeline_integration_with_hillaire_sky(render_ctx: RenderContext):
    """Verifies that RenderPipeline correctly initializes and executes with Hillaire sky pass."""
    pipeline = RenderPipeline(render_ctx)
    try:
        # The sky capability is provided by the atmosphere engine module and
        # exposed through the pipeline's environment provider.
        assert pipeline.environment is not None
        assert pipeline.environment.enabled is True
        assert pipeline.environment.sky_pass is not None
        assert pipeline.environment.cloud_pass.noise_texture is not None

        # Check that resolve program has the LUT samplers bound
        assert "u_SkyViewLUT" in pipeline.resolve_prog
        assert "u_TransmittanceLUT" in pipeline.resolve_prog

        pipeline.environment.apply_preset("SPACE")
        ecs = EntityManager(max_entities=8)
        ecs.create_entity(position=(1000.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        pipeline.render_frame(
            ecs=ecs,
            camera_pos=(0.0, 0.0, 0.0),
            camera_target=(0.0, 0.0, -1.0),
            time_elapsed=0.25,
        )
        assert pipeline._u_sky_mode.value == 1
    finally:
        pipeline.destroy()
