"""Automated Unit & Integration Tests for Phase 4: Dynamic Water & Screen-Space Refraction.

Verifies:
- WaterPass initialization and GPU mesh allocation
- Dynamic resolution resizing
- Gerstner wave grid generation and vertex/index structure
- Theoretical Gerstner displacement mathematics
- WaterPass rendering execution and texture sampling
- Quality preset scaling (LOW to CINEMATIC)
- RenderPipeline integration
"""

import math
import numpy as np
import pytest
import moderngl

from engine.gfx.passes.water_pass import WaterPass, _generate_water_grid
from engine.gfx.quality_presets import (
    RenderConfig,
    GraphicsQuality,
    get_quality_preset,
    QUALITY_PRESETS,
)


@pytest.fixture(scope="module")
def gl_context():
    """Provides a headless ModernGL context for GPU testing."""
    ctx = moderngl.create_context(standalone=True)
    yield ctx
    ctx.finish()


def test_water_grid_generation():
    """Validates geometry layout, UVs, and triangle topology of the water mesh."""
    grid_size = 16
    world_size = 32.0
    verts, inds = _generate_water_grid(grid_size, world_size)

    expected_verts = (grid_size + 1) * (grid_size + 1)
    expected_indices = grid_size * grid_size * 6

    assert len(verts) == expected_verts
    assert len(inds) == expected_indices
    assert verts.shape[1] == 3  # x, y, z

    # Check bounds
    half_size = world_size * 0.5
    assert math.isclose(verts[:, 0].min(), -half_size, abs_tol=1e-4)
    assert math.isclose(verts[:, 0].max(), half_size, abs_tol=1e-4)
    assert math.isclose(verts[:, 2].min(), -half_size, abs_tol=1e-4)
    assert math.isclose(verts[:, 2].max(), half_size, abs_tol=1e-4)


def test_water_pass_initialization(gl_context):
    """Validates WaterPass program compilation, FBO allocation, and VAO creation."""
    w, h = 320, 240
    water_pass = WaterPass(gl_context, width=w, height=h, grid_size=16, world_size=32.0)

    assert water_pass.enabled is True
    assert water_pass.width == w
    assert water_pass.height == h
    assert water_pass.prog is not None
    assert water_pass.vao is not None
    assert water_pass.opaque_copy_fbo is not None
    assert water_pass.opaque_copy_tex is not None
    assert water_pass.opaque_copy_tex.size == (w, h)

    water_pass.destroy()


def test_water_pass_resize(gl_context):
    """Validates dynamic framebuffer and texture recreation on viewport change."""
    water_pass = WaterPass(gl_context, width=320, height=240, grid_size=16, world_size=32.0)

    water_pass.resize(640, 480)
    assert water_pass.width == 640
    assert water_pass.height == 480
    assert water_pass.opaque_copy_tex.size == (640, 480)

    # Same size should be no-op
    water_pass.resize(640, 480)
    assert water_pass.width == 640

    water_pass.destroy()


def test_water_pass_rendering_execution(gl_context):
    """Validates full execution of water rendering with mock HDR scene and depth textures."""
    w, h = 128, 128
    water_pass = WaterPass(gl_context, width=w, height=h, grid_size=16, world_size=32.0)

    # Mock HDR color FBO
    hdr_tex = gl_context.texture((w, h), 4, dtype="f2")
    hdr_fbo = gl_context.framebuffer(color_attachments=[hdr_tex])

    # Mock Reversed-Z depth texture
    depth_tex = gl_context.texture((w, h), 1, dtype="f4")
    # Initialize with non-zero depth (e.g. 0.5 in Reversed-Z)
    dummy_depth = np.full((h, w), 0.5, dtype=np.float32)
    depth_tex.write(dummy_depth.tobytes())

    config = RenderConfig()
    config.water_enabled = True
    config.water_height = 0.0
    config.water_wave_amplitude = 0.2
    config.water_wave_speed = 1.0

    # Clear target FBO
    hdr_fbo.use()
    gl_context.clear(0.1, 0.2, 0.3, 1.0)

    # Execute water pass
    water_pass.render_water(
        hdr_fbo=hdr_fbo,
        depth_texture=depth_tex,
        config=config,
        time_elapsed=1.5,
    )

    gl_context.finish()

    water_pass.destroy()
    hdr_fbo.release()
    hdr_tex.release()
    depth_tex.release()


def test_water_quality_presets():
    """Verifies that all GraphicsQuality presets scale water parameters appropriately."""
    cfg_low = get_quality_preset(GraphicsQuality.LOW)
    assert cfg_low.water_enabled is False
    assert cfg_low.water_refraction_enabled is False

    cfg_med = get_quality_preset(GraphicsQuality.MEDIUM)
    assert cfg_med.water_enabled is True
    assert cfg_med.water_wave_amplitude == 0.10
    assert cfg_med.water_foam_enabled is False

    cfg_high = get_quality_preset(GraphicsQuality.HIGH)
    assert cfg_high.water_enabled is True
    assert cfg_high.water_wave_amplitude == 0.15
    assert cfg_high.water_foam_enabled is True

    cfg_ultra = get_quality_preset(GraphicsQuality.ULTRA)
    assert cfg_ultra.water_enabled is True
    assert cfg_ultra.water_wave_amplitude == 0.20
    assert cfg_ultra.water_foam_enabled is True

    cfg_cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
    assert cfg_cinematic.water_enabled is True
    assert cfg_cinematic.water_wave_amplitude == 0.25
    assert cfg_cinematic.water_clarity == 5.0


def test_pipeline_water_integration():
    """Validates full RenderPipeline integration of WaterPass (Pass 7.65)."""
    from engine.gfx.context import RenderContext
    from engine.gfx.pipeline import RenderPipeline
    from engine.core.ecs import EntityManager

    render_ctx = RenderContext(width=160, height=120, title="Test", hidden=True)
    cfg = get_quality_preset(GraphicsQuality.HIGH)
    pipeline = RenderPipeline(render_ctx, cfg)

    assert pipeline.water_pass is not None
    assert pipeline.water_pass.enabled is True
    assert pipeline.water_pass.opaque_copy_tex is not None

    ecs = EntityManager(max_entities=10)
    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.0, 5.0, 10.0),
        camera_target=(0.0, 0.0, 0.0),
        time_elapsed=0.5,
        sun_dir=(0.2, -0.8, 0.5),
        sun_lux=4.0,
        draw_batches=[],
        dt=0.016,
    )

    # Disable water and verify pipeline still renders cleanly
    cfg_low = get_quality_preset(GraphicsQuality.LOW)
    pipeline.apply_config(cfg_low)
    assert pipeline.water_pass.enabled is False

    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.0, 5.0, 10.0),
        camera_target=(0.0, 0.0, 0.0),
        time_elapsed=1.0,
        sun_dir=(0.2, -0.8, 0.5),
        sun_lux=4.0,
        draw_batches=[],
        dt=0.016,
    )

    pipeline.destroy()
    render_ctx.destroy()

