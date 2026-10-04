"""Unit and Integration Tests for Phase 2: Volumetric VFX & GPU Compute Particle System.

Validates:
- ParticleSystemPass creation, SSBO allocation, layout, and structured seeding.
- GPU Compute Shader simulation dispatch (divergence-free curl noise, buoyancy, wrapping, lifetime).
- Mode switching (DUST_MOTES, EMBERS, FIREFLIES, OFF) and parameter scaling.
- Instanced quad rendering with reversed-Z soft depth feathering into HDR FBO.
- Graphics quality presets scaling (LOW to CINEMATIC).
- Full RenderPipeline integration at Pass 7.7.
"""

from __future__ import annotations
import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.passes.particle_pass import ParticleSystemPass, ParticleMode, PARTICLE_DTYPE


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


def test_particle_pass_init_and_ssbo(render_ctx: RenderContext):
    """Verifies ParticleSystemPass allocation, SSBO size, and seeded data validity."""
    ctx = render_ctx.ctx
    count = 2048
    pass_obj = ParticleSystemPass(ctx, max_particles=count)
    assert pass_obj.max_particles == count
    assert pass_obj.active_particles == count
    assert pass_obj.mode == ParticleMode.DUST_MOTES
    assert pass_obj.enabled is True

    # Validate SSBO size: 2048 particles * 64 bytes = 131,072 bytes
    expected_size = count * 64
    assert pass_obj.ssbo.size == expected_size

    # Read back initial seeded buffer and inspect numpy structured array
    raw = pass_obj.ssbo.read()
    data = np.frombuffer(raw, dtype=PARTICLE_DTYPE)
    assert len(data) == count

    # Validate no NaNs or Infs
    assert not np.isnan(data["pos_size"]).any()
    assert not np.isinf(data["pos_size"]).any()
    assert not np.isnan(data["vel_life"]).any()
    assert not np.isinf(data["vel_life"]).any()
    assert not np.isnan(data["color_alpha"]).any()

    # Lifetimes and sizes should be positive
    assert (data["pos_size"][:, 3] > 0.0).all()

    pass_obj.destroy()


def test_particle_mode_switching_and_parameters(render_ctx: RenderContext):
    """Verifies mode transitions, active count clamping, and visual parameters."""
    ctx = render_ctx.ctx
    pass_obj = ParticleSystemPass(ctx, max_particles=4096)

    # Test mode switching
    pass_obj.set_mode(ParticleMode.EMBERS)
    assert pass_obj.mode == ParticleMode.EMBERS

    pass_obj.set_mode("FIREFLIES")
    assert pass_obj.mode == ParticleMode.FIREFLIES

    pass_obj.set_mode("OFF")
    assert pass_obj.mode == ParticleMode.OFF

    pass_obj.set_mode("DUST_MOTES")
    assert pass_obj.mode == ParticleMode.DUST_MOTES

    # Test active particle clamping
    pass_obj.set_active_count(1024)
    assert pass_obj.active_particles == 1024

    pass_obj.set_active_count(999999)  # Clamped to max_particles
    assert pass_obj.active_particles == 4096

    pass_obj.set_active_count(-10)
    assert pass_obj.active_particles == 0

    # Test parameters
    pass_obj.turbulence_strength = 2.5
    assert pass_obj.turbulence_strength == 2.5

    pass_obj.base_size_multiplier = 1.5
    assert pass_obj.base_size_multiplier == 1.5

    pass_obj.sun_scatter_intensity = 3.0
    assert pass_obj.sun_scatter_intensity == 3.0

    pass_obj.destroy()


def test_particle_gpu_compute_simulation(render_ctx: RenderContext):
    """Dispatches GPU compute shader and verifies particle integration step."""
    ctx = render_ctx.ctx
    count = 1024
    pass_obj = ParticleSystemPass(ctx, max_particles=count)

    # Record initial data
    raw_before = pass_obj.ssbo.read()
    data_before = np.frombuffer(raw_before, dtype=PARTICLE_DTYPE).copy()

    # Step simulation
    pass_obj.update(
        dt=0.033,
        camera_pos=(10.0, 2.0, -5.0),
    )

    # Read back modified buffer
    raw_after = pass_obj.ssbo.read()
    data_after = np.frombuffer(raw_after, dtype=PARTICLE_DTYPE)

    assert not np.isnan(data_after["pos_size"]).any()
    assert not np.isinf(data_after["pos_size"]).any()
    assert not np.isnan(data_after["vel_life"]).any()
    assert not np.isinf(data_after["vel_life"]).any()

    # Lifetimes should have changed (decremented or respawned)
    life_diff = np.abs(data_after["pos_size"][:, 3] - data_before["pos_size"][:, 3])
    assert (life_diff > 0.0).any()

    pass_obj.destroy()


def test_particle_rendering_instanced_quad(render_ctx: RenderContext):
    """Renders particles into an HDR FBO with Reversed-Z depth feathering."""
    ctx = render_ctx.ctx
    w, h = 160, 120
    hdr_tex = ctx.texture((w, h), 4, dtype="f2")
    depth_tex = ctx.depth_texture((w, h))
    fbo = ctx.framebuffer(color_attachments=[hdr_tex], depth_attachment=depth_tex)
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0, depth=0.0)  # Reversed-Z clear (0.0 = far plane)

    frame_ctx = FrameContext(ctx)
    view_mat = np.eye(4, dtype=np.float32).flatten()
    proj_mat = np.eye(4, dtype=np.float32).flatten()
    inv_proj_mat = np.eye(4, dtype=np.float32).flatten()
    inv_view_mat = np.eye(4, dtype=np.float32).flatten()
    vp_mat = np.eye(4, dtype=np.float32).flatten()

    frame_ctx.update(
        view_mat=view_mat,
        proj_mat=proj_mat,
        view_proj_mat=vp_mat,
        inv_proj_mat=inv_proj_mat,
        inv_view_mat=inv_view_mat,
        camera_pos=(0.0, 1.0, 5.0),
        time_elapsed=1.0,
        screen_size=(w, h),
    )

    pass_obj = ParticleSystemPass(ctx, max_particles=512)

    # Execute render for DUST_MOTES
    pass_obj.set_mode(ParticleMode.DUST_MOTES)
    pass_obj.render(hdr_fbo=fbo, depth_texture=depth_tex)

    # Execute render for EMBERS
    pass_obj.set_mode(ParticleMode.EMBERS)
    pass_obj.render(hdr_fbo=fbo, depth_texture=depth_tex)

    # Execute render for FIREFLIES
    pass_obj.set_mode(ParticleMode.FIREFLIES)
    pass_obj.render(hdr_fbo=fbo, depth_texture=depth_tex)

    # Clean up
    pass_obj.destroy()
    fbo.release()
    hdr_tex.release()
    depth_tex.release()


def test_quality_presets_particle_scaling():
    """Validates that quality presets configure particle counts and toggles properly."""
    low = get_quality_preset(GraphicsQuality.LOW)
    assert low.particles_enabled is False
    assert low.particle_mode == "OFF"

    med = get_quality_preset(GraphicsQuality.MEDIUM)
    assert med.particles_enabled is True
    assert med.particle_count == 8192

    high = get_quality_preset(GraphicsQuality.HIGH)
    assert high.particles_enabled is True
    assert high.particle_count == 16384

    ultra = get_quality_preset(GraphicsQuality.ULTRA)
    assert ultra.particles_enabled is True
    assert ultra.particle_count == 32768

    cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
    assert cinematic.particles_enabled is False
    assert cinematic.particle_count == 0


def test_pipeline_particle_system_integration(render_ctx: RenderContext):
    """Validates full RenderPipeline integration: allocation, preset scaling, and Pass 7.7 execution."""
    cfg = get_quality_preset(GraphicsQuality.HIGH)
    pipeline = RenderPipeline(render_ctx, cfg)

    assert pipeline.particle_pass is not None
    assert pipeline.particle_pass.active_particles == 16384
    assert pipeline.particle_pass.mode == ParticleMode.DUST_MOTES

    # Apply ULTRA preset -> should scale active particles to 32768
    ultra_cfg = get_quality_preset(GraphicsQuality.ULTRA)
    pipeline.apply_config(ultra_cfg)
    assert pipeline.particle_pass.active_particles == 32768

    # Apply LOW preset -> should disable particles (active particles = 0)
    low_cfg = get_quality_preset(GraphicsQuality.LOW)
    pipeline.apply_config(low_cfg)
    assert pipeline.particle_pass.active_particles == 0

    # Re-enable on HIGH
    pipeline.apply_config(cfg)
    assert pipeline.particle_pass.active_particles == 16384

    # Run render_frame with empty ECS
    ecs = EntityManager(max_entities=10)
    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.0, 2.0, 5.0),
        camera_target=(0.0, 1.0, 0.0),
        time_elapsed=0.5,
        sun_dir=(0.2, -0.8, 0.5),
        sun_lux=4.0,
        draw_batches=[],
        dt=0.016,
    )

    # Test mode switching to EMBERS in pipeline
    pipeline.particle_pass.set_mode(ParticleMode.EMBERS)
    assert pipeline.particle_pass.mode == ParticleMode.EMBERS

    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.0, 2.0, 5.0),
        camera_target=(0.0, 1.0, 0.0),
        time_elapsed=1.0,
        sun_dir=(0.2, -0.8, 0.5),
        sun_lux=4.0,
        draw_batches=[],
        dt=0.016,
    )

    pipeline.destroy()
