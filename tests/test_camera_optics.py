"""Unit and Integration Tests for Phase 3: Cinematic Camera Optics & Lens Effects.

Validates:
- CameraOpticsPass initialization, framebuffer allocations, and dynamic resizing.
- Physical Bokeh Depth of Field execution across Circular, Hexagonal, and Anamorphic shapes.
- Camera Velocity Motion Blur reconstruction and directional tap integration.
- Anamorphic Lens Flare, Streak, Ghost reflections, and Halo ring generation.
- Master PostProcessPipeline integration (Chromatic Aberration, Vignette, Film Grain).
- Graphics quality presets scaling (LOW to CINEMATIC).
- Full RenderPipeline execution with Pass 7.8 and Pass 8.
"""

from pathlib import Path
import numpy as np
import pytest

from engine.core.ecs import EntityManager
from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.quality_presets import GraphicsQuality, get_quality_preset, RenderConfig
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.passes.camera_optics_pass import CameraOpticsPass
from engine.gfx.post_process import PostProcessPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=320, height=240)
    yield ctx
    ctx.destroy()


def test_camera_optics_pass_init_and_resize(render_ctx: RenderContext):
    """Verifies CameraOpticsPass allocation, shaders, FBOs, and dynamic resizing."""
    ctx = render_ctx.ctx
    w, h = 320, 240
    optics = CameraOpticsPass(ctx, width=w, height=h)

    assert optics.width == w
    assert optics.height == h
    assert optics.optics_texture is not None
    assert optics.optics_texture.size == (w, h)
    assert optics.flare_texture is not None
    assert optics.flare_texture.size == (w // 2, h // 2)

    # Test resizing
    new_w, new_h = 160, 120
    optics.resize(new_w, new_h)
    assert optics.width == new_w
    assert optics.height == new_h
    assert optics.optics_texture.size == (new_w, new_h)
    assert optics.flare_texture.size == (new_w // 2, new_h // 2)

    optics.destroy()


def test_camera_optics_render_dof_and_motion_blur(render_ctx: RenderContext):
    """Tests execution of Depth of Field and Motion Blur passes with various bokeh diaphragms."""
    ctx = render_ctx.ctx
    w, h = 160, 120

    hdr_tex = ctx.texture((w, h), 4, dtype="f2")
    depth_tex = ctx.depth_texture((w, h))
    hdr_fbo = ctx.framebuffer(color_attachments=[hdr_tex], depth_attachment=depth_tex)
    hdr_fbo.use()
    ctx.clear(0.2, 0.4, 0.8, 1.0, depth=0.5)

    frame_ctx = FrameContext(ctx)
    view_mat = np.eye(4, dtype=np.float32).flatten()
    proj_mat = np.eye(4, dtype=np.float32).flatten()
    inv_proj_mat = np.eye(4, dtype=np.float32).flatten()
    inv_view_mat = np.eye(4, dtype=np.float32).flatten()
    vp_mat = np.eye(4, dtype=np.float32).flatten()
    prev_vp_mat = np.eye(4, dtype=np.float32).flatten()

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

    optics = CameraOpticsPass(ctx, width=w, height=h)
    cfg = RenderConfig()
    cfg.dof_enabled = True
    cfg.motion_blur_enabled = True

    # 1. Circular Bokeh
    cfg.dof_bokeh_shape = "CIRCULAR"
    res1 = optics.render_optics(hdr_tex, depth_tex, prev_vp_mat, cfg)
    assert res1 is not None

    # 2. Hexagonal Bokeh
    cfg.dof_bokeh_shape = "HEXAGONAL"
    res2 = optics.render_optics(hdr_tex, depth_tex, prev_vp_mat, cfg)
    assert res2 is not None

    # 3. Anamorphic 2x Bokeh
    cfg.dof_bokeh_shape = "ANAMORPHIC"
    cfg.dof_anamorphic_ratio = 2.0
    res3 = optics.render_optics(hdr_tex, depth_tex, prev_vp_mat, cfg)
    assert res3 is not None

    optics.destroy()
    hdr_fbo.release()
    hdr_tex.release()
    depth_tex.release()


def test_anamorphic_lens_flare_generation(render_ctx: RenderContext):
    """Verifies anamorphic streak, chromatic ghost reflections, and halo generation."""
    ctx = render_ctx.ctx
    w, h = 160, 120

    hdr_tex = ctx.texture((w, h), 4, dtype="f2")
    hdr_fbo = ctx.framebuffer(color_attachments=[hdr_tex])
    hdr_fbo.use()
    # Draw bright spot (> threshold 1.8)
    ctx.clear(3.5, 3.2, 2.8, 1.0)

    optics = CameraOpticsPass(ctx, width=w, height=h)
    cfg = RenderConfig()
    cfg.lens_flare_enabled = True
    cfg.lens_flare_threshold = 1.8
    cfg.lens_flare_streak_intensity = 0.8
    cfg.lens_flare_ghost_intensity = 0.5
    cfg.lens_flare_halo_intensity = 0.3

    flare_out = optics.render_flare(hdr_tex, cfg)
    assert flare_out is not None
    assert flare_out.size == (w // 2, h // 2)

    optics.destroy()
    hdr_fbo.release()
    hdr_tex.release()


def test_post_process_cinematic_lens_imperfections(render_ctx: RenderContext):
    """Validates Chromatic Aberration, Vignette, and Film Grain composite in PostProcessPipeline."""
    ctx = render_ctx.ctx
    w, h = 160, 120
    quad_vert = (ROOT_DIR / "shaders" / "fullscreen_quad.vert").read_text(encoding="utf-8")
    post_frag = (ROOT_DIR / "shaders" / "post_process.frag").read_text(encoding="utf-8")

    cfg = RenderConfig()
    cfg.chromatic_aberration_enabled = True
    cfg.chromatic_aberration_intensity = 0.010
    cfg.vignette_enabled = True
    cfg.vignette_intensity = 0.50
    cfg.film_grain_enabled = True
    cfg.film_grain_intensity = 0.08
    cfg.lens_flare_enabled = True

    post = PostProcessPipeline(ctx, w, h, cfg, post_frag, quad_vert)

    # Create dummy flare texture
    flare_tex = ctx.texture((w // 2, h // 2), 4, dtype="f2")
    flare_tex.write(b"\x3f" * (w // 2 * h // 2 * 8))

    post.render(target_fbo=post.final_fbo, flare_texture=flare_tex, time_elapsed=2.5)
    assert post.output_texture_id > 0

    flare_tex.release()
    post.destroy()


def test_quality_presets_camera_optics_scaling():
    """Validates that all presets configure Phase 3 camera optics appropriately."""
    low = get_quality_preset(GraphicsQuality.LOW)
    assert low.dof_enabled is False
    assert low.motion_blur_enabled is False
    assert low.lens_flare_enabled is False
    assert low.chromatic_aberration_enabled is False
    assert low.vignette_enabled is True
    assert low.film_grain_enabled is False

    med = get_quality_preset(GraphicsQuality.MEDIUM)
    assert med.dof_enabled is False
    assert med.motion_blur_enabled is True
    assert med.motion_blur_samples == 6
    assert med.chromatic_aberration_enabled is True
    assert med.vignette_enabled is True
    assert med.film_grain_enabled is True

    high = get_quality_preset(GraphicsQuality.HIGH)
    assert high.dof_enabled is True
    assert high.dof_bokeh_shape == "CIRCULAR"
    assert high.motion_blur_enabled is True
    assert high.motion_blur_samples == 12
    assert high.lens_flare_enabled is True
    assert high.chromatic_aberration_enabled is True
    assert high.film_grain_enabled is True

    ultra = get_quality_preset(GraphicsQuality.ULTRA)
    assert ultra.dof_enabled is True
    assert ultra.dof_bokeh_shape == "HEXAGONAL"
    assert ultra.motion_blur_enabled is True
    assert ultra.motion_blur_samples == 16
    assert ultra.lens_flare_enabled is True

    cinematic = get_quality_preset(GraphicsQuality.CINEMATIC)
    assert cinematic.dof_enabled is True
    assert cinematic.dof_bokeh_shape == "ANAMORPHIC"
    assert cinematic.dof_anamorphic_ratio == 2.0
    assert cinematic.motion_blur_enabled is True
    assert cinematic.motion_blur_samples == 24
    assert cinematic.lens_flare_enabled is True


def test_pipeline_camera_optics_integration(render_ctx: RenderContext):
    """Validates full RenderPipeline integration of CameraOpticsPass (Pass 7.8 & Pass 8)."""
    cfg = get_quality_preset(GraphicsQuality.HIGH)
    pipeline = RenderPipeline(render_ctx, cfg)

    assert pipeline.camera_optics_pass is not None
    assert pipeline.camera_optics_pass.optics_texture is not None

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

    # Switch to CINEMATIC (Anamorphic DoF + 24 motion blur taps)
    cinematic_cfg = get_quality_preset(GraphicsQuality.CINEMATIC)
    pipeline.apply_config(cinematic_cfg)
    pipeline.render_frame(
        ecs=ecs,
        camera_pos=(0.5, 2.2, 4.8),
        camera_target=(0.0, 1.0, 0.0),
        time_elapsed=1.0,
        sun_dir=(0.2, -0.8, 0.5),
        sun_lux=4.0,
        draw_batches=[],
        dt=0.016,
    )

    pipeline.destroy()
