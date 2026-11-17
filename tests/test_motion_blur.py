"""Tests for the velocity-buffer motion blur pass (todo p6-motion-blur).

Validates:
- Static camera + static scene produces an unblurred (identity) result.
- Moving camera produces a bounded, non-identity result.
- MotionBlurPass init / resize / destroy lifecycle.
- New RenderConfig motion blur fields round-trip.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.render_graph import RenderGraphContext
from engine.gfx.passes.motion_blur_pass import MotionBlurPass

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=128, height=96)
    yield ctx
    ctx.destroy()


def _make_scene(ctx, w: int, h: int):
    """Builds a gradient HDR scene, a depth texture, and a velocity texture."""
    scene = ctx.texture((w, h), 4, dtype="f2")
    pixels = np.zeros((h, w, 4), dtype=np.float16)
    ramp_x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ramp_y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    pixels[..., 0] = ramp_x[None, :]
    pixels[..., 1] = ramp_y[:, None]
    pixels[..., 2] = 0.5
    pixels[..., 3] = 1.0
    scene.write(pixels.tobytes())

    depth = ctx.depth_texture((w, h))
    depth_tex = ctx.texture((w, h), 1, dtype="f4")
    depth_tex.write(np.full((h, w, 1), 0.5, dtype=np.float32).tobytes())

    velocity = ctx.texture((w, h), 2, dtype="f2")
    velocity.filter = (0, 0)
    return scene, depth, depth_tex, velocity


def _run_pass(ctx, w, h, scene, depth_tex, velocity, config):
    frame_ctx = FrameContext(ctx)
    frame_ctx.update(
        view_mat=np.eye(4, dtype=np.float32).flatten(),
        proj_mat=np.eye(4, dtype=np.float32).flatten(),
        view_proj_mat=np.eye(4, dtype=np.float32).flatten(),
        inv_proj_mat=np.eye(4, dtype=np.float32).flatten(),
        inv_view_mat=np.eye(4, dtype=np.float32).flatten(),
        camera_pos=(0.0, 1.0, 5.0),
        time_elapsed=1.0,
        screen_size=(w, h),
    )
    g_buffer = SimpleNamespace(velocity_texture=velocity, depth_texture=depth_tex)
    context = RenderGraphContext(
        ctx=ctx,
        width=w,
        height=h,
        config=config,
        frame_context=frame_ctx,
        resources={"hdr_color": scene, "g_buffer": g_buffer},
    )
    blur = MotionBlurPass(ctx, w, h)
    try:
        blur.execute(context)
        out_tex = context.resources["motion_blur_output"]
        assert out_tex is not None
        return np.frombuffer(out_tex.read(), dtype=np.float16).reshape(h, w, 4).copy()
    finally:
        blur.destroy()


def test_static_scene_is_unchanged(render_ctx: RenderContext):
    """Zero velocity must pass the input through untouched."""
    ctx = render_ctx.ctx
    w, h = 128, 96
    scene, _depth, depth_tex, velocity = _make_scene(ctx, w, h)
    velocity.write(np.zeros((h, w, 2), dtype=np.float16).tobytes())

    cfg = RenderConfig(motion_blur_enabled=True)
    out = _run_pass(ctx, w, h, scene, depth_tex, velocity, cfg)
    src = np.frombuffer(scene.read(), dtype=np.float16).reshape(h, w, 4)

    assert np.allclose(out[..., :3], src[..., :3], atol=1e-3)

    scene.release()
    depth_tex.release()
    velocity.release()


def test_moving_camera_blurs_and_stays_bounded(render_ctx: RenderContext):
    """Non-zero velocity must change the image, but only within a bounded range."""
    ctx = render_ctx.ctx
    w, h = 128, 96
    scene, _depth, depth_tex, velocity = _make_scene(ctx, w, h)

    vel = np.zeros((h, w, 2), dtype=np.float16)
    vel[..., 0] = 0.05  # 5% of screen width of motion
    velocity.write(vel.tobytes())

    cfg = RenderConfig(motion_blur_enabled=True, motion_blur_intensity=1.0)
    out = _run_pass(ctx, w, h, scene, depth_tex, velocity, cfg)
    src = np.frombuffer(scene.read(), dtype=np.float16).reshape(h, w, 4)

    diff = np.abs(out[..., :3].astype(np.float32) - src[..., :3].astype(np.float32))
    assert float(diff.max()) > 1e-3, "moving camera should blur the image"
    assert float(diff.max()) <= 1.0, "blur difference must stay bounded"
    assert np.all(np.isfinite(out[..., :3]))

    scene.release()
    depth_tex.release()
    velocity.release()


def test_motion_blur_pass_lifecycle(render_ctx: RenderContext):
    """Init, resize, and destroy must allocate and release cleanly."""
    ctx = render_ctx.ctx
    blur = MotionBlurPass(ctx, 128, 96)
    assert blur.tile_texture is not None
    assert blur.tile_texture.size == (8, 6)  # 128/16, 96/16
    assert blur.output_texture.size == (128, 96)

    blur.resize(64, 48)
    assert blur.output_texture.size == (64, 48)
    assert blur.tile_texture.size == (4, 3)

    blur.destroy()
    assert blur.tile_texture is None
    assert blur.output_texture is None


def test_motion_blur_config_round_trip():
    """New tile-size and max-sample fields must round-trip through RenderConfig."""
    cfg = RenderConfig(motion_blur_tile_size=8, motion_blur_max_samples=24)
    assert cfg.motion_blur_tile_size == 8
    assert cfg.motion_blur_max_samples == 24

    default = RenderConfig()
    assert default.motion_blur_tile_size == 16
    assert default.motion_blur_max_samples == 32
    assert default.motion_blur_enabled is False
    assert default.motion_blur_samples == 12
    assert default.motion_blur_intensity == 1.5
    assert default.motion_blur_max_radius == 48.0
