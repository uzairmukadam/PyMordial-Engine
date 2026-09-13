"""Unit and integration tests for BackgroundAssetLoader and LoadingScreen."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import moderngl

from engine.gfx.texture_atlas import (
    DecodedMaterialLayer,
    decode_material_folder,
    TextureArrayAtlas,
)
from engine.core.async_loader import BackgroundAssetLoader
from engine.gfx.loading_screen import LoadingScreen
from engine.events import publish_event, WindowResizeEvent


@pytest.fixture
def gl_ctx():
    ctx = moderngl.create_context(standalone=True)
    yield ctx
    ctx.release()


@pytest.fixture
def sample_material_dir(tmp_path: Path):
    """Creates synthetic PBR material directory with 64x64 textures."""
    mat_dir = tmp_path / "textures" / "test_cobble"
    mat_dir.mkdir(parents=True)

    # diff.png (RGBA)
    diff = Image.fromarray(np.full((64, 64, 4), (180, 160, 140, 255), dtype=np.uint8))
    diff.save(mat_dir / "diff.png")

    # nor.png (RGBA normal)
    nor = Image.fromarray(np.full((64, 64, 4), (128, 128, 255, 255), dtype=np.uint8))
    nor.save(mat_dir / "nor.png")

    # disp.png (L height)
    disp = Image.fromarray(np.full((64, 64), 130, dtype=np.uint8))
    disp.save(mat_dir / "disp.png")

    # arm.png (RGBA ARM)
    arm = Image.fromarray(np.full((64, 64, 4), (255, 128, 0, 255), dtype=np.uint8))
    arm.save(mat_dir / "arm.png")

    return tmp_path / "textures"


class TestAsyncLoaderAndLoadingScreen:
    def test_decode_material_folder_pure_cpu(self, sample_material_dir: Path):
        """Validates that decode_material_folder reads and resizes textures purely on CPU."""
        mat_folder = sample_material_dir / "test_cobble"
        decoded = decode_material_folder(mat_folder, width=128, height=128, name="test_cobble")

        assert isinstance(decoded, DecodedMaterialLayer)
        assert decoded.name == "test_cobble"
        assert decoded.width == 128
        assert decoded.height == 128
        assert len(decoded.diffuse_bytes) == 128 * 128 * 4
        assert len(decoded.normal_bytes) == 128 * 128 * 4
        assert len(decoded.disp_bytes) == 128 * 128 * 1
        assert len(decoded.arm_bytes) == 128 * 128 * 4

    def test_background_asset_loader_dispatch_and_poll(self, sample_material_dir: Path):
        """Validates that BackgroundAssetLoader queues tasks and streams completed layers."""
        loader = BackgroundAssetLoader(max_workers=2)
        names = loader.start_material_loading(sample_material_dir, width=64, height=64)
        assert "test_cobble" in names
        assert loader.total_materials == 1
        assert loader.completed_materials == 0

        # Wait briefly for worker thread to complete decode
        import time
        for _ in range(50):
            layer = loader.poll_material_layer()
            if layer is not None:
                break
            time.sleep(0.05)

        assert layer is not None
        assert layer.name == "test_cobble"
        assert loader.completed_materials == 1
        assert loader.material_progress == 1.0
        assert loader.is_material_loading_complete is True

        loader.shutdown(wait=True)

    def test_texture_atlas_upload_decoded_layer(self, gl_ctx, sample_material_dir: Path):
        """Validates that TextureArrayAtlas.upload_decoded_layer writes pre-decoded buffers to GPU."""
        atlas = TextureArrayAtlas(gl_ctx, width=64, height=64, max_layers=4)
        assert atlas.layer_count == 1  # Identity layer 0

        mat_folder = sample_material_dir / "test_cobble"
        decoded = decode_material_folder(mat_folder, width=64, height=64, name="test_cobble")

        assigned_layer = atlas.upload_decoded_layer(decoded, rebuild_mipmaps=False)
        assert assigned_layer == 1
        assert atlas.layer_count == 2
        assert atlas.name_to_layer["test_cobble"] == 1
        assert "test_cobble" in atlas.material_names

        atlas.rebuild_all_mipmaps()
        atlas.destroy()

    def test_loading_screen_headless_lifecycle(self, gl_ctx):
        """Validates LoadingScreen creation, progress lerping, and rendering in headless context."""
        screen = LoadingScreen(gl_ctx, width=1280, height=720, is_headless=True)
        assert screen.target_progress == 0.0
        assert screen.current_progress == 0.0

        screen.set_progress(0.75, status="Loading assets...", substatus="TESTING")
        assert screen.target_progress == 0.75
        assert screen.status_text == "Loading assets..."

        # Update 10 frames of 16ms
        for _ in range(10):
            screen.update(0.016)

        # Progress should smoothly interpolate toward 0.75
        assert screen.current_progress > 0.40
        assert screen.current_progress <= 0.75

        # Render headless
        screen.render()

        # Fade out
        screen.fade_out(duration=0.05)
        screen.destroy()
        assert screen.prog is None
        assert screen.vao is None

    def test_loading_screen_resize_adaptation(self, gl_ctx):
        """Validates that LoadingScreen dynamically adapts resolution, textures, and uniforms."""
        screen = LoadingScreen(gl_ctx, width=1280, height=720, is_headless=True)
        assert screen.width == 1280
        assert screen.height == 720
        assert screen.text_texture.size == (1280, 720)

        # Resize to 1920x1080 (16:9 Full HD)
        screen.resize(1920, 1080)
        assert screen.width == 1920
        assert screen.height == 1080
        assert screen.text_texture.size == (1920, 1080)
        assert screen.u_resolution.value == (1920.0, 1080.0)

        # Resize to 2560x1080 (21:9 Ultrawide)
        screen.resize(2560, 1080)
        assert screen.width == 2560
        assert screen.height == 1080
        assert screen.text_texture.size == (2560, 1080)
        assert screen.u_resolution.value == (2560.0, 1080.0)

        screen.destroy()

    def test_loading_screen_window_resize_event(self, gl_ctx):
        """Validates that LoadingScreen handles WindowResizeEvent from the central event bus."""
        screen = LoadingScreen(gl_ctx, width=1280, height=720, is_headless=True)
        assert screen.width == 1280

        # Dispatch resize event
        publish_event(WindowResizeEvent(width=1600, height=900))
        assert screen.width == 1600
        assert screen.height == 900
        assert screen.text_texture.size == (1600, 900)

        screen.destroy()

    def test_loading_screen_target_fbo_auto_adaptation(self, gl_ctx):
        """Validates that render() automatically detects target_fbo dimensions and resizes."""
        screen = LoadingScreen(gl_ctx, width=1280, height=720, is_headless=True)

        # Create an 800x600 FBO
        tex = gl_ctx.texture((800, 600), 4)
        fbo = gl_ctx.framebuffer(color_attachments=[tex])

        screen.render(target_fbo=fbo)
        assert screen.width == 800
        assert screen.height == 600
        assert screen.text_texture.size == (800, 600)

        fbo.release()
        tex.release()
        screen.destroy()

    def test_loading_screen_pure_black_background(self, gl_ctx):
        """Validates that the loading screen renders pure black background with corner icon."""
        w, h = 800, 600
        tex = gl_ctx.texture((w, h), 4)
        fbo = gl_ctx.framebuffer(color_attachments=[tex])

        screen = LoadingScreen(gl_ctx, width=w, height=h, is_headless=True)
        screen.set_progress(0.5, status="Loading world chunks...")
        screen.update(0.1)
        screen.render(target_fbo=fbo)

        raw = fbo.read(components=4, dtype="f1")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))

        # Top-left, center, and middle-left regions must be pure black (0, 0, 0)
        # Remember OpenGL FBO row 0 is bottom
        center_pixel = arr[h // 2, w // 2, :3]
        top_left_pixel = arr[h - 50, 50, :3]
        np.testing.assert_array_equal(center_pixel, [0, 0, 0])
        np.testing.assert_array_equal(top_left_pixel, [0, 0, 0])

        # Bottom-right corner region (around iconCenter) must have non-zero luminance
        scale = max(0.70, min(1.8, h / 1080.0))
        margin_x = int(56.0 * scale)
        margin_y = int(56.0 * scale)
        corner_region = arr[margin_y - 10:margin_y + 10, (w - margin_x) - 10:(w - margin_x) + 10, :3]
        assert np.max(corner_region) > 0, "Expected animated loading icon in bottom-right corner"

        fbo.release()
        tex.release()
        screen.destroy()

