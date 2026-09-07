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
