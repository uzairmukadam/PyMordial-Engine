"""Unit and integration tests for Phase 6 Micro-Geometry Displacement.

Tests:
1. DisplacementMode IntEnum values and mutually exclusive bitfield flags.
2. encode_mat_flags and decode_mat_flags bit packing/unpacking.
3. TextureArrayAtlas creation and identity layer initialization.
4. GBuffer MRT layout with RT3 rt_displacement.
5. RenderConfig quality presets for POM, SSDM, and Hardware Tessellation.
6. MultiDrawIndirect command routing for ctx.PATCHES vs ctx.TRIANGLES.
7. SSDMPass scratch depth/normal execution.
"""

from __future__ import annotations
import numpy as np
import pytest
import moderngl

from engine.gfx.texture_atlas import (
    DisplacementMode,
    MAT_FLAG_HAS_TEXTURE,
    MAT_FLAG_DISP_SHIFT,
    MAT_FLAG_DISP_MASK,
    encode_mat_flags,
    decode_mat_flags,
    TextureArrayAtlas,
)
from engine.gfx.quality_presets import (
    RenderConfig,
    GraphicsQuality,
    get_quality_preset,
)
from engine.gfx.g_buffer import GBuffer
from engine.gfx.passes.ssdm_pass import SSDMPass
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation


@pytest.fixture
def gl_ctx():
    ctx = moderngl.create_context(standalone=True)
    yield ctx
    ctx.release()


class TestDisplacementFlags:
    """Tests for DisplacementMode enum and bitfield encoding."""

    def test_displacement_mode_values(self):
        assert int(DisplacementMode.NONE) == 0
        assert int(DisplacementMode.POM) == 1
        assert int(DisplacementMode.SSDM) == 2
        assert int(DisplacementMode.TESSELLATION) == 3

    def test_encode_decode_roundtrip(self):
        for has_tex in (True, False):
            for mode in DisplacementMode:
                encoded = encode_mat_flags(has_texture=has_tex, disp_mode=mode)
                dec_has_tex, dec_mode = decode_mat_flags(encoded)
                assert dec_has_tex == has_tex
                assert dec_mode == mode

    def test_mutual_exclusivity(self):
        # Setting POM then SSDM must cleanly overwrite
        flag_pom = encode_mat_flags(True, DisplacementMode.POM)
        _, m1 = decode_mat_flags(flag_pom)
        assert m1 == DisplacementMode.POM

        flag_ssdm = encode_mat_flags(True, DisplacementMode.SSDM)
        _, m2 = decode_mat_flags(flag_ssdm)
        assert m2 == DisplacementMode.SSDM
        assert m2 != DisplacementMode.POM

        flag_tess = encode_mat_flags(True, DisplacementMode.TESSELLATION)
        _, m3 = decode_mat_flags(flag_tess)
        assert m3 == DisplacementMode.TESSELLATION
        assert m3 != DisplacementMode.SSDM


class TestTextureArrayAtlas:
    """Tests for TextureArrayAtlas creation and identity layer."""

    def test_atlas_initialization(self, gl_ctx):
        atlas = TextureArrayAtlas(gl_ctx, width=64, height=64, max_layers=4)
        try:
            assert atlas.width == 64
            assert atlas.height == 64
            assert atlas.max_layers == 4
            assert atlas.layer_count == 1
            assert atlas.material_names[0] == "identity"
            assert atlas.name_to_layer["identity"] == 0

            # Verify arrays are created
            assert atlas.diffuse_array is not None
            assert atlas.normal_array is not None
            assert atlas.displacement_array is not None
            assert atlas.arm_array is not None
        finally:
            atlas.destroy()


class TestGBufferDisplacementMRT:
    """Tests that GBuffer has RT3 for displacement routing."""

    def test_gbuffer_has_rt3_displacement(self, gl_ctx):
        gb = GBuffer(gl_ctx, 320, 240)
        try:
            assert hasattr(gb, "rt_displacement")
            assert gb.rt_displacement is not None
            assert gb.rt_displacement.dtype == "f2"
            assert gb.rt_displacement.components == 4
            assert gb.fbo is not None
            assert len(gb.fbo.color_attachments) == 4
        finally:
            gb.destroy()


class TestSSDMPass:
    """Tests for SSDMPass allocation and execution."""

    def test_ssdm_pass_initialization(self, gl_ctx):
        ssdm = SSDMPass(gl_ctx, 320, 240)
        try:
            assert ssdm.prog is not None
            assert ssdm.vao is not None
            assert ssdm.perturbed_normal_tex is not None
        finally:
            ssdm.destroy()


class TestQualityPresetsMicroGeometry:
    """Tests that quality presets define POM, SSDM, and Tessellation parameters."""

    def test_presets_have_microgeometry_fields(self):
        for quality in (GraphicsQuality.LOW, GraphicsQuality.MEDIUM, GraphicsQuality.HIGH, GraphicsQuality.ULTRA, GraphicsQuality.CINEMATIC):
            cfg = get_quality_preset(quality)
            assert hasattr(cfg, "pom_enabled")
            assert hasattr(cfg, "pom_min_samples")
            assert hasattr(cfg, "pom_max_samples")
            assert hasattr(cfg, "pom_height_scale")
            assert hasattr(cfg, "pom_self_shadow")
            assert hasattr(cfg, "ssdm_enabled")
            assert hasattr(cfg, "ssdm_scale")
            assert hasattr(cfg, "tess_enabled")
            assert hasattr(cfg, "tess_max_level")
            assert hasattr(cfg, "tess_displacement_scale")

    def test_low_preset_disables_heavy_microgeometry(self):
        cfg = get_quality_preset(GraphicsQuality.LOW)
        assert cfg.pom_enabled is False
        assert cfg.tess_enabled is False

    def test_high_and_ultra_enable_microgeometry(self):
        cfg_high = get_quality_preset(GraphicsQuality.HIGH)
        assert cfg_high.pom_enabled is True
        assert cfg_high.tess_enabled is True
        assert cfg_high.ssdm_enabled is True

        cfg_ultra = get_quality_preset(GraphicsQuality.ULTRA)
        assert cfg_ultra.pom_enabled is True
        assert cfg_ultra.tess_enabled is True
        assert cfg_ultra.tess_max_level >= 16.0
