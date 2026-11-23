"""Regression tests for Horizon 1 foundation fixes.

Covers the MDI command packing contract, spot-shadow MDI population, the
RenderConfig shape used by EngineTweaks, and project validation edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from engine.app.config import ProjectConfig
from engine.app.project_validator import (
    ProjectValidationError,
    resolve_project_path,
    validate_project_config,
)
from engine.debug.engine_tweaks import EngineTweaks, GBufferDebugMode
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.quality_presets import GIMode, GraphicsQuality, RenderConfig, get_quality_preset


class TestMDICommandPacking:
    """MegaBuffer pre-offsets indices, so the indirect baseVertex must stay 0."""

    def test_base_vertex_is_always_zero(self):
        mdi = MultiDrawIndirect.__new__(MultiDrawIndirect)
        mdi.max_commands = 8
        mdi.command_buffer = np.zeros(8, dtype=MultiDrawIndirect.COMMAND_DTYPE)
        mdi.command_count = 0
        mdi.dropped_commands = 0

        alloc = MeshAllocation(first_index=2196, index_count=6, base_vertex=449, vertex_count=4)
        mdi.add_command(alloc, instance_count=3, base_instance=17)

        cmd = mdi.command_buffer[0]
        assert cmd["base_vertex"] == 0, "baseVertex must be 0; indices are already globally offset"
        assert cmd["first_index"] == 2196
        assert cmd["count"] == 6
        assert cmd["instance_count"] == 3
        assert cmd["base_instance"] == 17

    def test_command_stride_is_twenty_bytes(self):
        assert MultiDrawIndirect.COMMAND_DTYPE.itemsize == 20
        assert MultiDrawIndirect.COMMAND_BYTES == 20

    def test_overflow_is_counted_not_silently_dropped(self):
        mdi = MultiDrawIndirect.__new__(MultiDrawIndirect)
        mdi.max_commands = 2
        mdi.command_buffer = np.zeros(2, dtype=MultiDrawIndirect.COMMAND_DTYPE)
        mdi.command_count = 0
        mdi.dropped_commands = 0

        alloc = MeshAllocation(first_index=0, index_count=36, base_vertex=0, vertex_count=24)
        for _ in range(5):
            mdi.add_command(alloc)

        assert mdi.command_count == 2
        assert mdi.dropped_commands == 3


class TestMDIRendering:
    """Pixel-level proof that MDI renders the same geometry as a direct draw."""

    @pytest.fixture(scope="class")
    @classmethod
    def gl(cls):
        import moderngl

        ctx = moderngl.create_standalone_context(require=450)
        yield ctx
        ctx.release()

    def _make_program(self, ctx):
        return ctx.program(
            vertex_shader="""
            #version 450
            in vec3 in_position;
            uniform mat4 u_MVP;
            void main() { gl_Position = u_MVP * vec4(in_position, 1.0); }
            """,
            fragment_shader="""
            #version 450
            out vec4 f_color;
            void main() { f_color = vec4(1.0); }
            """,
        )

    @pytest.mark.parametrize("mesh_name", ["cube", "sphere", "capsule"])
    def test_indirect_matches_direct_draw(self, gl, mesh_name):
        import moderngl

        mega = MegaBuffer(gl)
        alloc = mega.allocations[mesh_name]
        prog = self._make_program(gl)
        vao = mega.get_vao(prog)

        fbo = gl.framebuffer(color_attachments=[gl.texture((64, 64), 4)])
        fbo.use()
        gl.viewport = (0, 0, 64, 64)
        gl.enable(moderngl.DEPTH_TEST)

        mvp = np.eye(4, dtype=np.float32)
        mvp[0, 0] = 1.5
        mvp[1, 1] = 1.5
        mvp[2, 2] = 0.5
        prog["u_MVP"].write(mvp.tobytes())

        def lit_pixels() -> int:
            raw = np.frombuffer(fbo.read(components=4, dtype="f1"), dtype=np.uint8)
            return int((raw.reshape(64, 64, 4)[:, :, 0] > 0).sum())

        gl.clear(0, 0, 0, 1)
        vao.render(moderngl.TRIANGLES, vertices=alloc.index_count, first=alloc.first_index)
        direct = lit_pixels()

        mdi = MultiDrawIndirect(gl)
        mdi.begin_frame()
        mdi.add_command(alloc, instance_count=1, base_instance=0)
        gl.clear(0, 0, 0, 1)
        mdi.submit(vao, prog)
        indirect = lit_pixels()

        assert direct > 0, f"{mesh_name} direct draw produced no pixels"
        assert indirect == direct, (
            f"{mesh_name}: indirect draw covered {indirect} px but direct draw covered {direct} px"
        )

        mdi.destroy()
        fbo.release()


class TestSpotShadowMDIPopulation:
    """Spot shadow maps must receive a populated command list each frame."""

    def test_spot_shadow_commands_are_recorded(self):
        from engine.core.ecs import EntityManager
        from engine.gfx.context import RenderContext
        from engine.gfx.pipeline import RenderPipeline

        render_ctx = RenderContext.create_headless(width=160, height=120)
        try:
            ecs = EntityManager(max_entities=16)
            ecs.create_entity(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            ecs.create_entity(position=(2.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

            pipeline = RenderPipeline(render_ctx)
            pipeline.lights_pass.add_spot_light(
                position=(0.0, 5.0, 0.0),
                direction=(0.0, -1.0, 0.0),
                radius=25.0,
                inner_cone_angle=0.3,
                outer_cone_angle=0.6,
                color=(1.0, 1.0, 1.0),
                intensity=5.0,
                cast_shadow=True,
            )

            pipeline.render_frame(
                ecs=ecs,
                camera_pos=(0.0, 4.0, 8.0),
                camera_target=(0.0, 0.0, 0.0),
                time_elapsed=0.5,
            )

            assert pipeline.shadow_mdi.command_count > 0, (
                "spot shadow MDI was never populated; spot shadow maps render nothing"
            )
            pipeline.destroy()
        finally:
            render_ctx.destroy()


class TestRenderConfigShape:
    """EngineTweaks writes these fields, so RenderConfig must declare them."""

    @pytest.mark.parametrize(
        "field",
        ["frustum_cull_enabled", "hiz_cull_enabled", "hiz_debug_mip", "max_entities"],
    )
    def test_declared_fields_exist(self, field):
        assert field in RenderConfig.__dataclass_fields__

    def test_engine_tweaks_sync_targets_declared_fields(self):
        cfg = get_quality_preset(GraphicsQuality.HIGH)
        tweaks = EngineTweaks()
        tweaks.frustum_cull_enabled = False
        tweaks.hiz_cull_enabled = True
        tweaks.apply_to_pipeline(SimpleNamespace(config=cfg), force=True)
        assert cfg.frustum_cull_enabled is False
        assert cfg.hiz_cull_enabled is True

    def test_preset_copy_is_independent(self):
        first = get_quality_preset(GraphicsQuality.HIGH)
        first.max_entities = 123
        second = get_quality_preset(GraphicsQuality.HIGH)
        assert second.max_entities != 123

    def test_gi_mode_has_no_removed_modes(self):
        assert {m.value for m in GIMode} == {"OFF", "IBL", "SSGI", "RMGI"}

    def test_gbuffer_debug_has_no_lpv_slot(self):
        assert 9 not in {int(m) for m in GBufferDebugMode}


class TestProjectValidation:
    def test_valid_default_config(self):
        assert validate_project_config(ProjectConfig()).is_valid

    def test_empty_title_is_rejected(self):
        report = validate_project_config(ProjectConfig(title="   "))
        assert not report.is_valid
        assert any("title" in e.lower() for e in report.errors)

    def test_non_sized_gravity_reports_cleanly(self):
        cfg = ProjectConfig()
        cfg.gravity = None  # type: ignore[assignment]
        report = validate_project_config(cfg)
        assert not report.is_valid
        assert any("gravity" in e.lower() for e in report.errors)

    def test_headless_and_fullscreen_conflict(self):
        report = validate_project_config(ProjectConfig(headless=True, fullscreen=True))
        assert not report.is_valid
        assert any("headless" in e.lower() for e in report.errors)

    def test_negative_max_frames_is_rejected(self):
        report = validate_project_config(ProjectConfig(max_frames=-1))
        assert not report.is_valid

    def test_raise_for_errors_raises(self):
        with pytest.raises(ProjectValidationError):
            validate_project_config(ProjectConfig(width=0)).raise_for_errors()

    def test_bare_name_resolves_under_projects_root(self, tmp_path):
        resolved = resolve_project_path("my_game", engine_root=tmp_path)
        assert resolved == (tmp_path / "projects" / "my_game").resolve()
