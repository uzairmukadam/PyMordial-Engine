"""Unified Master Render Graph Coordinator for PyMordial Engine."""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import moderngl

from engine.core.ecs import EntityManager
from engine.core.math_utils import matrix_perspective, matrix_look_at, mat4_mul, mat4_inv
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.frame_context import FrameContext
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.g_buffer import GBuffer
from engine.gfx.shadow_csm import CascadedShadowMap
from engine.gfx.post_process import PostProcessPipeline


SHADER_DIR = Path(__file__).resolve().parent.parent.parent / "shaders"


def _load_shader(rel_path: str) -> str:
    path = SHADER_DIR / rel_path
    return path.read_text(encoding="utf-8")


class RenderPipeline:
    """Coordinates the full ModernGL 4.5 MDI, Reversed-Z, CSM, and deferred resolve pipeline."""

    __slots__ = (
        "ctx_wrapper",
        "ctx",
        "config",
        "frame_context",
        "mega_buffer",
        "mdi",
        "g_buffer",
        "csm",
        "post_process",
        "ssbo_transforms",
        "ssbo_materials",
        "gbuffer_prog",
        "csm_prog",
        "resolve_prog",
        "resolve_vao",
        "gbuffer_vao",
        "csm_vao",
        "_view_mat",
        "_proj_mat",
        "_vp_mat",
        "_inv_proj",
        "_inv_view",
        "_cam_pos",
        "_cam_fwd",
        "_sun_v",
        "_cube_alloc",
        "_sphere_alloc",
        "_capsule_alloc",
        "_csm_cascade_idx_uniform",
        "_u_pcf_samples",
        "_u_sscs_enabled",
        "_u_sscs_steps",
        "_u_sscs_thickness",
        "_u_cascade_count",
    )

    def __init__(
        self,
        ctx_wrapper: RenderContext,
        config: RenderConfig | None = None,
    ) -> None:
        self.ctx_wrapper = ctx_wrapper
        self.ctx = ctx_wrapper.ctx
        self.config = config if config is not None else ctx_wrapper.config

        # 1. Initialize UBO 0 (std140 binding 0)
        self.frame_context = FrameContext(self.ctx)

        # 2. Initialize Shared Mega-Buffer & MDI Batcher
        self.mega_buffer = MegaBuffer(self.ctx)
        self.mdi = MultiDrawIndirect(self.ctx)

        # 3. Initialize MRT G-Buffer (Reversed-Z 32F)
        self.g_buffer = GBuffer(
            self.ctx,
            self.ctx_wrapper.width,
            self.ctx_wrapper.height,
            reverse_z=self.config.reverse_z,
        )

        # 4. Initialize Cascaded Shadow Maps (CSM)
        self.csm = CascadedShadowMap(
            self.ctx,
            atlas_size=self.config.shadow_resolution,
            cascade_count=self.config.csm_cascades,
            max_distance=self.config.shadow_distance,
        )

        # 5. Initialize SSBO 1 (Transforms) & SSBO 2 (Materials)
        # Pre-allocate GPU buffers for 100,000 entities
        max_entities = 100_000
        self.ssbo_transforms = self.ctx.buffer(reserve=max_entities * 16 * 4)
        self.ssbo_transforms.bind_to_storage_buffer(binding=1)

        self.ssbo_materials = self.ctx.buffer(reserve=max_entities * 8 * 4)
        self.ssbo_materials.bind_to_storage_buffer(binding=2)

        # Pre-allocated scratch buffers for zero-allocation frame rendering
        self._view_mat = np.zeros(16, dtype=np.float32)
        self._proj_mat = np.zeros(16, dtype=np.float32)
        self._vp_mat = np.zeros(16, dtype=np.float32)
        self._inv_proj = np.zeros(16, dtype=np.float32)
        self._inv_view = np.zeros(16, dtype=np.float32)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_fwd = np.zeros(3, dtype=np.float32)
        self._sun_v = np.zeros(3, dtype=np.float32)

        # 6. Compile Shaders
        quad_vert = _load_shader("fullscreen_quad.vert")
        gbuffer_vert = _load_shader("gbuffer.vert")
        gbuffer_frag = _load_shader("gbuffer.frag")
        csm_vert = _load_shader("csm_depth.vert")
        csm_frag = _load_shader("csm_depth.frag")
        resolve_frag = _load_shader("deferred_resolve.frag")
        post_frag = _load_shader("post_process.frag")

        # Compile MDI G-Buffer program
        self.gbuffer_prog = self.ctx.program(
            vertex_shader=gbuffer_vert,
            fragment_shader=gbuffer_frag,
        )
        self.gbuffer_vao = self.mega_buffer.get_vao(self.gbuffer_prog)

        # Compile CSM Depth pass program
        self.csm_prog = self.ctx.program(
            vertex_shader=csm_vert,
            fragment_shader=csm_frag,
        )
        self.csm_vao = self.mega_buffer.get_vao(self.csm_prog)

        # Compile Consolidated Lighting & Fog Resolve pass
        self.resolve_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=resolve_frag,
        )
        self.resolve_vao = self.ctx.vertex_array(self.resolve_prog, [])

        # Texture unit binding points for resolve pass
        if "u_GBufferAlbedo" in self.resolve_prog:
            self.resolve_prog["u_GBufferAlbedo"].value = 0
        if "u_GBufferNormal" in self.resolve_prog:
            self.resolve_prog["u_GBufferNormal"].value = 1
        if "u_GBufferDepth" in self.resolve_prog:
            self.resolve_prog["u_GBufferDepth"].value = 2
        if "u_ShadowAtlas" in self.resolve_prog:
            self.resolve_prog["u_ShadowAtlas"].value = 3

        # Pre-cache mesh allocations and shader uniforms to eliminate per-frame dictionary lookups
        self._cube_alloc = self.mega_buffer.allocations["cube"]
        self._sphere_alloc = self.mega_buffer.allocations["sphere"]
        self._capsule_alloc = self.mega_buffer.allocations["capsule"]
        self._csm_cascade_idx_uniform = self.csm_prog.get("u_CascadeIndex", None)
        self._u_pcf_samples = self.resolve_prog.get("u_PCF_Samples", None)
        self._u_sscs_enabled = self.resolve_prog.get("u_SSCS_Enabled", None)
        self._u_sscs_steps = self.resolve_prog.get("u_SSCS_Steps", None)
        self._u_sscs_thickness = self.resolve_prog.get("u_SSCS_Thickness", None)
        self._u_cascade_count = self.resolve_prog.get("u_CascadeCount", None)

        # 7. Post-Processing & Tonemapping Pipeline
        self.post_process = PostProcessPipeline(
            self.ctx,
            self.ctx_wrapper.width,
            self.ctx_wrapper.height,
            self.config,
            post_frag,
            quad_vert,
        )

    def apply_config(self, new_config: RenderConfig) -> None:
        """Applies dynamic graphics quality configuration changes."""
        self.config = new_config
        self.csm.resize_atlas(new_config.shadow_resolution)
        self.post_process.config = new_config
        self.ctx_wrapper.config = new_config
        self.ctx_wrapper.ctx.depth_func = ">" if new_config.reverse_z else "<"

    def resize(self, width: int, height: int) -> None:
        """Resizes MRT G-Buffer and Post-Process framebuffers on window resize."""
        self.ctx_wrapper.width = width
        self.ctx_wrapper.height = height
        self.g_buffer.resize(width, height)
        self.post_process.resize(width, height)

    def render_frame(
        self,
        ecs: EntityManager,
        camera_pos: tuple[float, float, float] | np.ndarray = (0.0, 5.0, 10.0),
        camera_target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        time_elapsed: float = 0.0,
        sun_dir: tuple[float, float, float] = (0.35, -0.85, 0.40),
        sun_lux: float = 4.0,
        fovy_deg: float = 60.0,
        draw_batches: list[tuple[MeshAllocation | str, int, int]] | None = None,
    ) -> None:
        """Executes full 4-pass deferred render pipeline."""
        active_count = ecs.active_count
        if active_count <= 0:
            return

        w, h = self.ctx_wrapper.width, self.ctx_wrapper.height
        aspect = w / (h if h > 0 else 1)

        # 1. Update SSBO 1 & SSBO 2 from ECS contiguous memory tables (Zero-allocation direct view)
        self.ssbo_transforms.write(ecs.get_active_transforms_view())
        self.ssbo_materials.write(ecs.get_active_materials_view())

        # 2. Camera Matrices (Reversed-Z)
        self._cam_pos[0] = float(camera_pos[0])
        self._cam_pos[1] = float(camera_pos[1])
        self._cam_pos[2] = float(camera_pos[2])

        matrix_look_at(camera_pos, camera_target, up=(0.0, 1.0, 0.0), out=self._view_mat)
        matrix_perspective(
            math.radians(fovy_deg),
            aspect,
            near=0.1,
            far=self.config.shadow_distance * 2.0,
            reverse_z=self.config.reverse_z,
            out=self._proj_mat,
        )
        mat4_mul(self._proj_mat, self._view_mat, out=self._vp_mat)
        mat4_inv(self._proj_mat, out=self._inv_proj)
        mat4_inv(self._view_mat, out=self._inv_view)

        fx = float(camera_target[0] - camera_pos[0])
        fy = float(camera_target[1] - camera_pos[1])
        fz = float(camera_target[2] - camera_pos[2])
        inv_fwd_len = 1.0 / (math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-6)
        self._cam_fwd[0] = fx * inv_fwd_len
        self._cam_fwd[1] = fy * inv_fwd_len
        self._cam_fwd[2] = fz * inv_fwd_len

        # 3. Cascaded Shadow Maps Matrices
        self._sun_v[0] = float(sun_dir[0])
        self._sun_v[1] = float(sun_dir[1])
        self._sun_v[2] = float(sun_dir[2])
        csm_matrices = self.csm.compute_cascade_matrices(self._cam_pos, self._cam_fwd, self._sun_v)

        # 4. Upload to UBO 0
        self.frame_context.update(
            view_mat=self._view_mat,
            proj_mat=self._proj_mat,
            view_proj_mat=self._vp_mat,
            inv_proj_mat=self._inv_proj,
            inv_view_mat=self._inv_view,
            camera_pos=self._cam_pos,
            time_elapsed=time_elapsed,
            screen_size=(float(w), float(h)),
            sun_dir=sun_dir,
            sun_lux=sun_lux,
            csm_matrices=csm_matrices,
            cascade_splits=self.csm.split_distances,
            fog_density=self.config.fog_density,
            fog_height_falloff=self.config.fog_height_falloff,
        )

        # Prepare MDI batch commands (zero-allocation cached handles)
        self.mdi.begin_frame()
        if draw_batches is not None:
            for mesh_item, count, base_inst in draw_batches:
                if count > 0:
                    alloc = (
                        self.mega_buffer.allocations[mesh_item]
                        if isinstance(mesh_item, str)
                        else mesh_item
                    )
                    self.mdi.add_command(alloc, instance_count=count, base_instance=base_inst)
        else:
            # Batch 1: Ground Box (Entity 0)
            self.mdi.add_command(self._cube_alloc, instance_count=1, base_instance=0)
            # Batch 2: Instanced PBR Spheres (Entities 1 .. active_count - 1)
            if active_count > 1:
                self.mdi.add_command(self._sphere_alloc, instance_count=active_count - 1, base_instance=1)

        # ---- PASS 1: Cascaded Shadow Maps Pass ----
        self.csm.fbo.use()
        self.csm.clear()
        self.ctx.depth_func = "<"  # Shadow maps use standard depth
        self.ctx.enable(moderngl.DEPTH_TEST)

        for c in range(self.config.csm_cascades):
            self.csm.begin_cascade(c)
            if self._csm_cascade_idx_uniform is not None:
                self._csm_cascade_idx_uniform.value = c
            self.mdi.submit(self.csm_vao, self.csm_prog)

        # ---- PASS 2: G-Buffer Pass (Reversed-Z) ----
        self.g_buffer.clear()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.depth_func = ">" if self.config.reverse_z else "<"
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.mdi.submit(self.gbuffer_vao, self.gbuffer_prog)

        # ---- PASS 3: Deferred Lighting & Shadow Resolve ----
        self.post_process.hdr_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.g_buffer.bind_textures(base_unit=0)
        self.csm.depth_texture.use(location=3)

        # Configure resolve uniforms (via pre-cached handles)
        if self._u_pcf_samples is not None:
            self._u_pcf_samples.value = self.config.pcf_samples
        if self._u_sscs_enabled is not None:
            self._u_sscs_enabled.value = 1 if self.config.sscs_enabled else 0
        if self._u_sscs_steps is not None:
            self._u_sscs_steps.value = self.config.sscs_steps
        if self._u_sscs_thickness is not None:
            self._u_sscs_thickness.value = self.config.sscs_thickness
        if self._u_cascade_count is not None:
            self._u_cascade_count.value = self.config.csm_cascades

        self.resolve_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # ---- PASS 4: Post-Process & Tone-Mapping ----
        # Blits to final texture and screen
        self.post_process.render(target_fbo=None)
        if not self.ctx_wrapper.is_headless:
            # Blit final texture to window surface
            self.post_process.render(target_fbo=self.ctx.screen)

    @property
    def output_texture_id(self) -> int:
        """Exposes OpenGL texture handle for ImGui viewport dock."""
        return self.post_process.output_texture_id

    def destroy(self) -> None:
        self.frame_context.destroy()
        self.mega_buffer.destroy()
        self.mdi.destroy()
        self.g_buffer.destroy()
        self.csm.destroy()
        self.post_process.destroy()
        self.ssbo_transforms.release()
        self.ssbo_materials.release()
        self.gbuffer_prog.release()
        self.csm_prog.release()
        self.resolve_prog.release()
        self.gbuffer_vao.release()
        self.csm_vao.release()
        self.resolve_vao.release()
