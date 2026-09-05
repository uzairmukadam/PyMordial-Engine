"""Unified Master Render Graph Coordinator for PyMordial Engine."""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import moderngl

from engine.core.ecs import EntityManager
from engine.core.math_utils import matrix_perspective, matrix_look_at, mat4_mul, mat4_inv
from engine.gfx.context import RenderContext
from engine.gfx.quality_presets import RenderConfig, get_quality_preset, GraphicsQuality
from engine.gfx.frame_context import FrameContext
from engine.gfx.mega_buffer import MegaBuffer
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

        # 6. Compile Shaders
        quad_vert = _load_shader("fullscreen_quad.vert")
        gbuffer_vert = _load_shader("gbuffer.vert")
        gbuffer_frag = _load_shader("gbuffer.frag")
        csm_vert = _load_shader("csm_depth.vert")
        csm_frag = _load_shader("csm_depth.frag")
        resolve_frag = _load_shader("deferred_resolve.frag")
        post_frag = _load_shader("post_process.frag")

        self.gbuffer_prog = self.ctx.program(
            vertex_shader=gbuffer_vert,
            fragment_shader=gbuffer_frag,
        )
        self.csm_prog = self.ctx.program(
            vertex_shader=csm_vert,
            fragment_shader=csm_frag,
        )
        self.resolve_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=resolve_frag,
        )

        # Vertex Arrays
        self.gbuffer_vao = self.mega_buffer.get_vao(self.gbuffer_prog)
        self.csm_vao = self.mega_buffer.get_vao(self.csm_prog)
        self.resolve_vao = self.ctx.vertex_array(self.resolve_prog, [])

        # 7. Initialize Post-Processing
        self.post_process = PostProcessPipeline(
            self.ctx,
            self.ctx_wrapper.width,
            self.ctx_wrapper.height,
            self.config,
            post_process_glsl=post_frag,
            quad_vert_glsl=quad_vert,
        )

    def apply_config(self, new_config: RenderConfig) -> None:
        """Applies dynamic graphics quality configuration changes."""
        self.config = new_config
        self.post_process.config = new_config
        self.ctx_wrapper.config = new_config
        self.ctx_wrapper.ctx.depth_func = ">" if new_config.reverse_z else "<"

    def render_frame(
        self,
        ecs: EntityManager,
        camera_pos: tuple[float, float, float] | np.ndarray = (0.0, 5.0, 10.0),
        camera_target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        time_elapsed: float = 0.0,
        sun_dir: tuple[float, float, float] = (0.35, -0.85, 0.40),
        sun_lux: float = 4.0,
        fovy_deg: float = 60.0,
    ) -> None:
        """Executes full 4-pass deferred render pipeline."""
        active_count = ecs.active_count
        if active_count <= 0:
            return

        w, h = self.ctx_wrapper.width, self.ctx_wrapper.height
        aspect = w / (h if h > 0 else 1)

        # 1. Update SSBO 1 & SSBO 2 from ECS contiguous memory tables
        transforms_bytes = ecs.get_active_transforms_view().tobytes()
        self.ssbo_transforms.write(transforms_bytes)

        materials_bytes = ecs.get_active_materials_view().tobytes()
        self.ssbo_materials.write(materials_bytes)

        # 2. Camera Matrices (Reversed-Z)
        cam_p = np.asarray(camera_pos, dtype=np.float32)
        cam_t = np.asarray(camera_target, dtype=np.float32)
        view_mat = matrix_look_at(cam_p, cam_t, up=(0.0, 1.0, 0.0))
        proj_mat = matrix_perspective(
            math.radians(fovy_deg),
            aspect,
            near=0.1,
            far=self.config.shadow_distance * 2.0,
            reverse_z=self.config.reverse_z,
        )
        vp_mat = mat4_mul(proj_mat, view_mat)
        inv_proj = mat4_inv(proj_mat)
        inv_view = mat4_inv(view_mat)

        cam_fwd = cam_t - cam_p
        cam_fwd /= (np.linalg.norm(cam_fwd) + 1e-6)

        # 3. Cascaded Shadow Maps Matrices
        sun_v = np.asarray(sun_dir, dtype=np.float32)
        csm_matrices = self.csm.compute_cascade_matrices(cam_p, cam_fwd, sun_v)

        # 4. Upload to UBO 0
        self.frame_context.update(
            view_mat=view_mat,
            proj_mat=proj_mat,
            view_proj_mat=vp_mat,
            inv_proj_mat=inv_proj,
            inv_view_mat=inv_view,
            camera_pos=cam_p,
            time_elapsed=time_elapsed,
            screen_size=(float(w), float(h)),
            sun_dir=sun_dir,
            sun_lux=sun_lux,
            csm_matrices=csm_matrices,
            cascade_splits=self.csm.split_distances,
            fog_density=self.config.fog_density,
            fog_height_falloff=self.config.fog_height_falloff,
        )

        # Prepare MDI batch commands
        cube_alloc = self.mega_buffer.allocations["cube"]
        plane_alloc = self.mega_buffer.allocations["plane"]
        sphere_alloc = self.mega_buffer.allocations["sphere"]

        self.mdi.begin_frame()
        # Batch 1: Ground Box (Entity 0)
        self.mdi.add_command(cube_alloc, instance_count=1, base_instance=0)
        # Batch 2: Instanced PBR Spheres (Entities 1 .. active_count - 1)
        if active_count > 1:
            self.mdi.add_command(sphere_alloc, instance_count=active_count - 1, base_instance=1)

        # ---- PASS 1: Cascaded Shadow Maps Pass ----
        self.csm.fbo.use()
        self.csm.clear()
        self.ctx.depth_func = "<"  # Shadow maps use standard depth
        self.ctx.enable(moderngl.DEPTH_TEST)

        for c in range(self.config.csm_cascades):
            self.csm.begin_cascade(c)
            if "u_CascadeIndex" in self.csm_prog:
                self.csm_prog["u_CascadeIndex"].value = c
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

        # Configure resolve uniforms
        if "u_PCF_Samples" in self.resolve_prog:
            self.resolve_prog["u_PCF_Samples"].value = self.config.pcf_samples
        if "u_SSCS_Enabled" in self.resolve_prog:
            self.resolve_prog["u_SSCS_Enabled"].value = 1 if self.config.sscs_enabled else 0
        if "u_SSCS_Steps" in self.resolve_prog:
            self.resolve_prog["u_SSCS_Steps"].value = self.config.sscs_steps
        if "u_SSCS_Thickness" in self.resolve_prog:
            self.resolve_prog["u_SSCS_Thickness"].value = self.config.sscs_thickness
        if "u_CascadeCount" in self.resolve_prog:
            self.resolve_prog["u_CascadeCount"].value = self.config.csm_cascades

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
