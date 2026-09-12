"""Froxel-Based Volumetric Fog and Light Scattering Pass for PyMordial Engine.

Implements a 3D frustum voxel (Froxel) atmospheric scattering volume:
1. Injection Pass: Evaluates 4-cascade CSM shadows, Henyey-Greenstein sun scattering,
   ambient sky light, and height-based exponential fog density into a 3D texture.
2. Integration Pass: Raymarches front-to-back along camera frustum rays using the
   Beer-Lambert extinction law to accumulate transmittance and in-scattered radiance.
3. Composite Pass: Trilinearly reconstructs volumetric fog at full resolution against
   Reversed-Z depth and composites onto the HDR scene buffer before tonemapping.
"""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext
from engine.gfx.quality_presets import FROXEL_RESOLUTIONS

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


class VolumetricFogPass(RenderPass):
    """Physically based froxel volumetric lighting and atmospheric fog."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "grid_w",
        "grid_h",
        "grid_d",
        "scatter_ext_vol",
        "integrated_vol",
        "inject_prog",
        "integrate_prog",
        "composite_prog",
        "composite_vao",
        "_u_inj_cascades",
        "_u_inj_near",
        "_u_inj_far",
        "_u_inj_density",
        "_u_inj_height_falloff",
        "_u_inj_anisotropy",
        "_u_inj_ambient",
        "_u_inj_grid_size",
        "_u_inj_point_lights",
        "_u_inj_point_light_count",
        "_u_int_near",
        "_u_int_far",
        "_u_int_grid_size",
        "_u_comp_near",
        "_u_comp_far",
        "_u_comp_grid_depth",
        "_u_comp_debug",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        grid_res: tuple[int, int, int] = (160, 90, 64),
    ) -> None:
        super().__init__(name="VolumetricFogPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.grid_w, self.grid_h, self.grid_d = grid_res

        # 1. Ping-pong 3D volume textures (RGBA16F)
        self.scatter_ext_vol = self.ctx.texture3d((self.grid_w, self.grid_h, self.grid_d), 4, dtype="f2")
        self.scatter_ext_vol.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scatter_ext_vol.repeat_x = False
        self.scatter_ext_vol.repeat_y = False
        self.scatter_ext_vol.repeat_z = False

        self.integrated_vol = self.ctx.texture3d((self.grid_w, self.grid_h, self.grid_d), 4, dtype="f2")
        self.integrated_vol.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.integrated_vol.repeat_x = False
        self.integrated_vol.repeat_y = False
        self.integrated_vol.repeat_z = False

        # 2. Compile Compute and Composite Shaders
        inject_src = (SHADER_DIR / "volumetric_fog_inject.comp").read_text(encoding="utf-8")
        self.inject_prog = self.ctx.compute_shader(inject_src)

        integrate_src = (SHADER_DIR / "volumetric_fog_integrate.comp").read_text(encoding="utf-8")
        self.integrate_prog = self.ctx.compute_shader(integrate_src)

        quad_vert = (SHADER_DIR / "fullscreen_quad.vert").read_text(encoding="utf-8")
        comp_frag = (SHADER_DIR / "volumetric_fog_composite.frag").read_text(encoding="utf-8")
        self.composite_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=comp_frag)
        self.composite_vao = self.ctx.vertex_array(self.composite_prog, [])

        # 3. Cache Uniform Handles
        self._u_inj_cascades = self.inject_prog.get("u_CascadeCount", None)
        self._u_inj_near = self.inject_prog.get("u_FogNear", None)
        self._u_inj_far = self.inject_prog.get("u_FogFar", None)
        self._u_inj_density = self.inject_prog.get("u_FogDensity", None)
        self._u_inj_height_falloff = self.inject_prog.get("u_FogHeightFalloff", None)
        self._u_inj_anisotropy = self.inject_prog.get("u_FogAnisotropy", None)
        self._u_inj_ambient = self.inject_prog.get("u_FogAmbient", None)
        self._u_inj_grid_size = self.inject_prog.get("u_GridSize", None)
        self._u_inj_point_lights = self.inject_prog.get("u_FogPointLights", None)
        self._u_inj_point_light_count = self.inject_prog.get("u_PointLightCount", None)

        self._u_int_near = self.integrate_prog.get("u_FogNear", None)
        self._u_int_far = self.integrate_prog.get("u_FogFar", None)
        self._u_int_grid_size = self.integrate_prog.get("u_GridSize", None)

        self._u_comp_near = self.composite_prog.get("u_FogNear", None)
        self._u_comp_far = self.composite_prog.get("u_FogFar", None)
        self._u_comp_grid_depth = self.composite_prog.get("u_GridDepth", None)
        self._u_comp_debug = self.composite_prog.get("u_DebugMode", None)

    def set_grid_resolution(self, grid_w: int, grid_h: int, grid_d: int) -> None:
        """Dynamically reallocates the 3D froxel textures when resolution changes."""
        if (self.grid_w, self.grid_h, self.grid_d) == (grid_w, grid_h, grid_d):
            return
        self.grid_w, self.grid_h, self.grid_d = grid_w, grid_h, grid_d
        if self.scatter_ext_vol is not None:
            self.scatter_ext_vol.release()
        if self.integrated_vol is not None:
            self.integrated_vol.release()

        self.scatter_ext_vol = self.ctx.texture3d((self.grid_w, self.grid_h, self.grid_d), 4, dtype="f2")
        self.scatter_ext_vol.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scatter_ext_vol.repeat_x = False
        self.scatter_ext_vol.repeat_y = False
        self.scatter_ext_vol.repeat_z = False

        self.integrated_vol = self.ctx.texture3d((self.grid_w, self.grid_h, self.grid_d), 4, dtype="f2")
        self.integrated_vol.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.integrated_vol.repeat_x = False
        self.integrated_vol.repeat_y = False
        self.integrated_vol.repeat_z = False

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def execute(self, context: RenderGraphContext) -> None:
        fog_enabled = bool(getattr(context.config, "volumetric_fog_enabled", True))
        if not fog_enabled:
            return

        csm = context.resources.get("csm", None)
        g_buffer = context.resources.get("g_buffer", None)
        target_fbo = context.resources.get("hdr_fbo", None)
        if csm is None or g_buffer is None or target_fbo is None:
            return

        # Adapt froxel grid resolution to active quality preset or tweak
        res_key = getattr(context.config, "fog_resolution", "HIGH")
        if isinstance(res_key, str) and res_key.upper() in FROXEL_RESOLUTIONS:
            target_res = FROXEL_RESOLUTIONS[res_key.upper()]
            if target_res != (self.grid_w, self.grid_h, self.grid_d):
                self.set_grid_resolution(*target_res)

        fog_near = 0.1
        fog_far = float(getattr(context.config, "fog_distance", 400.0))
        fog_density = float(getattr(context.config, "fog_density", 0.005))
        fog_height_falloff = float(getattr(context.config, "fog_height_falloff", 0.10))
        fog_anisotropy = float(getattr(context.config, "fog_anisotropy", 0.65))
        fog_ambient = float(getattr(context.config, "fog_ambient", 0.35))
        debug_mode = int(getattr(context.config, "fog_debug_mode", 0))

        # ---- PASS 1: Light & Density Injection ----
        self.scatter_ext_vol.bind_to_image(0, read=False, write=True)
        csm.depth_texture.use(location=1)

        # Bind point light SSBO 3 and update local lighting uniforms
        point_light_buf = context.resources.get("point_light_buffer", None)
        point_light_count = int(context.resources.get("point_light_count", 0))
        fog_point_lights = bool(getattr(context.config, "fog_point_lights", True))

        if point_light_buf is not None:
            point_light_buf.bind_to_storage_buffer(3)

        if self._u_inj_point_lights is not None:
            self._u_inj_point_lights.value = fog_point_lights
        if self._u_inj_point_light_count is not None:
            self._u_inj_point_light_count.value = point_light_count

        if self._u_inj_cascades is not None:
            self._u_inj_cascades.value = int(getattr(context.config, "csm_cascades", 4))
        if self._u_inj_near is not None:
            self._u_inj_near.value = fog_near
        if self._u_inj_far is not None:
            self._u_inj_far.value = fog_far
        if self._u_inj_density is not None:
            self._u_inj_density.value = fog_density
        if self._u_inj_height_falloff is not None:
            self._u_inj_height_falloff.value = fog_height_falloff
        if self._u_inj_anisotropy is not None:
            self._u_inj_anisotropy.value = fog_anisotropy
        if self._u_inj_ambient is not None:
            self._u_inj_ambient.value = fog_ambient
        if self._u_inj_grid_size is not None:
            self._u_inj_grid_size.value = (self.grid_w, self.grid_h, self.grid_d)

        # Dispatch injection compute (8x8x1 threads per group)
        gx = (self.grid_w + 7) // 8
        gy = (self.grid_h + 7) // 8
        gz = self.grid_d
        self.inject_prog.run(gx, gy, gz)

        # ---- PASS 2: Front-to-Back Raymarch Integration ----
        self.scatter_ext_vol.bind_to_image(0, read=True, write=False)
        self.integrated_vol.bind_to_image(1, read=False, write=True)

        if self._u_int_near is not None:
            self._u_int_near.value = fog_near
        if self._u_int_far is not None:
            self._u_int_far.value = fog_far
        if self._u_int_grid_size is not None:
            self._u_int_grid_size.value = (self.grid_w, self.grid_h, self.grid_d)

        self.integrate_prog.run(gx, gy, 1)

        # ---- PASS 3: Fullscreen HDR Composite ----
        target_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.SRC_ALPHA

        self.integrated_vol.use(location=0)
        g_buffer.depth_texture.use(location=1)

        if self._u_comp_near is not None:
            self._u_comp_near.value = fog_near
        if self._u_comp_far is not None:
            self._u_comp_far.value = fog_far
        if self._u_comp_grid_depth is not None:
            self._u_comp_grid_depth.value = float(self.grid_d)
        if self._u_comp_debug is not None:
            self._u_comp_debug.value = debug_mode

        self.composite_vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.disable(moderngl.BLEND)

    def destroy(self) -> None:
        self.scatter_ext_vol.release()
        self.integrated_vol.release()
        self.inject_prog.release()
        self.integrate_prog.release()
        self.composite_vao.release()
        self.composite_prog.release()
