"""Screen-Space Displacement Mapping (SSDM) Post-G-Buffer Pass.

Executes between G-Buffer and Ambient Occlusion passes to modify the depth
buffer (gl_FragDepth) and G-Buffer normals based on displacement heightmaps.
This creates true silhouette extrusion and micro-geometry crevices that all
downstream screen-space effects (GTAO, SSGI, SSR) automatically interact with.

Strict Mutual Exclusivity:
Reads G-Buffer RT3 (Displacement Info). If disp_mode != 2, pixels are passed
through completely untouched.
"""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"


def _load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


class SSDMPass(RenderPass):
    """Screen-Space Displacement Mapping pass."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "prog",
        "vao",
        "perturbed_normal_tex",
        "ssdm_fbo",
        "copy_src_fbo",
        "copy_dst_fbo",
        "_u_ssdm_enabled",
        "_u_ssdm_scale",
        "_u_ssdm_max_distance",
        "_u_ssdm_tiling",
    )

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="SSDMPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height

        quad_vert = _load_shader("fullscreen_quad.vert")
        ssdm_frag = _load_shader("ssdm.frag")

        self.prog = ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=ssdm_frag,
        )
        self.vao = ctx.vertex_array(self.prog, [])

        # Assign texture unit bindings
        tex_bindings = {
            "u_GBufferAlbedoRoughness": 0,
            "u_GBufferNormalMetallic": 1,
            "u_GBufferDepth": 2,
            "u_GBufferDispInfo": 3,
            "u_DisplacementArray": 12,
        }
        for name, unit in tex_bindings.items():
            if name in self.prog:
                self.prog[name].value = unit

        # Cache uniform handles
        self._u_ssdm_enabled = self.prog.get("u_SSDMEnabled", None)
        self._u_ssdm_scale = self.prog.get("u_SSDMScale", None)
        self._u_ssdm_max_distance = self.prog.get("u_SSDMMaxDistance", None)
        self._u_ssdm_tiling = self.prog.get("u_SSDMTiling", None)
        self._u_disp_mid_radius = self.prog.get("u_DispMidRadius", None)
        self._u_ssdm_scale_multiplier = self.prog.get("u_SSDMScaleMultiplier", None)
        self._u_material_disp_depth = self.prog.get("u_MaterialDispDepth", None)

        # Scratch normal texture and FBOs to prevent read-write feedback hazards
        self.perturbed_normal_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.perturbed_normal_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.ssdm_fbo: moderngl.Framebuffer | None = None
        self.copy_src_fbo = self.ctx.framebuffer(color_attachments=[self.perturbed_normal_tex])
        self.copy_dst_fbo: moderngl.Framebuffer | None = None

    def resize(self, width: int, height: int) -> None:
        """Resizes textures and framebuffers."""
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        self.perturbed_normal_tex.release()
        self.copy_src_fbo.release()
        if self.ssdm_fbo is not None:
            self.ssdm_fbo.release()
            self.ssdm_fbo = None
        if self.copy_dst_fbo is not None:
            self.copy_dst_fbo.release()
            self.copy_dst_fbo = None

        self.perturbed_normal_tex = self.ctx.texture((self.width, self.height), 4, dtype="f2")
        self.perturbed_normal_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.copy_src_fbo = self.ctx.framebuffer(color_attachments=[self.perturbed_normal_tex])

    def execute(
        self,
        ctx: RenderGraphContext,
        g_buffer,
        displacement_array: moderngl.Texture,
        enabled: bool = True,
        scale: float = 1.0,
        max_distance: float = 30.0,
        tiling: float = 8.0,
        disp_mid_radius: float = 25.0,
        material_depths: np.ndarray | None = None,
    ) -> None:
        """Executes the SSDM pass."""
        if not enabled:
            return

        # Ensure FBOs are set up with current G-Buffer attachments
        if self.ssdm_fbo is None:
            self.ssdm_fbo = self.ctx.framebuffer(
                color_attachments=[self.perturbed_normal_tex],
                depth_attachment=g_buffer.depth_texture,
            )
        if self.copy_dst_fbo is None:
            self.copy_dst_fbo = self.ctx.framebuffer(
                color_attachments=[g_buffer.rt_normal_metallic],
            )

        self.ssdm_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        try:
            self.ctx.depth_func = "1"
        except Exception:
            pass

        # Bind inputs
        g_buffer.rt_albedo_roughness.use(location=0)
        g_buffer.rt_normal_metallic.use(location=1)
        g_buffer.depth_texture.use(location=2)
        g_buffer.rt_displacement.use(location=3)
        displacement_array.use(location=12)

        # Set uniforms
        if self._u_ssdm_enabled is not None:
            self._u_ssdm_enabled.value = 1 if enabled else 0
        if self._u_ssdm_scale is not None:
            self._u_ssdm_scale.value = scale
        if self._u_ssdm_scale_multiplier is not None:
            self._u_ssdm_scale_multiplier.value = scale
        if self._u_ssdm_max_distance is not None:
            self._u_ssdm_max_distance.value = max_distance
        if self._u_ssdm_tiling is not None:
            self._u_ssdm_tiling.value = tiling
        if self._u_disp_mid_radius is not None:
            self._u_disp_mid_radius.value = disp_mid_radius
        if material_depths is not None and self._u_material_disp_depth is not None:
            self._u_material_disp_depth.write(material_depths.tobytes())

        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)

        # Copy perturbed normals back into G-Buffer RT1
        self.ctx.copy_framebuffer(self.copy_dst_fbo, self.copy_src_fbo)

    def destroy(self) -> None:
        self.perturbed_normal_tex.release()
        self.copy_src_fbo.release()
        if self.ssdm_fbo is not None:
            self.ssdm_fbo.release()
        if self.copy_dst_fbo is not None:
            self.copy_dst_fbo.release()
        self.prog.release()
        self.vao.release()
