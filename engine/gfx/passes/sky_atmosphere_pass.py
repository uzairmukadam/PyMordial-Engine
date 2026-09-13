"""Sébastien Hillaire (Eurographics 2020) Physically Based Sky & Atmosphere Pass.

Implements the AAA production atmospheric scattering system:
- Transmittance LUT (256x64): Precomputes optical depth and multi-component extinction.
- Multiple-Scattering LUT (32x32): Precomputes 2nd-order and infinite geometric-series scattering.
- Sky-View LUT (192x108): High-precision real-time sky luminance with non-linear horizon mapping.
"""

from __future__ import annotations
from pathlib import Path
import moderngl

from engine.gfx.render_graph import RenderPass

SHADER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shaders"
ROOT_DIR = SHADER_DIR.parent


def _load_and_preprocess_shader(filename: str) -> str:
    path = SHADER_DIR / filename
    raw = path.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("#include"):
            start_q = line.find('"')
            end_q = line.rfind('"')
            if start_q != -1 and end_q > start_q:
                inc_rel = line[start_q + 1 : end_q]
                inc_path = ROOT_DIR / inc_rel if (ROOT_DIR / inc_rel).is_file() else SHADER_DIR / inc_rel
                inc_content = inc_path.read_text(encoding="utf-8")
                lines.append(inc_content)
            else:
                lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


class SkyAtmospherePass(RenderPass):
    """Manages GPU look-up tables and real-time execution of Sébastien Hillaire's sky model."""

    __slots__ = (
        "ctx",
        "transmittance_w",
        "transmittance_h",
        "multiscat_size",
        "sky_view_w",
        "sky_view_h",
        "transmittance_lut",
        "transmittance_fbo",
        "multiscattering_lut",
        "multiscattering_fbo",
        "sky_view_lut",
        "sky_view_fbo",
        "transmittance_prog",
        "multiscat_prog",
        "sky_view_prog",
        "transmittance_vao",
        "multiscat_vao",
        "sky_view_vao",
        "_luts_dirty",
        "_prev_params",
    )

    def __init__(self, ctx: moderngl.Context) -> None:
        super().__init__(name="SkyAtmospherePass", enabled=True)
        self.ctx = ctx

        # Dimensions adhering to Sébastien Hillaire 2020 specification
        self.transmittance_w = 256
        self.transmittance_h = 64
        self.multiscat_size = 32
        self.sky_view_w = 192
        self.sky_view_h = 108

        # 1. GPU LUT Textures & Framebuffers
        # Transmittance LUT (256x64 RGBA16F)
        self.transmittance_lut = self.ctx.texture(
            (self.transmittance_w, self.transmittance_h),
            components=4,
            dtype="f2",
        )
        self.transmittance_lut.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.transmittance_lut.repeat_x = False
        self.transmittance_lut.repeat_y = False
        self.transmittance_fbo = self.ctx.framebuffer(color_attachments=[self.transmittance_lut])

        # Multiple-Scattering LUT (32x32 RGBA16F)
        self.multiscattering_lut = self.ctx.texture(
            (self.multiscat_size, self.multiscat_size),
            components=4,
            dtype="f2",
        )
        self.multiscattering_lut.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.multiscattering_lut.repeat_x = False
        self.multiscattering_lut.repeat_y = False
        self.multiscattering_fbo = self.ctx.framebuffer(color_attachments=[self.multiscattering_lut])

        # Sky-View LUT (192x108 RGBA16F)
        self.sky_view_lut = self.ctx.texture(
            (self.sky_view_w, self.sky_view_h),
            components=4,
            dtype="f2",
        )
        self.sky_view_lut.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.sky_view_lut.repeat_x = False
        self.sky_view_lut.repeat_y = False
        self.sky_view_fbo = self.ctx.framebuffer(color_attachments=[self.sky_view_lut])

        # 2. Compile Shader Programs
        quad_vert = _load_and_preprocess_shader("fullscreen_quad.vert")
        trans_frag = _load_and_preprocess_shader("hillaire_transmittance.frag")
        multiscat_frag = _load_and_preprocess_shader("hillaire_multiscattering.frag")
        sky_view_frag = _load_and_preprocess_shader("hillaire_sky_view.frag")

        self.transmittance_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=trans_frag)
        self.multiscat_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=multiscat_frag)
        self.sky_view_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=sky_view_frag)

        self.transmittance_vao = self.ctx.vertex_array(self.transmittance_prog, [])
        self.multiscat_vao = self.ctx.vertex_array(self.multiscat_prog, [])
        self.sky_view_vao = self.ctx.vertex_array(self.sky_view_prog, [])

        self._luts_dirty = True
        self._prev_params: dict[str, tuple] = {}

    def mark_dirty(self) -> None:
        """Flags the static Transmittance and Multi-Scattering LUTs for regeneration."""
        self._luts_dirty = True

    def _update_common_uniforms(self, prog: moderngl.Program, atmo_config: any) -> None:
        """Sets physical parameters and scattering coefficients on a shader program."""
        r_bottom = getattr(atmo_config, "planet_radius", 6360000.0)
        r_top = getattr(atmo_config, "atmosphere_radius", 6460000.0)

        u_r_bottom = prog.get("u_RBottom", None)
        if u_r_bottom is not None:
            u_r_bottom.value = float(r_bottom)

        u_r_top = prog.get("u_RTop", None)
        if u_r_top is not None:
            u_r_top.value = float(r_top)

        u_h_r = prog.get("u_H_R", None)
        if u_h_r is not None:
            u_h_r.value = float(getattr(atmo_config, "h_r", 8000.0))

        u_h_m = prog.get("u_H_M", None)
        if u_h_m is not None:
            u_h_m.value = float(getattr(atmo_config, "h_m", 1200.0))

        u_rb = prog.get("u_RayleighBeta", None)
        if u_rb is not None:
            rb = atmo_config.rayleigh_beta
            u_rb.value = (float(rb[0]), float(rb[1]), float(rb[2]))

        u_m_scat = prog.get("u_MieBetaScat", None)
        if u_m_scat is not None:
            u_m_scat.value = float(getattr(atmo_config, "mie_beta_scat", atmo_config.mie_beta * 0.90))

        u_m_ext = prog.get("u_MieBetaExt", None)
        if u_m_ext is not None:
            u_m_ext.value = float(getattr(atmo_config, "mie_beta_ext", atmo_config.mie_beta * 1.0))

        u_mg = prog.get("u_MieG", None)
        if u_mg is not None:
            u_mg.value = float(getattr(atmo_config, "mie_asymmetry", 0.80))

        u_ob = prog.get("u_OzoneBeta", None)
        if u_ob is not None:
            ob = getattr(atmo_config, "ozone_beta", (0.650e-6, 1.881e-6, 0.085e-6))
            u_ob.value = (float(ob[0]), float(ob[1]), float(ob[2]))

        u_ga = prog.get("u_GroundAlbedo", None)
        if u_ga is not None:
            ga = getattr(atmo_config, "ground_color", (0.15, 0.15, 0.15))
            u_ga.value = (float(ga[0]), float(ga[1]), float(ga[2]))

        u_turb = prog.get("u_Turbidity", None)
        if u_turb is not None:
            u_turb.value = float(getattr(atmo_config, "turbidity", 1.0))

    def bake_static_luts(self, atmo_config: any) -> None:
        """Precomputes Transmittance LUT (256x64) and Multi-Scattering LUT (32x32)."""
        prev_fbo = self.ctx.fbo
        prev_viewport = self.ctx.viewport

        # Disable depth testing and blending for clean float write
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # 1. Bake Transmittance LUT
        self.transmittance_fbo.use()
        self.ctx.viewport = (0, 0, self.transmittance_w, self.transmittance_h)
        self._update_common_uniforms(self.transmittance_prog, atmo_config)
        self.transmittance_vao.render(moderngl.TRIANGLES, vertices=3)

        # 2. Bake Multiple-Scattering LUT (samples Transmittance LUT on unit 0)
        self.multiscattering_fbo.use()
        self.ctx.viewport = (0, 0, self.multiscat_size, self.multiscat_size)
        self.transmittance_lut.use(location=0)
        u_trans_sampler = self.multiscat_prog.get("u_TransmittanceLUT", None)
        if u_trans_sampler is not None:
            u_trans_sampler.value = 0

        self._update_common_uniforms(self.multiscat_prog, atmo_config)
        self.multiscat_vao.render(moderngl.TRIANGLES, vertices=3)

        self._luts_dirty = False
        self._restore_fbo(prev_fbo, prev_viewport)

    def _restore_fbo(self, prev_fbo: any, prev_viewport: tuple[int, int, int, int]) -> None:
        if prev_fbo is not None:
            try:
                prev_fbo.use()
                self.ctx.viewport = prev_viewport
                return
            except Exception:
                pass
        if getattr(self.ctx, "screen", None) is not None:
            try:
                self.ctx.screen.use()
            except Exception:
                pass
        self.ctx.viewport = prev_viewport

    def update(
        self,
        atmo_config: any,
        camera_pos: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
        sun_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sun_intensity: float = 4.0,
    ) -> None:
        """Bakes static LUTs if dirty, and evaluates the real-time Sky-View LUT (192x108)."""
        # Detect parameter changes to trigger rebake
        curr_sig = (
            tuple(atmo_config.rayleigh_beta),
            atmo_config.mie_beta,
            atmo_config.turbidity,
            getattr(atmo_config, "ground_color", (0.15, 0.15, 0.15)),
        )
        if curr_sig != self._prev_params.get("sig", None):
            self._prev_params["sig"] = curr_sig
            self._luts_dirty = True

        if self._luts_dirty:
            self.bake_static_luts(atmo_config)

        # Render real-time Sky-View LUT
        prev_fbo = self.ctx.fbo
        prev_viewport = self.ctx.viewport

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self.sky_view_fbo.use()
        self.ctx.viewport = (0, 0, self.sky_view_w, self.sky_view_h)

        self.transmittance_lut.use(location=0)
        self.multiscattering_lut.use(location=1)

        u_trans = self.sky_view_prog.get("u_TransmittanceLUT", None)
        if u_trans is not None:
            u_trans.value = 0
        u_ms = self.sky_view_prog.get("u_MultiScatteringLUT", None)
        if u_ms is not None:
            u_ms.value = 1

        self._update_common_uniforms(self.sky_view_prog, atmo_config)

        # Camera world position relative to planet center (offset by planet radius)
        r_bottom = getattr(atmo_config, "planet_radius", 6360000.0)
        cam_p = (
            float(camera_pos[0]),
            r_bottom + max(1.0, float(camera_pos[1])),
            float(camera_pos[2]),
        )
        u_cam = self.sky_view_prog.get("u_CameraWorldPos", None)
        if u_cam is not None:
            u_cam.value = cam_p

        # Sun direction points towards sun: (-sun_dir[0], -sun_dir[1], -sun_dir[2])
        u_sun = self.sky_view_prog.get("u_SunDirection", None)
        if u_sun is not None:
            u_sun.value = (-float(sun_dir[0]), -float(sun_dir[1]), -float(sun_dir[2]))

        u_col = self.sky_view_prog.get("u_SunColor", None)
        if u_col is not None:
            u_col.value = (float(sun_color[0]), float(sun_color[1]), float(sun_color[2]))

        u_lux = self.sky_view_prog.get("u_SunIntensity", None)
        if u_lux is not None:
            u_lux.value = float(sun_intensity)

        self.sky_view_vao.render(moderngl.TRIANGLES, vertices=3)
        self._restore_fbo(prev_fbo, prev_viewport)

    def execute(self, context: any) -> None:
        """Executes pass within RenderGraph."""
        atmo = getattr(context, "atmosphere", None)
        if atmo is not None:
            self.update(
                atmo_config=atmo.config,
                camera_pos=getattr(context, "camera_pos", (0.0, 10.0, 0.0)),
                sun_dir=getattr(context, "sun_dir", (0.35, -0.85, 0.40)),
                sun_color=getattr(context, "sun_color", (1.0, 1.0, 1.0)),
                sun_intensity=getattr(context, "sun_lux", 4.0),
            )
        if hasattr(context, "resources") and isinstance(context.resources, dict):
            context.resources["transmittance_lut"] = self.transmittance_lut
            context.resources["multiscattering_lut"] = self.multiscattering_lut
            context.resources["sky_view_lut"] = self.sky_view_lut

    def destroy(self) -> None:
        self.transmittance_fbo.release()
        self.transmittance_lut.release()
        self.multiscattering_fbo.release()
        self.multiscattering_lut.release()
        self.sky_view_fbo.release()
        self.sky_view_lut.release()
        self.transmittance_prog.release()
        self.multiscat_prog.release()
        self.sky_view_prog.release()
