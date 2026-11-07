"""HDR environment lighting and split-sum GGX image-based lighting."""

from __future__ import annotations

import math

import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext
from engine.gfx.shader_utils import load_shader


class IBLPass(RenderPass):
    """Builds a dynamic HDR sky, diffuse irradiance, and GGX-prefiltered radiance."""

    def __init__(self, ctx: moderngl.Context) -> None:
        super().__init__(name="IBLPass", enabled=True)
        self.ctx = ctx
        self._last_sun_dir: tuple[float, float, float] | None = None

        self.lut_tex = self.ctx.texture((256, 256), 2, dtype="f2")
        self.lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lut_tex.repeat_x = False
        self.lut_tex.repeat_y = False
        self._generate_brdf_lut()

        self.w, self.h = 512, 256
        self.max_mip = int(math.floor(math.log2(max(self.w, self.h))))
        self.mip_count = self.max_mip + 1

        self.raw_env_tex = self.ctx.texture((self.w, self.h), 4, dtype="f2")
        self.raw_env_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.raw_env_tex.repeat_x = True
        self.raw_env_tex.repeat_y = False

        self.env_tex = self.ctx.texture((self.w, self.h), 4, dtype="f2")
        self.env_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.env_tex.repeat_x = True
        self.env_tex.repeat_y = False
        self.env_tex.build_mipmaps()

        self.irradiance_tex = self.ctx.texture((128, 64), 4, dtype="f2")
        self.irradiance_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.irradiance_tex.repeat_x = True
        self.irradiance_tex.repeat_y = False

        sky_vert = """
            #version 330 core
            out vec2 uv;
            void main() {
                uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """
        sky_frag = """
            #version 330 core
            in vec2 uv;
            out vec4 fragColor;
            uniform vec3 u_sunDir;

            const float PI = 3.14159265359;

            void main() {
                float longitude = (uv.x - 0.5) * 2.0 * PI;
                float latitude = (uv.y - 0.5) * PI;
                float cosLatitude = cos(latitude);
                vec3 rayDir = normalize(vec3(cosLatitude * cos(longitude),
                                             sin(latitude),
                                             cosLatitude * sin(longitude)));
                vec3 sunDir = normalize(-u_sunDir);
                float sunCos = clamp(dot(rayDir, sunDir), -1.0, 1.0);
                float sunElevation = max(sunDir.y, -0.15);

                if (rayDir.y < -0.02) {
                    vec3 ground = vec3(0.13, 0.12, 0.10);
                    vec3 horizon = vec3(0.18, 0.20, 0.22);
                    fragColor = vec4(mix(horizon, ground, clamp(-rayDir.y * 8.0, 0.0, 1.0)), 1.0);
                    return;
                }

                float zenithCos = max(rayDir.y, 0.025);
                float airMass = 1.0 / (zenithCos + 0.15 * pow(max(93.885 - degrees(acos(zenithCos)), 0.0), -1.253));
                float rayleighPhase = 0.75 * (1.0 + sunCos * sunCos);
                const float mieG = 0.76;
                float mieDenom = max(1.0 + mieG * mieG - 2.0 * mieG * sunCos, 0.001);
                float miePhase = 1.5 * ((1.0 - mieG * mieG) / (2.0 + mieG * mieG))
                               * (1.0 + sunCos * sunCos) / pow(mieDenom, 1.5);

                vec3 betaRayleigh = vec3(0.0058, 0.0135, 0.0331);
                vec3 betaMie = vec3(0.0030);
                vec3 transmittance = exp(-(betaRayleigh + betaMie) * airMass);
                float daylight = smoothstep(-0.12, 0.08, sunElevation);
                vec3 sky = (betaRayleigh * rayleighPhase + betaMie * miePhase)
                         * transmittance * (18.0 * daylight);
                vec3 twilight = vec3(0.055, 0.025, 0.018)
                              * exp(-abs(sunElevation) * 12.0)
                              * pow(max(1.0 - rayDir.y, 0.0), 2.0);
                sky += twilight;

                float sunDisc = smoothstep(0.99994, 0.99999, sunCos);
                float sunGlow = pow(max(sunCos, 0.0), 512.0) * 0.12
                              + pow(max(sunCos, 0.0), 48.0) * 0.015;
                sky += vec3(24.0, 21.5, 17.0) * (sunDisc + sunGlow) * daylight;

                fragColor = vec4(max(sky, vec3(0.0)), 1.0);
            }
        """

        self._sky_prog = self.ctx.program(vertex_shader=sky_vert, fragment_shader=sky_frag)
        self._sky_vao = self.ctx.vertex_array(self._sky_prog, [])
        self._sky_fbo = self.ctx.framebuffer(color_attachments=[self.raw_env_tex])

        prefilter_src = load_shader("ibl_prefilter.comp")
        self._prefilter_prog = self.ctx.compute_shader(prefilter_src)
        self._prefilter_prog["u_EnvironmentSource"].value = 0

        irradiance_src = load_shader("ibl_irradiance.comp")
        self._irradiance_prog = self.ctx.compute_shader(irradiance_src)
        self._irradiance_prog["u_EnvironmentSource"].value = 0

        self._update_procedural_env_map((0.35, -0.75, -0.55))

    def _generate_brdf_lut(self) -> None:
        quad_vert = load_shader("fullscreen_quad.vert")
        brdf_frag = load_shader("brdf_lut.frag")
        prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=brdf_frag)
        vao = self.ctx.vertex_array(prog, [])
        fbo = self.ctx.framebuffer(color_attachments=[self.lut_tex])

        previous_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, 256, 256)
        fbo.use()
        vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.viewport = previous_viewport

        fbo.release()
        vao.release()
        prog.release()

    def _prefilter_environment(self) -> None:
        self.raw_env_tex.use(location=0)
        for level in range(self.mip_count):
            mip_width = max(self.w >> level, 1)
            mip_height = max(self.h >> level, 1)
            self.env_tex.bind_to_image(0, read=False, write=True, level=level)
            self._prefilter_prog["u_MipLevel"].value = level
            self._prefilter_prog["u_MaxSourceMip"].value = self.max_mip
            self._prefilter_prog["u_Roughness"].value = level / self.max_mip
            self._prefilter_prog["u_OutputSize"].value = (float(mip_width), float(mip_height))
            self._prefilter_prog.run(
                group_x=(mip_width + 7) // 8,
                group_y=(mip_height + 7) // 8,
                group_z=1,
            )
            self.ctx.memory_barrier()

        self.irradiance_tex.bind_to_image(0, read=False, write=True)
        self._irradiance_prog["u_OutputSize"].value = (128.0, 64.0)
        self._irradiance_prog.run(group_x=16, group_y=8, group_z=1)
        self.ctx.memory_barrier()

    def _update_procedural_env_map(self, sun_dir: tuple[float, float, float]) -> None:
        """Renders the linear HDR sky and rebuilds its diffuse/specular convolutions."""
        previous_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, self.w, self.h)
        self._sky_fbo.use()
        self._sky_prog["u_sunDir"].value = sun_dir
        self._sky_vao.render(moderngl.TRIANGLES, vertices=3)
        self.raw_env_tex.build_mipmaps()
        self.ctx.memory_barrier()
        self._prefilter_environment()
        self.ctx.viewport = previous_viewport
        self._last_sun_dir = sun_dir

    def execute(self, context: RenderGraphContext) -> None:
        sun_dir = getattr(context, "sun_dir", None)
        if sun_dir is not None:
            if self._last_sun_dir is None:
                self._update_procedural_env_map(sun_dir)
            else:
                dx = sun_dir[0] - self._last_sun_dir[0]
                dy = sun_dir[1] - self._last_sun_dir[1]
                dz = sun_dir[2] - self._last_sun_dir[2]
                if dx * dx + dy * dy + dz * dz > 0.0002:
                    self._update_procedural_env_map(sun_dir)

        context.resources["ibl_lut"] = self.lut_tex
        context.resources["ibl_env_map"] = self.env_tex
        context.resources["ibl_irradiance"] = self.irradiance_tex

    def destroy(self) -> None:
        self.lut_tex.release()
        self.raw_env_tex.release()
        self.env_tex.release()
        self.irradiance_tex.release()
        self._sky_fbo.release()
        self._sky_vao.release()
        self._sky_prog.release()
        self._prefilter_prog.release()
        self._irradiance_prog.release()
