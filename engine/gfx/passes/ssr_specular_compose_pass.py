"""Lightweight SSR specular compose pass.

Replaces the second full deferred resolve. The base deferred resolve already
produces the complete lit HDR image without SSR; this pass applies only the
specular delta that SSR introduces, so no lighting term is recomputed and
environment specular is never double-counted.

Resource flow (no read/write feedback):

    base_hdr_texture  (read)  ─┐
    ssr_texture       (read)  ─┼─> ssr_specular_compose.frag ─> hdr_texture (write)
    G-buffer / IBL    (read)  ─┘

The pass is skipped entirely when SSR is disabled; the pipeline then uses the
base HDR result directly.
"""

from __future__ import annotations

import moderngl

from engine.gfx.render_graph import RenderGraphContext, RenderPass
from engine.gfx.shader_utils import load_shader


class SSRSpecularComposePass(RenderPass):
    """Composes SSR into the base HDR result with a single fullscreen pass."""

    name = "SSRSpecularCompose"

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)

        quad_vert = load_shader("fullscreen_quad.vert")
        frag = load_shader("ssr_specular_compose.frag")
        self.prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=frag)
        self.vao = self.ctx.vertex_array(self.prog, [])

        # Fixed texture-unit contract (matches the deferred resolve bindings).
        tex_uniforms = {
            "u_GBufferAlbedoRoughness": 0,
            "u_GBufferNormalMetallic": 1,
            "u_GBufferDepth": 2,
            "u_AOTexture": 4,
            "u_BRDFLUT": 6,
            "u_EnvironmentMap": 7,
            "u_SSRTexture": 8,
            "u_BaseHDR": 14,
        }
        for name, unit in tex_uniforms.items():
            if name in self.prog:
                self.prog[name].value = unit

        self._u_ao_enabled = self.prog.get("u_AOEnabled", None)
        self._u_ibl_enabled = self.prog.get("u_IBLEnabled", None)
        self._u_energy_comp = self.prog.get("u_EnergyCompensationEnabled", None)

    def resize(self, width: int, height: int) -> None:
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)

    def execute(self, context: RenderGraphContext) -> None:
        g_buffer = context.resources.get("g_buffer")
        base_hdr = context.resources.get("base_hdr_texture")
        ssr_tex = context.resources.get("ssr_texture")
        target_fbo = context.resources.get("compose_target_fbo")
        if g_buffer is None or base_hdr is None or ssr_tex is None or target_fbo is None:
            return

        cfg = context.config
        if self._u_ao_enabled is not None:
            self._u_ao_enabled.value = 1 if getattr(cfg, "ao_mode", "OFF") != "OFF" else 0
        if self._u_ibl_enabled is not None:
            self._u_ibl_enabled.value = 1 if getattr(cfg, "ibl_enabled", True) else 0
        if self._u_energy_comp is not None:
            self._u_energy_comp.value = (
                1 if getattr(cfg, "energy_compensation_enabled", True) else 0
            )

        target_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        g_buffer.albedo_roughness_texture.use(location=0)
        g_buffer.normal_metallic_texture.use(location=1)
        g_buffer.depth_texture.use(location=2)
        ao_tex = context.resources.get("ao_texture")
        if ao_tex is not None:
            ao_tex.use(location=4)
        ibl = context.resources.get("ibl_pass")
        if ibl is not None:
            ibl.lut_tex.use(location=6)
            ibl.env_tex.use(location=7)
        ssr_tex.use(location=8)
        base_hdr.use(location=14)

        profiler = getattr(context, "profiler", None)
        if profiler is not None:
            with profiler.region("SSR compose"):
                self.vao.render(moderngl.TRIANGLES, vertices=3)
        else:
            self.vao.render(moderngl.TRIANGLES, vertices=3)

    def destroy(self) -> None:
        self.vao.release()
        self.prog.release()


__all__ = ["SSRSpecularComposePass"]
