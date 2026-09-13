"""Dynamic Water Simulation & Screen-Space Refraction Pass.

Coordinates:
- Multi-octave Gerstner wave surface displacement and analytic normal derivation
- Screen-space refraction with depth disparity protection against foreground bleeding
- Physical Beer-Lambert depth absorption and optical water column color extinction
- Cook-Torrance GGX microfacet specular sun glints and Schlick Fresnel reflectance
- Contact foam along shoreline geometry and wave crest foam
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import moderngl
import numpy as np

from engine.gfx.render_graph import RenderPass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SHADER_DIR = ROOT_DIR / "shaders"


def _load_shader(rel_path: str | Path, visited: set[Path] | None = None) -> str:
    """Loads a GLSL shader file and recursively expands #include directives."""
    if visited is None:
        visited = set()
    if isinstance(rel_path, str):
        path = ROOT_DIR / rel_path if (ROOT_DIR / rel_path).is_file() else SHADER_DIR / rel_path
    else:
        path = rel_path
    path = path.resolve()
    if path in visited:
        return ""
    visited.add(path)
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
                inc_content = _load_shader(inc_path, visited)
                lines.append(inc_content)
            else:
                lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _generate_water_grid(
    grid_size: int = 128,
    world_size: float = 160.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a high-tessellation quad grid mesh in the XZ plane centered at (0,0)."""
    n_verts = (grid_size + 1) * (grid_size + 1)
    vertices = np.zeros((n_verts, 3), dtype=np.float32)

    half_size = world_size * 0.5
    step = world_size / grid_size
    idx = 0

    for z_i in range(grid_size + 1):
        z = -half_size + z_i * step
        for x_i in range(grid_size + 1):
            x = -half_size + x_i * step
            vertices[idx] = [x, 0.0, z]
            idx += 1

    indices = np.zeros((grid_size * grid_size * 6,), dtype=np.uint32)
    i_idx = 0
    row_stride = grid_size + 1

    for z_i in range(grid_size):
        for x_i in range(grid_size):
            top_left = z_i * row_stride + x_i
            top_right = top_left + 1
            bottom_left = (z_i + 1) * row_stride + x_i
            bottom_right = bottom_left + 1

            indices[i_idx] = top_left
            indices[i_idx + 1] = bottom_left
            indices[i_idx + 2] = top_right

            indices[i_idx + 3] = top_right
            indices[i_idx + 4] = bottom_left
            indices[i_idx + 5] = bottom_right
            i_idx += 6

    return vertices, indices


class WaterPass(RenderPass):
    """Manages dynamic water simulation, screen-space refraction, and rendering."""

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        grid_size: int = 128,
        world_size: float = 160.0,
    ) -> None:
        super().__init__(name="WaterPass", enabled=True)
        self.ctx = ctx
        self.width = max(1, width)
        self.height = max(1, height)
        self.grid_size = grid_size
        self.world_size = world_size

        # 1. Compile Shaders
        water_vert = _load_shader("water.vert")
        water_frag = _load_shader("water.frag")
        self.prog = ctx.program(vertex_shader=water_vert, fragment_shader=water_frag)

        # 2. Build High-Resolution Water Plane Mesh
        verts, inds = _generate_water_grid(grid_size, world_size)
        self.index_count = len(inds)
        self.vbo = ctx.buffer(verts.tobytes())
        self.ibo = ctx.buffer(inds.tobytes())
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f", "in_position")],
            self.ibo,
        )

        # 3. Intermediate Opaque Scene Copy Texture & FBO
        self.opaque_copy_tex = ctx.texture((self.width, self.height), 4, dtype="f2")
        self.opaque_copy_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.opaque_copy_tex.repeat_x = False
        self.opaque_copy_tex.repeat_y = False
        self.opaque_copy_fbo = ctx.framebuffer(color_attachments=[self.opaque_copy_tex])

        # 4. Cache Uniform Locations
        self._u_model = self.prog.get("u_Model", None)
        self._u_water_height = self.prog.get("u_WaterHeight", None)
        self._u_wave_amplitude = self.prog.get("u_WaveAmplitude", None)
        self._u_wave_speed = self.prog.get("u_WaveSpeed", None)
        self._u_wave_steepness = self.prog.get("u_WaveSteepness", None)
        self._u_time = self.prog.get("u_Time", None)

        self._u_water_color_shallow = self.prog.get("u_WaterColorShallow", None)
        self._u_water_color_deep = self.prog.get("u_WaterColorDeep", None)
        self._u_extinction_coeff = self.prog.get("u_ExtinctionCoeff", None)
        self._u_refraction_strength = self.prog.get("u_RefractionStrength", None)
        self._u_roughness = self.prog.get("u_Roughness", None)
        self._u_foam_color = self.prog.get("u_FoamColor", None)
        self._u_foam_threshold = self.prog.get("u_FoamThreshold", None)
        self._u_foam_scale = self.prog.get("u_FoamScale", None)
        self._u_foam_intensity = self.prog.get("u_FoamIntensity", None)
        self._u_water_clarity = self.prog.get("u_WaterClarity", None)
        self._u_refraction_enabled = self.prog.get("u_RefractionEnabled", None)
        self._u_foam_enabled = self.prog.get("u_FoamEnabled", None)

        # Pre-assign constant texture units
        if "u_OpaqueSceneColor" in self.prog:
            self.prog["u_OpaqueSceneColor"].value = 0
        if "u_DepthTexture" in self.prog:
            self.prog["u_DepthTexture"].value = 1

        self._identity_model = np.eye(4, dtype=np.float32)

    def resize(self, width: int, height: int) -> None:
        """Resizes the intermediate opaque copy framebuffer."""
        w = max(1, width)
        h = max(1, height)
        if w == self.width and h == self.height:
            return
        self.width = w
        self.height = h

        if self.opaque_copy_fbo is not None:
            self.opaque_copy_fbo.release()
        if self.opaque_copy_tex is not None:
            self.opaque_copy_tex.release()

        self.opaque_copy_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.opaque_copy_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.opaque_copy_tex.repeat_x = False
        self.opaque_copy_tex.repeat_y = False
        self.opaque_copy_fbo = self.ctx.framebuffer(color_attachments=[self.opaque_copy_tex])

    def render_water(
        self,
        hdr_fbo: moderngl.Framebuffer,
        depth_texture: moderngl.Texture,
        config: Any,
        time_elapsed: float = 0.0,
    ) -> None:
        """Renders the displaced water plane with screen-space refraction into hdr_fbo."""
        if not self.enabled:
            return

        # 1. Blit current opaque scene from hdr_fbo to opaque_copy_fbo
        self.ctx.copy_framebuffer(self.opaque_copy_fbo, hdr_fbo)

        # 2. Bind textures
        self.opaque_copy_tex.use(location=0)
        depth_texture.use(location=1)

        # 3. Update Wave Simulation Uniforms
        if self._u_model is not None:
            self._u_model.write(self._identity_model.tobytes())
        if self._u_water_height is not None:
            self._u_water_height.value = float(getattr(config, "water_height", 0.0))
        if self._u_wave_amplitude is not None:
            self._u_wave_amplitude.value = float(getattr(config, "water_wave_amplitude", 0.15))
        if self._u_wave_speed is not None:
            self._u_wave_speed.value = float(getattr(config, "water_wave_speed", 1.0))
        if self._u_wave_steepness is not None:
            self._u_wave_steepness.value = float(getattr(config, "water_wave_steepness", 0.8))
        if self._u_time is not None:
            self._u_time.value = float(time_elapsed)

        # 4. Update Optical & Aesthetic Uniforms
        if self._u_water_color_shallow is not None:
            col = getattr(config, "water_color_shallow", (0.05, 0.45, 0.55))
            self._u_water_color_shallow.value = (float(col[0]), float(col[1]), float(col[2]))
        if self._u_water_color_deep is not None:
            col = getattr(config, "water_color_deep", (0.005, 0.04, 0.15))
            self._u_water_color_deep.value = (float(col[0]), float(col[1]), float(col[2]))
        if self._u_extinction_coeff is not None:
            coeff = getattr(config, "water_extinction", (0.35, 0.12, 0.06))
            self._u_extinction_coeff.value = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
        if self._u_refraction_strength is not None:
            self._u_refraction_strength.value = float(getattr(config, "water_refraction_strength", 0.03))
        if self._u_roughness is not None:
            self._u_roughness.value = float(getattr(config, "water_roughness", 0.05))
        if self._u_foam_color is not None:
            fcol = getattr(config, "water_foam_color", (0.92, 0.96, 1.0))
            self._u_foam_color.value = (float(fcol[0]), float(fcol[1]), float(fcol[2]))
        if self._u_foam_threshold is not None:
            self._u_foam_threshold.value = float(getattr(config, "water_foam_threshold", 0.40))
        if self._u_foam_scale is not None:
            self._u_foam_scale.value = float(getattr(config, "water_foam_scale", 6.0))
        if self._u_foam_intensity is not None:
            self._u_foam_intensity.value = float(getattr(config, "water_foam_intensity", 1.0))
        if self._u_water_clarity is not None:
            self._u_water_clarity.value = float(getattr(config, "water_clarity", 4.0))
        if self._u_refraction_enabled is not None:
            self._u_refraction_enabled.value = 1 if getattr(config, "water_refraction_enabled", True) else 0
        if self._u_foam_enabled is not None:
            self._u_foam_enabled.value = 1 if getattr(config, "water_foam_enabled", True) else 0

        # 5. Render water surface directly into hdr_fbo
        hdr_fbo.use()
        self.vao.render(mode=moderngl.TRIANGLES)

    def execute(self, context: Any) -> None:
        """Executes water rendering pass via RenderGraphContext."""
        if not self.enabled:
            return
        resources = getattr(context, "resources", {})
        hdr_fbo = resources.get("hdr_fbo")
        g_buffer = resources.get("g_buffer")
        depth_texture = g_buffer.depth_texture if g_buffer is not None else resources.get("depth_texture")
        config = getattr(context, "config", None)
        time_elapsed = getattr(context, "time_elapsed", 0.0)

        if hdr_fbo is not None and depth_texture is not None and config is not None:
            self.render_water(
                hdr_fbo=hdr_fbo,
                depth_texture=depth_texture,
                config=config,
                time_elapsed=time_elapsed,
            )

    def destroy(self) -> None:
        """Releases all GPU textures, buffers, and framebuffers."""
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None
        if self.opaque_copy_fbo is not None:
            self.opaque_copy_fbo.release()
            self.opaque_copy_fbo = None
        if self.opaque_copy_tex is not None:
            self.opaque_copy_tex.release()
            self.opaque_copy_tex = None
        if self.prog is not None:
            self.prog.release()
            self.prog = None
