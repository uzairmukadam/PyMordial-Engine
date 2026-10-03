"""GPU Compute Particle System & Volumetric VFX Pass.

Simulates and renders tens of thousands of volumetric airborne particles (Dust Motes,
Glowing Embers, Bioluminescent Fireflies) entirely on the GPU with zero CPU per-particle
overhead. Features 3D curl noise turbulence, camera-relative boundary wrapping,
and soft depth-feathering (Z-feathering) against G-Buffer depth.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
import moderngl
import numpy as np

from engine.gfx.render_graph import RenderPass

from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()
ROOT_DIR = SHADER_DIR.parent
_load_shader = load_shader

class ParticleMode(str, Enum):
    DUST_MOTES = "DUST_MOTES"
    EMBERS = "EMBERS"
    FIREFLIES = "FIREFLIES"
    OFF = "OFF"


PARTICLE_DTYPE = np.dtype([
    ("pos_size", np.float32, 4),      # xyz = world pos, w = remaining life (s)
    ("vel_life", np.float32, 4),      # xyz = vel, w = max life (s)
    ("color_alpha", np.float32, 4),   # rgb = tint, a = target opacity
    ("params", np.float32, 4),        # x = radius, y = rot, z = rot_spd, w = mode
])

class ParticleSystemPass(RenderPass):
    """Manages GPU compute simulation and instanced rendering of volumetric particles."""

    # Particle memory layout: 4 x vec4 = 16 floats = 64 bytes per particle
    # struct Particle {
    #     vec4 pos_life;      // xyz = world pos, w = remaining life (s)
    #     vec4 vel_maxlife;   // xyz = vel, w = max life (s)
    #     vec4 color_alpha;   // rgb = base tint, a = target opacity
    #     vec4 params;        // x = radius, y = rot, z = rot_spd, w = mode
    # };
    BYTES_PER_PARTICLE = 64

    MODE_MAP = {
        "DUST_MOTES": 0,
        "EMBERS": 1,
        "FIREFLIES": 2,
        "OFF": -1,
    }

    def __init__(
        self,
        ctx: moderngl.Context,
        max_particles: int = 65536,
        active_count: int = 16384,
        mode: str | ParticleMode = "DUST_MOTES",
    ) -> None:
        super().__init__(name="ParticleSystemPass", enabled=True)
        self.ctx = ctx
        self.max_particles = max(1024, max_particles)
        self.active_count = min(self.max_particles, max(0, active_count))
        self.mode_str = mode.value.upper() if isinstance(mode, ParticleMode) else str(mode).upper()
        self.mode_int = self.MODE_MAP.get(self.mode_str, 0)

        # Particle simulation parameters
        self.emitter_half_size: tuple[float, float, float] = (16.0, 8.0, 16.0)
        self.turbulence_strength: float = 0.85
        self.speed_scale: float = 1.0
        self.soft_particle_radius: float = 0.20
        self.sun_scatter_intensity: float = 2.5
        self.base_size_multiplier: float = 1.0
        self.total_time: float = 0.0

        # Mode-specific gravity / buoyancy defaults
        self._gravity_map = {
            0: (0.0, -0.04, 0.0),    # Dust motes: slow settling
            1: (0.0, 0.75, 0.0),     # Embers: thermal updraft
            2: (0.0, 0.0, 0.0),      # Fireflies: neutral drift
        }

        # 1. Compile Shaders
        comp_src = _load_shader("shaders/particle_simulate.comp")
        self.simulate_prog = self.ctx.compute_shader(comp_src)

        vert_src = _load_shader("shaders/particle_render.vert")
        frag_src = _load_shader("shaders/particle_render.frag")
        self.render_prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # 2. Allocate Particle SSBO and Seed Initial Population
        self.particle_buffer = self.ctx.buffer(reserve=self.max_particles * self.BYTES_PER_PARTICLE)
        self._seed_particles(self.max_particles)
        self.particle_buffer.bind_to_storage_buffer(binding=4)

        # 3. Create Unit Billboard Quad VAO
        # 6 vertices covering [-1, 1] quad
        quad_data = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                 1.0,  1.0,
                -1.0, -1.0,
                 1.0,  1.0,
                -1.0,  1.0,
            ],
            dtype=np.float32,
        )
        self.quad_vbo = self.ctx.buffer(quad_data.tobytes())
        self.quad_vao = self.ctx.vertex_array(self.render_prog, [(self.quad_vbo, "2f", "in_position")])

        # 4. Cache Uniform Locations
        self._u_sim_dt = self.simulate_prog.get("u_DeltaTime", None)
        self._u_sim_emitter_center = self.simulate_prog.get("u_EmitterCenter", None)
        self._u_sim_emitter_half = self.simulate_prog.get("u_EmitterHalfSize", None)
        self._u_sim_count = self.simulate_prog.get("u_ParticleCount", None)
        self._u_sim_mode = self.simulate_prog.get("u_Mode", None)
        self._u_sim_turb = self.simulate_prog.get("u_TurbulenceStrength", None)
        self._u_sim_speed = self.simulate_prog.get("u_SpeedScale", None)
        self._u_sim_grav = self.simulate_prog.get("u_Gravity", None)
        self._u_sim_time = self.simulate_prog.get("u_Time", None)

        self._u_rend_depth = self.render_prog.get("u_DepthTexture", None)
        self._u_rend_soft_radius = self.render_prog.get("u_SoftParticleRadius", None)
        self._u_rend_sun_scatter = self.render_prog.get("u_SunScatterIntensity", None)
        if self._u_rend_depth is not None:
            self._u_rend_depth.value = 0

    @property
    def active_particles(self) -> int:
        """Returns the active particle count."""
        return self.active_count

    @active_particles.setter
    def active_particles(self, val: int) -> None:
        self.set_active_count(val)

    @property
    def mode(self) -> ParticleMode | str:
        """Returns current particle mode enum if valid, or mode string."""
        try:
            return ParticleMode(self.mode_str)
        except ValueError:
            return self.mode_str

    @property
    def ssbo(self) -> moderngl.Buffer:
        """Returns the underlying GPU particle SSBO buffer."""
        return self.particle_buffer

    def _seed_particles(self, count: int) -> None:
        """Seeds random initial particle positions and states in memory."""
        # 16 float32s per particle
        table = np.zeros((count, 16), dtype=np.float32)
        rng = np.random.default_rng(seed=42)

        hx, hy, hz = self.emitter_half_size
        table[:, 0] = rng.uniform(-hx, hx, size=count)
        table[:, 1] = rng.uniform(-hy, hy, size=count)
        table[:, 2] = rng.uniform(-hz, hz, size=count)

        if self.mode_int == 1:
            # Embers
            max_life = rng.uniform(3.0, 7.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4] = rng.uniform(-0.3, 0.3, size=count)
            table[:, 5] = rng.uniform(0.5, 1.2, size=count)
            table[:, 6] = rng.uniform(-0.3, 0.3, size=count)
            table[:, 7] = max_life
            table[:, 8] = 1.0
            table[:, 9] = rng.uniform(0.4, 0.75, size=count)
            table[:, 10] = 0.08
            table[:, 11] = 0.95
            table[:, 12] = rng.uniform(0.02, 0.05, size=count)
        elif self.mode_int == 2:
            # Fireflies
            max_life = rng.uniform(5.0, 10.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4:7] = rng.uniform(-0.25, 0.25, size=(count, 3))
            table[:, 7] = max_life
            table[:, 8] = 0.5
            table[:, 9] = 0.95
            table[:, 10] = 0.25
            table[:, 11] = 0.90
            table[:, 12] = rng.uniform(0.04, 0.07, size=count)
        else:
            # Dust Motes
            max_life = rng.uniform(8.0, 20.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4:7] = rng.uniform(-0.06, 0.06, size=(count, 3))
            table[:, 7] = max_life
            table[:, 8] = 0.95
            table[:, 9] = 0.93
            table[:, 10] = 0.88
            table[:, 11] = rng.uniform(0.5, 0.85, size=count)
            table[:, 12] = rng.uniform(0.015, 0.040, size=count)

        # Rotations
        table[:, 13] = rng.uniform(0.0, 2.0 * np.pi, size=count)
        table[:, 14] = rng.uniform(-0.75, 0.75, size=count)
        table[:, 15] = float(max(0, self.mode_int))

        self.particle_buffer.write(table.tobytes())

    def set_mode(self, mode_name: str | ParticleMode) -> None:
        """Updates particle emitter mode (DUST_MOTES, EMBERS, FIREFLIES, OFF)."""
        mode_upper = mode_name.value.upper() if isinstance(mode_name, ParticleMode) else str(mode_name).upper()
        if mode_upper in self.MODE_MAP:
            new_mode_int = self.MODE_MAP[mode_upper]
            if new_mode_int != self.mode_int:
                self.mode_str = mode_upper
                self.mode_int = new_mode_int
                if self.mode_int >= 0:
                    self._seed_particles(self.active_count)

    def set_active_count(self, count: int) -> None:
        """Sets the active particle simulation count (clamped to max capacity)."""
        new_count = min(self.max_particles, max(0, int(count)))
        if new_count > self.active_count:
            self._seed_particles(new_count)
        self.active_count = new_count

    def update(
        self,
        dt: float,
        camera_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Advances particle simulation via ModernGL compute shader."""
        if self.mode_int < 0 or self.active_count <= 0 or dt <= 0.0:
            return

        self.total_time += dt

        # Bind Particle SSBO
        self.particle_buffer.bind_to_storage_buffer(binding=4)

        # Upload compute uniforms
        if self._u_sim_dt is not None:
            self._u_sim_dt.value = min(0.1, dt)
        if self._u_sim_emitter_center is not None:
            self._u_sim_emitter_center.value = (float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]))
        if self._u_sim_emitter_half is not None:
            self._u_sim_emitter_half.value = self.emitter_half_size
        if self._u_sim_count is not None:
            self._u_sim_count.value = self.active_count
        if self._u_sim_mode is not None:
            self._u_sim_mode.value = self.mode_int
        if self._u_sim_turb is not None:
            self._u_sim_turb.value = self.turbulence_strength
        if self._u_sim_speed is not None:
            self._u_sim_speed.value = self.speed_scale
        if self._u_sim_grav is not None:
            self._u_sim_grav.value = self._gravity_map.get(self.mode_int, (0.0, -0.04, 0.0))
        if self._u_sim_time is not None:
            self._u_sim_time.value = self.total_time

        # Dispatch compute: 64 threads per workgroup
        gx = (self.active_count + 63) // 64
        self.simulate_prog.run(gx, 1, 1)

    def render(
        self,
        hdr_fbo: moderngl.Framebuffer,
        depth_texture: moderngl.Texture,
    ) -> None:
        """Renders instanced billboard particles into the HDR scene color target."""
        if self.mode_int < 0 or self.active_count <= 0:
            return

        hdr_fbo.use()
        depth_texture.use(location=0)
        self.particle_buffer.bind_to_storage_buffer(binding=4)

        # Upload render uniforms
        if self._u_rend_soft_radius is not None:
            self._u_rend_soft_radius.value = self.soft_particle_radius
        if self._u_rend_sun_scatter is not None:
            self._u_rend_sun_scatter.value = self.sun_scatter_intensity

        # Configure blending
        self.ctx.enable(moderngl.BLEND)
        if self.mode_int == 1:
            # Embers: Additive HDR blending
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        else:
            # Dust motes / Fireflies: Alpha blending
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Render instanced unit quads
        self.quad_vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=self.active_count)

        # Restore standard alpha blend func
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.BLEND)

    def execute(self, context: Any) -> None:
        """Executes simulation compute pass and instanced quad render pass via RenderGraphContext."""
        if not self.enabled or self.mode_int < 0 or self.active_count <= 0:
            return
        dt = getattr(context, "delta_time", 0.016)
        cam_pos = getattr(context, "camera_pos", (0.0, 0.0, 0.0))
        self.update(dt=dt, camera_pos=cam_pos)
        resources = getattr(context, "resources", {})
        hdr_fbo = resources.get("hdr_fbo")
        g_buffer = resources.get("g_buffer")
        depth_texture = g_buffer.depth_texture if g_buffer is not None else resources.get("depth_texture")
        if hdr_fbo is not None and depth_texture is not None:
            self.render(hdr_fbo=hdr_fbo, depth_texture=depth_texture)

    def destroy(self) -> None:
        """Releases GPU buffers and VAOs."""
        if self.quad_vao is not None:
            self.quad_vao.release()
            self.quad_vao = None
        if self.quad_vbo is not None:
            self.quad_vbo.release()
            self.quad_vbo = None
        if self.particle_buffer is not None:
            self.particle_buffer.release()
            self.particle_buffer = None
