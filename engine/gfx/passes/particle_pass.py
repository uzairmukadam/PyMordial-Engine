"""GPU particle simulation and entity-attached visual effects.

Emitters own bounded particle ranges, follow their source entities, and simulate
on the GPU with curl-noise motion and soft depth-feathering.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
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
    SMOKE = "SMOKE"
    OFF = "OFF"


@dataclass(slots=True)
class ParticleEmitter:
    """A fixed-capacity particle effect attached to an ECS entity."""

    entity_id: int
    start: int
    count: int
    mode: str
    offset: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    size_multiplier: float
    turbulence_strength: float
    speed_scale: float
    color: tuple[float, float, float]
    follow_camera: bool = False


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
        "SMOKE": 3,
        "OFF": -1,
    }

    def __init__(
        self,
        ctx: moderngl.Context,
        max_particles: int = 65536,
        active_count: int = 0,
        mode: str | ParticleMode = "OFF",
    ) -> None:
        super().__init__(name="ParticleSystemPass", enabled=True)
        self.ctx = ctx
        self.max_particles = max(1024, max_particles)
        self.active_count = min(self.max_particles, max(0, active_count))
        self.mode_str = mode.value.upper() if isinstance(mode, ParticleMode) else str(mode).upper()
        if self.mode_str not in self.MODE_MAP:
            raise ValueError(f"Unknown particle mode {self.mode_str!r}.")
        self.mode_int = self.MODE_MAP[self.mode_str]

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
            3: (0.0, 0.01, 0.0),     # Smoke: gentle buoyancy
        }

        # 1. Compile Shaders
        comp_src = _load_shader("shaders/particle_simulate.comp")
        self.simulate_prog = self.ctx.compute_shader(comp_src)

        vert_src = _load_shader("shaders/particle_render.vert")
        frag_src = _load_shader("shaders/particle_render.frag")
        self.render_prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # 2. Allocate Particle SSBO and Seed Initial Population
        self.particle_buffer = self.ctx.buffer(reserve=self.max_particles * self.BYTES_PER_PARTICLE)
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
        self._u_sim_offset = self.simulate_prog.get("u_ParticleOffset", None)
        self._u_sim_mode = self.simulate_prog.get("u_Mode", None)
        self._u_sim_turb = self.simulate_prog.get("u_TurbulenceStrength", None)
        self._u_sim_speed = self.simulate_prog.get("u_SpeedScale", None)
        self._u_sim_grav = self.simulate_prog.get("u_Gravity", None)
        self._u_sim_time = self.simulate_prog.get("u_Time", None)
        self._u_sim_color = self.simulate_prog.get("u_EmitterColor", None)

        self._u_rend_depth = self.render_prog.get("u_DepthTexture", None)
        self._u_rend_soft_radius = self.render_prog.get("u_SoftParticleRadius", None)
        self._u_rend_sun_scatter = self.render_prog.get("u_SunScatterIntensity", None)
        self._u_rend_offset = self.render_prog.get("u_ParticleOffset", None)
        self._u_rend_size = self.render_prog.get("u_ParticleSizeMultiplier", None)
        if self._u_rend_depth is not None:
            self._u_rend_depth.value = 0

        self.emitters: list[ParticleEmitter] = []
        self._free_ranges: list[tuple[int, int]] = [(0, self.max_particles)]
        self._legacy_emitter: ParticleEmitter | None = None
        self.entity_manager = None
        if self.active_count > 0 and self.mode_int >= 0:
            self._legacy_emitter = self.add_emitter(
                entity_id=-1,
                mode=self.mode_str,
                count=self.active_count,
                half_extents=self.emitter_half_size,
                turbulence_strength=self.turbulence_strength,
                speed_scale=self.speed_scale,
                _follow_camera=True,
            )

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

    def _seed_emitter(self, emitter: ParticleEmitter, center: tuple[float, float, float]) -> None:
        """Seeds one emitter's contiguous particle range."""
        count = emitter.count
        table = np.zeros((count, 16), dtype=np.float32)
        mode = self.MODE_MAP[emitter.mode]
        rng = np.random.default_rng(
            seed=(emitter.start * 17 + emitter.count * 31 + mode) & 0xFFFFFFFF
        )
        half = np.asarray(emitter.half_extents, dtype=np.float32)
        table[:, 0:3] = np.asarray(center, dtype=np.float32) + rng.uniform(
            -1.0, 1.0, size=(count, 3)
        ) * half

        if mode == self.MODE_MAP["EMBERS"]:
            max_life = rng.uniform(1.5, 3.5, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4] = rng.uniform(-0.2, 0.2, size=count)
            table[:, 5] = rng.uniform(0.6, 1.4, size=count)
            table[:, 6] = rng.uniform(-0.2, 0.2, size=count)
            table[:, 7] = max_life
            table[:, 8] = 1.0
            table[:, 9] = rng.uniform(0.4, 0.75, size=count)
            table[:, 10] = 0.08
            table[:, 11] = 0.95
            table[:, 12] = rng.uniform(0.02, 0.05, size=count) * emitter.size_multiplier
        elif mode == self.MODE_MAP["FIREFLIES"]:
            max_life = rng.uniform(5.0, 10.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4:7] = rng.uniform(-0.25, 0.25, size=(count, 3))
            table[:, 7] = max_life
            table[:, 8:11] = (0.5, 0.95, 0.25)
            table[:, 11] = 0.90
            table[:, 12] = rng.uniform(0.04, 0.07, size=count) * emitter.size_multiplier
        elif mode == self.MODE_MAP["SMOKE"]:
            max_life = rng.uniform(3.0, 6.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4] = rng.uniform(-0.08, 0.08, size=count)
            table[:, 5] = rng.uniform(0.18, 0.42, size=count)
            table[:, 6] = rng.uniform(-0.08, 0.08, size=count)
            table[:, 7] = max_life
            table[:, 8:11] = rng.uniform(0.22, 0.42, size=(count, 3))
            table[:, 11] = rng.uniform(0.18, 0.32, size=count)
            table[:, 12] = rng.uniform(0.055, 0.11, size=count) * emitter.size_multiplier
        else:
            max_life = rng.uniform(8.0, 20.0, size=count)
            table[:, 3] = rng.uniform(0.1, max_life, size=count)
            table[:, 4:7] = rng.uniform(-0.06, 0.06, size=(count, 3))
            table[:, 7] = max_life
            table[:, 8:11] = (0.95, 0.93, 0.88)
            table[:, 11] = rng.uniform(0.5, 0.85, size=count)
            table[:, 12] = rng.uniform(0.015, 0.040, size=count) * emitter.size_multiplier

        table[:, 8:11] *= np.asarray(emitter.color, dtype=np.float32)
        table[:, 13] = rng.uniform(0.0, 2.0 * np.pi, size=count)
        table[:, 14] = rng.uniform(-0.75, 0.75, size=count)
        table[:, 15] = float(mode)
        self.particle_buffer.write(table.tobytes(), offset=emitter.start * self.BYTES_PER_PARTICLE)

    def _allocate_range(self, count: int) -> int:
        for index, (start, length) in enumerate(self._free_ranges):
            if length < count:
                continue
            if length == count:
                self._free_ranges.pop(index)
            else:
                self._free_ranges[index] = (start + count, length - count)
            return start
        available = sum(length for _, length in self._free_ranges)
        raise ValueError(f"Particle capacity exceeded: need {count} slots, but only {available} remain.")

    def _release_range(self, start: int, count: int) -> None:
        self._free_ranges.append((start, count))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for range_start, length in self._free_ranges:
            if merged and merged[-1][0] + merged[-1][1] == range_start:
                prev_start, prev_length = merged[-1]
                merged[-1] = (prev_start, prev_length + length)
            else:
                merged.append((range_start, length))
        self._free_ranges = merged

    def add_emitter(
        self,
        entity_id: int,
        mode: str | ParticleMode,
        count: int = 128,
        *,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        half_extents: tuple[float, float, float] = (0.2, 0.2, 0.2),
        size_multiplier: float = 1.0,
        turbulence_strength: float = 0.2,
        speed_scale: float = 1.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        _follow_camera: bool = False,
    ) -> ParticleEmitter:
        """Attaches a bounded particle effect to an ECS entity."""
        mode_name = mode.value.upper() if isinstance(mode, ParticleMode) else str(mode).upper()
        if mode_name not in self.MODE_MAP or mode_name == "OFF":
            modes = tuple(name for name in self.MODE_MAP if name != "OFF")
            raise ValueError(f"Emitter mode must be one of {modes}.")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Emitter count must be a positive integer.")
        if entity_id != -1 and self.entity_manager is not None and not self.entity_manager.pool.is_valid(entity_id):
            raise KeyError(f"Cannot attach particles to inactive entity {entity_id}.")
        if len(offset) != 3 or len(half_extents) != 3 or len(color) != 3:
            raise ValueError("Emitter offset, half_extents, and color must each have three values.")
        values = (*offset, *half_extents, size_multiplier, turbulence_strength, speed_scale, *color)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Emitter settings must contain only finite values.")
        if (
            any(value < 0.0 for value in half_extents)
            or size_multiplier <= 0.0
            or turbulence_strength < 0.0
            or speed_scale < 0.0
            or any(value < 0.0 for value in color)
        ):
            raise ValueError(
                "Emitter extents, color, turbulence, and speed must be non-negative; size must be positive."
            )
        start = self._allocate_range(int(count))
        emitter = ParticleEmitter(
            entity_id=int(entity_id),
            start=start,
            count=int(count),
            mode=mode_name,
            offset=tuple(float(v) for v in offset),
            half_extents=tuple(float(v) for v in half_extents),
            size_multiplier=float(size_multiplier),
            turbulence_strength=float(turbulence_strength),
            speed_scale=float(speed_scale),
            color=tuple(float(v) for v in color),
            follow_camera=_follow_camera,
        )
        self.emitters.append(emitter)
        self._seed_emitter(emitter, self._emitter_center(emitter, (0.0, 0.0, 0.0)))
        return emitter

    def add_fire(
        self,
        entity_id: int,
        *,
        ember_count: int = 96,
        smoke_count: int = 160,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> tuple[ParticleEmitter, ParticleEmitter]:
        """Adds paired ember and smoke emitters to a fire-source entity."""
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Fire effect scale must be a positive finite number.")
        if not isinstance(ember_count, int) or not isinstance(smoke_count, int) or ember_count <= 0 or smoke_count <= 0:
            raise ValueError("Fire effect particle counts must be positive.")
        available = sum(length for _, length in self._free_ranges)
        if ember_count + smoke_count > available:
            raise ValueError(
                f"Fire effect needs {ember_count + smoke_count} particle slots, but only {available} remain."
            )
        embers = self.add_emitter(
            entity_id,
            ParticleMode.EMBERS,
            ember_count,
            offset=offset,
            half_extents=(0.16 * scale, 0.14 * scale, 0.16 * scale),
            size_multiplier=scale,
            turbulence_strength=0.12,
        )
        smoke = self.add_emitter(
            entity_id,
            ParticleMode.SMOKE,
            smoke_count,
            offset=(offset[0], offset[1] + 0.28 * scale, offset[2]),
            half_extents=(0.32 * scale, 0.8 * scale, 0.32 * scale),
            size_multiplier=scale,
            turbulence_strength=0.32,
        )
        return embers, smoke

    def remove_emitter(self, emitter: ParticleEmitter) -> bool:
        """Removes an emitter and returns its particle capacity to the pool."""
        try:
            self.emitters.remove(emitter)
        except ValueError:
            return False
        if emitter is self._legacy_emitter:
            self._legacy_emitter = None
            self.active_count = 0
        self._release_range(emitter.start, emitter.count)
        return True

    def _emitter_center(
        self,
        emitter: ParticleEmitter,
        camera_pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if emitter.follow_camera:
            position = camera_pos
        elif self.entity_manager is not None:
            dense_idx = self.entity_manager.pool.get_dense_index(emitter.entity_id)
            if dense_idx < 0:
                return emitter.offset
            position = self.entity_manager.rigid_body_state[1, dense_idx, 0:3]
        else:
            position = (0.0, 0.0, 0.0)
        if not emitter.follow_camera and self.entity_manager is not None:
            dense_idx = self.entity_manager.pool.get_dense_index(emitter.entity_id)
            if dense_idx >= 0:
                state = self.entity_manager.rigid_body_state[1, dense_idx]
                qx, qy, qz, qw = (float(value) for value in state[3:7])
                ox, oy, oz = emitter.offset
                tx = 2.0 * (qy * oz - qz * oy)
                ty = 2.0 * (qz * ox - qx * oz)
                tz = 2.0 * (qx * oy - qy * ox)
                offset = (
                    ox + qw * tx + qy * tz - qz * ty,
                    oy + qw * ty + qz * tx - qx * tz,
                    oz + qw * tz + qx * ty - qy * tx,
                )
            else:
                offset = emitter.offset
        else:
            offset = emitter.offset
        return (
            float(position[0]) + offset[0],
            float(position[1]) + offset[1],
            float(position[2]) + offset[2],
        )

    def set_mode(self, mode_name: str | ParticleMode) -> None:
        """Sets the optional legacy camera-centered emitter mode."""
        mode_upper = mode_name.value.upper() if isinstance(mode_name, ParticleMode) else str(mode_name).upper()
        if mode_upper not in self.MODE_MAP:
            raise ValueError(f"Unknown particle mode {mode_upper!r}.")
        self.mode_str = mode_upper
        self.mode_int = self.MODE_MAP[mode_upper]
        if self._legacy_emitter is not None:
            self._legacy_emitter.mode = mode_upper
            if self.mode_int >= 0:
                self._seed_emitter(
                    self._legacy_emitter,
                    self._emitter_center(self._legacy_emitter, (0.0, 0.0, 0.0)),
                )
        elif self.active_count > 0 and self.mode_int >= 0:
            self._legacy_emitter = self.add_emitter(
                entity_id=-1,
                mode=mode_upper,
                count=self.active_count,
                half_extents=self.emitter_half_size,
                turbulence_strength=self.turbulence_strength,
                speed_scale=self.speed_scale,
                _follow_camera=True,
            )

    def set_active_count(self, count: int) -> None:
        """Resizes the optional camera-centered emitter."""
        new_count = min(self.max_particles, max(0, int(count)))
        old_emitter = self._legacy_emitter
        if old_emitter is not None:
            self.remove_emitter(old_emitter)
        self.active_count = new_count
        if self.active_count > 0 and self.mode_int >= 0:
            self._legacy_emitter = self.add_emitter(
                entity_id=-1,
                mode=self.mode_str,
                count=self.active_count,
                half_extents=self.emitter_half_size,
                turbulence_strength=self.turbulence_strength,
                speed_scale=self.speed_scale,
                _follow_camera=True,
            )

    def update(
        self,
        dt: float,
        camera_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Advances all attached particle emitters via GPU compute."""
        if not self.enabled or not self.emitters or dt <= 0.0:
            return

        self.total_time += dt
        self.particle_buffer.bind_to_storage_buffer(binding=4)
        if self._u_sim_dt is not None:
            self._u_sim_dt.value = min(0.1, dt)
        if self._u_sim_time is not None:
            self._u_sim_time.value = self.total_time

        index = 0
        while index < len(self.emitters):
            emitter = self.emitters[index]
            if (
                not emitter.follow_camera
                and self.entity_manager is not None
                and not self.entity_manager.pool.is_valid(emitter.entity_id)
            ):
                self.remove_emitter(emitter)
                continue
            index += 1
            mode = self.MODE_MAP[emitter.mode]
            if mode < 0:
                continue
            center = self._emitter_center(emitter, camera_pos)
            if self._u_sim_emitter_center is not None:
                self._u_sim_emitter_center.value = center
            if self._u_sim_emitter_half is not None:
                self._u_sim_emitter_half.value = emitter.half_extents
            if self._u_sim_count is not None:
                self._u_sim_count.value = emitter.count
            if self._u_sim_offset is not None:
                self._u_sim_offset.value = emitter.start
            if self._u_sim_mode is not None:
                self._u_sim_mode.value = mode
            if self._u_sim_turb is not None:
                self._u_sim_turb.value = emitter.turbulence_strength
            if self._u_sim_speed is not None:
                self._u_sim_speed.value = emitter.speed_scale
            if self._u_sim_grav is not None:
                self._u_sim_grav.value = self._gravity_map.get(mode, (0.0, -0.04, 0.0))
            if self._u_sim_color is not None:
                self._u_sim_color.value = emitter.color
            self.simulate_prog.run((emitter.count + 63) // 64, 1, 1)

    def render(
        self,
        hdr_fbo: moderngl.Framebuffer,
        depth_texture: moderngl.Texture,
    ) -> None:
        """Renders instanced billboard particles into the HDR scene color target."""
        if not self.enabled or not self.emitters:
            return

        hdr_fbo.use()
        depth_texture.use(location=0)
        self.particle_buffer.bind_to_storage_buffer(binding=4)

        # Upload render uniforms
        if self._u_rend_soft_radius is not None:
            self._u_rend_soft_radius.value = self.soft_particle_radius
        if self._u_rend_sun_scatter is not None:
            self._u_rend_sun_scatter.value = self.sun_scatter_intensity

        self.ctx.enable(moderngl.BLEND)
        try:
            for emitter in self.emitters:
                mode = self.MODE_MAP[emitter.mode]
                if mode < 0:
                    continue
                self.ctx.blend_func = (
                    (moderngl.SRC_ALPHA, moderngl.ONE)
                    if mode == self.MODE_MAP["EMBERS"]
                    else (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                )
                if self._u_rend_offset is not None:
                    self._u_rend_offset.value = emitter.start
                if self._u_rend_size is not None:
                    self._u_rend_size.value = emitter.size_multiplier * self.base_size_multiplier
                self.quad_vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=emitter.count)
        finally:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.disable(moderngl.BLEND)

    def execute(self, context: Any) -> None:
        """Executes simulation compute pass and instanced quad render pass via RenderGraphContext."""
        if not self.enabled or not self.emitters:
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
