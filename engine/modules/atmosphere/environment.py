"""Atmosphere Environment Provider.

Implements the ``EnvironmentProvider`` contract from ``engine.gfx.environment``
by owning the sky-atmosphere and cloud passes plus the resolve-shader uniform
upload. The render pipeline consumes this provider and never reaches into the
atmosphere configuration directly.
"""

from __future__ import annotations

from typing import Any

import moderngl

from engine.gfx.passes.sky_atmosphere_pass import SkyAtmospherePass
from engine.gfx.passes.sky_cloud_pass import SkyCloudPass
from engine.modules.atmosphere.config import AtmosphereConfig
from engine.modules.atmosphere.system import AtmosphereSystem


class AtmosphereEnvironment:
    """Physically based sky, clouds, and lighting for the render pipeline."""

    __slots__ = (
        "ctx",
        "enabled",
        "config",
        "system",
        "sky_pass",
        "cloud_pass",
        "_prev_atmo_key",
        "_atmo_sun_tuple",
        "_atmo_moon_tuple",
    )

    def __init__(self, ctx: moderngl.Context, config: AtmosphereConfig | None = None) -> None:
        self.ctx = ctx
        self.enabled = True
        self.config = config if config is not None else AtmosphereConfig()
        self.system = AtmosphereSystem(self.config)
        self.sky_pass = SkyAtmospherePass(ctx)
        self.cloud_pass = SkyCloudPass(ctx)
        self._prev_atmo_key: tuple | None = None
        self._atmo_sun_tuple: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._atmo_moon_tuple: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---------------- Simulation ----------------

    def update(self, dt: float) -> None:
        self.system.update(dt)

    def compute_sun_vector(self) -> tuple[float, float, float]:
        return self.system.compute_sun_vector()

    def compute_moon_vector(self) -> tuple[float, float, float]:
        return self.system.compute_moon_vector()

    @property
    def moon_phase(self) -> float:
        """Illuminated fraction of the moon: 0.0 at new moon, 1.0 at full moon."""
        return self.system.moon_phase

    def compute_lighting(self) -> tuple[
        tuple[float, float, float], float, tuple[float, float, float], float
    ]:
        return self.system.compute_lighting()

    def apply_preset(self, preset_name: str) -> None:
        self.system.apply_preset(preset_name)

    # ---------------- GPU ----------------

    @property
    def sky_view_lut(self) -> moderngl.Texture:
        return self.sky_pass.sky_view_lut

    @property
    def transmittance_lut(self) -> moderngl.Texture:
        return self.sky_pass.transmittance_lut

    @property
    def multiscattering_lut(self) -> moderngl.Texture:
        return self.sky_pass.multiscattering_lut

    def render_sky(
        self,
        camera_pos: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
        sun_color: tuple[float, float, float],
        sun_intensity: float,
    ) -> None:
        self.sky_pass.update(
            atmo_config=self.config,
            camera_pos=camera_pos,
            sun_dir=sun_dir,
            sun_color=sun_color,
            sun_intensity=sun_intensity,
        )

    def render_clouds(
        self,
        hdr_fbo: Any,
        depth_texture: Any,
        inverse_projection: Any,
        inverse_view: Any,
        camera_position: tuple[float, float, float],
        sun_direction: tuple[float, float, float],
        sun_color: tuple[float, float, float],
        time_elapsed: float,
    ) -> None:
        self.cloud_pass.render(
            hdr_fbo=hdr_fbo,
            depth_texture=depth_texture,
            inverse_projection=inverse_projection,
            inverse_view=inverse_view,
            camera_position=camera_position,
            sun_direction=sun_direction,
            sun_color=sun_color,
            moon_color=self.config.moon_color,
            moon_intensity=self.config.moon_intensity,
            time_elapsed=time_elapsed,
            cloud_config=self.config,
        )

    def apply_resolve_uniforms(self, program: moderngl.Program) -> None:
        """Uploads every atmosphere uniform the deferred resolve shader needs.

        Uniform writes are cached against a configuration signature so the
        per-frame cost is a single tuple comparison when nothing changed.
        """
        atmo = self.config
        atmo_key = (
            atmo.rayleigh_beta,
            atmo.mie_beta,
            atmo.mie_asymmetry,
            atmo.ozone_beta,
            atmo.turbidity,
            atmo.planet_radius,
            atmo.atmosphere_radius,
            atmo.h_r,
            atmo.h_m,
            atmo.ozone_center,
            atmo.ozone_width,
            atmo.sun_disc_size,
            atmo.moon_disc_size,
            atmo.moon_color,
            atmo.star_intensity,
            atmo.star_density,
            atmo.ground_color,
            atmo.night_zenith,
            atmo.night_horizon,
            atmo.sky_mode,
            atmo.space_zenith,
            atmo.nebula_color,
            atmo.nebula_intensity,
            atmo.moon_intensity,
            atmo.sky_exposure,
            atmo.sky_saturation,
        )
        if atmo_key == self._prev_atmo_key:
            return
        self._prev_atmo_key = atmo_key

        self.sky_pass._update_common_uniforms(program, atmo)

        def _set(name: str, value: Any) -> None:
            uniform = program.get(name, None)
            if uniform is not None:
                uniform.value = value

        _set("u_RayleighBeta", atmo.rayleigh_beta)
        mie = float(atmo.mie_beta)
        _set("u_MieBeta", (mie, mie, mie))
        _set("u_MieG", atmo.mie_asymmetry)
        _set("u_OzoneBeta", atmo.ozone_beta)
        _set("u_AtmosphereGroundColor", atmo.ground_color)
        _set("u_NightZenithColor", atmo.night_zenith)
        _set("u_NightHorizonColor", atmo.night_horizon)
        _set("u_SkyMode", 1 if atmo.sky_mode.upper() == "SPACE" else 0)
        _set("u_SpaceZenithColor", atmo.space_zenith)
        _set("u_NebulaColor", atmo.nebula_color)
        _set("u_NebulaIntensity", atmo.nebula_intensity)
        _set("u_SunDiscSize", atmo.sun_disc_size)
        _set("u_MoonDiscSize", atmo.moon_disc_size)
        _set("u_MoonColor", atmo.moon_color)
        _set("u_MoonIntensity", atmo.moon_intensity)
        _set("u_StarIntensity", atmo.star_intensity)
        _set("u_StarDensity", atmo.star_density)
        _set("u_AtmosphereTurbidity", atmo.turbidity)
        _set("u_SkyExposure", atmo.sky_exposure)
        _set("u_SkySaturation", atmo.sky_saturation)

    def apply_sun_direction(self, program: moderngl.Program, sun_dir: tuple[float, float, float]) -> None:
        """Uploads the celestial directions (sun + moon), skipping redundant writes.

        The moon is derived from its own hour angle, so it is uploaded alongside
        the sun to keep the sky disc and the lighting in lockstep.
        """
        sx, sy, sz = float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])
        if (sx, sy, sz) != self._atmo_sun_tuple:
            self._atmo_sun_tuple = (sx, sy, sz)
            uniform = program.get("u_AtmosphereSunDir", None)
            if uniform is not None:
                uniform.value = self._atmo_sun_tuple
        self.apply_moon_direction(program, self.system.compute_moon_vector())

    def apply_moon_direction(self, program: moderngl.Program, moon_dir: tuple[float, float, float]) -> None:
        """Uploads the atmosphere moon direction, skipping redundant writes."""
        mx, my, mz = float(moon_dir[0]), float(moon_dir[1]), float(moon_dir[2])
        if (mx, my, mz) == self._atmo_moon_tuple:
            return
        self._atmo_moon_tuple = (mx, my, mz)
        uniform = program.get("u_AtmosphereMoonDir", None)
        if uniform is not None:
            uniform.value = self._atmo_moon_tuple

    def bind_luts(self, sky_view_unit: int, transmittance_unit: int) -> None:
        self.sky_pass.sky_view_lut.use(location=sky_view_unit)
        self.sky_pass.transmittance_lut.use(location=transmittance_unit)

    def destroy(self) -> None:
        self.cloud_pass.destroy()
        self.sky_pass.destroy()


__all__ = ["AtmosphereEnvironment"]
