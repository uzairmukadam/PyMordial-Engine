"""Environment Provider Contract for PyMordial Engine.

The renderer needs sky lighting, a sun direction, and sky LUTs, but it must not
own a particular atmosphere implementation. This module defines the narrow
contract the pipeline consumes, plus a flat fallback used when no environment
module is enabled.

An environment provider is supplied by an engine module (for example
``pymordial.atmosphere``) and injected into ``RenderPipeline``. The pipeline
never imports a specific implementation at module scope; the default provider is
resolved lazily so a disabled feature is never imported during startup.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import moderngl


@runtime_checkable
class EnvironmentProvider(Protocol):
    """The environment surface consumed by ``RenderPipeline``."""

    #: Whether the provider contributes sky rendering at all.
    enabled: bool

    #: Opaque provider configuration (owned by the provider, not the pipeline).
    config: Any

    def update(self, dt: float) -> None:
        """Advances any time-of-day or animated state."""

    def compute_sun_vector(self) -> tuple[float, float, float]:
        """Returns the normalized light direction (pointing from light into the scene)."""

    def compute_lighting(self) -> tuple[
        tuple[float, float, float], float, tuple[float, float, float], float
    ]:
        """Returns ``(light_dir, lux, light_color, ambient_factor)``."""

    def render_sky(
        self,
        camera_pos: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
        sun_color: tuple[float, float, float],
        sun_intensity: float,
    ) -> None:
        """Bakes/refreshes sky LUTs for this frame."""

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
        """Composites any cloud deck over sky pixels."""

    def apply_resolve_uniforms(self, program: moderngl.Program) -> None:
        """Uploads every environment uniform the deferred resolve shader needs."""

    def bind_luts(self, sky_view_unit: int, transmittance_unit: int) -> None:
        """Binds the sky-view and transmittance LUTs to the given texture units."""

    def destroy(self) -> None:
        """Releases GPU resources owned by the provider."""


class FlatEnvironment:
    """Minimal environment used when no atmosphere module is enabled.

    Provides a constant sky color and neutral lighting so the deferred resolve
    still has valid LUT bindings. This is a documented fallback, not a silent
    substitute for a missing required module.
    """

    __slots__ = (
        "ctx",
        "enabled",
        "config",
        "sky_color",
        "sun_direction",
        "sun_lux",
        "sun_color",
        "ambient",
        "_sky_view_lut",
        "_transmittance_lut",
        "_multiscattering_lut",
        "_atmo_sun_tuple",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        sky_color: tuple[float, float, float] = (0.42, 0.55, 0.72),
        sun_direction: tuple[float, float, float] = (0.35, -0.85, 0.40),
        sun_lux: float = 4.0,
        sun_color: tuple[float, float, float] = (1.0, 0.96, 0.88),
        ambient: float = 0.04,
    ) -> None:
        self.ctx = ctx
        self.enabled = False
        self.config = None
        self.sky_color = sky_color
        self.sun_direction = sun_direction
        self.sun_lux = sun_lux
        self.sun_color = sun_color
        self.ambient = ambient

        # 1x1 LUT stand-ins so the resolve shader always has valid bindings.
        self._sky_view_lut = self._make_constant_texture(sky_color)
        self._transmittance_lut = self._make_constant_texture((1.0, 1.0, 1.0))
        self._multiscattering_lut = self._make_constant_texture((0.0, 0.0, 0.0))
        self._atmo_sun_tuple: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _make_constant_texture(self, color: tuple[float, float, float]) -> moderngl.Texture:
        tex = self.ctx.texture((1, 1), 4, dtype="f2")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.write(
            __import__("struct").pack("<4e", float(color[0]), float(color[1]), float(color[2]), 1.0)
        )
        return tex

    def update(self, dt: float) -> None:
        pass

    @property
    def sky_view_lut(self) -> moderngl.Texture:
        return self._sky_view_lut

    @property
    def transmittance_lut(self) -> moderngl.Texture:
        return self._transmittance_lut

    @property
    def multiscattering_lut(self) -> moderngl.Texture:
        return self._multiscattering_lut

    def compute_sun_vector(self) -> tuple[float, float, float]:
        return self.sun_direction

    def compute_lighting(self) -> tuple[
        tuple[float, float, float], float, tuple[float, float, float], float
    ]:
        return (self.sun_direction, self.sun_lux, self.sun_color, self.ambient)

    def render_sky(
        self,
        camera_pos: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
        sun_color: tuple[float, float, float],
        sun_intensity: float,
    ) -> None:
        pass

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
        pass

    def apply_resolve_uniforms(self, program: moderngl.Program) -> None:
        """Sets a flat sky: no scattering, no stars, no celestial bodies."""
        for name, value in (
            ("u_RayleighBeta", 0.0),
            ("u_MieBeta", (0.0, 0.0, 0.0)),
            ("u_MieG", 0.0),
            ("u_OzoneBeta", (0.0, 0.0, 0.0)),
            ("u_AtmosphereTurbidity", 1.0),
            ("u_StarIntensity", 0.0),
            ("u_StarDensity", 0.0),
            ("u_NebulaIntensity", 0.0),
            ("u_SunDiscSize", 0.0),
            ("u_MoonDiscSize", 0.0),
            ("u_MoonIntensity", 0.0),
            ("u_SkyMode", 0),
        ):
            uniform = program.get(name, None)
            if uniform is not None:
                uniform.value = value

    def apply_sun_direction(self, program: moderngl.Program, sun_dir: tuple[float, float, float]) -> None:
        """Uploads the atmosphere sun direction, skipping redundant writes."""
        sx, sy, sz = float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])
        if (sx, sy, sz) == self._atmo_sun_tuple:
            return
        self._atmo_sun_tuple = (sx, sy, sz)
        uniform = program.get("u_AtmosphereSunDir", None)
        if uniform is not None:
            uniform.value = self._atmo_sun_tuple

    def bind_luts(self, sky_view_unit: int, transmittance_unit: int) -> None:
        self._sky_view_lut.use(location=sky_view_unit)
        self._transmittance_lut.use(location=transmittance_unit)

    def destroy(self) -> None:
        for tex in (self._sky_view_lut, self._transmittance_lut, self._multiscattering_lut):
            try:
                tex.release()
            except Exception:
                pass


def create_default_environment(ctx: moderngl.Context, config: Any = None) -> EnvironmentProvider:
    """Resolves the configured environment provider.

    The atmosphere module is imported lazily so that a project which disables it
    never pays its import or GPU cost. If the module is unavailable or disabled,
    a ``FlatEnvironment`` is returned.
    """
    enabled = True
    settings: dict[str, Any] = {}
    if config is not None:
        enabled = bool(getattr(config, "atmosphere_enabled", True))
        settings = dict(getattr(config, "atmosphere_settings", {}) or {})

    if not enabled:
        return FlatEnvironment(ctx)

    try:
        from engine.modules.atmosphere import create_atmosphere_environment
    except Exception:
        return FlatEnvironment(ctx)

    return create_atmosphere_environment(ctx, settings)


__all__ = [
    "EnvironmentProvider",
    "FlatEnvironment",
    "create_default_environment",
]
