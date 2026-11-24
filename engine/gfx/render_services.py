"""Stable Renderer Surface Exposed to Render Feature Modules.

A render feature must never reach into ``RenderPipeline`` internals. This module
defines the narrow, documented surface a feature may use: the GL context, the
render config, the frame context, the shared G-buffer, the shadow atlas, the HDR
target, texture-unit reservations, and a capability query.

The pipeline constructs one ``RenderServices`` per frame and hands it to
registered features. Anything not exposed here is private implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import moderngl

from engine.gfx.quality_presets import RenderConfig

# Texture units reserved by the core deferred resolve pass. Features must not
# bind to these units; they are listed here so a feature can detect a conflict
# instead of silently corrupting the resolve.
RESERVED_TEXTURE_UNITS: dict[int, str] = {
    0: "u_GBufferAlbedoRoughness",
    1: "u_GBufferNormalMetallic",
    2: "u_GBufferDepth",
    3: "u_ShadowAtlas",
    4: "u_AOTexture",
    5: "u_SSGITexture",
    6: "u_BRDFLUT",
    7: "u_EnvironmentMap",
    8: "u_SSRTexture",
    10: "u_SkyViewLUT",
    11: "u_TransmittanceLUT",
    12: "u_SpotShadowAtlas",
    13: "u_IrradianceMap",
    15: "u_RMGITexture",
}

# Texture units a render feature may claim.
FEATURE_TEXTURE_UNITS: tuple[int, ...] = (9, 16, 17, 18, 19, 20, 21, 22, 23)


class RenderCapabilityError(Exception):
    """Raised when a render feature requires a capability the GPU does not provide."""


@dataclass
class RenderServices:
    """The documented renderer surface available to render feature modules."""

    ctx: moderngl.Context
    config: RenderConfig
    width: int
    height: int
    frame_context: Any = None
    g_buffer: Any = None
    csm: Any = None
    hdr_fbo: Any = None
    hdr_texture: Any = None
    depth_texture: Any = None
    mega_buffer: Any = None
    texture_atlas: Any = None
    material_registry: Any = None
    _claimed_units: set[int] = field(default_factory=set)
    _capabilities: dict[str, bool] = field(default_factory=dict)

    # ---------------- Capability queries ----------------

    def has_capability(self, name: str) -> bool:
        """Returns True when the GPU context provides a named capability.

        Recognized names: ``compute``, ``tessellation``, ``multi_draw_indirect``,
        ``shader_draw_parameters``, ``float_textures``.
        """
        cached = self._capabilities.get(name)
        if cached is not None:
            return cached

        extensions = getattr(self.ctx, "extensions", ()) or ()
        version_code = int(getattr(self.ctx, "version_code", 0) or 0)

        if name == "compute":
            result = version_code >= 430 or "GL_ARB_compute_shader" in extensions
        elif name == "tessellation":
            result = version_code >= 400 or "GL_ARB_tessellation_shader" in extensions
        elif name == "multi_draw_indirect":
            result = "GL_ARB_multi_draw_indirect" in extensions
        elif name == "shader_draw_parameters":
            result = "GL_ARB_shader_draw_parameters" in extensions
        elif name == "float_textures":
            result = version_code >= 300
        else:
            result = False

        self._capabilities[name] = result
        return result

    def require_capability(self, name: str, feature: str = "render feature") -> None:
        """Raises ``RenderCapabilityError`` when a required capability is missing."""
        if not self.has_capability(name):
            raise RenderCapabilityError(
                f"{feature} requires the '{name}' GPU capability, which this context does not provide."
            )

    # ---------------- Texture unit reservations ----------------

    def claim_texture_unit(self, unit: int, feature: str = "render feature") -> int:
        """Reserves a texture unit for a feature.

        Raises:
            RenderCapabilityError: if the unit is reserved by the core resolve
                pass or already claimed by another feature.
        """
        if unit in RESERVED_TEXTURE_UNITS:
            raise RenderCapabilityError(
                f"{feature} cannot claim texture unit {unit}: reserved for "
                f"'{RESERVED_TEXTURE_UNITS[unit]}' by the core deferred resolve."
            )
        if unit in self._claimed_units:
            raise RenderCapabilityError(
                f"{feature} cannot claim texture unit {unit}: already claimed by another feature."
            )
        self._claimed_units.add(unit)
        return unit

    def claim_next_texture_unit(self, feature: str = "render feature") -> int:
        """Claims the lowest free feature texture unit."""
        for unit in FEATURE_TEXTURE_UNITS:
            if unit not in self._claimed_units:
                return self.claim_texture_unit(unit, feature)
        raise RenderCapabilityError(f"{feature} cannot claim a texture unit: all feature units are in use.")

    def release_texture_unit(self, unit: int) -> None:
        self._claimed_units.discard(unit)

    @property
    def claimed_texture_units(self) -> frozenset[int]:
        return frozenset(self._claimed_units)


__all__ = [
    "FEATURE_TEXTURE_UNITS",
    "RESERVED_TEXTURE_UNITS",
    "RenderCapabilityError",
    "RenderServices",
]
