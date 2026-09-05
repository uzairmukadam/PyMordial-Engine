"""PyMordial ModernGL 4.5 Core Graphics Subsystem."""

from engine.gfx.quality_presets import (
    GraphicsQuality,
    RenderConfig,
    get_quality_preset,
    QUALITY_PRESETS,
)
from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.g_buffer import GBuffer
from engine.gfx.shadow_csm import CascadedShadowMap
from engine.gfx.post_process import PostProcessPipeline
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.hud_overlay import HudOverlay

__all__ = [
    "GraphicsQuality",
    "RenderConfig",
    "get_quality_preset",
    "QUALITY_PRESETS",
    "RenderContext",
    "FrameContext",
    "MegaBuffer",
    "MeshAllocation",
    "MultiDrawIndirect",
    "GBuffer",
    "CascadedShadowMap",
    "PostProcessPipeline",
    "RenderPipeline",
    "HudOverlay",
]
