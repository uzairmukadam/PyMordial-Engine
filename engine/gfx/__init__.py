"""PyMordial ModernGL 4.5 Core Graphics Subsystem."""

from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.g_buffer import GBuffer
from engine.gfx.hud_overlay import HudOverlay
from engine.gfx.loading_screen import LoadingScreen
from engine.gfx.material_registry import MaterialDef, MaterialRegistry
from engine.gfx.mdi import MultiDrawIndirect
from engine.gfx.mega_buffer import MegaBuffer, MeshAllocation
from engine.gfx.pipeline import RenderPipeline
from engine.gfx.post_process import PostProcessPipeline
from engine.gfx.quality_presets import (
    QUALITY_PRESETS,
    GIMode,
    GraphicsQuality,
    RenderConfig,
    get_quality_preset,
)
from engine.gfx.render_feature import RenderFeature, RenderFeatureRegistry
from engine.gfx.render_graph import RenderGraph, RenderGraphContext, RenderGraphError, RenderPass
from engine.gfx.render_services import RenderCapabilityError, RenderServices
from engine.gfx.shadow_csm import CascadedShadowMap

__all__ = [
    "GraphicsQuality",
    "GIMode",
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
    "RenderGraph",
    "RenderGraphContext",
    "RenderGraphError",
    "RenderPass",
    "RenderFeature",
    "RenderFeatureRegistry",
    "RenderCapabilityError",
    "RenderServices",
    "MaterialDef",
    "MaterialRegistry",
    "HudOverlay",
    "LoadingScreen",
]
