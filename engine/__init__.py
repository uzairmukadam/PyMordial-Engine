"""PyMordial Game Engine (Alpha).

A high-performance, modular 3D game engine built in Python 3.11+, ModernGL,
Rapier3D physics, and PyGame-CE. Designed for zero-allocation runtime performance
and clean isolation between core engine infrastructure and project-specific gameplay logic.
"""

from engine.app.project_app import ProjectApp
from engine.app.config import ProjectConfig
from engine.app.module import ProjectModule
from engine.window.window import WindowMode, VSyncMode
from engine.gfx.quality_presets import GraphicsQuality, RenderConfig, get_quality_preset
from engine.gfx.context import RenderContext
from engine.gfx.pipeline import RenderPipeline
from engine.core.ecs import EntityManager
from engine.camera.camera import VirtualCamera
from engine.camera.follow_camera import FollowCamera, FollowCameraConfig
from engine.camera.free_camera import FreeFlyCamera, FreeCamera
from engine.physics.rapier_world import PhysicsManager
from engine.physics.character_motor import CharacterMotor, CharacterMotorConfig
from engine.physics.character_controller import KinematicCharacterMotor
from engine.audio import AudioEngine, get_audio_engine, SoundCue
from engine.input import InputManager
from engine.core.state import GameState, GameStateManager
from engine.ui import (
    UIManager,
    UIScreen,
    UITheme,
    UIStyle,
    UIButton,
    UIPanel,
    UILabel,
    UISlider,
    UISegmentGroup,
    UIImage,
)

__version__ = "0.1.0b1"

__all__ = [
    "__version__",
    "ProjectApp",
    "ProjectConfig",
    "ProjectModule",
    "WindowMode",
    "VSyncMode",
    "GraphicsQuality",
    "RenderConfig",
    "get_quality_preset",
    "RenderContext",
    "RenderPipeline",
    "EntityManager",
    "VirtualCamera",
    "FollowCamera",
    "FollowCameraConfig",
    "FreeFlyCamera",
    "FreeCamera",
    "PhysicsManager",
    "CharacterMotor",
    "KinematicCharacterMotor",
    "CharacterMotorConfig",
    "AudioEngine",
    "get_audio_engine",
    "SoundCue",
    "InputManager",
    "GameState",
    "GameStateManager",
    "UIManager",
    "UIScreen",
    "UITheme",
    "UIStyle",
    "UIButton",
    "UIPanel",
    "UILabel",
    "UISlider",
    "UISegmentGroup",
    "UIImage",
]
