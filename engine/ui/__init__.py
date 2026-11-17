"""PyMordial Engine Native ModernGL UI Subsystem.

Provides GPU-accelerated 2D UI rendering, widgets, themes, and screens.
"""

from engine.ui.elements import (
    UIButton,
    UIElement,
    UIImage,
    UILabel,
    UIPanel,
    UISegmentGroup,
    UISlider,
)
from engine.ui.manager import UIManager, UIScreen
from engine.ui.renderer import UIRenderer
from engine.ui.shaders import UI_FRAG_GLSL, UI_VERT_GLSL
from engine.ui.theme import UIStyle, UITheme

__all__ = [
    "UI_VERT_GLSL",
    "UI_FRAG_GLSL",
    "UIStyle",
    "UITheme",
    "UIRenderer",
    "UIElement",
    "UIPanel",
    "UILabel",
    "UIButton",
    "UISlider",
    "UISegmentGroup",
    "UIImage",
    "UIScreen",
    "UIManager",
]
