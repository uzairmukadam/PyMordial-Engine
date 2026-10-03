"""PyMordial Engine Native ModernGL UI Subsystem.

Provides GPU-accelerated 2D UI rendering, widgets, themes, and screens.
"""

from engine.ui.shaders import UI_VERT_GLSL, UI_FRAG_GLSL
from engine.ui.theme import UIStyle, UITheme
from engine.ui.renderer import UIRenderer
from engine.ui.elements import (
    UIElement,
    UIPanel,
    UILabel,
    UIButton,
    UISlider,
    UISegmentGroup,
    UIImage,
)
from engine.ui.manager import UIScreen, UIManager

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
