"""Theme and Styling System for Native ModernGL UI.

Allows games to inject custom arcade or simulation aesthetics,
color palettes, fonts, and button interaction states.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class UIStyle:
    """Styling properties for an individual UI element."""
    # Fill colors (RGBA 0.0 - 1.0)
    bg_color_top: tuple[float, float, float, float] = (0.12, 0.14, 0.18, 0.92)
    bg_color_bottom: tuple[float, float, float, float] = (0.08, 0.09, 0.12, 0.96)

    # Border properties
    border_color: tuple[float, float, float, float] = (0.28, 0.32, 0.40, 0.80)
    border_width: float = 1.5
    corner_radius: float = 8.0

    # Text properties
    text_color: tuple[float, float, float, float] = (0.95, 0.96, 0.98, 1.0)
    font_size: int = 18
    text_align: str = "center"  # "left", "center", "right"

    # Glow / accent properties
    glow_color: tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.0)
    glow_radius: float = 0.0


@dataclass
class UITheme:
    """Master aesthetic theme configuration for native engine UI."""
    name: str = "DefaultDark"

    # Colors
    primary: tuple[float, float, float, float] = (0.10, 0.65, 0.95, 1.0)       # Electric Cyan
    secondary: tuple[float, float, float, float] = (1.0, 0.45, 0.10, 1.0)     # Neon Orange
    accent: tuple[float, float, float, float] = (0.10, 1.0, 0.55, 1.0)        # Acid Green

    bg_dark: tuple[float, float, float, float] = (0.06, 0.07, 0.09, 0.94)
    bg_panel: tuple[float, float, float, float] = (0.10, 0.12, 0.16, 0.92)
    bg_card: tuple[float, float, float, float] = (0.14, 0.17, 0.22, 0.88)

    border_subtle: tuple[float, float, float, float] = (0.22, 0.26, 0.34, 0.70)
    border_active: tuple[float, float, float, float] = (0.20, 0.80, 1.0, 1.0)

    text_main: tuple[float, float, float, float] = (0.95, 0.96, 0.98, 1.0)
    text_muted: tuple[float, float, float, float] = (0.60, 0.65, 0.75, 1.0)
    text_highlight: tuple[float, float, float, float] = (1.0, 0.90, 0.20, 1.0)

    # Default geometry
    corner_radius: float = 8.0
    panel_radius: float = 12.0
    border_width: float = 1.5

    def create_panel_style(self) -> UIStyle:
        return UIStyle(
            bg_color_top=self.bg_panel,
            bg_color_bottom=self.bg_dark,
            border_color=self.border_subtle,
            border_width=self.border_width,
            corner_radius=self.panel_radius,
            text_color=self.text_main,
        )

    def create_button_style(self, variant: str = "primary", is_hovered: bool = False, is_pressed: bool = False) -> UIStyle:
        if variant == "primary":
            base_top = (0.12, 0.60, 0.92, 0.95)
            base_bot = (0.08, 0.42, 0.78, 0.95)
            border = (0.35, 0.85, 1.0, 1.0) if is_hovered else (0.20, 0.65, 0.95, 0.80)
        elif variant == "accent":
            base_top = (1.0, 0.50, 0.12, 0.95)
            base_bot = (0.85, 0.32, 0.08, 0.95)
            border = (1.0, 0.80, 0.30, 1.0) if is_hovered else (1.0, 0.50, 0.12, 0.80)
        elif variant == "success":
            base_top = (0.15, 0.85, 0.45, 0.95)
            base_bot = (0.10, 0.65, 0.32, 0.95)
            border = (0.40, 1.0, 0.65, 1.0) if is_hovered else (0.20, 0.85, 0.45, 0.80)
        elif variant == "danger":
            base_top = (0.85, 0.18, 0.22, 0.95)
            base_bot = (0.65, 0.10, 0.14, 0.95)
            border = (1.0, 0.40, 0.45, 1.0) if is_hovered else (0.85, 0.20, 0.25, 0.80)
        else: # secondary/neutral
            base_top = (0.18, 0.22, 0.28, 0.90)
            base_bot = (0.12, 0.15, 0.20, 0.95)
            border = (0.45, 0.52, 0.65, 0.90) if is_hovered else (0.28, 0.34, 0.42, 0.70)

        if is_pressed:
            # Darken and shift down
            return UIStyle(
                bg_color_top=tuple(c * 0.75 for c in base_bot[:3]) + (base_bot[3],),
                bg_color_bottom=tuple(c * 0.75 for c in base_top[:3]) + (base_top[3],),
                border_color=border,
                border_width=self.border_width + 0.5,
                corner_radius=self.corner_radius,
                text_color=self.text_main,
                font_size=18,
            )
        elif is_hovered:
            # Brighten on hover
            return UIStyle(
                bg_color_top=tuple(min(1.0, c * 1.25) for c in base_top[:3]) + (base_top[3],),
                bg_color_bottom=tuple(min(1.0, c * 1.20) for c in base_bot[:3]) + (base_bot[3],),
                border_color=border,
                border_width=self.border_width + 0.5,
                corner_radius=self.corner_radius,
                text_color=(1.0, 1.0, 1.0, 1.0),
                font_size=18,
                glow_color=border,
                glow_radius=8.0,
            )
        else:
            return UIStyle(
                bg_color_top=base_top,
                bg_color_bottom=base_bot,
                border_color=border,
                border_width=self.border_width,
                corner_radius=self.corner_radius,
                text_color=self.text_main,
                font_size=18,
            )
