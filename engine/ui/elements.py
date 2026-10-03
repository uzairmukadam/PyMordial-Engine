"""Native ModernGL UI Elements and Interactive Widgets.

Provides modular UI components (Panel, Label, Button, Slider, SegmentGroup, Image)
with event hit-testing and styling hooks.
"""

from __future__ import annotations
from typing import Callable, Any
import moderngl

from engine.ui.theme import UIStyle, UITheme
from engine.ui.renderer import UIRenderer


class UIElement:
    """Base class for all native OpenGL UI widgets."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 100.0,
        h: float = 40.0,
        style: UIStyle | None = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.style = style if style is not None else UIStyle()
        self.visible = True
        self.enabled = True
        self.parent: UIElement | None = None
        self.children: list[UIElement] = []

        # Interactive state
        self.is_hovered = False
        self.is_pressed = False

    @property
    def absolute_x(self) -> float:
        return self.x + (self.parent.absolute_x if self.parent else 0.0)

    @property
    def absolute_y(self) -> float:
        return self.y + (self.parent.absolute_y if self.parent else 0.0)

    def add_child(self, child: UIElement) -> UIElement:
        child.parent = self
        self.children.append(child)
        return child

    def remove_child(self, child: UIElement) -> None:
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def hit_test(self, px: float, py: float) -> bool:
        """Tests if coordinate is within absolute bounds."""
        if not self.visible:
            return False
        ax, ay = self.absolute_x, self.absolute_y
        return ax <= px <= ax + self.w and ay <= py <= ay + self.h

    def on_mouse_move(self, mx: float, my: float) -> bool:
        if not self.visible or not self.enabled:
            return False
        handled = False
        for c in reversed(self.children):
            if c.on_mouse_move(mx, my):
                handled = True
                break

        self.is_hovered = self.hit_test(mx, my)
        return handled or self.is_hovered

    def on_mouse_down(self, mx: float, my: float, button: int) -> bool:
        if not self.visible or not self.enabled:
            return False
        for c in reversed(self.children):
            if c.on_mouse_down(mx, my, button):
                return True
        if self.hit_test(mx, my):
            self.is_pressed = True
            return True
        return False

    def on_mouse_up(self, mx: float, my: float, button: int) -> bool:
        if not self.visible or not self.enabled:
            return False
        handled = False
        for c in reversed(self.children):
            if c.on_mouse_up(mx, my, button):
                handled = True
                break

        if self.is_pressed:
            self.is_pressed = False
            if self.hit_test(mx, my):
                self.on_click()
            handled = True
        return handled

    def on_click(self) -> None:
        pass

    def render(self, renderer: UIRenderer, theme: UITheme) -> None:
        if not self.visible:
            return
        self.draw(renderer, theme)
        for c in self.children:
            c.render(renderer, theme)

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        pass


class UIPanel(UIElement):
    """Container card with background fill and border."""

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        renderer.draw_rect(self.absolute_x, self.absolute_y, self.w, self.h, self.style)


class UILabel(UIElement):
    """Text label supporting left, center, right alignment."""

    def __init__(
        self,
        text: str,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 200.0,
        h: float = 30.0,
        font_size: int = 18,
        color: tuple[float, float, float, float] | None = None,
        align: str = "left",
        bold: bool = False,
    ) -> None:
        super().__init__(x, y, w, h)
        self.text = text
        self.font_size = font_size
        self.color = color if color is not None else (0.95, 0.96, 0.98, 1.0)
        self.align = align
        self.bold = bold

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        renderer.flush()  # Draw text with font texture
        ax, ay = self.absolute_x, self.absolute_y
        tx = ax + (self.w * 0.5 if self.align == "center" else (self.w if self.align == "right" else 0.0))
        # Center vertically inside element height
        ty = ay + max(0.0, (self.h - self.font_size) * 0.4)
        renderer.draw_text(
            self.text,
            tx,
            ty,
            font_size=self.font_size,
            color=self.color,
            align=self.align,
            bold=self.bold,
        )


class UIButton(UIElement):
    """Interactive button with hover and click state styles."""

    def __init__(
        self,
        text: str,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 180.0,
        h: float = 44.0,
        variant: str = "primary",  # "primary", "accent", "success", "neutral", "danger"
        font_size: int | None = None,
        style: UIStyle | None = None,
        on_click: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.text = text
        self.variant = variant
        self.font_size = font_size
        self.style = style
        self.click_callback = on_click

    def on_click(self) -> None:
        if self.click_callback:
            self.click_callback()

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        if self.style is not None:
            btn_style = self.style
        else:
            btn_style = theme.create_button_style(
                variant=self.variant,
                is_hovered=self.is_hovered,
                is_pressed=self.is_pressed,
            )
        f_size = self.font_size if self.font_size is not None else btn_style.font_size
        ax, ay = self.absolute_x, self.absolute_y
        renderer.draw_rect(ax, ay, self.w, self.h, btn_style)

        # Button text
        if self.text:
            renderer.flush()
            ty = ay + max(0.0, (self.h - f_size) * 0.42)
            renderer.draw_text(
                self.text,
                ax + self.w * 0.5,
                ty,
                font_size=f_size,
                color=btn_style.text_color,
                align="center",
                bold=True,
            )


class UISlider(UIElement):
    """Numeric slider with draggable thumb and formatted value readout."""

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        val: float,
        step: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 240.0,
        h: float = 40.0,
        format_str: str = "{label}: {val:.0f}",
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.label = label
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.val = float(val)
        self.step = float(step)
        self.format_str = format_str
        self.on_change = on_change
        self.is_dragging = False

    def on_mouse_down(self, mx: float, my: float, button: int) -> bool:
        if self.hit_test(mx, my):
            self.is_dragging = True
            self._update_val_from_mouse(mx)
            return True
        return False

    def on_mouse_move(self, mx: float, my: float) -> bool:
        super().on_mouse_move(mx, my)
        if self.is_dragging:
            self._update_val_from_mouse(mx)
            return True
        return self.is_hovered

    def on_mouse_up(self, mx: float, my: float, button: int) -> bool:
        if self.is_dragging:
            self.is_dragging = False
            return True
        return super().on_mouse_up(mx, my, button)

    def _update_val_from_mouse(self, mx: float) -> None:
        ax = self.absolute_x
        ratio = max(0.0, min(1.0, (mx - ax) / max(1.0, self.w)))
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        if self.step > 0.0:
            stepped = round((raw - self.min_val) / self.step) * self.step + self.min_val
        else:
            stepped = raw
        stepped = max(self.min_val, min(self.max_val, stepped))
        if stepped != self.val:
            self.val = stepped
            if self.on_change:
                self.on_change(self.val)

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        ax, ay = self.absolute_x, self.absolute_y
        track_h = 8.0
        track_y = ay + 26.0

        # Background Track
        track_style = UIStyle(
            bg_color_top=(0.14, 0.17, 0.22, 0.95),
            bg_color_bottom=(0.08, 0.10, 0.14, 0.95),
            border_color=(0.25, 0.30, 0.38, 0.8),
            border_width=1.0,
            corner_radius=4.0,
        )
        renderer.draw_rect(ax, track_y, self.w, track_h, track_style)

        # Active Fill Track
        ratio = (self.val - self.min_val) / max(1e-4, self.max_val - self.min_val)
        fill_w = max(4.0, self.w * ratio)
        fill_style = UIStyle(
            bg_color_top=theme.primary,
            bg_color_bottom=(theme.primary[0] * 0.7, theme.primary[1] * 0.7, theme.primary[2] * 0.7, 1.0),
            corner_radius=4.0,
        )
        renderer.draw_rect(ax, track_y, fill_w, track_h, fill_style)

        # Thumb Knob
        thumb_r = 9.0
        thumb_x = ax + fill_w - thumb_r
        thumb_y = track_y + (track_h * 0.5) - thumb_r
        thumb_style = UIStyle(
            bg_color_top=(1.0, 1.0, 1.0, 1.0),
            bg_color_bottom=(0.85, 0.85, 0.90, 1.0),
            border_color=theme.border_active if self.is_hovered or self.is_dragging else theme.border_subtle,
            border_width=2.0,
            corner_radius=thumb_r,
            glow_color=theme.primary if self.is_dragging else (0.0, 0.0, 0.0, 0.0),
            glow_radius=8.0 if self.is_dragging else 0.0,
        )
        renderer.draw_rect(thumb_x, thumb_y, thumb_r * 2.0, thumb_r * 2.0, thumb_style)

        # Header readout label
        renderer.flush()
        disp_text = self.format_str.format(label=self.label, val=self.val)
        renderer.draw_text(
            disp_text,
            ax,
            ay,
            font_size=15,
            color=theme.text_main,
            align="left",
            bold=False,
        )


class UISegmentGroup(UIElement):
    """Horizontal segmented radio button group."""

    def __init__(
        self,
        options: list[tuple[str, Any]],  # [(label, value), ...]
        selected_index: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 320.0,
        h: float = 38.0,
        on_change: Callable[[Any], None] | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.options = options
        self.selected_index = max(0, min(len(options) - 1, selected_index))
        self.on_change = on_change

    @property
    def selected_value(self) -> Any:
        return self.options[self.selected_index][1]

    def on_click(self) -> None:
        # Note: on_click called after on_mouse_up hit-tested; selection handled in on_mouse_up
        pass

    def on_mouse_up(self, mx: float, my: float, button: int) -> bool:
        if not self.visible or not self.enabled:
            return False
        if self.is_pressed and self.hit_test(mx, my):
            self.is_pressed = False
            ax = self.absolute_x
            btn_w = self.w / len(self.options)
            idx = int((mx - ax) // btn_w)
            idx = max(0, min(len(self.options) - 1, idx))
            if idx != self.selected_index:
                self.selected_index = idx
                if self.on_change:
                    self.on_change(self.selected_value)
            return True
        self.is_pressed = False
        return False

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        ax, ay = self.absolute_x, self.absolute_y
        num_opts = len(self.options)
        if num_opts == 0:
            return

        btn_w = self.w / num_opts

        # Base Frame
        base_style = UIStyle(
            bg_color_top=(0.10, 0.12, 0.16, 0.95),
            bg_color_bottom=(0.06, 0.08, 0.10, 0.98),
            border_color=theme.border_subtle,
            border_width=1.5,
            corner_radius=6.0,
        )
        renderer.draw_rect(ax, ay, self.w, self.h, base_style)

        # Draw individual segments
        for i, (label, val) in enumerate(self.options):
            bx = ax + i * btn_w
            is_selected = (i == self.selected_index)

            if is_selected:
                seg_style = UIStyle(
                    bg_color_top=theme.primary,
                    bg_color_bottom=(theme.primary[0] * 0.75, theme.primary[1] * 0.75, theme.primary[2] * 0.75, 1.0),
                    border_color=theme.border_active,
                    border_width=1.0,
                    corner_radius=5.0,
                )
                renderer.draw_rect(bx + 2.0, ay + 2.0, btn_w - 4.0, self.h - 4.0, seg_style)

            renderer.flush()
            t_col = (1.0, 1.0, 1.0, 1.0) if is_selected else theme.text_muted
            renderer.draw_text(
                label,
                bx + btn_w * 0.5,
                ay + (self.h - 16) * 0.45,
                font_size=15,
                color=t_col,
                align="center",
                bold=is_selected,
            )


class UIImage(UIElement):
    """Draws a ModernGL texture with optional border and tint."""

    def __init__(
        self,
        texture: moderngl.Texture,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 120.0,
        h: float = 120.0,
        tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        corner_radius: float = 8.0,
    ) -> None:
        super().__init__(x, y, w, h)
        self.texture = texture
        self.tint = tint
        self.corner_radius = corner_radius

    def draw(self, renderer: UIRenderer, theme: UITheme) -> None:
        if self.texture is not None:
            renderer.draw_textured_rect(
                x=self.absolute_x,
                y=self.absolute_y,
                w=self.w,
                h=self.h,
                texture=self.texture,
                tint=self.tint,
                corner_radius=self.corner_radius,
            )
