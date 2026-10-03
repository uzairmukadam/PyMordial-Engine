"""Unit Tests for Engine Native ModernGL UI Subsystem."""


from engine.ui import (
    UIPanel,
    UIButton,
    UISlider,
    UISegmentGroup,
    UIScreen,
)


def test_ui_element_hierarchy_and_bounds():
    root = UIPanel(x=100, y=50, w=400, h=300)
    child = UIButton(text="Test", x=20, y=30, w=100, h=40)
    root.add_child(child)

    assert child.absolute_x == 120.0
    assert child.absolute_y == 80.0
    assert child.hit_test(150, 100) is True
    assert child.hit_test(50, 50) is False


def test_ui_button_click():
    clicked = [False]

    def on_click():
        clicked[0] = True

    btn = UIButton(text="Action", x=50, y=50, w=120, h=40, on_click=on_click)

    # Hover
    btn.on_mouse_move(80, 70)
    assert btn.is_hovered is True

    # Mouse down
    btn.on_mouse_down(80, 70, 1)
    assert btn.is_pressed is True

    # Mouse up inside -> triggers click
    btn.on_mouse_up(80, 70, 1)
    assert btn.is_pressed is False
    assert clicked[0] is True


def test_ui_slider_value_change():
    slider_val = [0.0]

    def on_change(v):
        slider_val[0] = v

    slider = UISlider(label="Laps", min_val=1.0, max_val=10.0, val=3.0, step=1.0, x=0, y=0, w=100, h=40, on_change=on_change)
    assert slider.val == 3.0

    # Drag to 80% of width
    slider.on_mouse_down(80, 20, 1)
    # val should be close to 1 + 0.8 * 9 = 8.2 -> stepped to 8.0
    assert slider.val == 8.0
    assert slider_val[0] == 8.0


def test_ui_segment_group():
    selected = ["pro"]

    def on_change(v):
        selected[0] = v

    segments = UISegmentGroup(
        options=[("Rookie", "rookie"), ("Pro", "pro"), ("Legend", "legend")],
        selected_index=1,
        x=0,
        y=0,
        w=300,
        h=40,
        on_change=on_change,
    )
    assert segments.selected_value == "pro"

    # Click on third segment (x in [200, 300])
    segments.on_mouse_down(250, 20, 1)
    segments.on_mouse_up(250, 20, 1)
    assert segments.selected_value == "legend"
    assert selected[0] == "legend"


def test_ui_screen_stack_lifecycle():
    from engine.ui import UIManager
    mgr = UIManager(ctx=None)

    entered = []
    exited = []

    class TrackedScreen(UIScreen):
        def on_enter(self, app):
            entered.append(self.name)

        def on_exit(self, app):
            exited.append(self.name)

    screen_a = TrackedScreen(name="ScreenA")
    screen_b = TrackedScreen(name="ScreenB")

    mgr.set_screen(screen_a)
    assert mgr.active_screen is screen_a
    assert mgr.active_screen.name == "ScreenA"

    mgr.push_screen(screen_b)
    assert len(mgr.screen_stack) == 2
    assert mgr.active_screen.name == "ScreenB"

    popped = mgr.pop_screen()
    assert popped.name == "ScreenB"
    assert mgr.active_screen.name == "ScreenA"

    mgr.set_screen(None)
    assert mgr.active_screen is None


def test_ui_event_dispatch():
    import pygame
    from engine.ui import UIManager

    mgr = UIManager(ctx=None)
    screen = UIScreen(name="EventScreen")
    clicked = [False]

    btn = UIButton(text="ClickMe", x=10, y=10, w=100, h=50, on_click=lambda: clicked.__setitem__(0, True))
    screen.add_child(btn)
    mgr.set_screen(screen)

    # 1. Mouse motion event
    ev_motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(50, 30), rel=(0, 0), buttons=(0, 0, 0))
    handled = mgr.handle_event(ev_motion)
    assert handled is True
    assert btn.is_hovered is True

    # 2. Mouse down event
    ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(50, 30), button=1)
    handled = mgr.handle_event(ev_down)
    assert handled is True
    assert btn.is_pressed is True

    # 3. Mouse up event triggers click
    ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(50, 30), button=1)
    handled = mgr.handle_event(ev_up)
    assert handled is True
    assert clicked[0] is True
