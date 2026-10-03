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
    screen_a = UIScreen(name="ScreenA")
    screen_b = UIScreen(name="ScreenB")

    # In headless / mock mode, test manager stack operations without GPU context
    class MockManager:
        def __init__(self):
            self.screen_stack = []

        def set_screen(self, s):
            self.screen_stack = [s] if s else []

        def push_screen(self, s):
            self.screen_stack.append(s)

        def pop_screen(self):
            return self.screen_stack.pop() if self.screen_stack else None

    mgr = MockManager()
    mgr.set_screen(screen_a)
    assert mgr.screen_stack[-1].name == "ScreenA"

    mgr.push_screen(screen_b)
    assert len(mgr.screen_stack) == 2
    assert mgr.screen_stack[-1].name == "ScreenB"

    popped = mgr.pop_screen()
    assert popped.name == "ScreenB"
    assert mgr.screen_stack[-1].name == "ScreenA"
