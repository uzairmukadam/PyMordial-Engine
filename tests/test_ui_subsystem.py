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


def test_ui_non_modal_hud_passthrough():
    """Ensures non-modal HUD screens pass unhandled mouse motions through to gameplay."""
    import pygame
    from engine.ui import UIManager

    mgr = UIManager(ctx=None)
    hud = UIScreen(name="GameplayHUD", is_modal=False)
    mgr.set_screen(hud)

    # Moving mouse on empty HUD should return False (pass-through)
    ev_motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(500, 300), rel=(10, -5), buttons=(0, 0, 0))
    handled = mgr.handle_event(ev_motion)
    assert handled is False, "Non-modal HUD must pass through mouse motion to camera!"


def test_capsule_mesh_primitive_proportions():
    """Validates that capsule mesh primitive has correct radius and smooth geometry without pinch distortion."""
    import numpy as np
    import moderngl
    from engine.gfx.mega_buffer import MegaBuffer

    ctx = moderngl.create_context(standalone=True)
    mega = MegaBuffer(ctx)
    alloc = mega.allocations["capsule"]
    assert alloc is not None
    assert alloc.vertex_count > 0
    assert alloc.index_count > 0

    verts_chunk = np.concatenate(mega.vertex_data, axis=0)
    capsule_verts = verts_chunk[alloc.base_vertex : alloc.base_vertex + alloc.vertex_count]
    pos = capsule_verts["position"]
    xs = pos[:, 0]
    ys = pos[:, 1]
    zs = pos[:, 2]

    # Total height should be roughly 1.8m (cylinder height 1.0 + 2*radius 0.4)
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    total_height = max_y - min_y
    assert abs(total_height - 1.8) < 1e-3, f"Expected height ~1.8m, got {total_height}"

    # Cylinder wall at y=-0.5 and y=+0.5 should have exact radius 0.4
    cyl_bottom = np.isclose(ys, -0.5, atol=1e-3)
    cyl_top = np.isclose(ys, 0.5, atol=1e-3)
    assert np.any(cyl_bottom)
    assert np.any(cyl_top)
    r_bottom = np.sqrt(xs[cyl_bottom] ** 2 + zs[cyl_bottom] ** 2)
    r_top = np.sqrt(xs[cyl_top] ** 2 + zs[cyl_top] ** 2)
    assert np.allclose(r_bottom, 0.4, atol=1e-3)
    assert np.allclose(r_top, 0.4, atol=1e-3)


def test_ui_captured_virtual_cursor():
    """Validates that UIManager tracks and accumulates relative mouse delta when captured."""
    import pygame
    from engine.ui import UIManager, UIScreen, UIButton

    mgr = UIManager(ctx=None, width=1280, height=720)
    screen = UIScreen(name="Menu", is_modal=True)
    clicked = [False]
    btn = UIButton(text="Play", x=600, y=340, w=100, h=40, on_click=lambda: clicked.__setitem__(0, True))
    screen.add_child(btn)
    mgr.set_screen(screen)
    mgr.cursor_visible = True

    # Initial cursor is at center: (640, 360)
    assert mgr.cursor_pos == [640.0, 360.0]

    # Button is at x in [600, 700], y in [340, 380]. (640, 360) is INSIDE the button!
    ev_motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(0, 0), rel=(10, 5), buttons=(0, 0, 0))
    if not pygame.display.get_init():
        pygame.display.init()
    try:
        pygame.event.set_grab(True)
    except Exception:
        pass
    try:
        mgr.handle_event(ev_motion)
        assert mgr.cursor_pos == [650.0, 365.0]
        assert btn.is_hovered is True

        ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(0, 0), button=1)
        mgr.handle_event(ev_down)
        assert btn.is_pressed is True

        ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(0, 0), button=1)
        mgr.handle_event(ev_up)
        assert clicked[0] is True
    finally:
        try:
            pygame.event.set_grab(False)
        except Exception:
            pass


def test_ui_canvas_scaling_and_resolution_independence():
    """Validates that UIManager scales canvas proportionally to screen resolution with accurate hit-testing."""
    import pygame
    from engine.ui import UIManager, UIScreen, UIButton
    from engine.events import WindowResizeEvent

    # Reference resolution: 1280x720
    mgr = UIManager(ctx=None, width=1280, height=720, reference_width=1280.0, reference_height=720.0)
    screen = UIScreen(name="ScalingMenu", is_modal=True)
    clicked = [False]

    # Button centered at x=540, y=320, w=200, h=80 in virtual 1280x720 canvas
    btn = UIButton(text="Start", x=540, y=320, w=200, h=80, on_click=lambda: clicked.__setitem__(0, True))
    screen.add_child(btn)
    mgr.set_screen(screen)

    # 1. At 1280x720 (native reference)
    assert abs(mgr.ui_scale - 1.0) < 1e-4
    assert abs(mgr.offset_x) < 1e-4
    assert abs(mgr.offset_y) < 1e-4

    # 2. Resize to 1920x1080 (1.5x scaling)
    mgr._on_window_resize(WindowResizeEvent(width=1920, height=1080))
    assert abs(mgr.ui_scale - 1.5) < 1e-4
    assert abs(mgr.offset_x) < 1e-4
    assert abs(mgr.offset_y) < 1e-4

    # Window click at physical screen center: (1920 * 0.5, 1080 * 0.5) = (960, 540)
    # Virtual canvas position: (960 / 1.5, 540 / 1.5) = (640, 360)
    # Button bounds: [540..740, 320..400]. (640, 360) is exactly in the center!
    ev_motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(960, 540), rel=(0, 0), buttons=(0, 0, 0))
    mgr.handle_event(ev_motion)
    assert abs(mgr.cursor_pos[0] - 640.0) < 1e-3
    assert abs(mgr.cursor_pos[1] - 360.0) < 1e-3
    assert btn.is_hovered is True

    # Mouse down and up at physical (960, 540)
    ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(960, 540), button=1)
    mgr.handle_event(ev_down)
    assert btn.is_pressed is True

    ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(960, 540), button=1)
    mgr.handle_event(ev_up)
    assert btn.is_pressed is False
    assert clicked[0] is True

    # 3. Resize to 2560x1440 (2.0x scaling)
    mgr._on_window_resize(WindowResizeEvent(width=2560, height=1440))
    assert abs(mgr.ui_scale - 2.0) < 1e-4

    # 4. Ultrawide 21:9 resolution: 2560x1080
    mgr._on_window_resize(WindowResizeEvent(width=2560, height=1080))
    # Vertical scale is 1080 / 720 = 1.5; horizontal scale is 2560 / 1280 = 2.0.
    # Uniform aspect-ratio scale is min(1.5, 2.0) = 1.5
    assert abs(mgr.ui_scale - 1.5) < 1e-4
    # Centering offset_x = (2560 - 1280 * 1.5) * 0.5 = (2560 - 1920) * 0.5 = 320.0
    assert abs(mgr.offset_x - 320.0) < 1e-4
    assert abs(mgr.offset_y - 0.0) < 1e-4

    # Physical click at center of ultrawide: x = 1280, y = 540
    # Virtual canvas: vx = (1280 - 320) / 1.5 = 960 / 1.5 = 640.0; vy = 540 / 1.5 = 360.0
    ev_motion_uw = pygame.event.Event(pygame.MOUSEMOTION, pos=(1280, 540), rel=(0, 0), buttons=(0, 0, 0))
    mgr.handle_event(ev_motion_uw)
    assert abs(mgr.cursor_pos[0] - 640.0) < 1e-3
    assert abs(mgr.cursor_pos[1] - 360.0) < 1e-3
    assert btn.is_hovered is True


def test_ui_renderer_quad_scaling():
    """Validates that UIRenderer correctly scales quad vertex data based on active scale and offsets."""
    import moderngl
    from engine.ui import UIRenderer, UIStyle

    ctx = moderngl.create_context(standalone=True)
    renderer = UIRenderer(ctx, screen_width=1920, screen_height=1080)
    renderer.set_scale(ui_scale=1.5, offset_x=100.0, offset_y=50.0)

    # Draw a 200x100 rectangle at (10, 20) in virtual space
    style = UIStyle(corner_radius=8.0, border_width=2.0)
    renderer.draw_rect(10.0, 20.0, 200.0, 100.0, style)

    # In vertex buffer:
    # Position: [x*1.5 + 100, y*1.5 + 50, w*1.5, h*1.5] = [115.0, 80.0, 300.0, 150.0]
    vd = renderer._vertex_data
    assert abs(vd[0] - 115.0) < 1e-3
    assert abs(vd[1] - 80.0) < 1e-3
    assert abs(vd[2] - 300.0) < 1e-3
    assert abs(vd[3] - 150.0) < 1e-3

    # Params: [corner_radius*1.5, border_width*1.5, ...] = [12.0, 3.0, ...]
    assert abs(vd[20] - 12.0) < 1e-3
    assert abs(vd[21] - 3.0) < 1e-3


