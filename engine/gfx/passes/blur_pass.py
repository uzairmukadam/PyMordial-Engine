"""Reusable separable bilateral blur helper for low-resolution features.

Ambient occlusion, screen-space reflections, screen-space GI and volumetric fog
all need the same "blur horizontally into A, then vertically into B" dance, each
with its own edge-preserving kernel. :class:`BlurPass` owns that ping-pong pair
of render targets so a feature only has to supply its source texture plus the
G-Buffer inputs the kernel needs.
"""

from __future__ import annotations

import moderngl

from engine.gfx.shader_utils import load_shader


class BlurPass:
    """Owns a ping-pong texture pair and runs a horizontal + vertical blur.

    Args:
        ctx: ModernGL context.
        width: Target width in pixels.
        height: Target height in pixels.
        shader: Fragment shader implementing the kernel (defaults to the
            depth-aware ``bilateral_blur.frag``).
        components: Channel count of the ping-pong textures (1 for AO, 4 for
            colour features).
        dtype: ModernGL texture dtype (``"f1"`` for LDR, ``"f2"`` for HDR).
    """

    __slots__ = (
        "ctx",
        "width",
        "height",
        "components",
        "dtype",
        "textures",
        "fbos",
        "prog",
        "vao",
        "_u_direction",
        "_u_sigma",
        "_u_radius",
        "_dir_h",
        "_dir_v",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        *,
        shader: str = "bilateral_blur.frag",
        components: int = 1,
        dtype: str = "f1",
    ) -> None:
        self.ctx = ctx
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)
        self.components = components
        self.dtype = dtype
        self.textures: list[moderngl.Texture] = []
        self.fbos: list[moderngl.Framebuffer] = []
        self._create_targets()

        vertex_shader = load_shader("fullscreen_quad.vert")
        self.prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=load_shader(shader))
        self.vao = ctx.vertex_array(self.prog, [])

        self._u_direction = self.prog.get("u_BlurDirection", None)
        self._u_sigma = self.prog.get("u_DepthSigma", None)
        self._u_radius = self.prog.get("u_BlurRadius", None)
        self._dir_h = (1.0 / self.width, 0.0)
        self._dir_v = (0.0, 1.0 / self.height)

    def _create_targets(self) -> None:
        for _ in range(2):
            texture = self.ctx.texture((self.width, self.height), self.components, dtype=self.dtype)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            self.textures.append(texture)
            self.fbos.append(self.ctx.framebuffer(color_attachments=[texture]))

    def _release_targets(self) -> None:
        for framebuffer in self.fbos:
            framebuffer.release()
        for texture in self.textures:
            texture.release()
        self.fbos.clear()
        self.textures.clear()

    def resize(self, width: int, height: int) -> None:
        """Reallocates the ping-pong pair when the target resolution changes."""
        new_w = max(int(width), 1)
        new_h = max(int(height), 1)
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h
        self._release_targets()
        self._create_targets()
        self._dir_h = (1.0 / self.width, 0.0)
        self._dir_v = (0.0, 1.0 / self.height)

    def blur_hv(
        self,
        source_tex: moderngl.Texture,
        depth_tex: moderngl.Texture,
        normal_tex: moderngl.Texture | None = None,
        radius: float = 1.0,
        sigma: float | None = None,
    ) -> moderngl.Texture:
        """Runs the horizontal then vertical blur and returns the result texture.

        Args:
            source_tex: Texture to filter (bound to unit 0).
            depth_tex: G-Buffer depth used by the edge-preserving kernel (unit 1).
            normal_tex: Optional G-Buffer normal texture (unit 2).
            radius: Kernel tap spacing multiplier.
            sigma: Edge tolerance; ignored when the kernel does not expose one.
        """
        if self._u_radius is not None:
            self._u_radius.value = float(radius)
        if sigma is not None and self._u_sigma is not None:
            self._u_sigma.value = float(sigma)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Horizontal: source -> textures[0]
        self.fbos[0].use()
        source_tex.use(location=0)
        depth_tex.use(location=1)
        if normal_tex is not None:
            normal_tex.use(location=2)
        if self._u_direction is not None:
            self._u_direction.value = self._dir_h
        self.vao.render(moderngl.TRIANGLES, vertices=3)

        # Vertical: textures[0] -> textures[1]
        self.fbos[1].use()
        self.textures[0].use(location=0)
        depth_tex.use(location=1)
        if normal_tex is not None:
            normal_tex.use(location=2)
        if self._u_direction is not None:
            self._u_direction.value = self._dir_v
        self.vao.render(moderngl.TRIANGLES, vertices=3)

        return self.textures[1]

    def destroy(self) -> None:
        self._release_targets()
        self.vao.release()
        self.prog.release()
