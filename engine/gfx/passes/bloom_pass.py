"""Thresholded HDR bloom pyramid."""

from __future__ import annotations

import moderngl

from engine.gfx.render_graph import RenderPass
from engine.gfx.shader_utils import load_shader


class BloomPass(RenderPass):
    """Builds a soft, multiscale glow from bright HDR scene regions."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "downsample_prog",
        "upsample_prog",
        "downsample_vao",
        "upsample_vao",
        "_u_threshold",
        "downsample_textures",
        "downsample_fbos",
        "upsample_textures",
        "upsample_fbos",
    )

    MAX_LEVELS = 6

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        super().__init__(name="BloomPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        vertex_shader = load_shader("fullscreen_quad.vert")
        self.downsample_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=load_shader("bloom_downsample.frag"),
        )
        self.upsample_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=load_shader("bloom_upsample.frag"),
        )
        self.downsample_vao = self.ctx.vertex_array(self.downsample_prog, [])
        self.upsample_vao = self.ctx.vertex_array(self.upsample_prog, [])
        self.downsample_prog["u_Source"].value = 0
        self.upsample_prog["u_LowResolution"].value = 0
        self.upsample_prog["u_HighResolution"].value = 1
        self._u_threshold = self.downsample_prog["u_Threshold"]
        self.downsample_textures: list[moderngl.Texture] = []
        self.downsample_fbos: list[moderngl.Framebuffer] = []
        self.upsample_textures: list[moderngl.Texture] = []
        self.upsample_fbos: list[moderngl.Framebuffer] = []
        self._create_buffers()

    def _create_buffers(self) -> None:
        self._destroy_buffers()
        for level in range(self.MAX_LEVELS):
            level_width = max(1, self.width >> (level + 1))
            level_height = max(1, self.height >> (level + 1))
            texture = self.ctx.texture((level_width, level_height), 4, dtype="f2")
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.downsample_textures.append(texture)
            self.downsample_fbos.append(self.ctx.framebuffer(color_attachments=[texture]))

            upsample_texture = self.ctx.texture((level_width, level_height), 4, dtype="f2")
            upsample_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.upsample_textures.append(upsample_texture)
            self.upsample_fbos.append(self.ctx.framebuffer(color_attachments=[upsample_texture]))

    def _destroy_buffers(self) -> None:
        for framebuffer in self.downsample_fbos:
            framebuffer.release()
        for texture in self.downsample_textures:
            texture.release()
        for framebuffer in self.upsample_fbos:
            framebuffer.release()
        for texture in self.upsample_textures:
            texture.release()
        self.downsample_fbos.clear()
        self.downsample_textures.clear()
        self.upsample_fbos.clear()
        self.upsample_textures.clear()

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0 or (width == self.width and height == self.height):
            return
        self.width = width
        self.height = height
        self._create_buffers()

    def render(
        self,
        scene_hdr: moderngl.Texture,
        threshold: float,
        iterations: int,
    ) -> moderngl.Texture:
        """Returns a half-resolution HDR texture containing the blurred highlights."""
        level_count = max(1, min(self.MAX_LEVELS, int(iterations)))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        source = scene_hdr
        for level in range(level_count):
            framebuffer = self.downsample_fbos[level]
            framebuffer.use()
            self.ctx.viewport = (0, 0, *self.downsample_textures[level].size)
            source.use(location=0)
            self._u_threshold.value = float(threshold) if level == 0 else -1.0
            self.downsample_vao.render(mode=moderngl.TRIANGLES, vertices=3)
            source = self.downsample_textures[level]

        for level in range(level_count - 2, -1, -1):
            framebuffer = self.upsample_fbos[level]
            framebuffer.use()
            self.ctx.viewport = (0, 0, *self.upsample_textures[level].size)
            lower_level = level + 1
            lower_texture = (
                self.upsample_textures[lower_level]
                if lower_level < level_count - 1
                else self.downsample_textures[lower_level]
            )
            lower_texture.use(location=0)
            self.downsample_textures[level].use(location=1)
            self.upsample_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        return self.upsample_textures[0] if level_count > 1 else self.downsample_textures[0]

    def execute(self, context) -> None:
        resources = getattr(context, "resources", {})
        scene_hdr = resources.get("hdr_color")
        config = getattr(context, "config", None)
        if scene_hdr is not None and config is not None and getattr(config, "bloom_enabled", True):
            resources["bloom_texture"] = self.render(
                scene_hdr,
                float(getattr(config, "bloom_threshold", 1.0)),
                int(getattr(config, "bloom_iterations", 5)),
            )

    def destroy(self) -> None:
        self._destroy_buffers()
        self.downsample_vao.release()
        self.upsample_vao.release()
        self.downsample_prog.release()
        self.upsample_prog.release()
