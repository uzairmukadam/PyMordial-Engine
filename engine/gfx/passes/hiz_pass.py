"""Hierarchical-Z (Hi-Z) Depth Pyramid Pass for PyMordial Engine.

Downsamples the full-resolution Reversed-Z depth buffer into an analytical
depth mipmap pyramid using GPU compute shaders. Accelerates screen-space
raymarching (SSR, SSGI), contact shadows, and GPU occlusion culling.
"""

from __future__ import annotations
import math
import moderngl

from engine.gfx.render_graph import RenderPass, RenderGraphContext


from engine.gfx.shader_utils import get_shader_dir, load_shader

SHADER_DIR = get_shader_dir()
_load_shader = load_shader



class HiZPass(RenderPass):
    """GPU Compute-based Hierarchical-Z Depth Pyramid Pass."""

    __slots__ = (
        "ctx",
        "width",
        "height",
        "reverse_z",
        "num_mips",
        "mip_sizes",
        "dispatch_groups",
        "hiz_texture",
        "copy_prog",
        "downsample_prog",
        "_u_copy_depth",
        "_u_down_src_tex",
        "_u_down_src_level",
        "_u_down_reverse_z",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        reverse_z: bool = True,
    ) -> None:
        super().__init__(name="HiZPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.reverse_z = reverse_z

        self.copy_prog = self.ctx.compute_shader(_load_shader("hiz_copy.comp"))
        self.downsample_prog = self.ctx.compute_shader(_load_shader("hiz_downsample.comp"))

        self._u_copy_depth = self.copy_prog.get("u_DepthTexture", None)
        if self._u_copy_depth is not None:
            self._u_copy_depth.value = 0

        self._u_down_src_tex = self.downsample_prog.get("u_SourceTexture", None)
        if self._u_down_src_tex is not None:
            self._u_down_src_tex.value = 0

        self._u_down_src_level = self.downsample_prog.get("u_SrcLevel", None)
        self._u_down_reverse_z = self.downsample_prog.get("u_ReverseZ", None)
        if self._u_down_reverse_z is not None:
            self._u_down_reverse_z.value = 1 if self.reverse_z else 0

        self._allocate_pyramid()

    def _allocate_pyramid(self) -> None:
        self.num_mips = int(math.floor(math.log2(max(self.width, self.height)))) + 1
        self.mip_sizes = []
        self.dispatch_groups = []
        w, h = self.width, self.height
        for _ in range(self.num_mips):
            self.mip_sizes.append((w, h))
            self.dispatch_groups.append(((w + 7) // 8, (h + 7) // 8))
            w = max(1, w // 2)
            h = max(1, h // 2)

        self.hiz_texture = self.ctx.texture((self.width, self.height), 1, dtype="f4")
        self.hiz_texture.filter = (moderngl.NEAREST_MIPMAP_NEAREST, moderngl.NEAREST)
        self.hiz_texture.repeat_x = False
        self.hiz_texture.repeat_y = False
        self.hiz_texture.build_mipmaps()

    def resize(self, width: int, height: int) -> None:
        if self.width == width and self.height == height:
            return
        self.width = width
        self.height = height
        if hasattr(self, "hiz_texture") and self.hiz_texture is not None:
            self.hiz_texture.release()
        self._allocate_pyramid()

    def execute(
        self,
        ctx: RenderGraphContext,
        depth_texture: moderngl.Texture | None = None,
    ) -> None:
        if not self.enabled:
            return

        src_depth = depth_texture
        if src_depth is None:
            g_buffer = ctx.resources.get("g_buffer", None)
            if g_buffer is not None:
                src_depth = getattr(g_buffer, "depth_texture", None)

        if src_depth is None:
            return

        # 1. Level 0 Copy: Copy hardware/SSDM depth into Hi-Z mip 0
        src_depth.use(location=0)
        self.hiz_texture.bind_to_image(0, read=False, write=True, level=0)
        gx0, gy0 = self.dispatch_groups[0]
        self.copy_prog.run(group_x=gx0, group_y=gy0)

        # 2. Downsample: Iteratively build mips 1..num_mips-1
        self.hiz_texture.use(location=0)

        for k in range(1, self.num_mips):
            if self._u_down_src_level is not None:
                self._u_down_src_level.value = k - 1

            self.hiz_texture.bind_to_image(0, read=False, write=True, level=k)
            gx, gy = self.dispatch_groups[k]
            self.downsample_prog.run(group_x=gx, group_y=gy)

        # Publish Hi-Z depth pyramid resources to graph context
        ctx.resources["hiz_texture"] = self.hiz_texture
        ctx.resources["hiz_mip_count"] = self.num_mips

    def destroy(self) -> None:
        if hasattr(self, "hiz_texture") and self.hiz_texture is not None:
            self.hiz_texture.release()
