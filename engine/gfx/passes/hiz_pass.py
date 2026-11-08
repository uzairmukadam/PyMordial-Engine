"""Hierarchical-Z (Hi-Z) Depth Pyramid Pass for PyMordial Engine.

Downsamples the full-resolution Reversed-Z depth buffer into an analytical
depth mipmap pyramid using GPU compute shaders. Accelerates screen-space
raymarching (SSR, SSGI), contact shadows, and GPU occlusion culling.
"""

from __future__ import annotations
import math
import moderngl
import numpy as np

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
        "cull_prog",
        "_u_cull_vp",
        "_u_cull_screen",
        "_u_cull_count",
        "_u_cull_mips",
        "_u_cull_reverse_z",
        "ssbo_aabb",
        "ssbo_visibility",
        "_aabb_scratch",
        "_max_cull_entities",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        reverse_z: bool = True,
        max_entities: int = 100_000,
    ) -> None:
        super().__init__(name="HiZPass", enabled=True)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.reverse_z = reverse_z
        self._max_cull_entities = max_entities

        self.copy_prog = self.ctx.compute_shader(_load_shader("hiz_copy.comp"))
        self.downsample_prog = self.ctx.compute_shader(_load_shader("hiz_downsample.comp"))
        self.cull_prog = self.ctx.compute_shader(_load_shader("cull_indirect.comp"))

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

        # Uniforms for GPU occlusion culling
        self._u_cull_vp = self.cull_prog.get("u_ViewProjection", None)
        self._u_cull_screen = self.cull_prog.get("u_ScreenSize", None)
        self._u_cull_count = self.cull_prog.get("u_EntityCount", None)
        self._u_cull_mips = self.cull_prog.get("u_HiZMipCount", None)
        self._u_cull_reverse_z = self.cull_prog.get("u_ReverseZ", None)
        if self._u_cull_reverse_z is not None:
            self._u_cull_reverse_z.value = 1 if self.reverse_z else 0

        # Pre-allocated GPU storage buffers for AABBs and Visibility mask
        # AABB: 2 * vec4 (32 bytes) per entity. Visibility: uint (4 bytes) per entity.
        self.ssbo_aabb = self.ctx.buffer(reserve=max_entities * 32)
        self.ssbo_visibility = self.ctx.buffer(reserve=max_entities * 4)
        self._aabb_scratch = np.zeros((max_entities, 2, 4), dtype=np.float32)

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

    def cull_aabbs_gpu(
        self,
        aabbs: np.ndarray,
        active_count: int,
        vp_matrix: np.ndarray,
        screen_size: tuple[float, float],
    ) -> np.ndarray:
        """Executes GPU-driven Hi-Z occlusion culling and returns a visibility mask uint8/bool array.

        Args:
            aabbs: Contiguous (active_count, 6) float32 slice: [cx, cy, cz, hx, hy, hz].
            active_count: Total active entities.
            vp_matrix: 16-element column-major float32 array.
            screen_size: (width, height) in pixels.

        Returns:
            Boolean ndarray of shape (active_count,) where True indicates visible geometry.
        """
        if active_count <= 0:
            return np.ones(0, dtype=bool)

        count = min(active_count, self._max_cull_entities)

        # Pack into (count, 2, 4) std430 layout without runtime reallocation
        self._aabb_scratch[:count, 0, 0:3] = aabbs[:count, 0:3] # center
        self._aabb_scratch[:count, 0, 3] = aabbs[:count, 3]     # extent.x
        self._aabb_scratch[:count, 1, 0:2] = aabbs[:count, 4:6] # extent.y, extent.z
        self._aabb_scratch[:count, 1, 2:4] = 0.0                # padding/flags

        # Upload to SSBO 5
        self.ssbo_aabb.write(self._aabb_scratch[:count].tobytes(), offset=0)
        self.ssbo_aabb.bind_to_storage_buffer(5)
        self.ssbo_visibility.bind_to_storage_buffer(6)

        # Bind Hi-Z pyramid sampler
        self.hiz_texture.use(location=0)

        # Set Uniforms
        if self._u_cull_vp is not None:
            self._u_cull_vp.write(vp_matrix.tobytes())
        if self._u_cull_screen is not None:
            self._u_cull_screen.value = (float(screen_size[0]), float(screen_size[1]))
        if self._u_cull_count is not None:
            self._u_cull_count.value = count
        if self._u_cull_mips is not None:
            self._u_cull_mips.value = self.num_mips
        if self._u_cull_reverse_z is not None:
            self._u_cull_reverse_z.value = 1 if self.reverse_z else 0

        # Run compute shader: 64 threads per workgroup
        group_x = (count + 63) // 64
        self.cull_prog.run(group_x=group_x, group_y=1, group_z=1)

        # Read back visibility bitmask
        raw_mask = self.ssbo_visibility.read(count * 4)
        vis_uint32 = np.frombuffer(raw_mask, dtype=np.uint32)
        return vis_uint32 != 0

    def destroy(self) -> None:
        if hasattr(self, "hiz_texture") and self.hiz_texture is not None:
            self.hiz_texture.release()
        if hasattr(self, "ssbo_aabb") and self.ssbo_aabb is not None:
            self.ssbo_aabb.release()
        if hasattr(self, "ssbo_visibility") and self.ssbo_visibility is not None:
            self.ssbo_visibility.release()
