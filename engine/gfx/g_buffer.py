"""MRT G-Buffer Framebuffer with Reversed-Z Depth32F."""

from __future__ import annotations
import moderngl


class GBuffer:
    """Manages the Multi-Render-Target (MRT) G-Buffer Framebuffer.

    Attachments:
    - RT0: Albedo (RGB) + Roughness (A) (RGBA8)
    - RT1: Octahedral Normal (RG) + Metallic (B) + AO (A) (RGBA16F)
    - RT2: Velocity Vectors (RG16F)
    - RT3: Displacement Info (RG16F) -> R=TextureLayerIndex, G=DisplacementMode
    - Depth: 32-bit Floating Point Depth (Reversed-Z)
    """

    __slots__ = (
        "ctx",
        "width",
        "height",
        "reverse_z",
        "rt_albedo_roughness",
        "rt_normal_metallic",
        "rt_velocity",
        "rt_displacement",
        "depth_texture",
        "fbo",
    )

    @property
    def albedo_roughness_texture(self) -> moderngl.Texture:
        return self.rt_albedo_roughness

    @property
    def normal_metallic_texture(self) -> moderngl.Texture:
        return self.rt_normal_metallic

    @property
    def velocity_texture(self) -> moderngl.Texture:
        return self.rt_velocity

    @property
    def displacement_texture(self) -> moderngl.Texture:
        return self.rt_displacement

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        reverse_z: bool = True,
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.reverse_z = reverse_z

        self._create_buffers()

    def _create_buffers(self) -> None:
        # RT0: Albedo (RGB) + Roughness (A) (RGBA8)
        self.rt_albedo_roughness = self.ctx.texture(
            (self.width, self.height), 4, dtype="f1"
        )
        self.rt_albedo_roughness.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # RT1: Octahedral Normal (RG) + Metallic (B) + AO (A) (RGBA16F)
        self.rt_normal_metallic = self.ctx.texture(
            (self.width, self.height), 4, dtype="f2"
        )
        self.rt_normal_metallic.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # RT2: Velocity Vectors (RG16F)
        self.rt_velocity = self.ctx.texture(
            (self.width, self.height), 2, dtype="f2"
        )
        self.rt_velocity.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # RT3: Displacement Info (RGBA16F) -> RG=Mesh UV, B=TexLayerIndex, A=DisplacementMode
        self.rt_displacement = self.ctx.texture(
            (self.width, self.height), 4, dtype="f2"
        )
        self.rt_displacement.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Depth: 32-bit Floating Point Depth (Reversed-Z)
        self.depth_texture = self.ctx.depth_texture((self.width, self.height))
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False

        self.fbo = self.ctx.framebuffer(
            color_attachments=[
                self.rt_albedo_roughness,
                self.rt_normal_metallic,
                self.rt_velocity,
                self.rt_displacement,
            ],
            depth_attachment=self.depth_texture,
        )

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width <= 0 or new_height <= 0:
            return
        if new_width == self.width and new_height == self.height:
            return
        self.destroy()
        self.width = new_width
        self.height = new_height
        self._create_buffers()

    def clear(self) -> None:
        """Clears G-Buffer attachments. In Reversed-Z, depth is cleared to 0.0."""
        self.fbo.use()
        clear_depth = 0.0 if self.reverse_z else 1.0
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=clear_depth)

    def bind_textures(self, base_unit: int = 0) -> None:
        """Binds RT0, RT1, and Depth to consecutive texture units for lighting resolve."""
        self.rt_albedo_roughness.use(location=base_unit)
        self.rt_normal_metallic.use(location=base_unit + 1)
        self.depth_texture.use(location=base_unit + 2)

    def destroy(self) -> None:
        self.rt_albedo_roughness.release()
        self.rt_normal_metallic.release()
        self.rt_velocity.release()
        self.rt_displacement.release()
        self.depth_texture.release()
        self.fbo.release()
