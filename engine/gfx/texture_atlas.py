"""GPU Texture Array Atlas for PBR Material Binding and Displacement Modes.

Manages layered sampler2DArray atlases for per-entity PBR texture maps:
- Diffuse (Albedo): RGBA8 sRGB array (Texture Unit 10)
- Normal Map: RGBA8 tangent-space normal array (Texture Unit 11)
- Displacement (Height): R8/R16F heightfield array for Tessellation, POM and SSDM (Texture Unit 12)
- ARM (AO/Roughness/Metallic): RGBA8 packed channel array (Texture Unit 13)

All layers within each array share identical resolution (e.g. 4096, 2048, 1024).
Layer 0 is reserved as the "no texture" identity layer (flat white, flat normal, zero height).
Entities reference their texture set via layer index in material_data[entity, 6]
and their displacement mode via material_data[entity, 7].
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import moderngl


@dataclass(slots=True)
class DecodedMaterialLayer:
    """Pre-decoded raw pixel buffers for all 4 PBR channels ready for GPU upload."""
    name: str
    diffuse_bytes: bytes
    normal_bytes: bytes
    disp_bytes: bytes
    arm_bytes: bytes
    width: int
    height: int


class DisplacementMode(IntEnum):
    """Mutually exclusive displacement modes per entity."""
    NONE = 0          # Standard PBR (normal mapping only, no displacement)
    POM = 1           # Parallax Occlusion Mapping (fragment shader raymarch + self-shadowing)
    SSDM = 2          # Screen-Space Displacement Mapping (post-G-buffer depth extrusion)
    TESSELLATION = 3  # Hardware Tessellation (GPU TCS+TES vertex displacement along normal)


# Material flag bitfield layout stored in material_data[entity, 7]:
# Bit 0: MAT_FLAG_HAS_TEXTURE (1 = sample texture arrays)
# Bits 1..2: Displacement Mode (2 bits: 0=None, 1=POM, 2=SSDM, 3=Tessellation)
MAT_FLAG_HAS_TEXTURE = 1 << 0
MAT_FLAG_DISP_SHIFT = 1
MAT_FLAG_DISP_MASK = 0x3


def encode_mat_flags(has_texture: bool, disp_mode: DisplacementMode = DisplacementMode.NONE) -> float:
    """Encodes material flags and displacement mode into a float for material_data[entity, 7]."""
    flags = (1 if has_texture else 0) | ((int(disp_mode) & MAT_FLAG_DISP_MASK) << MAT_FLAG_DISP_SHIFT)
    return float(flags)


def decode_mat_flags(flag_val: float | int) -> tuple[bool, DisplacementMode]:
    """Decodes material flags and displacement mode from material_data[entity, 7]."""
    val = int(flag_val)
    has_texture = bool(val & MAT_FLAG_HAS_TEXTURE)
    disp_mode = DisplacementMode((val >> MAT_FLAG_DISP_SHIFT) & MAT_FLAG_DISP_MASK)
    return has_texture, disp_mode


def _fit_image(img: Image.Image, w: int, h: int) -> Image.Image:
    if img.size == (w, h):
        return img
    resample = Image.Resampling.BOX if (img.width >= w and img.height >= h) else Image.Resampling.BILINEAR
    return img.resize((w, h), resample)


def decode_material_folder(
    folder: str | Path,
    width: int,
    height: int,
    name: str | None = None,
) -> DecodedMaterialLayer:
    """Pure CPU worker function to read and decode all 4 PBR texture maps.

    Safe to execute on background worker threads without OpenGL/ModernGL context.
    """
    p = Path(folder)
    mat_name = name or p.name
    diff_path = p / "diff.png"
    nor_path = p / "nor.png"
    disp_path = p / "disp.png"
    arm_path = p / "arm.png"

    # 1. Diffuse (RGBA)
    if diff_path.is_file():
        img = Image.open(diff_path).convert("RGBA")
        img = _fit_image(img, width, height)
        diff_bytes = img.tobytes()
    else:
        diff_bytes = np.full(width * height * 4, 255, dtype=np.uint8).tobytes()

    # 2. Normal (RGBA tangent space)
    if nor_path.is_file():
        img = Image.open(nor_path).convert("RGBA")
        img = _fit_image(img, width, height)
        nor_bytes = img.tobytes()
    else:
        flat = np.zeros(width * height * 4, dtype=np.uint8)
        flat[0::4] = 128
        flat[1::4] = 128
        flat[2::4] = 255
        flat[3::4] = 255
        nor_bytes = flat.tobytes()

    # 3. Displacement (R8)
    if disp_path.is_file():
        raw_disp = Image.open(disp_path)
        if raw_disp.mode == "I;16":
            arr = np.array(raw_disp, dtype=np.uint16)
            arr_8 = (arr / 256).astype(np.uint8)
            disp_img = Image.fromarray(arr_8, mode="L")
        elif raw_disp.mode in ("RGBA", "RGB"):
            arr = np.array(raw_disp)
            disp_img = Image.fromarray(arr[..., 0], mode="L")
        else:
            disp_img = raw_disp.convert("L")
        disp_img = _fit_image(disp_img, width, height)
        disp_bytes = disp_img.tobytes()
    else:
        disp_bytes = np.full(width * height, 128, dtype=np.uint8).tobytes()

    # 4. ARM (AO, Roughness, Metallic, Unused)
    if arm_path.is_file():
        img = Image.open(arm_path).convert("RGBA")
        img = _fit_image(img, width, height)
        arm_bytes = img.tobytes()
    else:
        arm = np.zeros(width * height * 4, dtype=np.uint8)
        arm[0::4] = 255
        arm[1::4] = 128
        arm[2::4] = 0
        arm[3::4] = 255
        arm_bytes = arm.tobytes()

    return DecodedMaterialLayer(
        name=mat_name,
        diffuse_bytes=diff_bytes,
        normal_bytes=nor_bytes,
        disp_bytes=disp_bytes,
        arm_bytes=arm_bytes,
        width=width,
        height=height,
    )



class TextureArrayAtlas:
    """Manages layered sampler2DArray GPU atlases for PBR material maps.

    Each atlas is a ModernGL Texture3D (sampler2DArray) where the Z axis
    indexes individual material layers. All layers share a fixed resolution.
    """

    __slots__ = (
        "ctx",
        "width",
        "height",
        "max_layers",
        "_next_layer",
        "material_names",
        "name_to_layer",
        "diffuse_array",
        "normal_array",
        "displacement_array",
        "arm_array",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int = 2048,
        height: int = 2048,
        max_layers: int = 32,
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.max_layers = max_layers
        self._next_layer = 0
        self.material_names: list[str] = ["identity"]
        self.name_to_layer: dict[str, int] = {"identity": 0}

        # Create 2D array textures (sampler2DArray) with pre-allocated layers
        # Diffuse: RGBA8 (sRGB albedo)
        self.diffuse_array = ctx.texture_array(
            (width, height, max_layers), 4, dtype="f1",
        )
        self.diffuse_array.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
        self.diffuse_array.anisotropy = 16.0
        self.diffuse_array.repeat_x = True
        self.diffuse_array.repeat_y = True

        # Normal: RGBA8 (tangent-space XYZ packed as [0,1])
        self.normal_array = ctx.texture_array(
            (width, height, max_layers), 4, dtype="f1",
        )
        self.normal_array.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
        self.normal_array.anisotropy = 16.0
        self.normal_array.repeat_x = True
        self.normal_array.repeat_y = True

        # Displacement: single-channel R8 heightfield (0.0=low, 1.0=high)
        self.displacement_array = ctx.texture_array(
            (width, height, max_layers), 1, dtype="f1",
        )
        self.displacement_array.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
        self.displacement_array.anisotropy = 16.0
        self.displacement_array.repeat_x = True
        self.displacement_array.repeat_y = True

        # ARM: RGBA8 (R=AO, G=Roughness, B=Metallic, A=unused)
        self.arm_array = ctx.texture_array(
            (width, height, max_layers), 4, dtype="f1",
        )
        self.arm_array.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
        self.arm_array.anisotropy = 16.0
        self.arm_array.repeat_x = True
        self.arm_array.repeat_y = True

        # Write identity layer 0 (no texture / flat defaults)
        self._write_identity_layer()

    def _write_identity_layer(self) -> None:
        """Writes layer 0 as identity: white diffuse, flat normal, zero height, default ARM."""
        w, h = self.width, self.height
        pixel_count = w * h

        # Diffuse: solid white (255, 255, 255, 255)
        white = np.full(pixel_count * 4, 255, dtype=np.uint8)
        self.diffuse_array.write(white.tobytes(), viewport=(0, 0, 0, w, h, 1))

        # Normal: flat up (128, 128, 255, 255) = tangent-space (0, 0, 1)
        flat_normal = np.zeros(pixel_count * 4, dtype=np.uint8)
        flat_normal[0::4] = 128  # X
        flat_normal[1::4] = 128  # Y
        flat_normal[2::4] = 255  # Z
        flat_normal[3::4] = 255  # W
        self.normal_array.write(flat_normal.tobytes(), viewport=(0, 0, 0, w, h, 1))

        # Displacement: mid height (128)
        mid_height = np.full(pixel_count, 128, dtype=np.uint8)
        self.displacement_array.write(mid_height.tobytes(), viewport=(0, 0, 0, w, h, 1))

        # ARM: default (AO=255, Roughness=128, Metallic=0, A=255)
        arm_data = np.zeros(pixel_count * 4, dtype=np.uint8)
        arm_data[0::4] = 255  # AO = 1.0
        arm_data[1::4] = 128  # Roughness = 0.5
        arm_data[2::4] = 0    # Metallic = 0.0
        arm_data[3::4] = 255  # Unused
        self.arm_array.write(arm_data.tobytes(), viewport=(0, 0, 0, w, h, 1))

        self._next_layer = 1

    def add_layer_from_images(
        self,
        diffuse_path: str | Path | None = None,
        normal_path: str | Path | None = None,
        displacement_path: str | Path | None = None,
        arm_path: str | Path | None = None,
        name: str = "",
        rebuild_mipmaps: bool = True,
    ) -> int:
        """Loads PBR texture files, resizes to atlas resolution, and uploads as a new layer."""
        if self._next_layer >= self.max_layers:
            raise RuntimeError(
                f"TextureArrayAtlas full: {self._next_layer}/{self.max_layers} layers"
            )

        layer = self._next_layer
        w, h = self.width, self.height

        # Diffuse
        if diffuse_path is not None and Path(diffuse_path).is_file():
            img = Image.open(diffuse_path).convert("RGBA")
            img = _fit_image(img, w, h)
            self.diffuse_array.write(img.tobytes(), viewport=(0, 0, layer, w, h, 1))
        else:
            white = np.full(w * h * 4, 255, dtype=np.uint8)
            self.diffuse_array.write(white.tobytes(), viewport=(0, 0, layer, w, h, 1))

        # Normal
        if normal_path is not None and Path(normal_path).is_file():
            img = Image.open(normal_path).convert("RGBA")
            img = _fit_image(img, w, h)
            self.normal_array.write(img.tobytes(), viewport=(0, 0, layer, w, h, 1))
        else:
            flat = np.zeros(w * h * 4, dtype=np.uint8)
            flat[0::4] = 128
            flat[1::4] = 128
            flat[2::4] = 255
            flat[3::4] = 255
            self.normal_array.write(flat.tobytes(), viewport=(0, 0, layer, w, h, 1))

        # Displacement
        if displacement_path is not None and Path(displacement_path).is_file():
            raw_disp = Image.open(displacement_path)
            # Handle 16-bit or RGBA
            if raw_disp.mode == "I;16":
                arr = np.array(raw_disp, dtype=np.uint16)
                arr_8 = (arr / 256).astype(np.uint8)
                disp_img = Image.fromarray(arr_8, mode="L")
            elif raw_disp.mode in ("RGBA", "RGB"):
                # Take Red channel
                arr = np.array(raw_disp)
                disp_img = Image.fromarray(arr[..., 0], mode="L")
            else:
                disp_img = raw_disp.convert("L")

            disp_img = _fit_image(disp_img, w, h)
            self.displacement_array.write(disp_img.tobytes(), viewport=(0, 0, layer, w, h, 1))
        else:
            mid = np.full(w * h, 128, dtype=np.uint8)
            self.displacement_array.write(mid.tobytes(), viewport=(0, 0, layer, w, h, 1))

        # ARM (AO, Roughness, Metallic)
        if arm_path is not None and Path(arm_path).is_file():
            img = Image.open(arm_path).convert("RGBA")
            img = _fit_image(img, w, h)
            self.arm_array.write(img.tobytes(), viewport=(0, 0, layer, w, h, 1))
        else:
            arm = np.zeros(w * h * 4, dtype=np.uint8)
            arm[0::4] = 255
            arm[1::4] = 128
            arm[2::4] = 0
            arm[3::4] = 255
            self.arm_array.write(arm.tobytes(), viewport=(0, 0, layer, w, h, 1))

        mat_name = name or f"material_{layer}"
        self.material_names.append(mat_name)
        self.name_to_layer[mat_name] = layer
        self._next_layer = layer + 1

        if rebuild_mipmaps:
            self.rebuild_all_mipmaps()

        return layer

    def upload_decoded_layer(
        self,
        decoded: DecodedMaterialLayer,
        rebuild_mipmaps: bool = False,
    ) -> int:
        """Uploads pre-decoded PBR texture buffers into the GPU texture array on the main thread."""
        if self._next_layer >= self.max_layers:
            raise RuntimeError(
                f"TextureArrayAtlas full: {self._next_layer}/{self.max_layers} layers"
            )

        layer = self._next_layer
        w, h = self.width, self.height

        diff_bytes = decoded.diffuse_bytes
        norm_bytes = decoded.normal_bytes
        disp_bytes = decoded.disp_bytes
        arm_bytes = decoded.arm_bytes

        if decoded.width != w or decoded.height != h:
            img_diff = Image.frombytes("RGBA", (decoded.width, decoded.height), diff_bytes)
            diff_bytes = _fit_image(img_diff, w, h).tobytes()
            img_norm = Image.frombytes("RGBA", (decoded.width, decoded.height), norm_bytes)
            norm_bytes = _fit_image(img_norm, w, h).tobytes()
            img_disp = Image.frombytes("L", (decoded.width, decoded.height), disp_bytes)
            disp_bytes = _fit_image(img_disp, w, h).tobytes()
            img_arm = Image.frombytes("RGBA", (decoded.width, decoded.height), arm_bytes)
            arm_bytes = _fit_image(img_arm, w, h).tobytes()

        self.diffuse_array.write(diff_bytes, viewport=(0, 0, layer, w, h, 1))
        self.normal_array.write(norm_bytes, viewport=(0, 0, layer, w, h, 1))
        self.displacement_array.write(disp_bytes, viewport=(0, 0, layer, w, h, 1))
        self.arm_array.write(arm_bytes, viewport=(0, 0, layer, w, h, 1))

        mat_name = decoded.name or f"material_{layer}"
        self.material_names.append(mat_name)
        self.name_to_layer[mat_name] = layer
        self._next_layer = layer + 1

        if rebuild_mipmaps:
            self.rebuild_all_mipmaps()

        return layer

    def load_materials_from_folder(
        self,
        textures_dir: str | Path,
        preferred_order: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Loads all Poly Haven material folders from directory into atlas layers."""
        p = Path(textures_dir)
        if not p.is_dir():
            return self.name_to_layer

        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if preferred_order:
            ordered = []
            for name in preferred_order:
                for d in subdirs:
                    if d.name == name:
                        ordered.append(d)
                        break
            for d in subdirs:
                if d not in ordered:
                    ordered.append(d)
            subdirs = ordered
        else:
            subdirs.sort(key=lambda d: d.name)

        total = len(subdirs)
        for i, d in enumerate(subdirs):
            diff = d / "diff.png"
            nor = d / "nor.png"
            disp = d / "disp.png"
            arm = d / "arm.png"
            self.add_layer_from_images(
                diffuse_path=diff if diff.exists() else None,
                normal_path=nor if nor.exists() else None,
                displacement_path=disp if disp.exists() else None,
                arm_path=arm if arm.exists() else None,
                name=d.name,
                rebuild_mipmaps=False,
            )
            if progress_callback is not None:
                progress_callback(i + 1, total, d.name)

        self.rebuild_all_mipmaps()
        return self.name_to_layer

    def rebuild_all_mipmaps(self) -> None:
        """Rebuilds mipmaps for all 4 texture arrays."""
        self.diffuse_array.build_mipmaps()
        self.normal_array.build_mipmaps()
        self.displacement_array.build_mipmaps()
        self.arm_array.build_mipmaps()

    @property
    def layer_count(self) -> int:
        """Returns the number of allocated material layers (including identity layer 0)."""
        return self._next_layer

    def bind(
        self,
        diffuse_unit: int = 10,
        normal_unit: int = 11,
        displacement_unit: int = 12,
        arm_unit: int = 13,
    ) -> None:
        """Binds all 4 sampler2DArray atlases to the specified texture units."""
        self.diffuse_array.use(location=diffuse_unit)
        self.normal_array.use(location=normal_unit)
        self.displacement_array.use(location=displacement_unit)
        self.arm_array.use(location=arm_unit)

    def destroy(self) -> None:
        """Releases all GPU texture array resources."""
        self.diffuse_array.release()
        self.normal_array.release()
        self.displacement_array.release()
        self.arm_array.release()
