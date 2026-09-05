"""Compressed & Pre-Filtered GPU Texture Format (.pm_tex) for PyMordial Engine.

Provides zero-runtime-filtering texture streaming:
- Binary header with format enum (RGBA8, BC7, BC5), dimensions, and mip levels
- Explicit per-level mipmap table (width, height, byte offset, byte size)
- Pre-baked mipmap pyramid generated offline for zero runtime CPU overhead
- Direct level-by-level streaming into ModernGL Texture handles
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    import moderngl

# 4-byte magic string: "PMTX" (PyMordial TeXture) = 0x504D5458
PM_TEX_MAGIC = b"PMTX"
PM_TEX_VERSION = 1
PM_TEX_HEADER_SIZE = 28
MIP_DESCRIPTOR_SIZE = 16


class TextureFormat(IntEnum):
    """Supported GPU texture format encodings."""

    RGBA8_UNORM = 1     # Standard 32-bit RGBA (4 bytes/pixel)
    BC7_UNORM = 2       # BPTC 8-bit/channel block compression (1 byte/pixel)
    BC5_UNORM = 3       # RGTC2 dual-channel normal compression (1 byte/pixel)
    RGB8_UNORM = 4      # Standard 24-bit RGB (3 bytes/pixel)


@dataclass(slots=True)
class MipLevelDescriptor:
    """Byte offset and dimensions for a single pre-baked mip level."""

    level: int
    width: int
    height: int
    offset: int
    byte_length: int


@dataclass(slots=True)
class PMTex:
    """Pre-cooked binary texture containing complete pre-baked mipmap chain."""

    width: int
    height: int
    format: TextureFormat
    mip_levels: list[MipLevelDescriptor]
    data: bytes | memoryview

    @property
    def mip_count(self) -> int:
        return len(self.mip_levels)

    def get_mip_bytes(self, level: int) -> bytes | memoryview:
        """Extracts the slice of raw bytes corresponding to a specific mip level."""
        if 0 <= level < len(self.mip_levels):
            desc = self.mip_levels[level]
            return self.data[desc.offset : desc.offset + desc.byte_length]
        raise IndexError(f"Mip level {level} out of range [0, {len(self.mip_levels)})")

    def serialize(self) -> bytes:
        """Serializes the texture into a contiguous .pm_tex binary buffer."""
        # 28-byte Header: 4s (magic) + 6I (version, w, h, fmt, mips, flags)
        header = struct.pack(
            "<4sIIIIII",
            PM_TEX_MAGIC,
            PM_TEX_VERSION,
            self.width,
            self.height,
            int(self.format),
            len(self.mip_levels),
            0,  # flags
        )
        assert len(header) == PM_TEX_HEADER_SIZE

        # Mip descriptor table
        table_parts = []
        for desc in self.mip_levels:
            table_parts.append(
                struct.pack("<IIII", desc.width, desc.height, desc.offset, desc.byte_length)
            )
        table_bytes = b"".join(table_parts)
        payload = bytes(self.data) if isinstance(self.data, memoryview) else self.data

        return header + table_bytes + payload

    def save(self, filepath: str | Path) -> None:
        """Writes texture to disk at the given path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.serialize())

    @classmethod
    def from_bytes(cls, raw: bytes | memoryview) -> PMTex:
        """Deserializes a .pm_tex binary buffer."""
        if not isinstance(raw, memoryview):
            mv = memoryview(raw)
        else:
            mv = raw

        if len(mv) < PM_TEX_HEADER_SIZE:
            raise ValueError(f"Data size ({len(mv)}) smaller than header ({PM_TEX_HEADER_SIZE})")

        header = mv[:PM_TEX_HEADER_SIZE]
        magic, version, width, height, fmt_int, mip_count, _ = struct.unpack("<4sIIIIII", header)

        if magic != PM_TEX_MAGIC:
            raise ValueError(f"Invalid .pm_tex magic: {magic!r}, expected {PM_TEX_MAGIC!r}")
        if version != PM_TEX_VERSION:
            raise ValueError(f"Unsupported .pm_tex version {version}")

        table_start = PM_TEX_HEADER_SIZE
        table_end = table_start + mip_count * MIP_DESCRIPTOR_SIZE

        if len(mv) < table_end:
            raise ValueError("Corrupted .pm_tex: incomplete mip descriptor table")

        descriptors: list[MipLevelDescriptor] = []
        for i in range(mip_count):
            entry_offset = table_start + i * MIP_DESCRIPTOR_SIZE
            mw, mh, data_off, data_len = struct.unpack(
                "<IIII", mv[entry_offset : entry_offset + MIP_DESCRIPTOR_SIZE]
            )
            descriptors.append(
                MipLevelDescriptor(
                    level=i,
                    width=mw,
                    height=mh,
                    offset=data_off,
                    byte_length=data_len,
                )
            )

        data_payload = mv[table_end:]
        return cls(
            width=width,
            height=height,
            format=TextureFormat(fmt_int),
            mip_levels=descriptors,
            data=data_payload,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> PMTex:
        """Loads and deserializes a .pm_tex file from disk."""
        return cls.from_bytes(Path(filepath).read_bytes())

    def upload_to_gpu(self, ctx: moderngl.Context) -> moderngl.Texture:
        """Streams pre-filtered mipmap levels directly into a ModernGL Texture handle."""
        components = 4 if self.format in (TextureFormat.RGBA8_UNORM, TextureFormat.BC7_UNORM) else 3
        # Create base texture using Level 0 dimensions
        tex = ctx.texture((self.width, self.height), components=components)
        if self.mip_count > 1:
            tex.build_mipmaps(0, self.mip_count - 1)

        for i, desc in enumerate(self.mip_levels):
            mip_bytes = self.get_mip_bytes(i)
            tex.write(mip_bytes, level=i)

        return tex


def cook_image_to_pm_tex(
    img: Image.Image,
    target_format: TextureFormat = TextureFormat.RGBA8_UNORM,
    generate_mips: bool = True,
) -> PMTex:
    """Cooks a PIL Image into a complete PMTex structure with pre-filtered mipmaps."""
    rgba_img = img.convert("RGBA")
    w, h = rgba_img.size

    mip_images: list[Image.Image] = [rgba_img]
    if generate_mips:
        curr_w, curr_h = w, h
        curr_img = rgba_img
        while curr_w > 1 or curr_h > 1:
            curr_w = max(1, curr_w // 2)
            curr_h = max(1, curr_h // 2)
            downsampled = curr_img.resize((curr_w, curr_h), Image.Resampling.BOX)
            mip_images.append(downsampled)
            curr_img = downsampled

    descriptors: list[MipLevelDescriptor] = []
    data_parts: list[bytes] = []
    current_offset = 0

    for i, mip in enumerate(mip_images):
        raw_bytes = mip.tobytes()
        b_len = len(raw_bytes)
        descriptors.append(
            MipLevelDescriptor(
                level=i,
                width=mip.width,
                height=mip.height,
                offset=current_offset,
                byte_length=b_len,
            )
        )
        data_parts.append(raw_bytes)
        current_offset += b_len

    all_data = b"".join(data_parts)
    return PMTex(
        width=w,
        height=h,
        format=target_format,
        mip_levels=descriptors,
        data=all_data,
    )
