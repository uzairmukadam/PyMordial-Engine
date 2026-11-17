"""PBR Material Registry Module (Re-exports from engine.core.materials)."""

from engine.core.materials import (
    DEFAULT_MATERIAL_DEPTHS,
    MAT_FLAG_DISP_MASK,
    MAT_FLAG_DISP_SHIFT,
    MAT_FLAG_HAS_TEXTURE,
    DisplacementMode,
    MaterialDef,
    MaterialRegistry,
    decode_mat_flags,
    encode_mat_flags,
)

__all__ = [
    "DisplacementMode",
    "MAT_FLAG_HAS_TEXTURE",
    "MAT_FLAG_DISP_SHIFT",
    "MAT_FLAG_DISP_MASK",
    "encode_mat_flags",
    "decode_mat_flags",
    "DEFAULT_MATERIAL_DEPTHS",
    "MaterialDef",
    "MaterialRegistry",
]
