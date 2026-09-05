"""PyMordial Engine Asset Pipeline & Virtual File System (VFS).

Exposes:
- PMMesh, MeshBoundingVolumes, build_pm_mesh (32-byte binary mesh format)
- PMTex, TextureFormat, cook_image_to_pm_tex (Pre-filtered GPU textures)
- VFS, PakReader, PakWriter (Memory-mapped archive streaming)
- ResourceCache (Deduplicated asset and GPU handle manager)
"""

from engine.assets.mesh_format import (
    PMMesh,
    MeshBoundingVolumes,
    VERTEX_DTYPE,
    VERTEX_STRIDE,
    build_pm_mesh,
    compute_bounding_volumes,
)
from engine.assets.texture_format import (
    PMTex,
    TextureFormat,
    MipLevelDescriptor,
    cook_image_to_pm_tex,
)
from engine.assets.vfs import (
    VFS,
    PakReader,
    PakWriter,
    PakEntry,
    normalize_vpath,
    fnv1a_64,
)
from engine.assets.resource_cache import ResourceCache

__all__ = [
    "PMMesh",
    "MeshBoundingVolumes",
    "VERTEX_DTYPE",
    "VERTEX_STRIDE",
    "build_pm_mesh",
    "compute_bounding_volumes",
    "PMTex",
    "TextureFormat",
    "MipLevelDescriptor",
    "cook_image_to_pm_tex",
    "VFS",
    "PakReader",
    "PakWriter",
    "PakEntry",
    "normalize_vpath",
    "fnv1a_64",
    "ResourceCache",
]
