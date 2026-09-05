"""Resource Cache and Asset Manager for PyMordial Engine.

Provides deduplicated, O(1) cached access to cooked meshes and textures:
- Seamlessly resolves assets from Virtual File System (.pak or loose files)
- Caches deserialized PMMesh and PMTex structures
- Caches uploaded ModernGL GPU Texture objects to guarantee zero VRAM duplication
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from engine.assets.mesh_format import PMMesh
from engine.assets.texture_format import PMTex
from engine.assets.vfs import VFS, normalize_vpath

if TYPE_CHECKING:
    import moderngl
    from engine.gfx.mega_buffer import MeshAllocation


class ResourceCache:
    """Coordinates asset retrieval, memory mapping, and GPU handle deduplication."""

    __slots__ = (
        "vfs",
        "_mesh_cache",
        "_tex_cache",
        "_gpu_tex_cache",
        "_gpu_mesh_cache",
    )

    def __init__(self, vfs: VFS | None = None) -> None:
        self.vfs = vfs if vfs is not None else VFS()
        self._mesh_cache: dict[str, PMMesh] = {}
        self._tex_cache: dict[str, PMTex] = {}
        self._gpu_tex_cache: dict[str, moderngl.Texture] = {}
        self._gpu_mesh_cache: dict[str, MeshAllocation] = {}

    def mount_pak(self, pak_path: str | Path) -> None:
        """Mounts a .pak archive into the underlying VFS."""
        self.vfs.mount_pak(pak_path)

    def mount_dir(self, dir_path: str | Path) -> None:
        """Mounts a disk folder into the underlying VFS."""
        self.vfs.mount_dir(dir_path)

    def load_mesh(self, vpath: str) -> PMMesh:
        """Loads and caches a .pm_mesh asset from VFS."""
        norm = normalize_vpath(vpath)
        if norm in self._mesh_cache:
            return self._mesh_cache[norm]

        raw_data = self.vfs.read(norm)
        mesh = PMMesh.from_bytes(raw_data)
        self._mesh_cache[norm] = mesh
        return mesh

    def load_texture(self, vpath: str) -> PMTex:
        """Loads and caches a .pm_tex asset from VFS."""
        norm = normalize_vpath(vpath)
        if norm in self._tex_cache:
            return self._tex_cache[norm]

        raw_data = self.vfs.read(norm)
        tex = PMTex.from_bytes(raw_data)
        self._tex_cache[norm] = tex
        return tex

    def load_gpu_texture(self, vpath: str, ctx: moderngl.Context) -> moderngl.Texture:
        """Loads, uploads, and caches a ModernGL GPU Texture handle."""
        norm = normalize_vpath(vpath)
        if norm in self._gpu_tex_cache:
            return self._gpu_tex_cache[norm]

        pm_tex = self.load_texture(norm)
        gpu_tex = pm_tex.upload_to_gpu(ctx)
        self._gpu_tex_cache[norm] = gpu_tex
        return gpu_tex

    def register_gpu_mesh(self, name: str, alloc: MeshAllocation) -> None:
        """Caches a MegaBuffer MeshAllocation for a named cooked mesh."""
        self._gpu_mesh_cache[normalize_vpath(name)] = alloc

    def get_gpu_mesh(self, name: str) -> MeshAllocation | None:
        """Retrieves a cached MegaBuffer MeshAllocation if previously loaded."""
        return self._gpu_mesh_cache.get(normalize_vpath(name))

    def clear(self) -> None:
        """Releases cached CPU and GPU objects."""
        for tex in self._gpu_tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._mesh_cache.clear()
        self._tex_cache.clear()
        self._gpu_tex_cache.clear()
        self._gpu_mesh_cache.clear()

    def close(self) -> None:
        """Cleans up cache and shuts down VFS mounts."""
        self.clear()
        self.vfs.close()
