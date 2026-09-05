"""Virtual File System (VFS) and Memory-Mapped .pak Container for PyMordial Engine.

Provides zero-copy asset streaming:
- Contiguous single-file .pak archive container
- 64-bit FNV-1a hash table for O(1) directory lookups
- Raw uncompressed streaming via mmap for zero-copy direct OpenGL uploads
- Optional Zstandard compression for distribution
- Multi-mount VFS supporting packed archives and development directory overlays
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import mmap
import os
from pathlib import Path
import struct
from typing import BinaryIO
import zstandard

# 4-byte magic: "PMPK" (PyMordial PacK) = 0x504D504B
PAK_MAGIC = b"PMPK"
PAK_VERSION = 1
PAK_HEADER_SIZE = 24

COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1


def fnv1a_64(text: str) -> int:
    """Computes a 64-bit FNV-1a hash of a normalized virtual path."""
    h = 0xCBF29CE484222325
    for b in text.lower().replace("\\", "/").strip("/").encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def normalize_vpath(vpath: str) -> str:
    """Standardizes path separators and case for uniform VFS resolution."""
    return vpath.replace("\\", "/").strip("/")


@dataclass(slots=True)
class PakEntry:
    """Metadata describing a single file stored inside a .pak archive."""

    vpath: str
    path_hash: int
    offset: int
    compressed_size: int
    uncompressed_size: int
    compression: int


class PakWriter:
    """Constructs a contiguous .pak archive containing multiple assets."""

    def __init__(self) -> None:
        self._files: list[tuple[str, bytes, bool]] = []

    def add_file(self, vpath: str, data: bytes, compress: bool = False) -> None:
        """Queues a file to be bundled into the archive."""
        self._files.append((normalize_vpath(vpath), data, compress))

    def write(self, output_path: str | Path) -> None:
        """Encodes all queued files into the destination .pak file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        compressor = zstandard.ZstdCompressor(level=3)
        entries: list[PakEntry] = []
        data_blobs: list[bytes] = []

        current_offset = PAK_HEADER_SIZE
        for vpath, raw_bytes, do_compress in self._files:
            uncomp_len = len(raw_bytes)
            if do_compress:
                comp_bytes = compressor.compress(raw_bytes)
                comp_len = len(comp_bytes)
                method = COMPRESSION_ZSTD
                payload = comp_bytes
            else:
                comp_bytes = raw_bytes
                comp_len = uncomp_len
                method = COMPRESSION_NONE
                payload = raw_bytes

            entries.append(
                PakEntry(
                    vpath=vpath,
                    path_hash=fnv1a_64(vpath),
                    offset=current_offset,
                    compressed_size=comp_len,
                    uncompressed_size=uncomp_len,
                    compression=method,
                )
            )
            data_blobs.append(payload)
            current_offset += comp_len

        table_offset = current_offset

        # Build directory table:
        # For each entry:
        # Q (path_hash) + Q (offset) + Q (comp_size) + Q (uncomp_size) + H (method) + H (path_len) + path_bytes
        table_parts = []
        for e in entries:
            p_bytes = e.vpath.encode("utf-8")
            table_parts.append(
                struct.pack(
                    "<QQQQHH",
                    e.path_hash,
                    e.offset,
                    e.compressed_size,
                    e.uncompressed_size,
                    e.compression,
                    len(p_bytes),
                )
                + p_bytes
            )
        table_bytes = b"".join(table_parts)

        # Header: 4s (magic) + I (version) + I (entry_count) + Q (table_offset) + I (flags)
        header = struct.pack(
            "<4sIIQI",
            PAK_MAGIC,
            PAK_VERSION,
            len(entries),
            table_offset,
            0,
        )
        assert len(header) == PAK_HEADER_SIZE

        with open(out_p, "wb") as f:
            f.write(header)
            for blob in data_blobs:
                f.write(blob)
            f.write(table_bytes)


class PakReader:
    """Provides memory-mapped random access to files packed in a .pak container."""

    __slots__ = (
        "filepath",
        "_file",
        "_mmap",
        "_entries",
        "_hash_to_entry",
        "_decompressor",
    )

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f".pak file not found: {self.filepath}")

        self._file: BinaryIO = open(self.filepath, "rb")
        file_size = os.path.getsize(self.filepath)
        if file_size < PAK_HEADER_SIZE:
            raise ValueError(f"Corrupt .pak: file size ({file_size}) smaller than header ({PAK_HEADER_SIZE})")

        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._entries: dict[str, PakEntry] = {}
        self._hash_to_entry: dict[int, PakEntry] = {}
        self._decompressor = zstandard.ZstdDecompressor()

        self._read_directory()

    def _read_directory(self) -> None:
        """Parses directory index table at the end of the archive."""
        header = self._mmap[:PAK_HEADER_SIZE]
        magic, version, count, table_offset, _ = struct.unpack("<4sIIQI", header)

        if magic != PAK_MAGIC:
            raise ValueError(f"Invalid .pak magic: {magic!r}, expected {PAK_MAGIC!r}")
        if version != PAK_VERSION:
            raise ValueError(f"Unsupported .pak version: {version}")

        pos = table_offset
        m_len = len(self._mmap)

        for _ in range(count):
            if pos + 36 > m_len:
                raise ValueError("Corrupt .pak table: truncated entry header")

            p_hash, off, comp_sz, uncomp_sz, method, p_len = struct.unpack(
                "<QQQQHH", self._mmap[pos : pos + 36]
            )
            pos += 36
            if pos + p_len > m_len:
                raise ValueError("Corrupt .pak table: truncated path string")

            p_str = self._mmap[pos : pos + p_len].decode("utf-8")
            pos += p_len

            entry = PakEntry(
                vpath=p_str,
                path_hash=p_hash,
                offset=off,
                compressed_size=comp_sz,
                uncompressed_size=uncomp_sz,
                compression=method,
            )
            self._entries[p_str] = entry
            self._hash_to_entry[p_hash] = entry

    def has_file(self, vpath: str) -> bool:
        """Checks whether a virtual file path exists inside this archive."""
        norm = normalize_vpath(vpath)
        return norm in self._entries or fnv1a_64(norm) in self._hash_to_entry

    def list_files(self) -> list[str]:
        """Returns a list of all normalized file paths contained in the archive."""
        return list(self._entries.keys())

    def try_read(self, vpath: str, zero_copy: bool = True) -> memoryview | bytes | None:
        """Attempts to read a file by virtual path, returning None if not found."""
        norm = normalize_vpath(vpath)
        entry = self._entries.get(norm)
        if entry is None:
            entry = self._hash_to_entry.get(fnv1a_64(norm))

        if entry is None:
            return None

        raw_slice = memoryview(self._mmap)[entry.offset : entry.offset + entry.compressed_size]

        if entry.compression == COMPRESSION_NONE:
            return raw_slice if zero_copy else bytes(raw_slice)
        elif entry.compression == COMPRESSION_ZSTD:
            return self._decompressor.decompress(raw_slice, max_output_size=entry.uncompressed_size)
        else:
            raise ValueError(f"Unknown compression method {entry.compression} for {vpath}")

    def read(self, vpath: str, zero_copy: bool = True) -> memoryview | bytes:
        """Reads a file by virtual path.

        Args:
            vpath: Virtual path of asset.
            zero_copy: If True and uncompressed, returns a zero-copy memoryview.
                       If False, returns a standalone bytes object.

        Returns:
            A memoryview or bytes object containing the file payload.
        """
        res = self.try_read(vpath, zero_copy=zero_copy)
        if res is None:
            raise FileNotFoundError(f"File not found in .pak: '{vpath}'")
        return res

    def read_bytes(self, vpath: str) -> bytes:
        """Reads and returns a standalone bytes copy of the requested virtual file."""
        return bytes(self.read(vpath, zero_copy=False))

    def close(self) -> None:
        """Releases memory map and file handle."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self) -> PakReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VFS:
    """Unified Virtual File System coordinating mounts across archives and folders."""

    def __init__(self) -> None:
        self._pak_mounts: list[PakReader] = []
        self._dir_mounts: list[Path] = []

    def mount_pak(self, pak_path: str | Path) -> PakReader:
        """Mounts a .pak container into the virtual file system."""
        reader = PakReader(pak_path)
        self._pak_mounts.insert(0, reader)  # Prepend for highest priority
        return reader

    def mount_dir(self, directory_path: str | Path) -> None:
        """Mounts a loose directory on disk for development live overrides."""
        p = Path(directory_path).resolve()
        if p.is_dir():
            self._dir_mounts.insert(0, p)

    def exists(self, vpath: str) -> bool:
        """Returns True if the virtual path exists in any mounted source."""
        norm = normalize_vpath(vpath)
        # Check loose directories first
        for d in self._dir_mounts:
            if (d / norm).exists():
                return True
        # Check mounted .pak containers
        for pak in self._pak_mounts:
            if pak.has_file(norm):
                return True
        return False

    def read(self, vpath: str) -> memoryview | bytes:
        """Reads and returns the contents of a virtual path."""
        norm = normalize_vpath(vpath)
        # Check loose directories
        for d in self._dir_mounts:
            target = d / norm
            if target.is_file():
                return target.read_bytes()

        # Check mounted pak containers with single lookup
        for pak in self._pak_mounts:
            data = pak.try_read(norm)
            if data is not None:
                return data

        raise FileNotFoundError(f"VFS could not resolve '{vpath}' in any mounted container")

    def open_stream(self, vpath: str) -> io.BytesIO:
        """Returns a binary BytesIO stream for the requested virtual path."""
        data = self.read(vpath)
        return io.BytesIO(data)

    def list_files(self, prefix: str = "") -> list[str]:
        """Lists all files accessible in the VFS matching an optional prefix."""
        results: set[str] = set()
        norm_pre = normalize_vpath(prefix)

        for pak in self._pak_mounts:
            for f in pak.list_files():
                if not norm_pre or f.startswith(norm_pre):
                    results.add(f)

        for d in self._dir_mounts:
            for root, _, files in os.walk(d):
                for f in files:
                    rel = Path(root, f).relative_to(d).as_posix()
                    if not norm_pre or rel.startswith(norm_pre):
                        results.add(rel)

        return sorted(results)

    def close(self) -> None:
        """Closes all mounted archives."""
        for pak in self._pak_mounts:
            pak.close()
        self._pak_mounts.clear()
        self._dir_mounts.clear()
