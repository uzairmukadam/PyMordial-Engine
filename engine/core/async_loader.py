"""Multi-threaded background asset loader for PyMordial Engine.

Executes CPU-intensive file I/O, image decoding, resampling, and data conversions
across a worker thread pool, streaming decoded data back to the main thread via
thread-safe queues for GPU upload without locking the OS event loop.
"""

from __future__ import annotations
import concurrent.futures
import os
from pathlib import Path
import queue
from typing import Callable, Any
from engine.gfx.texture_atlas import DecodedMaterialLayer, decode_material_folder
from engine.logging import log_info, log_error


class BackgroundAssetLoader:
    """Coordinates asynchronous background CPU decoding of materials and assets."""

    def __init__(self, max_workers: int | None = None) -> None:
        # Default to min(8, cpu_count) workers for parallel 4K decoding
        cpu_workers = max(2, min(8, (os.cpu_count() or 4)))
        self._max_workers = max_workers or cpu_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="PyMordialLoader",
        )
        self._decoded_materials_queue: queue.Queue[DecodedMaterialLayer] = queue.Queue()
        self._material_futures: list[concurrent.futures.Future] = []
        self._total_materials = 0
        self._completed_materials = 0
        self._current_material_name: str = ""
        self._is_active = False

    @property
    def total_materials(self) -> int:
        return self._total_materials

    @property
    def completed_materials(self) -> int:
        return self._completed_materials

    @property
    def current_material_name(self) -> str:
        return self._current_material_name

    @property
    def material_progress(self) -> float:
        """Normalized material loading progress (0.0 to 1.0)."""
        if self._total_materials <= 0:
            return 1.0
        return min(1.0, self._completed_materials / float(self._total_materials))

    @property
    def is_material_loading_complete(self) -> bool:
        """Returns True if all queued material layers have finished decoding and been polled."""
        if not self._is_active:
            return True
        return self._completed_materials >= self._total_materials and self._decoded_materials_queue.empty()

    def start_material_loading(
        self,
        textures_dir: str | Path,
        width: int = 4096,
        height: int = 4096,
        preferred_order: list[str] | None = None,
    ) -> list[str]:
        """Discovers material directories and dispatches parallel decode workers."""
        p = Path(textures_dir)
        if not p.is_dir():
            return []

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

        if not subdirs:
            return []

        self._total_materials = len(subdirs)
        self._completed_materials = 0
        self._is_active = True
        self._material_futures.clear()

        log_info(
            "AsyncLoader",
            f"Dispatching {self._total_materials} material layers across {self._max_workers} worker threads...",
        )

        for d in subdirs:
            future = self._executor.submit(decode_material_folder, d, width, height, d.name)
            future.add_done_callback(self._on_material_decoded)
            self._material_futures.append(future)

        return [d.name for d in subdirs]

    def _on_material_decoded(self, future: concurrent.futures.Future[DecodedMaterialLayer]) -> None:
        """Worker completion callback: puts decoded layer into main thread queue."""
        try:
            layer_data = future.result()
            self._decoded_materials_queue.put(layer_data)
        except Exception as e:
            log_error("AsyncLoader", f"Material decode task failed: {e}")

    def poll_material_layer(self) -> DecodedMaterialLayer | None:
        """Retrieves the next decoded material layer from queue, or None if none available."""
        try:
            layer = self._decoded_materials_queue.get_nowait()
            self._completed_materials += 1
            self._current_material_name = layer.name
            return layer
        except queue.Empty:
            return None

    def submit_generic(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submits an arbitrary task to the thread pool."""
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = False) -> None:
        """Shuts down the background thread pool."""
        self._is_active = False
        self._executor.shutdown(wait=wait)
