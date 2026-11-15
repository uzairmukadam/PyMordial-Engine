"""Multi-Draw Indirect (MDI) Command Batcher and Submitter."""

from __future__ import annotations
import numpy as np
import moderngl

from engine.gfx.mega_buffer import MeshAllocation
from engine.logging.logger import log_warn

# Extensions required for the true single-dispatch hardware path.
# - GL_ARB_multi_draw_indirect: glMultiDrawElementsIndirect
# - GL_ARB_shader_draw_parameters: gl_BaseInstanceARB in the vertex shader
_REQUIRED_EXTENSIONS = ("GL_ARB_multi_draw_indirect", "GL_ARB_shader_draw_parameters")


class MultiDrawIndirect:
    """Batches and executes GPU Multi-Draw Indirect commands.

    Command packing contract
    ------------------------
    ``MegaBuffer`` pre-offsets every mesh's indices into the shared global IBO
    (``global_indices = indices + base_vertex``). The indirect ``baseVertex``
    field is therefore always written as ``0``; adding ``alloc.base_vertex``
    again would double-apply the offset and drop the first ``base_vertex``
    vertices of every mesh. Per-instance entity lookup is carried by
    ``baseInstance`` (``gl_BaseInstanceARB``), not by ``baseVertex``.
    """

    # struct DrawElementsIndirectCommand { uint32 count; uint32 instanceCount; uint32 firstIndex; int32 baseVertex; uint32 baseInstance; };
    # 5 uint32 elements = 20 bytes per command
    COMMAND_DTYPE = np.dtype(
        [
            ("count", np.uint32),
            ("instance_count", np.uint32),
            ("first_index", np.uint32),
            ("base_vertex", np.int32),
            ("base_instance", np.uint32),
        ]
    )
    COMMAND_BYTES = 20

    __slots__ = (
        "ctx",
        "max_commands",
        "command_buffer",
        "gpu_indirect_buffer",
        "command_count",
        "hardware_mdi",
        "dropped_commands",
    )

    def __init__(self, ctx: moderngl.Context, max_commands: int = 10_000) -> None:
        self.ctx = ctx
        self.max_commands = max_commands
        self.command_buffer = np.zeros(max_commands, dtype=self.COMMAND_DTYPE)
        self.command_count = 0
        self.dropped_commands = 0

        # Resolve the hardware capability once; the driver extension set is immutable.
        extensions = getattr(ctx, "extensions", None) or ()
        self.hardware_mdi = all(ext in extensions for ext in _REQUIRED_EXTENSIONS)
        if not self.hardware_mdi:
            missing = [ext for ext in _REQUIRED_EXTENSIONS if ext not in extensions]
            log_warn(
                "gfx.mdi",
                f"Hardware multi-draw indirect unavailable (missing {', '.join(missing)}); "
                "falling back to per-command draws.",
            )

        # GPU indirect buffer
        byte_size = max_commands * self.COMMAND_BYTES
        self.gpu_indirect_buffer = self.ctx.buffer(reserve=byte_size)

    def begin_frame(self) -> None:
        """Resets indirect command counter for the frame."""
        self.command_count = 0

    def add_command(
        self,
        alloc: MeshAllocation,
        instance_count: int = 1,
        base_instance: int = 0,
    ) -> None:
        """Records an indirect draw command into contiguous memory."""
        if self.command_count >= self.max_commands:
            self.dropped_commands += 1
            return

        cmd = self.command_buffer[self.command_count]
        cmd["count"] = alloc.index_count
        cmd["instance_count"] = instance_count
        cmd["first_index"] = alloc.first_index
        # Indices are already offset into global IBO space by MegaBuffer.
        cmd["base_vertex"] = 0
        cmd["base_instance"] = base_instance
        self.command_count += 1

    def submit(
        self,
        vao: moderngl.VertexArray,
        program: moderngl.Program | None = None,
        mode: int | None = None,
    ) -> None:
        """Uploads commands and submits via hardware Multi-Draw Indirect."""
        if self.command_count == 0:
            return

        render_mode = mode if mode is not None else vao.mode

        if self.hardware_mdi:
            # Fast path: true hardware multi-draw indirect (single GPU dispatch)
            raw_bytes = self.command_buffer[: self.command_count].tobytes()
            self.gpu_indirect_buffer.write(raw_bytes)
            vao.render_indirect(
                self.gpu_indirect_buffer,
                mode=render_mode,
                count=self.command_count,
                first=0,
            )
            return

        # Fallback path: one draw call per command, carrying baseInstance via uniform.
        base_inst_uniform = program.get("u_BaseInstance", None) if program is not None else None
        for i in range(self.command_count):
            cmd = self.command_buffer[i]
            if base_inst_uniform is not None:
                base_inst_uniform.value = int(cmd["base_instance"])
            vao.render(
                mode=render_mode,
                vertices=int(cmd["count"]),
                first=int(cmd["first_index"]),
                instances=int(cmd["instance_count"]),
            )

    def destroy(self) -> None:
        self.gpu_indirect_buffer.release()
