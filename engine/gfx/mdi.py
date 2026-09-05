"""Multi-Draw Indirect (MDI) Command Batcher and Submitter."""

from __future__ import annotations
import numpy as np
import moderngl

from engine.gfx.mega_buffer import MeshAllocation


class MultiDrawIndirect:
    """Batches and executes GPU Multi-Draw Indirect commands."""

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

    __slots__ = (
        "ctx",
        "max_commands",
        "command_buffer",
        "gpu_indirect_buffer",
        "command_count",
    )

    def __init__(self, ctx: moderngl.Context, max_commands: int = 10_000) -> None:
        self.ctx = ctx
        self.max_commands = max_commands
        self.command_buffer = np.zeros(max_commands, dtype=self.COMMAND_DTYPE)
        self.command_count = 0

        # GPU indirect buffer
        byte_size = max_commands * 20
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
            return

        cmd = self.command_buffer[self.command_count]
        cmd["count"] = alloc.index_count
        cmd["instance_count"] = instance_count
        cmd["first_index"] = alloc.first_index
        cmd["base_vertex"] = alloc.base_vertex
        cmd["base_instance"] = base_instance
        self.command_count += 1

    def submit(
        self,
        vao: moderngl.VertexArray,
        program: moderngl.Program,
    ) -> None:
        """Uploads commands and submits via hardware Multi-Draw Indirect."""
        if self.command_count == 0:
            return

        base_inst_uniform = program.get("u_BaseInstance", None)
        for i in range(self.command_count):
            cmd = self.command_buffer[i]
            if base_inst_uniform is not None:
                base_inst_uniform.value = int(cmd["base_instance"])
            vao.render(
                mode=moderngl.TRIANGLES,
                vertices=int(cmd["count"]),
                first=int(cmd["first_index"]),
                instances=int(cmd["instance_count"]),
            )

    def destroy(self) -> None:
        self.gpu_indirect_buffer.release()
