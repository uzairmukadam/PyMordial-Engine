"""Temporal pass output-buffer tests.

Guards the ping-pong contract: the texture a temporal pass reports as its output
must be the one it wrote *this* frame. A swap performed before the caller reads
the result hands back the previous frame's history, which shows up as temporal
wobble and ghosting.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.gfx.context import RenderContext
from engine.gfx.frame_context import FrameContext
from engine.gfx.g_buffer import GBuffer
from engine.gfx.passes.smaa_pass import SMAAPass
from engine.gfx.passes.taa_pass import TAAPass
from engine.gfx.quality_presets import RenderConfig
from engine.gfx.render_graph import RenderGraphContext


@pytest.fixture(scope="module")
def render_ctx():
    ctx = RenderContext.create_headless(width=64, height=64)
    yield ctx
    ctx.destroy()


def _make_context(render_ctx: RenderContext, config: RenderConfig) -> RenderGraphContext:
    g_buffer = GBuffer(render_ctx.ctx, 64, 64)
    frame_context = FrameContext(render_ctx.ctx)
    context = RenderGraphContext(
        ctx=render_ctx.ctx,
        width=64,
        height=64,
        config=config,
        frame_context=frame_context,
        resources={"g_buffer": g_buffer},
    )
    return context


class TestTemporalOutputContract:
    """The reported output must be the buffer written during this execute call."""

    def test_taa_output_is_the_buffer_written_this_frame(self, render_ctx: RenderContext) -> None:
        config = RenderConfig(aa_mode="TAA", taa_enabled=True)
        taa = TAAPass(render_ctx.ctx, 64, 64)
        context = _make_context(render_ctx, config)

        source = render_ctx.ctx.texture((64, 64), 4, dtype="f2")
        source.write(np.full((64, 64, 4), 0.5, dtype=np.float16).tobytes())
        context.resources["hdr_color"] = source

        try:
            taa.execute(context)
            reported = context.resources["taa_output"]
            written_fbo = taa.output_fbo
            assert reported is written_fbo.color_attachments[0], (
                "TAAPass reported a texture that is not the attachment of the "
                "framebuffer it just wrote. The ping-pong swap ran too early."
            )
        finally:
            taa.destroy()
            source.release()

    def test_taa_output_alternates_between_history_buffers(self, render_ctx: RenderContext) -> None:
        """Consecutive frames must write different buffers, and each must be reported."""
        config = RenderConfig(aa_mode="TAA", taa_enabled=True)
        taa = TAAPass(render_ctx.ctx, 64, 64)
        context = _make_context(render_ctx, config)

        source = render_ctx.ctx.texture((64, 64), 4, dtype="f2")
        source.write(np.full((64, 64, 4), 0.5, dtype=np.float16).tobytes())
        context.resources["hdr_color"] = source

        try:
            seen: list[int] = []
            for _ in range(4):
                taa.execute(context)
                reported = context.resources["taa_output"]
                assert reported is taa.output_fbo.color_attachments[0]
                seen.append(reported.glo)

            assert len(set(seen)) == 2, (
                f"TAA should alternate between exactly two history buffers, saw {len(set(seen))}."
            )
            assert seen[0] != seen[1], "Consecutive TAA frames wrote the same buffer."
        finally:
            taa.destroy()
            source.release()

    def test_smaa_output_is_the_buffer_written_this_frame(self, render_ctx: RenderContext) -> None:
        config = RenderConfig(aa_mode="SMAA_4X")
        smaa = SMAAPass(render_ctx.ctx, 64, 64)
        context = _make_context(render_ctx, config)

        ldr = render_ctx.ctx.texture((64, 64), 4, dtype="f1")
        ldr.write(np.full((64, 64, 4), 128, dtype=np.uint8).tobytes())
        context.resources["ldr_color"] = ldr

        try:
            smaa.execute(context)
            reported = context.resources["smaa_output"]
            assert reported is smaa.output_fbo.color_attachments[0], (
                "SMAAPass reported a texture that is not the attachment of the "
                "framebuffer it just wrote. The ping-pong swap ran too early."
            )
        finally:
            smaa.destroy()
            ldr.release()

    def test_smaa_1x_reports_neighborhood_result(self, render_ctx: RenderContext) -> None:
        """SMAA 1x has no temporal stage, so it must report the neighborhood buffer."""
        config = RenderConfig(aa_mode="SMAA_1X")
        smaa = SMAAPass(render_ctx.ctx, 64, 64)
        context = _make_context(render_ctx, config)

        ldr = render_ctx.ctx.texture((64, 64), 4, dtype="f1")
        ldr.write(np.full((64, 64, 4), 128, dtype=np.uint8).tobytes())
        context.resources["ldr_color"] = ldr

        try:
            smaa.execute(context)
            assert context.resources["smaa_output"] is smaa.tex_smaa
        finally:
            smaa.destroy()
            ldr.release()
