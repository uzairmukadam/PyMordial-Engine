"""Temporary AO probe: one resolution per process to avoid multi-context artifacts."""

import sys

import numpy as np
from PIL import Image

W = int(sys.argv[1])
H = int(sys.argv[2])
MODE = int(sys.argv[3]) if len(sys.argv) > 3 else 7

from engine.app.project_app import ProjectApp
from projects.playground.config import get_project_config
from projects.playground.world.arena import PlaygroundArenaBuilder

cfg = get_project_config()
cfg.width, cfg.height = W, H
cfg.headless = True
cfg.enable_debug = False
app = ProjectApp(cfg)
app.set_world_builder(PlaygroundArenaBuilder())
p = app.pipeline
p.config.debug_gbuffer = MODE
p.config.ao_intensity = 1.0
p.apply_config(p.config)

for _ in range(4):
    app.step_frame(1.0 / 60.0)

ao = p.ao_pass
raw = np.frombuffer(ao.raw_ao_tex.read(), dtype=np.uint8).reshape((ao.height, ao.width))
blur = np.frombuffer(ao.blur_pass.textures[1].read(), dtype=np.uint8).reshape((ao.height, ao.width))
fin = np.frombuffer(p.post_process.final_fbo.read(components=3, dtype="f1"), dtype=np.uint8).reshape((H, W, 3))

print(f"RES {W}x{H} mode={MODE}")
print(f"  ao_pass={ao.width}x{ao.height} scale={ao.resolution_scale}")
print(f"  raw_tex={ao.raw_ao_tex.size} blur_tex={ao.blur_pass.textures[1].size}")
print(f"  g_buffer={p.g_buffer.width}x{p.g_buffer.height} hdr={p.post_process.hdr_texture.size}")
print(f"  raw mean={raw.mean():.1f} blur mean={blur.mean():.1f}")
print(f"  blur bottom-half={blur[: ao.height // 2].mean():.1f} top-half={blur[ao.height // 2 :].mean():.1f}")

Image.fromarray(raw[::-1]).save(f"C:/Users/uzair/AppData/Local/Temp/probe_raw_{W}x{H}.png")
Image.fromarray(blur[::-1]).save(f"C:/Users/uzair/AppData/Local/Temp/probe_blur_{W}x{H}.png")
Image.fromarray(fin[::-1]).save(f"C:/Users/uzair/AppData/Local/Temp/probe_fin_{W}x{H}.png")
app.shutdown()
