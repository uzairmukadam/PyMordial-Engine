"""Default Engine World and Map Builder.

Provides out-of-the-box ground plane generation with POM displacement,
material atlas upload, static physics box collider, and atmospheric directional lighting.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

from engine.gfx.texture_atlas import (
    DisplacementMode,
    decode_material_folder,
    encode_mat_flags,
)
from engine.assets.mesh_format import PMMesh, build_pm_mesh
from engine.world.base import BaseWorldBuilder

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp


def make_tiled_plane_pm_mesh(size: float = 100.0, uv_tiles: float = 50.0) -> PMMesh:
    """Builds a high-precision 32-byte cache-aligned PMMesh quad with tiled UVs and tangents for POM."""
    half_s = size * 0.5
    pos = np.array(
        [
            [-half_s, 0.0, half_s],
            [half_s, 0.0, half_s],
            [half_s, 0.0, -half_s],
            [-half_s, 0.0, -half_s],
        ],
        dtype=np.float32,
    )
    nor = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    uvs = np.array(
        [
            [0.0, 0.0],
            [uv_tiles, 0.0],
            [uv_tiles, uv_tiles],
            [0.0, uv_tiles],
        ],
        dtype=np.float32,
    )
    tan = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return build_pm_mesh(pos, nor, uvs, tan, idx)


class DefaultWorldBuilder(BaseWorldBuilder):
    """Standard engine world generator creating a tiled POM ground plane and physics boundary."""

    __slots__ = (
        "size",
        "uv_tiles",
        "material_name",
        "friction",
        "restitution",
        "ground_entity_id",
    )

    def __init__(
        self,
        size: float = 100.0,
        uv_tiles: float = 50.0,
        material_name: str = "mud_cracked_dry_03",
        friction: float = 0.80,
        restitution: float = 0.05,
    ) -> None:
        self.size = float(size)
        self.uv_tiles = float(uv_tiles)
        self.material_name = material_name
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.ground_entity_id: int | None = None

    def build_world(self, app: ProjectApp) -> None:
        """Constructs the tiled ground mesh, loads PBR material, and creates the static collider."""
        pipeline = app.pipeline
        ecs = app.ecs
        physics = app.physics

        # 1. PBR Texture Array Atlas upload
        raw_mat_dir = Path("assets/textures") / self.material_name
        cooked_mat_dir = Path("build/cooked_assets/textures") / self.material_name
        target_res = 4096

        layer_idx = 1
        if raw_mat_dir.is_dir():
            decoded = decode_material_folder(
                folder=raw_mat_dir,
                width=target_res,
                height=target_res,
                name=self.material_name,
                cooked_folder=cooked_mat_dir if cooked_mat_dir.is_dir() else None,
            )
            layer_idx = pipeline.texture_atlas.upload_decoded_layer(decoded, rebuild_mipmaps=True)

        # 2. Add tiled plane to MegaBuffer
        tiled_pm = make_tiled_plane_pm_mesh(size=self.size, uv_tiles=self.uv_tiles)
        alloc_plane = pipeline.mega_buffer.add_pm_mesh("pom_plane", tiled_pm)

        # Rebind pipeline VAOs with new mesh
        pipeline.csm_vao = pipeline.mega_buffer.get_vao(pipeline.csm_prog)
        pipeline.gbuffer_vao = pipeline.mega_buffer.get_vao(pipeline.gbuffer_prog)
        if pipeline.gbuffer_tess_prog is not None:
            pipeline.gbuffer_tess_vao = pipeline.mega_buffer.get_vao(pipeline.gbuffer_tess_prog, mode=pipeline.ctx.PATCHES)
        if pipeline.csm_tess_prog is not None:
            pipeline.csm_tess_vao = pipeline.mega_buffer.get_vao(pipeline.csm_tess_prog, mode=pipeline.ctx.PATCHES)

        # 3. Create static ground entity & collider
        plane_id = ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            color=(0.65, 0.55, 0.42),
            roughness=0.85,
            metallic=0.02,
        )
        physics.create_body(plane_id, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(
            plane_id,
            half_x=self.size * 0.5,
            half_y=0.5,
            half_z=self.size * 0.5,
            friction=self.friction,
            restitution=self.restitution,
        )

        d_plane = ecs.pool.get_dense_index(plane_id)
        if d_plane >= 0:
            ecs.material_data[d_plane, 6] = float(layer_idx)
            ecs.material_data[d_plane, 7] = encode_mat_flags(
                has_texture=(layer_idx > 0),
                disp_mode=DisplacementMode.POM,
            )

        self.ground_entity_id = plane_id

        # Register draw batch with the app's MDI submission
        app.register_draw_batch(alloc_plane, 1, d_plane, False)

    def teardown_world(self, app: ProjectApp) -> None:
        """Removes the ground plane entity and collider."""
        if self.ground_entity_id is not None:
            app.physics.remove_body(self.ground_entity_id)
            app.ecs.destroy_entity(self.ground_entity_id)
            self.ground_entity_id = None
