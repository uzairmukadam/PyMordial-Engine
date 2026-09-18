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


def make_grid_terrain_pm_mesh(size: float = 100.0, grid_cells: int = 32, uv_tiles: float = 1.0) -> PMMesh:
    """Builds a flat regular grid terrain mesh ready for heightmap displacement."""
    cells = max(1, int(grid_cells))
    verts_per_side = cells + 1
    total_verts = verts_per_side * verts_per_side

    half_s = size * 0.5
    xs = np.linspace(-half_s, half_s, verts_per_side, dtype=np.float32)
    zs = np.linspace(-half_s, half_s, verts_per_side, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)

    pos = np.zeros((total_verts, 3), dtype=np.float32)
    pos[:, 0] = grid_x.ravel()
    pos[:, 1] = 0.0
    pos[:, 2] = grid_z.ravel()

    nor = np.zeros((total_verts, 4), dtype=np.float32)
    nor[:, 1] = 1.0
    nor[:, 3] = 1.0

    uvs = np.zeros((total_verts, 2), dtype=np.float32)
    uvs[:, 0] = ((grid_x.ravel() + half_s) / size) * uv_tiles
    uvs[:, 1] = ((grid_z.ravel() + half_s) / size) * uv_tiles

    tan = np.zeros((total_verts, 4), dtype=np.float32)
    tan[:, 0] = 1.0
    tan[:, 3] = 1.0

    indices = []
    for r in range(cells):
        for c in range(cells):
            v0 = r * verts_per_side + c
            v1 = v0 + 1
            v2 = (r + 1) * verts_per_side + c
            v3 = v2 + 1
            indices.extend([v0, v1, v2, v1, v3, v2])

    idx = np.array(indices, dtype=np.uint32)
    return build_pm_mesh(pos, nor, uvs, tan, idx)


class DefaultWorldBuilder(BaseWorldBuilder):
    """Standard engine world generator creating a tiled POM or flat untextured ground plane and physics boundary."""

    __slots__ = (
        "size",
        "uv_tiles",
        "material_name",
        "color",
        "roughness",
        "metallic",
        "friction",
        "restitution",
        "ground_entity_id",
        "ground_collider_id",
    )

    def __init__(
        self,
        size: float = 100.0,
        uv_tiles: float = 50.0,
        material_name: str | None = "mud_cracked_dry_03",
        color: tuple[float, float, float] = (0.65, 0.55, 0.42),
        roughness: float = 0.85,
        metallic: float = 0.02,
        friction: float = 0.80,
        restitution: float = 0.05,
    ) -> None:
        self.size = float(size)
        self.uv_tiles = float(uv_tiles)
        self.material_name = material_name
        self.color = color
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.ground_entity_id: int | None = None
        self.ground_collider_id: int | None = None

    def build_world(self, app: ProjectApp) -> None:
        """Constructs the tiled ground mesh, loads PBR material (if specified), and creates the static collider."""
        pipeline = app.pipeline
        ecs = app.ecs
        physics = app.physics

        layer_idx = 0
        disp_mode = DisplacementMode.NONE

        # 1. PBR Texture Array Atlas upload (if material is specified)
        if self.material_name:
            raw_mat_dir = Path("assets/textures") / self.material_name
            cooked_mat_dir = Path("build/cooked_assets/textures") / self.material_name
            target_res = 4096

            if raw_mat_dir.is_dir():
                decoded = decode_material_folder(
                    folder=raw_mat_dir,
                    width=target_res,
                    height=target_res,
                    name=self.material_name,
                    cooked_folder=cooked_mat_dir if cooked_mat_dir.is_dir() else None,
                )
                layer_idx = pipeline.texture_atlas.upload_decoded_layer(decoded, rebuild_mipmaps=True)
                pipeline.material_registry.sync_with_atlas(pipeline.texture_atlas)
                disp_mode = DisplacementMode.POM

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

        # 3. Create visual ground plane entity (at exact y=0.0)
        plane_id = ecs.create_entity(
            position=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            material_id=self.material_name or "flat_land",
            layer_idx=layer_idx,
            disp_mode=disp_mode,
            color=self.color,
            roughness=self.roughness,
            metallic=self.metallic,
            is_static=True,
        )

        # 4. Create dedicated static physics box collider (centered at y=-0.5, top face flush with y=0.0)
        col_id = ecs.create_entity(
            position=(0.0, -0.5, 0.0),
            scale=(self.size, 1.0, self.size),
            color=(0.0, 0.0, 0.0),
            is_static=True,
        )
        physics.create_body(col_id, body_type="fixed", position=(0.0, -0.5, 0.0))
        physics.attach_box_collider(
            col_id,
            half_x=self.size * 0.5,
            half_y=0.5,
            half_z=self.size * 0.5,
            friction=self.friction,
            restitution=self.restitution,
        )

        ecs.set_entity_material(
            plane_id,
            material_id=self.material_name,
            layer_idx=layer_idx,
            disp_mode=DisplacementMode.POM,
        )

        self.ground_entity_id = plane_id
        self.ground_collider_id = col_id

        # Register draw batch with the app's MDI submission
        d_plane = ecs.pool.get_dense_index(plane_id)
        if d_plane >= 0:
            half_s = self.size * 0.5 + 50.0
            ecs.aabbs[d_plane, 0:3] = (0.0, 0.0, 0.0)
            ecs.aabbs[d_plane, 3] = half_s
            ecs.aabbs[d_plane, 4] = 20.0
            ecs.aabbs[d_plane, 5] = half_s
        app.register_draw_batch(alloc_plane, 1, d_plane, False, cast_shadow=False)

    def teardown_world(self, app: ProjectApp) -> None:
        """Removes the ground plane entity and collider."""
        if self.ground_collider_id is not None:
            app.physics.remove_body(self.ground_collider_id)
            app.ecs.destroy_entity(self.ground_collider_id)
            self.ground_collider_id = None
        if self.ground_entity_id is not None:
            app.ecs.destroy_entity(self.ground_entity_id)
            self.ground_entity_id = None
