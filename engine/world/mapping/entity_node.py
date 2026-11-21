"""Hierarchical Map Entity Node Runtime.

Manages runtime instantiations of atomic and compound entity definitions into
PyMordial's contiguous ECS tables and Rapier3D physics simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.world.mapping.schema import CollisionDef, EntityDef, TransformDef
from engine.world.mapping.transform_math import combine_transforms

if TYPE_CHECKING:
    from engine.app.project_app import ProjectApp
    from engine.world.mapping.prefab_manager import PrefabManager


class MapEntityNode:
    """Runtime node binding an EntityDef to its ECS entity ID and Rapier physics body."""

    __slots__ = (
        "entity_def",
        "world_transform",
        "ecs_id",
        "has_rigid_body",
        "children",
        "properties",
        "scripts",
    )

    def __init__(self, entity_def: EntityDef) -> None:
        self.entity_def = entity_def
        self.world_transform: TransformDef = entity_def.transform
        self.ecs_id: int | None = None
        self.has_rigid_body: bool = False
        self.children: list[MapEntityNode] = []
        self.properties: dict[str, Any] = dict(entity_def.properties)
        self.scripts: list[Any] = []

    @property
    def id(self) -> str:
        return self.entity_def.id

    def spawn(
        self,
        app: ProjectApp,
        parent_transform: TransformDef | None = None,
        prefab_mgr: PrefabManager | None = None,
    ) -> MapEntityNode:
        """Instantiates this entity and all children into ECS and Rapier physics."""
        # 1. Resolve world transform: M_world = M_parent * M_local
        if parent_transform is not None:
            self.world_transform = combine_transforms(parent_transform, self.entity_def.transform)
        else:
            self.world_transform = self.entity_def.transform

        # 2. If this entity references a prefab, resolve template and children
        if self.entity_def.prefab is not None and prefab_mgr is not None:
            prefab_inst = prefab_mgr.instantiate(
                self.entity_def.prefab,
                instance_id=self.entity_def.id,
                property_overrides=self.entity_def.property_overrides,
            )
            # Copy prefab visual/collision if not overridden
            if self.entity_def.visual is None and prefab_inst.visual is not None:
                self.entity_def.visual = prefab_inst.visual
            if self.entity_def.collision is None and prefab_inst.collision is not None:
                self.entity_def.collision = prefab_inst.collision
            if not self.entity_def.children and prefab_inst.children:
                self.entity_def.children = prefab_inst.children
            # Merge properties
            self.properties.update(prefab_inst.properties)

        # 3. Determine visual & collision requirements
        has_visual = (self.entity_def.visual is not None) and self.entity_def.visual.is_visible
        collision = self.entity_def.resolved_collision()
        has_trigger = self.entity_def.trigger is not None

        # 4. Allocate ECS Entity if visual or collision is active
        # (pure triggers without visual or physics do not need an ECS entity)
        if has_visual or (collision is not None and collision.enabled):
            pos = self.world_transform.position
            rot = self.world_transform.rotation
            scale = self.world_transform.scale if has_visual else (0.0, 0.0, 0.0)

            vis = self.entity_def.visual
            shape = vis.shape if (has_visual and vis is not None) else "cube"
            color = vis.color if (has_visual and vis is not None) else (0.0, 0.0, 0.0)
            roughness = vis.roughness if (has_visual and vis is not None) else 0.5
            metallic = vis.metallic if (has_visual and vis is not None) else 0.0
            ao = vis.ao if (has_visual and vis is not None) else 1.0
            emissive = vis.emissive_intensity if (has_visual and vis is not None) else 0.0
            layer_idx = vis.layer_idx if (has_visual and vis is not None) else 0
            mat_id = vis.material if (has_visual and vis is not None) else None

            is_static = (collision.body_type == "fixed") if (collision and collision.enabled) else True

            # Register with ECS
            entity_id = app.ecs.create_entity(
                shape=shape,
                position=pos,
                rotation=rot,
                scale=scale,
                color=color,
                roughness=roughness,
                metallic=metallic,
                ao=ao,
                emissive_intensity=emissive,
                layer_idx=layer_idx,
                material_id=mat_id,
                is_static=is_static,
            )
            self.ecs_id = entity_id

            if not has_visual:
                # Suppress AABB culling extent so invisible colliders/triggers are not drawn
                dense_idx = app.ecs.pool.get_dense_index(entity_id)
                if dense_idx >= 0:
                    app.ecs.local_half_extents[dense_idx] = (0.0, 0.0, 0.0)
                    app.ecs.scales[dense_idx] = (0.0, 0.0, 0.0)
                    app.ecs.recompute_matrix(dense_idx)

            # 5. Attach Rapier physics rigid body and collider if enabled
            if collision is not None and collision.enabled:
                app.physics.create_body(
                    entity_id=entity_id,
                    body_type=collision.body_type,
                    position=pos,
                    rotation=rot,
                )
                self.has_rigid_body = True

                # Determine effective collider dimensions
                dims = collision.dimensions
                app.physics.attach_primitive_collider(
                    entity_id=entity_id,
                    shape_type=collision.shape,
                    dimensions=dims,
                    density=collision.density,
                    friction=collision.friction,
                    restitution=collision.restitution,
                )

                # Configure dynamic properties and register with prop manager
                if collision.body_type == "dynamic":
                    phys = self.entity_def.resolved_physics()
                    lin_damp = phys.linear_damping if phys is not None else 0.05
                    ang_damp = phys.angular_damping if phys is not None else 0.1
                    app.physics.set_damping(entity_id, linear=lin_damp, angular=ang_damp)

                    if hasattr(app, "props") and app.props is not None:
                        from engine.physics.props import PropInstance, PropType

                        shape_lower = collision.shape.lower()
                        if shape_lower in ("box", "cube"):
                            ptype = PropType.CRATE
                        elif shape_lower == "sphere":
                            ptype = PropType.SPHERE
                        elif shape_lower in ("cylinder", "barrel"):
                            ptype = PropType.BARREL if "barrel" in self.entity_def.id.lower() else PropType.CYLINDER
                        elif shape_lower == "cone":
                            ptype = PropType.CONE
                        else:
                            ptype = PropType.CRATE

                        destructible = bool(self.properties.get("destructible", False))
                        health = float(self.properties.get("health", 40.0))

                        prop_inst = PropInstance(
                            entity_id=entity_id,
                            prop_type=ptype,
                            shape_name=collision.shape,
                            scale=self.world_transform.scale,
                            is_destructible=destructible,
                            health=health,
                            max_health=health,
                        )
                        app.props.register_prop(prop_inst)

        # 6. Instantiate attached behavioral scripts
        from engine.world.mapping.script_runner import get_script_class

        for script_def in self.entity_def.scripts:
            cls = get_script_class(script_def.script_type)
            if cls is not None:
                instance = cls()
                instance.on_spawn(self, app, script_def.params)
                self.scripts.append(instance)
                if hasattr(app, "map_manager") and app.map_manager is not None:
                    app.map_manager.script_runner.attach_script(instance)

        # 7. Register trigger zone if defined or auto-derived from SENSOR role
        trigger_def = self.entity_def.trigger
        if trigger_def is None and self.entity_def.physics is not None:
            from engine.world.mapping.schema import PhysicsRole, TriggerDef

            if self.entity_def.physics.role == PhysicsRole.SENSOR:
                dim = self.entity_def.physics.dimensions
                if dim is None:
                    dim = tuple(0.5 * s for s in self.world_transform.scale)
                trigger_def = TriggerDef(shape=self.entity_def.physics.shape or "box", dimensions=dim)
                self.entity_def.trigger = trigger_def

        if trigger_def is not None:
            if hasattr(app, "map_manager") and app.map_manager is not None:
                app.map_manager.trigger_system.register_trigger(self, trigger_def)

        # 8. Recursively spawn children
        for child_def in self.entity_def.children:
            child_node = MapEntityNode(child_def)
            child_node.spawn(app, parent_transform=self.world_transform, prefab_mgr=prefab_mgr)
            self.children.append(child_node)

        return self

    def teardown(self, app: ProjectApp) -> None:
        """Tears down all children and destroys ECS entities and Rapier bodies."""
        # 1. Teardown children first
        for child in self.children:
            child.teardown(app)
        self.children.clear()

        # 2. Detach scripts and trigger zones
        if hasattr(app, "map_manager") and app.map_manager is not None:
            for script in self.scripts:
                app.map_manager.script_runner.detach_script(script)
            if self.entity_def.trigger is not None:
                app.map_manager.trigger_system.unregister_node_triggers(self)

        for script in self.scripts:
            script.on_despawn()
        self.scripts.clear()

        # 3. Unregister dynamic prop and remove Rapier physics body
        if self.ecs_id is not None:
            if hasattr(app, "props") and app.props is not None:
                app.props.unregister_prop(self.ecs_id)
            if self.has_rigid_body:
                app.physics.remove_body(self.ecs_id)
                self.has_rigid_body = False

        # 3. Destroy ECS entity
        if self.ecs_id is not None:
            if app.ecs.is_alive(self.ecs_id):
                app.ecs.destroy_entity(self.ecs_id)
            self.ecs_id = None

    def collect_all_ecs_ids(self) -> list[int]:
        """Collects all active ECS entity IDs in this entity node and its descendant subtree."""
        ids: list[int] = []
        if self.ecs_id is not None:
            ids.append(self.ecs_id)
        for child in self.children:
            ids.extend(child.collect_all_ecs_ids())
        return ids

    def find_by_id(self, target_id: str) -> MapEntityNode | None:
        """Finds this node or any descendant matching the given string ID."""
        if self.entity_def.id == target_id:
            return self
        for child in self.children:
            found = child.find_by_id(target_id)
            if found is not None:
                return found
        return None
