use pyo3::prelude::*;
use rapier3d::prelude::*;
use rapier3d::control::*;

#[pyclass]
pub struct PyRapierWorld {
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    physics_hooks: (),
    event_handler: (),
    // Keep track of handle mapping: entity_id -> RigidBodyHandle
    handles: std::collections::HashMap<u32, RigidBodyHandle>,
    // Character controllers: entity_id -> (KinematicCharacterController, SharedShape)
    character_controllers: std::collections::HashMap<u32, (KinematicCharacterController, SharedShape)>,
}

#[pymethods]
impl PyRapierWorld {
    #[new]
    #[pyo3(signature = (gravity_x=0.0, gravity_y=-9.81, gravity_z=0.0))]
    pub fn new(gravity_x: f32, gravity_y: f32, gravity_z: f32) -> Self {
        Self {
            gravity: vector![gravity_x, gravity_y, gravity_z],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            physics_hooks: (),
            event_handler: (),
            handles: std::collections::HashMap::new(),
            character_controllers: std::collections::HashMap::new(),
        }
    }

    /// Fixed timestep simulation step. Releases the Python GIL during stepping.
    pub fn step(&mut self, py: Python<'_>, dt: f32) {
        self.integration_parameters.dt = dt;
        let gravity = self.gravity;

        // Release the Python GIL so native threads can run in parallel
        py.allow_threads(|| {
            self.physics_pipeline.step(
                &gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                &mut self.ccd_solver,
                Some(&mut self.query_pipeline),
                &self.physics_hooks,
                &self.event_handler,
            );
        });
    }

    /// Add a rigid body to the simulation for an entity
    pub fn create_rigid_body(
        &mut self,
        entity_id: u32,
        body_type: &str, // "dynamic", "fixed", "kinematic_position", "kinematic_velocity"
        x: f32,
        y: f32,
        z: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
    ) -> PyResult<()> {
        let b_type = match body_type {
            "dynamic" => RigidBodyType::Dynamic,
            "fixed" => RigidBodyType::Fixed,
            "kinematic_position" => RigidBodyType::KinematicPositionBased,
            "kinematic_velocity" => RigidBodyType::KinematicVelocityBased,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown body type: {}", body_type))),
        };

        let rot = nalgebra::UnitQuaternion::new_normalize(
            nalgebra::Quaternion::new(qw, qx, qy, qz),
        );
        let rb = RigidBodyBuilder::new(b_type)
            .translation(vector![x, y, z])
            .rotation(rot.scaled_axis())
            .build();

        let handle = self.rigid_body_set.insert(rb);
        self.handles.insert(entity_id, handle);
        Ok(())
    }

    /// Attach a box collider to an entity's rigid body
    pub fn attach_box_collider(&mut self, entity_id: u32, half_x: f32, half_y: f32, half_z: f32) -> PyResult<()> {
        let handle = self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;
        let collider = ColliderBuilder::cuboid(half_x, half_y, half_z).build();
        self.collider_set.insert_with_parent(collider, *handle, &mut self.rigid_body_set);
        Ok(())
    }

    /// Attach a sphere collider
    pub fn attach_sphere_collider(&mut self, entity_id: u32, radius: f32) -> PyResult<()> {
        let handle = self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;
        let collider = ColliderBuilder::ball(radius).build();
        self.collider_set.insert_with_parent(collider, *handle, &mut self.rigid_body_set);
        Ok(())
    }

    /// Attach a capsule collider (Y-axis aligned)
    pub fn attach_capsule_collider(&mut self, entity_id: u32, half_height: f32, radius: f32) -> PyResult<()> {
        let handle = self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;
        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        self.collider_set.insert_with_parent(collider, *handle, &mut self.rigid_body_set);
        Ok(())
    }

    /// Remove an entity's rigid body
    pub fn remove_rigid_body(&mut self, entity_id: u32) -> bool {
        self.character_controllers.remove(&entity_id);
        if let Some(handle) = self.handles.remove(&entity_id) {
            self.rigid_body_set.remove(
                handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
            true
        } else {
            false
        }
    }

    /// Get position and quaternion for an entity: (x, y, z, qx, qy, qz, qw)
    pub fn get_transform(&self, entity_id: u32) -> PyResult<(f32, f32, f32, f32, f32, f32, f32)> {
        let handle = self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;
        let rb = self.rigid_body_set.get(*handle).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("Rigid body handle invalid")
        })?;
        let pos = rb.translation();
        let rot = rb.rotation();
        let q = rot.quaternion();
        Ok((pos.x, pos.y, pos.z, q.i, q.j, q.k, q.w))
    }

    /// Set position and rotation for an entity
    pub fn set_transform(
        &mut self,
        entity_id: u32,
        x: f32,
        y: f32,
        z: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
    ) -> PyResult<()> {
        let handle = self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;
        let rb = self.rigid_body_set.get_mut(*handle).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("Rigid body handle invalid")
        })?;
        let rot = nalgebra::UnitQuaternion::new_normalize(
            nalgebra::Quaternion::new(qw, qx, qy, qz),
        );
        rb.set_position(Isometry::from_parts(
            Translation::from(vector![x, y, z]),
            rot,
        ), true);
        Ok(())
    }

    /// Batch sync state: writes transform [x, y, z, qx, qy, qz, qw] for each tracked entity
    pub fn sync_transforms(&self, entity_ids: Vec<u32>) -> Vec<(u32, f32, f32, f32, f32, f32, f32, f32)> {
        let mut results = Vec::with_capacity(entity_ids.len());
        for id in entity_ids {
            if let Some(handle) = self.handles.get(&id) {
                if let Some(rb) = self.rigid_body_set.get(*handle) {
                    let pos = rb.translation();
                    let q = rb.rotation().quaternion();
                    results.push((id, pos.x, pos.y, pos.z, q.i, q.j, q.k, q.w));
                }
            }
        }
        results
    }

    /// Direct zero-copy transform sync straight into a contiguous NumPy float32 buffer
    pub fn sync_transforms_direct(
        &self,
        _py: Python<'_>,
        entity_ids: Vec<u32>,
        dense_indices: Vec<u32>,
        target_buffer: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let buf = pyo3::buffer::PyBuffer::<f32>::get_bound(target_buffer)?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                buf.buf_ptr() as *mut f32,
                buf.item_count(),
            )
        };
        let n = entity_ids.len().min(dense_indices.len());
        for i in 0..n {
            let ent_id = entity_ids[i];
            let dense_idx = dense_indices[i] as usize;
            if let Some(handle) = self.handles.get(&ent_id) {
                if let Some(rb) = self.rigid_body_set.get(*handle) {
                    let pos = rb.translation();
                    let q = rb.rotation().quaternion();
                    let base = dense_idx * 7;
                    if base + 6 < slice.len() {
                        slice[base] = pos.x;
                        slice[base + 1] = pos.y;
                        slice[base + 2] = pos.z;
                        slice[base + 3] = q.i;
                        slice[base + 4] = q.j;
                        slice[base + 5] = q.k;
                        slice[base + 6] = q.w;
                    }
                }
            }
        }
        Ok(())
    }

    /// Raycast query: returns (hit_entity_id, hit_distance, normal_x, normal_y, normal_z) if hit
    pub fn cast_ray(
        &mut self,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
        max_toi: f32,
        solid: bool,
    ) -> Option<(u32, f32, f32, f32, f32)> {
        self.query_pipeline.update(&self.collider_set);
        let ray = Ray::new(
            point![origin_x, origin_y, origin_z],
            vector![dir_x, dir_y, dir_z],
        );
        if let Some((handle, hit)) = self.query_pipeline.cast_ray_and_get_normal(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_toi,
            solid,
            QueryFilter::default(),
        ) {
            if let Some(collider) = self.collider_set.get(handle) {
                if let Some(parent_handle) = collider.parent() {
                    // Reverse map to entity_id
                    for (ent_id, rb_h) in &self.handles {
                        if *rb_h == parent_handle {
                            return Some((*ent_id, hit.time_of_impact, hit.normal.x, hit.normal.y, hit.normal.z));
                        }
                    }
                }
            }
            return Some((0, hit.time_of_impact, hit.normal.x, hit.normal.y, hit.normal.z));
        }
        None
    }

    /// Creates a Kinematic Character Controller with capsule collider for an entity
    #[pyo3(signature = (
        entity_id,
        half_height=0.5,
        radius=0.4,
        max_slope_deg=45.0,
        step_height=0.3,
        snap_to_ground=0.2,
        x=0.0,
        y=0.0,
        z=0.0,
    ))]
    pub fn create_character_controller(
        &mut self,
        entity_id: u32,
        half_height: f32,
        radius: f32,
        max_slope_deg: f32,
        step_height: f32,
        snap_to_ground: f32,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<()> {
        let max_slope_rad = max_slope_deg.to_radians();
        let kcc = KinematicCharacterController {
            up: Vector::y_axis(),
            offset: CharacterLength::Absolute(0.02),
            slide: true,
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Absolute(step_height),
                min_width: CharacterLength::Absolute(0.2),
                include_dynamic_bodies: false,
            }),
            max_slope_climb_angle: max_slope_rad,
            min_slope_slide_angle: max_slope_rad,
            snap_to_ground: if snap_to_ground > 0.0 {
                Some(CharacterLength::Absolute(snap_to_ground))
            } else {
                None
            },
            ..Default::default()
        };

        // Create kinematic position-based rigid body
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![x, y, z])
            .build();
        let handle = self.rigid_body_set.insert(rb);
        self.handles.insert(entity_id, handle);

        // Attach capsule collider
        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        self.collider_set.insert_with_parent(collider, handle, &mut self.rigid_body_set);

        // Store controller with shared capsule shape for sweep queries
        let shape = SharedShape::capsule_y(half_height, radius);
        self.character_controllers.insert(entity_id, (kcc, shape));

        Ok(())
    }

    /// Moves a character entity using the Kinematic Character Controller.
    /// Returns: (effective_dx, effective_dy, effective_dz, is_grounded, is_sliding)
    pub fn move_character(
        &mut self,
        entity_id: u32,
        desired_dx: f32,
        desired_dy: f32,
        desired_dz: f32,
        dt: f32,
    ) -> PyResult<(f32, f32, f32, bool, bool)> {
        let (kcc, shape) = self.character_controllers.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no character controller", entity_id))
        })?.clone();

        let handle = *self.handles.get(&entity_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Entity {} has no rigid body", entity_id))
        })?;

        let current_pos = *self.rigid_body_set.get(handle).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("Rigid body handle invalid")
        })?.position();

        let desired_translation = vector![desired_dx, desired_dy, desired_dz];

        // Exclude the character's own body from obstacle queries
        let filter = QueryFilter::default().exclude_rigid_body(handle);

        // Ensure query pipeline has latest collider transforms
        self.query_pipeline.update(&self.collider_set);

        // Perform character shape sweep query against spatial query pipeline
        let movement = kcc.move_shape(
            dt,
            &self.rigid_body_set,
            &self.collider_set,
            &self.query_pipeline,
            shape.as_ref(),
            &current_pos,
            desired_translation,
            filter,
            |_collision| {},
        );

        // Apply effective translation directly to rigid body position
        let new_translation = current_pos.translation.vector + movement.translation;
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.set_next_kinematic_translation(new_translation);
            rb.set_translation(new_translation, true);
        }

        Ok((
            movement.translation.x,
            movement.translation.y,
            movement.translation.z,
            movement.grounded,
            movement.is_sliding_down_slope,
        ))
    }
}

#[pymodule]
fn pymordial_rapier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRapierWorld>()?;
    Ok(())
}
