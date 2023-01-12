use crate::hgrid::HGrid;
use crate::marching_cubes::march_cube;
use crate::poisson_layer::PoissonLayer;
use crate::poisson_vector_field::PoissonVectorField;
use crate::polynomial::{eval_bspline, eval_bspline_diff};
use crate::Real;
use na::{vector, Point3, Vector3};
use parry::bounding_volume::{Aabb, BoundingVolume};
use parry::partitioning::IndexedData;
use std::collections::HashMap;
use std::ops::{AddAssign, Mul};

/// An implicit surface reconstructed with the Screened Poisson reconstruction algorithm.
#[derive(Clone)]
pub struct PoissonReconstruction {
    layers: Vec<PoissonLayer>,
    isovalue: Real,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct CellWithId {
    pub cell: Point3<i64>,
    pub id: usize,
}

impl IndexedData for CellWithId {
    fn default() -> Self {
        Self {
            cell: Point3::default(),
            id: 0,
        }
    }

    fn index(&self) -> usize {
        self.id
    }
}

impl PoissonReconstruction {
    /// Reconstruct a surface using the Screened Poisson reconstruction algorithm,
    /// given a set of sample points and normals at these points.
    ///
    /// # Parameters
    /// - `points`: the sample points.
    /// - `normals`: the normals at the sample points. Must have the same length as `points`.
    /// - `screening`: the screening coefficient. Larger values increase the fitting of the
    ///   reconstructed surface relative to the sample point’s positions. Setting this to `0.0`
    ///   disables screening (but reduces computation times).
    /// - `density_estimation_depth`: the depth on the multigrid solver where point density estimation
    ///   is calculated. The estimation kernel radius will be equal to the maximum extent of the
    ///   input point’s AABB, divided by `2.pow(max_depth)`. Smaller value of this parameter results
    ///   in more robustness wrt. occasional holes and sampling irregularities, but reduces the
    ///   detail accuracies.
    /// - `max_depth`: the max depth of the multigrid solver. Larger values result in higher accuracy
    ///   (which requires higher sampling densities, or a `density_estimation_depth` set to a smaller
    ///   value). Higher values increases computation times.
    /// - `max_relaxation_iters`: the maximum number of iterations for the internal
    ///   conjugate-gradient solver. Values around `10` should be enough for most cases.
    pub fn from_points_and_normals(
        points: &[Point3<Real>],
        normals: &[Vector3<Real>],
        screening: Real,
        density_estimation_depth: usize,
        max_depth: usize,
        max_relaxation_iters: usize,
    ) -> Self {
        assert_eq!(
            points.len(),
            normals.len(),
            "Exactly one normal per point must be provided."
        );
        assert!(density_estimation_depth <= max_depth);
        let mut root_aabb = Aabb::from_points(points);
        let max_extent = root_aabb.extents().max();
        let leaf_cell_width = max_extent / (2.0 as Real).powi(max_depth as i32);
        root_aabb.loosen(leaf_cell_width);
        let grid_origin = root_aabb.mins;

        let mut layers = vec![];
        layers.push(PoissonLayer::from_points(
            points,
            grid_origin,
            leaf_cell_width,
        ));

        for i in 0..max_depth {
            let layer = PoissonLayer::from_next_layer(points, &layers[i]);
            layers.push(layer);
        }

        // Reverse so the coarser layers go first.
        layers.reverse();

        let vector_field =
            PoissonVectorField::new(&layers, points, normals, density_estimation_depth);

        for i in 0..layers.len() {
            let result = PoissonLayer::solve(
                &layers,
                i,
                &vector_field,
                points,
                normals,
                screening,
                max_relaxation_iters,
            );
            layers[i].node_weights = result;
        }

        let mut total_weight = 0.0;
        let mut result = Self {
            layers,
            isovalue: 0.0,
        };
        let mut isovalue = 0.0;

        for (pt, w) in points.iter().zip(vector_field.densities.iter()) {
            isovalue += result.eval(pt) / *w;
            total_weight += 1.0 / *w;
        }

        result.isovalue = isovalue / total_weight;
        result
    }

    /// The domain where the surface’s implicit function is defined.
    pub fn aabb(&self) -> &Aabb {
        self.layers.last().unwrap().cells_qbvh.root_aabb()
    }

    /// Does the given AABB intersect any of the smallest grid cells of the reconstruction?
    pub fn leaf_cells_intersect_aabb(&self, aabb: &Aabb) -> bool {
        let mut intersections = vec![];
        self.layers
            .last()
            .unwrap()
            .cells_qbvh
            .intersect_aabb(aabb, &mut intersections);
        !intersections.is_empty()
    }

    /// Evaluates the value of the implicit function at the given 3D point.
    ///
    /// In order to get a meaningful value, the point must be located inside of [`Self::aabb`].
    pub fn eval(&self, pt: &Point3<Real>) -> Real {
        let mut result = 0.0;

        for layer in &self.layers {
            result += layer.eval_triquadratic(pt);
        }

        result - self.isovalue
    }

    /// Evaluates the value of the implicit function’s gradient at the given 3D point.
    ///
    /// In order to get a meaningful value, the point must be located inside of [`Self::aabb`].
    pub fn eval_gradient(&self, pt: &Point3<Real>) -> Vector3<Real> {
        let mut result = Vector3::zeros();

        for layer in &self.layers {
            result += layer.eval_triquadratic_gradient(pt);
        }

        result
    }

    /// Reconstructs a mesh from this implicit function using a simple marching-cubes, extracting
    /// the isosurface at 0.
    pub fn reconstruct_mesh(&self) -> Vec<Point3<Real>> {
        let mut vertices = vec![];

        if let Some(last_layer) = self.layers.last() {
            for cell in last_layer.cells_qbvh.raw_proxies() {
                let aabb = last_layer.cells_qbvh.node_aabb(cell.node).unwrap();
                let mut vertex_values = [0.0; 8];

                for (pt, val) in aabb.vertices().iter().zip(vertex_values.iter_mut()) {
                    *val = self.eval(pt);
                }

                march_cube(&aabb.mins, &aabb.maxs, &vertex_values, 0.0, &mut vertices);
            }
        }

        vertices
    }
}

pub fn eval_triquadratic<T: Mul<Real, Output = T> + AddAssign + Copy + Default>(
    pt: &Point3<Real>,
    grid: &HGrid<usize>,
    grid_node_idx: &HashMap<Point3<i64>, usize>,
    node_weights: &[T],
) -> T {
    let cell_width = grid.cell_width();
    let ref_cell = grid.key(pt);
    let mut result = T::default();

    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                let curr_cell = ref_cell + vector![i, j, k];

                if let Some(node_id) = grid_node_idx.get(&curr_cell) {
                    let spline_origin = grid.cell_center(&curr_cell);
                    let valx = eval_bspline(pt.x, spline_origin.x, cell_width);
                    let valy = eval_bspline(pt.y, spline_origin.y, cell_width);
                    let valz = eval_bspline(pt.z, spline_origin.z, cell_width);
                    result += node_weights[*node_id] * valx * valy * valz;
                }
            }
        }
    }

    result
}

pub fn eval_triquadratic_gradient(
    pt: &Point3<Real>,
    grid: &HGrid<usize>,
    grid_node_idx: &HashMap<Point3<i64>, usize>,
    node_weights: &[Real],
) -> Vector3<Real> {
    let cell_width = grid.cell_width();
    let ref_cell = grid.key(pt);
    let mut result = Vector3::default();

    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                let curr_cell = ref_cell + vector![i, j, k];

                if let Some(node_id) = grid_node_idx.get(&curr_cell) {
                    let spline_origin = grid.cell_center(&curr_cell);

                    let valx = eval_bspline(pt.x, spline_origin.x, cell_width);
                    let valy = eval_bspline(pt.y, spline_origin.y, cell_width);
                    let valz = eval_bspline(pt.z, spline_origin.z, cell_width);

                    let diffx = eval_bspline_diff(pt.x, spline_origin.x, cell_width);
                    let diffy = eval_bspline_diff(pt.y, spline_origin.y, cell_width);
                    let diffz = eval_bspline_diff(pt.z, spline_origin.z, cell_width);

                    result += Vector3::new(
                        diffx * valy * valz,
                        valx * diffy * valz,
                        valx * valy * diffz,
                    ) * node_weights[*node_id];
                }
            }
        }
    }

    result
}
