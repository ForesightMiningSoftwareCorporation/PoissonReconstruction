use crate::conjugate_gradient::solve_conjugate_gradient;
use crate::hgrid::HGrid;
use crate::poisson_vector_field::PoissonVectorField;
use crate::polynomial::TriQuadraticBspline;
use crate::{
    poisson::{self, CellWithId},
    polynomial, Real,
};
use na::{vector, DVector, Point3, Vector3};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use parry::bounding_volume::Aabb;
use parry::partitioning::Qbvh;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct PoissonLayer {
    pub grid: HGrid<usize>,
    pub cells_qbvh: Qbvh<CellWithId>,
    pub grid_node_idx: HashMap<Point3<i64>, usize>,
    pub ordered_nodes: Vec<Point3<i64>>,
    pub node_weights: DVector<Real>,
}

impl PoissonLayer {
    pub fn cell_width(&self) -> Real {
        self.grid.cell_width()
    }
}

impl PoissonLayer {
    pub fn from_points(
        points: &[Point3<Real>],
        grid_origin: Point3<Real>,
        cell_width: Real,
    ) -> Self {
        let mut grid = HGrid::new(grid_origin, cell_width);
        let mut grid_node_idx = HashMap::new();
        let mut ordered_nodes = vec![];

        // for pt in points {
        //     let ref_node = grid.key(pt);
        //
        //     for corner_shift in CORNERS.iter() {
        //         let node = ref_node + corner_shift;
        //         let _ = grid_node_idx.entry(node).or_insert_with(|| {
        //             let center = grid.cell_center(&node);
        //             grid.insert(&center, 0);
        //             ordered_nodes.push(node);
        //             ordered_nodes.len() - 1
        //         });
        //     }
        // }

        // TODO: do we still need this when using the multigrid solver?
        for (pid, pt) in points.iter().enumerate() {
            let ref_node = grid.key(pt);
            let ref_center = grid.cell_center(&ref_node);
            grid.insert(&ref_center, pid);

            for i in -2..=2 {
                for j in -2..=2 {
                    for k in -2..=2 {
                        let node = ref_node + vector![i, j, k];
                        let center = grid.cell_center(&node);
                        let _ = grid_node_idx.entry(node).or_insert_with(|| {
                            grid.insert(&center, usize::MAX);
                            ordered_nodes.push(node);
                            ordered_nodes.len() - 1
                        });
                    }
                }
            }
        }

        Self::from_populated_grid(grid, grid_node_idx, ordered_nodes)
    }

    pub fn from_next_layer(points: &[Point3<Real>], layer: &Self) -> Self {
        let cell_width = layer.cell_width() * 2.0;
        let mut grid = HGrid::new(*layer.grid.origin(), cell_width);
        let mut grid_node_idx = HashMap::new();
        let mut ordered_nodes = vec![];

        // Add nodes to the new grid to form a comforming "octree".
        for sub_node_key in &layer.ordered_nodes {
            let pt = layer.grid.cell_center(sub_node_key);
            let my_key = grid.key(&pt);
            let my_center = grid.cell_center(&my_key);
            let quadrant = pt - my_center;

            let range = |x| {
                if x < 0.0 {
                    -2..=1
                } else {
                    -1..=2
                }
            };

            for i in range(quadrant.x) {
                for j in range(quadrant.y) {
                    for k in range(quadrant.z) {
                        let adj_key = my_key + vector![i, j, k];

                        let _ = grid_node_idx.entry(adj_key).or_insert_with(|| {
                            let adj_center = grid.cell_center(&adj_key);
                            grid.insert(&adj_center, usize::MAX);
                            ordered_nodes.push(adj_key);
                            ordered_nodes.len() - 1
                        });
                    }
                }
            }
        }

        for (pid, pt) in points.iter().enumerate() {
            let ref_node = grid.key(pt);
            let ref_center = grid.cell_center(&ref_node);
            grid.insert(&ref_center, pid);
        }

        Self::from_populated_grid(grid, grid_node_idx, ordered_nodes)
    }

    fn from_populated_grid(
        grid: HGrid<usize>,
        grid_node_idx: HashMap<Point3<i64>, usize>,
        ordered_nodes: Vec<Point3<i64>>,
    ) -> Self {
        let cell_width = grid.cell_width();
        let mut cells_qbvh = Qbvh::new();
        cells_qbvh.clear_and_rebuild(
            ordered_nodes.iter().map(|key| {
                let center = grid.cell_center(key);
                let id = grid_node_idx[key];
                let half_width = Vector3::repeat(cell_width / 2.0);
                (
                    CellWithId { cell: *key, id },
                    Aabb::from_half_extents(center, half_width),
                )
            }),
            0.0,
        );

        let node_weights = DVector::zeros(grid_node_idx.len());

        Self {
            grid,
            cells_qbvh,
            ordered_nodes,
            grid_node_idx,
            node_weights,
        }
    }

    pub(crate) fn solve(
        layers: &[Self],
        curr_layer: usize,
        vector_field: &PoissonVectorField,
        points: &[Point3<Real>],
        normals: &[Vector3<Real>],
        screening: Real,
        niters: usize,
    ) -> DVector<Real> {
        let my_layer = &layers[curr_layer];
        let cell_width = my_layer.cell_width();
        assert_eq!(points.len(), normals.len());
        let convolution = polynomial::compute_quadratic_bspline_convolution_coeffs(cell_width);
        let num_nodes = my_layer.ordered_nodes.len();

        // Compute the gradient matrix.
        let mut grad_matrix = CooMatrix::new(num_nodes, num_nodes);
        let screen_factor =
            (2.0 as Real).powi(curr_layer as i32) * screening * vector_field.area_approximation()
                / (points.len() as Real);

        for (nid, node) in my_layer.ordered_nodes.iter().enumerate() {
            let center1 = my_layer.grid.cell_center(node);

            for i in -2..=2 {
                for j in -2..=2 {
                    for k in -2..=2 {
                        let other_node = node + vector![i, j, k];
                        let center2 = my_layer.grid.cell_center(&other_node);

                        if let Some(other_nid) = my_layer.grid_node_idx.get(&other_node) {
                            let ii = (i + 2) as usize;
                            let jj = (j + 2) as usize;
                            let kk = (k + 2) as usize;

                            let mut laplacian = convolution.laplacian[ii][jj][kk];

                            if screening != 0.0 {
                                for si in -1..=1 {
                                    for sj in -1..=1 {
                                        for sk in -1..=1 {
                                            let adj = node + vector![si, sj, sk];

                                            if let Some(pt_ids) = my_layer.grid.cell(&adj) {
                                                for pid in pt_ids {
                                                    // Use get to ignore the sentinel.
                                                    if let Some(pt) = points.get(*pid) {
                                                        let poly1 = TriQuadraticBspline::new(
                                                            center1, cell_width,
                                                        );
                                                        let poly2 = TriQuadraticBspline::new(
                                                            center2, cell_width,
                                                        );
                                                        laplacian += screen_factor
                                                            * poly1.eval(*pt)
                                                            * poly2.eval(*pt);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            grad_matrix.push(nid, *other_nid, laplacian);
                        }
                    }
                }
            }
        }

        // Build rhs
        let mut rhs = DVector::zeros(my_layer.ordered_nodes.len());
        vector_field.build_rhs(layers, curr_layer, &mut rhs);

        // Subtract the results from the coarser layers.
        rhs.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(rhs_id, rhs)| {
                let node_key = my_layer.ordered_nodes[rhs_id];
                let node_center = my_layer.grid.cell_center(&node_key);
                let poly1 = TriQuadraticBspline::new(node_center, my_layer.cell_width());

                for coarser_layer in &layers[0..curr_layer] {
                    let aabb = Aabb::from_half_extents(
                        node_center,
                        Vector3::repeat(
                            my_layer.cell_width() * 1.5 + coarser_layer.cell_width() * 1.5,
                        ),
                    );

                    for (coarser_node_key, _) in coarser_layer
                        .grid
                        .cells_intersecting_aabb(&aabb.mins, &aabb.maxs)
                    {
                        let coarser_node_center = coarser_layer.grid.cell_center(&coarser_node_key);
                        let poly2 = TriQuadraticBspline::new(
                            coarser_node_center,
                            coarser_layer.cell_width(),
                        );
                        let mut coeff = poly1.grad_grad(poly2, true, true).sum();
                        let coarser_rhs_id = coarser_layer.grid_node_idx[&coarser_node_key];

                        if screening != 0.0 {
                            for si in -1..=1 {
                                for sj in -1..=1 {
                                    for sk in -1..=1 {
                                        let adj = node_key + vector![si, sj, sk];

                                        if let Some(pt_ids) = my_layer.grid.cell(&adj) {
                                            for pid in pt_ids {
                                                // Use get to ignore the sentinel.
                                                if let Some(pt) = points.get(*pid) {
                                                    coeff += screen_factor
                                                        * poly1.eval(*pt)
                                                        * poly2.eval(*pt);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        *rhs -= coarser_layer.node_weights[coarser_rhs_id] * coeff;
                    }
                }
            });

        // Solve the sparse system.
        let lhs = CscMatrix::from(&grad_matrix);
        solve_conjugate_gradient(&lhs, &mut rhs, niters);
        // let chol = CscCholesky::factor(&lhs).unwrap();
        // chol.solve_mut(&mut rhs);

        rhs
    }

    pub fn eval_triquadratic(&self, pt: &Point3<Real>) -> Real {
        poisson::eval_triquadratic(
            pt,
            &self.grid,
            &self.grid_node_idx,
            self.node_weights.as_slice(),
        )
    }

    pub fn eval_triquadratic_gradient(&self, pt: &Point3<Real>) -> Vector3<Real> {
        poisson::eval_triquadratic_gradient(
            pt,
            &self.grid,
            &self.grid_node_idx,
            self.node_weights.as_slice(),
        )
    }
}
